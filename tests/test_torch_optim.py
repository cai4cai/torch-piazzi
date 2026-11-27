"""``piazzi.minimize`` with a ``torch.optim.Optimizer`` subclass as ``method``."""

import math

import pytest
import torch
from problems import F64, SCALAR_PROBLEMS, rosen

import piazzi

SCALAR_IDS = [p.name for p in SCALAR_PROBLEMS]
STRONG_WOLFE = {"line_search_fn": "strong_wolfe"}


def upstream_lbfgs(fun, x0, **options):
    """A single upstream ``LBFGS.step``: the final point, evaluation count and iterations."""
    x = x0.clone().requires_grad_(True)
    opt = torch.optim.LBFGS([x], **options)
    evals = 0

    def closure():
        nonlocal evals
        evals += 1
        opt.zero_grad()
        loss = fun(x)
        loss.backward()
        return loss

    opt.step(closure)
    return x.detach(), evals, opt.state[x]["n_iter"]


@pytest.mark.parametrize(
    "x0",
    [torch.tensor([-1.2, 1.0], dtype=F64), torch.tensor([1.0, 8.0], dtype=F64), torch.zeros(10, dtype=F64)],
    ids=["rosen_2d", "rosen_2d_far", "rosen_10d"],
)
@pytest.mark.parametrize(
    "options", [STRONG_WOLFE, {**STRONG_WOLFE, "history_size": 5}, {}, {"lr": 0.1}], ids=["sw", "sw_h5", "fixed", "lr"]
)
@pytest.mark.parametrize("max_iter", [20, 200])
def test_lbfgs_matches_a_single_upstream_step(x0, options, max_iter):
    x_ref, evals_ref, iters_ref = upstream_lbfgs(rosen, x0, max_iter=max_iter, **options)
    res = piazzi.minimize(rosen, x0, method=torch.optim.LBFGS, options={"max_iter": max_iter, **options})
    assert torch.equal(res.x, x_ref)
    assert res.n_iter == iters_ref
    # Upstream does not evaluate the last point of a fixed-step run cut off by max_iter; piazzi
    # does, for result.fun. The point a line search accepts is not evaluated twice.
    extra = "line_search_fn" not in options and res.message == "max_iter reached"
    assert res.n_fun_grad == evals_ref + extra
    assert res.fun.item() == pytest.approx(rosen(res.x).item(), rel=1e-12)


@pytest.mark.parametrize("problem", SCALAR_PROBLEMS, ids=SCALAR_IDS)
def test_lbfgs_strong_wolfe_problems(problem):
    # Upstream's default tolerance_change (1e-9) stops at about 1e-5 from the minimiser.
    options = {"max_iter": 500, "tolerance_change": 1e-14, **STRONG_WOLFE}
    res = piazzi.minimize(problem.fun, problem.x0, method=torch.optim.LBFGS, options=options)
    assert res.converged, res.message
    assert res.x.shape == problem.x0.shape
    assert torch.allclose(res.x, problem.x_opt, atol=1e-5), res.x
    assert isinstance(res.raw, torch.optim.LBFGS)


def test_evaluations_are_counted_and_memoised():
    calls = 0

    def fun(x):
        nonlocal calls
        calls += 1
        return rosen(x)

    res = piazzi.minimize(fun, torch.tensor([-1.2, 1.0], dtype=F64), method=torch.optim.LBFGS, options=STRONG_WOLFE)
    assert res.n_fun_grad == calls and res.n_fun == 0
    # One evaluation per iteration at the very least, and no re-evaluation of accepted points.
    assert res.n_iter < calls < res.n_iter + 20


def test_max_eval_is_a_budget_for_the_whole_run():
    res = piazzi.minimize(
        rosen, torch.zeros(10, dtype=F64), method=torch.optim.LBFGS, options={"max_eval": 30, **STRONG_WOLFE}
    )
    assert res.message == "max_eval reached" and not res.converged
    assert 30 <= res.n_fun_grad < 30 + 25  # the last line search may overrun, as upstream


def test_gradient_below_tolerance_at_x0():
    res = piazzi.minimize(rosen, torch.ones(4, dtype=F64), method=torch.optim.LBFGS, options=STRONG_WOLFE)
    assert res.converged and res.n_iter == 0 and res.n_fun_grad == 1


@pytest.mark.parametrize(
    ("optimizer", "options"),
    [(torch.optim.Adam, {"lr": 0.1}), (torch.optim.SGD, {"lr": 0.1, "momentum": 0.5}), (torch.optim.Rprop, {})],
)
def test_first_order_optimizers(optimizer, options):
    target = torch.tensor([[1.0, -2.0], [0.5, 3.0]], dtype=F64)
    seen = []

    def fun(x):
        return ((x - target) ** 2).sum()

    res = piazzi.minimize(
        fun,
        torch.zeros(2, 2, dtype=F64),
        method=optimizer,
        callback=lambda info: seen.append(info.x),
        options={"max_iter": 500, **options},
    )
    assert res.x.shape == (2, 2)
    assert torch.allclose(res.x, target, atol=1e-3), res.x
    # At most one evaluation per step plus the final point (a zero step, as when Rprop skips
    # one, reuses the last); the callback sees every iterate, unaliased.
    assert res.n_fun_grad <= res.n_iter + 1 and res.n_iter == len(seen)
    assert torch.equal(seen[-1], res.x) and not torch.equal(seen[0], seen[-1])


@pytest.mark.parametrize("optimizer", [torch.optim.LBFGS, torch.optim.SGD])
def test_aux_float32_and_stop_by_callback(optimizer):
    # aux holds views of x; they must keep the values of the point they came from even
    # though the optimiser updates its parameter in place.
    def fun(x):
        return rosen(x), {"x": x, "first": x[0]}

    seen = []

    def callback(info):
        seen.append((info.x.clone(), info.aux))
        return info.iteration == 4

    options = STRONG_WOLFE if optimizer is torch.optim.LBFGS else {"lr": 1e-3}
    res = piazzi.minimize(
        fun, torch.tensor([-1.2, 1.0]), method=optimizer, has_aux=True, callback=callback, options=options
    )
    assert res.message == "stopped by callback" and res.n_iter == len(seen) == 5
    assert res.x.dtype == torch.float32
    assert all(torch.equal(aux["x"], x) and torch.equal(aux["first"], x[0]) for x, aux in seen)
    assert not torch.equal(seen[0][0], seen[-1][0])
    assert torch.equal(res.aux["x"], res.x) and res.aux["x"].data_ptr() != res.x.data_ptr()


def test_step_tol():
    res = piazzi.minimize(
        lambda x: (x**2).sum(), torch.ones(3, dtype=F64), method=torch.optim.Adam, options={"lr": 0.1, "step_tol": 1e-3}
    )
    assert res.converged and res.message == "step below step_tol"


def test_non_finite_values():
    res = piazzi.minimize(lambda x: x.sum() * math.nan, torch.ones(2, dtype=F64), method=torch.optim.Adam)
    assert not res.converged and res.n_iter == 0 and res.message == "non-finite value at x0"

    def fun(x):  # NaN beyond x = 1, minimiser at x = 3
        return ((x - 3) ** 2).sum() + torch.where(x > 1, math.nan, 0.0).sum()

    seen = []
    res = piazzi.minimize(
        fun, torch.zeros(1, dtype=F64), method=torch.optim.SGD, callback=seen.append, options={"lr": 0.5}
    )
    # The step to x = 3 gives NaN: the run stops at the last finite point, and the callback
    # sees that iteration as rejected, with the NaN evaluation counted.
    assert res.message == "non-finite value" and not res.converged
    assert torch.equal(res.x, torch.zeros(1, dtype=F64)) and res.fun.item() == 9.0
    assert len(seen) == res.n_iter == 1 and not seen[0].accepted
    assert torch.equal(seen[0].x, res.x) and seen[0].n_fun_grad == res.n_fun_grad == 2


def test_method_must_be_a_name_or_an_optimizer_class():
    x = torch.zeros(2, dtype=F64, requires_grad=True)
    with pytest.raises(TypeError, match="not an instance"):
        piazzi.minimize(rosen, torch.zeros(2, dtype=F64), method=torch.optim.LBFGS([x]))


def test_max_eval_needs_lbfgs():
    with pytest.raises(ValueError, match="max_eval"):
        piazzi.minimize(rosen, torch.zeros(2, dtype=F64), method=torch.optim.Adam, options={"max_eval": 10})
