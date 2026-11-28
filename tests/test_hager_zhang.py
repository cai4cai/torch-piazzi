"""The Hager-Zhang line search of :func:`piazzi.lbfgs`, ported from LineSearches.jl and TensorFlow Probability.

Sources:

- LineSearches.jl (``test/hagerzhangls.jl``, MIT licence): the 1-D ``find_steplength`` cases
  (quadratic, quartic, Rosenbrock slice, non-finite initial step, Wolfe at the initial step, custom
  constants, Float32, Himmelblau along a direction).
- TensorFlow Probability (``optimizer/linesearch/hager_zhang_test.py``, Apache-2.0 licence,
  Copyright 2018 The TensorFlow Probability Authors): ``test_quadratic``, ``test_multiple_minima``,
  ``test_rosenbrock``, ``test_approx_wolfe``, ``test_determinism``, ``test_consistency`` and
  ``test_inf_recovery``. ``test_multiple_minima`` checks the exact or the approximate Wolfe
  conditions instead of the exact ones, as the implementation follows LineSearches.jl (below).
- The implementation follows ``LineSearches.jl``'s ``HagerZhang`` (``src/hagerzhang.jl``) and
  Optim.jl's L-BFGS-B ``hz.jl``; the full MIT notice is in ``piazzi/_line_search.py``.

Differences: the references run the search on a function of the step length, and so does the
harness here (a one-dimensional problem with ``x0 = 0`` and ``d = 1``). piazzi's search enforces a
budget of evaluations. Like the references, it accepts a point satisfying the approximate Wolfe
conditions even if it is higher than the start, by at most ``epsilon |f|``
(``test_approx_wolfe_may_exceed_the_loss_by_epsilon``). TFP's batched and graph-mode tests have no
counterpart. The solver-level tests (``piazzi.lbfgs(..., line_search="hager_zhang")``) compare with the
other line searches and with ``torch.optim.LBFGS``.
"""

import math

import pytest
import torch
from problems import F64, SCALAR_PROBLEMS, rosen

import piazzi
from piazzi._line_search import hager_zhang_search

DELTA, SIGMA, EPSILON = 0.1, 0.9, 1e-6


def search(phi, c, *, delta=DELTA, sigma=SIGMA, epsilon=EPSILON, max_ls=50, t_max=math.inf, dtype=F64):
    """Run the search on ``phi(t) -> (value, slope)``; return the result, the step lengths tried and phi(0)."""
    tried = []

    def eval_full(xv):
        t = xv.item()
        tried.append(t)
        f, df = phi(t)
        return torch.tensor(f, dtype=dtype), None, torch.tensor([df], dtype=dtype)

    f0, df0 = phi(0.0)
    out = hager_zhang_search(
        torch.zeros(1, dtype=dtype),
        f0,
        torch.tensor([df0], dtype=dtype),
        None,
        torch.ones(1, dtype=dtype),
        c,
        t_max,
        df0,
        False,
        None,
        delta,
        sigma,
        epsilon,
        max_ls,
        eval_full,
        lambda _m, x, delta_x: x + delta_x,
    )
    return out, tried, (f0, df0)


def is_exact_wolfe(t, f_t, df_t, f0, df0, delta=DELTA, sigma=SIGMA):
    return f_t - f0 <= delta * df0 * t and df_t >= sigma * df0


def is_approx_wolfe(f_t, df_t, f0, df0, delta=DELTA, sigma=SIGMA, epsilon=EPSILON):
    return f_t <= f0 + epsilon * abs(f0) and (2 * delta - 1) * df0 >= df_t >= sigma * df0


def quadratic(t):
    return (t - 2.0) ** 2, 2.0 * (t - 2.0)


# ---- LineSearches.jl: test/hagerzhangls.jl ----


@pytest.mark.parametrize("start", [1.0, 2.0, 7.5])
def test_julia_quadratic(start):
    (_, f_t, _, _, t, _, _, ok), _, (f0, _) = search(quadratic, start)
    assert ok and t > 0 and f_t.item() < f0


def test_julia_wolfe_at_the_initial_step_costs_one_evaluation():
    (_, f_t, _, _, t, _, _, ok), tried, _ = search(quadratic, 2.0)
    assert ok and t == 2.0 and abs(f_t.item()) < 1e-10 and tried == [2.0]


def test_julia_quartic():
    (_, f_t, _, _, _, _, _, ok), _, (f0, _) = search(lambda t: ((t - math.pi) ** 4, 4 * (t - math.pi) ** 3), 10.0)
    assert ok and f_t.item() < f0


def test_julia_rosenbrock_slice():
    def phi(a):
        return (1 - a) ** 2 + 100 * (a * a - a) ** 2, -2 * (1 - a) + 200 * (a * a - a) * (2 * a - 1)

    (_, f_t, _, _, _, _, _, ok), _, (f0, _) = search(phi, 0.01)
    assert ok and f_t.item() < f0


def test_julia_non_finite_initial_step_backtracks():
    def phi(a):
        return (math.inf, math.inf) if a > 5 else quadratic(a)

    (_, f_t, _, _, t, _, _, ok), _, _ = search(phi, 100.0)
    assert ok and t <= 5 and math.isfinite(f_t.item())


def test_julia_custom_constants():
    (_, f_t, _, _, _, _, _, ok), _, _ = search(quadratic, 1.0, delta=0.01, sigma=0.1)
    assert ok and abs(f_t.item()) < 1e-6


def test_julia_float32():
    (x_t, f_t, _, g_t, _, _, _, ok), _, _ = search(quadratic, 1.0, dtype=torch.float32)
    assert ok and x_t.dtype == f_t.dtype == g_t.dtype == torch.float32


def test_julia_himmelblau_along_a_direction():
    s = torch.tensor([42.0, 18.0], dtype=F64)
    x0 = torch.tensor([2.0, 2.0], dtype=F64)

    def himmelblau(x):
        return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2

    def phi(a):
        x = (x0 + a * s).requires_grad_(True)
        f = himmelblau(x)
        (g,) = torch.autograd.grad(f, x)
        return f.item(), torch.dot(g, s).item()

    f0, df0 = phi(0.0)
    assert df0 < 0
    (_, f_t, _, _, _, _, _, ok), _, _ = search(phi, 1.0)
    assert ok and f_t.item() < f0


# ---- TensorFlow Probability: hager_zhang_test.py ----


@pytest.mark.parametrize("start", [0.1, 7.0], ids=["close_start", "far_start"])
def test_tfp_quadratic(start):
    (_, f_t, _, g_t, t, _, _, ok), _, (f0, df0) = search(lambda x: ((x - 1.3) ** 2, 2 * (x - 1.3)), start)
    assert ok and is_exact_wolfe(t, f_t.item(), g_t.item(), f0, df0)


def multiple_minima(x):
    # Two minima in the direction of positive x: around 0.46 and around 2.65.
    val = 0.988 * x**5 - 4.96 * x**4 + 4.978 * x**3 + 5.015 * x**2 - 6.043 * x - 1
    dval = 4.94 * x**4 - 19.84 * x**3 + 14.934 * x**2 + 10.03 * x - 6.043
    return val, dval


@pytest.mark.parametrize("start", [0.1, 1.5, 2.0, 4.0])
def test_tfp_multiple_minima(start):
    (_, f_t, _, g_t, t, _, _, ok), _, (f0, df0) = search(multiple_minima, start)
    # TFP asserts the exact Wolfe conditions. Its bracketing differs from LineSearches.jl's, which
    # (as here) accepts the first trial that satisfies the exact or the approximate conditions;
    # from a start of 2 that is a point on the way down to the first minimum, with a positive slope.
    assert ok and f_t.item() <= f0
    assert is_exact_wolfe(t, f_t.item(), g_t.item(), f0, df0) or is_approx_wolfe(f_t.item(), g_t.item(), f0, df0)


def test_tfp_rosenbrock_along_a_descent_direction():
    x0 = torch.tensor([-3.0, -4.0], dtype=F64)
    direction = torch.tensor([0.5, 1.0], dtype=F64)

    def phi(t):
        x = (x0 + t * direction).requires_grad_(True)
        f = rosen(x)
        (g,) = torch.autograd.grad(f, x)
        return f.item(), torch.dot(g, direction).item()

    assert phi(0.0)[1] < 0
    (_, f_t, _, g_t, t, _, _, ok), _, (f0, df0) = search(phi, 1.0)
    assert ok and is_exact_wolfe(t, f_t.item(), g_t.item(), f0, df0)


def test_tfp_approx_wolfe_conditions_apply_close_to_the_minimum():
    # Based on Hager and Zhang (2005): the argument is shifted so that the minimum is extremely
    # close to zero, where the loss differences are rounding noise.
    shift = 0.99999999255000149

    def phi(x):
        return 1.0 - 2.0 * (x + shift) + (x + shift) ** 2, -2.0 + 2.0 * (x + shift)

    (_, f_t, _, g_t, _, _, _, ok), tried, (f0, df0) = search(phi, 1e-8)
    # TFP asserts that the exact conditions fail here, which depends on how its arithmetic rounds
    # the loss (f0 is about 1e-16, all noise); what matters is that a point is accepted at once,
    # without chasing noise, and that it satisfies the approximate conditions.
    assert ok and tried == [1e-8]
    assert f_t.item() <= f0 and is_approx_wolfe(f_t.item(), g_t.item(), f0, df0)


def test_approx_wolfe_may_exceed_the_loss_by_epsilon():
    # The cubic is 1 at 0 with slope -1e-6 and 1 + 5e-7 at 1 with slope -0.85e-6: the first trial
    # satisfies the approximate Wolfe conditions, as in the references, though it is higher than the
    # starting value (by less than epsilon * |f|). With epsilon = 0 it is not acceptable.
    def phi(t):
        a, b, e = -1e-6, 4.35e-6, -2.85e-6
        return 1 + a * t + b * t**2 + e * t**3, a + 2 * b * t + 3 * e * t**2

    (_, f_t, _, g_t, t, _, _, ok), tried, (f0, df0) = search(phi, 1.0)
    assert is_approx_wolfe(f_t.item(), g_t.item(), f0, df0) and f_t.item() > f0
    assert ok and t == 1.0 and tried == [1.0]
    (_, f_t, _, _, _, _, _, ok), _, _ = search(phi, 1.0, epsilon=0.0, max_ls=3)
    assert not ok or f_t.item() <= f0


def test_tfp_determinism():
    runs = [search(lambda x: ((x - 1.8) ** 2, 2 * (x - 1.8)), 0.9) for _ in range(2)]
    assert runs[0][1] == runs[1][1]
    assert runs[0][0][4] == runs[1][0][4] and runs[0][0][7] and runs[1][0][7]


def test_tfp_consistency():
    def rastrigin(x):
        z = x - 0.25
        return 10.0 + z * z - 10 * math.cos(2 * math.pi * z), 2 * z + 20 * math.pi * math.sin(2 * math.pi * z)

    (_, f_t, _, g_t, t, _, _, ok), _, _ = search(rastrigin, 0.1)
    assert ok
    f_ref, df_ref = rastrigin(t)
    assert f_t.item() == pytest.approx(f_ref) and g_t.item() == pytest.approx(df_ref)


def test_tfp_inf_recovery():
    # The first step does not bracket, and the expansion hits an infinite value (cut at 4).
    def phi(x):
        return (math.inf, math.nan) if x > 4 else ((x - 2.0) ** 2, 2 * (x - 2.0))

    (_, f_t, _, g_t, t, _, _, ok), _, (f0, df0) = search(phi, 1.0)
    assert ok and is_exact_wolfe(t, f_t.item(), g_t.item(), f0, df0)


# ---- the search's own contracts ----


def test_t_max_is_respected():
    (_, _, _, _, t, _, _, ok), tried, _ = search(lambda t: ((t - 100.0) ** 2, 2 * (t - 100.0)), 1.0, t_max=3.0)
    assert ok and t <= 3.0 and max(tried) <= 3.0


@pytest.mark.parametrize("max_ls", [0, 1, 3, 6])
def test_budget_is_exact_and_no_point_is_evaluated_twice(max_ls):
    gen = torch.Generator().manual_seed(max_ls)
    for _ in range(40):
        w = torch.rand(3, generator=gen, dtype=F64) * 6 + 0.5
        p = torch.rand(3, generator=gen, dtype=F64) * 6.28

        def phi(t, w=w, p=p):
            f = -t + 0.3 * t * t + sum(0.4 * math.sin(w[i].item() * t + p[i].item()) for i in range(3))
            df = -1 + 0.6 * t + sum(0.4 * w[i].item() * math.cos(w[i].item() * t + p[i].item()) for i in range(3))
            return f, df

        if phi(0.0)[1] >= 0:
            continue
        (_, f_t, _, g_t, t, _, _, ok), tried, (f0, df0) = search(phi, 2.0, max_ls=max_ls)
        assert len(tried) <= max_ls + 1
        assert len(set(tried)) == len(tried)
        if ok:  # at most epsilon above the start, and enough decrease or the Wolfe conditions
            assert f_t.item() <= f0 + EPSILON * abs(f0)
            assert f_t.item() <= f0 + 1e-4 * t * df0 or is_exact_wolfe(t, f_t.item(), g_t.item(), f0, df0)


def test_a_step_never_ends_at_zero():
    # The loss falls only up to t = 1e-5 and is far higher at the first trial. With no budget to
    # look further, the search fails at that trial; with a budget it may find a point, but it
    # never returns the start (t = 0), whether it counts as a success or not.
    def phi(t):
        return -1e-3 * t + 50 * t * t, -1e-3 + 100 * t

    (_, _, _, _, t, _, _, ok), tried, _ = search(phi, 1.0, max_ls=0)
    assert not ok and t == 1.0 and tried == [1.0]
    for max_ls in (1, 2, 3, 10):
        (_, f_t, _, _, t, _, _, ok), _, (f0, _) = search(phi, 1.0, max_ls=max_ls)
        assert t > 0
        assert not ok or f_t.item() <= f0


def test_broken_bracket_falls_back_instead_of_looping():
    # A gradient that disagrees with the values (here: always positive) breaks the bracket
    # invariant; the search must end within its budget, as a failure.
    (_, _, _, _, t, _, _, _), tried, _ = search(
        lambda t: (-t if t == 0 else 1.0 + t, -1.0 if t == 0 else 1.0), 1.0, max_ls=20
    )
    assert len(tried) <= 21 and t > 0


# ---- piazzi.lbfgs(line_search="hager_zhang") ----


def ceres_rosenbrock(x):
    return (1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2


OPTIONS = {"max_iter": 500, "step_tol": 1e-12, "line_search": "hager_zhang"}


def test_lbfgs_ceres_rosenbrock_and_final_cost_is_zero():
    res = piazzi.lbfgs(ceres_rosenbrock, torch.tensor([-1.2, 0.0], dtype=F64), **OPTIONS)
    assert res.converged, res.message
    assert torch.allclose(res.x, torch.ones(2, dtype=F64), atol=1e-9, rtol=0)
    res = piazzi.lbfgs(lambda x: (x**2).sum(), torch.tensor([2.0], dtype=F64), **OPTIONS)
    assert res.converged and res.fun.item() <= torch.finfo(F64).eps


@pytest.mark.parametrize("problem", SCALAR_PROBLEMS, ids=[p.name for p in SCALAR_PROBLEMS])
def test_lbfgs_scalar_problems(problem):
    res = piazzi.lbfgs(problem.fun, problem.x0, **OPTIONS)
    assert res.converged, res.message
    assert torch.allclose(res.x, problem.x_opt, atol=1e-5)


def test_lbfgs_counts_and_ignores_value_fun():
    calls = 0

    def fun(x):
        nonlocal calls
        calls += 1
        return rosen(x)

    def value_fun(x):
        raise AssertionError("Hager-Zhang needs gradients and must not call value_fun")

    res = piazzi.lbfgs(fun, torch.tensor([-1.2, 1.0], dtype=F64), value_fun=value_fun, **OPTIONS)
    assert res.converged and res.n_fun == 0 and res.n_fun_grad == calls


def test_lbfgs_steps_never_rise_by_more_than_epsilon():
    seen = []
    piazzi.lbfgs(
        rosen,
        torch.tensor([-1.2, 1.0], dtype=F64),
        callback=lambda info: seen.append((info.fun_start.item(), info.fun.item(), info.accepted)) and False,
        **OPTIONS,
    )
    assert seen
    assert all(f <= f_start + 1e-6 * abs(f_start) for f_start, f, accepted in seen if accepted)
    seen.clear()
    piazzi.lbfgs(
        rosen,
        torch.tensor([-1.2, 1.0], dtype=F64),
        hz_epsilon=0.0,
        callback=lambda info: seen.append((info.fun_start.item(), info.fun.item(), info.accepted)) and False,
        **{k: v for k, v in OPTIONS.items() if k != "hz_epsilon"},
    )
    assert all(f <= f_start for f_start, f, accepted in seen if accepted)


@pytest.mark.parametrize("scale", [1e-9, 1.0, 1e9])
def test_lbfgs_is_scale_invariant(scale):
    x0 = torch.tensor([-1.2, 1.0], dtype=F64)
    ref = piazzi.lbfgs(rosen, x0, **OPTIONS)
    res = piazzi.lbfgs(lambda x: scale * rosen(x), x0, **OPTIONS)
    assert res.converged, res.message
    assert torch.allclose(res.x, ref.x, atol=1e-6)


def test_lbfgs_float32():
    res = piazzi.lbfgs(rosen, torch.tensor([-1.2, 1.0]), max_iter=500, line_search="hager_zhang")
    assert res.converged, res.message
    assert torch.allclose(res.x, torch.ones(2), atol=1e-3)


def test_lbfgs_respects_max_step():
    xs = []
    piazzi.lbfgs(
        lambda x: 0.5 * ((x - 100.0) ** 2).sum(),
        torch.zeros(3, dtype=F64),
        line_search="hager_zhang",
        max_step=0.5,
        callback=lambda info: xs.append(info.x.clone()) and False,
        max_iter=20,
    )
    steps = [(b - a).abs().max().item() for a, b in zip([torch.zeros(3, dtype=F64), *xs], xs, strict=False)]
    assert max(steps) <= 0.5 * (1 + 1e-12)


def test_lbfgs_non_finite_trial_never_passes():
    def fun(x):
        return torch.where(x < 0.5, (x - 0.3) ** 2, torch.full_like(x, float("nan"))).sum()

    res = piazzi.lbfgs(fun, torch.tensor([0.0], dtype=F64), **OPTIONS)
    assert math.isfinite(res.fun.item()) and abs(res.x.item() - 0.3) < 1e-6


@pytest.mark.parametrize(("c1", "c2", "epsilon"), [(0.5, 0.9, 1e-6), (0.1, 0.05, 1e-6), (0.1, 0.9, -1.0)])
def test_lbfgs_rejects_inconsistent_constants(c1, c2, epsilon):
    with pytest.raises(ValueError, match="hager_zhang needs"):
        piazzi.lbfgs(rosen, torch.zeros(2, dtype=F64), line_search="hager_zhang", c1=c1, c2=c2, hz_epsilon=epsilon)


def test_lbfgs_matches_the_other_searches_on_the_minimiser():
    x0 = torch.tensor([-1.2, 1.0], dtype=F64)
    ref = piazzi.lbfgs(
        rosen, x0, line_search="strong_wolfe", **{k: v for k, v in OPTIONS.items() if k != "line_search"}
    )
    res = piazzi.lbfgs(rosen, x0, **OPTIONS)
    assert res.converged and ref.converged
    assert torch.allclose(res.x, ref.x, atol=1e-6)
