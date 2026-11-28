"""Tests of ``piazzi.torchmin``: pytorch-minimize behind piazzi's calling convention."""

import itertools
import subprocess
import sys

import pytest
import torch
from problems import F64, SCALAR_PROBLEMS, entropy, rosen

import piazzi

try:
    import piazzi.torchmin as pzt
except ImportError:  # pytorch-minimize is optional
    pzt = None

needs_torchmin = pytest.mark.skipif(pzt is None, reason="pytorch-minimize is not installed")

# trust-krylov is left out of the problem tests: torchmin 0.1.0 raises "reducible T matrix
# encountered" when its Lanczos process exhausts the space (see test_trust_krylov_upstream_failure).
METHODS = ["bfgs", "l-bfgs", "cg", "newton-cg", "newton-exact", "dogleg", "trust-ncg", "trust-exact"]
# BFGS and L-BFGS also stop on |f change| <= xtol (1e-8 by default), which can be before the gradient is small.
STRICT = {"bfgs": {"xtol": 1e-14, "gtd_tol": 0.0}, "l-bfgs": {"xtol": 1e-14, "gtd_tol": 0.0}}
SCALAR_IDS = [p.name for p in SCALAR_PROBLEMS]


def test_import_piazzi_does_not_import_torchmin():
    code = "import sys, piazzi; assert 'torchmin' not in sys.modules and 'piazzi.torchmin' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


@needs_torchmin
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("problem", SCALAR_PROBLEMS, ids=SCALAR_IDS)
def test_problems(problem, method):
    """SciPy: CheckOptimize (entropy), test_rosenbrock, test_result_x_shape_when_len_x_is_one."""
    if (problem.name, method) == ("entropy", "dogleg"):
        pytest.skip("torchmin's dogleg fails its Cholesky factorisation on a Hessian that is not positive definite")
    res = pzt.minimize(problem.fun, problem.x0, method, options={"max_iter": 2000, **STRICT.get(method, {})})
    assert res.converged, res.message
    assert res.x.shape == problem.x0.shape and res.x.dtype == problem.x0.dtype
    assert res.fun.item() == pytest.approx(problem.fun(problem.x_opt).item(), abs=1e-6)
    if problem.name != "entropy":  # whose minimiser is not unique in its first entry
        assert torch.allclose(res.x, problem.x_opt, atol=1e-2), res.x
    assert torch.allclose(res.fun, problem.fun(res.x))


@needs_torchmin
@pytest.mark.parametrize("method", METHODS)
def test_counts_are_the_calls_made(method):
    calls = []

    def fun(x):
        calls.append(x.detach().clone())
        return rosen(x)

    res = pzt.minimize(fun, torch.tensor([-1.2, 1.0], dtype=F64), method)
    assert res.n_fun_grad == len(calls) == res.raw.nfev
    assert res.n_fun == 0 and res.n_system == 0
    repeats = len(calls) - len({tuple(c.tolist()) for c in calls})
    if method in ("newton-cg", "newton-exact"):
        # torchmin evaluates the point its line search accepts a second time, to build the Hessian.
        assert repeats > 0
    else:
        assert repeats == 0


@needs_torchmin
@pytest.mark.parametrize("method", ["l-bfgs", "trust-exact", "newton-cg"])
def test_callback_info_and_aux(method):
    infos = []
    x0 = torch.tensor([-1.2, 1.0], dtype=F64)
    res = pzt.minimize(
        lambda x: (rosen(x), {"x0": x[0]}),
        x0,
        method,
        has_aux=True,
        callback=lambda info: infos.append(info),
    )
    assert [i.iteration for i in infos] == list(range(res.n_iter))
    for i in infos:
        assert i.x.shape == x0.shape and not i.fun.requires_grad and not i.aux["x0"].requires_grad
        assert torch.allclose(i.fun, rosen(i.x)) and torch.equal(i.aux["x0"], i.x[0])
    assert infos[-1].n_fun_grad <= res.n_fun_grad
    assert torch.equal(res.aux["x0"], res.x[0]) and torch.allclose(res.fun, rosen(res.x))
    assert torch.allclose(infos[-1].x, res.x, atol=1e-4)
    assert torch.equal(infos[0].fun_start, rosen(x0))
    # The iterates were cloned for the callback: later iterations did not overwrite them.
    assert len({tuple(i.x.tolist()) for i in infos}) > 1


@needs_torchmin
def test_callback_stops_and_reports_the_stopping_point():
    seen = []

    def callback(info):
        seen.append(info)
        return info.iteration == 2

    res = pzt.minimize(rosen, torch.tensor([-1.2, 1.0], dtype=F64), "bfgs", callback=callback)
    assert len(seen) == res.n_iter == 3
    assert not res.converged and "callback" in res.message
    assert torch.equal(res.x, seen[-1].x) and torch.allclose(res.fun, rosen(res.x))


@needs_torchmin
def test_rejected_trust_region_step_is_not_accepted():
    infos = []
    pzt.minimize(rosen, torch.tensor([-1.2, 1.0], dtype=F64), "trust-exact", callback=infos.append)
    rejected = [i for i in infos if not i.accepted]
    assert rejected  # this start has rejected steps
    for a, b in itertools.pairwise(infos):
        if not b.accepted:
            assert torch.equal(a.x, b.x)


@needs_torchmin
def test_x0_is_not_modified_or_aliased():
    x0 = torch.tensor([-1.2, 1.0], dtype=F64)
    keep = x0.clone()
    points = []
    res = pzt.minimize(lambda x: (points.append(x), rosen(x))[1], x0, "bfgs")
    assert torch.equal(x0, keep)
    assert res.x.data_ptr() != x0.data_ptr()
    assert all(p.data_ptr() != x0.data_ptr() for p in points)


@needs_torchmin
def test_float32():
    x0 = torch.tensor([-1.2, 1.0])
    res = pzt.minimize(rosen, x0, "l-bfgs", options={"gtol": 1e-3, "xtol": 1e-6})
    assert res.x.dtype == torch.float32 and res.fun.dtype == torch.float32
    assert torch.allclose(res.x, torch.ones(2), atol=1e-2)


@needs_torchmin
def test_x_of_any_shape():
    res = pzt.minimize(lambda x: ((x - 3.0) ** 2).sum(), torch.zeros(2, 3, dtype=F64), "trust-exact")
    assert res.x.shape == (2, 3) and torch.allclose(res.x, torch.full((2, 3), 3.0, dtype=F64))


@needs_torchmin
def test_manifold_is_rejected():
    with pytest.raises(ValueError, match="manifold"):
        pzt.minimize(rosen, torch.zeros(3, dtype=F64), manifold=piazzi.Euclidean())


@needs_torchmin
def test_bad_arguments():
    x0 = torch.zeros(2, dtype=F64)
    with pytest.raises(ValueError, match="method"):
        pzt.minimize(rosen, x0, "nelder-mead")
    with pytest.raises(ValueError, match="callback"):
        pzt.minimize(rosen, x0, options={"callback": print})
    with pytest.raises(TypeError, match="floating"):
        pzt.minimize(rosen, torch.zeros(2, dtype=torch.int64))
    assert pzt.minimize(rosen, x0, "L-BFGS").converged  # case insensitive
    options = {"max_iter": 3}
    pzt.minimize(rosen, x0, options=options)
    assert options == {"max_iter": 3}  # not modified


@needs_torchmin
def test_nonfinite_value_is_not_converged():
    res = pzt.minimize(lambda x: x.sum() * torch.nan, torch.ones(2, dtype=F64), "l-bfgs")
    assert not res.converged


@needs_torchmin
def test_registered_in_minimize_methods():
    x0 = torch.tensor([-1.2, 1.0], dtype=F64)
    res = piazzi.minimize(rosen, x0, method="torchmin-trust-exact", options={"gtol": 1e-8})
    ref = pzt.minimize(rosen, x0, "trust-exact", options={"gtol": 1e-8})
    assert torch.equal(res.x, ref.x) and res.n_fun_grad == ref.n_fun_grad
    assert {f"torchmin-{m}" for m in pzt.METHODS} <= set(piazzi.MINIMIZE_METHODS)
    with pytest.raises(ValueError, match="manifold"):
        piazzi.minimize(rosen, x0, method="torchmin-bfgs", manifold=piazzi.Euclidean())


@needs_torchmin
def test_entropy_matches_native_value():
    x0 = torch.zeros(3, dtype=F64)
    res = pzt.minimize(entropy, x0, "trust-exact")
    ref = piazzi.minimize(entropy, x0, options={"max_iter": 2000, "step_tol": 1e-10})
    assert res.fun.item() == pytest.approx(ref.fun.item(), abs=1e-8)


@needs_torchmin
def test_trust_krylov_upstream_failure():
    """torchmin 0.1.0's trust-krylov raises when its Lanczos process breaks down; piazzi does not hide it."""
    with pytest.raises(RuntimeError, match="reducible"):
        pzt.minimize(rosen, torch.tensor([-1.2, 1.0], dtype=F64), "trust-krylov")


@needs_torchmin
@pytest.mark.parametrize("method", ["bfgs", "l-bfgs", "newton-exact"])
def test_xtol_stall_at_a_large_gradient_is_not_converged(method):
    options = {"xtol": 0.5}  # torchmin then reports success after a step or a decrease below 0.5
    res = pzt.minimize(rosen, torch.tensor([-1.2, 1.0], dtype=F64), method, options=options)
    assert res.raw.success and res.raw.grad.abs().max() > 1e-2
    assert not res.converged and "not a minimiser" in res.message


@needs_torchmin
def test_converged_scale_invariant():
    scale = 1e-9
    res = pzt.minimize(lambda x: scale * rosen(x), torch.tensor([-1.2, 1.0], dtype=F64), "bfgs", options={"xtol": 0.5})
    assert not res.converged


@needs_torchmin
def test_callback_stop_without_a_recorded_evaluation_evaluates_again(monkeypatch):
    from piazzi.torchmin import _minimize

    monkeypatch.setattr(_minimize, "_HISTORY", 0)  # every lookup misses
    res = pzt.minimize(rosen, torch.tensor([-1.2, 1.0], dtype=F64), "bfgs", callback=lambda info: info.iteration == 1)
    assert torch.isfinite(res.fun) and torch.allclose(res.fun, rosen(res.x))
    assert res.n_fun_grad == res.raw.nfev + 1


@needs_torchmin
def test_aux_is_not_shared_between_callback_and_result():
    seen = []

    def callback(info):
        seen.append(info.aux["r"])
        info.aux["r"].zero_()  # a callback is free to scribble on what it is handed

    res = pzt.minimize(
        lambda x: (rosen(x), {"r": x.detach() * 1.0}),
        torch.tensor([-1.2, 1.0], dtype=F64),
        "bfgs",
        has_aux=True,
        callback=callback,
    )
    assert torch.equal(res.aux["r"], res.x) and res.aux["r"].abs().sum() > 0
