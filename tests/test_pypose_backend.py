import importlib
import subprocess
import sys

import pytest
import torch

import piazzi

try:
    import piazzi.pypose as pzp  # first: it silences PyPose's import-time warnings, which tests treat as errors
except ImportError:  # PyPose is optional
    pzp = pp = None
else:
    import pypose as pp

needs_pypose = pytest.mark.skipif(pzp is None, reason="PyPose is not installed")

THETA = torch.tensor([2.5, 1.3], dtype=torch.float64)


def exp_fit(dtype=torch.float64):
    t = torch.linspace(0, 4, 40, dtype=dtype)
    y = (THETA.to(dtype)[0] * torch.exp(-THETA.to(dtype)[1] * t)) + 0.01 * torch.sin(7 * t)
    return (lambda th: th[0] * torch.exp(-th[1] * t) - y), torch.tensor([1.0, 0.5], dtype=dtype)


def rosenbrock_residuals(x):
    return torch.stack([10 * (x[1] - x[0] ** 2), 1 - x[0]])


def test_import_piazzi_does_not_load_pypose():
    code = "import sys, piazzi; assert 'pypose' not in sys.modules and 'piazzi.pypose' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_import_error_without_pypose(monkeypatch):
    monkeypatch.setitem(sys.modules, "pypose", None)
    for name in ("piazzi.pypose", "piazzi.pypose._least_squares"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    with pytest.raises(ImportError, match=r"piazzi\[pypose\]"):
        importlib.import_module("piazzi.pypose")


@needs_pypose
@pytest.mark.parametrize("method", ["gn", "lm"])
@pytest.mark.parametrize(("dtype", "tol"), [(torch.float64, 1e-8), (torch.float32, 1e-3)])
def test_matches_native_solver(method, dtype, tol):
    fun, x0 = exp_fit(dtype)
    ref = piazzi.least_squares(fun, x0, method=method)
    res = pzp.least_squares(fun, x0, method=method)
    assert res.converged, res.message
    assert res.x.dtype == dtype and res.x.shape == x0.shape
    assert torch.allclose(res.x, ref.x, atol=tol, rtol=tol)
    assert abs(float(res.fun) - float(ref.fun)) <= tol * max(1.0, float(ref.fun))
    assert res.n_fun_grad == 0


@needs_pypose
@pytest.mark.parametrize("method", ["gn", "lm"])
def test_counts_are_exact_and_nothing_is_evaluated_twice(method):
    fun, x0 = exp_fit()
    plain, grad = [], []

    def counted(x):
        (grad if torch.is_grad_enabled() else plain).append(x.detach().clone())
        return fun(x)

    res = pzp.least_squares(counted, x0, method=method)
    assert res.n_fun == len(plain) and res.n_system == len(grad)
    # Plain evaluations are never repeated at a point (only the Jacobian's evaluation repeats
    # the residuals at its own point, as PyPose's does).
    for i in range(len(plain)):
        for j in range(i):
            assert not torch.equal(plain[i], plain[j])
    # x0, then one trial per iteration (a rejected Levenberg-Marquardt trial costs one more).
    assert res.n_system == res.n_iter
    if method == "gn":
        assert res.n_fun == res.n_iter + 1


@needs_pypose
def test_no_aliasing():
    fun, x0 = exp_fit()
    x0_before = x0.clone()
    seen = []

    def keep(x):
        seen.append(x)
        return fun(x)

    res = pzp.least_squares(keep, x0)
    assert torch.equal(x0, x0_before)
    res.x.add_(1.0)
    assert torch.equal(x0, x0_before)
    assert all(s is not res.x for s in seen)


@needs_pypose
@pytest.mark.parametrize("solver", ["pinv", "lstsq", "cholesky", "cg"])
@pytest.mark.parametrize("strategy", ["constant", "adaptive", "trust_region"])
def test_solver_and_strategy_options(solver, strategy):
    fun, x0 = exp_fit()
    options = {"cg": {"tol": 1e-12}}.get(solver)
    res = pzp.least_squares(
        fun,
        x0,
        solver=solver,
        strategy=strategy,
        solver_options=options,
        strategy_options={"damping": 1e-4} if strategy != "trust_region" else {"radius": 1e4},
    )
    assert res.converged, res.message
    assert torch.allclose(res.x, THETA, atol=1e-2)
    assert isinstance(res.raw, pp.optim.LevenbergMarquardt)


@needs_pypose
def test_accepts_pypose_objects():
    fun, x0 = exp_fit()
    res = pzp.least_squares(
        fun, x0, solver=pp.optim.solver.Cholesky(), strategy=pp.optim.strategy.Adaptive(damping=1e-3), reject=4
    )
    assert res.converged and res.raw.reject == 4


@needs_pypose
def test_gauss_newton_uphill_step_is_not_taken():
    # PyPose's Gauss-Newton has no line search: the first full step from here raises the cost.
    x0 = torch.tensor([-1.2, 1.0], dtype=torch.float64)
    res = pzp.least_squares(rosenbrock_residuals, x0, method="gn")
    assert not res.converged and "did not decrease" in res.message
    assert torch.equal(res.x, x0)
    assert float(res.fun) == pytest.approx(0.5 * float(rosenbrock_residuals(x0).square().sum()))


@needs_pypose
def test_levenberg_marquardt_never_returns_a_worse_point():
    x0 = torch.tensor([-1.2, 1.0], dtype=torch.float64)
    f0 = 0.5 * float(rosenbrock_residuals(x0).square().sum())
    res = pzp.least_squares(
        rosenbrock_residuals, x0, strategy="constant", strategy_options={"damping": 1e-3}, solver="lstsq"
    )
    assert float(res.fun) <= f0
    assert float(res.fun) == pytest.approx(0.5 * float(rosenbrock_residuals(res.x).square().sum()))
    res = pzp.least_squares(rosenbrock_residuals, x0, strategy="adaptive")
    assert res.converged, res.message
    assert torch.allclose(res.x, torch.ones(2, dtype=torch.float64), atol=1e-6)


@needs_pypose
def test_non_finite_cost_is_not_accepted():
    x0 = torch.tensor([1.0, 1.0], dtype=torch.float64)

    def fun(x):
        r = torch.stack([x[0] - 3, x[1] - 2])
        return r * torch.where(x[0] > 1.5, torch.nan, 1.0)

    res = pzp.least_squares(fun, x0, method="gn")
    assert not res.converged and torch.equal(res.x, x0) and torch.isfinite(res.fun)


@needs_pypose
def test_callback_info_and_stop():
    fun, x0 = exp_fit()
    infos = []
    res = pzp.least_squares(fun, x0, callback=lambda info: infos.append(info) and False)
    assert [i.iteration for i in infos] == list(range(res.n_iter))
    assert infos[-1].n_fun == res.n_fun and infos[-1].n_system == res.n_system
    assert torch.equal(infos[-1].x, res.x) and torch.allclose(infos[-1].fun, res.fun)
    assert all(i.accepted for i in infos) and infos[0].fun_start > infos[0].fun
    stopped = pzp.least_squares(fun, x0, callback=lambda info: True)
    assert stopped.n_iter == 1 and not stopped.converged and "callback" in stopped.message


@needs_pypose
def test_has_aux():
    fun, x0 = exp_fit()
    res = pzp.least_squares(lambda x: (fun(x), {"norm": x.norm()}), x0, has_aux=True)
    assert res.aux["norm"].requires_grad is False
    assert torch.allclose(res.aux["norm"], res.x.norm())


@needs_pypose
def test_lie_tensor_x0():
    gen = torch.Generator().manual_seed(0)
    target = pp.Exp(pp.se3(0.3 * torch.randn(6, generator=gen, dtype=torch.float64)))
    x0 = pp.identity_SE3(dtype=torch.float64)
    res = pzp.least_squares(lambda X: (X.Inv() @ target).Log(), x0)
    assert res.converged, res.message
    assert isinstance(res.x, pp.LieTensor) and res.x.ltype == x0.ltype
    assert torch.allclose(res.x.tensor(), target.tensor(), atol=1e-8)
    assert torch.equal(x0.tensor(), pp.identity_SE3(dtype=torch.float64).tensor())
    assert res.n_system == res.n_iter


@needs_pypose
def test_argument_errors():
    fun, x0 = exp_fit()
    with pytest.raises(ValueError, match="method"):
        pzp.least_squares(fun, x0, method="dogleg")
    with pytest.raises(ValueError, match="lm"):
        pzp.least_squares(fun, x0, method="gn", strategy="constant")
    with pytest.raises(ValueError, match="solver"):
        pzp.least_squares(fun, x0, solver="svd")
    with pytest.raises(ValueError, match="by name"):
        pzp.least_squares(fun, x0, solver=pp.optim.solver.PINV(), solver_options={"rcond": 1e-3})
    with pytest.raises(TypeError):
        pzp.least_squares(fun, torch.tensor([1, 2]))
    with pytest.raises(TypeError):
        pzp.least_squares(fun, x0, manifold=piazzi.Euclidean())


@needs_pypose
@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
def test_stays_on_the_device_of_x0(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is not available")
    t = torch.linspace(0, 4, 40, device=device)
    y = 2.5 * torch.exp(-1.3 * t)
    x0 = torch.tensor([1.0, 0.5], device=device)
    res = pzp.least_squares(lambda p: p[0] * torch.exp(-p[1] * t) - y, x0)
    assert res.converged and res.x.device == x0.device and res.fun.device == x0.device
    assert torch.allclose(res.x.cpu(), torch.tensor([2.5, 1.3]), atol=1e-4)


@needs_pypose
def test_user_prints_are_not_swallowed(capsys):
    fun, x0 = exp_fit()
    pzp.least_squares(lambda x: (print("evaluated"), fun(x))[1], x0, max_iter=1)
    assert "evaluated" in capsys.readouterr().out


@needs_pypose
def test_solver_failure_is_reported():
    def failing(A, b):
        raise RuntimeError("singular")

    fun, x0 = exp_fit()
    res = pzp.least_squares(fun, x0, solver=failing)
    assert not res.converged and "linear solver failed" in res.message and torch.equal(res.x, x0)


@needs_pypose
def test_convergence_is_relative_to_the_scale_of_x():
    t = torch.linspace(1, 2, 20, dtype=torch.float64)
    scale = 1e-8
    res = pzp.least_squares(lambda x: x[0] * t - 2.0 * scale * t, torch.tensor([0.5 * scale], dtype=torch.float64))
    assert torch.allclose(res.x, torch.tensor([2.0 * scale], dtype=torch.float64), rtol=1e-6)
