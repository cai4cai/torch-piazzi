import importlib
import sys

import pytest
import torch

import piazzi

try:
    import piazzi.scipy as pzs
except ImportError:  # SciPy is optional
    pzs = None

needs_scipy = pytest.mark.skipif(pzs is None, reason="SciPy is not installed")

THETA = torch.tensor([2.5, 1.3], dtype=torch.float64)


@pytest.fixture
def outlier_data():
    t = torch.linspace(0, 4, 40, dtype=torch.float64)
    y = 2.5 * torch.exp(-1.3 * t)
    y[::10] += 2.0
    return t, y


def residuals(t, y):
    return lambda th: th[0] * torch.exp(-th[1] * t) - y


def test_import_error_without_scipy(monkeypatch):
    monkeypatch.setitem(sys.modules, "scipy", None)
    monkeypatch.setitem(sys.modules, "scipy.optimize", None)
    for name in ("piazzi.scipy", "piazzi.scipy._least_squares"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    with pytest.raises(ImportError, match=r"piazzi\[scipy\]"):
        importlib.import_module("piazzi.scipy")


@needs_scipy
@pytest.mark.parametrize("loss", ["linear", "huber", "soft_l1", "cauchy", "arctan"])
def test_matches_native_least_squares(loss, outlier_data):
    t, y = outlier_data
    fun, x0 = residuals(t, y), torch.tensor([1.0, 0.5], dtype=torch.float64)
    ref = piazzi.least_squares(fun, x0, loss=loss, f_scale=0.1, step_tol=1e-12, max_iter=500)
    res = pzs.least_squares(fun, x0, loss=loss, f_scale=0.1, xtol=1e-14, ftol=1e-14, gtol=1e-14)
    assert res.converged, res.message
    assert torch.allclose(res.x, ref.x, atol=1e-6)
    assert torch.allclose(res.fun, ref.fun, rtol=1e-8)
    assert res.raw.cost == pytest.approx(res.fun.item())
    assert res.n_fun == res.raw.nfev and res.n_system == res.raw.njev


@needs_scipy
@pytest.mark.parametrize("jac", ["autodiff", "2-point", "callable"])
def test_jacobian_options(jac, outlier_data):
    t, y = outlier_data
    y = 2.5 * torch.exp(-1.3 * t)  # no outliers: exact fit
    if jac == "callable":
        jac = lambda th: torch.stack([torch.exp(-th[1] * t), -th[0] * t * torch.exp(-th[1] * t)], dim=1)  # noqa: E731
    res = pzs.least_squares(residuals(t, y), torch.tensor([1.0, 0.5], dtype=torch.float64), jac=jac)
    assert torch.allclose(res.x, THETA, atol=1e-6)


@needs_scipy
def test_callable_loss_matches_builtin(outlier_data):
    t, y = outlier_data
    fun, x0 = residuals(t, y), torch.tensor([1.0, 0.5], dtype=torch.float64)
    a = pzs.least_squares(fun, x0, loss="cauchy", f_scale=0.1)
    b = pzs.least_squares(fun, x0, loss=torch.log1p, f_scale=0.1)
    assert torch.allclose(a.x, b.x, atol=1e-10)


@needs_scipy
def test_keeps_shape_dtype_and_converts_bounds():
    target = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    lb = torch.full((2, 3), -1.0)
    ub = torch.full((2, 3), 3.5)
    res = pzs.least_squares(lambda x: (x - target).reshape(-1), torch.ones(2, 3), bounds=(lb, ub))
    assert res.x.shape == (2, 3) and res.x.dtype == torch.float32 and res.fun.dtype == torch.float32
    assert torch.allclose(res.x, target.clamp(max=3.5), atol=1e-5)


@needs_scipy
def test_lm_method_and_counts(outlier_data):
    t, y = outlier_data
    y = 2.5 * torch.exp(-1.3 * t)
    res = pzs.least_squares(residuals(t, y), torch.tensor([1.0, 0.5], dtype=torch.float64), method="lm")
    assert torch.allclose(res.x, THETA, atol=1e-6)
    assert res.n_iter >= 1 and res.n_fun >= 1
