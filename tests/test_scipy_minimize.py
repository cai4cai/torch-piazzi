import pytest
import torch

import piazzi

try:
    import scipy.optimize

    import piazzi.scipy as pzs
except ImportError:  # SciPy is optional
    pzs = None

needs_scipy = pytest.mark.skipif(pzs is None, reason="SciPy is not installed")

ONES = torch.ones(2, dtype=torch.float64)


def rosenbrock(x):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


@pytest.fixture
def x0():
    return torch.tensor([-1.2, 1.0], dtype=torch.float64)


# piazzi.minimize


def test_minimize_dispatches_to_lbfgs_armijo(x0):
    options = {"max_iter": 500, "step_tol": 1e-10}
    res = piazzi.minimize(rosenbrock, x0, options=options)
    ref = piazzi.lbfgs_armijo(rosenbrock, x0, **options)
    assert torch.equal(res.x, ref.x) and res.n_fun_grad == ref.n_fun_grad
    assert torch.allclose(res.x, ONES, atol=1e-4)
    assert torch.equal(piazzi.minimize(rosenbrock, x0, method="LBFGS_Armijo", options=options).x, ref.x)


def test_minimize_aux_and_callback(x0):
    seen = []
    res = piazzi.minimize(
        lambda x: (rosenbrock(x), {"x0": x[0]}),
        x0,
        has_aux=True,
        callback=lambda info: seen.append(info.iteration),
        options={"max_iter": 5},
    )
    assert torch.equal(res.aux["x0"], res.x[0]) and seen == list(range(res.n_iter))


def test_minimize_unknown_method(x0):
    with pytest.raises(ValueError, match="lbfgs_armijo"):
        piazzi.minimize(rosenbrock, x0, method="BFGS")


# piazzi.scipy.minimize


@needs_scipy
@pytest.mark.parametrize("method", [None, "BFGS", "L-BFGS-B", "CG"])
def test_scipy_first_order_methods(method, x0):
    res = pzs.minimize(rosenbrock, x0, method=method, options={"gtol": 1e-9} if method != "L-BFGS-B" else None)
    assert res.converged, res.message
    assert torch.allclose(res.x, ONES, atol=1e-4)
    assert res.fun.item() == pytest.approx(res.raw.fun)
    assert res.n_fun_grad == res.raw.nfev and res.n_fun == 0 and res.n_iter == res.raw.nit


@needs_scipy
@pytest.mark.parametrize(
    "method, kw",
    [
        ("Newton-CG", {"hess": "autodiff"}),
        ("trust-exact", {"hess": "autodiff"}),
        ("trust-ncg", {"hessp": "autodiff"}),
        ("trust-constr", {"hess": "autodiff"}),
    ],
)
def test_scipy_second_order_methods(method, kw, x0):
    res = pzs.minimize(rosenbrock, x0, method=method, **kw)
    assert torch.allclose(res.x, ONES, atol=1e-4), res.message


@needs_scipy
def test_scipy_callable_derivatives_match_autodiff(x0):
    def grad(x):
        return torch.stack([-2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2), 200 * (x[1] - x[0] ** 2)])

    def hessp(x, p):
        H = torch.tensor([[1200 * x[0] ** 2 - 400 * x[1] + 2, -400 * x[0]], [-400 * x[0], 200.0]], dtype=x.dtype)
        return H @ p

    a = pzs.minimize(rosenbrock, x0, method="trust-ncg", jac=grad, hessp=hessp)
    b = pzs.minimize(rosenbrock, x0, method="trust-ncg", hessp="autodiff")
    assert torch.allclose(a.x, b.x, atol=1e-10)
    assert a.n_fun == a.raw.nfev and a.n_fun_grad == a.raw.njev


@needs_scipy
def test_scipy_finite_difference_jac(x0):
    res = pzs.minimize(rosenbrock, x0, method="BFGS", jac="2-point")
    assert torch.allclose(res.x, ONES, atol=1e-3)
    assert res.n_fun == res.raw.nfev


@needs_scipy
def test_scipy_keeps_shape_dtype_aux_and_converts_bounds():
    target = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    def fun(x):
        loss = ((x - target) ** 2).sum()
        return loss, {"half": 0.5 * loss}

    bounds = scipy.optimize.Bounds(torch.full((2, 3), -1.0), torch.full((2, 3), 3.5))
    res = pzs.minimize(fun, torch.ones(2, 3), method="L-BFGS-B", bounds=bounds, has_aux=True)
    assert res.x.shape == (2, 3) and res.x.dtype == torch.float32 and res.fun.dtype == torch.float32
    assert torch.allclose(res.x, target.clamp(max=3.5), atol=1e-4)
    assert torch.allclose(res.aux["half"], 0.5 * res.fun)

    pairs = torch.tensor([[-1.0, 3.5]] * 6)
    res2 = pzs.minimize(fun, torch.ones(2, 3), method="L-BFGS-B", bounds=pairs, has_aux=True)
    assert torch.allclose(res2.x, res.x, atol=1e-5)


@needs_scipy
def test_scipy_matches_native_minimize(x0):
    ref = piazzi.minimize(rosenbrock, x0, options={"max_iter": 500, "step_tol": 1e-12})
    res = pzs.minimize(rosenbrock, x0, method="BFGS", options={"gtol": 1e-10})
    assert torch.allclose(res.x, ref.x, atol=1e-5)
