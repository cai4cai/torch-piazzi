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


def test_minimize_dispatches_to_lbfgs(x0):
    options = {"max_iter": 500, "step_tol": 1e-10}
    res = piazzi.minimize(rosenbrock, x0, options=options)
    ref = piazzi.lbfgs(rosenbrock, x0, **options)
    assert torch.equal(res.x, ref.x) and res.n_fun_grad == ref.n_fun_grad
    assert torch.allclose(res.x, ONES, atol=1e-4)
    assert torch.equal(piazzi.minimize(rosenbrock, x0, method="LBFGS", options=options).x, ref.x)


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
    with pytest.raises(ValueError, match="lbfgs"):
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
    ("method", "kw"),
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


@needs_scipy
def test_scipy_args_are_bound():
    target = torch.tensor([1.0, -2.0], dtype=torch.float64)

    def fun(x, c):
        return ((x - c) ** 2).sum()

    def grad(x, c):
        return 2 * (x - c)

    def hessp(x, p, c):
        return 2 * p

    x0 = torch.zeros(2, dtype=torch.float64)
    for kw in ({}, {"jac": grad}, {"jac": grad, "hessp": hessp, "method": "trust-ncg"}):
        res = pzs.minimize(fun, x0, args=(target,), **kw)
        assert torch.allclose(res.x, target, atol=1e-6), kw
    res = pzs.minimize(lambda x, c: ((x - c) ** 2).sum(), torch.zeros(1, dtype=torch.float64), args=3.0)
    assert torch.allclose(res.x, torch.tensor([3.0], dtype=torch.float64), atol=1e-6)


@needs_scipy
def test_scipy_rejects_complex_step(x0):
    with pytest.raises(ValueError, match="complex-step"):
        pzs.minimize(rosenbrock, x0, jac="cs")
    with pytest.raises(ValueError, match="complex-step"):
        pzs.minimize(rosenbrock, x0, method="trust-constr", hess="cs")


@needs_scipy
def test_scipy_hessp_autodiff_affine_objective():
    """An affine objective has a constant gradient with no graph; its Hessian-vector product is 0."""
    res = pzs.minimize(
        lambda x: x.sum(),
        torch.zeros(2, dtype=torch.float64),
        method="trust-ncg",
        hessp="autodiff",
        options={"maxiter": 3},
    )
    assert res.n_iter <= 3


# --- Regression tests from the code review of the SciPy wrappers ----------------------------


@needs_scipy
@pytest.mark.parametrize("method", ["Nelder-Mead", "Powell", "COBYLA"])
def test_scipy_gradient_free_methods_get_no_gradient(method, x0):
    calls = []

    def fun(x):
        calls.append(x.requires_grad)
        return rosenbrock(x)

    res = pzs.minimize(fun, x0, method=method)  # SciPy would warn about an unused gradient
    assert res.n_fun_grad == 0 and res.n_fun in (len(calls), len(calls) - 1)  # final re-evaluation not counted
    assert not any(calls)


@needs_scipy
@pytest.mark.parametrize(
    ("method", "kind"), [("trust-exact", "hess"), ("Newton-CG", "hessp"), ("trust-krylov", "hessp")]
)
def test_scipy_counts_hessian_evaluations(method, kind, x0):
    calls = []

    def hess(x):
        calls.append(None)
        return torch.autograd.functional.hessian(rosenbrock, x)

    def hessp(x, p):
        return hess(x) @ p

    res = pzs.minimize(rosenbrock, x0, method=method, **{kind: hess if kind == "hess" else hessp})
    assert res.n_system == len(calls) > 0
    ref = pzs.minimize(rosenbrock, x0, method=method, **{kind: "autodiff"})
    assert ref.n_system == res.n_system  # the same path, so the same count with autodiff


@needs_scipy
@pytest.mark.parametrize("jac", ["2-point", "3-point"])
def test_scipy_finite_differences_are_value_only_calls(jac, x0):
    calls = []

    def fun(x):
        calls.append(None)
        return rosenbrock(x)

    res = pzs.minimize(fun, x0, method="BFGS", jac=jac)
    assert res.n_fun_grad == 0
    assert res.n_fun in (len(calls), len(calls) - 1)  # a final re-evaluation is not counted


@needs_scipy
def test_scipy_callback_gets_iteration_info_and_can_stop(x0):
    seen = []

    def callback(info):
        seen.append(info)
        return info.iteration == 4

    res = pzs.minimize(lambda x: (rosenbrock(x), {"x0": x[0] * 1}), x0, method="BFGS", has_aux=True, callback=callback)
    assert len(seen) == 5 and res.n_iter == 5 and not res.converged
    assert all(isinstance(info, piazzi.IterationInfo) for info in seen)
    assert seen[0].fun_start.item() == pytest.approx(rosenbrock(x0).item())
    assert seen[-1].aux is not None and torch.equal(seen[-1].aux["x0"], seen[-1].x[0])
    assert torch.equal(seen[-1].x, res.x)
