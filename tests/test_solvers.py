import pytest
import torch

import piazzi


def rosenbrock(x):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


def exp_residuals(theta, t, y):
    return theta[0] * torch.exp(-theta[1] * t) - y


@pytest.fixture
def exp_data():
    t = torch.linspace(0, 4, 40, dtype=torch.float64)
    y = 2.5 * torch.exp(-1.3 * t)
    return t, y


def test_lbfgs_armijo_rosenbrock():
    x0 = torch.tensor([-1.2, 1.0], dtype=torch.float64)
    res = piazzi.lbfgs_armijo(rosenbrock, x0, max_iter=500, step_tol=1e-10)
    assert torch.allclose(res.x, torch.ones(2, dtype=torch.float64), atol=1e-4)
    assert res.n_fun_grad > 0


def test_lbfgs_armijo_keeps_shape_aux_and_callback():
    target = torch.arange(6, dtype=torch.float64).reshape(2, 3)

    def fun(x):
        loss = ((x - target) ** 2).sum()
        return loss, {"half": 0.5 * loss}

    seen = []
    res = piazzi.lbfgs_armijo(
        fun,
        torch.zeros(2, 3, dtype=torch.float64),
        has_aux=True,
        max_step=10.0,
        callback=lambda info: seen.append(info.iteration),
    )
    assert res.x.shape == (2, 3)
    assert torch.allclose(res.x, target, atol=1e-6)
    assert torch.allclose(res.aux["half"], 0.5 * res.fun)
    assert seen == list(range(res.n_iter))


def test_lbfgs_callback_stop():
    res = piazzi.lbfgs_armijo(rosenbrock, torch.tensor([-1.2, 1.0]), max_iter=100, callback=lambda info: True)
    assert res.n_iter == 1 and res.message == "stopped by callback"


@pytest.mark.parametrize("method", ["gn", "lm"])
def test_gauss_newton_exponential_fit(method, exp_data):
    t, y = exp_data
    system = piazzi.least_squares_system(lambda th: exp_residuals(th, t, y))
    res = piazzi.gauss_newton(system, torch.tensor([1.0, 0.5], dtype=torch.float64), method=method, step_tol=1e-10)
    assert torch.allclose(res.x, torch.tensor([2.5, 1.3], dtype=torch.float64), atol=1e-6)
    assert res.converged, res.message


def test_gauss_newton_linear_one_step():
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 7.0]], dtype=torch.float64)
    b = torch.tensor([1.0, 0.0, 2.0], dtype=torch.float64)
    system = piazzi.least_squares_system(lambda x: A @ x - b)
    res = piazzi.gauss_newton(system, torch.zeros(2, dtype=torch.float64), method="gn", max_step=100.0, gn_ridge=0.0)
    expected = torch.linalg.lstsq(A, b).solution
    assert torch.allclose(res.x, expected, atol=1e-10)
    assert res.n_iter <= 2


def test_least_squares_system_matches_autograd(exp_data):
    t, y = exp_data
    theta = torch.tensor([1.7, 0.9], dtype=torch.float64)
    loss, g, H = piazzi.least_squares_system(lambda th: exp_residuals(th, t, y))(theta)
    th = theta.clone().requires_grad_(True)
    ref = 0.5 * (exp_residuals(th, t, y) ** 2).sum()
    (g_ref,) = torch.autograd.grad(ref, th)
    assert torch.allclose(loss, ref.detach())
    assert torch.allclose(g, g_ref)
    assert torch.allclose(H, H.T) and torch.linalg.eigvalsh(H).min() >= 0


def test_lm_float32_objective_value_fun(exp_data):
    t, y = (v.float() for v in exp_data)
    calls = {"value": 0}

    def value_fun(th):
        calls["value"] += 1
        return 0.5 * (exp_residuals(th, t, y) ** 2).sum()

    system = piazzi.least_squares_system(lambda th: exp_residuals(th, t, y))
    res = piazzi.levenberg_marquardt(system, torch.tensor([1.0, 0.5]), value_fun=value_fun)
    assert res.x.dtype == torch.float32
    assert torch.allclose(res.x, torch.tensor([2.5, 1.3]), atol=1e-3)
    assert calls["value"] == res.n_fun


def test_bad_method():
    with pytest.raises(ValueError):
        piazzi.gauss_newton(lambda x: None, torch.zeros(2), method="dogleg")


# --- Regression tests from the code review of the solvers ----------------------------------


def rosen_nd(x):
    return (100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2).sum()


@pytest.mark.parametrize("scale", [1.0, 1e-3, 1e-6, 1e-9])
def test_lbfgs_armijo_is_scale_invariant(scale):
    """Absolute thresholds made tiny objectives stall (1e-6) or stop at x0 as converged (1e-9)."""
    x0 = torch.tensor([-1.2, 1.0], dtype=torch.float64)
    res = piazzi.lbfgs_armijo(lambda x: scale * rosen_nd(x), x0, max_iter=500, step_tol=1e-9)
    assert res.converged, res.message
    assert torch.allclose(res.x, torch.ones(2, dtype=torch.float64), atol=1e-5)


def test_lbfgs_armijo_tiny_gradient_is_not_converged_at_x0():
    target = torch.tensor([3.0, -2.0], dtype=torch.float64)
    res = piazzi.lbfgs_armijo(lambda x: 1e-7 * ((x - target) ** 2).sum(), torch.zeros(2, dtype=torch.float64))
    assert res.converged and torch.allclose(res.x, target, atol=1e-6), (res.x, res.message)


def test_lbfgs_armijo_stops_on_a_non_finite_gradient():
    # sqrt((x - 1)^2) has a NaN gradient exactly at its minimiser, which the first step hits.
    res = piazzi.lbfgs_armijo(lambda x: ((x - 1) ** 2).sqrt().sum(), torch.zeros(1, dtype=torch.float64))
    assert res.message == "gradient is not finite" and res.n_fun_grad == 2 and res.n_fun == 0
    assert res.x.item() == 1.0 and res.fun.item() == 0.0


def test_lbfgs_armijo_does_not_repeat_a_failed_steepest_descent_search():
    points = []

    def wrong_gradient(x):  # value 2 |x|^2, but gradient -2 x: every search direction goes uphill
        points.append(x.detach().clone())
        return 3 * (x.detach() ** 2).sum() - (x**2).sum()

    res = piazzi.lbfgs_armijo(wrong_gradient, torch.ones(2, dtype=torch.float64))
    assert not res.converged and res.message == "line search failed"
    assert len({tuple(p.tolist()) for p in points}) == len(points)


def test_lbfgs_armijo_at_the_minimiser_evaluates_once():
    res = piazzi.lbfgs_armijo(rosen_nd, torch.ones(3, dtype=torch.float64))
    assert res.converged and res.message == "gradient is zero" and res.n_fun_grad == 1


@pytest.mark.parametrize("n", [2, 5, 10])
def test_lbfgs_armijo_float32_reports_convergence(n):
    x0 = torch.tensor([-1.2, 1.0]).repeat(n // 2) if n % 2 == 0 else torch.zeros(n)
    res = piazzi.lbfgs_armijo(rosen_nd, x0, max_iter=1000)
    assert res.converged, res.message
    assert torch.allclose(res.x, torch.ones(n), atol=1e-4)


@pytest.mark.parametrize("solver", ["lbfgs", "gn"])
def test_value_fun_may_return_aux(solver, exp_data):
    t, y = exp_data

    def fun_aux(th):
        r = exp_residuals(th, t, y)
        return 0.5 * (r**2).sum(), {"r0": r[0]}

    x0 = torch.tensor([1.0, 0.5], dtype=torch.float64)
    if solver == "lbfgs":
        res = piazzi.lbfgs_armijo(fun_aux, x0, has_aux=True, value_fun=fun_aux, lr=50.0, max_step=100.0, max_iter=500)
    else:
        system = piazzi.least_squares_system(lambda th: (exp_residuals(th, t, y), {"r0": th[0]}), has_aux=True)
        x0 = torch.tensor([50.0, 10.0], dtype=torch.float64)  # far away: the line search backtracks
        res = piazzi.gauss_newton(system, x0, has_aux=True, method="gn", value_fun=fun_aux, max_step=10.0)
    assert res.n_fun > 0, "the line search never used value_fun"
    assert torch.allclose(res.x, torch.tensor([2.5, 1.3], dtype=torch.float64), atol=1e-4)


@pytest.mark.parametrize("method", ["gn", "lm"])
def test_gauss_newton_copes_with_a_system_that_reuses_its_buffers(method, exp_data):
    t, y = exp_data
    fresh = piazzi.least_squares_system(lambda th: exp_residuals(th, t, y))
    g_buf, H_buf = torch.empty(2, dtype=torch.float64), torch.empty(2, 2, dtype=torch.float64)

    def buffered(th):
        loss, g, H = fresh(th)
        g_buf.copy_(g)
        H_buf.copy_(H)
        return loss, g_buf, H_buf

    x0 = torch.tensor([50.0, 10.0], dtype=torch.float64)
    a = piazzi.gauss_newton(fresh, x0, method=method, max_step=10.0, step_tol=1e-10)
    b = piazzi.gauss_newton(buffered, x0, method=method, max_step=10.0, step_tol=1e-10)
    assert torch.equal(a.x, b.x) and a.n_iter == b.n_iter and a.message == b.message


@pytest.mark.parametrize("method", ["gn", "lm"])
def test_gauss_newton_stops_on_non_finite_curvature(method):
    def system(x):
        return (x**2).sum(), 2 * x, torch.full((2, 2), float("nan"), dtype=x.dtype)

    res = piazzi.gauss_newton(system, torch.ones(2, dtype=torch.float64), method=method)
    assert not res.converged and res.message == "gradient or curvature is not finite"
    assert res.n_system == 1 and res.n_fun == 0


def test_gauss_newton_never_steps_uphill_with_an_indefinite_curvature():
    def system(x):  # f = |x|^2 with a negative-definite "curvature"
        return (x**2).sum(), 2 * x, -torch.eye(2, dtype=x.dtype)

    x0 = torch.tensor([1.0, -2.0], dtype=torch.float64)
    values = []
    res = piazzi.gauss_newton(system, x0, method="gn", callback=lambda info: values.append(info.fun.item()))
    assert all(b <= a for a, b in zip([5.0, *values], values, strict=False)), values
    assert res.fun.item() < 5.0


def test_least_squares_aux_may_hold_non_tensors(exp_data):
    t, y = exp_data

    def fun(th):
        r = exp_residuals(th, t, y)
        return r, {"name": "fit", "n": 3, "worst": r.abs().max(), "nested": [r[0], None]}

    res = piazzi.least_squares(fun, torch.tensor([1.0, 0.5], dtype=torch.float64), has_aux=True)
    assert res.aux["name"] == "fit" and res.aux["n"] == 3 and res.aux["nested"][1] is None
    assert torch.equal(res.aux["worst"], exp_residuals(res.x, t, y).abs().max())
