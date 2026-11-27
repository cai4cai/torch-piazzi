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
