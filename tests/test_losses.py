"""Robust losses. ``test_options``, ``test_grad``, ``test_jac`` and ``test_robustness`` port
``LossFunctionMixin`` from SciPy's ``scipy/optimize/tests/test_least_squares.py`` (BSD-3-Clause).

SciPy checks ``res.grad`` and ``res.jac`` after a single evaluation (``max_nfev=1``); here the
same quantities are read straight from the normal equations ``(F, g, H)``: SciPy's ``grad``
is ``g`` and its scaled Jacobian satisfies ``jac^T jac = H``.
"""

import math

import pytest
import torch

import piazzi

EPS = torch.finfo(torch.float64).eps


def fun_trivial(x, a=0.0):
    return (x - a) ** 2 + 5.0


def cubic_soft_l1(z):
    return 3 * ((1 + z) ** (1 / 3) - 1)


LOSSES = [*piazzi.LOSSES, cubic_soft_l1]
ROBUST_LOSSES = [loss for loss in LOSSES if loss != "linear"]


def loss_id(loss):
    return loss if isinstance(loss, str) else loss.__name__


def trivial_system(x, **kwargs):
    x = torch.tensor([x], dtype=torch.float64)
    return piazzi.least_squares_system(fun_trivial, **kwargs)(x)


@pytest.mark.parametrize("loss", LOSSES, ids=loss_id)
def test_options(loss):
    res = piazzi.least_squares(fun_trivial, torch.tensor([2.0], dtype=torch.float64), loss=loss, step_tol=1e-12)
    assert abs(res.x.item()) < 1e-8, res.message


def test_bad_loss_and_f_scale():
    x0 = torch.tensor([2.0], dtype=torch.float64)
    with pytest.raises(ValueError, match="loss"):
        piazzi.least_squares(fun_trivial, x0, loss="hinge")
    with pytest.raises(ValueError, match="f_scale"):
        piazzi.least_squares(fun_trivial, x0, loss="huber", f_scale=0.0)
    with pytest.raises(ValueError, match="loss"):
        piazzi.robust_cost(torch.ones(3), loss="hinge")


def test_grad():
    # g = J^T rho'(z) r for r = x^2 + 5 at x = 2, with f_scale = 1.
    x = 2.0
    f = x**2 + 5
    expected = {
        "linear": 2 * x * f,
        "huber": 2 * x,
        "soft_l1": 2 * x * f / (1 + f**2) ** 0.5,
        "cauchy": 2 * x * f / (1 + f**2),
        "arctan": 2 * x * f / (1 + f**4),
        cubic_soft_l1: 2 * x * f / (1 + f**2) ** (2 / 3),
    }
    for loss, g_ref in expected.items():
        _, g, _ = trivial_system(x, loss=loss)
        assert g.item() == pytest.approx(g_ref, rel=1e-12), loss_id(loss)


def test_jac():
    # SciPy's scaled Jacobian is jac = J * sqrt(max(rho' + 2 rho'' z, EPS)), so H = jac^2
    # with curvature="scipy".
    x = 2.0
    f = x**2 + 5
    cases = [
        ("linear", 1.0, 2 * x),
        # The huber correction is identically zero for outliers, so it is floored at EPS.
        ("huber", 1.0, 2 * x * EPS**0.5),
        # With f_scale = 10 the residual is an inlier and huber is linear.
        ("huber", 10.0, 2 * x),
        # soft_l1 always gives a positive weight.
        ("soft_l1", 1.0, 2 * x * (1 + f**2) ** -0.75),
        # For cauchy the correction is negative here, and floored at EPS.
        ("cauchy", 1.0, 2 * x * EPS**0.5),
        ("cauchy", 10.0, 2 * x * (1 - (f / 10) ** 2) ** 0.5 / (1 + (f / 10) ** 2)),
        ("arctan", 1.0, 2 * x * EPS**0.5),
        ("arctan", 20.0, 2 * x * (1 - 3 * (f / 20) ** 4) ** 0.5 / (1 + (f / 20) ** 4)),
        (cubic_soft_l1, 1.0, 2 * x * EPS**0.5),
        (cubic_soft_l1, 6.0, 2 * x * (1 - (f / 6) ** 2 / 3) ** 0.5 * (1 + (f / 6) ** 2) ** (-5 / 6)),
    ]
    for loss, f_scale, jac in cases:
        _, _, H = trivial_system(x, loss=loss, f_scale=f_scale, curvature="scipy")
        assert H.item() == pytest.approx(jac**2, rel=1e-9), (loss_id(loss), f_scale)


class ExponentialFittingProblem:
    """y = a + exp(b * x) + noise, with outliers. Same shape as SciPy's test problem,
    with data drawn from a torch generator instead of NumPy."""

    def __init__(self, a, b, noise, n_outliers=1, x_range=(-1, 1), n_points=11, seed=0):
        gen = torch.Generator().manual_seed(seed)
        self.p0 = torch.zeros(2, dtype=torch.float64)
        self.x = torch.linspace(x_range[0], x_range[1], n_points, dtype=torch.float64)
        self.y = a + torch.exp(b * self.x)
        self.y = self.y + noise * torch.randn(n_points, generator=gen, dtype=torch.float64)
        outliers = torch.randint(0, n_points, (n_outliers,), generator=gen)
        self.y[outliers] += 50 * noise * torch.rand(n_outliers, generator=gen, dtype=torch.float64)
        self.p_opt = torch.tensor([a, b], dtype=torch.float64)

    def fun(self, p):
        return p[0] + torch.exp(p[1] * self.x) - self.y


@pytest.mark.parametrize("noise", [0.1, 1.0])
@pytest.mark.parametrize("method", ["gn", "lm"])
def test_robustness(noise, method):
    p = ExponentialFittingProblem(1, 0.1, noise, seed=12220903)
    res_lsq = piazzi.least_squares(p.fun, p.p0, method=method, step_tol=1e-10)
    assert gradient_norm(p, res_lsq.x, "linear", 1.0) < 1e-6
    for loss in ROBUST_LOSSES:
        res = piazzi.least_squares(p.fun, p.p0, loss=loss, f_scale=noise, method=method, step_tol=1e-10)
        assert gradient_norm(p, res.x, loss, noise) < 1e-6, loss_id(loss)
        assert (res.x - p.p_opt).norm() < (res_lsq.x - p.p_opt).norm(), loss_id(loss)


def gradient_norm(p, x, loss, f_scale):
    x = x.clone().requires_grad_(True)
    (g,) = torch.autograd.grad(piazzi.robust_cost(p.fun(x), loss, f_scale), x)
    return g.norm().item()


@pytest.mark.parametrize("loss", LOSSES, ids=loss_id)
@pytest.mark.parametrize("f_scale", [0.5, 3.0])
def test_system_matches_robust_cost_autograd(loss, f_scale):
    p = ExponentialFittingProblem(1, 0.1, 1.0, n_outliers=3, seed=1)
    x = torch.tensor([0.7, -0.4], dtype=torch.float64)
    F, g, H = piazzi.least_squares_system(p.fun, loss=loss, f_scale=f_scale)(x)
    xg = x.clone().requires_grad_(True)
    F_ref = piazzi.robust_cost(p.fun(xg), loss, f_scale)
    (g_ref,) = torch.autograd.grad(F_ref, xg)
    assert torch.allclose(F, F_ref.detach()) and torch.allclose(g, g_ref)
    assert torch.allclose(H, H.T) and torch.linalg.eigvalsh(H).min() >= 0


@pytest.mark.parametrize("loss", ROBUST_LOSSES, ids=loss_id)
def test_irls_weights_are_default(loss):
    p = ExponentialFittingProblem(1, 0.1, 1.0, n_outliers=3, seed=1)
    x = torch.tensor([0.7, -0.4], dtype=torch.float64)
    J = torch.autograd.functional.jacobian(p.fun, x)  # reverse mode: a reference independent of jacfwd
    r = p.fun(x)
    rho = loss if callable(loss) else (lambda z: piazzi.LOSSES[loss](z)[0])
    zg = (r / 0.5) ** 2
    zg.requires_grad_(True)
    (w,) = torch.autograd.grad(rho(zg).sum(), zg)
    _, _, H = piazzi.least_squares_system(p.fun, loss=loss, f_scale=0.5)(x)
    assert torch.allclose(H, J.T @ (w[:, None] * J))


def test_scipy_curvature_gn_converges():
    # With most residuals starting as outliers, SciPy's curvature floors almost every weight
    # at EPS; Gauss-Newton's line search still reaches the same minimiser as IRLS.
    p = ExponentialFittingProblem(1, 0.1, 0.1, seed=12220903)
    ref = piazzi.least_squares(p.fun, p.p0, loss="cauchy", f_scale=0.1, step_tol=1e-12)
    res = piazzi.least_squares(p.fun, p.p0, loss="cauchy", f_scale=0.1, curvature="scipy", method="gn", step_tol=1e-12)
    assert torch.allclose(res.x, ref.x, atol=1e-8)


def test_bad_curvature():
    with pytest.raises(ValueError, match="curvature"):
        piazzi.least_squares_system(fun_trivial, loss="huber", curvature="exact")


def test_huber_derivatives_finite_at_zero():
    z = torch.tensor([0.0, 0.5, 1.0, 4.0], dtype=torch.float64)
    rho, d1, d2 = piazzi.LOSSES["huber"](z)
    assert torch.allclose(rho, torch.tensor([0.0, 0.5, 1.0, 3.0], dtype=torch.float64))
    assert torch.allclose(d1, torch.tensor([1.0, 1.0, 1.0, 0.5], dtype=torch.float64))
    assert torch.allclose(d2, torch.tensor([0.0, 0.0, 0.0, -0.5 * 4**-1.5], dtype=torch.float64))


def test_least_squares_aux_and_float32():
    p = ExponentialFittingProblem(1, 0.1, 0.1, seed=3)
    x, y = p.x.float(), p.y.float()

    def fun(q):
        r = q[0] + torch.exp(q[1] * x) - y
        return r, {"max_abs": r.abs().max()}

    res = piazzi.least_squares(fun, torch.zeros(2), loss="soft_l1", f_scale=0.1, has_aux=True)
    assert res.x.dtype == torch.float32
    assert torch.allclose(res.x, p.p_opt.float(), atol=0.1)
    assert math.isfinite(res.aux["max_abs"].item())


@pytest.mark.parametrize("loss", LOSSES, ids=loss_id)
@pytest.mark.parametrize("noise", [0.1, 1.0])
def test_matches_scipy(loss, noise):
    scipy_optimize = pytest.importorskip("scipy.optimize")
    np = pytest.importorskip("numpy")
    p = ExponentialFittingProblem(1, 0.1, noise, n_outliers=2, seed=12220903)
    xs, ys = p.x.numpy(), p.y.numpy()

    def fun_np(q):
        return q[0] + np.exp(q[1] * xs) - ys

    if callable(loss):

        def scipy_loss(z):
            t = 1 + z
            return np.stack([3 * (t ** (1 / 3) - 1), t ** (-2 / 3), -2 / 3 * t ** (-5 / 3)])
    else:
        scipy_loss = loss
    ref = scipy_optimize.least_squares(fun_np, np.zeros(2), loss=scipy_loss, f_scale=noise, xtol=1e-14, ftol=1e-14)
    res = piazzi.least_squares(p.fun, p.p0, loss=loss, f_scale=noise, step_tol=1e-12)
    assert torch.allclose(res.x, torch.from_numpy(ref.x), atol=1e-7), (res.x, ref.x)
