import math

import pytest
import torch

import piazzi
from piazzi._common import quiet_jvp_warnings

D = torch.float64


def twist_matrix(xi):
    """4 x 4 matrix of the se(3) element (rho, omega)."""
    out = torch.zeros(4, 4, dtype=xi.dtype)
    out[:3, :3] = piazzi.so3_hat(xi[3:])
    out[:3, 3] = xi[:3]
    return out


def translation(c):
    out = torch.eye(4, dtype=c.dtype)
    out[:3, 3] = c
    return out


def random_pose(seed, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return piazzi.se3_exp(scale * torch.randn(6, generator=g, dtype=D))


ANGLES = [0.0, 1e-12, 1e-6, 1e-3, 0.0999, 0.1001, 0.5, 2.0, 3.1]


@pytest.mark.parametrize("angle", ANGLES)
def test_so3_exp_matches_matrix_exp(angle):
    axis = torch.tensor([0.3, -0.5, 0.8], dtype=D)
    w = angle * axis / axis.norm()
    R = piazzi.so3_exp(w)
    torch.testing.assert_close(R, torch.linalg.matrix_exp(piazzi.so3_hat(w)), atol=1e-15, rtol=0)
    torch.testing.assert_close(R @ R.T, torch.eye(3, dtype=D), atol=1e-15, rtol=0)


@pytest.mark.parametrize("angle", ANGLES)
def test_se3_exp_matches_matrix_exp(angle):
    xi = torch.tensor([1.0, -2.0, 0.5, 0.3, -0.5, 0.8], dtype=D)
    xi[3:] *= angle / xi[3:].norm()
    torch.testing.assert_close(piazzi.se3_exp(xi), torch.linalg.matrix_exp(twist_matrix(xi)), atol=1e-14, rtol=0)


@pytest.mark.parametrize("angle", ANGLES)
def test_exp_float32_accuracy(angle):
    xi = torch.tensor([1.0, -2.0, 0.5, 0.3, -0.5, 0.8], dtype=D)
    xi[3:] *= angle / xi[3:].norm()
    torch.testing.assert_close(piazzi.se3_exp(xi.float()).double(), piazzi.se3_exp(xi), atol=1e-6, rtol=0)


def test_exp_batched():
    xi = torch.randn(5, 2, 6, dtype=D)
    T = piazzi.se3_exp(xi)
    assert T.shape == (5, 2, 4, 4)
    torch.testing.assert_close(T[3, 1], piazzi.se3_exp(xi[3, 1]))
    torch.testing.assert_close(piazzi.so3_exp(xi[..., 3:]), T[..., :3, :3])


@pytest.mark.parametrize("mode", ["fwd", "rev"])
def test_exp_derivative_at_zero_is_finite_and_exact(mode):
    jac = torch.func.jacfwd if mode == "fwd" else torch.func.jacrev
    for dtype in (torch.float32, D):
        with quiet_jvp_warnings():
            J_so3 = jac(piazzi.so3_exp)(torch.zeros(3, dtype=dtype))  # dR / dw_k = hat(e_k)
            J_se3 = jac(piazzi.se3_exp)(torch.zeros(6, dtype=dtype))
        assert J_so3.dtype == J_se3.dtype == dtype
        torch.testing.assert_close(J_so3.permute(2, 0, 1), piazzi.so3_hat(torch.eye(3, dtype=dtype)))
        basis = torch.stack([twist_matrix(e) for e in torch.eye(6, dtype=dtype)])
        torch.testing.assert_close(J_se3.permute(2, 0, 1), basis)


@pytest.mark.parametrize("side", ["right", "left"])
@pytest.mark.parametrize("retraction", ["decoupled", "exp"])
@pytest.mark.parametrize("with_pivot", [False, True])
def test_se3_plus_matches_reference(side, retraction, with_pivot):
    T = random_pose(0)
    xi = torch.tensor([0.4, -0.2, 0.7, 0.5, 0.1, -0.9], dtype=D)
    c = torch.tensor([10.0, -3.0, 2.0], dtype=D)
    if retraction == "exp":
        inc = torch.linalg.matrix_exp(twist_matrix(xi))
    else:
        inc = translation(xi[:3]) @ torch.linalg.matrix_exp(twist_matrix(torch.cat([torch.zeros(3, dtype=D), xi[3:]])))
    if with_pivot:
        inc = translation(c) @ inc @ translation(-c)
    expected = T @ inc if side == "right" else inc @ T
    M = piazzi.SE3(side=side, retraction=retraction, pivot=c if with_pivot else None)
    torch.testing.assert_close(M.plus(T, xi), expected, atol=1e-13, rtol=0)
    torch.testing.assert_close(M.plus(T.reshape(-1), xi), expected.reshape(-1), atol=1e-13, rtol=0)
    torch.testing.assert_close(M.plus(T, torch.zeros(6, dtype=D)), T)


@pytest.mark.parametrize("side", ["right", "left"])
def test_se3_decoupled_and_exp_share_jacobian_at_zero(side):
    T, c = random_pose(1), torch.tensor([1.0, 2.0, 3.0], dtype=D)
    zero = torch.zeros(6, dtype=D)
    with quiet_jvp_warnings():
        Js = [
            torch.func.jacfwd(piazzi.SE3(side=side, retraction=r, pivot=c).plus, argnums=1)(T, zero)
            for r in ("decoupled", "exp")
        ]
    torch.testing.assert_close(Js[0], Js[1])


@pytest.mark.parametrize("side", ["right", "left"])
def test_so3_plus(side):
    R, w = piazzi.so3_exp(torch.tensor([0.3, 1.0, -0.4], dtype=D)), torch.tensor([0.2, -0.1, 0.5], dtype=D)
    expected = R @ piazzi.so3_exp(w) if side == "right" else piazzi.so3_exp(w) @ R
    torch.testing.assert_close(piazzi.SO3(side=side).plus(R, w), expected)


def test_invalid_options():
    with pytest.raises(ValueError, match="side"):
        piazzi.SE3(side="middle")
    with pytest.raises(ValueError, match="retraction"):
        piazzi.SE3(retraction="cayley")
    with pytest.raises(ValueError, match="side"):
        piazzi.SO3(side="up")
    with pytest.raises(ValueError, match="no fixed size"):
        piazzi.Product(piazzi.SE3(), piazzi.Euclidean())


def test_product_split_join_plus():
    M = piazzi.Product(piazzi.SE3(), piazzi.Euclidean((2, 2)), piazzi.SO3())
    T, A, R = random_pose(2), torch.arange(4.0, dtype=D).reshape(2, 2), piazzi.so3_exp(torch.ones(3, dtype=D))
    x = M.join(T, A, R)
    assert x.shape == (16 + 4 + 9,) and M.tangent_size(x) == 6 + 4 + 3
    for got, want in zip(M.split(x), (T, A, R), strict=True):
        torch.testing.assert_close(got, want)
    delta = torch.linspace(-1, 1, 13, dtype=D)
    T2, A2, R2 = M.split(M.plus(x, delta))
    torch.testing.assert_close(T2, piazzi.SE3().plus(T, delta[:6]))
    torch.testing.assert_close(A2, A + delta[6:10].reshape(2, 2))
    torch.testing.assert_close(R2, piazzi.SO3().plus(R, delta[10:]))
    with pytest.raises(ValueError, match="expected 3 blocks"):
        M.join(T, A)
    with pytest.raises(ValueError, match="elements"):
        M.join(T, A, torch.zeros(4, dtype=D))


# --- Solvers on SE(3): rigid point-set registration -----------------------------------


@pytest.fixture
def registration():
    """Points far from the origin, rotated by ~92 degrees: a hard case without a pivot."""
    g = torch.Generator().manual_seed(0)
    P = 5 * torch.randn(30, 3, generator=g, dtype=D) + torch.tensor([100.0, 80.0, -60.0], dtype=D)
    T_true = piazzi.se3_exp(torch.tensor([3.0, -2.0, 5.0, 0.9, -1.2, 0.6], dtype=D))
    Q = P @ T_true[:3, :3].T + T_true[:3, 3]

    def residuals(T):
        return (P @ T[:3, :3].T + T[:3, 3] - Q).reshape(-1)

    return residuals, T_true, P.mean(0)


def assert_rigid(T, atol=1e-12):
    R = T[:3, :3]
    torch.testing.assert_close(R @ R.T, torch.eye(3, dtype=T.dtype), atol=atol, rtol=0)
    assert abs(torch.linalg.det(R).item() - 1) < atol
    torch.testing.assert_close(T[3], torch.tensor([0.0, 0, 0, 1], dtype=T.dtype))


@pytest.mark.parametrize("method", ["lm", "gn"])
@pytest.mark.parametrize("retraction", ["decoupled", "exp"])
@pytest.mark.parametrize("side", ["right", "left"])
def test_least_squares_se3_registration(registration, method, retraction, side):
    residuals, T_true, _ = registration
    M = piazzi.SE3(side=side, retraction=retraction)
    x0 = torch.eye(4, dtype=D)
    res = piazzi.least_squares(residuals, x0, manifold=M, method=method, max_step=1e3, step_tol=1e-10)
    assert res.converged, res.message
    assert res.x.shape == (4, 4)
    torch.testing.assert_close(res.x, T_true, atol=1e-8, rtol=0)
    assert_rigid(res.x)


@pytest.mark.parametrize("retraction", ["decoupled", "exp"])
def test_pivot_speeds_up_registration(registration, retraction):
    # Without a pivot, a rotation about the far-away origin plus a compensating translation
    # barely moves the points: H has eigenvalues ~0.1 next to ~6e5, and Gauss-Newton crawls.
    residuals, T_true, centre = registration
    results = {
        name: piazzi.least_squares(
            residuals,
            torch.eye(4, dtype=D),
            manifold=piazzi.SE3(retraction=retraction, pivot=pivot),
            method="gn",
            max_step=1e3,
            max_iter=10,
        )
        for name, pivot in (("origin", None), ("centre", centre))
    }
    assert results["centre"].converged, results["centre"].message
    torch.testing.assert_close(results["centre"].x, T_true, atol=1e-8, rtol=0)
    assert not results["origin"].converged


def test_least_squares_se3_float32(registration):
    residuals, T_true, centre = registration
    M = piazzi.SE3(pivot=centre.float())
    res = piazzi.least_squares(lambda T: residuals(T.double()).float(), torch.eye(4), manifold=M, max_step=1e3)
    assert res.x.dtype == torch.float32
    torch.testing.assert_close(res.x.double(), T_true, atol=1e-3, rtol=0)
    assert_rigid(res.x.double(), atol=1e-5)


def test_least_squares_se3_robust(registration):
    residuals, T_true, centre = registration

    def corrupted(T):
        r = residuals(T).clone()
        return r + torch.where(torch.arange(r.numel()) % 15 == 0, 50.0, 0.0).to(r)

    errors = {}
    for loss in ("linear", "cauchy"):
        M = piazzi.SE3(pivot=centre)
        res = piazzi.least_squares(corrupted, torch.eye(4, dtype=D), manifold=M, loss=loss, max_step=1e3)
        assert_rigid(res.x)
        errors[loss] = (res.x - T_true).abs().max().item()
    # The outliers still pull a little on the Cauchy fit (weight ~1/z), hence the loose bound.
    assert errors["cauchy"] < 0.05 and errors["linear"] > 1.0


@pytest.mark.parametrize("retraction", ["decoupled", "exp"])
def test_lbfgs_and_minimize_se3(registration, retraction):
    residuals, T_true, centre = registration
    M = piazzi.SE3(retraction=retraction, pivot=centre)

    def cost(T):
        return 0.5 * (residuals(T) ** 2).sum()

    opts = {"max_iter": 500, "max_step": 1e3, "step_tol": 1e-10}
    for res in (
        piazzi.lbfgs_armijo(cost, torch.eye(4, dtype=D), manifold=M, **opts),
        piazzi.minimize(cost, torch.eye(4, dtype=D), manifold=M, options=opts),
    ):
        assert res.converged, res.message
        torch.testing.assert_close(res.x, T_true, atol=1e-6, rtol=0)
        assert_rigid(res.x)


def test_product_similarity_registration(registration):
    """Pose and scale together: ``Product(SE3, Euclidean(1))``, scaling about the centre."""
    _, T_true, centre = registration
    g = torch.Generator().manual_seed(5)
    P = 5 * torch.randn(30, 3, generator=g, dtype=D) + centre
    s_true = 1.5
    Q = (centre + s_true * (P - centre)) @ T_true[:3, :3].T + T_true[:3, 3]
    M = piazzi.Product(piazzi.SE3(pivot=centre), piazzi.Euclidean(1))

    def fun(x):
        T, s = M.split(x)
        return ((centre + s * (P - centre)) @ T[:3, :3].T + T[:3, 3] - Q).reshape(-1)

    x0 = M.join(torch.eye(4, dtype=D), torch.ones(1, dtype=D))
    res = piazzi.least_squares(fun, x0, manifold=M, max_step=1e3, step_tol=1e-10)
    T, s = M.split(res.x)
    assert res.converged, res.message
    torch.testing.assert_close(s, torch.tensor([s_true], dtype=D), atol=1e-8, rtol=0)
    torch.testing.assert_close(T, T_true, atol=1e-8, rtol=0)
    assert_rigid(T)


def test_so3_rotation_fit():
    g = torch.Generator().manual_seed(3)
    V = torch.randn(20, 3, generator=g, dtype=D)
    R_true = piazzi.so3_exp(torch.tensor([0.0, 0.0, math.pi * 0.9], dtype=D))
    x0 = torch.eye(3, dtype=D)
    res = piazzi.least_squares(lambda R: (V @ R.T - V @ R_true.T).reshape(-1), x0, manifold=piazzi.SO3())
    assert res.converged, res.message
    torch.testing.assert_close(res.x, R_true, atol=1e-8, rtol=0)


def test_least_squares_system_tangent_shapes(registration):
    residuals, *_ = registration
    loss, g, H = piazzi.least_squares_system(residuals, manifold=piazzi.SE3())(random_pose(4))
    assert g.shape == (6,) and H.shape == (6, 6)
    assert torch.linalg.matrix_rank(H).item() == 6


def test_euclidean_manifold_matches_default():
    t = torch.linspace(0, 4, 40, dtype=D)
    y = 2.5 * torch.exp(-1.3 * t)
    x0 = torch.tensor([1.0, 0.5], dtype=D)

    def fun(th):
        return th[0] * torch.exp(-th[1] * t) - y

    a = piazzi.least_squares(fun, x0)
    b = piazzi.least_squares(fun, x0, manifold=piazzi.Euclidean())
    torch.testing.assert_close(a.x, b.x)
    assert a.n_iter == b.n_iter


def test_subset_plus_scales_linearly():
    """plus must not build a size x size matrix: this block would need 320 GB as one."""
    n = 200_000
    M = piazzi.Subset(n, [0, 5])
    x = torch.arange(n, dtype=D)
    out = M.plus(x, torch.ones(n - 2, dtype=D))
    assert out[0] == 0 and out[5] == 5 and out[1] == 2 and out[-1] == n
    torch.testing.assert_close(M.minus(out, x), torch.ones(n - 2, dtype=D))


def test_minimize_registered_method_without_manifold():
    calls = []

    def plain(fun, x0, *, has_aux, callback):
        calls.append(x0)
        return piazzi.lbfgs_armijo(fun, x0, has_aux=has_aux, callback=callback)

    piazzi.MINIMIZE_METHODS["plain_test"] = plain
    try:
        res = piazzi.minimize(lambda x: ((x - 1) ** 2).sum(), torch.zeros(2, dtype=D), method="plain_test")
        assert calls and res.converged
    finally:
        del piazzi.MINIMIZE_METHODS["plain_test"]


def test_minimize_torch_optim_rejects_manifold():
    with pytest.raises(ValueError, match="manifold"):
        piazzi.minimize(lambda T: (T**2).sum(), torch.eye(4, dtype=D), method=torch.optim.Adam, manifold=piazzi.SE3())
