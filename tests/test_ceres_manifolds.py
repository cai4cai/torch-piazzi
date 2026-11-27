"""Tests ported from Ceres Solver's manifold and rotation test suites.

Sources (ceres-solver, master):

- ``include/ceres/manifold_test_utils.h``: the invariants of
  ``EXPECT_THAT_MANIFOLD_INVARIANTS_HOLD``, ported as :func:`assert_manifold_invariants`.
- ``internal/ceres/manifold_test.cc``: EuclideanManifold, SubsetManifold and
  ProductManifold tests, and the QuaternionManifold tests (GenericDelta, SmallDelta,
  DeltaJustBelowPi, PlusPiBy2), ported to :class:`piazzi.SO3` and :class:`piazzi.SE3`.
- ``internal/ceres/rotation_test.cc``: the angle-axis <-> rotation-matrix tests, ported to
  :func:`piazzi.so3_exp` / :func:`piazzi.so3_log`, and the derivative-at-small-angle Jet
  tests, ported as Jacobians of ``log(exp(w))``.

Differences from Ceres:

- Ceres checks a hand-written ``PlusJacobian`` / ``MinusJacobian`` against Ridders'
  numerical differentiation. piazzi takes these Jacobians by autodiff, so the port checks
  forward- and reverse-mode autodiff against each other and against central differences.
- ``RightMultiplyByPlusJacobian`` has no piazzi counterpart and is not ported.
- Ceres's quaternion delta is half the rotation angle; piazzi's rotation vectors are the
  full angle. "Delta just below pi" therefore means a rotation just below pi here, which
  exercises the near-pi branch of :func:`piazzi.so3_log`.
- Sphere and Line manifolds are not implemented in piazzi, so their tests are not ported.
- Ceres loops over 1000 (manifolds) or 10000 (rotations) random trials. The rotation
  tests here are batched and keep those counts; the manifold tests use fewer trials, with
  the Jacobian checks on a subset, to keep the suite fast.
"""

import math

import pytest
import torch

import piazzi
from piazzi._common import quiet_jvp_warnings

D = torch.float64
EPS = torch.finfo(D).eps
K_TOLERANCE = 1e-9  # manifold_test.cc
K_ROT_TOLERANCE = 10 * EPS  # rotation_test.cc kTolerance
K_LOOSE_TOLERANCE = 1e-9  # rotation_test.cc kLooseTolerance
N_TRIALS = 100
N_JACOBIAN_TRIALS = 10


def gen(seed=0):
    return torch.Generator().manual_seed(seed)


def uniform(*shape, g, low=-1.0, high=1.0):
    """Like Eigen's ``Vector::Random``: uniform in [low, high)."""
    return low + (high - low) * torch.rand(*shape, generator=g, dtype=D)


def unit(n, g):
    v = uniform(n, g=g)
    return v / v.norm()


def random_rotation(g):
    return piazzi.so3_exp(unit(3, g) * uniform(1, g=g, low=0.0, high=math.pi))


def random_pose(g):
    return piazzi.SE3().plus(
        torch.eye(4, dtype=D), torch.cat([uniform(3, g=g), unit(3, g) * math.pi * torch.rand(1, generator=g, dtype=D)])
    )


def rel_err(actual, expected):
    """Relative error, or absolute when ``expected`` is zero (as in manifold_test_utils.h)."""
    n = expected.norm().item()
    e = (actual - expected).norm().item()
    return e / n if n > 0 else e


# --- manifold_test_utils.h -------------------------------------------------------------


def numerical_jacobian(f, x, h=1e-6):
    """Central differences of ``f`` (flat in, flat out) at ``x``."""
    cols = []
    for e in torch.eye(x.numel(), dtype=x.dtype):
        cols.append((f(x + h * e) - f(x - h * e)).reshape(-1) / (2 * h))
    return torch.stack(cols, -1)


def plus_jacobian(M, x):
    zero = torch.zeros(M.tangent_size(x), dtype=x.dtype)
    with quiet_jvp_warnings():
        fwd = torch.func.jacfwd(lambda d: M.plus(x, d).reshape(-1))(zero)
    rev = torch.func.jacrev(lambda d: M.plus(x, d).reshape(-1))(zero)
    return fwd, rev, numerical_jacobian(lambda d: M.plus(x, d), zero)


def minus_jacobian(M, x):
    flat = x.reshape(-1)
    with quiet_jvp_warnings():
        fwd = torch.func.jacfwd(lambda y: M.minus(y.reshape(x.shape), x))(flat)
    rev = torch.func.jacrev(lambda y: M.minus(y.reshape(x.shape), x))(flat)
    return fwd, rev, numerical_jacobian(lambda y: M.minus(y.reshape(x.shape), x), flat)


def assert_manifold_invariants(M, x, delta, y, tol=K_TOLERANCE, jacobians=True):
    """Port of ``EXPECT_THAT_MANIFOLD_INVARIANTS_HOLD`` (without RightMultiplyByPlusJacobian)."""
    zero = torch.zeros(M.tangent_size(x), dtype=x.dtype)
    # XPlusZeroIsXAt
    assert rel_err(M.plus(x, zero), x) <= tol
    # XMinusXIsZeroAt
    assert M.minus(x, x).norm().item() <= tol
    # MinusPlusIsIdentityAt, with delta and with zero
    assert rel_err(M.minus(M.plus(x, delta), x), delta) <= tol
    assert rel_err(M.minus(M.plus(x, zero), x), zero) <= tol
    # PlusMinusIsIdentityAt, with x and with y
    assert rel_err(M.plus(x, M.minus(x, x)), x) <= tol
    assert rel_err(M.plus(x, M.minus(y, x)), y) <= tol
    if not jacobians:
        return
    # HasCorrectPlusJacobianAt (autodiff, both modes, vs central differences)
    fwd, rev, num = plus_jacobian(M, x)
    assert rel_err(fwd, rev) <= tol
    assert rel_err(fwd, num) <= 1e-7  # central differences are only good to ~h^2 and eps/h
    # HasCorrectMinusJacobianAt
    mfwd, mrev, mnum = minus_jacobian(M, x)
    assert rel_err(mfwd, mrev) <= tol
    assert rel_err(mfwd, mnum) <= 1e-7
    # MinusPlusJacobianIsIdentityAt
    eye = torch.eye(fwd.shape[1], dtype=x.dtype)
    assert rel_err(mfwd @ fwd, eye) <= tol


# --- EuclideanManifold ------------------------------------------------------------------


@pytest.mark.parametrize("M", [piazzi.Euclidean(3), piazzi.Euclidean()], ids=["static", "dynamic"])
def test_euclidean_normal_function(M):
    g = gen(1)
    for i in range(N_TRIALS):
        x, delta, y = uniform(3, g=g), uniform(3, g=g), uniform(3, g=g)
        assert M.tangent_size(x) == 3
        assert rel_err(M.plus(x, delta), x + delta) <= K_TOLERANCE
        assert_manifold_invariants(M, x, delta, y, jacobians=i < N_JACOBIAN_TRIALS)


# --- SubsetManifold ---------------------------------------------------------------------


def test_subset_empty_constant_parameters():
    M = piazzi.Subset(3, [])
    g = gen(2)
    for i in range(N_TRIALS):
        x, delta, y = uniform(3, g=g), uniform(3, g=g), uniform(3, g=g)
        assert rel_err(M.plus(x, delta), x + delta) <= K_TOLERANCE
        assert_manifold_invariants(M, x, delta, y, jacobians=i < N_JACOBIAN_TRIALS)


@pytest.mark.parametrize(
    ("size", "constant", "match"),
    [(3, [-1], "greater than or equal to zero"), (2, [2], "less than the size"), (3, [1, 1], "duplicates")],
)
def test_subset_invalid_constant_parameters(size, constant, match):
    with pytest.raises(ValueError, match=match):
        piazzi.Subset(size, constant)


@pytest.mark.parametrize("i", range(4))
def test_subset_normal_function(i):
    M = piazzi.Subset(4, [i])
    assert M.ambient_size == 4 and M.tangent_size(torch.zeros(4)) == 3
    g = gen(3 + i)
    for trial in range(N_TRIALS):
        x, delta, y = uniform(4, g=g), uniform(3, g=g), uniform(4, g=g)
        y[i] = x[i]  # y must be on the same "manifold"
        out = M.plus(x, delta)
        expected = x.clone()
        expected[[j for j in range(4) if j != i]] += delta
        assert torch.equal(out, expected)
        assert_manifold_invariants(M, x, delta, y, jacobians=trial < N_JACOBIAN_TRIALS)


# --- ProductManifold --------------------------------------------------------------------


def four_subsets():
    return [piazzi.Subset(5, [2]), piazzi.Subset(3, [0, 1]), piazzi.Subset(4, [1]), piazzi.Subset(2, [0])]


@pytest.mark.parametrize("n", [2, 3, 4])
def test_product_sizes(n):
    factors = four_subsets()[:n]
    M = piazzi.Product(*factors)
    x = torch.zeros(M.ambient_size, dtype=D)
    assert M.ambient_size == sum(f.ambient_size for f in factors)
    assert M.tangent_size(x) == sum(f.tangent_size(None) for f in factors)


def test_product_normal_function():
    factors = four_subsets()
    M = piazzi.Product(*factors)
    assert M.ambient_size == 14 and M.tangent_size(torch.zeros(14)) == 9
    g = gen(10)
    for trial in range(N_TRIALS):
        x, delta = uniform(14, g=g), uniform(9, g=g)
        out = M.plus(x, delta)
        expected = torch.cat(
            [
                f.plus(xb, db)
                for f, xb, db in zip(factors, x.split([5, 3, 4, 2]), delta.split([4, 1, 3, 1]), strict=True)
            ]
        )
        assert torch.equal(out, expected)
        assert_manifold_invariants(M, x, delta, out, jacobians=trial < N_JACOBIAN_TRIALS)


@pytest.mark.parametrize("zero_first", [True, False])
def test_product_zero_tangent_size_and_euclidean(zero_first):
    factors = [piazzi.Subset(1, [0]), piazzi.Euclidean(2)]
    M = piazzi.Product(*(factors if zero_first else factors[::-1]))
    assert M.ambient_size == 3 and M.tangent_size(torch.zeros(3)) == 2
    g = gen(11)
    for trial in range(N_TRIALS):
        x, delta, y = uniform(3, g=g), uniform(2, g=g), uniform(3, g=g)
        out = M.plus(x, delta)
        if zero_first:
            y[0] = x[0]
            assert out[0] == x[0] and out[1] == x[1] + delta[0] and out[2] == x[2] + delta[1]
        else:
            y[2] = x[2]
            assert out[0] == x[0] + delta[0] and out[1] == x[1] + delta[1] and out[2] == x[2]
        assert_manifold_invariants(M, x, delta, y, jacobians=trial < N_JACOBIAN_TRIALS)


def test_product_of_poses_and_rotations():
    M = piazzi.Product(piazzi.SE3(pivot=[1.0, 2.0, 3.0]), piazzi.Euclidean(2), piazzi.SO3(side="left"))
    g = gen(12)
    for trial in range(20):
        x = M.join(random_pose(g), uniform(2, g=g), random_rotation(g))
        y = M.join(random_pose(g), uniform(2, g=g), random_rotation(g))
        delta = torch.cat([uniform(3, g=g), unit(3, g) * 2.0, uniform(2, g=g), unit(3, g) * 2.5])
        assert_manifold_invariants(M, x, delta, y, jacobians=trial < 3)


# --- QuaternionManifold, ported to SO3 and SE3 --------------------------------------------


def hat(w):
    return piazzi.so3_hat(w)


def twist(xi):
    out = torch.zeros(4, 4, dtype=xi.dtype)
    out[:3, :3] = hat(xi[3:])
    out[:3, 3] = xi[:3]
    return out


def translation(c):
    out = torch.eye(4, dtype=D)
    out[:3, 3] = torch.as_tensor(c, dtype=D)
    return out


def so3_plus_reference(M, x, delta):
    """Like QuaternionManifoldPlusIsCorrectAt: compose with an independently computed rotation."""
    dR = torch.linalg.matrix_exp(hat(delta))
    return x @ dR if M.side == "right" else dR @ x


def se3_plus_reference(M, x, delta):
    if M.retraction == "exp":
        inc = torch.linalg.matrix_exp(twist(delta))
    else:
        rot = torch.eye(4, dtype=D)
        rot[:3, :3] = torch.linalg.matrix_exp(hat(delta[3:]))
        inc = translation(delta[:3]) @ rot
    if M.pivot is not None:
        inc = translation(M.pivot) @ inc @ translation(-M.pivot)
    return x @ inc if M.side == "right" else inc @ x


SO3_MANIFOLDS = [piazzi.SO3(side="right"), piazzi.SO3(side="left")]
SE3_MANIFOLDS = [
    piazzi.SE3(side=side, retraction=r, pivot=p)
    for side in ("right", "left")
    for r in ("decoupled", "exp")
    for p in (None, [0.5, -2.0, 1.0])
]


def ids(ms):
    return [
        f"{m.side}"
        + (f"-{m.retraction}" if hasattr(m, "retraction") else "")
        + ("-pivot" if getattr(m, "pivot", None) is not None else "")
        for m in ms
    ]


# Rotation part of delta: GenericDelta (uniform [-1, 1]^3), SmallDelta (1e-6), DeltaJustBelowPi.
DELTA_KINDS = {
    "generic": lambda g: uniform(3, g=g),
    "small": lambda g: unit(3, g) * 1e-6,
    "just_below_pi": lambda g: unit(3, g) * (math.pi - 1e-6),
}


@pytest.mark.parametrize("kind", DELTA_KINDS)
@pytest.mark.parametrize("M", SO3_MANIFOLDS, ids=ids(SO3_MANIFOLDS))
def test_so3_manifold(M, kind):
    g = gen(20)
    for trial in range(N_TRIALS):
        x, y, delta = random_rotation(g), random_rotation(g), DELTA_KINDS[kind](g)
        assert rel_err(M.plus(x, delta), so3_plus_reference(M, x, delta)) <= K_TOLERANCE
        assert_manifold_invariants(M, x, delta, y, jacobians=trial < N_JACOBIAN_TRIALS)


@pytest.mark.parametrize("kind", DELTA_KINDS)
@pytest.mark.parametrize("M", SE3_MANIFOLDS, ids=ids(SE3_MANIFOLDS))
def test_se3_manifold(M, kind):
    g = gen(21)
    for trial in range(N_TRIALS // 2):
        x, y = random_pose(g), random_pose(g)
        delta = torch.cat([uniform(3, g=g), DELTA_KINDS[kind](g)])
        assert rel_err(M.plus(x, delta), se3_plus_reference(M, x, delta)) <= K_TOLERANCE
        assert_manifold_invariants(M, x, delta, y, jacobians=trial < N_JACOBIAN_TRIALS // 2)


@pytest.mark.parametrize("M", SO3_MANIFOLDS, ids=ids(SO3_MANIFOLDS))
def test_so3_plus_pi_by_2(M):
    """QuaternionManifold.PlusPiBy2: a quarter turn about each axis from the identity."""
    x = torch.eye(3, dtype=D)
    for i in range(3):
        delta = torch.zeros(3, dtype=D)
        delta[i] = math.pi / 2
        out = M.plus(x, delta)
        # The axis column stays e_i; the other two rotate by 90 degrees.
        j, k = (i + 1) % 3, (i + 2) % 3
        expected = torch.zeros(3, 3, dtype=D)
        expected[i, i] = 1
        expected[k, j] = 1
        expected[j, k] = -1
        torch.testing.assert_close(out, expected, atol=EPS, rtol=0)
        assert_manifold_invariants(M, x, delta, out)


# --- rotation_test.cc: angle-axis <-> rotation matrix -------------------------------------


def assert_orthonormal(R, tol=K_ROT_TOLERANCE):
    eye = torch.eye(3, dtype=D).expand_as(R)
    assert (R @ R.mT - eye).abs().max().item() <= tol
    assert (R.mT @ R - eye).abs().max().item() <= tol


def assert_near_angle_axis(actual, expected, tol=K_LOOSE_TOLERANCE):
    """IsNearAngleAxis: relative error, with the sign ambiguity at pi, absolute at zero."""
    norm = expected.norm(dim=-1)
    err = (actual - expected).norm(dim=-1)
    flipped = (actual + expected).norm(dim=-1)
    err = torch.where((norm - math.pi).abs() < tol, torch.minimum(err, flipped), err)
    err = torch.where(norm > 0, err / norm.clamp(min=1e-300), err)
    assert err.max().item() <= tol


def test_zero_angle_axis_to_rotation_matrix():
    R = piazzi.so3_exp(torch.zeros(3, dtype=D))
    assert_orthonormal(R)
    assert (R - torch.eye(3, dtype=D)).abs().max().item() <= K_ROT_TOLERANCE


def test_near_zero_angle_axis_to_rotation_matrix():
    R = piazzi.so3_exp(torch.tensor([1e-24, 2e-24, 3e-24], dtype=D))
    assert_orthonormal(R)
    assert (R - torch.eye(3, dtype=D)).abs().max().item() <= K_ROT_TOLERANCE


S3 = math.sqrt(3) / 2


@pytest.mark.parametrize(
    ("w", "R"),
    [
        ([math.pi / 2, 0, 0], [[1, 0, 0], [0, 0, -1], [0, 1, 0]]),  # XRotationToRotationMatrix
        ([0, math.pi, 0], [[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),  # YRotationToRotationMatrix
        ([0, 0, math.pi / 3], [[0.5, -S3, 0], [S3, 0.5, 0], [0, 0, 1]]),  # ZRotationToRotationMatrix
    ],
    ids=["x_pi_by_2", "y_pi", "z_pi_by_3"],
)
def test_axis_rotations(w, R):
    w, R = torch.tensor(w, dtype=D), torch.tensor(R, dtype=D)
    out = piazzi.so3_exp(w)
    assert_orthonormal(out)
    assert (out - R).abs().max().item() <= K_ROT_TOLERANCE
    assert_near_angle_axis(piazzi.so3_log(out), w)


def test_at_pi_angle_axis_round_trip():
    R = torch.diag(torch.tensor([1.0, -1.0, -1.0], dtype=D))
    w = piazzi.so3_log(R)
    assert_near_angle_axis(w, torch.tensor([math.pi, 0, 0], dtype=D))
    assert (piazzi.so3_exp(w) - R).abs().max().item() <= K_ROT_TOLERANCE


def random_angle_axis(n, g, low, high):
    axis = uniform(n, 3, g=g)
    axis = axis / axis.norm(dim=-1, keepdim=True)
    return axis * uniform(n, 1, g=g, low=low, high=high)


def test_angle_axis_to_rotation_matrix_and_back():
    w = random_angle_axis(10000, gen(30), -math.pi, math.pi)
    R = piazzi.so3_exp(w)
    assert_orthonormal(R)
    # Rotations by theta < 0 come back as the same rotation with theta > 0 about -axis.
    assert_near_angle_axis(piazzi.so3_log(R), w)


def test_angle_axis_to_rotation_matrix_and_back_near_zero():
    w = random_angle_axis(10000, gen(31), -1e-16 * math.pi, 1e-16 * math.pi)
    assert (piazzi.so3_log(piazzi.so3_exp(w)) - w).abs().max().item() <= EPS


def test_near_pi_angle_axis_round_trip():
    w = random_angle_axis(10000, gen(32), math.pi - 1e-8, math.pi)
    assert_near_angle_axis(piazzi.so3_log(piazzi.so3_exp(w)), w)


def test_exhaustive_round_trip():
    """RotationMatrixToAngleAxis.ExhaustiveRoundTrip, on a 200 x 200 grid instead of 1000 x 1000."""
    n, max_small_angle = 200, 1e-8
    theta = torch.arange(n, dtype=D) / n * 2 * math.pi
    phi = torch.arange(n, dtype=D) / n * math.pi
    theta, phi = torch.meshgrid(theta, phi, indexing="ij")
    axis = torch.stack([phi.sin() * theta.cos(), phi.sin() * theta.sin(), phi.cos()], -1).reshape(-1, 3)
    g = gen(33)
    for angle in (
        torch.full((axis.shape[0], 1), math.pi, dtype=D),
        uniform(axis.shape[0], 1, g=g, low=math.pi - max_small_angle, high=math.pi),
        uniform(axis.shape[0], 1, g=g, low=-1.0, high=2 * max_small_angle - 1.0),
    ):
        w = angle * axis
        assert_near_angle_axis(piazzi.so3_log(piazzi.so3_exp(w)), w)


def test_rotation_matrix_matches_matrix_exp_for_large_angles():
    """AngleAxis.RotatePointGivesSameAnswerAsRotationMatrix: angles from -pi to ~21 pi."""
    g = gen(34)
    theta = (2 * torch.arange(10000, dtype=D) * 0.0011 - 1) * math.pi
    axis = uniform(10000, 3, g=g)
    w = theta[:, None] * axis / axis.norm(dim=-1, keepdim=True)
    torch.testing.assert_close(piazzi.so3_exp(w), torch.linalg.matrix_exp(hat(w)), atol=1e-12, rtol=0)


@pytest.mark.parametrize("exponent", [*range(-2, -32, -4), -31, -100, -200, -306])
def test_log_exp_jacobian_at_small_angles(exponent):
    """Small/Tiny AngleAxisToQuaternionForJets and QuaternionToAngleAxisForJets.

    Ceres checks the dual parts of each conversion at theta = 10^i, from 1e-2 down to
    1e-306. Here the conversions are exp and log, and ``log(exp(w)) = w`` for any angle
    below pi, so the Jacobian of the round trip must be the identity.
    """
    w = torch.tensor([10.0**exponent, 0.0, 0.0], dtype=D)
    with quiet_jvp_warnings():
        J = torch.func.jacfwd(lambda v: piazzi.so3_log(piazzi.so3_exp(v)))(w)
    torch.testing.assert_close(J, torch.eye(3, dtype=D), atol=K_ROT_TOLERANCE, rtol=0)
    J = torch.func.jacrev(lambda v: piazzi.se3_log(piazzi.se3_exp(v)))(torch.cat([torch.ones(3, dtype=D), w]))
    torch.testing.assert_close(J, torch.eye(6, dtype=D), atol=K_ROT_TOLERANCE, rtol=0)


@pytest.mark.parametrize("angle", [0.5, 2.0, 2.5, math.pi - 1e-3])
def test_log_exp_jacobian_at_larger_angles(angle):
    w = unit(3, gen(35)) * angle
    with quiet_jvp_warnings():
        J = torch.func.jacfwd(lambda v: piazzi.so3_log(piazzi.so3_exp(v)))(w)
    torch.testing.assert_close(J, torch.eye(3, dtype=D), atol=1e-9, rtol=0)


def test_tiny_angle_axis():
    """TinyAngleAxisToQuaternion / TinyQuaternionToAngleAxis: theta = DBL_MIN ** 0.75."""
    theta = torch.finfo(D).tiny ** 0.75
    w = torch.tensor([theta, 0.0, 0.0], dtype=D)
    R = piazzi.so3_exp(w)
    assert R[2, 1].item() == theta and R[1, 2].item() == -theta
    assert piazzi.so3_log(R)[0].item() == theta
