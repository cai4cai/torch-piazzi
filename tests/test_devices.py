"""The solvers on every available device, in float32 and float64 (issue #7).

Each test runs on the CPU and, when present, on CUDA and Apple MPS; unavailable devices are
skipped, and so is float64 on MPS, which has no float64. CI runs only the CPU cases; run
``uv run pytest tests/test_devices.py`` on a Mac or a CUDA machine to cover the others.

The least-squares tests cover both placements of the normal equations: the default (float64
on the CPU, whatever the device of the objective) and ``solve_device`` / ``solve_dtype`` set
to the objective's own device and dtype.
"""

import pytest
import torch

import piazzi

CSR_BETA = pytest.mark.filterwarnings("ignore:Sparse CSR tensor support is in beta:UserWarning")
AVAILABLE = {"cpu": True, "cuda": torch.cuda.is_available(), "mps": torch.backends.mps.is_available()}


def _cases():
    for device, available in AVAILABLE.items():
        for dtype in (torch.float32, torch.float64):
            marks = []
            if not available:
                marks.append(pytest.mark.skip(reason=f"{device} is not available"))
            elif device == "mps" and dtype == torch.float64:
                marks.append(pytest.mark.skip(reason="MPS has no float64"))
            yield pytest.param(device, dtype, id=f"{device}-{str(dtype).removeprefix('torch.')}", marks=marks)


CASES = list(_cases())
device_dtype = pytest.mark.parametrize(("device", "dtype"), CASES)
placement = pytest.mark.parametrize("placement", ["cpu64", "device"])


def atol(dtype):
    return 1e-3 if dtype == torch.float32 else 1e-7


def solve_options(placement, device, dtype):
    """The default float64 CPU solve, or a solve on the objective's own device and dtype."""
    return {} if placement == "cpu64" else {"solve_device": device, "solve_dtype": dtype}


def tensor(values, device, dtype):
    # Build on the CPU in float64 and narrow there: MPS cannot hold float64 even briefly.
    return torch.tensor(values, dtype=torch.float64).to(dtype).to(device)


def assert_result(res, device, dtype, expected):
    assert res.x.device.type == device and res.x.dtype == dtype
    assert res.fun.device.type == device
    torch.testing.assert_close(res.x.cpu(), expected.to(dtype), atol=atol(dtype), rtol=0)


def exp_problem(device, dtype):
    t = torch.linspace(0, 4, 40, dtype=dtype, device=device)
    y = 2.5 * torch.exp(-1.3 * t)
    return (lambda th: th[0] * torch.exp(-th[1] * t) - y), tensor([1.0, 0.5], device, dtype)


EXP_SOLUTION = torch.tensor([2.5, 1.3], dtype=torch.float64)


def rosen(x):
    return (100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2).sum()


@device_dtype
@placement
@pytest.mark.parametrize("method", ["gn", "lm", "dogleg"])
@pytest.mark.parametrize("damping", ["identity", "marquardt"])
def test_least_squares(device, dtype, placement, method, damping):
    fun, x0 = exp_problem(device, dtype)
    res = piazzi.least_squares(fun, x0, method=method, damping=damping, **solve_options(placement, device, dtype))
    assert res.converged, res.message
    assert_result(res, device, dtype, EXP_SOLUTION)


@device_dtype
@placement
@pytest.mark.parametrize("loss", ["huber", "soft_l1", "cauchy", "arctan"])
def test_robust_least_squares(device, dtype, placement, loss):
    """Every 10th point is off by 1; each device must find the minimiser the CPU finds in float64.

    The minimisers differ between losses (arctan even lands in another local minimum), so the
    reference is the same fit in float64 on the CPU rather than the clean solution.
    """

    def problem(device, dtype):
        fun, x0 = exp_problem(device, dtype)
        outliers = (torch.arange(40, device=device) % 10 == 0).to(dtype)
        return (lambda th: fun(th) + outliers), x0

    ref = piazzi.least_squares(*problem("cpu", torch.float64), loss=loss, f_scale=0.1, step_tol=1e-12)
    fun, x0 = problem(device, dtype)
    options = solve_options(placement, device, dtype)
    res = piazzi.least_squares(fun, x0, loss=loss, f_scale=0.1, step_tol=1e-10, **options)
    assert ref.converged and res.converged, res.message
    assert_result(res, device, dtype, ref.x)


@device_dtype
@placement
@pytest.mark.parametrize(
    ("layout", "solver"),
    [("coo", None), ("coo", "dense"), pytest.param("csr", None, marks=CSR_BETA)],
    ids=["coo-cg", "coo-dense", "csr-cg"],
)
def test_sparse_normal_equations(device, dtype, placement, layout, solver):
    if layout == "csr" and device == "mps":
        pytest.skip("PyTorch cannot build CSR tensors on MPS")
    pytest.importorskip("torchsparsegradutils")
    fun, x0 = exp_problem(device, dtype)
    dense_system = piazzi.least_squares_system(fun)

    def sparse_system(x):
        loss, g, H = dense_system(x)
        return loss, g, H.to_sparse_coo() if layout == "coo" else H.to_sparse_csr()

    linear_solver = piazzi.dense_solver if solver == "dense" else None
    options = solve_options(placement, device, dtype)
    res = piazzi.gauss_newton(sparse_system, x0, linear_solver=linear_solver, **options)
    assert res.converged, res.message
    assert_result(res, device, dtype, EXP_SOLUTION)


@device_dtype
def test_lbfgs_armijo(device, dtype):
    seen = []
    res = piazzi.minimize(rosen, tensor([-1.2, 1.0], device, dtype), callback=seen.append, options={"max_iter": 200})
    assert res.converged, res.message
    assert_result(res, device, dtype, torch.ones(2, dtype=torch.float64))
    assert all(info.x.device.type == device and info.fun.device.type == device for info in seen)


@device_dtype
def test_torch_lbfgs_strong_wolfe(device, dtype):
    res = piazzi.minimize(
        rosen,
        tensor([-1.2, 1.0], device, dtype),
        method=torch.optim.LBFGS,
        options={"line_search_fn": "strong_wolfe", "tolerance_change": 1e-12},
    )
    assert res.converged, res.message
    assert_result(res, device, dtype, torch.ones(2, dtype=torch.float64))


@device_dtype
@pytest.mark.parametrize("solver", ["least_squares", "minimize"])
def test_se3_registration(device, dtype, solver):
    """Rigid registration of points far from the origin, with a pivot at their centroid."""
    g = torch.Generator().manual_seed(0)
    P = 5 * torch.randn(30, 3, generator=g, dtype=torch.float64) + torch.tensor([100.0, 80.0, -60.0])
    T_true = piazzi.se3_exp(torch.tensor([3.0, -2.0, 5.0, 0.9, -1.2, 0.6], dtype=torch.float64))
    Q = P @ T_true[:3, :3].T + T_true[:3, 3]
    P, Q = (v.to(dtype).to(device) for v in (P, Q))

    def residuals(T):
        return (P @ T[:3, :3].T + T[:3, 3] - Q).reshape(-1)

    manifold = piazzi.SE3(pivot=P.mean(0))
    T0 = torch.eye(4, dtype=dtype, device=device)
    if solver == "least_squares":
        res = piazzi.least_squares(residuals, T0, manifold=manifold, max_step=1e3)
    else:
        res = piazzi.minimize(
            lambda T: 0.5 * residuals(T).square().sum(),
            T0,
            manifold=manifold,
            # A step_tol the dtype can resolve: float32 cannot move a coordinate near 1 by 1e-10.
            # L-BFGS is slow here (rotation and translation are scaled ~100 apart): ~190 iterations
            # usually, up to ~550 when float32 rounding differs by an ulp, as across platforms.
            options={"max_iter": 1000, "step_tol": 1e-10 if dtype == torch.float64 else 1e-6},
        )
    assert res.converged, res.message
    assert res.x.device.type == device and res.x.dtype == dtype
    # Translations are ~100 times the rotation entries, so compare them with a relative tolerance.
    torch.testing.assert_close(res.x.cpu().double(), T_true, atol=atol(dtype), rtol=atol(dtype))


@device_dtype
@pytest.mark.parametrize("with_manifold", [False, True])
@pytest.mark.parametrize("solver", ["least_squares", "minimize"])
def test_finite_difference(device, dtype, solver, with_manifold):
    """The wrapper on every device: differences, the tangent pseudo-inverse and the gradient stay on ``device``."""
    g = torch.Generator().manual_seed(0)
    P = torch.randn(12, 3, generator=g, dtype=torch.float64)
    T_true = piazzi.se3_exp(torch.tensor([0.3, -0.2, 0.1, 0.2, -0.1, 0.15], dtype=torch.float64))
    Q = P @ T_true[:3, :3].T + T_true[:3, 3]
    P, Q = (v.to(dtype).to(device) for v in (P, Q))

    def residuals(T):
        return (P @ T[:3, :3].T + T[:3, 3] - Q).reshape(-1)

    manifold = piazzi.SE3()
    T0 = torch.eye(4, dtype=dtype, device=device)
    # float32 differences are accurate to ~1e-3 (the step is ~5e-3), so only ask for that much.
    if solver == "least_squares":
        fd = piazzi.finite_difference(residuals, manifold=manifold if with_manifold else None)
        res = piazzi.least_squares(fd, T0, manifold=manifold, step_tol=1e-9 if dtype == torch.float64 else 1e-4)
    else:
        fd = piazzi.finite_difference(
            lambda T: 0.5 * residuals(T).square().sum(), manifold=manifold if with_manifold else None
        )
        res = piazzi.minimize(
            fd, T0, manifold=manifold, options={"max_iter": 300, "step_tol": 1e-9 if dtype == torch.float64 else 1e-4}
        )
    assert res.x.device.type == device and res.x.dtype == dtype
    torch.testing.assert_close(res.x.cpu().double(), T_true, atol=5e-3 if dtype == torch.float32 else 1e-5, rtol=0)


@device_dtype
def test_scipy_wrappers_return_to_the_device(device, dtype):
    pytest.importorskip("scipy")
    import piazzi.scipy

    fun, x0 = exp_problem(device, dtype)
    res = piazzi.scipy.least_squares(fun, x0)
    assert_result(res, device, dtype, EXP_SOLUTION)
    res = piazzi.scipy.minimize(rosen, tensor([-1.2, 1.0], device, dtype), method="BFGS")
    assert_result(res, device, dtype, torch.ones(2, dtype=torch.float64))
