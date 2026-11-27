"""Tests of :func:`piazzi.finite_difference`, checked against autodiff and the ported SciPy problems."""

import pytest
import torch
from problems import F64, LSQ_PROBLEMS, SCALAR_PROBLEMS

import piazzi
from piazzi._common import quiet_jvp_warnings

# Problems small enough that 2n evaluations per Jacobian stay cheap; the 100-unknown bvp is left out.
LSQ_FD = [p for p in LSQ_PROBLEMS if p.name not in ("bvp", "trivial")]


def numpy_style(x):
    """A function autograd cannot see through: it detaches, goes through NumPy and comes back."""
    y = x.detach().cpu().numpy()
    return torch.from_numpy(y**2 + y[::-1] * 3.0).to(x)


@pytest.mark.parametrize("scheme, tol", [("central", 1e-8), ("forward", 1e-5)])
def test_jacobians_match_autodiff(scheme, tol):
    f = lambda x: torch.stack([x[0] ** 2 + x[1], torch.sin(x[1]) * x[0], x.sum() * 100])  # noqa: E731
    x = torch.tensor([1.5, -2.0], dtype=F64)
    fd = piazzi.finite_difference(f, scheme=scheme)
    with quiet_jvp_warnings():
        exact = torch.func.jacrev(f)(x)
        torch.testing.assert_close(torch.func.jacfwd(fd)(x), exact, atol=tol * 100, rtol=tol)
        torch.testing.assert_close(torch.func.jacrev(fd)(x), exact, atol=tol * 100, rtol=tol)
    xg = x.clone().requires_grad_()
    fd(xg).sum().backward()
    torch.testing.assert_close(xg.grad, exact.sum(0), atol=tol * 100, rtol=tol)


def test_values_are_unchanged_and_function_is_a_black_box():
    x = torch.tensor([1.0, 2.0, 3.0], dtype=F64)
    fd = piazzi.finite_difference(numpy_style)
    assert torch.equal(fd(x), numpy_style(x))
    with quiet_jvp_warnings():
        J = torch.func.jacfwd(fd)(x)
    torch.testing.assert_close(J, 2 * torch.diag(x) + 3 * torch.eye(3, dtype=F64).flip(0), atol=1e-8, rtol=0)


@pytest.mark.parametrize("scheme, per_jacobian", [("central", 2), ("forward", 1)])
def test_evaluation_counts(scheme, per_jacobian):
    """One call for the value, plus ``n`` or ``2n`` per Jacobian, in forward and in reverse mode."""
    n = 4
    fd = piazzi.finite_difference(lambda x: x**2, scheme=scheme)
    x = torch.arange(1.0, n + 1, dtype=F64)
    with quiet_jvp_warnings():
        torch.func.jacfwd(fd)(x)
    assert fd.n_evals == 1 + per_jacobian * n
    before = fd.n_evals
    xg = x.clone().requires_grad_()
    fd(xg).sum().backward()
    assert fd.n_evals - before == 1 + per_jacobian * n


def test_function_gets_a_copy_and_result_is_copied():
    buffer = torch.zeros(2, dtype=F64)

    def fun(x):
        x.mul_(2)  # must not change the caller's tensor
        buffer.copy_(x)
        return buffer  # reused output buffer

    fd = piazzi.finite_difference(fun)
    x = torch.tensor([1.0, 2.0], dtype=F64)
    y = fd(x)
    assert torch.equal(x, torch.tensor([1.0, 2.0], dtype=F64))
    buffer.zero_()
    assert torch.equal(y, torch.tensor([2.0, 4.0], dtype=F64))


def test_step_is_relative_to_the_coordinate():
    """At x ~ 1e6 a step of 1e6 * eps is needed; an absolute step would be lost in the function's slow scale."""
    f = lambda x: torch.sin(x * 1e-6)  # noqa: E731
    x = torch.tensor([1e6, 2e6], dtype=F64)
    with quiet_jvp_warnings():
        J = torch.func.jacfwd(piazzi.finite_difference(f))(x)
    torch.testing.assert_close(J, torch.diag(1e-6 * torch.cos(x * 1e-6)), atol=1e-12, rtol=1e-6)


def test_float32_default_step_is_usable():
    f = lambda x: torch.stack([x[0] ** 2, x[0] * x[1]])  # noqa: E731
    x = torch.tensor([1.0, 2.0])
    with quiet_jvp_warnings():
        J = torch.func.jacfwd(piazzi.finite_difference(f))(x)
    torch.testing.assert_close(J, torch.tensor([[2.0, 0.0], [2.0, 1.0]]), atol=2e-3, rtol=2e-3)


def test_no_parameters_and_zero_tangent():
    fd = piazzi.finite_difference(lambda x: x.sum().reshape(1) + 1)
    xg = torch.zeros(0, dtype=F64, requires_grad=True)
    fd(xg).sum().backward()
    assert xg.grad.shape == (0,)


def test_invalid_arguments():
    with pytest.raises(ValueError, match="scheme"):
        piazzi.finite_difference(lambda x: x, scheme="backward")
    with pytest.raises(ValueError, match="eps"):
        piazzi.finite_difference(lambda x: x, eps=0.0)


@pytest.mark.parametrize("method", ["lm", "gn"])
@pytest.mark.parametrize("problem", LSQ_FD, ids=lambda p: p.name)
def test_least_squares_reaches_the_autodiff_solution(problem, method):
    fd = piazzi.finite_difference(problem.fun)
    res = piazzi.least_squares(fd, problem.x0, method=method, max_iter=500, step_tol=1e-8)
    if problem.x_opt is not None:
        assert torch.allclose(res.x, problem.x_opt, atol=1e-4), (res.x, res.message)
    else:
        assert res.fun.item() <= max(problem.max_cost, 1e-9), (res.fun, res.message)
    # Every solver evaluation costs 2n more calls of fun, which only n_evals reveals.
    assert fd.n_evals > res.n_system + res.n_fun


@pytest.mark.parametrize("problem", SCALAR_PROBLEMS, ids=lambda p: p.name)
def test_minimize_reaches_the_autodiff_solution(problem):
    fd = piazzi.finite_difference(problem.fun)
    res = piazzi.minimize(fd, problem.x0, options={"max_iter": 500, "step_tol": 1e-9})
    assert torch.allclose(res.x, problem.x_opt, atol=1e-4), (res.x, res.message)


def test_robust_loss_with_finite_differences():
    t = torch.linspace(0, 1, 30, dtype=F64)
    y = 2.0 * torch.exp(-1.5 * t)
    y[3] += 5.0  # an outlier
    model = lambda p: p[0] * torch.exp(-p[1] * t) - y  # noqa: E731
    x0 = torch.tensor([1.0, 1.0], dtype=F64)
    ref = piazzi.least_squares(model, x0, loss="cauchy", f_scale=0.1)
    res = piazzi.least_squares(piazzi.finite_difference(model), x0, loss="cauchy", f_scale=0.1)
    torch.testing.assert_close(res.x, ref.x, atol=1e-5, rtol=0)


def test_least_squares_float32():
    t = torch.linspace(0, 1, 20)
    y = 2.0 * torch.exp(-1.5 * t)
    fd = piazzi.finite_difference(lambda p: p[0] * torch.exp(-p[1] * t) - y)
    res = piazzi.least_squares(fd, torch.tensor([1.0, 1.0]), step_tol=1e-5)
    torch.testing.assert_close(res.x, torch.tensor([2.0, 1.5]), atol=2e-3, rtol=0)


@pytest.mark.parametrize("with_manifold", [False, True])
@pytest.mark.parametrize("solver", ["least_squares", "minimize"])
def test_manifold_registration_matches_autodiff(solver, with_manifold):
    """Rotate points with an SE(3) pose; the differences are taken in the tangent space through ``plus``."""
    gen = torch.Generator().manual_seed(0)
    pts = torch.randn(12, 3, dtype=F64, generator=gen)
    truth = piazzi.se3_exp(torch.tensor([0.3, -0.2, 0.1, 0.2, -0.1, 0.15], dtype=F64))
    obs = pts @ truth[:3, :3].T + truth[:3, 3]

    def residuals(T):
        return (pts @ T[:3, :3].T + T[:3, 3] - obs).reshape(-1)

    M = piazzi.SE3()
    T0 = torch.eye(4, dtype=F64)
    if solver == "least_squares":
        run = lambda f: piazzi.least_squares(f, T0, manifold=M, step_tol=1e-10)  # noqa: E731
        fd = piazzi.finite_difference(residuals, manifold=M if with_manifold else None)
    else:
        run = lambda f: piazzi.minimize(f, T0, manifold=M, options={"step_tol": 1e-10, "max_iter": 200})  # noqa: E731
        fd = piazzi.finite_difference(
            lambda T: 0.5 * residuals(T).square().sum(), manifold=M if with_manifold else None
        )
        residuals_scalar = lambda T: 0.5 * residuals(T).square().sum()  # noqa: E731
    res = run(fd)
    ref = run(residuals if solver == "least_squares" else residuals_scalar)
    assert torch.allclose(res.x, truth, atol=1e-5), (res.x, res.message)
    torch.testing.assert_close(res.x, ref.x, atol=1e-5, rtol=0)


def random_pose(seed):
    gen = torch.Generator().manual_seed(seed)
    return piazzi.se3_exp(0.7 * torch.randn(6, dtype=F64, generator=gen))


@pytest.mark.parametrize("scheme", ["central", "forward"])
@pytest.mark.parametrize("manifold, point", [(piazzi.SE3(), random_pose(1)), (piazzi.SO3(), random_pose(2)[:3, :3])])
def test_manifold_differences_stay_on_the_manifold(manifold, point, scheme):
    """With manifold=, fun only sees valid rotations, and the tangent-space Jacobian matches autodiff."""
    seen = []
    pts = torch.randn(5, 3, dtype=F64, generator=torch.Generator().manual_seed(3))

    def fun(T):
        R = T[:3, :3]
        seen.append((R @ R.T - torch.eye(3, dtype=F64)).abs().max().item())
        return (pts @ R.T + (T[:3, 3] if T.shape[0] == 4 else 0)).reshape(-1)

    fd = piazzi.finite_difference(fun, manifold=manifold, scheme=scheme)
    delta0 = torch.zeros(manifold.tangent_size(point), dtype=F64)
    tol = 1e-5 if scheme == "central" else 1e-3
    with quiet_jvp_warnings():
        exact = torch.func.jacfwd(lambda d: fun(manifold.plus(point, d)))(delta0)
        seen.clear()
        got_fwd = torch.func.jacfwd(lambda d: fd(manifold.plus(point, d)))(delta0)
        got_rev = torch.func.jacrev(lambda d: fd(manifold.plus(point, d)))(delta0)
    assert max(seen) < 1e-12  # on the manifold to rounding, not O(eps^2) = 1e-11 and growing with eps
    torch.testing.assert_close(got_fwd, exact, atol=tol, rtol=tol)
    torch.testing.assert_close(got_rev, exact, atol=tol, rtol=tol)


def test_manifold_central_cost_is_two_per_tangent_coordinate():
    M = piazzi.SE3()
    fd = piazzi.finite_difference(lambda T: T[:3, 3] * 2, manifold=M)
    delta0 = torch.zeros(6, dtype=F64)
    with quiet_jvp_warnings():
        torch.func.jacfwd(lambda d: fd(M.plus(torch.eye(4, dtype=F64), d)))(delta0)
    assert fd.n_evals == 1 + 2 * 6  # tangent size, not the 16 entries of the matrix
