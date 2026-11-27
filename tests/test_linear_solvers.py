import pytest
import torch

import piazzi
from piazzi._linear_solvers import diagonal, matvec

D = torch.float64
CSR_BETA = pytest.mark.filterwarnings("ignore:Sparse CSR tensor support is in beta:UserWarning")


def exp_problem():
    t = torch.linspace(0, 4, 40, dtype=D)
    y = 2.5 * torch.exp(-1.3 * t)
    return (lambda th: th[0] * torch.exp(-th[1] * t) - y), torch.tensor([1.0, 0.5], dtype=D)


def spd(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n, n, generator=g, dtype=D)
    H = A @ A.T + 0.1 * torch.eye(n, dtype=D)
    H[H.abs() < 0.5] = 0  # make it sparse-ish, then restore definiteness
    return H + n * torch.eye(n, dtype=D)


@CSR_BETA
@pytest.mark.parametrize("layout", ["coo", "csr"])
def test_diagonal_and_matvec_sparse(layout):
    H = spd(12)
    Hs = H.to_sparse_coo() if layout == "coo" else H.to_sparse_csr()
    v = torch.arange(12, dtype=D)
    torch.testing.assert_close(diagonal(Hs), H.diagonal())
    torch.testing.assert_close(matvec(Hs, v), H @ v)


@CSR_BETA
@pytest.mark.parametrize("preconditioner", ["jacobi", None, "block"])
def test_cg_solver_matches_dense(preconditioner):
    H, g = spd(30, 1), torch.linspace(-1, 1, 30, dtype=D)
    damping = torch.full((30,), 0.3, dtype=D)
    if preconditioner == "block":

        def preconditioner(H, damping):
            M = torch.linalg.inv(H.to_dense() + torch.diag(damping))
            return lambda v: M @ v

    solver = piazzi.cg_solver(tol=1e-12, max_iter=200, preconditioner=preconditioner)
    expected = piazzi.dense_solver(H, g, damping)
    torch.testing.assert_close(solver(H.to_sparse_coo(), g, damping), expected, atol=1e-9, rtol=0)
    torch.testing.assert_close(piazzi.dense_solver(H.to_sparse_csr(), g, damping), expected)


def test_cg_solver_inexact_step_is_silent():
    H, g = spd(30, 2), torch.ones(30, dtype=D)
    d = piazzi.cg_solver(tol=1e-14, max_iter=2)(H.to_sparse_coo(), g, torch.zeros(30, dtype=D))
    assert torch.isfinite(d).all()  # no warning (pytest turns warnings into errors)


def test_cg_solver_rejects_bad_preconditioner():
    with pytest.raises(ValueError, match="preconditioner"):
        piazzi.cg_solver(preconditioner="ilu")


@CSR_BETA
@pytest.mark.parametrize("method", ["lm", "gn"])
@pytest.mark.parametrize("damping", ["identity", "marquardt"])
@pytest.mark.parametrize("layout", ["coo", "csr"])
def test_gauss_newton_with_sparse_hessian(method, damping, layout):
    fun, x0 = exp_problem()
    dense_system = piazzi.least_squares_system(fun)

    def sparse_system(x):
        loss, g, H = dense_system(x)
        return loss, g, H.to_sparse_coo() if layout == "coo" else H.to_sparse_csr()

    kwargs = {"method": method, "damping": damping, "step_tol": 1e-10}
    a = piazzi.gauss_newton(dense_system, x0, **kwargs)
    b = piazzi.gauss_newton(sparse_system, x0, **kwargs)  # default sparse solver: CG
    c = piazzi.gauss_newton(sparse_system, x0, linear_solver=piazzi.dense_solver, **kwargs)
    expected = torch.tensor([2.5, 1.3], dtype=D)
    for res in (a, b, c):
        assert res.converged, res.message
        torch.testing.assert_close(res.x, expected, atol=1e-6, rtol=0)
    torch.testing.assert_close(a.x, c.x)
    assert a.n_iter == c.n_iter


def test_marquardt_damping_handles_badly_scaled_parameters():
    """A model whose two parameters differ in scale by 1e4: Marquardt damping is scale-free."""
    t = torch.linspace(0, 4, 40, dtype=D)
    y = 2.5e4 * torch.exp(-1.3 * t)

    def fun(th):
        return th[0] * torch.exp(-th[1] * t) - y

    x0 = torch.tensor([1.0e4, 0.5], dtype=D)
    res = {
        d: piazzi.least_squares(fun, x0, damping=d, max_step=1e6, step_tol=1e-10, max_iter=200)
        for d in ("identity", "marquardt")
    }
    for r in res.values():
        torch.testing.assert_close(r.x, torch.tensor([2.5e4, 1.3], dtype=D), rtol=1e-8, atol=0)
    assert res["marquardt"].n_iter < res["identity"].n_iter


def test_custom_linear_solver_is_used():
    fun, x0 = exp_problem()
    calls = []

    def solver(H, g, damping):
        calls.append(damping.shape)
        return piazzi.dense_solver(H, g, damping)

    res = piazzi.least_squares(fun, x0, linear_solver=solver)
    assert res.converged and calls and calls[0] == (2,)


def test_invalid_damping():
    fun, x0 = exp_problem()
    with pytest.raises(ValueError, match="damping"):
        piazzi.least_squares(fun, x0, damping="levenberg")
