"""Tests ported from SciPy's ``test_least_squares.py`` and ``test_optimize.py`` (BSD-3-Clause;
see ``problems.py`` for the licence), plus differential tests that run each problem through
both piazzi and the ``piazzi.scipy`` wrappers.

Options that piazzi does not have (bounds, sparse Jacobians, ``x_scale``, ``tr_solver``,
complex ``x0``) are left out. ``SciPy:`` in a docstring names the test it ports.
"""

import math

import pytest
import torch
from problems import (
    F64,
    LSQ_PROBLEMS,
    SCALAR_PROBLEMS,
    entropy,
    fun_rosenbrock,
    fun_trivial,
    rosen,
)

import piazzi

try:
    import piazzi.scipy as pzs
except ImportError:  # SciPy is optional
    pzs = None

needs_scipy = pytest.mark.skipif(pzs is None, reason="SciPy is not installed")

LSQ_IDS = [p.name for p in LSQ_PROBLEMS]
SCALAR_IDS = [p.name for p in SCALAR_PROBLEMS]


def check_lsq_solution(problem, res):
    if problem.x_opt is not None:
        assert torch.allclose(res.x, problem.x_opt, atol=1e-4), (res.x, res.message)
    else:
        assert res.fun.item() <= problem.max_cost, (res.fun, res.message)


# piazzi.least_squares


# Not a singularity problem: cond(J^T J) peaks near 1e6, fine in float64. This large-residual
# problem makes the GN direction poorly aligned with the gradient, so Armijo only accepts tiny
# steps and GN crawls (cost ~25 after 500 iterations). Truncating small singular values helps
# only with a cutoff (~1e-3) so aggressive that GN then stalls on ill-conditioned linear
# problems. LM (and a dog-leg prototype) solve it.
GN_STALLS = pytest.mark.xfail(
    reason="pure Gauss-Newton crawls on this large-residual problem; Levenberg-Marquardt solves it", strict=True
)
LSQ_CASES = [
    pytest.param(p, m, id=f"{p.name}-{m}", marks=GN_STALLS if (p.name, m) == ("bvp", "gn") else ())
    for p in LSQ_PROBLEMS
    for m in ("lm", "gn")
]


@pytest.mark.parametrize("problem, method", LSQ_CASES)
def test_least_squares_problems(problem, method):
    """SciPy: test_basic, test_rosenbrock, test_rosenbrock_cropped, test_bvp, BroydenTridiagonal."""
    res = piazzi.least_squares(problem.fun, problem.x0, method=method, max_iter=500, step_tol=1e-8)
    check_lsq_solution(problem, res)
    assert torch.isfinite(res.fun)


@pytest.mark.parametrize("method", ["lm", "gn"])
def test_least_squares_full_result(method):
    """SciPy: test_full_result (cost 12.5 at x = 0, where the Jacobian vanishes)."""
    res = piazzi.least_squares(fun_trivial, torch.tensor([2.0], dtype=F64), method=method, max_iter=500)
    assert torch.allclose(res.x, torch.zeros(1, dtype=F64), atol=1e-4)
    assert res.fun.item() == pytest.approx(12.5)


def test_least_squares_single_evaluation():
    """SciPy: test_full_result_single_fev; here ``max_iter=0`` evaluates x0 only."""
    res = piazzi.least_squares(fun_trivial, torch.tensor([2.0], dtype=F64), max_iter=0)
    assert torch.equal(res.x, torch.tensor([2.0], dtype=F64))
    assert res.fun.item() == 40.5  # 0.5 * 9^2
    assert res.n_system == 1 and res.n_fun == 0 and not res.converged


@pytest.mark.parametrize("method", ["lm", "gn"])
def test_least_squares_starts_at_optimum(method):
    """x0 is the exact minimiser (zero residuals, so g = 0): stop before any step."""
    res = piazzi.least_squares(fun_rosenbrock, torch.ones(2, dtype=F64), method=method)
    assert torch.equal(res.x, torch.ones(2, dtype=F64))
    assert res.converged and res.n_iter == 0 and res.message == "gradient is zero"


@pytest.mark.parametrize("method", ["lm", "gn"])
@pytest.mark.parametrize("k", range(-4, 5))
def test_least_squares_zero_step_at_nonzero_cost_minimiser(method, k):
    """Starting within rounding of a minimiser with nonzero residuals still converges.

    The exact minimiser is (0.4, 0). Starting a few ulp away, every LM step is ~1e-16 and its
    decrease is lost to rounding; such steps used to be rejected until max_rejections (seen in
    CI on Linux, where lstsq returned (0.4, 1.24e-16)).
    """
    t = torch.linspace(0, 1, 5, dtype=F64)
    y = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0], dtype=F64)
    fun = lambda th: th[0] + th[1] * t - y  # noqa: E731
    res = piazzi.least_squares(fun, torch.tensor([0.4, k * 3.1e-17], dtype=F64), method=method)
    assert torch.allclose(res.x, torch.tensor([0.4, 0.0], dtype=F64), atol=1e-12)
    assert res.converged, res.message
    assert res.n_iter <= 2


@pytest.mark.parametrize("method", ["lm", "gn"])
def test_least_squares_zero_jacobian_at_x0(method):
    """At x0 = 0 the trivial problem has J = 0, so g = 0 and H = 0: x0 is already stationary."""
    res = piazzi.least_squares(fun_trivial, torch.zeros(1, dtype=F64), method=method)
    assert torch.equal(res.x, torch.zeros(1, dtype=F64))
    assert res.converged, res.message


def test_least_squares_x0_multidimensional():
    """SciPy: test_x0_multidimensional raises, but piazzi accepts any x0 shape."""
    target = torch.arange(4, dtype=F64).reshape(2, 2)
    res = piazzi.least_squares(lambda x: (x - target).reshape(-1), torch.zeros(2, 2, dtype=F64), max_step=10.0)
    assert res.x.shape == (2, 2) and torch.allclose(res.x, target)


def test_least_squares_rejects_complex_x0():
    """SciPy: test_x0_complex_scalar, test_x0_complex_array."""
    with pytest.raises(TypeError):
        piazzi.least_squares(fun_trivial, torch.tensor([2.0 + 0j]))
    with pytest.raises(TypeError):
        piazzi.least_squares(fun_trivial, torch.tensor([1, 2]))  # integer


@pytest.mark.parametrize("method", ["lm", "gn"])
def test_least_squares_respects_max_iter(method):
    """SciPy: test_nfev / test_respect_maxiter."""
    for max_iter in (1, 2, 3):
        res = piazzi.least_squares(
            fun_rosenbrock, torch.tensor([-2.0, 1.0], dtype=F64), method=method, max_iter=max_iter
        )
        assert res.n_iter <= max_iter


@pytest.mark.parametrize("method", ["lm", "gn"])
def test_least_squares_callback_does_not_alias(method):
    """SciPy: test_minimize_callback_copies_array; iterates seen by callbacks stay intact."""
    seen = []
    res = piazzi.least_squares(
        fun_rosenbrock, torch.tensor([-2.0, 1.0], dtype=F64), method=method, callback=lambda i: seen.append(i.x.clone())
    )
    xs = []
    piazzi.least_squares(
        fun_rosenbrock, torch.tensor([-2.0, 1.0], dtype=F64), method=method, callback=lambda i: xs.append(i.x)
    )
    assert len(xs) == len(seen) > 1
    assert all(torch.equal(a, b) for a, b in zip(xs, seen, strict=True))
    assert torch.equal(xs[-1], res.x)


@pytest.mark.parametrize("method", ["lm", "gn"])
def test_least_squares_nan_residuals(method):
    """SciPy: test_nan_values; NaN residuals end the run as not converged, without a crash."""
    res = piazzi.least_squares(lambda x: x * math.nan, torch.ones(2, dtype=F64), method=method)
    assert not res.converged and torch.isnan(res.fun)
    assert res.n_iter <= 1


@pytest.mark.parametrize("method", ["lm", "gn"])
def test_least_squares_nan_trial_point_is_rejected(method):
    """A trial point with NaN residuals must not be accepted; the solver backs off instead."""

    def fun(x):  # NaN beyond x = 1.5, minimiser at x = 1
        # NaN added as a constant: torch.where(c, r * nan, r) would also give NaN gradients.
        return (x - 1) + torch.where(x > 1.5, math.nan, 0.0)

    res = piazzi.least_squares(fun, torch.zeros(1, dtype=F64), method=method, max_step=10.0)
    assert torch.allclose(res.x, torch.ones(1, dtype=F64), atol=1e-6), res
    assert torch.isfinite(res.fun)


# piazzi.minimize


@pytest.mark.parametrize("problem", SCALAR_PROBLEMS, ids=SCALAR_IDS)
def test_minimize_problems(problem):
    """SciPy: CheckOptimize (entropy), test_rosenbrock, test_result_x_shape_when_len_x_is_one."""
    res = piazzi.minimize(problem.fun, problem.x0, options={"max_iter": 2000, "step_tol": 1e-10})
    assert res.converged, res.message
    assert res.x.shape == problem.x0.shape
    assert torch.allclose(res.x, problem.x_opt, atol=1e-5), res.x


def test_minimize_entropy_value():
    """SciPy: CheckOptimize compares function values at the solution."""
    res = piazzi.minimize(entropy, torch.zeros(3, dtype=F64))
    assert res.fun.item() == pytest.approx(entropy(torch.tensor([0.0, -0.524869316, 0.487525860], dtype=F64)).item())


def test_minimize_respects_max_iter():
    """SciPy: test_respect_maxiter."""
    for max_iter in (1, 5, 20):
        res = piazzi.minimize(rosen, torch.zeros(10, dtype=F64), options={"max_iter": max_iter})
        assert res.n_iter <= max_iter


def test_minimize_callback_does_not_alias():
    """SciPy: test_minimize_callback_copies_array."""
    xs = []
    res = piazzi.minimize(rosen, torch.zeros(10, dtype=F64), callback=lambda i: xs.append(i.x))
    clones = []
    piazzi.minimize(rosen, torch.zeros(10, dtype=F64), callback=lambda i: clones.append(i.x.clone()))
    assert len(xs) == len(clones) > 1
    assert all(torch.equal(a, b) for a, b in zip(xs, clones, strict=True))
    assert torch.equal(xs[-1], res.x)


def test_minimize_nan_return():
    """SciPy: test_bfgs_nan_return (NaN from the first call)."""
    res = piazzi.minimize(lambda x: x.sum() * math.nan, torch.zeros(1, dtype=F64))
    assert torch.isnan(res.fun) and not res.converged
    assert res.n_iter <= 1


def test_minimize_nan_x0():
    """SciPy: test_bfgs_nan (x0 = NaN)."""
    res = piazzi.minimize(lambda x: x.sum(), torch.tensor([math.nan], dtype=F64))
    assert torch.isnan(res.fun) and not res.converged


def test_minimize_nan_trial_point_is_rejected():
    """SciPy: test_nan_values (func2 turns NaN after a few calls); NaN trial points are not accepted."""

    def fun(x):  # NaN beyond x = 1.5, minimiser at x = 1
        return ((x - 1) ** 2).sum() + torch.where(x.sum() > 1.5, math.nan, 0.0)

    res = piazzi.minimize(fun, torch.zeros(1, dtype=F64), options={"max_step": 10.0, "step_tol": 1e-10})
    assert torch.allclose(res.x, torch.ones(1, dtype=F64), atol=1e-6), res
    assert torch.isfinite(res.fun)


# Differential tests: the same problems through SciPy.


@needs_scipy
@pytest.mark.parametrize("problem", LSQ_PROBLEMS, ids=LSQ_IDS)
def test_least_squares_matches_scipy(problem):
    ours = piazzi.least_squares(problem.fun, problem.x0, max_iter=500, step_tol=1e-8)
    theirs = pzs.least_squares(problem.fun, problem.x0)
    check_lsq_solution(problem, theirs)
    # Where the minimiser is unique, both must find it with the same cost. Otherwise (several
    # local minima, or a manifold of zero-cost solutions) each only has to reach max_cost.
    if problem.x_opt is not None:
        assert torch.allclose(ours.x, theirs.x, atol=1e-4)
        assert ours.fun.item() == pytest.approx(theirs.fun.item(), rel=1e-6, abs=1e-12)


@needs_scipy
@pytest.mark.parametrize("method", ["BFGS", "L-BFGS-B"])
@pytest.mark.parametrize("problem", SCALAR_PROBLEMS, ids=SCALAR_IDS)
def test_minimize_matches_scipy(problem, method):
    ours = piazzi.minimize(problem.fun, problem.x0, options={"max_iter": 2000, "step_tol": 1e-10})
    options = {"gtol": 1e-10, "ftol": 1e-15} if method == "L-BFGS-B" else {"gtol": 1e-10}
    theirs = pzs.minimize(problem.fun, problem.x0, method=method, options=options)
    assert torch.allclose(theirs.x, problem.x_opt, atol=1e-4), theirs.message
    assert torch.allclose(ours.x, theirs.x, atol=1e-4)
    # L-BFGS-B stops on its relative-decrease test (ftol), so near-zero minima differ by ~1e-12.
    assert ours.fun.item() == pytest.approx(theirs.fun.item(), rel=1e-8, abs=1e-10)
