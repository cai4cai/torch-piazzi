"""Tests of :func:`piazzi.dogleg` beyond the ported suites: counts, dtypes, odd curvature."""

import math

import pytest
import torch
from problems import fun_rosenbrock

import piazzi

F64 = torch.float64


def exp_problem(dtype):
    t = torch.linspace(0, 4, 40, dtype=dtype)
    y = 2.5 * torch.exp(-1.3 * t)
    return (lambda th: th[0] * torch.exp(-th[1] * t) - y), torch.tensor([1.0, 0.5], dtype=dtype)


@pytest.mark.parametrize("dogleg_type", ["traditional", "subspace"])
@pytest.mark.parametrize("damping", ["identity", "marquardt"])
def test_rosenbrock_converges(dogleg_type, damping):
    res = piazzi.least_squares(
        fun_rosenbrock, torch.tensor([-1.2, 1.0], dtype=F64), method="dogleg", dogleg_type=dogleg_type, damping=damping
    )
    assert res.converged, res.message
    assert torch.allclose(res.x, torch.ones(2, dtype=F64), atol=1e-6)


def test_counts_match_the_callback():
    fun, x0 = exp_problem(F64)
    infos = []
    res = piazzi.least_squares(fun, x0, method="dogleg", callback=infos.append, initial_radius=0.1)
    assert res.converged, res.message
    assert infos[-1].n_system == res.n_system and infos[-1].n_fun == res.n_fun


def run_counts(first_trial):
    infos = []
    res = piazzi.least_squares(
        fun_rosenbrock,
        torch.tensor([-3.0, 2.0], dtype=F64),
        method="dogleg",
        initial_radius=1e4,
        first_trial=first_trial,
        callback=infos.append,
    )
    return res, infos


def test_value_first_trial_never_wastes_a_system():
    # With value-only first trials every full system is evaluated at an accepted point.
    res, infos = run_counts("value")
    assert res.converged, res.message
    assert res.n_system == 1 + sum(i.accepted for i in infos[:-1]) + infos[-1].accepted


def test_system_first_trial_wastes_systems_on_rejections():
    res, infos = run_counts("system")
    assert res.converged, res.message
    assert res.n_system > 1 + sum(i.accepted for i in infos)  # a system at a rejected first trial


def test_auto_first_trial_beats_always_system_when_rejections_are_common():
    auto, _ = run_counts("auto")
    system, _ = run_counts("system")
    assert auto.converged and system.converged
    assert auto.n_system <= system.n_system


def test_a_rejected_trial_costs_no_linear_solve():
    # Rosenbrock from a huge radius: the full Gauss-Newton step is rejected, and the model is reused
    # for the smaller radii. The solves are counted by a custom linear solver.
    solves = []

    def counting(H, g, damping):
        solves.append(1)
        return piazzi.dense_solver(H, g, damping)

    infos = []
    res = piazzi.least_squares(
        fun_rosenbrock,
        torch.tensor([-3.0, 2.0], dtype=F64),
        method="dogleg",
        linear_solver=counting,
        callback=infos.append,
        initial_radius=1e4,
    )
    assert res.converged, res.message
    accepted = sum(i.accepted for i in infos)
    assert len(infos) > accepted, "the huge radius should force at least one rejection"
    assert len(solves) <= accepted + 1  # at most one per linearisation, none per rejected trial


def test_never_steps_uphill_with_an_indefinite_curvature():
    # f = x^4 / 16 - x^2 / 2 per coordinate: negative curvature around 0, minima at +-sqrt(4).
    # H is a negative definite matrix at the start, so there is no Gauss-Newton descent step and
    # the Cauchy (steepest-descent) fallback has to make progress.
    def system(x):
        loss = (x**4 / 16 - x**2 / 2).sum()
        return loss, x**3 / 4 - x, torch.diag(0.75 * x**2 - 1.0)

    x0 = torch.tensor([0.1, -0.2], dtype=F64)
    steps = []
    res = piazzi.dogleg(system, x0, callback=steps.append, initial_radius=0.5)
    accepted = [i for i in steps if i.accepted]
    assert accepted
    assert all(i.fun <= i.fun_start for i in accepted)  # no accepted step goes uphill
    assert res.fun < system(x0)[0]
    assert torch.allclose(res.x.abs(), torch.full((2,), 2.0, dtype=F64), atol=1e-3), res


def test_zero_gradient_and_non_finite_start():
    system = piazzi.least_squares_system(lambda x: x * 0.0 + 1.0)
    res = piazzi.dogleg(system, torch.zeros(2, dtype=F64))
    assert res.converged and res.message == "gradient is zero"
    res = piazzi.least_squares(lambda x: x * math.nan, torch.ones(2, dtype=F64), method="dogleg")
    assert not res.converged and "not finite" in res.message


@pytest.mark.parametrize("dogleg_type", ["traditional", "subspace"])
def test_float32_reports_convergence(dogleg_type):
    fun, x0 = exp_problem(torch.float32)
    res = piazzi.least_squares(fun, x0, method="dogleg", dogleg_type=dogleg_type)
    assert res.converged, res.message
    assert res.x.dtype == torch.float32
    assert torch.allclose(res.x, torch.tensor([2.5, 1.3]), atol=1e-4)


def test_objective_scale_does_not_matter():
    fun, x0 = exp_problem(F64)
    ref = piazzi.least_squares(fun, x0, method="dogleg")
    scaled = piazzi.least_squares(lambda th: 1e-5 * fun(th), x0, method="dogleg")
    assert scaled.converged, scaled.message
    assert torch.allclose(ref.x, scaled.x, atol=1e-5)


def test_bad_arguments():
    system = piazzi.least_squares_system(lambda x: x)
    x0 = torch.ones(1, dtype=F64)
    with pytest.raises(ValueError, match="dogleg_type"):
        piazzi.dogleg(system, x0, dogleg_type="exact")
    with pytest.raises(ValueError, match="damping"):
        piazzi.dogleg(system, x0, damping="x")
    with pytest.raises(ValueError, match="first_trial"):
        piazzi.dogleg(system, x0, first_trial="x")
    with pytest.raises(ValueError, match="radius"):
        piazzi.dogleg(system, x0, initial_radius=0.0)


@pytest.mark.parametrize("dogleg_type", ["traditional", "subspace"])
def test_singular_minimiser_ends_converged(dogleg_type):
    """``r = x^2 + 5`` has J = 0 at its minimiser x = 0: the Gauss-Newton step is no distance estimate there.

    The trust region collapses instead; the run used to end with "too many rejected steps in a row".
    """
    res = piazzi.least_squares(
        lambda x: x**2 + 5.0,
        torch.tensor([2.0], dtype=F64),
        method="dogleg",
        dogleg_type=dogleg_type,
        step_tol=1e-8,
        max_iter=500,
    )
    assert res.converged, res.message
    assert res.x.abs().item() < 1e-6
