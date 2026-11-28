"""Tests ported from Ceres Solver's dog-leg trust-region strategy and minimiser tests.

Sources (ceres-solver, master):

- ``internal/ceres/dogleg_strategy_test.cc``: the ellipse and valley fixtures and their tests
  (TrustRegionObeyed, CorrectGaussNewtonStep, CorrectStepLocalOptimumAlongGradient,
  CorrectStepGlobalOptimumAlongGradient).
- ``internal/ceres/trust_region_minimizer_test.cc``: ``PowellsSingularFunctionUsingDogleg``.

Differences from Ceres:

- Ceres tests ``DoglegStrategy::ComputeStep`` directly, on a Jacobian and a residual. piazzi has
  no strategy object, so the one step is taken through :func:`piazzi.dogleg` with
  ``max_iter=1``: on these exactly quadratic problems the step from the origin is always
  accepted, and the returned ``x`` is that step. ``initial_radius == max_radius`` pins the
  radius, as in the Ceres tests.
- Ceres's ``ValidSubspaceBasis`` inspects the strategy's private basis; it has no counterpart
  here. The subspace step is instead checked against a brute-force minimisation of the model
  on the boundary (``test_subspace_step_is_the_boundary_minimiser_of_the_subspace_model``).
- Ceres's ``min_lm_diagonal = max_lm_diagonal = 1`` (an identity trust-region scaling) is
  ``damping="identity"``.

Ceres Solver - A fast non-linear least squares minimizer
Copyright 2015 Google Inc. All rights reserved.
http://ceres-solver.org/

Redistribution and use in source and binary forms, with or without modification, are
permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this list of
  conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice, this list of
  conditions and the following disclaimer in the documentation and/or other materials
  provided with the distribution.
* Neither the name of Google Inc. nor the names of its contributors may be used to endorse
  or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS
OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
OF THE POSSIBILITY OF SUCH DAMAGE.

Author of the original tests: moll.markus@arcor.de (Markus Moll)
"""
# ruff: noqa: E501  (the basis matrix is copied from Ceres as is)

import itertools
import math

import pytest
import torch

import piazzi

D = torch.float64
EPS = torch.finfo(D).eps
K_TOLERANCE_LOOSE = 1e-5

# dogleg_strategy_test.cc, DoglegStrategyFixtureEllipse.
BASIS = torch.tensor(
    [
        [-0.1046920933796121, -0.7449367449921986, -0.4190744502875876, -0.4480450716142566, 0.2375351607929440, -0.0363053418882862],
        [0.4064975684355914, 0.2681113508511354, -0.7463625494601520, -0.0803264850508117, -0.4463149623021321, 0.0130224954867195],
        [-0.5514387729089798, 0.1026621026168657, -0.5008316122125011, 0.5738122212666414, 0.2974664724007106, 0.1296020877535158],
        [0.5037835370947156, 0.2668479925183712, -0.1051754618492798, -0.0272739396578799, 0.7947481647088278, -0.1776623363955670],
        [-0.4005458426625444, 0.2939330589634109, -0.0682629380550051, -0.2895448882503687, -0.0457239396341685, -0.8139899477847840],
        [-0.3247764582762654, 0.4528151365941945, -0.0276683863102816, -0.6155994592510784, 0.1489240599972848, 0.5362574892189350],
    ],
    dtype=D,
)  # fmt: skip
DDIAG = torch.tensor([1.0, 2.0, 4.0, 8.0, 16.0, 32.0], dtype=D)


def ellipse():
    """``J^T J = Q diag(1..32) Q^T`` with the minimum of the quadratic at ``(1, ..., 1)``."""
    J = DDIAG.sqrt().diag() @ BASIS
    return J, -J @ torch.ones(6, dtype=D)


def valley():
    """``J^T J = diag(1..32)`` with the minimum at ``(0, 0, 1, 0, 0, 0)``."""
    J = DDIAG.diag()
    minimum = torch.zeros(6, dtype=D)
    minimum[2] = 1.0
    return J, -J @ minimum


def one_step(fixture, dogleg_type, radius):
    """The step from the origin that Ceres's ``ComputeStep`` returns, taken through ``piazzi.dogleg``."""
    J, r0 = fixture
    system = piazzi.least_squares_system(lambda x: J @ x + r0)
    res = piazzi.dogleg(
        system,
        torch.zeros(6, dtype=D),
        dogleg_type=dogleg_type,
        initial_radius=radius,
        max_radius=radius,
        max_iter=1,
    )
    return res


@pytest.mark.parametrize("dogleg_type", ["traditional", "subspace"])
def test_trust_region_is_obeyed(dogleg_type):
    # The minimum is at distance sqrt(6) from the origin: a radius of 2 must be respected.
    res = one_step(ellipse(), dogleg_type, 2.0)
    assert res.x.norm() <= 2.0 * (1.0 + 4.0 * EPS)
    assert res.x.norm() > 1.5  # and the step uses the region
    assert res.fun < 0.5 * ellipse()[1].square().sum()  # accepted: the model is exact


@pytest.mark.parametrize("dogleg_type", ["traditional", "subspace"])
def test_correct_gauss_newton_step(dogleg_type):
    res = one_step(ellipse(), dogleg_type, 10.0)
    assert torch.allclose(res.x, torch.ones(6, dtype=D), atol=K_TOLERANCE_LOOSE)


@pytest.mark.parametrize("dogleg_type", ["traditional", "subspace"])
def test_correct_step_local_optimum_along_gradient(dogleg_type):
    # The gradient and the Gauss-Newton step are parallel and the trust region is active.
    res = one_step(valley(), dogleg_type, 0.25)
    expected = torch.zeros(6, dtype=D)
    expected[2] = 0.25
    assert torch.allclose(res.x, expected, atol=K_TOLERANCE_LOOSE)


@pytest.mark.parametrize("dogleg_type", ["traditional", "subspace"])
def test_correct_step_global_optimum_along_gradient(dogleg_type):
    # Parallel again, with the trust region inactive.
    res = one_step(valley(), dogleg_type, 2.0)
    expected = torch.zeros(6, dtype=D)
    expected[2] = 1.0
    assert torch.allclose(res.x, expected, atol=K_TOLERANCE_LOOSE)


def test_subspace_step_is_the_boundary_minimiser_of_the_subspace_model():
    # Replaces Ceres's ValidSubspaceBasis: the step lies in span(gradient, Gauss-Newton step) and,
    # among the points of that plane on the boundary, minimises the quadratic model.
    J, r0 = ellipse()
    H, g = J.T @ J, J.T @ r0
    gn = -torch.linalg.solve(H, g)
    radius = 2.0
    x = one_step(ellipse(), "subspace", radius).x
    basis, _ = torch.linalg.qr(torch.stack([-g, gn], dim=1))
    # The solver's Gauss-Newton step carries a ridge of 1e-8, hence the tolerance.
    assert (x - basis @ (basis.T @ x)).norm() < 1e-6
    assert math.isclose(x.norm().item(), radius, rel_tol=1e-12)

    def model(v):
        return (g @ v + 0.5 * v @ H @ v).item()

    angles = torch.linspace(0, 2 * math.pi, 20001, dtype=D)
    best = min(model(radius * basis @ torch.stack([a.cos(), a.sin()])) for a in angles)
    assert model(x) <= best + 1e-6
    # ... which beats the traditional step's model value at the same radius.
    assert model(x) <= model(one_step(ellipse(), "traditional", radius).x) + 1e-12


def powell(z, active):
    """Powell's singular function (``PowellEvaluator2``) on the active entries; the rest are 0."""
    x = torch.zeros(4, dtype=z.dtype)
    x[list(active)] = z
    f1 = x[0] + 10.0 * x[1]
    f2 = math.sqrt(5.0) * (x[2] - x[3])
    f3 = (x[1] - 2.0 * x[2]) ** 2
    f4 = math.sqrt(10.0) * (x[0] - x[3]) ** 2
    return torch.stack([f1, f2, f3, f4])


# Ceres excludes (True, True, False, True) and (True, True, True, True): they reach a local minimum.
POWELL_CASES = [
    mask
    for mask in itertools.product([False, True], repeat=4)
    if any(mask) and mask not in [(True, True, False, True), (True, True, True, True)]
]


@pytest.mark.parametrize("dogleg_type", ["traditional", "subspace"])
@pytest.mark.parametrize("mask", POWELL_CASES)
def test_powells_singular_function(mask, dogleg_type):
    active = [i for i, m in enumerate(mask) if m]
    start = torch.tensor([3.0, -1.0, 0.0, 1.0], dtype=D)[active]
    res = piazzi.least_squares(
        lambda z: powell(z, active),
        start,
        method="dogleg",
        dogleg_type=dogleg_type,
        initial_radius=1e4,
        max_radius=1e20,
        step_tol=1e-12,
        max_iter=500,
    )
    assert torch.allclose(res.x, torch.zeros_like(res.x), atol=1e-3)
