"""Derivative checks for the Lie-group maps and the manifolds' ``plus``.

The exp/log maps switch between closed forms and Taylor series (near the identity) and, for
``so3_log``, a symmetric-part formula (near a half-turn). Each branch must give finite,
correct first and second derivatives in reverse and forward mode, including at the switches,
because the solvers differentiate through them with ``jacfwd`` and users backpropagate through
them. ``torch.autograd.gradcheck`` compares every derivative with finite differences in
float64.

The log maps are checked along the group, as ``w -> log(R0 @ exp(w))`` near ``w = 0``: their
finite differences then stay on SO(3) / SE(3), where the maps are defined.
"""

import math

import pytest
import torch
from torch.autograd import gradcheck, gradgradcheck

import piazzi
from piazzi._common import quiet_jvp_warnings

D = torch.float64
AXIS = torch.tensor([0.36, -0.48, 0.8], dtype=D)  # unit vector, no symmetry with the coordinate axes

# Rotation angles at and around each branch switch:
# - so3_exp / se3_exp use series below theta^2 = 1e-2 (theta = 0.1);
# - so3_log uses series below sin^2(theta) = 1e-3 (theta ~ 0.0316) and the symmetric-part
#   formula above theta = 2 pi / 3 (cos(theta) < -0.5).
EXP_ANGLES = [0.0, 1e-8, 1e-3, 0.0999, 0.1001, 1.0, 2.5, math.pi - 1e-3]
LOG_ANGLES = [1e-8, 1e-3, 0.0316, 0.0317, 1.0, 2 * math.pi / 3 - 1e-4, 2 * math.pi / 3 + 1e-4, 3.0, math.pi - 1e-4]


def check(fn, *inputs):
    """First derivatives (reverse and forward mode) and second derivatives (reverse-over-reverse
    and forward-over-reverse) against finite differences."""
    inputs = tuple(x.detach().clone().requires_grad_(True) for x in inputs)
    with quiet_jvp_warnings():
        assert gradcheck(fn, inputs, check_forward_ad=True, check_batched_grad=True)
        assert gradgradcheck(fn, inputs, check_fwd_over_rev=True)


def rotvec(angle):
    return angle * AXIS


@pytest.mark.parametrize("angle", EXP_ANGLES)
def test_so3_exp(angle):
    check(piazzi.so3_exp, rotvec(angle))


@pytest.mark.parametrize("angle", EXP_ANGLES)
def test_se3_exp(angle):
    check(piazzi.se3_exp, torch.cat([torch.tensor([0.3, -1.2, 2.0], dtype=D), rotvec(angle)]))


def test_so3_hat():
    check(piazzi.so3_hat, torch.tensor([0.3, -1.2, 2.0], dtype=D))


def test_exp_batched():
    """A batch mixing every branch: no branch's safe input may leak into another's derivative."""
    omega = torch.stack([rotvec(a) for a in EXP_ANGLES])
    check(piazzi.so3_exp, omega)
    check(piazzi.se3_exp, torch.cat([torch.ones_like(omega), omega], -1))


@pytest.mark.parametrize("angle", LOG_ANGLES)
def test_so3_log(angle):
    R0 = piazzi.so3_exp(rotvec(angle))
    check(lambda w: piazzi.so3_log(R0 @ piazzi.so3_exp(w)), torch.zeros(3, dtype=D))


@pytest.mark.parametrize("angle", LOG_ANGLES)
def test_se3_log(angle):
    T0 = piazzi.se3_exp(torch.cat([torch.tensor([0.3, -1.2, 2.0], dtype=D), rotvec(angle)]))
    check(lambda xi: piazzi.se3_log(T0 @ piazzi.se3_exp(xi)), torch.zeros(6, dtype=D))


@pytest.mark.parametrize("mode", ["fwd", "rev"])
def test_log_derivatives_finite_at_exact_half_turn(mode):
    """At an exact half-turn ``sin(theta) = 0``: the log maps aren't differentiable there (the axis
    flips sign), so gradcheck doesn't apply, but their derivatives must stay finite. The square
    root of ``sin^2`` is floored to keep its derivative finite."""
    jac = torch.func.jacfwd if mode == "fwd" else torch.func.jacrev
    R = 2 * torch.outer(AXIS, AXIS) - torch.eye(3, dtype=D)  # exactly symmetric, so sin^2 is exactly 0
    T = torch.eye(4, dtype=D)
    T[:3, :3], T[:3, 3] = R, torch.tensor([0.3, -1.2, 2.0], dtype=D)
    with quiet_jvp_warnings():
        assert torch.isfinite(jac(piazzi.so3_log)(R)).all()
        assert torch.isfinite(jac(piazzi.se3_log)(T)).all()


def test_log_inverts_exp_on_each_branch():
    """``log(exp(w)) == w`` on every branch, including both sides of each switch."""
    for angle in LOG_ANGLES:
        w = rotvec(angle)
        torch.testing.assert_close(piazzi.so3_log(piazzi.so3_exp(w)), w, atol=1e-12, rtol=0)
        xi = torch.cat([torch.tensor([0.3, -1.2, 2.0], dtype=D), w])
        torch.testing.assert_close(piazzi.se3_log(piazzi.se3_exp(xi)), xi, atol=1e-10, rtol=0)


POSE = piazzi.se3_exp(torch.tensor([0.5, -0.2, 1.0, 0.4, -0.3, 1.1], dtype=D))
PIVOT = torch.tensor([10.0, -5.0, 3.0], dtype=D)
STEPS = {"zero": torch.zeros(6, dtype=D), "generic": torch.tensor([0.1, -0.3, 0.2, 0.5, 0.2, -0.4], dtype=D)}


@pytest.mark.parametrize("step", STEPS)
@pytest.mark.parametrize("side", ["right", "left"])
@pytest.mark.parametrize("retraction", ["decoupled", "exp"])
@pytest.mark.parametrize("pivot", [None, PIVOT], ids=["origin", "pivot"])
def test_se3_plus(step, side, retraction, pivot):
    M = piazzi.SE3(side=side, retraction=retraction, pivot=pivot)
    check(lambda d: M.plus(POSE, d), STEPS[step])


@pytest.mark.parametrize("step", STEPS)
@pytest.mark.parametrize("side", ["right", "left"])
def test_so3_plus(step, side):
    check(lambda d: piazzi.SO3(side=side).plus(POSE[:3, :3], d), STEPS[step][3:])


@pytest.mark.parametrize("step", STEPS)
def test_product_plus(step):
    M = piazzi.Product(piazzi.SE3(pivot=PIVOT), piazzi.Subset(3, constant=[1]), piazzi.SO3())
    x = M.join(POSE, torch.tensor([1.0, 2.0, 3.0], dtype=D), POSE[:3, :3])
    delta = torch.cat([STEPS[step], torch.tensor([0.1, -0.2], dtype=D), STEPS[step][3:]])
    check(lambda d: M.plus(x, d), delta)
