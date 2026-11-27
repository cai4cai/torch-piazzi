"""Manifolds for the solvers' parameters, in the style of Ceres Solver's ``Manifold``.

A manifold tells a solver how to move on a constrained parameter set. ``plus(x, delta)``
(Ceres's ``Plus``, written x ⊞ δ) moves the point ``x`` by a tangent vector ``delta`` and
returns a point on the manifold again. The solvers then work in tangent coordinates:

- the Jacobian (or gradient) is taken with respect to ``delta`` at ``delta = 0``, by
  autodiff through ``plus``, so a pose stored as a 4 x 4 matrix gives 6 columns, not 16;
- each step is applied as ``x <- plus(x, step)``, so the chart is re-centred at the current
  iterate after every step and never drifts far from it.

Unlike Ceres, the solvers need no ``PlusJacobian`` (autodiff composes through ``plus``) and
no ``Minus`` (steps are already tangent vectors). The built-in manifolds still provide
``minus(y, x)`` (the tangent vector with ``plus(x, minus(y, x)) == y``), to measure errors
between points (for example pose errors) and to check the same invariants as Ceres's
``manifold_test_utils.h``.

Design choices, and how they relate to other libraries:

- Ceres Solver (https://ceres-solver.readthedocs.io/latest/nnls_modeling.html#manifold)
  attaches a ``Manifold`` (``Plus``/``Minus`` and their Jacobians) to each parameter block,
  and builds rigid transforms as a ``ProductManifold`` of a quaternion and a 3-vector; it
  leaves Lie groups such as SE(3) to Sophus. piazzi follows this design: a manifold is a
  retraction passed to the solver, and :class:`Product` combines blocks (for example a pose
  and camera intrinsics). It adds a dedicated :class:`SE3` for convenience, since a pose is
  the common case.
- PyPose (https://github.com/pypose/pypose/tree/main/pypose/lietensor) puts the group in
  the tensor type instead: ``LieTensor`` is a ``torch.Tensor`` subclass (SE3 stored as
  ``[t, q]``), updated by ``X <- Exp(delta) @ X``, with custom autograd functions that pad
  gradients to the 7 stored numbers. piazzi keeps parameters as plain tensors, so any
  PyTorch code (and forward-mode ``torch.func.jacfwd``) works on them unchanged, and the
  group lives only in ``plus``. Its tangent order ``(rho, omega)`` (translation first)
  matches PyPose and Sophus.
- GeoTorch (https://github.com/lezcano/geotorch) constrains ``nn.Module`` weights with
  ``torch.nn.utils.parametrize``: an unconstrained tensor is mapped onto the manifold
  (``matrix_exp`` or Cayley of a skew matrix for SO(n)). It targets first-order optimisers,
  stores SO(3) as a full 3 x 3 raw tensor (a rank-deficient 9-column Jacobian for
  Gauss-Newton) and has no SE(3). piazzi uses a minimal tangent chart instead, re-centred
  at every step, so Gauss-Newton sees exactly the manifold's degrees of freedom.

Two choices matter most in practice for pose problems:

- Which point rotations are about. Ceres's quaternion manifold and PyPose perturb on the
  left (about the world origin); for an object far from the origin a small rotation then
  also moves it a long way, which couples rotation and translation. :class:`SE3` perturbs
  on the right by default and takes a ``pivot`` (as nanodrr does about the isocentre).
- Decoupled vs screw retraction, and SE(3) vs ``Product(SO3, R^3)``. All of these describe
  the same set of poses, so they share minimisers. The decoupled and screw retractions have
  identical derivatives at ``delta = 0`` (``V(0) = I``), so they give the same Gauss-Newton
  step and differ only in the curve along which it is applied. A product with a
  world-frame translation has a Jacobian that differs by an invertible linear map (the
  translation columns are rotated by ``R``); Gauss-Newton steps are invariant to that, but
  Levenberg-Marquardt damping, L-BFGS scaling and ``max_step`` are not. The decoupled update
  (``rho`` added as a body-frame translation) is the simpler default; the true SE(3)
  exponential is available as ``retraction="exp"``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

__all__ = [
    "SE3",
    "SO3",
    "Euclidean",
    "Manifold",
    "Product",
    "Subset",
    "se3_exp",
    "se3_log",
    "so3_exp",
    "so3_hat",
    "so3_log",
]


class Manifold:
    """Base class: a parameter set with a retraction ``plus``.

    Subclasses set ``ambient_shape`` (the shape a point is stored in, or None for "any") and
    implement :meth:`tangent_size` and :meth:`plus`. Points may be passed flattened or in
    ``ambient_shape``; ``plus`` returns the shape it was given.
    """

    ambient_shape: tuple[int, ...] | None = None

    @property
    def ambient_size(self) -> int:
        if self.ambient_shape is None:
            raise ValueError(f"{type(self).__name__} has no fixed size; give it a shape")
        return math.prod(self.ambient_shape)

    def tangent_size(self, x: torch.Tensor) -> int:
        """Number of degrees of freedom at ``x`` (the length of ``delta``)."""
        raise NotImplementedError

    def plus(self, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """Move ``x`` by the tangent vector ``delta``; ``plus(x, 0) == x``."""
        raise NotImplementedError

    def minus(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Tangent vector at ``x`` that reaches ``y``: ``plus(x, minus(y, x)) == y``.

        Optional: the solvers do not use it. Where ``plus`` wraps around (rotations by more
        than pi), ``minus`` returns the shortest such vector.
        """
        raise NotImplementedError


class Euclidean(Manifold):
    """Plain vector space: ``plus(x, delta) = x + delta``.

    This is what the solvers do without a manifold. ``shape`` is only needed inside a
    :class:`Product`, where each block must have a known size.
    """

    def __init__(self, shape: int | Sequence[int] | None = None):
        self.ambient_shape = None if shape is None else ((shape,) if isinstance(shape, int) else tuple(shape))

    def tangent_size(self, x):
        return x.numel()

    def plus(self, x, delta):
        return x + delta.reshape(x.shape)

    def minus(self, y, x):
        return (y - x).reshape(-1)


class Subset(Manifold):
    """A vector of ``size`` entries whose ``constant`` entries are held fixed.

    Like Ceres's ``SubsetManifold``: the tangent vector holds the free entries, in order.
    Inside a :class:`Product` this fixes part of a block, for example one camera intrinsic.
    """

    def __init__(self, size: int, constant: Sequence[int]):
        constant = list(constant)
        if any(i < 0 for i in constant):
            raise ValueError("constant indices must be greater than or equal to zero")
        if any(i >= size for i in constant):
            raise ValueError(f"constant indices must be less than the size ({size})")
        if len(set(constant)) != len(constant):
            raise ValueError("constant indices contain duplicates")
        self.ambient_shape = (size,)
        self.free = [i for i in range(size) if i not in set(constant)]

    def tangent_size(self, x):
        return len(self.free)

    def plus(self, x, delta):
        embed = torch.eye(self.ambient_size, dtype=x.dtype, device=x.device)[:, self.free]
        return (x.reshape(-1) + embed @ delta.reshape(-1)).reshape(x.shape)

    def minus(self, y, x):
        return (y - x).reshape(-1)[self.free]


# Taylor coefficients of A = sin(t)/t, B = (1 - cos t)/t^2, C = (t - sin t)/t^3 in powers of t^2.
_SERIES = {
    "A": (1.0, -1 / 6, 1 / 120, -1 / 5040, 1 / 362880),
    "B": (1 / 2, -1 / 24, 1 / 720, -1 / 40320, 1 / 3628800),
    "C": (1 / 6, -1 / 120, 1 / 5040, -1 / 362880, 1 / 39916800),
}
# Below this theta^2 the series are used: their truncation error (theta^10 / 11!) is below
# float64 rounding, and the closed forms (C especially) lose digits to cancellation.
_SMALL_THETA2 = 1e-2


def _horner(coeffs, z):
    out = torch.full_like(z, coeffs[-1])
    for c in reversed(coeffs[:-1]):
        out = out * z + c
    return out


def _exp_coeffs(omega: torch.Tensor):
    """``A, B, C`` of the SO(3)/SE(3) exponential for rotation vectors (..., 3), shaped (..., 1, 1).

    Both branches are evaluated with a safe angle, so values and derivatives (forward and
    reverse mode) are finite at ``omega = 0``.
    """
    # keepdim: in forward-mode AD (torch 2.15), a 0-dim float32 tensor combined with a Python
    # scalar gets a float64 tangent, which then breaks matmuls with float32 tensors.
    theta2 = (omega * omega).sum(-1, keepdim=True)
    small = theta2 < _SMALL_THETA2
    theta2_safe = torch.where(small, torch.ones_like(theta2), theta2)
    theta = theta2_safe.sqrt()
    sin_t = torch.sin(theta)
    half = torch.sin(0.5 * theta)
    A = torch.where(small, _horner(_SERIES["A"], theta2), sin_t / theta)
    B = torch.where(small, _horner(_SERIES["B"], theta2), 2 * half * half / theta2_safe)
    C = torch.where(small, _horner(_SERIES["C"], theta2), (theta - sin_t) / (theta2_safe * theta))
    return A[..., None], B[..., None], C[..., None]


def so3_hat(omega: torch.Tensor) -> torch.Tensor:
    """Skew-symmetric matrix (..., 3, 3) with ``hat(w) @ v == cross(w, v)``."""
    x, y, z = omega.unbind(-1)
    o = torch.zeros_like(x)
    return torch.stack([o, -z, y, z, o, -x, -y, x, o], -1).reshape(*omega.shape[:-1], 3, 3)


def so3_exp(omega: torch.Tensor) -> torch.Tensor:
    """Rotation matrix (..., 3, 3) of the rotation vector ``omega`` (..., 3) (Rodrigues)."""
    A, B, _ = _exp_coeffs(omega)
    K = so3_hat(omega)
    eye = torch.eye(3, dtype=omega.dtype, device=omega.device)
    return eye + A * K + B * (K @ K)


def _se3_exp_parts(xi: torch.Tensor, screw: bool):
    rho, omega = xi[..., :3], xi[..., 3:]
    A, B, C = _exp_coeffs(omega)
    K = so3_hat(omega)
    K2 = K @ K
    eye = torch.eye(3, dtype=xi.dtype, device=xi.device)
    R = eye + A * K + B * K2
    t = ((eye + B * K + C * K2) @ rho[..., None])[..., 0] if screw else rho
    return R, t


def _vee(M):
    return torch.stack([M[..., 2, 1], M[..., 0, 2], M[..., 1, 0]], -1)


# Taylor coefficients of asin(s)/s in powers of s^2: theta/sin(theta) as a function of sin^2.
_ASIN_SERIES = (1.0, 1 / 6, 3 / 40, 5 / 112, 35 / 1152, 63 / 2816)
_SMALL_SIN2 = 1e-3


def so3_log(R: torch.Tensor) -> torch.Tensor:
    """Rotation vector (..., 3) of the rotation matrix ``R`` (..., 3, 3), with angle in [0, pi].

    Three branches, all evaluated with safe inputs so derivatives stay finite: a series
    near the identity, ``theta / sin(theta) * vee(R - R^T) / 2`` for moderate angles, and,
    for angles above 2 pi / 3 (where ``sin`` loses precision), the axis from the symmetric
    part ``(R + R^T) / 2 - cos(theta) I = (1 - cos(theta)) a a^T``.
    """
    eye = torch.eye(3, dtype=R.dtype, device=R.device)
    v = 0.5 * _vee(R - R.mT)  # sin(theta) * axis
    cos = ((R.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True) - 1) / 2).clamp(-1, 1)
    s2 = (v * v).sum(-1, keepdim=True)
    small = (s2 < _SMALL_SIN2) & (cos > 0)
    near_pi = cos < -0.5
    # Moderate angles (and the series near zero).
    s2_safe = torch.where(small | near_pi, torch.ones_like(s2), s2)
    sin = s2_safe.sqrt()
    factor = torch.where(small, _horner(_ASIN_SERIES, s2), torch.atan2(sin, cos) / sin)
    generic = factor * v
    # Near pi.
    one_minus_cos = torch.where(near_pi, 1 - cos, torch.ones_like(cos))
    S = (0.5 * (R + R.mT) - cos[..., None] * eye) / one_minus_cos[..., None]
    diag = S.diagonal(dim1=-2, dim2=-1)
    k = diag.argmax(-1, keepdim=True)
    col = torch.take_along_dim(S, k[..., None, :], dim=-1)[..., 0]  # a * a_k
    a_k = torch.take_along_dim(diag, k, dim=-1).clamp(min=torch.finfo(R.dtype).tiny).sqrt()
    axis = col / a_k
    axis = torch.where((axis * v).sum(-1, keepdim=True) < 0, -axis, axis)
    sin_pi = torch.where(near_pi, s2, torch.ones_like(s2)).sqrt()
    theta_pi = torch.where(near_pi, torch.atan2(sin_pi, cos), torch.zeros_like(cos))
    return torch.where(near_pi, theta_pi * axis, generic)


def _se3_V(omega):
    _, B, C = _exp_coeffs(omega)
    K = so3_hat(omega)
    return torch.eye(3, dtype=omega.dtype, device=omega.device) + B * K + C * (K @ K)


def se3_log(T: torch.Tensor) -> torch.Tensor:
    """Twist ``(rho, omega)`` (..., 6) of the rigid transform ``T`` (..., 4, 4); inverts :func:`se3_exp`."""
    omega = so3_log(T[..., :3, :3])
    rho = torch.linalg.solve(_se3_V(omega), T[..., :3, 3:])[..., 0]
    return torch.cat([rho, omega], -1)


def _rigid_inv(T):
    Rt = T[:3, :3].T
    return _make_se3(Rt, -Rt @ T[:3, 3])


def _make_se3(R, t):
    top = torch.cat([R, t[..., None]], -1)
    bottom = torch.zeros(*R.shape[:-2], 1, 4, dtype=R.dtype, device=R.device)
    bottom[..., 0, 3] = 1
    return torch.cat([top, bottom], -2)


def se3_exp(xi: torch.Tensor) -> torch.Tensor:
    """SE(3) exponential: 4 x 4 rigid transform (..., 4, 4) of the twist ``xi = (rho, omega)`` (..., 6).

    The translation part is ``V(omega) @ rho`` (a screw motion), as in Sophus and PyPose.
    """
    return _make_se3(*_se3_exp_parts(xi, screw=True))


class SO3(Manifold):
    """Rotations stored as 3 x 3 matrices; 3 degrees of freedom.

    ``side="right"`` (default): ``plus(R, w) = R @ so3_exp(w)``, a rotation in the body
    frame. ``side="left"``: ``so3_exp(w) @ R``, a rotation in the world frame (the convention
    of Ceres's ``QuaternionManifold`` and PyPose).
    """

    ambient_shape = (3, 3)

    def __init__(self, side: str = "right"):
        if side not in ("right", "left"):
            raise ValueError(f"side must be 'right' or 'left', got {side!r}")
        self.side = side

    def tangent_size(self, x):
        return 3

    def plus(self, x, delta):
        R, dR = x.reshape(3, 3), so3_exp(delta.reshape(3))
        return (R @ dR if self.side == "right" else dR @ R).reshape(x.shape)

    def minus(self, y, x):
        Rx, Ry = x.reshape(3, 3), y.reshape(3, 3)
        return so3_log(Rx.T @ Ry if self.side == "right" else Ry @ Rx.T)


class SE3(Manifold):
    """Rigid transforms stored as 4 x 4 matrices ``[[R, t], [0, 1]]``; 6 degrees of freedom.

    The tangent vector is ``delta = (rho, omega)``: translation first, then a rotation
    vector, as in Sophus and PyPose. It builds an increment ``D = [[so3_exp(omega), u], [0, 1]]``.

    Args:
        side: ``"right"`` (default) composes ``T @ D``: the increment is expressed in the
            body frame (the frame ``T`` maps from). ``"left"`` composes ``D @ T``, in the
            world frame.
        retraction: ``"decoupled"`` (default) uses ``u = rho``, so rotation and translation
            update independently. ``"exp"`` uses the true SE(3) exponential,
            ``u = V(omega) @ rho`` (a screw motion). Both give the same Jacobian at
            ``delta = 0`` and the same Gauss-Newton step; they differ only at second order.
        pivot: optional point ``c`` (3,) that ``omega`` rotates about, in the frame of the
            increment (body frame for ``side="right"``). The increment becomes
            ``C @ D @ C^-1`` with ``C`` the translation by ``c``. For registration, put it
            at the centre of the object (the isocentre): rotating about a far-away origin
            couples rotation and translation and slows convergence.
    """

    ambient_shape = (4, 4)

    def __init__(self, side: str = "right", retraction: str = "decoupled", pivot=None):
        if side not in ("right", "left"):
            raise ValueError(f"side must be 'right' or 'left', got {side!r}")
        if retraction not in ("decoupled", "exp"):
            raise ValueError(f"retraction must be 'decoupled' or 'exp', got {retraction!r}")
        self.side, self.retraction = side, retraction
        self.pivot = None if pivot is None else torch.as_tensor(pivot).reshape(3)

    def tangent_size(self, x):
        return 6

    def plus(self, x, delta):
        T = x.reshape(4, 4)
        dR, dt = _se3_exp_parts(delta.reshape(6), screw=self.retraction == "exp")
        if self.pivot is not None:
            c = self.pivot.to(dtype=x.dtype, device=x.device)
            dt = dt + c - dR @ c
        R, t = T[:3, :3], T[:3, 3]
        if self.side == "right":
            R_new, t_new = R @ dR, R @ dt + t
        else:
            R_new, t_new = dR @ R, dR @ t + dt
        return torch.cat([torch.cat([R_new, t_new[:, None]], 1), T[3:]], 0).reshape(x.shape)

    def minus(self, y, x):
        Tx, Ty = x.reshape(4, 4), y.reshape(4, 4)
        inc = _rigid_inv(Tx) @ Ty if self.side == "right" else Ty @ _rigid_inv(Tx)
        dR, dt = inc[:3, :3], inc[:3, 3]
        omega = so3_log(dR)
        if self.pivot is not None:
            c = self.pivot.to(dtype=x.dtype, device=x.device)
            dt = dt - c + dR @ c
        if self.retraction == "exp":
            dt = torch.linalg.solve(_se3_V(omega), dt)
        return torch.cat([dt, omega])


class Product(Manifold):
    """Cartesian product of manifolds, stored as one flat vector (blocks concatenated).

    Every factor needs a fixed ``ambient_shape`` (give :class:`Euclidean` a shape). Use
    :meth:`join` to build ``x0`` and :meth:`split` inside the objective to unpack ``x``::

        M = piazzi.Product(piazzi.SE3(), piazzi.Euclidean(1))
        x0 = M.join(torch.eye(4), torch.ones(1))
        res = piazzi.least_squares(lambda x: residuals(*M.split(x)), x0, manifold=M)
    """

    def __init__(self, *manifolds: Manifold):
        if not manifolds:
            raise ValueError("Product needs at least one manifold")
        self.manifolds = manifolds
        self.sizes = [m.ambient_size for m in manifolds]
        self.ambient_shape = (sum(self.sizes),)

    def tangent_size(self, x):
        return sum(m.tangent_size(b) for m, b in zip(self.manifolds, self._blocks(x), strict=True))

    def _blocks(self, x):
        return x.reshape(-1).split(self.sizes)

    def split(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Unpack a flat point into its blocks, each in its factor's ``ambient_shape``."""
        return [b.reshape(m.ambient_shape) for m, b in zip(self.manifolds, self._blocks(x), strict=True)]

    def join(self, *blocks: torch.Tensor) -> torch.Tensor:
        """Pack one block per factor into a flat point."""
        if len(blocks) != len(self.manifolds):
            raise ValueError(f"expected {len(self.manifolds)} blocks, got {len(blocks)}")
        for m, b, n in zip(self.manifolds, blocks, self.sizes, strict=True):
            if b.numel() != n:
                raise ValueError(f"block for {type(m).__name__} has {b.numel()} elements, expected {n}")
        return torch.cat([b.reshape(-1) for b in blocks])

    def plus(self, x, delta):
        blocks = self._blocks(x)
        tangent = [m.tangent_size(b) for m, b in zip(self.manifolds, blocks, strict=True)]
        deltas = delta.reshape(-1).split(tangent)
        out = [m.plus(b, d) for m, b, d in zip(self.manifolds, blocks, deltas, strict=True)]
        return torch.cat(out).reshape(x.shape)

    def minus(self, y, x):
        pairs = zip(self.manifolds, self._blocks(y), self._blocks(x), strict=True)
        return torch.cat([m.minus(by, bx).reshape(-1) for m, by, bx in pairs])
