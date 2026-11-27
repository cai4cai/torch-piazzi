"""Finite-difference derivatives for objectives that autograd cannot see through."""

from __future__ import annotations

from collections.abc import Callable

import torch

from ._common import quiet_jvp_warnings, transfer
from ._manifolds import Manifold

__all__ = ["finite_difference"]

_SCHEMES = ("central", "forward")


class _Counter:
    """Holds the user's function, the step settings and the number of calls."""

    def __init__(self, fun: Callable, eps: float | None, scheme: str, manifold: Manifold | None):
        self.fun, self.eps, self.scheme, self.manifold, self.n_evals = fun, eps, scheme, manifold, 0

    def default_eps(self, dtype: torch.dtype) -> float:
        if self.eps is not None:
            return self.eps
        # the step that balances truncation and rounding error for the scheme
        return torch.finfo(dtype).eps ** (1 / 3 if self.scheme == "central" else 1 / 2)

    def tangent_pinv(self, x: torch.Tensor) -> torch.Tensor:
        """Pseudo-inverse of ``P = d plus(x, delta) / d delta`` at ``delta = 0``, shape (tangent, numel).

        ``P`` maps tangent coordinates to the ambient tangent vectors that autograd passes to
        ``jvp``, so ``pinv(P) @ t`` recovers the coordinates ``delta`` of a tangent direction ``t``
        (exactly: ``P`` has full column rank). It costs no evaluation of ``fun``.
        """
        m = self.manifold
        zero = x.new_zeros(m.tangent_size(x))
        if zero.numel() == 0:  # every parameter fixed: no directions (jacfwd cannot differentiate in nothing)
            return x.new_zeros(0, x.numel())
        with torch.enable_grad(), quiet_jvp_warnings():
            P = torch.func.jacfwd(lambda d: m.plus(x, d).reshape(-1))(zero)
        # an SVD of a small matrix: in float64 on the CPU, as for the normal equations (MPS has no float64)
        return transfer(torch.linalg.pinv(transfer(P, "cpu", torch.float64)), x.device, x.dtype)

    def displaced(self, x: torch.Tensor, t: torch.Tensor | None = None, delta: torch.Tensor | None = None):
        """Return the points ``x +- h d`` and the step ``h``.

        Without a manifold, ``d`` is the ambient direction ``t`` (see :meth:`step`). With one,
        ``d`` is the tangent direction with coordinates ``delta``, and the points are
        ``manifold.plus(x, +-h * delta)``, so they lie on the manifold.
        """
        if self.manifold is None:
            d, h = self.step(x, t)
            return x + d, x - d, h
        h = self.default_eps(x.dtype)  # tangent coordinates are already relative to the point
        return self.manifold.plus(x, h * delta), self.manifold.plus(x, -h * delta), h

    def step(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the displacement ``h * t`` and the step ``h``, with ``h`` relative to ``|x|``.

        For a coordinate direction ``e_i`` this is ``eps * max(|x_i|, 1) * e_i``, the per-coordinate
        step of a numerical Jacobian; for a general direction the scale is the ``|t|``-weighted mean
        of ``max(|x_i|, 1)``.
        """
        eps = self.default_eps(x.dtype)
        a = t.abs()
        total = a.sum()
        # a zero direction still gets a positive step (the difference is then exactly zero, not 0 / 0)
        scale = torch.where(
            total > 0, (a * x.abs().clamp(min=1)).sum() / total.clamp(min=torch.finfo(x.dtype).tiny), 1.0
        )
        return eps * scale * t, eps * scale

    def call(self, x: torch.Tensor) -> torch.Tensor:
        self.n_evals += 1
        with torch.no_grad():  # fun gets a copy, so that it may modify its argument
            out = self.fun(x.detach().clone())
        if not isinstance(out, torch.Tensor):
            raise TypeError(
                f"fun must return a tensor, got {type(out).__name__} (finite_difference does not support has_aux)"
            )
        return out.detach().clone()  # a copy, so that fun may reuse its output buffer


class _FiniteDifference(torch.autograd.Function):
    """``fun(x)`` with derivatives by finite differences, in forward mode, reverse mode and under ``vmap``."""

    @staticmethod
    def forward(state, x):
        return state.call(x)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.state, ctx.x, ctx.y = inputs[0], inputs[1].detach(), output.detach()
        ctx.pinv = None  # computed on first use: the manifold's Jacobian is not needed for the value

    @staticmethod
    def vmap(info, in_dims, state, x):
        # The user's function is a black box that cannot be batched: call it once per batch entry.
        if in_dims[1] is None:
            return _FiniteDifference.apply(state, x), None
        return torch.stack([_FiniteDifference.apply(state, xi) for xi in x.movedim(in_dims[1], 0)]), 0

    @staticmethod
    def jvp(ctx, _, t):
        # Called by jacfwd once per tangent direction (under vmap), so J costs n or 2n evaluations.
        state, x = ctx.state, ctx.x
        pinv = _pinv(ctx)
        x_up, x_lo, h = state.displaced(x, t) if pinv is None else state.displaced(x, delta=pinv @ t.reshape(-1))
        up = _FiniteDifference.apply(state, x_up)
        if state.scheme == "forward":
            return (up - ctx.y) / h
        return (up - _FiniteDifference.apply(state, x_lo)) / (2 * h)

    @staticmethod
    def backward(ctx, v):
        # Reverse mode needs J^T v: one directional difference per coordinate (n or 2n evaluations),
        # each contracted with v, so J is never stored. With a manifold the coordinates are tangent
        # ones, giving g = P^T J_amb^T v; the ambient gradient handed back is pinv(P)^T g, which the
        # chain rule through ``plus`` maps back to g (P^T pinv(P)^T = I).
        state, x = ctx.state, ctx.x
        pinv = _pinv(ctx)
        n = x.numel() if pinv is None else pinv.shape[0]
        g = []
        for i in range(n):
            e = torch.zeros(n, dtype=x.dtype, device=x.device)
            e[i] = 1
            x_up, x_lo, h = state.displaced(x, e.reshape(x.shape)) if pinv is None else state.displaced(x, delta=e)
            up = _FiniteDifference.apply(state, x_up)
            lo = ctx.y if state.scheme == "forward" else _FiniteDifference.apply(state, x_lo)
            g.append(torch.sum(v * (up - lo)) / ((1 if state.scheme == "forward" else 2) * h))
        if not g:
            return None, torch.zeros_like(x)
        g = torch.stack(g)
        return None, (g if pinv is None else pinv.T @ g).reshape(x.shape)


def _pinv(ctx):
    if ctx.state.manifold is None:
        return None
    if ctx.pinv is None:
        ctx.pinv = ctx.state.tangent_pinv(ctx.x)
    return ctx.pinv


class FiniteDifference:
    """The callable returned by :func:`finite_difference`; ``n_evals`` counts calls of the wrapped function."""

    def __init__(self, fun: Callable, eps: float | None, scheme: str, manifold: Manifold | None):
        if scheme not in _SCHEMES:
            raise ValueError(f"scheme must be one of {_SCHEMES}, got {scheme!r}")
        if eps is not None and not eps > 0:
            raise ValueError(f"eps must be positive, got {eps!r}")
        self._state = _Counter(fun, eps, scheme, manifold)

    @property
    def n_evals(self) -> int:
        """Calls of the wrapped function so far, the finite-difference ones included."""
        return self._state.n_evals

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return _FiniteDifference.apply(self._state, x)


def finite_difference(
    fun: Callable, *, eps: float | None = None, scheme: str = "central", manifold: Manifold | None = None
) -> FiniteDifference:
    """Make a black-box function look differentiable, so that any solver can use it.

    Use this when ``fun(x) -> tensor`` cannot be differentiated by autograd: a non-differentiable
    renderer or simulator, a library working on NumPy arrays, a measurement. The returned
    function gives the same values as ``fun`` and has finite-difference derivatives, in forward
    mode (``torch.func.jacfwd``, which :func:`piazzi.least_squares` uses) and in reverse mode
    (``backward``, which :func:`piazzi.minimize` uses), so solvers, ``manifold=`` and robust
    losses work unchanged::

        fd_fun = piazzi.finite_difference(simulate)
        result = piazzi.least_squares(lambda x: fd_fun(x) - target, x0)

    Every Jacobian costs ``n`` (``"forward"``) or ``2n`` (``"central"``) extra calls of ``fun``
    for ``n`` parameters, which is why this is a last resort: prefer autograd where it works. The
    solver's ``n_fun_grad`` counts only its own evaluations, so read the total from the
    ``n_evals`` attribute of the returned function, which counts every call of ``fun``.

    ``fun`` is called without gradient tracking, on a copy of its argument, and its result is
    copied. It must return a tensor and be deterministic: noise larger than the step ruins the
    differences. It runs on the device and in the dtype of ``x``, and so do the differences;
    prefer float64, since in float32 the default step is large (about ``5e-3``) and a central
    difference is accurate to roughly 1e-3 relative error at best.

    The numerics follow ``torch.autograd.gradcheck``'s numerical Jacobian (central differences,
    one coordinate at a time), with the step relative to the coordinate: ``eps * max(|x_i|, 1)``.
    With ``manifold=M``, pass the same manifold to the solver (``least_squares(..., manifold=M)``).
    The solver differentiates ``fun(M.plus(x, delta))`` at ``delta = 0``; the wrapper then recovers
    the tangent coordinates ``delta`` of each direction and evaluates ``fun(M.plus(x, +-h * delta))``,
    so ``fun`` only ever sees points on the manifold (a valid rotation matrix, say) and the step
    ``h`` is in tangent coordinates (``eps``, not scaled by ``|x|``). Without ``manifold=``, the
    differences are taken in ambient space along the tangent directions, so ``fun`` is evaluated
    at points that leave the manifold by ``O(eps^2)``.

    Args:
        fun: the black-box function of a tensor, returning a tensor.
        eps: relative step. Default ``finfo.eps ** (1/3)`` for ``"central"`` (about ``6e-6`` in
            float64) and ``finfo.eps ** (1/2)`` for ``"forward"``, the steps that balance
            truncation and rounding error.
        scheme: ``"central"`` (default; error ``O(eps^2)``, ``2n`` calls) or ``"forward"``
            (error ``O(eps)``, ``n`` calls, reusing the value at ``x`` that the solver has
            just computed).
        manifold: the :class:`piazzi.Manifold` that the solver is given, if any, as above.
    """
    return FiniteDifference(fun, eps, scheme, manifold)
