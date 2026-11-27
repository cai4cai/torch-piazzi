"""Finite-difference derivatives for objectives that autograd cannot see through."""

from __future__ import annotations

from collections.abc import Callable

import torch

__all__ = ["finite_difference"]

_SCHEMES = ("central", "forward")


class _Counter:
    """Holds the user's function, the step settings and the number of calls."""

    def __init__(self, fun: Callable, eps: float | None, scheme: str):
        self.fun, self.eps, self.scheme, self.n_evals = fun, eps, scheme, 0

    def step(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Return the displacement ``h * t`` and the step ``h``, with ``h`` relative to ``|x|``.

        For a coordinate direction ``e_i`` this is ``eps * max(|x_i|, 1) * e_i``, the per-coordinate
        step of a numerical Jacobian; for a general direction the scale is the ``|t|``-weighted mean
        of ``max(|x_i|, 1)``.
        """
        eps = self.eps
        if eps is None:  # the step that balances truncation and rounding error for the scheme
            eps = torch.finfo(x.dtype).eps ** (1 / 3 if self.scheme == "central" else 1 / 2)
        a = t.abs()
        scale = (a * x.abs().clamp(min=1)).sum() / a.sum().clamp(min=torch.finfo(x.dtype).tiny)
        return eps * scale * t, eps * scale

    def call(self, x: torch.Tensor) -> torch.Tensor:
        self.n_evals += 1
        with torch.no_grad():  # a copy, so that fun may modify its argument or reuse its output
            return self.fun(x.detach().clone()).detach().clone()


class _FiniteDifference(torch.autograd.Function):
    """``fun(x)`` with derivatives by finite differences, in forward mode, reverse mode and under ``vmap``."""

    @staticmethod
    def forward(state, x):
        return state.call(x)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.state, ctx.x, ctx.y = inputs[0], inputs[1].detach(), output.detach()

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
        d, h = state.step(x, t)
        up = _FiniteDifference.apply(state, x + d)
        if state.scheme == "forward":
            return (up - ctx.y) / h
        return (up - _FiniteDifference.apply(state, x - d)) / (2 * h)

    @staticmethod
    def backward(ctx, v):
        # Reverse mode needs J^T v: one directional difference per coordinate (n or 2n evaluations),
        # each contracted with v, so J is never stored.
        state, x = ctx.state, ctx.x
        flat = x.reshape(-1)
        g = []
        for i in range(flat.numel()):
            e = torch.zeros_like(flat)
            e[i] = 1
            d, h = state.step(x, e.reshape(x.shape))
            up = _FiniteDifference.apply(state, x + d)
            lo = ctx.y if state.scheme == "forward" else _FiniteDifference.apply(state, x - d)
            g.append(torch.sum(v * (up - lo)) / ((1 if state.scheme == "forward" else 2) * h))
        return None, torch.stack(g).reshape(x.shape) if g else torch.zeros_like(x)


class FiniteDifference:
    """The callable returned by :func:`finite_difference`; ``n_evals`` counts calls of the wrapped function."""

    def __init__(self, fun: Callable, eps: float | None, scheme: str):
        if scheme not in _SCHEMES:
            raise ValueError(f"scheme must be one of {_SCHEMES}, got {scheme!r}")
        if eps is not None and not eps > 0:
            raise ValueError(f"eps must be positive, got {eps!r}")
        self._state = _Counter(fun, eps, scheme)

    @property
    def n_evals(self) -> int:
        """Calls of the wrapped function so far, the finite-difference ones included."""
        return self._state.n_evals

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return _FiniteDifference.apply(self._state, x)


def finite_difference(fun: Callable, *, eps: float | None = None, scheme: str = "central") -> FiniteDifference:
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
    With a ``manifold``, the solvers differentiate ``fun(manifold.plus(x, delta))`` at ``delta = 0``;
    the differences are then taken along the manifold's tangent directions at the current point, so
    ``fun`` is evaluated at points that leave the manifold by ``O(eps^2)`` (a rotation matrix
    perturbed this way is orthogonal only to second order).

    Args:
        fun: the black-box function of a tensor, returning a tensor.
        eps: relative step. Default ``finfo.eps ** (1/3)`` for ``"central"`` (about ``6e-6`` in
            float64) and ``finfo.eps ** (1/2)`` for ``"forward"``, the steps that balance
            truncation and rounding error.
        scheme: ``"central"`` (default; error ``O(eps^2)``, ``2n`` calls) or ``"forward"``
            (error ``O(eps)``, ``n`` calls, reusing the value at ``x`` that the solver has
            just computed).
    """
    return FiniteDifference(fun, eps, scheme)
