"""Robust loss functions for least squares, following ``scipy.optimize.least_squares``.

A loss ``rho(z)`` is applied to the scaled squared residuals ``z = (r / C)^2``, where
``C = f_scale``, and the cost is ``F = 0.5 * sum(C^2 * rho(z))``. Built-in losses:

- ``"linear"``: ``rho(z) = z`` (ordinary least squares).
- ``"huber"``: ``rho(z) = z`` if ``z <= 1``, else ``2 * sqrt(z) - 1``.
- ``"soft_l1"``: ``rho(z) = 2 * (sqrt(1 + z) - 1)``.
- ``"cauchy"``: ``rho(z) = log(1 + z)``.
- ``"arctan"``: ``rho(z) = arctan(z)``.

A callable ``loss(z) -> rho(z)`` (elementwise, written in PyTorch) can be used too; its
first and second derivatives come from autograd.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

__all__ = ["LOSSES", "robust_cost"]

# Each built-in returns (rho, rho', rho'') evaluated at z. The derivatives are closed form,
# so evaluating the normal equations needs no autograd.


def _linear(z):
    return z, torch.ones_like(z), torch.zeros_like(z)


def _huber(z):
    inlier = z <= 1
    zc = z.clamp(min=1)  # keeps the outlier branch finite where it is not selected
    rho = torch.where(inlier, z, 2 * zc.sqrt() - 1)
    d1 = torch.where(inlier, torch.ones_like(z), zc.rsqrt())
    d2 = torch.where(inlier, torch.zeros_like(z), -0.5 * zc**-1.5)
    return rho, d1, d2


def _soft_l1(z):
    t = 1 + z
    return 2 * (t.sqrt() - 1), t.rsqrt(), -0.5 * t**-1.5


def _cauchy(z):
    t = 1 + z
    return torch.log1p(z), 1 / t, -1 / t**2


def _arctan(z):
    t = 1 + z**2
    return torch.arctan(z), 1 / t, -2 * z / t**2


LOSSES: dict[str, Callable] = {
    "linear": _linear,
    "huber": _huber,
    "soft_l1": _soft_l1,
    "cauchy": _cauchy,
    "arctan": _arctan,
}

Loss = str | Callable[[torch.Tensor], torch.Tensor]


def check_loss(loss: Loss, f_scale: float) -> None:
    if not callable(loss) and loss not in LOSSES:
        raise ValueError(f"loss must be one of {sorted(LOSSES)} or a callable, got {loss!r}")
    if not f_scale > 0:
        raise ValueError(f"f_scale must be positive, got {f_scale!r}")


def rho_and_derivatives(z: torch.Tensor, loss: Loss) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(rho(z), rho'(z), rho''(z))``; for a callable loss the derivatives come from autograd."""
    if not callable(loss):
        return LOSSES[loss](z)
    zg = z.detach().requires_grad_(True)
    with torch.enable_grad():
        rho = loss(zg)
        (d1,) = torch.autograd.grad(rho.sum(), zg, create_graph=True)
        if d1.requires_grad:
            (d2,) = torch.autograd.grad(d1.sum(), zg, allow_unused=True)
        else:
            d2 = None
    d2 = torch.zeros_like(z) if d2 is None else d2
    return rho.detach(), d1.detach(), d2.detach()


def robust_cost(r: torch.Tensor, loss: Loss = "linear", f_scale: float = 1.0) -> torch.Tensor:
    """``0.5 * sum(f_scale^2 * rho((r / f_scale)^2))``, differentiable with autograd.

    This is the cost that :func:`piazzi.least_squares` minimises. Use it with
    :func:`piazzi.lbfgs_armijo` to minimise the same cost without a Jacobian.
    """
    check_loss(loss, f_scale)
    z = (r / f_scale) ** 2
    rho = loss(z) if callable(loss) else LOSSES[loss](z)[0]
    return 0.5 * f_scale**2 * rho.sum()
