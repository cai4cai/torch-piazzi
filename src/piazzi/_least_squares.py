"""Nonlinear least squares, like ``scipy.optimize.least_squares``, on top of :func:`gauss_newton`."""

from __future__ import annotations

from collections.abc import Callable

import torch

from ._common import OptimizeResult, quiet_jvp_warnings
from ._gauss_newton import gauss_newton
from ._losses import Loss, check_loss, rho_and_derivatives, robust_cost

__all__ = ["least_squares", "least_squares_system"]


def least_squares_system(
    residual_fun: Callable,
    has_aux: bool = False,
    *,
    loss: Loss = "linear",
    f_scale: float = 1.0,
    curvature: str = "irls",
) -> Callable:
    """Turn ``residual_fun(x) -> r`` into a normal-equations function for :func:`gauss_newton`.

    The returned ``system(x)`` gives ``(loss, g, H)`` (plus ``aux`` with ``has_aux``) for the
    cost ``F = 0.5 * sum(C^2 * rho(z))`` with ``z = (r / C)^2`` and ``C = f_scale``, as in
    ``scipy.optimize.least_squares``. With the default ``loss="linear"`` this is
    ``F = 0.5 * ||r||^2``, ``g = J^T r`` and ``H = J^T J``. The Jacobian ``J`` comes from
    forward-mode autodiff (cheap when there are few parameters). For large or structured
    problems, write ``system`` by hand and reduce ``J^T J`` without materialising ``J``.

    Args:
        loss: a name from :data:`piazzi.LOSSES` (``"linear"``, ``"huber"``, ``"soft_l1"``,
            ``"cauchy"``, ``"arctan"``) or a callable ``rho(z)`` written in PyTorch, applied
            elementwise; its derivatives come from autograd.
        f_scale: soft margin ``C`` between inlier and outlier residuals.
        curvature: curvature model ``H = J^T diag(w) J`` for a robust loss; the gradient
            is always ``g = J^T (rho'(z) * r)``, so the choice changes the path, not the
            minimiser.

            - ``"irls"`` (default): ``w = rho'(z)``, as in iteratively reweighted least
              squares. Ceres Solver uses the same weights for losses with
              ``rho'' <= 0``, which includes every built-in loss.
            - ``"scipy"``: ``w = max(rho'(z) + 2 * rho''(z) * z, eps)``, the model of
              ``scipy.optimize.least_squares``. It is more accurate for inliers, but for
              outliers the bracket is negative and floored at ``eps``. When most
              residuals start as outliers, ``H`` is then nearly singular and
              Levenberg-Marquardt can stall; ``method="gn"`` copes better.
    """
    check_loss(loss, f_scale)
    if curvature not in ("irls", "scipy"):
        raise ValueError(f"curvature must be 'irls' or 'scipy', got {curvature!r}")
    robust = loss != "linear"

    def system(x):
        shape = x.shape

        def flat(xf):
            out = residual_fun(xf.reshape(shape))
            if has_aux:
                r, aux = out
                return r.reshape(-1), (r.reshape(-1), aux)
            return out.reshape(-1), (out.reshape(-1),)

        with quiet_jvp_warnings():
            J, extra = torch.func.jacfwd(flat, has_aux=True)(x.reshape(-1))
        r, aux = extra[0], (extra[1] if has_aux else None)
        if robust:
            z = (r / f_scale) ** 2
            rho, d1, d2 = rho_and_derivatives(z, loss)
            loss_val = 0.5 * f_scale**2 * rho.sum()
            w = d1 if curvature == "irls" else (d1 + 2 * d2 * z).clamp(min=torch.finfo(r.dtype).eps)
            g, H = J.T @ (d1 * r), J.T @ (w[:, None] * J)
        else:
            loss_val = 0.5 * torch.dot(r, r)
            g, H = J.T @ r, J.T @ J
        return (loss_val, g, H, aux) if has_aux else (loss_val, g, H)

    return system


def least_squares(
    fun: Callable,
    x0: torch.Tensor,
    *,
    loss: Loss = "linear",
    f_scale: float = 1.0,
    method: str = "lm",
    has_aux: bool = False,
    curvature: str = "irls",
    **kwargs,
) -> OptimizeResult:
    """Minimise ``0.5 * sum(f_scale^2 * rho((fun(x) / f_scale)^2))``, like ``scipy.optimize.least_squares``.

    ``fun(x)`` returns the residuals (``(r, aux)`` with ``has_aux``). This builds the normal
    equations with :func:`least_squares_system` and solves them with :func:`gauss_newton`
    (``method="lm"`` or ``"gn"``), using the residuals alone, without the Jacobian, for
    value-only evaluations. ``loss``, ``f_scale`` and ``curvature`` are as in
    :func:`least_squares_system`, and other keyword arguments go to :func:`gauss_newton`.
    ``result.fun`` is the cost, not the residuals.
    """
    system = least_squares_system(fun, has_aux, loss=loss, f_scale=f_scale, curvature=curvature)

    def value_fun(x):
        out = fun(x)
        return robust_cost((out[0] if has_aux else out).reshape(-1), loss, f_scale)

    kwargs.setdefault("value_fun", value_fun)
    return gauss_newton(system, x0, method=method, has_aux=has_aux, **kwargs)
