"""PyTorch front end for ``scipy.optimize.least_squares``."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import scipy.optimize
import torch

from .._common import OptimizeResult, check_x0, quiet_jvp_warnings
from .._losses import Loss, check_loss, rho_and_derivatives
from ._convert import n2t, t2n

__all__ = ["least_squares"]


def _to_scipy(v):
    """Convert a tensor argument (bounds, x_scale, diff_step) to a flat float64 array."""
    return t2n(v).reshape(-1) if isinstance(v, torch.Tensor) else v


def least_squares(
    fun: Callable,
    x0: torch.Tensor,
    *,
    jac: str | Callable = "autodiff",
    loss: Loss = "linear",
    f_scale: float = 1.0,
    bounds=(-np.inf, np.inf),
    **kwargs,
) -> OptimizeResult:
    """Call ``scipy.optimize.least_squares`` on a residual function written in PyTorch.

    ``fun(x)`` takes a tensor shaped like ``x0`` and returns the residuals. It runs in the
    dtype and on the device of ``x0``; SciPy itself works in float64 on the CPU, and the
    wrapper converts at each call. Unlike SciPy, the default Jacobian comes from
    forward-mode autodiff, as in :func:`piazzi.least_squares`, so the two run on the same
    derivatives.

    Args:
        jac: ``"autodiff"`` (default) for ``torch.func.jacfwd``; a callable
            ``jac(x) -> J`` that returns a tensor of shape (m, n); or one of SciPy's
            finite-difference schemes (``"2-point"``, ``"3-point"``). Finite differences
            need a float64 ``x0``: in float32 their steps are lost to rounding.
        loss: a SciPy loss name, or a callable ``rho(z)`` written in PyTorch as for
            :func:`piazzi.least_squares`. The wrapper gives SciPy its derivatives from
            autograd.
        f_scale: soft margin between inlier and outlier residuals.
        bounds: as in SciPy; tensors in a ``(lb, ub)`` pair are converted (and flattened).
        **kwargs: passed to ``scipy.optimize.least_squares`` (``method``, ``ftol``,
            ``xtol``, ``gtol``, ``max_nfev``, ``x_scale``, ``callback``, ...). Tensors given
            as ``x_scale`` or ``diff_step`` are converted. A ``callback`` receives SciPy's
            intermediate result, with NumPy arrays.

    Returns:
        An :class:`piazzi.OptimizeResult`. ``x`` has the shape, dtype and device of ``x0``,
        and ``fun`` is the cost. SciPy does not report iterations, so ``n_iter`` is its
        Jacobian evaluation count (``njev``), or ``nfev`` when ``njev`` is not reported.
        ``n_fun`` is ``nfev`` and ``n_system`` is ``njev``. ``converged`` is SciPy's
        ``success``. ``raw`` is SciPy's own result.
    """
    check_loss(loss, f_scale)
    like, shape = check_x0(x0)

    def residuals(xn):
        with torch.no_grad():
            r = fun(n2t(xn, like).reshape(shape))
        return t2n(r.reshape(-1))

    if jac == "autodiff":

        def jacobian(xn):
            with quiet_jvp_warnings():
                J = torch.func.jacfwd(lambda xf: fun(xf.reshape(shape)).reshape(-1))(n2t(xn, like))
            return t2n(J)

    elif callable(jac):

        def jacobian(xn):
            with torch.no_grad():
                J = jac(n2t(xn, like).reshape(shape))
            return t2n(J.reshape(-1, like.numel()))

    else:
        jacobian = jac

    if callable(loss):

        def scipy_loss(z):
            return np.stack([t2n(v) for v in rho_and_derivatives(torch.from_numpy(z), loss)])

    else:
        scipy_loss = loss

    for key in ("x_scale", "diff_step"):
        if key in kwargs:
            kwargs[key] = _to_scipy(kwargs[key])

    res = scipy.optimize.least_squares(
        residuals,
        t2n(like),
        jac=jacobian,
        loss=scipy_loss,
        f_scale=f_scale,
        bounds=tuple(_to_scipy(b) for b in bounds) if isinstance(bounds, (tuple, list)) else bounds,
        **kwargs,
    )
    njev = res.njev if res.njev is not None else 0
    return OptimizeResult(
        x=n2t(res.x, like).reshape(shape),
        fun=n2t(res.cost, like),
        aux=None,
        n_iter=res.njev if res.njev is not None else res.nfev,
        n_fun=res.nfev,
        n_system=njev,
        converged=bool(res.success),
        message=res.message,
        raw=res,
    )
