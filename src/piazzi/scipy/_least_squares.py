"""PyTorch front end for ``scipy.optimize.least_squares``."""

from __future__ import annotations

import inspect
from collections.abc import Callable

import numpy as np
import scipy.optimize
import torch

from .._common import OptimizeResult, check_x0, quiet_jvp_warnings
from .._losses import Loss, check_loss, rho_and_derivatives
from ._convert import bounds_to_scipy, iteration_callback, n2t, t2n, to_scipy_array

__all__ = ["least_squares"]


def _bind(f, args, kwargs):
    """``f`` with SciPy-style extra arguments appended: ``f(*a, *args, **kwargs)``."""

    def bound(*a):
        return f(*a, *args, **kwargs)

    return bound


def least_squares(
    fun: Callable,
    x0: torch.Tensor,
    *,
    jac: str | Callable = "autodiff",
    loss: Loss = "linear",
    f_scale: float = 1.0,
    bounds=(-np.inf, np.inf),
    callback: Callable | None = None,
    **kwargs,
) -> OptimizeResult:
    """Run :func:`scipy.optimize.least_squares` on a residual function written in PyTorch.

    ``fun`` runs in the dtype and on the device of ``x0``; SciPy itself works in float64 on
    the CPU, and the wrapper converts at each call. Unlike SciPy, the default Jacobian comes
    from forward-mode autodiff, as in :func:`piazzi.least_squares`, so the two run on the
    same derivatives.

    Args:
        fun (Callable): ``fun(x) -> r``, where ``x`` has the shape of ``x0`` and ``r`` is a
            tensor of residuals of any shape.
        x0 (Tensor): starting point, of any shape.

    Keyword Args:
        jac (str or Callable, optional): ``"autodiff"`` for :func:`torch.func.jacfwd`; a
            callable ``jac(x) -> J`` returning a tensor of shape ``(m, n)``; or one of
            SciPy's finite-difference schemes, ``"2-point"`` or ``"3-point"`` (not
            ``"cs"``). Finite differences need a float64 ``x0``: in float32 their steps are
            lost to rounding. Default: ``"autodiff"``
        loss (str or Callable, optional): a SciPy loss name, or a callable ``rho(z)``
            written in PyTorch as for :func:`piazzi.least_squares`; the wrapper gives SciPy
            its derivatives from autograd. Default: ``"linear"``
        f_scale (float, optional): soft margin between inlier and outlier residuals, in the
            units of ``r``. Default: ``1.0``
        bounds (tuple or scipy.optimize.Bounds, optional): as in SciPy, a ``(lb, ub)`` pair
            or a :class:`scipy.optimize.Bounds`. Tensors and arrays in it are converted and
            flattened to match the flat ``x0``; 0-dim ones are scalars.
            Default: ``(-inf, inf)``
        callback (Callable, optional): called after every iteration with an
            :class:`piazzi.IterationInfo`, as for the native solvers (``fun`` is the cost;
            ``aux`` is ``None``); returning ``True`` stops the run. Needs SciPy 1.16 or
            later. Default: ``None``
        args (tuple, optional): as in SciPy, extra positional arguments for ``fun`` and a
            callable ``jac``.
        kwargs (dict, optional): as in SciPy, extra keyword arguments for ``fun`` and a
            callable ``jac``.
        **kwargs: other keyword arguments of :func:`scipy.optimize.least_squares`
            (``method``, ``ftol``, ``xtol``, ``gtol``, ``max_nfev``, ``x_scale``, ...).
            Tensors given as ``x_scale`` or ``diff_step`` are converted like ``bounds``.

    Returns:
        OptimizeResult: ``x`` with the shape, dtype and device of ``x0``, and ``fun`` the
        cost. SciPy does not report iterations, so ``n_iter`` is its Jacobian evaluation
        count (``njev``), or ``nfev`` when ``njev`` is not reported. The counts are calls the
        wrapper made: ``n_fun`` residual evaluations (including those of finite
        differences, which SciPy's ``nfev`` leaves out for ``"trf"`` and ``"dogbox"``) and
        ``n_system`` Jacobian evaluations (``jac="autodiff"`` or a callable).
        ``converged`` is SciPy's ``success``, and ``raw`` is SciPy's own result.

    See Also:
        - :func:`piazzi.least_squares`: the native solver with the same calling convention.

    Example::

        >>> t = torch.linspace(0, 4, 40, dtype=torch.float64)
        >>> y = 2.5 * torch.exp(-1.3 * t)
        >>> x0 = torch.tensor([1.0, 0.5], dtype=torch.float64)
        >>> res = piazzi.scipy.least_squares(lambda p: p[0] * torch.exp(-p[1] * t) - y, x0)
        >>> res.x
        tensor([2.5000, 1.3000], dtype=torch.float64)
    """
    check_loss(loss, f_scale)
    like, shape = check_x0(x0)
    if jac == "cs":
        raise ValueError("jac='cs' (complex step) is not supported: use 'autodiff' or a finite-difference scheme")
    # SciPy's extra arguments are bound to the PyTorch functions, which the adapters call with x only.
    args, fun_kwargs = tuple(kwargs.pop("args", ())), dict(kwargs.pop("kwargs", None) or {})
    if args or fun_kwargs:
        fun = _bind(fun, args, fun_kwargs)
        if callable(jac):
            jac = _bind(jac, args, fun_kwargs)

    counts = {"fg": 0, "f": 0, "sys": 0}

    def residuals(xn):
        counts["f"] += 1
        with torch.no_grad():
            r = fun(n2t(xn, like).reshape(shape))
        return t2n(r.reshape(-1))

    if jac == "autodiff":

        def jacobian(xn):
            counts["sys"] += 1
            with quiet_jvp_warnings():
                J = torch.func.jacfwd(lambda xf: fun(xf.reshape(shape)).reshape(-1))(n2t(xn, like))
            return t2n(J)

    elif callable(jac):

        def jacobian(xn):
            counts["sys"] += 1
            with torch.no_grad():
                J = jac(n2t(xn, like).reshape(shape))
            return t2n(J.reshape(-1, like.numel()))

    else:
        jacobian = jac

    if callable(loss):

        def scipy_loss(z):
            # Evaluated in the dtype and on the device of x0, as with piazzi.least_squares.
            return np.stack([t2n(v) for v in rho_and_derivatives(n2t(z, like), loss)])

    else:
        scipy_loss = loss

    for key in ("x_scale", "diff_step"):
        if key in kwargs:
            kwargs[key] = to_scipy_array(kwargs[key])
    if callback is not None:
        if "callback" not in inspect.signature(scipy.optimize.least_squares).parameters:
            raise TypeError("callback needs SciPy 1.16 or later for least_squares")
        kwargs["callback"] = iteration_callback(callback, like, shape, {}, counts, lambda r: r.cost)

    res = scipy.optimize.least_squares(
        residuals,
        t2n(like),
        jac=jacobian,
        loss=scipy_loss,
        f_scale=f_scale,
        bounds=bounds_to_scipy(tuple(bounds) if isinstance(bounds, list) else bounds),
        **kwargs,
    )
    return OptimizeResult(
        x=n2t(res.x, like).reshape(shape),
        fun=n2t(res.cost, like),
        aux=None,
        n_iter=res.njev if res.njev is not None else res.nfev,
        n_fun=counts["f"],
        n_system=counts["sys"],
        converged=bool(res.success),
        message=res.message,
        raw=res,
    )
