"""PyTorch front end for ``scipy.optimize.minimize``."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np
import scipy.optimize
import torch

from .._common import OptimizeResult, check_x0, detach_tree, split_aux
from ._convert import bounds_to_scipy, iteration_callback, n2t, t2n
from ._least_squares import _bind

__all__ = ["minimize"]

# Methods that use no derivatives: SciPy warns if given a gradient, and computing one is waste.
GRADIENT_FREE = {"nelder-mead", "powell", "cobyla", "cobyqa"}


def minimize(
    fun: Callable,
    x0: torch.Tensor,
    method: str | Callable | None = None,
    *,
    jac: str | Callable | None = "autodiff",
    hess: str | Callable | Any = None,
    hessp: str | Callable | None = None,
    bounds=None,
    has_aux: bool = False,
    callback: Callable | None = None,
    options: dict[str, Any] | None = None,
    **kwargs,
) -> OptimizeResult:
    """Run :func:`scipy.optimize.minimize` on a scalar function written in PyTorch.

    It has the calling convention of :func:`piazzi.minimize` (``fun``, ``x0``, ``method``,
    ``has_aux``, ``callback`` and ``options``), so you can swap backends to compare them;
    ``method`` and the keys of ``options`` are SciPy's. ``fun`` runs in the dtype and on the
    device of ``x0``; SciPy itself works in float64 on the CPU, and the wrapper converts at
    each call. Use a float64 ``x0`` for tight tolerances: SciPy's line searches and finite
    differences struggle with float32 noise.

    Args:
        fun (Callable): ``fun(x) -> loss``, or ``(loss, aux)`` with ``has_aux=True``, where
            ``x`` has the shape of ``x0`` and ``loss`` is a scalar tensor.
        x0 (Tensor): starting point, of any shape.
        method (str or Callable, optional): any SciPy method (``"BFGS"``, ``"L-BFGS-B"``,
            ``"Newton-CG"``, ``"trust-exact"``, ...). Default: ``None``, which lets SciPy
            choose.

    Keyword Args:
        jac (str or Callable, optional): ``"autodiff"`` computes the value and gradient
            together with autograd, in one evaluation; a callable ``jac(x) -> grad`` in
            PyTorch; or SciPy's finite differences (``"2-point"``, ``"3-point"``, or
            ``None``; not ``"cs"``). Methods that use no gradient (Nelder-Mead, Powell,
            COBYLA, COBYQA) get none. Default: ``"autodiff"``
        hess (str or Callable, optional): ``"autodiff"`` for the exact Hessian from
            autograd; a callable ``hess(x) -> H`` in PyTorch returning a tensor of shape
            ``(n, n)``; or anything SciPy accepts (``"2-point"``, or a
            :class:`scipy.optimize.HessianUpdateStrategy` such as
            :class:`scipy.optimize.BFGS`). Only Newton-type and trust-region methods use it.
            Default: ``None``
        hessp (str or Callable, optional): ``"autodiff"`` for Hessian-vector products from
            autograd (double backward), or a callable ``hessp(x, p) -> H @ p`` in PyTorch.
            Default: ``None``
        bounds (scipy.optimize.Bounds or Tensor, optional): as in SciPy; a
            :class:`scipy.optimize.Bounds` holding tensors, or a tensor of shape ``(n, 2)``
            of ``(min, max)`` pairs, is converted. Default: ``None``
        has_aux (bool, optional): whether ``fun`` also returns an auxiliary output (any tree
            of tensors), returned detached in ``result.aux``. Default: ``False``
        callback (Callable, optional): called after every iteration with an
            :class:`piazzi.IterationInfo` (tensors, with the evaluation counts so far), as
            for the native solvers; returning ``True`` stops the run. ``aux`` is set when
            SciPy's last evaluation was at the iterate. TNC reports only ``x`` (``fun`` is
            then NaN unless known) and cannot be stopped. Default: ``None``
        options (dict, optional): SciPy's method-specific options. Default: ``None``
        args (tuple, optional): as in SciPy, extra arguments for ``fun`` and callable
            ``jac``, ``hess`` and ``hessp``. Default: ``()``
        **kwargs: other keyword arguments of :func:`scipy.optimize.minimize` (``tol``,
            ``constraints``, ...). Constraint functions receive NumPy arrays.

    Returns:
        OptimizeResult: ``x`` with the shape, dtype and device of ``x0``, ``fun`` the final
        value and ``aux`` its auxiliary output (with ``has_aux``). ``n_iter`` is SciPy's
        ``nit``. The counts are calls the wrapper made, not SciPy's ``nfev`` / ``njev`` /
        ``nhev``: ``n_fun_grad`` counts value-and-gradient evaluations (``jac="autodiff"``)
        or calls to a callable ``jac``, ``n_fun`` value-only ones (including
        finite-difference ones), and ``n_system`` Hessian or Hessian-vector product
        evaluations. ``converged`` is SciPy's ``success``, and ``raw`` is SciPy's own
        result. ``fun`` and ``aux`` come from the last evaluation at SciPy's final ``x``; if
        SciPy did not evaluate there last, the wrapper evaluates ``fun`` once more, which
        the counts do not include.

    See Also:
        - :func:`piazzi.minimize`: the native interface with the same calling convention.

    Example::

        >>> def rosenbrock(x):
        ...     return (100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2).sum()
        >>> res = piazzi.scipy.minimize(rosenbrock, torch.tensor([-1.2, 1.0], dtype=torch.float64), method="BFGS")
        >>> res.x
        tensor([1.0000, 1.0000], dtype=torch.float64)
    """
    like, shape = check_x0(x0)
    if jac == "cs" or hess == "cs":
        raise ValueError("complex-step derivatives ('cs') are not supported: use 'autodiff' or finite differences")
    # SciPy's extra arguments are bound to the PyTorch functions, which the adapters call without them.
    args = kwargs.pop("args", ())
    args = args if isinstance(args, tuple) else (args,)
    if args:
        fun = _bind(fun, args, {})
        jac = _bind(jac, args, {}) if callable(jac) else jac
        if callable(hess) and not isinstance(hess, scipy.optimize.HessianUpdateStrategy):
            hess = _bind(hess, args, {})
        hessp = _bind(hessp, args, {}) if callable(hessp) else hessp
    n = like.numel()
    last: dict[str, Any] = {}
    counts = {"fg": 0, "f": 0, "sys": 0}

    def to_torch(xn):
        return n2t(xn, like).reshape(shape)

    def record(xn, loss, aux):
        last.update(x=np.array(xn), fun=loss.detach(), aux=detach_tree(aux))
        last.setdefault("start", last["fun"])

    def value(xn, count=True):
        with torch.no_grad():
            loss, aux = split_aux(fun(to_torch(xn)), has_aux)
        record(xn, loss, aux)
        counts["f"] += count
        return loss.item()

    def value_and_grad(xn):
        xg = to_torch(xn).requires_grad_(True)
        with torch.enable_grad():
            loss, aux = split_aux(fun(xg), has_aux)
            (g,) = torch.autograd.grad(loss, xg)
        record(xn, loss, aux)
        counts["fg"] += 1
        return loss.item(), t2n(g).reshape(-1)

    def scalar(x):
        return split_aux(fun(x), has_aux)[0]

    # What SciPy gets for fun, jac, hess and hessp: callables, or values it understands.
    scipy_fun: Callable = value
    scipy_jac: Any = jac
    scipy_hess: Any = hess
    scipy_hessp: Any = hessp
    gradient_free = isinstance(method, str) and method.lower() in GRADIENT_FREE
    if gradient_free and jac == "autodiff":
        scipy_jac = None
    elif jac == "autodiff":
        scipy_fun, scipy_jac = value_and_grad, True
    elif callable(jac):
        user_jac: Callable[[torch.Tensor], torch.Tensor] = jac

        def callable_jac(xn):
            counts["fg"] += 1
            with torch.no_grad():
                return t2n(user_jac(to_torch(xn))).reshape(-1)

        scipy_jac = callable_jac

    if hess == "autodiff":

        def autodiff_hess(xn):
            counts["sys"] += 1
            return t2n(torch.autograd.functional.hessian(scalar, to_torch(xn))).reshape(n, n)

        scipy_hess = autodiff_hess
    elif callable(hess) and not isinstance(hess, scipy.optimize.HessianUpdateStrategy):
        user_hess = cast(Callable[[torch.Tensor], torch.Tensor], hess)

        def callable_hess(xn):
            counts["sys"] += 1
            with torch.no_grad():
                return t2n(user_hess(to_torch(xn))).reshape(n, n)

        scipy_hess = callable_hess

    if hessp == "autodiff":

        def autodiff_hessp(xn, pn):
            counts["sys"] += 1
            xg = to_torch(xn).requires_grad_(True)
            with torch.enable_grad():
                (g,) = torch.autograd.grad(scalar(xg), xg, create_graph=True)
                # An affine objective has a constant gradient with no graph: its Hessian is zero.
                hv = None
                if g.requires_grad:
                    (hv,) = torch.autograd.grad(g, xg, grad_outputs=to_torch(pn), allow_unused=True)
            if hv is None:
                hv = torch.zeros_like(xg)
            return t2n(hv).reshape(-1)

        scipy_hessp = autodiff_hessp
    elif callable(hessp):
        user_hessp: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = hessp

        def callable_hessp(xn, pn):
            counts["sys"] += 1
            with torch.no_grad():
                return t2n(user_hessp(to_torch(xn), to_torch(pn))).reshape(-1)

        scipy_hessp = callable_hessp

    res = scipy.optimize.minimize(
        scipy_fun,
        t2n(like),
        method=method,
        jac=scipy_jac,
        hess=scipy_hess,
        hessp=scipy_hessp,
        bounds=bounds_to_scipy(bounds),
        callback=iteration_callback(callback, like, shape, last, counts, lambda r: r.fun),
        options=options,
        **kwargs,
    )
    # SciPy's final x is usually the last point evaluated; otherwise evaluate it once more.
    if "x" not in last or not np.array_equal(last["x"], res.x):
        value(res.x, count=False)
    return OptimizeResult(
        x=to_torch(res.x),
        fun=last["fun"],
        aux=last["aux"],
        n_iter=getattr(res, "nit", 0) or 0,
        n_fun_grad=counts["fg"],
        n_fun=counts["f"],
        n_system=counts["sys"],
        converged=bool(res.success),
        message=str(res.message),
        raw=res,
    )
