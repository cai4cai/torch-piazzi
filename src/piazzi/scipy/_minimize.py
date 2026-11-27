"""PyTorch front end for ``scipy.optimize.minimize``."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

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
    """Call ``scipy.optimize.minimize`` on a scalar function written in PyTorch.

    It has the calling convention of :func:`piazzi.minimize` (``fun``, ``x0``, ``method``,
    ``has_aux``, ``callback`` and ``options``), so you can swap backends to compare them;
    ``method`` and the keys of ``options`` are SciPy's. ``fun(x)`` takes a tensor shaped
    like ``x0`` and returns a scalar tensor (``(loss, aux)`` with ``has_aux``). It runs in
    the dtype and on the device of ``x0``; SciPy itself works in float64 on the CPU, and
    the wrapper converts at each call. Use a float64 ``x0`` for tight tolerances: SciPy's
    line searches and finite differences struggle with float32 noise.

    Args:
        method: any SciPy method (``"BFGS"``, ``"L-BFGS-B"``, ``"Newton-CG"``,
            ``"trust-exact"``, ...). The default ``None`` lets SciPy choose.
        jac: ``"autodiff"`` (default) computes the value and gradient together with
            autograd, in one evaluation; a callable ``jac(x) -> grad`` in PyTorch; or
            SciPy's finite differences (``"2-point"``, ``"3-point"``, or ``None``; not ``"cs"``).
            Methods that use no gradient (Nelder-Mead, Powell, COBYLA, COBYQA) get none.
        hess: ``"autodiff"`` for the exact Hessian from autograd; a callable
            ``hess(x) -> H`` in PyTorch returning (n, n); or anything SciPy accepts
            (``"2-point"``, a ``HessianUpdateStrategy`` such as ``scipy.optimize.BFGS()``).
            Only Newton-type and trust-region methods use it.
        hessp: ``"autodiff"`` for Hessian-vector products from autograd (double backward),
            or a callable ``hessp(x, p) -> H @ p`` in PyTorch.
        bounds: as in SciPy; a ``Bounds`` holding tensors, or an (n, 2) tensor of
            ``(min, max)`` pairs, is converted.
        callback: called after every iteration with an :class:`piazzi.IterationInfo`
            (tensors, with the evaluation counts so far), as for the native solvers;
            returning True stops the run. ``aux`` is set when SciPy's last evaluation was
            at the iterate. TNC reports only ``x`` (``fun`` is then NaN unless known) and
            cannot be stopped.
        options: SciPy's method-specific options.
        args: as in SciPy, extra arguments for ``fun`` and callable ``jac``, ``hess`` and
            ``hessp``.
        **kwargs: passed to ``scipy.optimize.minimize`` (``tol``, ``constraints``, ...).
            Constraint functions receive NumPy arrays.

    Returns:
        An :class:`piazzi.OptimizeResult`. ``x`` has the shape, dtype and device of ``x0``,
        ``fun`` is the final value and ``aux`` its auxiliary output (with ``has_aux``).
        ``n_iter`` is SciPy's ``nit``. The counts are calls the wrapper made, not SciPy's
        ``nfev`` / ``njev`` / ``nhev``: ``n_fun_grad`` counts value-and-gradient evaluations
        (``jac="autodiff"``) or calls to a callable ``jac``, ``n_fun`` value-only ones
        (including finite-difference ones), and ``n_system`` Hessian or Hessian-vector
        product evaluations. ``converged`` is SciPy's ``success``. ``raw`` is SciPy's own
        result. ``fun`` and ``aux`` come from the last evaluation at SciPy's final ``x``; if
        SciPy did not evaluate there last, the wrapper evaluates ``fun`` once more, which
        the counts do not include.
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

    gradient_free = isinstance(method, str) and method.lower() in GRADIENT_FREE
    if gradient_free and jac == "autodiff":
        scipy_fun, scipy_jac = value, None
    elif jac == "autodiff":
        scipy_fun, scipy_jac = value_and_grad, True
    elif callable(jac):
        scipy_fun = value

        def scipy_jac(xn):
            counts["fg"] += 1
            with torch.no_grad():
                return t2n(jac(to_torch(xn))).reshape(-1)

    else:
        scipy_fun, scipy_jac = value, jac

    if hess == "autodiff":

        def scipy_hess(xn):
            counts["sys"] += 1
            return t2n(torch.autograd.functional.hessian(scalar, to_torch(xn))).reshape(n, n)

    elif callable(hess) and not isinstance(hess, scipy.optimize.HessianUpdateStrategy):

        def scipy_hess(xn):
            counts["sys"] += 1
            with torch.no_grad():
                return t2n(hess(to_torch(xn))).reshape(n, n)

    else:
        scipy_hess = hess

    if hessp == "autodiff":

        def scipy_hessp(xn, pn):
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

    elif callable(hessp):

        def scipy_hessp(xn, pn):
            counts["sys"] += 1
            with torch.no_grad():
                return t2n(hessp(to_torch(xn), to_torch(pn))).reshape(-1)

    else:
        scipy_hessp = hessp

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
