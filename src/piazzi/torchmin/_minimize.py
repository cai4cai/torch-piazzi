"""piazzi's calling convention for ``torchmin.minimize``."""

from __future__ import annotations

import collections
import contextlib
import warnings
from collections.abc import Callable
from typing import Any

import torch
import torchmin

from .._common import Callback, IterationInfo, OptimizeResult, check_x0, detach_tree, split_aux
from .._manifolds import Manifold

__all__ = ["METHODS", "minimize"]

#: The methods of ``torchmin.minimize``.
METHODS = ("bfgs", "l-bfgs", "cg", "newton-cg", "newton-exact", "dogleg", "trust-ncg", "trust-exact", "trust-krylov")

# Evaluations kept to look up the value at the iterates torchmin reports (its line searches
# evaluate a handful of trial points per iteration).
_HISTORY = 32


# torchmin's default gradient tolerance per method (Newton's methods have none: 1e-5 is the
# default of its quasi-Newton ones), and the fraction of the gradient at x0 that also counts as small.
_GTOL = dict.fromkeys(METHODS, 1e-5) | {"dogleg": 1e-4, "trust-ncg": 1e-4, "trust-exact": 1e-4, "trust-krylov": 1e-4}
_GRAD_RTOL = 1e-3


@contextlib.contextmanager
def quiet_torchmin_warnings():
    """Silence a deprecation warning that torchmin's exact trust-region solver raises on every call.

    ``torch.triangular_solve`` is deprecated in torch >= 1.11; torchmin 0.1.0 still uses it in
    ``trustregion/exact.py``. Remove this filter once a torchmin release uses
    ``torch.linalg.solve_triangular`` (upstream: https://github.com/rfeinman/pytorch-minimize).
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"torch\.triangular_solve is deprecated", UserWarning, r"torchmin")
        yield


def _clone_tree(aux):
    """Clone every tensor in a (nested) dict / list / tuple."""
    if isinstance(aux, torch.Tensor):
        return aux.clone()
    if isinstance(aux, dict):
        return {k: _clone_tree(v) for k, v in aux.items()}
    if isinstance(aux, (list, tuple)):
        return type(aux)(_clone_tree(v) for v in aux)
    return aux


def minimize(
    fun: Callable,
    x0: torch.Tensor,
    method: str = "l-bfgs",
    *,
    has_aux: bool = False,
    manifold: Manifold | None = None,
    callback: Callback | None = None,
    options: dict[str, Any] | None = None,
) -> OptimizeResult:
    """Run ``torchmin.minimize`` (pytorch-minimize) on a scalar function written in PyTorch.

    It has the calling convention of :func:`piazzi.minimize` (``fun``, ``x0``, ``method``,
    ``has_aux``, ``callback`` and ``options``), so you can swap backends to compare them;
    ``method`` and the keys of ``options`` are torchmin's. Everything stays in PyTorch: ``fun``
    and torchmin's solver run on the device and in the dtype of ``x0``. Gradients, Hessians and
    Hessian-vector products come from autograd, so the Newton-type and trust-region methods
    need an objective that is twice differentiable.

    Args:
        fun (Callable): ``fun(x) -> loss``, or ``(loss, aux)`` with ``has_aux=True``, where
            ``x`` has the shape of ``x0`` and ``loss`` is a scalar tensor.
        x0 (Tensor): starting point, of any shape.
        method (str, optional): one of :data:`piazzi.torchmin.METHODS`: ``"bfgs"``,
            ``"l-bfgs"``, ``"cg"``, ``"newton-cg"``, ``"newton-exact"``, ``"dogleg"``,
            ``"trust-ncg"``, ``"trust-exact"`` or ``"trust-krylov"`` (case insensitive).
            Default: ``"l-bfgs"``

    Keyword Args:
        has_aux (bool, optional): whether ``fun`` also returns an auxiliary output (any tree
            of tensors), returned detached in ``result.aux`` and ``info.aux``. Default: ``False``
        manifold (Manifold, optional): not supported, since torchmin has no retraction: giving
            one raises :class:`ValueError`. Use :func:`piazzi.lbfgs` for manifolds.
            Default: ``None``
        callback (Callable, optional): called after every iteration with an
            :class:`piazzi.IterationInfo` (tensors, with the evaluation counts so far), as for
            the native solvers; returning ``True`` stops the run. ``info.accepted`` is
            ``False`` when the iterate did not move (a rejected trust-region step), and then
            ``info.x`` is the previous iterate. Default: ``None``
        options (dict, optional): torchmin's method-specific options (``max_iter``, ``gtol``,
            ``xtol``, ``history_size``, ``line_search``, ...). ``callback`` and ``return_all``
            are not accepted; torchmin's own ``disp`` is. Default: ``None``

    Returns:
        OptimizeResult: ``x`` with the shape, dtype and device of ``x0``, ``fun`` the value and
        ``aux`` the auxiliary output there. ``n_iter`` is torchmin's ``nit``. Every
        evaluation of ``fun`` in torchmin is a value-and-gradient one (its line searches
        included), so ``n_fun_grad`` counts the calls to ``fun``, ``n_fun`` is ``0`` and
        ``n_system`` is ``0``: the Hessian and Hessian-vector products of the Newton-type
        and trust-region methods are autograd backward passes through the graph kept from
        the evaluation, not evaluations of ``fun``, and are not counted. The trust-region
        methods compute the gradient and the Hessian at every trial point, rejected ones
        included. ``converged`` is torchmin's ``success`` with two more checks: the final value is
        finite, and the max-norm of the gradient there is at most ``options["gtol"]`` (torchmin's
        default for the method if absent: ``1e-4`` for the trust-region methods, ``1e-5``
        otherwise) or ``1e-3`` times the max-norm of the gradient at ``x0``,
        whichever is larger. This matters for ``bfgs``, ``l-bfgs``,
        ``newton-cg`` and ``newton-exact``, which also report success when a step or a decrease falls below
        ``xtol``; such a stall at a large gradient is not converged, and ``message`` says so.
        ``message`` is otherwise torchmin's, and ``raw`` is torchmin's own result. Those methods
        stop on ``xtol`` before moving to the last trial point, so ``x`` is then the previous
        iterate, and the last ``info.x`` the callback saw can differ from it. If the final
        point was not among the last evaluations (this can only happen when a callback
        stops the run), ``fun`` is evaluated there once more and the counts include it.

    See Also:
        - :func:`piazzi.minimize`: the native interface with the same calling convention.
        - :func:`piazzi.scipy.minimize`: the same problem with SciPy's methods.

    Example::

        >>> def rosenbrock(x):
        ...     return (100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2).sum()
        >>> x0 = torch.tensor([-1.2, 1.0], dtype=torch.float64)
        >>> res = piazzi.torchmin.minimize(rosenbrock, x0, "trust-exact")
        >>> res.x
        tensor([1.0000, 1.0000], dtype=torch.float64)
    """
    if manifold is not None:
        raise ValueError("manifold= is not supported by piazzi.torchmin; use method='lbfgs'")
    if not isinstance(method, str) or method.lower() not in METHODS:
        raise ValueError(f"method must be one of {list(METHODS)}, got {method!r}")
    method = method.lower()
    options = dict(options or {})
    for key in ("callback", "return_all"):
        if key in options:
            raise ValueError(f"options[{key!r}] is not supported: pass callback= to piazzi.torchmin.minimize")
    like, shape = check_x0(x0)  # like: an owned flat copy
    counts = {"fg": 0}
    first_grad: list[float] = []  # max-norm of the gradient at x0, the scale for the convergence check
    history: collections.deque = collections.deque(maxlen=_HISTORY)  # (flat x, loss, aux)

    def wrapped(x):
        loss, aux = split_aux(fun(x.clone()), has_aux)  # fun gets its own copy; the graph still leads to x
        if x.requires_grad:
            # torchmin copies the gradient in place, which fails when autograd returns an expanded
            # view (the gradient of a sum, say): hand it a contiguous one.
            def contiguous(g):
                if not first_grad:
                    first_grad.append(g.abs().max().item())
                return g.contiguous().clone()

            x.register_hook(contiguous)
        # torchmin updates the value it gets in place (``f[...] = f_new``), so keep our own copies.
        history.append((x.detach().reshape(-1).clone(), loss.detach().clone(), detach_tree(aux)))
        counts["fg"] += 1
        return loss

    def lookup(xflat):
        for xh, loss, aux in reversed(history):
            if torch.equal(xh, xflat):
                return loss, aux
        return None

    state: dict[str, Any] = {"it": 0, "x": like.clone(), "stopped": None}

    def on_iteration(xflat):
        xflat = xflat.detach().reshape(-1)
        found = lookup(xflat)
        nan = torch.full((), torch.nan, dtype=like.dtype, device=like.device)
        loss, aux = found if found is not None else (nan, None)
        if "start" not in state:
            at_x0 = lookup(like)
            state["start"] = at_x0[0] if at_x0 else loss
        info = IterationInfo(
            state["it"],
            xflat.clone().reshape(shape),
            loss.clone(),
            state["start"],
            _clone_tree(aux),
            not torch.equal(xflat, state["x"]),
            counts["fg"],
            0,
            0,
        )
        state["it"] += 1
        state["x"] = xflat.clone()
        assert callback is not None  # on_iteration is only passed to torchmin when there is a callback
        stop = bool(callback(info))
        if stop:
            state["stopped"] = (state["x"], found)
        return stop

    with quiet_torchmin_warnings():
        res = torchmin.minimize(
            wrapped,
            like.clone().reshape(shape),  # torchmin keeps views of what it is given
            method,
            options=options,
            callback=None if callback is None else on_iteration,
        )
    xflat = res.x.detach().reshape(-1).clone()
    found = None
    if state["stopped"] is not None:
        # torchmin returns the iterate before the one the callback stopped at; report the one it saw.
        xflat, found = state["stopped"]
    else:
        found = lookup(xflat)
    if found is None:
        # Not a recent evaluation (e.g. nothing was evaluated): evaluate once more, and count it.
        with torch.enable_grad():
            loss, aux = split_aux(fun(xflat.clone().reshape(shape)), has_aux)
        counts["fg"] += 1
        found = (loss.detach().clone(), detach_tree(aux))
    loss, aux = found
    message = str(res.message)
    converged = bool(res.success) and bool(torch.isfinite(loss)) and state["stopped"] is None
    grad = getattr(res, "grad", None)
    if converged and grad is not None and first_grad:
        gmax = grad.abs().max().item()
        limit = max(options.get("gtol", _GTOL[method]), _GRAD_RTOL * first_grad[0])
        if not gmax <= limit:
            converged = False
            message += f" (but the gradient max-norm is {gmax:.2e}, above {limit:.2e}: not a minimiser)"
    return OptimizeResult(
        x=xflat.reshape(shape),
        fun=loss,
        aux=_clone_tree(aux),
        n_iter=int(res.nit),
        n_fun_grad=counts["fg"],
        n_fun=0,
        n_system=0,
        converged=converged,
        message=message,
        raw=res,
    )
