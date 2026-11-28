"""A ``scipy.optimize.minimize``-style front end for piazzi's scalar minimisers."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

import torch

from ._common import Callback, OptimizeResult
from ._lbfgs import lbfgs_armijo
from ._manifolds import Manifold
from ._torch_optim import torch_optimizer

__all__ = ["MINIMIZE_METHODS", "minimize"]

#: Scalar minimisers available to :func:`piazzi.minimize` by name: ``"lbfgs_armijo"``, and
#: ``"torchmin-<method>"`` once :mod:`piazzi.torchmin` is imported (``piazzi.minimize`` does so
#: when it meets such a name). To add one, register a callable
#: ``method(fun, x0, *, has_aux, callback, **options)`` under a lower-case name.
#: :func:`piazzi.minimize` passes ``manifold=`` only when one is given, so a method that
#: doesn't support manifolds need not accept it.
MINIMIZE_METHODS: dict[str, Callable[..., OptimizeResult]] = {
    "lbfgs_armijo": lbfgs_armijo,
}

# Optional backends register their methods, under a prefix, when their module is imported;
# a prefixed name that isn't registered yet imports the module that owns the prefix.
_BACKEND_MODULES = {"torchmin-": "piazzi.torchmin"}


def minimize(
    fun: Callable,
    x0: torch.Tensor,
    method: str | type[torch.optim.Optimizer] = "lbfgs_armijo",
    *,
    has_aux: bool = False,
    manifold: Manifold | None = None,
    callback: Callback | None = None,
    options: dict[str, Any] | None = None,
) -> OptimizeResult:
    """Minimise a scalar function of a tensor, like :func:`scipy.optimize.minimize`.

    Gradients come from autograd. As in SciPy, ``method`` names the solver and its settings
    go in ``options``. :func:`piazzi.scipy.minimize` has the same calling convention and runs
    SciPy's methods, so you can swap backends to compare them.

    ``method`` can also be a :class:`torch.optim.Optimizer` subclass, such as
    :class:`torch.optim.LBFGS` or :class:`torch.optim.Adam`. ``options`` then holds its
    constructor arguments (``lr``, ``line_search_fn``, ...) plus piazzi's ``max_iter``
    (default ``100``), ``step_tol`` (stop once a step is below it in max-norm; off by
    default, since some optimisers, such as Rprop, skip steps) and, for LBFGS,
    ``max_eval``. Each iteration is one optimiser step, ``callback`` is called after each,
    and every evaluation computes the value and gradient together.
    :class:`torch.optim.LBFGS` runs one L-BFGS iteration per step, with ``max_iter``,
    ``max_eval`` (default ``max_iter * 5 // 4``) and its ``tolerance_grad`` and
    ``tolerance_change`` tests applied to the whole run, so it follows the same path as a
    single upstream ``step`` call. The point its line search accepts is not evaluated twice.
    ``result.raw`` is the optimiser.

    Args:
        fun (Callable): ``fun(x) -> loss``, or ``(loss, aux)`` with ``has_aux=True``, where
            ``loss`` is a scalar tensor differentiable with respect to ``x`` and ``x`` has
            the shape of ``x0``.
        x0 (Tensor): starting point, of any shape, or a point on ``manifold``.
        method (str or type, optional): a name from :data:`piazzi.MINIMIZE_METHODS` (case
            insensitive; ``"lbfgs_armijo"``, that is :func:`piazzi.lbfgs_armijo`, or
            ``"torchmin-bfgs"``, ``"torchmin-trust-exact"``, ... for the methods of
            :func:`piazzi.torchmin.minimize`, which needs pytorch-minimize), or a
            :class:`torch.optim.Optimizer` subclass (the class, not an instance). Default: ``"lbfgs_armijo"``

    Keyword Args:
        has_aux (bool, optional): whether ``fun`` also returns an auxiliary output (any tree
            of tensors), returned detached in ``result.aux`` and ``info.aux``.
            Default: ``False``
        manifold (Manifold, optional): keeps ``x`` on a manifold such as
            :class:`piazzi.SE3`; see :func:`piazzi.lbfgs_armijo`. Not supported with
            :class:`torch.optim.Optimizer` methods. Default: ``None``
        callback (Callable, optional): called with an :class:`piazzi.IterationInfo` after
            every iteration; returning ``True`` stops the run. Default: ``None``
        options (dict, optional): settings for the method, for example
            ``{"max_iter": 500, "step_tol": 1e-8}``. Default: ``None``

    Returns:
        OptimizeResult: the method's result; see :func:`piazzi.lbfgs_armijo`.

    See Also:
        - :func:`piazzi.lbfgs_armijo`: the default method, which documents its ``options``.
        - :func:`piazzi.scipy.minimize`: the same problem with SciPy's methods.

    Example::

        >>> def rosenbrock(x):
        ...     return (100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2).sum()
        >>> x0 = torch.tensor([-1.2, 1.0], dtype=torch.float64)
        >>> piazzi.minimize(rosenbrock, x0).x
        tensor([1.0000, 1.0000], dtype=torch.float64)
        >>> res = piazzi.minimize(rosenbrock, x0, method=torch.optim.LBFGS, options={"line_search_fn": "strong_wolfe"})
        >>> res.x
        tensor([1.0000, 1.0000], dtype=torch.float64)
    """
    if isinstance(method, type) and issubclass(method, torch.optim.Optimizer):
        if manifold is not None:
            raise ValueError("manifold= is not supported with torch.optim optimisers; use method='lbfgs_armijo'")
        return torch_optimizer(fun, x0, method, has_aux=has_aux, callback=callback, **(options or {}))
    if not isinstance(method, str):
        raise TypeError(
            f"method must be a name from {sorted(MINIMIZE_METHODS)} or a torch.optim.Optimizer subclass"
            f" (the class, not an instance), got {method!r}"
        )
    key = method.lower()
    if key not in MINIMIZE_METHODS:
        for prefix, module in _BACKEND_MODULES.items():
            if key.startswith(prefix):
                importlib.import_module(module)  # registers its methods; raises if the backend is missing
    if key not in MINIMIZE_METHODS:
        raise ValueError(f"method must be one of {sorted(MINIMIZE_METHODS)}, got {method!r}")
    extra = {} if manifold is None else {"manifold": manifold}  # registered methods need not know manifolds
    return MINIMIZE_METHODS[key](fun, x0, has_aux=has_aux, callback=callback, **extra, **(options or {}))
