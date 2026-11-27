"""A ``scipy.optimize.minimize``-style front end for piazzi's scalar minimisers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from ._common import Callback, OptimizeResult
from ._lbfgs import lbfgs_armijo
from ._torch_optim import torch_optimizer

__all__ = ["MINIMIZE_METHODS", "minimize"]

MINIMIZE_METHODS: dict[str, Callable[..., OptimizeResult]] = {
    "lbfgs_armijo": lbfgs_armijo,
}


def minimize(
    fun: Callable,
    x0: torch.Tensor,
    method: str | type[torch.optim.Optimizer] = "lbfgs_armijo",
    *,
    has_aux: bool = False,
    callback: Callback | None = None,
    options: dict[str, Any] | None = None,
) -> OptimizeResult:
    """Minimise a scalar function of a tensor, like ``scipy.optimize.minimize``.

    ``fun(x)`` returns a scalar tensor (``(loss, aux)`` with ``has_aux``), and gradients come
    from autograd. ``method`` names a solver from :data:`piazzi.MINIMIZE_METHODS` (case
    insensitive); ``"lbfgs_armijo"`` (:func:`piazzi.lbfgs_armijo`) is currently the only
    one. As in SciPy, solver-specific settings go in ``options``, for example
    ``options={"max_iter": 500, "step_tol": 1e-8}``.

    ``method`` can also be a ``torch.optim.Optimizer`` subclass, such as
    ``torch.optim.LBFGS`` or ``torch.optim.Adam``. ``options`` then holds its constructor
    arguments (``lr``, ``line_search_fn``, ...) plus piazzi's ``max_iter`` (default 100),
    ``step_tol`` (stop once a step is below it in max-norm; off by default, since some
    optimisers, such as Rprop, skip steps) and, for LBFGS, ``max_eval``.
    Each iteration is one optimiser step, ``callback`` is called after each, and every
    evaluation computes the value and gradient together. ``torch.optim.LBFGS`` runs one
    L-BFGS iteration per step, with ``max_iter``, ``max_eval`` (default
    ``max_iter * 5 // 4``) and its ``tolerance_grad`` and ``tolerance_change`` tests applied
    to the whole run, so it follows the same path as a single upstream ``step`` call, for
    example ``method=torch.optim.LBFGS, options={"line_search_fn": "strong_wolfe"}``. The
    point its line search accepts is not evaluated twice. ``result.raw`` is the optimiser.

    :func:`piazzi.scipy.minimize` has the same calling convention and runs SciPy's methods,
    so you can swap backends to compare them.
    """
    if isinstance(method, type) and issubclass(method, torch.optim.Optimizer):
        return torch_optimizer(fun, x0, method, has_aux=has_aux, callback=callback, **(options or {}))
    if not isinstance(method, str):
        raise TypeError(
            f"method must be a name from {sorted(MINIMIZE_METHODS)} or a torch.optim.Optimizer subclass"
            f" (the class, not an instance), got {method!r}"
        )
    key = method.lower()
    if key not in MINIMIZE_METHODS:
        raise ValueError(f"method must be one of {sorted(MINIMIZE_METHODS)}, got {method!r}")
    return MINIMIZE_METHODS[key](fun, x0, has_aux=has_aux, callback=callback, **(options or {}))
