"""A ``scipy.optimize.minimize``-style front end for piazzi's scalar minimisers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from ._common import Callback, OptimizeResult
from ._lbfgs import lbfgs_armijo
from ._manifolds import Manifold

__all__ = ["MINIMIZE_METHODS", "minimize"]

MINIMIZE_METHODS: dict[str, Callable[..., OptimizeResult]] = {
    "lbfgs_armijo": lbfgs_armijo,
}


def minimize(
    fun: Callable,
    x0: torch.Tensor,
    method: str = "lbfgs_armijo",
    *,
    has_aux: bool = False,
    manifold: Manifold | None = None,
    callback: Callback | None = None,
    options: dict[str, Any] | None = None,
) -> OptimizeResult:
    """Minimise a scalar function of a tensor, like ``scipy.optimize.minimize``.

    ``fun(x)`` returns a scalar tensor (``(loss, aux)`` with ``has_aux``), and gradients come
    from autograd. ``method`` names a solver from :data:`piazzi.MINIMIZE_METHODS` (case
    insensitive); ``"lbfgs_armijo"`` (:func:`piazzi.lbfgs_armijo`) is currently the only
    one. As in SciPy, solver-specific settings go in ``options``, for example
    ``options={"max_iter": 500, "step_tol": 1e-8}``. ``manifold`` (for example
    :class:`piazzi.SE3`) keeps ``x`` on a constrained set; see :func:`piazzi.lbfgs_armijo`.

    :func:`piazzi.scipy.minimize` has the same calling convention and runs SciPy's methods,
    so you can swap backends to compare them.
    """
    key = method.lower()
    if key not in MINIMIZE_METHODS:
        raise ValueError(f"method must be one of {sorted(MINIMIZE_METHODS)}, got {method!r}")
    return MINIMIZE_METHODS[key](fun, x0, has_aux=has_aux, manifold=manifold, callback=callback, **(options or {}))
