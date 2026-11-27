"""Shared types and helpers."""

from __future__ import annotations

import contextlib
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class IterationInfo:
    """Passed to ``callback`` after every iteration (accepted or not).

    ``x`` and ``fun`` are the current iterate and its value: the new point after an
    accepted step, the unchanged point after a rejected one. ``fun_start`` is the value
    at the start of the iteration.

    ``n_fun_grad``, ``n_fun`` and ``n_system`` count the evaluations so far, including this
    iteration's, as in :class:`OptimizeResult`. Use them to plot progress against cost
    rather than iterations, which hide the evaluations a line search spends.
    """

    iteration: int
    x: torch.Tensor
    fun: torch.Tensor
    fun_start: torch.Tensor
    aux: Any
    accepted: bool
    n_fun_grad: int = 0
    n_fun: int = 0
    n_system: int = 0


@dataclass
class OptimizeResult:
    """Outcome of a solver run.

    ``n_fun_grad``: value + gradient evaluations, ``n_fun``: value-only evaluations,
    ``n_system``: evaluations of second-order information (the normal equations of
    Gauss-Newton / Levenberg-Marquardt, or SciPy's Hessian or Hessian-vector products).
    ``converged`` is True when the run stopped at a minimiser to working precision: on a step
    or decrease below tolerance (``step_tol``, rounding noise, or an optimiser's own
    tolerances), at a zero gradient, or with no free parameters. ``message`` says which.
    ``raw`` holds the backend's own result object for third-party wrappers (``piazzi.scipy``),
    and is None for the native solvers.
    """

    x: torch.Tensor
    fun: torch.Tensor
    aux: Any
    n_iter: int
    n_fun_grad: int = 0
    n_fun: int = 0
    n_system: int = 0
    converged: bool = False
    message: str = ""
    raw: Any = None


Callback = Callable[[IterationInfo], "bool | None"]


def split_aux(out, has_aux: bool):
    if has_aux:
        if not isinstance(out, tuple) or len(out) != 2:
            raise ValueError("with has_aux=True the function must return (value, aux)")
        return out
    return out, None


def detach_tree(aux):
    """Detach every tensor in a (nested) dict / list / tuple."""
    if isinstance(aux, torch.Tensor):
        return aux.detach()
    if isinstance(aux, dict):
        return {k: detach_tree(v) for k, v in aux.items()}
    if isinstance(aux, (list, tuple)):
        return type(aux)(detach_tree(v) for v in aux)
    return aux


def check_x0(x0: torch.Tensor) -> tuple[torch.Tensor, torch.Size]:
    if not isinstance(x0, torch.Tensor) or not x0.is_floating_point():
        raise TypeError("x0 must be a floating-point tensor")
    return x0.detach().reshape(-1).clone(), x0.shape


def transfer(t: torch.Tensor, device, dtype: torch.dtype) -> torch.Tensor:
    """Move ``t`` to ``device`` and cast it to ``dtype``, casting on the side that supports the wider dtype.

    Narrowing (float64 -> float32) casts first, since the target device may lack float64 (MPS);
    widening moves first, since the source device may lack it.
    """
    if torch.finfo(dtype).bits < torch.finfo(t.dtype).bits:
        return t.to(dtype).to(device)
    return t.to(device).to(dtype)


# How many units in the last place an entry of x may change by and still count as unmoved.
# Tuned on a float32 benchmark (Rosenbrock from random starts, SE(3) registration with data
# perturbed by an ulp, robust and far-off exponential fits, scaled objectives, runs capped by
# max_iter; see #19 and #25): there, it let every run that reached a minimiser report
# convergence, and no run that had not. 32 left an L-BFGS registration on 1 in 200 perturbed
# data sets one step short (45 ulps), so 64 leaves a margin. A heuristic, not a guarantee.
RESOLUTION_ULPS = 64


def entry_scales(manifold, x: torch.Tensor) -> torch.Tensor:
    """The scale of each entry of ``x`` (``|x|`` or the manifold's
    :meth:`~piazzi.Manifold.entry_scale`), floored at ``sqrt(eps)`` times the largest one.

    The floor gives entries at or near 0 (a parameter whose value is 0, a translation at the
    origin) a usable scale, and is small enough that a small parameter next to a large one
    still moves visibly.
    """
    scale = x.abs() if manifold is None else manifold.entry_scale(x)
    return scale.clamp(min=torch.finfo(x.dtype).eps ** 0.5 * scale.max().item())


def within(manifold, x: torch.Tensor, x_new: torch.Tensor, relative: float) -> bool:
    """Whether every entry of ``x_new`` is within ``relative`` times its scale of ``x``."""
    return bool(((x_new - x).abs() <= relative * entry_scales(manifold, x)).all())


def unmoved(manifold, x: torch.Tensor, x_new: torch.Tensor) -> bool:
    """Whether ``x_new`` is within the resolution of ``x``: every entry within ``RESOLUTION_ULPS``
    units in the last place of its scale, ``|x|`` entry by entry or the manifold's
    :meth:`~piazzi.Manifold.entry_scale`, floored at ``sqrt(eps)`` times the largest scale.

    A Newton-like step (the Gauss-Newton step, an L-BFGS step from its memory) estimates the
    distance to the minimiser, so when it leaves ``x`` unmoved in this sense, ``x`` is a
    minimiser to working precision and any change in the loss along the step is rounding
    noise. Comparing points entry by entry (rather than the step with ``max|x|``) keeps small
    parameters next to large ones from counting as converged early, and works on manifolds,
    where the step lives in tangent coordinates.
    """
    return within(manifold, x, x_new, RESOLUTION_ULPS * torch.finfo(x.dtype).eps)


def retract(manifold, x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """``x + d`` on a flat point, or ``manifold.plus(x, d)`` when a manifold is given."""
    return x + d if manifold is None else manifold.plus(x, d)


@contextlib.contextmanager
def quiet_jvp_warnings():
    """Silence a spurious deprecation warning on the first forward-mode AD call.

    The first forward-mode AD call makes torch load its JVP decompositions, which call the
    deprecated torch.jit.script internally in torch 2.15 and earlier. Fixed upstream by
    https://github.com/pytorch/pytorch/pull/203621 (landed 2026-11-09, after 2.15.0); drop
    this filter once the minimum torch version includes it.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"`torch\.jit\.script` is deprecated", FutureWarning, r"torch\.jit")
        yield
