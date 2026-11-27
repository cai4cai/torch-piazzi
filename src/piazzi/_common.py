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
    ``n_system``: normal-equation evaluations (Gauss-Newton / Levenberg-Marquardt).
    ``converged`` is True when the run stopped on ``step_tol`` or at a zero gradient;
    ``message`` says why it stopped.
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
