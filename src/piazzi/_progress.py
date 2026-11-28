"""A ready-made progress printer for the solvers' ``callback``."""

from __future__ import annotations

import math
import sys
from collections.abc import Callable
from typing import TextIO

from ._common import IterationInfo

__all__ = ["print_progress"]

_HEADER = (
    f"{'iter':>5}  {'fun':>13}  {'rel_decrease':>12}  {'step':>8}  {'n_fun_grad':>10}  {'n_fun':>6}  {'n_system':>8}"
)


def print_progress(every: int = 1, file: TextIO | None = None) -> Callable[[IterationInfo], None]:
    """Make a ``callback`` that prints a table of the solver's progress, one row per iteration.

    The columns are the iteration, the objective value after it, its relative decrease over
    the iteration, whether the step was accepted or rejected, and the evaluations so far by
    kind (see :class:`piazzi.IterationInfo`), in the spirit of Ceres Solver's per-iteration
    summary. A header is printed before the first iteration of each run, so the same
    callback can be reused. It never stops the run; to stop early as well, call it from your
    own callback.

    Args:
        every (int, optional): print every ``every``-th iteration. Default: ``1``
        file (TextIO, optional): where to print. Default: ``None``, which prints to
            :data:`sys.stdout` (looked up at each call).

    Returns:
        Callable: ``callback(info)`` for any solver or wrapper's ``callback`` argument.

    Example::

        >>> t = torch.linspace(0, 1, 20, dtype=torch.float64)
        >>> y = 2 * torch.exp(-3 * t)
        >>> res = piazzi.least_squares(
        ...     lambda p: p[0] * torch.exp(-p[1] * t) - y,
        ...     torch.tensor([1.0, 1.0], dtype=torch.float64),
        ...     callback=piazzi.print_progress(every=2),
        ... )  # doctest: +ELLIPSIS
         iter            fun  rel_decrease      step  n_fun_grad   n_fun  n_system
            0   3.876612e-01     7.208e-01  accepted           0       0         2
            2   4.938297e-10     1.000e+00  accepted           0       0         4...
    """
    if not (isinstance(every, int) and every >= 1):
        raise ValueError(f"every must be a positive integer, got {every!r}")

    def callback(info: IterationInfo) -> None:
        out = sys.stdout if file is None else file
        if info.iteration == 0:
            print(_HEADER, file=out)
        if info.iteration % every:
            return
        fun, start = float(info.fun), float(info.fun_start)
        rel = (start - fun) / abs(start) if start != 0 and math.isfinite(start) else math.nan
        step = "accepted" if info.accepted else "rejected"
        row = (
            f"{info.iteration:>5}  {fun:>13.6e}  {rel:>12.3e}  {step:>8}  "
            f"{info.n_fun_grad:>10}  {info.n_fun:>6}  {info.n_system:>8}"
        )
        print(row, file=out)

    return callback
