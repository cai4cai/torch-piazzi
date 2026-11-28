"""Line search algorithms shared by gradient-based solvers.

Provides:

- :func:`cubic_interp`: cubic Hermite minimiser used by :func:`wolfe_search`.
- :func:`armijo_search`: cheap Armijo backtracking (one forward pass per rejected trial).
- :func:`wolfe_search`: strong Wolfe bracketing and zoom with cubic interpolation.

Both search routines accept pre-wired ``eval_*`` callables from the caller so that
evaluation counting and graph management stay in the solver.  The common return tuple is::

    (x_new, f_new, aux_new, g_new_or_None, t_accepted, x_first, first_unmoved, search_ok)

``g_new_or_None`` is ``None`` only from :func:`armijo_search` when the accepted point came
from a ``value_fun`` evaluation (the caller must call its full evaluator to get the gradient).
``first_unmoved`` is ``True`` when the first trial left ``x`` unmoved within the resolution of
the dtype, which signals a minimiser to working precision when the search was from memory.
``search_ok`` is ``False`` when no sufficient decrease was found.
"""

from __future__ import annotations

import math

import torch

from ._common import unmoved

__all__: list[str] = []  # private module; nothing exported directly


def cubic_interp(
    t0: float,
    f0: float,
    g0: float,
    t1: float,
    f1: float,
    g1: float,
    lo: float,
    hi: float,
) -> float:
    """Cubic minimiser of the Hermite interpolant through ``(t0, f0, g0)`` and ``(t1, f1, g1)``.

    Clamped to ``[lo, hi]``.  When the discriminant is negative (complex root) the midpoint
    is returned.  Ported from ``torch/optim/lbfgs.py::_cubic_interpolate``, which itself is
    ported from ``torch/optim/lswolfe.lua`` (BSD-3-Clause).
    """
    d1 = g0 + g1 - 3.0 * (f0 - f1) / (t0 - t1)
    d2sq = d1 * d1 - g0 * g1
    if d2sq >= 0.0:
        d2 = math.sqrt(d2sq)
        if t0 <= t1:
            t_min = t1 - (t1 - t0) * (g1 + d2 - d1) / (g1 - g0 + 2.0 * d2)
        else:
            t_min = t0 - (t0 - t1) * (g0 + d2 - d1) / (g0 - g1 + 2.0 * d2)
    else:
        t_min = (lo + hi) / 2.0
    return min(max(t_min, lo), hi)


def _armijo_ok(fn: float, bound: float) -> bool:
    """``True`` iff ``fn`` is finite and satisfies ``fn <= bound``."""
    return math.isfinite(fn) and fn <= bound


def armijo_search(
    x: torch.Tensor,
    f_val: float,
    d: torch.Tensor,
    t: float,
    gtd: float,
    used_memory: bool,
    manifold,
    c1: float,
    max_backtracks: int,
    eval_trial,
    eval_value,
    accept,
    reject,
    retract,
    value_fun,
):
    """Armijo backtracking line search.

    Each trial that fails the sufficient-decrease condition costs one forward pass (via
    ``eval_trial`` or ``eval_value``); the backward pass runs only once a point is accepted
    (via ``accept``).  When the accepted point came from a ``value_fun`` evaluation the
    returned ``g_new`` is ``None``; the caller must call its full evaluator.

    Args:
        x (Tensor): current flat iterate.
        f_val (float): objective value at ``x``.
        d (Tensor): search direction (flat).
        t (float): initial step length.
        gtd (float): directional derivative ``g · d`` at ``x``.
        used_memory (bool): whether the direction came from L-BFGS memory (vs steepest descent).
        manifold: manifold object or ``None``.
        c1 (float): Armijo sufficient-decrease constant.
        max_backtracks (int): maximum number of step reductions.
        eval_trial: ``eval_trial(xv)`` → ``(f, aux, pending)``.
        eval_value: ``eval_value(xv)`` → ``f`` (no-grad, only called when ``value_fun`` is set).
        accept: ``accept(pending)`` → ``g``; runs the deferred backward pass and counts it.
        reject: ``reject(pending)``; counts the rejected forward pass.
        retract: ``retract(manifold, x, delta)`` → ``x_new``.
        value_fun: the value-only callable, or ``None``.

    Returns:
        tuple: ``(x_new, f_new, aux_new, g_new_or_None, t, x_first, first_unmoved, search_ok)``
    """
    x_new = retract(manifold, x, t * d)
    f_new, aux_new, pending = eval_trial(x_new)
    first_unmoved = used_memory and math.isfinite(f_new.item()) and unmoved(manifold, x, x_new)
    x_first = x_new
    n_bt = 0
    while not _armijo_ok(f_new.item(), f_val + c1 * t * gtd) and n_bt < max_backtracks:
        reject(pending)
        pending = None  # free the rejected trial's graph before the next forward pass
        denom = 2.0 * (f_new.item() - f_val - gtd * t)
        t_q = -gtd * t * t / denom if denom > 0 else 0.5 * t
        t = min(max(t_q, 0.1 * t), 0.5 * t)
        x_new = retract(manifold, x, t * d)
        if value_fun is None:
            f_new, aux_new, pending = eval_trial(x_new)
        else:
            f_new, pending = eval_value(x_new), None
        n_bt += 1
    if not _armijo_ok(f_new.item(), f_val + c1 * t * gtd):  # no sufficient decrease found
        reject(pending)
        return x_new, f_new, aux_new, None, t, x_first, first_unmoved, False
    # Accepted.  Resolve the deferred backward pass, or flag that the caller must re-evaluate.
    # pending is None when the accepted point came from a value_fun evaluation (no graph kept).
    g_new = None if pending is None else accept(pending)
    return x_new, f_new, aux_new, g_new, t, x_first, first_unmoved, True


def wolfe_search(
    x: torch.Tensor,
    f_val: float,
    g: torch.Tensor,
    aux,
    d: torch.Tensor,
    t_init: float,
    t_max: float,
    gtd: float,
    used_memory: bool,
    manifold,
    c1: float,
    c2: float,
    max_ls: int,
    eval_full,
    retract,
):
    """Strong Wolfe line search: bracketing phase then zoom with cubic interpolation.

    Implements Nocedal and Wright (2006) Algorithms 3.5 and 3.6.  Every trial point requires
    a gradient evaluation (the curvature condition needs it), so all evaluations are counted
    in ``n_fun_grad`` (via ``eval_full``).  The curvature condition guarantees ``sᵀy > 0``
    at any point that fully satisfies the strong Wolfe conditions; when the budget is exhausted
    the best point found is returned with ``search_ok=True`` if it moved and satisfies Armijo,
    or ``search_ok=False`` otherwise (a step of length zero is never a success).

    The implementation is ported from ``torch/optim/lbfgs.py::_strong_wolfe`` (BSD-3-Clause),
    which is itself ported from ``torch/optim/lswolfe.lua``.  Manifold support follows Ceres
    Solver's line-search minimiser: steps are applied via the manifold's ``plus`` operator and
    directional derivatives are computed in tangent coordinates.

    Args:
        x (Tensor): current flat iterate.
        f_val (float): objective value at ``x``.
        g (Tensor): gradient at ``x`` (flat, tangent coordinates).
        aux: auxiliary output at ``x`` (returned if the search ends back at ``x``).
        d (Tensor): search direction (flat, tangent coordinates).
        t_init (float): initial step length.
        t_max (float): largest step length (``max_step`` over the largest entry of ``d``); the
            expansion phase never goes beyond it.
        gtd (float): directional derivative ``g · d`` at ``x`` (must be negative).
        used_memory (bool): whether the direction came from L-BFGS memory (vs steepest descent).
        manifold: manifold object or ``None``.
        c1 (float): Armijo sufficient-decrease constant (``0 < c1 < c2``).
        c2 (float): curvature constant (``c1 < c2 < 1``).
        max_ls (int): number of trial points after the first one (as ``max_backtracks`` for
            Armijo), for the bracketing and zoom phases together.
        eval_full: ``eval_full(xv)`` → ``(f, aux, g)`` (value, aux, gradient); counts one
            ``n_fun_grad``.
        retract: ``retract(manifold, x, delta)`` → ``x_new``.

    Returns:
        tuple: ``(x_new, f_new, aux_new, g_new, t, x_first, first_unmoved, search_ok)``
    """
    d_norm = d.abs().max().item()

    # ---- bracketing phase (Algorithm 3.5) ----
    t_prev, f_prev, gtd_prev = 0.0, f_val, gtd
    g_prev: torch.Tensor = g
    aux_prev = aux
    x_prev: torch.Tensor = x
    t = t_init
    ls_iter = 0
    bracket: list[float] = []
    bracket_f: list[float] = []
    bracket_x: list[torch.Tensor] = []
    bracket_aux: list = []
    bracket_g: list[torch.Tensor] = []
    bracket_gtd: list[float] = []
    done = False

    x_first = retract(manifold, x, t * d)
    f_new, aux_new, g_new = eval_full(x_first)
    f_new_v = f_new.item()
    g_new_v: torch.Tensor = g_new
    gtd_new = torch.dot(g_new_v, d).item()
    first_unmoved = used_memory and math.isfinite(f_new_v) and unmoved(manifold, x, x_first)
    x_new = x_first

    while ls_iter < max_ls:
        # Violation of Armijo, or the function went back up (monotone bracketing).
        if not math.isfinite(f_new_v) or f_new_v > f_val + c1 * t * gtd or (ls_iter > 0 and f_new_v >= f_prev):
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new_v]
            bracket_g = [g_prev.clone(), g_new_v.clone()]
            bracket_aux = [aux_prev, aux_new]
            bracket_x = [x_prev, x_new]
            bracket_gtd = [gtd_prev, gtd_new]
            break
        if abs(gtd_new) <= -c2 * gtd:
            # Strong Wolfe conditions already satisfied.
            bracket = [t]
            bracket_f = [f_new_v]
            bracket_g = [g_new_v]
            bracket_aux = [aux_new]
            bracket_x = [x_new]
            done = True
            break
        if gtd_new >= 0:
            # Derivative turned positive: bracket is [t, t_prev].
            bracket = [t, t_prev]
            bracket_f = [f_new_v, f_prev]
            bracket_g = [g_new_v.clone(), g_prev.clone()]
            bracket_aux = [aux_new, aux_prev]
            bracket_x = [x_new, x_prev]
            bracket_gtd = [gtd_new, gtd_prev]
            break
        if t >= t_max:
            # Cannot expand beyond max_step: take this point (Armijo holds, curvature may not).
            return x_new, f_new, aux_new, g_new_v, t, x_first, first_unmoved, True
        # Expand: interpolate a new trial step in [t + 0.01*(t-t_prev), min(10*t, t_max)].
        min_step = min(t + 0.01 * (t - t_prev), t_max)
        max_step = min(t * 10.0, t_max)
        t_new = cubic_interp(t_prev, f_prev, gtd_prev, t, f_new_v, gtd_new, min_step, max_step)
        t_prev, f_prev, gtd_prev = t, f_new_v, gtd_new
        g_prev = g_new_v.clone()
        aux_prev = aux_new
        x_prev = x_new
        t = t_new
        x_new = retract(manifold, x, t * d)
        f_new, aux_new, g_new = eval_full(x_new)
        f_new_v = f_new.item()
        g_new_v = g_new
        gtd_new = torch.dot(g_new_v, d).item()
        ls_iter += 1

    if ls_iter == max_ls and not done and not bracket:
        # Budget exhausted during expansion without bracketing. The last trial was never put
        # through the tests above: if it went back up (or is not finite), the previous trial,
        # which satisfied Armijo with a descending slope, is the better point.
        ok = t > 0 and math.isfinite(f_new_v) and f_new_v <= f_val + c1 * t * gtd
        if t_prev > 0 and (not ok or f_new_v >= f_prev):
            return (
                x_prev,
                torch.tensor(f_prev, dtype=g.dtype, device=g.device),
                aux_prev,
                g_prev,
                t_prev,
                x_first,
                first_unmoved,
                True,
            )
        return x_new, f_new, aux_new, g_new_v, t, x_first, first_unmoved, ok

    if done:
        return (
            bracket_x[0],
            torch.tensor(bracket_f[0], dtype=g.dtype, device=g.device),
            bracket_aux[0],
            bracket_g[0],
            bracket[0],
            x_first,
            first_unmoved,
            True,
        )

    # ---- zoom phase (Algorithm 3.6) ----
    low_pos = 0 if bracket_f[0] <= bracket_f[-1] else 1
    high_pos = 1 - low_pos
    insuf_progress = False
    while ls_iter < max_ls:
        # Bracket collapsed below floating-point resolution.
        if abs(bracket[1] - bracket[0]) * d_norm < torch.finfo(g.dtype).eps:
            break
        # Cubic interpolation within the current bracket.
        t = cubic_interp(
            bracket[0],
            bracket_f[0],
            bracket_gtd[0],
            bracket[1],
            bracket_f[1],
            bracket_gtd[1],
            min(bracket),
            max(bracket),
        )
        # Guard against insufficient progress (t lands too close to a boundary).
        eps_bracket = 0.1 * (max(bracket) - min(bracket))
        if min(max(bracket) - t, t - min(bracket)) < eps_bracket:
            if insuf_progress or t >= max(bracket) or t <= min(bracket):
                if abs(t - max(bracket)) < abs(t - min(bracket)):
                    t = max(bracket) - eps_bracket
                else:
                    t = min(bracket) + eps_bracket
                insuf_progress = False
            else:
                insuf_progress = True
        else:
            insuf_progress = False

        x_new = retract(manifold, x, t * d)
        f_new, aux_new, g_new = eval_full(x_new)
        f_new_v = f_new.item()
        g_new_v = g_new
        gtd_new = torch.dot(g_new_v, d).item()
        ls_iter += 1

        if not math.isfinite(f_new_v) or f_new_v > f_val + c1 * t * gtd or f_new_v >= bracket_f[low_pos]:
            # Armijo failed or not better than the current low: update the high end.
            bracket[high_pos] = t
            bracket_f[high_pos] = f_new_v
            bracket_g[high_pos] = g_new_v.clone()
            bracket_aux[high_pos] = aux_new
            bracket_x[high_pos] = x_new
            bracket_gtd[high_pos] = gtd_new
            low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[1] else (1, 0)
        else:
            if abs(gtd_new) <= -c2 * gtd:
                # Strong Wolfe conditions satisfied.
                return x_new, f_new, aux_new, g_new_v, t, x_first, first_unmoved, True
            if gtd_new * (bracket[high_pos] - bracket[low_pos]) >= 0:
                # Old high becomes new low.
                bracket[high_pos] = bracket[low_pos]
                bracket_f[high_pos] = bracket_f[low_pos]
                bracket_g[high_pos] = bracket_g[low_pos]
                bracket_aux[high_pos] = bracket_aux[low_pos]
                bracket_x[high_pos] = bracket_x[low_pos]
                bracket_gtd[high_pos] = bracket_gtd[low_pos]
            # New point becomes the new low.
            bracket[low_pos] = t
            bracket_f[low_pos] = f_new_v
            bracket_g[low_pos] = g_new_v
            bracket_aux[low_pos] = aux_new
            bracket_x[low_pos] = x_new
            bracket_gtd[low_pos] = gtd_new

    # Return the best Armijo-satisfying point found (strong Wolfe may not have been met).
    t = bracket[low_pos]
    x_new = bracket_x[low_pos]
    f_new_v = bracket_f[low_pos]
    g_new_v = bracket_g[low_pos]
    aux_new = bracket_aux[low_pos]
    f_new = torch.tensor(f_new_v, dtype=g.dtype, device=g.device)
    search_ok = t > 0 and math.isfinite(f_new_v) and f_new_v <= f_val + c1 * t * gtd
    return x_new, f_new, aux_new, g_new_v, t, x_first, first_unmoved, search_ok
