"""Line search algorithms shared by gradient-based solvers.

Provides:

- :func:`cubic_interp`: cubic Hermite minimiser used by :func:`wolfe_search`.
- :func:`armijo_search`: cheap Armijo backtracking (one forward pass per rejected trial).
- :func:`wolfe_search`: strong Wolfe bracketing and zoom with cubic interpolation.
- :func:`hager_zhang_search`: the Hager-Zhang search of CG_DESCENT, with approximate Wolfe conditions.

The search routines accept pre-wired ``eval_*`` callables from the caller so that
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

from ._common import unmoved, within

__all__: list[str] = []  # private module; nothing exported directly

# :func:`hager_zhang_search` is a port of LineSearches.jl's ``HagerZhang`` (src/hagerzhang.jl), which
# is licensed under the MIT licence:
#
# Copyright (c) 2012: John Myles White and other contributors (Optim.jl).
# Copyright (c) 2016: Asbjørn Nilsen Riseth and other contributors (LineSearches.jl).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
# OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


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


def _bracket_resolved(manifold, retract, x: torch.Tensor, d: torch.Tensor, a: float, b: float) -> bool:
    """Whether the points at steps ``a`` and ``b`` along ``d`` cannot be told apart at the resolution of ``x``.

    Compared entry by entry, each at its own scale (see :func:`piazzi._common.within`), so that a
    problem whose parameters are all tiny, or huge, is judged like any other.
    """
    return within(manifold, retract(manifold, x, a * d), retract(manifold, x, b * d), torch.finfo(x.dtype).eps)


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
        # Bracket collapsed below the resolution of x.
        if _bracket_resolved(manifold, retract, x, d, bracket[0], bracket[1]):
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


class _SearchEnded(Exception):
    """Ends the Hager-Zhang search early; the search then falls back to its best point."""


class _BudgetExhausted(_SearchEnded):
    """Raised by the Hager-Zhang search's evaluator once its trial budget is spent."""


class _BrokenBracket(_SearchEnded):
    """Raised when a bracket end that must descend does not: the gradient is inconsistent with the values."""


def hager_zhang_search(
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
    epsilon: float,
    max_ls: int,
    eval_full,
    retract,
):
    """Hager-Zhang line search: bracketing, secant steps and bisection with approximate Wolfe conditions.

    Implements Algorithm 851 (CG_DESCENT) of Hager and Zhang (2006), in the form of
    ``LineSearches.jl::HagerZhang`` (MIT licence, itself from Optim.jl): bracket (stages B0-B3),
    then secant steps (S1-S4) with the bracket update (U0-U3) and bisection. A trial point is
    accepted when it satisfies the Wolfe conditions with ``delta = c1`` and ``sigma = c2``, or the
    approximate Wolfe conditions ``(2 c1 - 1) g.d <= g_t.d <= c2 g.d`` with ``f_t <= f + epsilon |f|``,
    which stay meaningful where the loss differences are lost in rounding noise. Every trial
    needs the gradient, so every evaluation is counted by ``eval_full``.

    As in the references, a point that satisfies the approximate conditions may be higher than
    the starting value, by at most ``epsilon |f|``; ``epsilon=0`` makes the search monotone.

    If the gradient is inconsistent with the values, so that a bracket end that must descend does
    not, the search ends like a spent budget: it returns its best acceptable point, or fails.

    Difference from the Julia code: the budget is a number of evaluations (``max_ls`` after the
    first), enforced exactly. When the budget is spent, or no finite point is found, the best
    evaluated point whose loss is at most ``f + epsilon |f|`` and which satisfies Armijo (or the
    Wolfe conditions) is returned, else ``search_ok=False``. A non-finite value or slope shrinks
    the step (first trial) or bisects towards the last good step (expansion).

    Args:
        x (Tensor): current flat iterate.
        f_val (float): objective value at ``x``.
        g (Tensor): gradient at ``x`` (flat, tangent coordinates).
        aux: auxiliary output at ``x`` (returned if the search ends back at ``x``).
        d (Tensor): search direction (flat, tangent coordinates).
        t_init (float): initial step length.
        t_max (float): largest step length (``max_step`` over the largest entry of ``d``).
        gtd (float): directional derivative ``g . d`` at ``x`` (must be negative).
        used_memory (bool): whether the direction came from L-BFGS memory (vs steepest descent).
        manifold: manifold object or ``None``.
        c1 (float): sufficient-decrease constant ``delta`` (``0 < c1 < min(c2, 1/2)``).
        c2 (float): curvature constant ``sigma`` (``c1 < c2 < 1``).
        epsilon (float): relative tolerance of the approximate Wolfe conditions.
        max_ls (int): number of trial points after the first one.
        eval_full: ``eval_full(xv)`` -> ``(f, aux, g)``; counts one ``n_fun_grad``.
        retract: ``retract(manifold, x, delta)`` -> ``x_new``.

    Returns:
        tuple: ``(x_new, f_new, aux_new, g_new, t, x_first, first_unmoved, search_ok)``
    """
    rho, gamma, psi3 = 5.0, 2.0 / 3.0, 0.1  # HZ's expansion factor, contraction test and shrink factor
    dtype_eps = torch.finfo(g.dtype).eps
    phi_lim = f_val + epsilon * abs(f_val)

    # Every evaluated point; index 0 is the start.
    alphas, values, slopes = [0.0], [f_val], [gtd]
    pts_x, pts_f, pts_aux, pts_g = [x], [torch.tensor(f_val, dtype=g.dtype, device=g.device)], [aux], [g]

    def evaluate(t: float) -> int:
        if len(alphas) - 1 > max_ls:
            raise _BudgetExhausted
        x_t = retract(manifold, x, t * d)
        f_t, aux_t, g_t = eval_full(x_t)
        alphas.append(t)
        values.append(f_t.item())
        slopes.append(torch.dot(g_t, d).item())
        pts_x.append(x_t)
        pts_f.append(f_t)
        pts_aux.append(aux_t)
        pts_g.append(g_t)
        return len(alphas) - 1

    def finite(i: int) -> bool:
        return math.isfinite(values[i]) and math.isfinite(slopes[i])

    def collapsed(a: float, b: float) -> bool:  # bracket below the resolution of t, or of x (entry by entry)
        return b - a <= 2.2e-16 * abs(b) or _bracket_resolved(manifold, retract, x, d, a, b)

    def satisfies_wolfe(i: int) -> bool:
        t, f_t, df_t = alphas[i], values[i], slopes[i]
        exact = c1 * gtd >= (f_t - f_val) / t and df_t >= c2 * gtd
        approx = (2 * c1 - 1) * gtd >= df_t >= c2 * gtd and f_t <= phi_lim
        return exact or approx

    def acceptable(i: int) -> bool:  # a finite step with enough decrease (up to epsilon for approximate Wolfe)
        t = alphas[i]
        return (
            i > 0
            and t > 0
            and finite(i)
            and values[i] <= phi_lim
            and (values[i] <= f_val + c1 * t * gtd or satisfies_wolfe(i))
        )

    def result(i: int):
        ok = acceptable(i)
        return pts_x[i], pts_f[i], pts_aux[i], pts_g[i], alphas[i], x_first, first_unmoved, ok

    def secant(ia: int, ib: int) -> float:
        den = slopes[ib] - slopes[ia]
        if den == 0.0 or not math.isfinite(den):
            return math.nan
        return (alphas[ia] * slopes[ib] - alphas[ib] * slopes[ia]) / den

    def bisect(ia: int, ib: int):  # HZ, stage U3 with theta = 1/2: returns (ia, ib, wolfe found at ib)
        a, b = alphas[ia], alphas[ib]
        while not collapsed(a, b):
            mid = 0.5 * (a + b)
            im = evaluate(mid)
            if satisfies_wolfe(im):
                return ia, im, True
            if slopes[im] >= 0.0:
                return ia, im, False
            if values[im] <= phi_lim and math.isfinite(slopes[im]):
                a, ia = mid, im
            else:
                b, ib = mid, im
        return ia, ib, False

    def update(ia: int, ib: int, ic: int):  # HZ, stages U0-U3
        a, b, c = alphas[ia], alphas[ib], alphas[ic]
        if not a < c < b:  # only a point strictly inside can narrow the bracket
            return ia, ib, False
        if slopes[ic] >= 0.0:
            return ia, ic, False
        if values[ic] <= phi_lim and math.isfinite(slopes[ic]):
            return ic, ib, False
        return bisect(ia, ic)

    def secant2(ia: int, ib: int):  # HZ, stages S1-S4: returns (wolfe found, iA, iB)
        if not slopes[ia] < 0.0:  # the bracket invariant: inaccurate gradients can break it
            raise _BrokenBracket
        a, b = alphas[ia], alphas[ib]
        c = secant(ia, ib)
        if not a < c < b:  # not finite, or on an endpoint: bisect instead
            c = 0.5 * (a + b)
        ic = evaluate(c)
        if satisfies_wolfe(ic):
            return True, ic, ic
        i_a, i_b, found = update(ia, ib, ic)
        if found:
            return True, i_b, i_b
        if i_b == ic:  # b was updated: also update a
            c = secant(ib, i_b)
        elif i_a == ic:
            c = secant(ia, i_a)
        if (i_a == ic or i_b == ic) and alphas[i_a] < c < alphas[i_b]:
            ic = evaluate(c)
            if satisfies_wolfe(ic):
                return True, ic, ic
            i_a, i_b, found = update(i_a, i_b, ic)
            if found:
                return True, i_b, i_b
        return False, i_a, i_b

    x_first, first_unmoved = x, False
    try:
        # ---- first trial: shrink while the value or slope is not finite ----
        c = min(t_init, t_max)
        ic = evaluate(c)
        x_first = pts_x[ic]
        first_unmoved = used_memory and math.isfinite(values[ic]) and unmoved(manifold, x, x_first)
        for _ in range(math.ceil(-math.log2(dtype_eps))):
            if finite(ic):
                break
            c *= psi3
            ic = evaluate(c)
        if not finite(ic):
            raise _BudgetExhausted
        if satisfies_wolfe(ic):
            return result(ic)

        # ---- bracketing (HZ, stages B0-B3) ----
        ia, ib = 0, ic
        alpha_max = t_max
        while True:
            if slopes[ic] >= 0.0:  # past the minimum: b found, look back for a
                ib = ic
                ia = next((i for i in range(ib - 1, -1, -1) if finite(i) and values[i] <= phi_lim), 0)
                break
            if values[ic] > phi_lim:  # downhill slope but higher value: crested over a peak
                ia, ib, found = bisect(0, ic)
                if found:
                    return result(ib)
                break
            # Still going downhill: expand. c_old is a viable step if bracketing fails.
            c_old, i_old = alphas[ic], ic
            if math.nextafter(c_old, math.inf) >= alpha_max:
                return result(i_old)
            c = min(c_old * rho, alpha_max)
            ic = evaluate(c)
            for _ in range(math.ceil(-math.log2(dtype_eps))):
                if finite(ic) or c <= math.nextafter(c_old, math.inf):
                    break
                alpha_max = c  # steps from c on are assumed to be infeasible
                c = 0.5 * (c_old + c)
                ic = evaluate(c)
            if not finite(ic):
                return result(i_old)
            if satisfies_wolfe(ic):
                return result(ic)

        # ---- secant steps and bisection inside the bracket (HZ, stages L1-L3) ----
        while True:
            a, b = alphas[ia], alphas[ib]
            if collapsed(a, b):
                return result(ia)
            found, i_a, i_b = secant2(ia, ib)
            if found:
                return result(i_a)
            width = alphas[i_b] - alphas[i_a]
            if width < gamma * (b - a):
                ia, ib = i_a, i_b
            else:  # the secant steps converge too slowly: bisect
                mid = 0.5 * (alphas[i_a] + alphas[i_b])
                im = evaluate(mid)
                if satisfies_wolfe(im):
                    return result(im)
                ia, ib, found = update(i_a, i_b, im)
                if found:
                    return result(ib)
    except _SearchEnded:
        pass
    # Budget spent, or no finite point: the best acceptable point, if any.
    best = min((i for i in range(1, len(alphas)) if acceptable(i)), key=lambda i: values[i], default=None)
    return result(best if best is not None else min(1, len(alphas) - 1))
