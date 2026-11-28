"""L-BFGS with Armijo backtracking or strong Wolfe line search."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal

import torch

from ._common import (
    RESOLUTION_ULPS,
    Callback,
    IterationInfo,
    OptimizeResult,
    check_x0,
    detach_tree,
    retract,
    split_aux,
    within,
)
from ._line_search import armijo_search, wolfe_search
from ._manifolds import Manifold
from ._progress import with_progress

__all__ = ["lbfgs"]

# How far (in units in the last place of each entry's scale) the full step of a backtracked
# L-BFGS step may move x for that step to count towards step_tol: 16 times the resolution of x,
# a 6x margin over the largest full step seen at a float32 noise floor (171 ulps), and far
# below sqrt(eps) (~2900 ulps in float32), which admitted full steps of 0.35 at |x| ~ 1000.
NEAR_ULPS = 16 * RESOLUTION_ULPS


def lbfgs(
    fun: Callable,
    x0: torch.Tensor,
    *,
    has_aux: bool = False,
    manifold: Manifold | None = None,
    value_fun: Callable | None = None,
    max_iter: int = 100,
    lr: float = 1.0,
    history_size: int = 10,
    line_search: Literal["armijo", "strong_wolfe"] = "armijo",
    c1: float = 1e-4,
    c2: float = 0.9,
    max_backtracks: int = 10,
    max_step: float = 1.0,
    step_tol: float = 1e-5,
    callback: Callback | None = None,
    verbose: bool | int = False,
) -> OptimizeResult:
    """Minimise a scalar function with L-BFGS and a line search.

    Built for expensive, deterministic objectives, where every evaluation counts.

    **Armijo backtracking** (``line_search="armijo"``, the default): each iteration
    tries the full step (``lr`` times the L-BFGS direction, scaled by
    ``min(1, 1 / ||g||_1)`` while there is no curvature history, at the start or after a
    reset), capped so that no coordinate moves more than ``max_step``. If the
    sufficient-decrease condition (Armijo, parameter ``c1``) fails, the step is shrunk by
    quadratic interpolation up to ``max_backtracks`` times. Each trial point is evaluated
    with its autograd graph kept, and the backward pass runs only once a point is accepted:
    a typical iteration costs one value-and-gradient evaluation, a rejected trial costs a
    forward pass only, and no point is evaluated twice.

    **Strong Wolfe** (``line_search="strong_wolfe"``): a bracketing phase followed by a
    zoom phase with cubic interpolation, following Nocedal and Wright Algorithm 3.5/3.6
    (ported from ``torch.optim.LBFGS``). Every trial point requires a gradient evaluation
    (the curvature condition needs it), so rejected trials cost one ``n_fun_grad`` count
    instead of one ``n_fun`` count. ``value_fun`` is ignored when
    ``line_search="strong_wolfe"``. A step that satisfies the curvature condition (parameter
    ``c2``) has ``sᵀy > 0``, so its curvature pair is kept; if the search ends on the Armijo
    condition alone (its trial budget is spent, or the step reached ``max_step``), the pair is
    kept only if ``sᵀy > 0``, as with Armijo. On a failed search the memory is still
    reset and the solver retries with steepest descent, exactly as with Armijo.

    **Convergence** (both line searches): curvature pairs with ``s·y ≤ 0`` are skipped
    (the strong Wolfe curvature condition rules them out when it holds). If no sufficient
    decrease is found, the memory is reset and the iterate does not move; the run retries
    once with steepest descent. It stops after two failures in a row, after one failure of a
    steepest-descent step, or once a step is below ``step_tol`` (max-norm). The stop counts
    as convergence when the last failed search from the L-BFGS memory showed a minimiser to
    working precision: its step left ``x`` unmoved (every entry within 64 units in the last
    place of its scale, ``|x|`` or the manifold's :meth:`~piazzi.Manifold.entry_scale`, at a
    finite value), or the loss did not measurably go up (by more than ``8 * eps * |loss|``)
    while that step was below ``step_tol`` or predicted a decrease below rounding noise. A
    trial point with a non-finite value never passes the line search, and a non-finite value
    at ``x0`` stops the run at once.

    Args:
        fun (Callable): ``fun(x) -> loss``, or ``(loss, aux)`` with ``has_aux=True``, where
            ``loss`` is a scalar tensor differentiable with respect to ``x`` and ``x`` has
            the shape of ``x0``.
        x0 (Tensor): starting point, of any shape (it is flattened internally).

    Keyword Args:
        has_aux (bool, optional): whether ``fun`` also returns an auxiliary output (any tree
            of tensors), returned detached in ``result.aux`` and ``info.aux``.
            Default: ``False``
        manifold (Manifold, optional): keeps ``x`` on a manifold such as
            :class:`piazzi.SE3`. The gradient is then taken with respect to the tangent
            vector ``delta`` of ``fun(manifold.plus(x, delta))`` at ``delta = 0``, steps are
            applied with ``manifold.plus``, and the curvature pairs are kept in tangent
            coordinates without vector transport (as in Ceres Solver's line-search
            solver). ``max_step`` and ``step_tol`` apply to the tangent step.
            Default: ``None``
        value_fun (Callable, optional): a cheaper ``value_fun(x) -> loss`` for the Armijo
            backtracking evaluations, run under :class:`torch.no_grad`. The gradient at an
            accepted backtracked point then needs one more call to ``fun``. With the default,
            the backtracking evaluates ``fun`` itself and records its graph for the deferred
            backward pass; if ``fun`` is much cheaper without a graph, pass
            ``value_fun=fun`` to backtrack under :class:`torch.no_grad` instead. Ignored
            when ``line_search="strong_wolfe"``, which always needs the gradient.
            Default: ``None``
        max_iter (int, optional): maximum number of iterations. Default: ``100``
        lr (float, optional): initial step length along the L-BFGS direction.
            Default: ``1.0``
        history_size (int, optional): number of curvature pairs kept. Default: ``10``
        line_search (str, optional): ``"armijo"`` for cheap Armijo backtracking (one
            forward pass per rejected trial) or ``"strong_wolfe"`` for the bracketing and
            zoom method with cubic interpolation (one value-and-gradient evaluation per
            trial). Default: ``"armijo"``
        c1 (float, optional): sufficient-decrease (Armijo) constant, used by both line
            searches. Default: ``1e-4``
        c2 (float, optional): curvature constant for the strong Wolfe condition; must
            satisfy ``c1 < c2 < 1`` (checked with ``line_search="strong_wolfe"``). Ignored
            when ``line_search="armijo"``.
            Default: ``0.9``
        max_backtracks (int, optional): maximum number of trial points in one line search
            after the first one (step reductions for Armijo, bracketing and zoom trials for
            strong Wolfe), so a search makes at most ``max_backtracks + 1`` evaluations.
            Default: ``10``
        max_step (float, optional): largest change of any coordinate in one step (in the
            units of ``x``, or of the tangent vector). Default: ``1.0``
        step_tol (float, optional): stop, as converged, once an accepted step from the
            L-BFGS memory has a max-norm below this. A step shortened by backtracking counts
            only if the full step was below it too, or if the step made no measurable progress
            (at a minimiser the line search may accept a step that changes nothing), or if the
            full step barely moved ``x`` (1024 units in the last place of each entry's scale),
            as at a float32 noise floor.
            Default: ``1e-5``
        callback (Callable, optional): called with an :class:`piazzi.IterationInfo` after
            every iteration, accepted or not; returning ``True`` stops the run.
            Default: ``None``
        verbose (bool or int, optional): print a table of the run's progress with
            :func:`piazzi.print_progress` before calling ``callback``: ``True`` prints every
            iteration and an integer ``k`` every ``k``-th one. Default: ``False``

    Returns:
        OptimizeResult: the solution ``x`` (shaped like ``x0``), the final loss ``fun`` and
        ``aux``, ``n_iter``, the counts ``n_fun_grad`` (value and gradient) and ``n_fun``
        (value only, Armijo backtracking only), and ``converged`` and ``message``.

    See Also:
        - :func:`piazzi.minimize`: the high-level interface, which runs this method by default
          and passes its ``options`` here.

    Example::

        >>> def rosenbrock(x):
        ...     return (100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2).sum()
        >>> res = piazzi.lbfgs(rosenbrock, torch.tensor([-1.2, 1.0], dtype=torch.float64))
        >>> res.x, res.converged
        (tensor([1.0000, 1.0000], dtype=torch.float64), True)
        >>> res_sw = piazzi.lbfgs(
        ...     rosenbrock, torch.tensor([-1.2, 1.0], dtype=torch.float64),
        ...     line_search="strong_wolfe",
        ... )
        >>> res_sw.x, res_sw.converged
        (tensor([1.0000, 1.0000], dtype=torch.float64), True)
    """
    if line_search not in ("armijo", "strong_wolfe"):
        raise ValueError(f"line_search must be 'armijo' or 'strong_wolfe', got {line_search!r}")
    if line_search == "strong_wolfe" and not 0 < c1 < c2 < 1:
        raise ValueError(f"strong Wolfe needs 0 < c1 < c2 < 1, got c1={c1!r}, c2={c2!r}")
    callback = with_progress(callback, verbose)
    x, shape = check_x0(x0)
    counts = {"fg": 0, "f": 0}

    def leaf(xv):
        # The variable to differentiate with respect to, and the point to pass to fun: x itself,
        # or, on a manifold, the tangent vector delta = 0 and the point plus(x, delta).
        if manifold is None:
            var = xv.detach().reshape(shape).requires_grad_(True)
            return var, var
        var = xv.new_zeros(manifold.tangent_size(xv)).requires_grad_(True)
        return var, manifold.plus(xv.detach(), var).reshape(shape)

    def eval_full(xv):
        with torch.enable_grad():
            var, xg = leaf(xv)
            loss, aux = split_aux(fun(xg), has_aux)
            (grad,) = torch.autograd.grad(loss, var)
        counts["fg"] += 1
        return loss.detach(), detach_tree(aux), grad.detach().reshape(-1)

    def eval_trial(xv):
        """Value and aux at ``xv``, keeping the graph so that ``accept`` can add the gradient."""
        with torch.enable_grad():
            var, xg = leaf(xv)
            loss, aux = split_aux(fun(xg), has_aux)
        return loss.detach(), detach_tree(aux), (loss, var)

    def accept(pending):
        loss, var = pending
        (grad,) = torch.autograd.grad(loss, var)
        counts["fg"] += 1
        return grad.detach().reshape(-1)

    def reject(pending):
        if pending is not None:  # a value_fun evaluation was counted when it was made
            counts["f"] += 1

    def eval_value(xv):
        with torch.no_grad():
            assert value_fun is not None  # only called with a value_fun
            loss = value_fun(xv.reshape(shape))
        if isinstance(loss, tuple):  # value_fun=fun with has_aux: drop the aux
            loss = loss[0]
        counts["f"] += 1
        return loss.detach()

    def mk_result(n_iter, converged, message):
        return OptimizeResult(
            x=x.reshape(shape),
            fun=f,
            aux=aux,
            n_iter=n_iter,
            n_fun_grad=counts["fg"],
            n_fun=counts["f"],
            converged=converged,
            message=message,
        )

    f, aux, g = eval_full(x)
    if not torch.isfinite(f):
        return mk_result(0, False, "objective is not finite at x0")
    if g.numel() == 0:  # a manifold with no degrees of freedom (e.g. Subset with every entry fixed)
        return mk_result(0, True, "no free parameters")
    S: list[torch.Tensor] = []
    Y: list[torch.Tensor] = []
    fails, stalled, x_stalled = 0, None, None
    eps = torch.finfo(g.dtype).eps
    for it in range(max_iter):
        f_val = f.item()
        if g.abs().max().item() == 0:  # stationary: nothing to do, and no direction to search
            return mk_result(it, True, "gradient is zero")
        # Two-loop recursion.
        qv = -g
        alphas = []
        for s_, y_ in reversed(list(zip(S, Y, strict=True))):
            rho = 1.0 / torch.dot(y_, s_)
            a = rho * torch.dot(s_, qv)
            qv = qv - a * y_
            alphas.append((rho, a))
        if S:
            qv = qv * (torch.dot(S[-1], Y[-1]) / torch.dot(Y[-1], Y[-1]))
        for (s_, y_), (rho, a) in zip(zip(S, Y, strict=True), reversed(alphas), strict=True):
            b = rho * torch.dot(y_, qv)
            qv = qv + s_ * (a - b)
        d = qv
        gtd = torch.dot(g, d).item()
        # Not a descent direction (relative to the scale of g and d, so that objectives with
        # tiny gradients still count as descending): restart from steepest descent.
        if not math.isfinite(gtd) or gtd >= -eps * (torch.linalg.vector_norm(g) * torch.linalg.vector_norm(d)).item():
            S.clear()
            Y.clear()
            d, gtd = -g, -torch.dot(g, g).item()
        t = lr if S else lr * min(1.0, 1.0 / max(g.abs().sum().item(), 1e-12))
        t = min(t, max_step / max(d.abs().max().item(), 1e-12))
        used_memory = bool(S)
        pred, step0 = -t * gtd, t * d.abs().max().item()

        # ---- line search ----
        if line_search == "armijo":
            x_new, f_new, aux_new, g_new, t, x_first, first_unmoved, search_ok = armijo_search(
                x,
                f_val,
                d,
                t,
                gtd,
                used_memory,
                manifold,
                c1,
                max_backtracks,
                eval_trial,
                eval_value,
                accept,
                reject,
                retract,
                value_fun,
            )
        else:
            x_new, f_new, aux_new, g_new, t, x_first, first_unmoved, search_ok = wolfe_search(
                x,
                f_val,
                g,
                aux,
                d,
                t,
                max_step / max(d.abs().max().item(), 1e-12),
                gtd,
                used_memory,
                manifold,
                c1,
                c2,
                max_backtracks,
                eval_full,
                retract,
            )

        if not search_ok:  # no sufficient decrease found
            S.clear()
            Y.clear()
            fails += 1
            stop = callback is not None and callback(
                IterationInfo(it, x.reshape(shape), f, f, aux, False, counts["fg"], counts["f"])
            )
            # Whether this failure shows a minimiser to working precision. A failure from the
            # L-BFGS memory gives the verdict; steepest-descent attempts, whose unscaled step and
            # prediction say nothing about the distance to the minimiser, keep the verdict of the
            # last such failure, even across a steepest-descent step that succeeded on noise.
            noise = 8 * torch.finfo(f.dtype).eps * abs(f_val)
            flat = math.isfinite(f_new.item()) and f_new.item() <= f_val + noise
            verdict = None
            if first_unmoved:
                verdict = "step below the resolution of x"
            elif flat and step0 < step_tol:
                verdict = "step below step_tol"
            elif flat and pred <= noise:
                verdict = "predicted decrease below rounding noise"
            if used_memory or stalled is None:
                stalled, x_stalled = verdict, x
            if fails >= 2 or not used_memory:
                if stalled is not None:
                    return mk_result(it + 1, True, stalled)
                message = "line search failed twice in a row" if fails >= 2 else "line search failed"
                return mk_result(it + 1, False, message)
            if stop:
                return mk_result(it + 1, False, "stopped by callback")
            continue
        fails = 0
        # An earlier verdict holds across steepest-descent steps that succeed on noise at a noise
        # floor, not across real progress: an accepted step from memory clears it, and so does
        # leaving the neighbourhood where it was given. A minimiser is only determined to about
        # sqrt(eps) (the loss is flat to second order), so within that distance (per entry, at
        # its scale) every point is a minimiser to working precision. Near a zero residual the
        # loss's noise is absolute, so the size of the decrease cannot tell noise from progress.
        if (
            stalled is not None
            and x_stalled is not None
            and (used_memory or not within(manifold, x_stalled, x_new, torch.finfo(x.dtype).eps ** 0.5))
        ):
            stalled = None
        if g_new is None:  # Armijo accepted on a value_fun evaluation: need a full eval for g
            f_new, aux_new, g_new = eval_full(x_new)
        if not torch.isfinite(g_new).all():  # the value decreased, but the gradient is unusable
            f_start = f
            x, f, aux = x_new, f_new, aux_new
            if callback is not None:
                callback(IterationInfo(it, x.reshape(shape), f, f_start, aux, True, counts["fg"], counts["f"]))
            return mk_result(it + 1, False, "gradient is not finite")
        s_vec, y_vec = t * d, g_new - g  # tangent step (equals x_new - x without a manifold)
        # Keep only pairs with positive curvature, relative to the scale of s and y. Armijo does
        # not guarantee it, nor does a strong Wolfe search that ended on Armijo alone (budget
        # spent, or step capped by max_step).
        sy = torch.dot(s_vec, y_vec).item()
        if sy > eps * (torch.linalg.vector_norm(s_vec) * torch.linalg.vector_norm(y_vec)).item():
            S.append(s_vec)
            Y.append(y_vec)
            if len(S) > history_size:
                S.pop(0)
                Y.pop(0)
        f_start, x_prev = f, x
        x, f, aux, g = x_new, f_new, aux_new, g_new
        stop = callback is not None and callback(
            IterationInfo(it, x.reshape(shape), f, f_start, aux, True, counts["fg"], counts["f"])
        )
        # An unscaled steepest-descent step (no curvature pairs yet) says nothing about
        # convergence: with a tiny gradient it is tiny too.
        # A backtracked step is short without the minimiser being any closer: it counts only if
        # the full step was short too, if the full step barely moved x (within NEAR_ULPS units
        # in the last place of each entry's scale: at a float32 noise floor near a zero residual,
        # where the loss's absolute noise makes backtracked steps succeed on noise, the full
        # steps measured 0.25 to 171 ulps), or if it made no measurable progress (at a minimiser
        # the Armijo bound can round to the loss itself, so a step that changes nothing passes).
        step = s_vec.abs().max().item()
        no_progress = f_start.item() - f.item() <= 8 * torch.finfo(f.dtype).eps * abs(f_start.item())
        full_near = within(manifold, x_prev, x_first, NEAR_ULPS * torch.finfo(x.dtype).eps)
        if used_memory and step < step_tol and (step0 < step_tol or full_near or no_progress):
            return mk_result(it + 1, True, "step below step_tol")
        if stop:
            return mk_result(it + 1, False, "stopped by callback")
    return mk_result(max_iter, False, "max_iter reached")
