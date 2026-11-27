"""L-BFGS with a cheap Armijo backtracking line search."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

from ._common import Callback, IterationInfo, OptimizeResult, check_x0, detach_tree, retract, split_aux
from ._manifolds import Manifold

__all__ = ["lbfgs_armijo"]


def lbfgs_armijo(
    fun: Callable,
    x0: torch.Tensor,
    *,
    has_aux: bool = False,
    manifold: Manifold | None = None,
    value_fun: Callable | None = None,
    max_iter: int = 100,
    lr: float = 1.0,
    history_size: int = 10,
    c1: float = 1e-4,
    max_backtracks: int = 10,
    max_step: float = 1.0,
    step_tol: float = 1e-5,
    callback: Callback | None = None,
) -> OptimizeResult:
    """Minimise a scalar function with L-BFGS and an Armijo backtracking line search.

    Built for expensive, deterministic objectives, where every evaluation counts. Each
    iteration first tries the full step: ``lr`` times the L-BFGS direction (scaled by
    ``min(1, 1 / ||g||_1)`` while there is no curvature history, at the start or after a
    reset), capped so that no coordinate moves more than ``max_step``. If the sufficient-decrease condition fails,
    the step is shrunk by quadratic interpolation. Each trial point is evaluated with its
    autograd graph kept, and the backward pass runs only once a point is accepted: a
    typical iteration costs one value-and-gradient evaluation, a rejected trial costs a
    forward pass only, and no point is evaluated twice. Only the latest trial's graph is
    held at any time.

    Curvature pairs with ``s.y <= 0`` are skipped (Armijo does not guarantee positive
    curvature). If no decrease is found the memory is reset and the iterate does not move;
    the run then retries once with steepest descent. It stops after two failures in a row,
    after one failure of a steepest-descent step (a retry would repeat it exactly), or once
    a step is below ``step_tol`` (max-norm). A failure counts as convergence when the loss
    did not measurably go up (by more than ``8 * eps * |loss|`` in its dtype) and either
    the first trial step was already below ``step_tol`` or its predicted decrease was below
    that rounding noise: near the optimum of a float32 objective, the line search otherwise
    keeps failing on noise. A trial point with a non-finite value never passes the line
    search, and a non-finite value at ``x0`` stops the run at once.

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
        value_fun (Callable, optional): a cheaper ``value_fun(x) -> loss`` for the
            backtracking evaluations of the line search, run under
            :class:`torch.no_grad`. The gradient at an accepted backtracked point then needs
            one more call to ``fun``. With the default, the backtracking evaluates ``fun``
            itself and records its graph for the deferred backward pass; if ``fun`` is much
            cheaper without a graph, pass ``value_fun=fun`` to backtrack under
            :class:`torch.no_grad` instead. Default: ``None``
        max_iter (int, optional): maximum number of iterations. Default: ``100``
        lr (float, optional): initial step length along the L-BFGS direction.
            Default: ``1.0``
        history_size (int, optional): number of curvature pairs kept. Default: ``10``
        c1 (float, optional): Armijo sufficient-decrease constant. Default: ``1e-4``
        max_backtracks (int, optional): maximum number of step reductions in one line
            search. Default: ``10``
        max_step (float, optional): largest change of any coordinate in one step (in the
            units of ``x``, or of the tangent vector). Default: ``1.0``
        step_tol (float, optional): stop, as converged, once a step's max-norm is below
            this. Default: ``1e-5``
        callback (Callable, optional): called with an :class:`piazzi.IterationInfo` after
            every iteration, accepted or not; returning ``True`` stops the run.
            Default: ``None``

    Returns:
        OptimizeResult: the solution ``x`` (shaped like ``x0``), the final loss ``fun`` and
        ``aux``, ``n_iter``, the counts ``n_fun_grad`` (value and gradient) and ``n_fun``
        (value only), and ``converged`` and ``message``.

    Example::

        >>> def rosenbrock(x):
        ...     return (100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2).sum()
        >>> res = piazzi.lbfgs_armijo(rosenbrock, torch.tensor([-1.2, 1.0], dtype=torch.float64))
        >>> res.x, res.converged
        (tensor([1.0000, 1.0000], dtype=torch.float64), True)
    """
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
            loss = value_fun(xv.reshape(shape))
        if isinstance(loss, tuple):  # value_fun=fun with has_aux: drop the aux
            loss = loss[0]
        counts["f"] += 1
        return loss.detach()

    def armijo_ok(fn, bound):
        # A non-finite trial value (NaN, or -inf, which would pass `<=`) is never accepted.
        return math.isfinite(fn) and fn <= bound

    def result(n_iter, converged, message):
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
        return result(0, False, "objective is not finite at x0")
    if g.numel() == 0:  # a manifold with no degrees of freedom (e.g. Subset with every entry fixed)
        return result(0, True, "no free parameters")
    S: list[torch.Tensor] = []
    Y: list[torch.Tensor] = []
    fails = 0
    eps = torch.finfo(g.dtype).eps
    for it in range(max_iter):
        f_val = f.item()
        if g.abs().max().item() == 0:  # stationary: nothing to do, and no direction to search
            return result(it, True, "gradient is zero")
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
        pred, step0 = -t * gtd, t * d.abs().max().item()  # predicted decrease and size of the first trial
        x_new = retract(manifold, x, t * d)
        f_new, aux_new, pending = eval_trial(x_new)
        n_bt = 0
        while not armijo_ok(f_new.item(), f_val + c1 * t * gtd) and n_bt < max_backtracks:
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
        if not armijo_ok(f_new.item(), f_val + c1 * t * gtd):  # no sufficient decrease found
            reject(pending)
            pending = None
            S.clear()
            Y.clear()
            fails += 1
            stop = callback is not None and callback(
                IterationInfo(it, x.reshape(shape), f, f, aux, False, counts["fg"], counts["f"])
            )
            if fails >= 2 or not used_memory:
                noise = 8 * torch.finfo(f.dtype).eps * abs(f_val)
                if math.isfinite(f_new.item()) and f_new.item() <= f_val + noise:
                    if step0 < step_tol:
                        return result(it + 1, True, "step below step_tol")
                    if pred <= noise:
                        return result(it + 1, True, "predicted decrease below rounding noise")
                message = "line search failed twice in a row" if fails >= 2 else "line search failed"
                return result(it + 1, False, message)
            if stop:
                return result(it + 1, False, "stopped by callback")
            continue
        fails = 0
        if pending is None:  # accepted on a value_fun evaluation
            f_new, aux_new, g_new = eval_full(x_new)
        else:
            g_new = accept(pending)
            pending = None
        if not torch.isfinite(g_new).all():  # the value decreased, but the gradient is unusable
            f_start = f
            x, f, aux = x_new, f_new, aux_new
            if callback is not None:
                callback(IterationInfo(it, x.reshape(shape), f, f_start, aux, True, counts["fg"], counts["f"]))
            return result(it + 1, False, "gradient is not finite")
        s_vec, y_vec = t * d, g_new - g  # tangent step (equals x_new - x without a manifold)
        # Keep the pair only with positive curvature, relative to the scale of s and y.
        sy = torch.dot(s_vec, y_vec).item()
        if sy > eps * (torch.linalg.vector_norm(s_vec) * torch.linalg.vector_norm(y_vec)).item():
            S.append(s_vec)
            Y.append(y_vec)
            if len(S) > history_size:
                S.pop(0)
                Y.pop(0)
        step = s_vec.abs().max().item()
        f_start = f
        x, f, aux, g = x_new, f_new, aux_new, g_new
        stop = callback is not None and callback(
            IterationInfo(it, x.reshape(shape), f, f_start, aux, True, counts["fg"], counts["f"])
        )
        # An unscaled steepest-descent step (no curvature pairs yet) says nothing about
        # convergence: with a tiny gradient it is tiny too.
        if used_memory and step < step_tol:
            return result(it + 1, True, "step below step_tol")
        if stop:
            return result(it + 1, False, "stopped by callback")
    return result(max_iter, False, "max_iter reached")
