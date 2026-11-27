"""L-BFGS with a cheap Armijo backtracking line search."""

from __future__ import annotations

from collections.abc import Callable

import torch

from ._common import Callback, IterationInfo, OptimizeResult, check_x0, detach_tree, split_aux

__all__ = ["lbfgs_armijo"]


def lbfgs_armijo(
    fun: Callable,
    x0: torch.Tensor,
    *,
    has_aux: bool = False,
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
    """Minimise ``fun`` with L-BFGS and an Armijo backtracking line search.

    Built for expensive, deterministic objectives with few parameters, where every
    evaluation counts. Each iteration first tries the full step (``lr`` times the L-BFGS
    direction, capped so that no coordinate moves more than ``max_step``), evaluating
    value and gradient there. That is usually accepted, so a typical iteration costs one
    evaluation. If the sufficient-decrease condition fails, the step is shrunk by
    quadratic interpolation using value-only evaluations, then one gradient is computed
    at the accepted point.

    Curvature pairs with ``s.y <= 0`` are skipped (Armijo does not guarantee positive
    curvature). If no decrease is found the memory is reset and the iterate does not
    move; two such failures in a row, or a step below ``step_tol`` (max-norm), stop the
    run: near the optimum the line search otherwise keeps failing on float32 noise. A trial
    point with a non-finite value never passes the line search, and a non-finite value at
    ``x0`` stops the run at once.

    Args:
        fun: ``fun(x) -> loss`` (or ``(loss, aux)`` with ``has_aux``), a scalar tensor
            differentiable with respect to ``x``. ``x`` has the shape of ``x0``.
        x0: starting point (any shape; it is flattened internally).
        value_fun: optional cheaper ``value_fun(x) -> loss`` for the value-only
            evaluations of the line search (run under ``torch.no_grad``). Defaults to ``fun``.
        lr: initial step length along the L-BFGS direction.
        callback: called with an :class:`IterationInfo` after every iteration; returning
            True stops the run.
    """
    x, shape = check_x0(x0)
    counts = {"fg": 0, "f": 0}

    def eval_full(xv):
        xg = xv.detach().reshape(shape).requires_grad_(True)
        with torch.enable_grad():
            loss, aux = split_aux(fun(xg), has_aux)
            (grad,) = torch.autograd.grad(loss, xg)
        counts["fg"] += 1
        return loss.detach(), detach_tree(aux), grad.detach().reshape(-1)

    def eval_value(xv):
        with torch.no_grad():
            if value_fun is not None:
                loss = value_fun(xv.reshape(shape))
            else:
                loss, _ = split_aux(fun(xv.reshape(shape)), has_aux)
        counts["f"] += 1
        return loss.detach()

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
    S: list[torch.Tensor] = []
    Y: list[torch.Tensor] = []
    fails = 0
    for it in range(max_iter):
        f_val = f.item()
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
        if gtd > -1e-12:  # not a descent direction: restart from steepest descent
            S.clear()
            Y.clear()
            d, gtd = -g, -torch.dot(g, g).item()
        t = lr if S else lr * min(1.0, 1.0 / max(g.abs().sum().item(), 1e-12))
        t = min(t, max_step / max(d.abs().max().item(), 1e-12))
        x_new = x + t * d
        f_new, aux_new, g_new = eval_full(x_new)
        n_bt = 0
        # Written as `not <=` so that a NaN trial value counts as insufficient decrease.
        while not f_new.item() <= f_val + c1 * t * gtd and n_bt < max_backtracks:
            denom = 2.0 * (f_new.item() - f_val - gtd * t)
            t_q = -gtd * t * t / denom if denom > 0 else 0.5 * t
            t = min(max(t_q, 0.1 * t), 0.5 * t)
            x_new = x + t * d
            f_new, g_new = eval_value(x_new), None
            n_bt += 1
        if not f_new.item() <= f_val + c1 * t * gtd:  # no sufficient decrease found
            S.clear()
            Y.clear()
            fails += 1
            stop = callback is not None and callback(IterationInfo(it, x.reshape(shape), f, f, aux, False))
            if fails >= 2:
                return result(it + 1, False, "line search failed twice in a row")
            if stop:
                return result(it + 1, False, "stopped by callback")
            continue
        fails = 0
        if g_new is None:
            f_new, aux_new, g_new = eval_full(x_new)
        s_vec, y_vec = x_new - x, g_new - g
        if torch.dot(s_vec, y_vec).item() > 1e-10:
            S.append(s_vec)
            Y.append(y_vec)
            if len(S) > history_size:
                S.pop(0)
                Y.pop(0)
        step = s_vec.abs().max().item()
        f_start = f
        x, f, aux, g = x_new, f_new, aux_new, g_new
        stop = callback is not None and callback(IterationInfo(it, x.reshape(shape), f, f_start, aux, True))
        if step < step_tol:
            return result(it + 1, True, "step below step_tol")
        if stop:
            return result(it + 1, False, "stopped by callback")
    return result(max_iter, False, "max_iter reached")
