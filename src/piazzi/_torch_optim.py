"""Run ``torch.optim`` optimisers with piazzi's calling convention."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

from ._common import Callback, IterationInfo, OptimizeResult, check_x0, detach_tree, split_aux

__all__ = ["torch_optimizer"]


def torch_optimizer(
    fun: Callable,
    x0: torch.Tensor,
    optimizer: type[torch.optim.Optimizer],
    *,
    has_aux: bool = False,
    max_iter: int = 100,
    max_eval: int | None = None,
    step_tol: float | None = None,
    callback: Callback | None = None,
    **optimizer_options,
) -> OptimizeResult:
    """Minimise ``fun`` with a ``torch.optim`` optimiser; see :func:`piazzi.minimize`.

    Each iteration is one ``optimizer.step(closure)``. Evaluations are memoised by point, so
    an optimiser that evaluates the same ``x`` twice (``torch.optim.LBFGS`` re-evaluates
    the point its line search accepted at the start of the next step) is charged once.
    ``torch.optim.LBFGS`` is run one L-BFGS iteration per step, with ``max_iter`` and
    ``max_eval`` counted over the whole run and its ``tolerance_grad`` and
    ``tolerance_change`` tests applied after every iteration, so it follows the same path
    and stops at the same point as a single upstream ``step`` with those settings.
    """
    x, shape = check_x0(x0)
    lbfgs = issubclass(optimizer, torch.optim.LBFGS)
    if lbfgs:
        max_eval = max_iter * 5 // 4 if max_eval is None else max_eval
        optimizer_options["max_iter"] = 1
    elif max_eval is not None:
        raise ValueError("max_eval applies to torch.optim.LBFGS only")
    param = x.reshape(shape).clone().requires_grad_(True)
    opt = optimizer([param], **optimizer_options)
    cache: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, object]] = []
    counts = {"fg": 0}

    def evaluate():
        """``(x, value, gradient, aux)`` at the current point, from the cache if possible."""
        xv = param.detach()
        for entry in cache:
            if torch.equal(entry[0], xv):
                return entry
        # fun gets a copy: the optimiser updates param in place, which would change any aux
        # (or anything else fun keeps) that is a view of x.
        xg = xv.clone().requires_grad_(True)
        with torch.enable_grad():
            loss, aux = split_aux(fun(xg), has_aux)
            (grad,) = torch.autograd.grad(loss, xg)
        entry = (xv.clone(), loss.detach(), grad.detach(), detach_tree(aux))
        cache.append(entry)
        counts["fg"] += 1
        return entry

    def closure():
        _, loss, grad, _ = evaluate()
        param.grad = grad.clone()  # the optimiser may modify it in place
        return loss

    def current():
        entry = evaluate()
        cache[:] = [entry]  # the optimisers never return to earlier points
        return entry[1:]

    def result(n_iter, converged, message):
        return OptimizeResult(
            x=param.detach().clone(),
            fun=f,
            aux=aux,
            n_iter=n_iter,
            n_fun_grad=counts["fg"],
            converged=converged,
            message=message,
            raw=opt,
        )

    f, g, aux = current()
    if not math.isfinite(f.item()):
        return result(0, False, "objective is not finite at x0")
    group = opt.param_groups[0]
    if lbfgs and g.abs().max().item() <= group["tolerance_grad"]:
        return result(0, True, "gradient below tolerance_grad")
    for it in range(max_iter):
        if lbfgs:
            if counts["fg"] >= max_eval:
                return result(it, False, "max_eval reached")
            # LBFGS counts the (memoised) evaluation at the start of the step against max_eval.
            group["max_eval"] = max_eval - counts["fg"] + 1
        x_old, f_old, aux_old = param.detach().clone(), f, aux
        opt.step(closure)
        f, g, aux = current()
        if not math.isfinite(f.item()):
            with torch.no_grad():
                param.copy_(x_old)
            f, aux = f_old, aux_old
            if callback is not None:
                callback(IterationInfo(it, x_old, f, f, aux, False, n_fun_grad=counts["fg"]))
            return result(it + 1, False, "non-finite value")
        step = (param.detach() - x_old).abs().max().item()
        info = IterationInfo(it, param.detach().clone(), f, f_old, aux, True, n_fun_grad=counts["fg"])
        stop = callback is not None and bool(callback(info))
        if lbfgs:
            # Upstream's tests; a zero step means the directional derivative was above
            # -tolerance_change, and LBFGS stopped without moving.
            if g.abs().max().item() <= group["tolerance_grad"]:
                return result(it + 1, True, "gradient below tolerance_grad")
            if step <= group["tolerance_change"]:
                return result(it + 1, True, "step below tolerance_change")
            if abs(f.item() - f_old.item()) < group["tolerance_change"]:
                return result(it + 1, True, "change in value below tolerance_change")
        if step_tol is not None and step <= step_tol:
            return result(it + 1, True, "step below step_tol")
        if stop:
            return result(it + 1, False, "stopped by callback")
    return result(max_iter, False, "max_iter reached")
