"""Gauss-Newton and Levenberg-Marquardt on user-supplied normal equations."""

from __future__ import annotations

from collections.abc import Callable

import torch

from ._common import Callback, IterationInfo, OptimizeResult, check_x0, detach_tree

__all__ = ["gauss_newton", "levenberg_marquardt"]


def gauss_newton(
    system: Callable,
    x0: torch.Tensor,
    *,
    method: str = "lm",
    has_aux: bool = False,
    value_fun: Callable | None = None,
    max_iter: int = 100,
    max_step: float = 1.0,
    step_tol: float = 1e-5,
    c1: float = 1e-4,
    max_backtracks: int = 10,
    gn_ridge: float = 1e-6,
    tau: float = 1e-3,
    max_rejections: int = 6,
    solve_dtype: torch.dtype = torch.float64,
    solve_device: str | torch.device = "cpu",
    callback: Callback | None = None,
) -> OptimizeResult:
    """Gauss-Newton (``method="gn"``) or Levenberg-Marquardt (``"lm"``).

    ``system(x)`` returns ``(loss, g, H)`` (``(loss, g, H, aux)`` with ``has_aux``): the
    loss, its gradient (n,) and a positive semi-definite curvature model (n, n), usually
    ``J^T W J``. It is called without gradient tracking. Only the n x n system is moved to
    ``solve_device`` and solved in ``solve_dtype`` (float64 on the CPU by default, which
    avoids float32 trouble and is cheap for small n). Steps are capped so that no
    coordinate moves more than ``max_step``.

    - ``"gn"``: step ``-(H + ridge I)^{-1} g`` with ``ridge = gn_ridge * mean(diag H)``,
      then Armijo backtracking (halving) with value-only evaluations; stops when no
      decrease is found.
    - ``"lm"``: step ``-(H + lam I)^{-1} g`` with ``lam`` initialised to
      ``tau * max(diag H)`` and updated from the gain ratio (actual / predicted decrease)
      by Nielsen's rule. After an accepted step the next trial evaluates the full system
      (it is usually accepted too); after a rejection the trial is value-only. Stops
      after ``max_rejections`` rejections in a row.

    Both stop once a step is below ``step_tol`` (max-norm), or when the gradient is exactly
    zero (a stationary point, where ``H`` may be singular). A trial point with a non-finite
    value is never accepted, and a non-finite value at ``x0`` stops the run at once.

    Args:
        value_fun: optional cheaper ``value_fun(x) -> loss`` for value-only evaluations.
            Defaults to the loss returned by ``system``.
        callback: called with an :class:`IterationInfo` after every iteration; returning
            True stops the run.
    """
    if method not in ("gn", "lm"):
        raise ValueError("method must be 'gn' or 'lm'")
    x, shape = check_x0(x0)
    counts = {"sys": 0, "f": 0}

    def eval_sys(xv):
        with torch.no_grad():
            out = system(xv.reshape(shape))
        if has_aux:
            loss, g, H, aux = out
        else:
            (loss, g, H), aux = out, None
        counts["sys"] += 1
        return loss.detach(), detach_tree(aux), g.detach().reshape(-1), H.detach()

    def eval_value(xv):
        with torch.no_grad():
            loss = value_fun(xv.reshape(shape)) if value_fun is not None else system(xv.reshape(shape))[0]
        counts["f"] += 1
        return loss.detach()

    def host(g, H):
        # Move first, then cast: casting on the source device fails where it lacks float64 (MPS).
        return g.to(solve_device).to(solve_dtype), H.to(solve_device).to(solve_dtype)

    def at_noise_floor(f_new, f_val, step):
        # A step below step_tol whose value went up by no more than rounding noise: we are at a
        # minimiser to working precision, so the step counts as accepted (and converged).
        noise = 8 * torch.finfo(f.dtype).eps * abs(f_val)
        return step < step_tol and f_new <= f_val + noise

    def armijo_ok(fn, f_val, t, gtd, d):
        # `<=` (rather than `not >`) so that a NaN trial value counts as insufficient decrease.
        return fn <= f_val + c1 * t * gtd or at_noise_floor(fn, f_val, t * d.abs().max().item())

    def solve(Hc, gc, lam):
        A = Hc + lam * torch.eye(Hc.shape[0], dtype=Hc.dtype, device=Hc.device)
        d = torch.linalg.solve(A, -gc)
        return d * min(1.0, max_step / max(d.abs().max().item(), 1e-12))

    def result(n_iter, converged, message):
        return OptimizeResult(
            x=x.reshape(shape),
            fun=f,
            aux=aux,
            n_iter=n_iter,
            n_system=counts["sys"],
            n_fun=counts["f"],
            converged=converged,
            message=message,
        )

    def notify(it, f_start, accepted):
        return callback is not None and bool(callback(IterationInfo(it, x.reshape(shape), f, f_start, aux, accepted)))

    f, aux, g, H = eval_sys(x)
    if not torch.isfinite(f):
        return result(0, False, "objective is not finite at x0")
    gc, Hc = host(g, H)
    lam = tau * Hc.diagonal().max().item()
    nu, rejections, have_sys = 2.0, 0, True
    for it in range(max_iter):
        f_val, f_start = f.item(), f
        if gc.abs().max().item() == 0:  # stationary, and H may be singular (e.g. J = 0)
            return result(it, True, "gradient is zero")
        if method == "gn":
            d = solve(Hc, gc, gn_ridge * Hc.diagonal().mean().item())
            gtd = torch.dot(gc, d).item()
            t = 1.0
            dx = d.to(x)
            x_new = x + dx
            f_new, aux_new, g_new, H_new = eval_sys(x_new)
            n_bt = 0

            while not armijo_ok(f_new.item(), f_val, t, gtd, d) and n_bt < max_backtracks:
                t *= 0.5
                x_new = x + t * dx
                f_new, g_new = eval_value(x_new), None
                n_bt += 1
            if not armijo_ok(f_new.item(), f_val, t, gtd, d):
                notify(it, f_start, False)
                return result(it + 1, False, "line search failed")
            if g_new is None:
                f_new, aux_new, g_new, H_new = eval_sys(x_new)
            step = t * d.abs().max().item()
            x, f, aux = x_new, f_new, aux_new
            gc, Hc = host(g_new, H_new)
            stop = notify(it, f_start, True)
            if step < step_tol:
                return result(it + 1, True, "step below step_tol")
            if stop:
                return result(it + 1, False, "stopped by callback")
            continue
        # Levenberg-Marquardt.
        d = solve(Hc, gc, lam)
        pred = -(torch.dot(gc, d) + 0.5 * d @ Hc @ d).item()
        x_new = x + d.to(x)
        if rejections == 0:  # after an accepted step the trial is usually accepted too
            f_new, aux_new, g_new, H_new = eval_sys(x_new)
            have_sys = True
        else:
            f_new, have_sys = eval_value(x_new), False
        rho = (f_val - f_new.item()) / max(pred, 1e-20)
        # At a minimiser the model predicts no decrease (pred = 0), so the gain ratio is 0 and
        # would reject the zero step forever; accept it when the value did not go up. Likewise
        # a tiny step whose decrease is lost to rounding (pred ~ 1e-32) is accepted.
        fn = f_new.item()
        if rho > 0 or (pred <= 0 and fn <= f_val) or at_noise_floor(fn, f_val, d.abs().max().item()):
            if not have_sys:
                f_new, aux_new, g_new, H_new = eval_sys(x_new)
            lam *= max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3)
            nu, rejections = 2.0, 0
            step = d.abs().max().item()
            x, f, aux = x_new, f_new, aux_new
            gc, Hc = host(g_new, H_new)
            stop = notify(it, f_start, True)
            if step < step_tol:
                return result(it + 1, True, "step below step_tol")
            if stop:
                return result(it + 1, False, "stopped by callback")
        else:
            lam *= nu
            nu *= 2.0
            rejections += 1
            stop = notify(it, f_start, False)
            if rejections >= max_rejections:
                return result(it + 1, False, "too many rejected steps in a row")
            if stop:
                return result(it + 1, False, "stopped by callback")
    return result(max_iter, False, "max_iter reached")


def levenberg_marquardt(system: Callable, x0: torch.Tensor, **kwargs) -> OptimizeResult:
    """:func:`gauss_newton` with ``method="lm"``."""
    return gauss_newton(system, x0, method="lm", **kwargs)
