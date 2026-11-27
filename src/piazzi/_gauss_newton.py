"""Gauss-Newton and Levenberg-Marquardt on user-supplied normal equations."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

from ._common import Callback, IterationInfo, OptimizeResult, check_x0, detach_tree, retract, transfer, unmoved
from ._linear_solvers import LinearSolver, cg_solver, dense_solver, diagonal, is_sparse, matvec
from ._manifolds import Manifold

__all__ = ["gauss_newton", "levenberg_marquardt"]


def gauss_newton(
    system: Callable,
    x0: torch.Tensor,
    *,
    method: str = "lm",
    has_aux: bool = False,
    manifold: Manifold | None = None,
    value_fun: Callable | None = None,
    max_iter: int = 100,
    max_step: float = 1.0,
    step_tol: float = 1e-5,
    c1: float = 1e-4,
    max_backtracks: int = 10,
    gn_ridge: float = 1e-6,
    tau: float = 1e-3,
    max_rejections: int = 6,
    damping: str = "identity",
    solve_dtype: torch.dtype = torch.float64,
    solve_device: str | torch.device = "cpu",
    linear_solver: LinearSolver | None = None,
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

    ``damping="marquardt"`` replaces ``I`` by ``D = diag(H)`` (floored at
    ``1e-12 * max(diag H)``), as in Ceres Solver: the ridge becomes ``gn_ridge * D`` and the
    Levenberg-Marquardt damping ``lam * D`` with ``lam`` initialised to ``tau``. The damping
    then scales with each parameter's curvature, which helps when parameters have very
    different units (it plays the role of ``x_scale="jac"`` in
    ``scipy.optimize.least_squares``).

    Both stop once a step is below ``step_tol`` (max-norm), or when the gradient is exactly
    zero (a stationary point, where ``H`` may be singular). They also stop, as converged,
    when a step fails but the model predicted a decrease below the rounding noise of the
    loss, ``8 * eps * |loss|`` in the loss's dtype (for Levenberg-Marquardt, the prediction
    of the first trial from the current point): no step can then show measurable progress.
    This is what ends a float32 objective at its minimiser, where rounding noise in the
    gradient keeps steps above ``step_tol``; it plays the role of Ceres Solver's
    ``function_tolerance``. Near a zero residual the loss's rounding noise is absolute and
    that test cannot see it, so a failed step also counts as converged when the Gauss-Newton
    step, which estimates the distance to the minimiser, leaves ``x`` unmoved: every entry
    within 32 units in the last place of its scale (``|x|``, or the manifold's
    :meth:`~piazzi.Manifold.entry_scale`). The loss cannot change measurably along
    such a step. For Levenberg-Marquardt, whose first trial from a point is damped, the
    undamped Gauss-Newton step is checked as well (one extra linear solve), and a trial with
    a non-finite value never counts. A trial point with a non-finite
    value is never accepted, and a non-finite value at ``x0`` stops the run at once.

    With a ``manifold`` (see :class:`piazzi.Manifold`), ``g`` and ``H`` are in its tangent
    coordinates at ``x`` (:func:`piazzi.least_squares_system` builds them that way when given
    the same manifold), and each step is applied as ``x <- manifold.plus(x, step)``.
    ``max_step`` and ``step_tol`` then apply to the tangent step.

    ``H`` may also be a sparse COO/CSR tensor, for problems with many parameters but few
    nonzeros in ``J^T J`` (bundle adjustment, for example). The solver then only needs its
    diagonal and matrix-vector products.

    Args:
        value_fun: optional cheaper ``value_fun(x) -> loss`` for value-only evaluations.
            Defaults to the loss returned by ``system``.
        linear_solver: ``solver(H, g, damping) -> d`` solving ``(H + diag(damping)) d = -g`` (see
            :func:`piazzi.dense_solver` and :func:`piazzi.cg_solver`). Defaults to a direct
            dense solve for a dense ``H`` and to Jacobi-preconditioned conjugate gradients
            (:func:`piazzi.cg_solver`, which needs ``torchsparsegradutils``) for a sparse one.
        callback: called with an :class:`IterationInfo` after every iteration; returning
            True stops the run.
    """
    if method not in ("gn", "lm"):
        raise ValueError("method must be 'gn' or 'lm'")
    if damping not in ("identity", "marquardt"):
        raise ValueError(f"damping must be 'identity' or 'marquardt', got {damping!r}")
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
        # Copies: a system may write into the same output buffers at every call, and the solver
        # keeps g and H of the current point while it evaluates trial points.
        return loss.detach(), detach_tree(aux), g.detach().reshape(-1).clone(), H.detach().clone()

    def eval_value(xv):
        with torch.no_grad():
            loss = value_fun(xv.reshape(shape)) if value_fun is not None else system(xv.reshape(shape))[0]
        if isinstance(loss, tuple):  # a value_fun that also returns aux
            loss = loss[0]
        counts["f"] += 1
        return loss.detach()

    def all_finite(t):
        values = t if not is_sparse(t) else (t.coalesce().values() if t.layout == torch.sparse_coo else t.values())
        return bool(torch.isfinite(values).all())

    def host(t):
        return transfer(t, solve_device, solve_dtype)

    def noise(f_val):
        """Rounding noise of a loss value ``f_val`` in the objective's dtype."""
        return 8 * torch.finfo(f.dtype).eps * abs(f_val)

    def at_noise_floor(f_new, f_val, step):
        # A step below step_tol whose value went up by no more than rounding noise: we are at a
        # minimiser to working precision, so the step counts as accepted (and converged).
        return step < step_tol and f_new <= f_val + noise(f_val)

    def below_noise(pred, f_new, f_val):
        # The model promises less than rounding noise and the value did not measurably go up:
        # a failed step here means we are at a minimiser to working precision.
        return pred <= noise(f_val) and math.isfinite(f_new) and f_new <= f_val + noise(f_val)

    def armijo_ok(fn, f_val, t, gtd, d):
        # A non-finite trial value (NaN, or -inf, which would pass `<=`) is never accepted.
        return math.isfinite(fn) and (fn <= f_val + c1 * t * gtd or at_noise_floor(fn, f_val, t * d.abs().max().item()))

    solvers = {}

    def damping_vector(Hc, lam):
        diag = diagonal(Hc)
        if damping == "identity":
            return torch.full_like(diag, lam)
        return lam * diag.clamp(min=1e-12 * diag.max().item())

    def gauss_newton_unmoved(Hc, gc, x):
        """Whether the (lightly ridged) Gauss-Newton step from ``x`` leaves it unmoved."""
        d = solve(Hc, gc, gn_ridge * (diagonal(Hc).mean().item() if damping == "identity" else 1.0))
        if not torch.dot(gc, d).item() < 0:  # not a descent direction: no estimate of the distance
            return False
        return unmoved(manifold, x, retract(manifold, x, transfer(d, x.device, x.dtype)))

    def solve(Hc, gc, lam):
        if linear_solver is not None:
            solver = linear_solver
        else:
            key = is_sparse(Hc)
            if key not in solvers:
                solvers[key] = cg_solver() if key else dense_solver
            solver = solvers[key]
        with torch.no_grad():  # the step is never differentiated; keep custom solvers from recording a graph
            d = solver(Hc, gc, damping_vector(Hc, lam))
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
        if callback is None:
            return False
        info = IterationInfo(it, x.reshape(shape), f, f_start, aux, accepted, n_fun=counts["f"], n_system=counts["sys"])
        return bool(callback(info))

    f, aux, g, H = eval_sys(x)
    if not torch.isfinite(f):
        return result(0, False, "objective is not finite at x0")
    if g.numel() == 0:  # a manifold with no degrees of freedom (e.g. Subset with every entry fixed)
        return result(0, True, "no free parameters")
    gc, Hc = host(g), host(H)
    lam = tau * (diagonal(Hc).max().item() if damping == "identity" else 1.0)
    nu, rejections, have_sys = 2.0, 0, True
    for it in range(max_iter):
        f_val, f_start = f.item(), f
        if not (all_finite(gc) and all_finite(Hc)):
            return result(it, False, "gradient or curvature is not finite")
        if gc.abs().max().item() == 0:  # stationary, and H may be singular (e.g. J = 0)
            return result(it, True, "gradient is zero")
        if method == "gn":
            d = solve(Hc, gc, gn_ridge * (diagonal(Hc).mean().item() if damping == "identity" else 1.0))
            gtd = torch.dot(gc, d).item()
            newton = gtd < 0
            if not newton:  # an indefinite H or an inexact custom solver: fall back to steepest descent
                d = -gc * min(1.0, max_step / gc.abs().max().item())
                gtd = torch.dot(gc, d).item()
            t = 1.0
            dx = transfer(d, x.device, x.dtype)
            x_new = retract(manifold, x, dx)
            f_new, aux_new, g_new, H_new = eval_sys(x_new)
            full_unmoved = newton and math.isfinite(f_new.item()) and unmoved(manifold, x, x_new)
            n_bt = 0

            while not armijo_ok(f_new.item(), f_val, t, gtd, d) and n_bt < max_backtracks:
                t *= 0.5
                x_new = retract(manifold, x, t * dx)
                f_new, g_new = eval_value(x_new), None
                n_bt += 1
            if not armijo_ok(f_new.item(), f_val, t, gtd, d):
                notify(it, f_start, False)
                if full_unmoved:
                    return result(it + 1, True, "step below the resolution of x")
                if below_noise(-gtd, f_new.item(), f_val):
                    return result(it + 1, True, "predicted decrease below rounding noise")
                return result(it + 1, False, "line search failed")
            if g_new is None:
                f_new, aux_new, g_new, H_new = eval_sys(x_new)
            step = t * d.abs().max().item()
            x, f, aux = x_new, f_new, aux_new
            gc, Hc = host(g_new), host(H_new)
            stop = notify(it, f_start, True)
            if step < step_tol:
                return result(it + 1, True, "step below step_tol")
            if stop:
                return result(it + 1, False, "stopped by callback")
            continue
        # Levenberg-Marquardt.
        d = solve(Hc, gc, lam)
        pred = -(torch.dot(gc, d) + 0.5 * torch.dot(d, matvec(Hc, d))).item()
        if rejections == 0:
            # The prediction of the first trial from this point: rejections raise lam and shrink
            # later predictions, which must not pass for convergence far from a minimiser.
            pred_first = pred
        x_new = retract(manifold, x, transfer(d, x.device, x.dtype))
        if rejections == 0:
            # The first trial is damped, so its size alone does not estimate the distance to the
            # minimiser: when it leaves x unmoved, the Gauss-Newton step is checked as well, once.
            first_unmoved, gn_checked = unmoved(manifold, x, x_new), None
        if rejections == 0:  # after an accepted step the trial is usually accepted too
            f_new, aux_new, g_new, H_new = eval_sys(x_new)
            have_sys = True
        else:
            f_new, have_sys = eval_value(x_new), False
        rho = (f_val - f_new.item()) / max(pred, 1e-20)
        # At a minimiser the model predicts no decrease (pred = 0), so the gain ratio is 0 and
        # would reject the zero step forever; accept it when the value did not go up. Likewise
        # a tiny step whose decrease is lost to rounding (pred ~ 1e-32) is accepted.
        # A non-finite trial value is never accepted: -inf would give rho = +inf.
        fn = f_new.item()
        accept = rho > 0 or (pred <= 0 and fn <= f_val) or at_noise_floor(fn, f_val, d.abs().max().item())
        if math.isfinite(fn) and accept:
            if not have_sys:
                f_new, aux_new, g_new, H_new = eval_sys(x_new)
            lam *= max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3)
            nu, rejections = 2.0, 0
            step = d.abs().max().item()
            x, f, aux = x_new, f_new, aux_new
            gc, Hc = host(g_new), host(H_new)
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
            if first_unmoved and math.isfinite(fn):
                if gn_checked is None:
                    gn_checked = gauss_newton_unmoved(Hc, gc, x)
                if gn_checked:
                    return result(it + 1, True, "step below the resolution of x")
            if below_noise(pred_first, fn, f_val):
                return result(it + 1, True, "predicted decrease below rounding noise")
            if rejections >= max_rejections:
                return result(it + 1, False, "too many rejected steps in a row")
            if stop:
                return result(it + 1, False, "stopped by callback")
    return result(max_iter, False, "max_iter reached")


def levenberg_marquardt(system: Callable, x0: torch.Tensor, **kwargs) -> OptimizeResult:
    """:func:`gauss_newton` with ``method="lm"``."""
    return gauss_newton(system, x0, method="lm", **kwargs)
