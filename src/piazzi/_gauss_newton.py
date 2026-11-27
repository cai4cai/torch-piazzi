"""Gauss-Newton and Levenberg-Marquardt on user-supplied normal equations."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

from ._common import Callback, IterationInfo, OptimizeResult, check_x0, detach_tree, retract, transfer
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
    """Gauss-Newton or Levenberg-Marquardt on user-supplied normal equations.

    Minimises a loss given its gradient ``g`` and a curvature model ``H`` (usually
    ``J^T W J``) at each point. Only the ``(n, n)`` system is moved to ``solve_device`` and
    solved in ``solve_dtype``: float64 on the CPU by default, which avoids float32 trouble
    and is cheap when ``n`` is small. Steps are capped so that no coordinate moves more
    than ``max_step``.

    - ``method="lm"`` (Levenberg-Marquardt): step ``-(H + lam D)^{-1} g``, with ``lam``
      initialised from ``tau`` and updated from the gain ratio (actual / predicted
      decrease) by Nielsen's rule. After an accepted step the next trial evaluates the
      full system (it is usually accepted too); after a rejection the trial is
      value-only.
    - ``method="gn"`` (Gauss-Newton): step ``-(H + ridge D)^{-1} g``, then Armijo
      backtracking (halving) with value-only evaluations. If ``H`` is indefinite, or a
      custom ``linear_solver`` returns an uphill step, the iteration falls back to
      steepest descent.

    ``D`` is the identity, or ``diag(H)`` with ``damping="marquardt"``.

    The run stops, as converged, once a step is below ``step_tol`` (max-norm), when the
    gradient is exactly zero (a stationary point, where ``H`` may be singular), or when a
    step fails but the model predicted a decrease below the rounding noise of the loss,
    ``8 * eps * |loss|`` in the loss's dtype (for Levenberg-Marquardt, the prediction of
    the first trial from the current point). That last test ends a float32 objective at
    its minimiser, where rounding noise in the gradient keeps steps above ``step_tol``; it
    plays the role of Ceres Solver's ``function_tolerance``. A trial point with a
    non-finite value is never accepted, and a non-finite value at ``x0`` stops the run at
    once.

    ``H`` may be a sparse COO or CSR tensor, for problems with many parameters but few
    nonzeros in ``J^T J`` (bundle adjustment, for example). The solver then only needs its
    diagonal and matrix-vector products.

    Args:
        system (Callable): ``system(x) -> (loss, g, H)``, or ``(loss, g, H, aux)`` with
            ``has_aux=True``. ``loss`` is a scalar tensor, ``g`` its gradient of shape
            ``(n,)`` and ``H`` a positive semi-definite curvature model of shape ``(n, n)``,
            dense or sparse. ``x`` has the shape of ``x0``. Called without gradient
            tracking. :func:`piazzi.least_squares_system` builds one from residuals.
        x0 (Tensor): starting point, of any shape. ``n`` is its number of elements, or the
            tangent size of ``manifold``.

    Keyword Args:
        method (str, optional): ``"lm"`` (Levenberg-Marquardt) or ``"gn"`` (Gauss-Newton).
            Default: ``"lm"``
        has_aux (bool, optional): whether ``system`` also returns an auxiliary output
            (any tree of tensors), returned detached in ``result.aux`` and
            ``info.aux``. Default: ``False``
        manifold (Manifold, optional): keeps ``x`` on a manifold such as
            :class:`piazzi.SE3`. ``g`` and ``H`` are then in tangent coordinates at ``x``
            (:func:`piazzi.least_squares_system` builds them that way when given the same
            manifold), each step is applied as ``x <- manifold.plus(x, step)``, and
            ``max_step`` and ``step_tol`` apply to the tangent step. Default: ``None``
        value_fun (Callable, optional): a cheaper ``value_fun(x) -> loss`` (or
            ``(loss, aux)``) for value-only evaluations. Default: ``None``, which uses
            the loss that ``system`` returns.
        max_iter (int, optional): maximum number of iterations, accepted or rejected.
            Default: ``100``
        max_step (float, optional): largest change of any coordinate in one step (in the
            units of ``x``, or of the tangent vector). Default: ``1.0``
        step_tol (float, optional): stop, as converged, once a step's max-norm is below
            this. Default: ``1e-5``
        c1 (float, optional): Armijo sufficient-decrease constant (``method="gn"``).
            Default: ``1e-4``
        max_backtracks (int, optional): maximum number of step halvings in one line search
            (``method="gn"``). Default: ``10``
        gn_ridge (float, optional): ridge added to ``H`` for ``method="gn"``, relative to
            ``mean(diag H)`` (or to ``D`` with ``damping="marquardt"``). Default: ``1e-6``
        tau (float, optional): initial Levenberg-Marquardt damping, relative to
            ``max(diag H)`` (or to ``D`` with ``damping="marquardt"``). Default: ``1e-3``
        max_rejections (int, optional): stop after this many rejected Levenberg-Marquardt
            steps in a row. Default: ``6``
        damping (str, optional): ``"identity"`` (``D = I``) or ``"marquardt"``
            (``D = diag(H)``, floored at ``1e-12 * max(diag H)``, as in Ceres Solver). The
            Marquardt damping scales with each parameter's curvature, which helps when
            parameters have very different units; it plays the role of
            ``x_scale="jac"`` in :func:`scipy.optimize.least_squares`.
            Default: ``"identity"``
        solve_dtype (torch.dtype, optional): dtype of the linear solve.
            Default: ``torch.float64``
        solve_device (str or torch.device, optional): device of the linear solve.
            Default: ``"cpu"``
        linear_solver (LinearSolver, optional): ``solver(H, g, damping) -> d`` solving
            ``(H + diag(damping)) d = -g`` (see :data:`piazzi.LinearSolver`).
            Default: ``None``, which uses :func:`piazzi.dense_solver` for a dense ``H`` and
            :func:`piazzi.cg_solver` (Jacobi-preconditioned conjugate gradients, which
            needs ``torchsparsegradutils``) for a sparse one.
        callback (Callable, optional): called with an :class:`piazzi.IterationInfo` after
            every iteration, accepted or not; returning ``True`` stops the run.
            Default: ``None``

    Returns:
        OptimizeResult: the solution ``x`` (shaped like ``x0``), the final loss ``fun`` and
        ``aux``, ``n_iter``, the counts ``n_system`` (full system evaluations) and ``n_fun``
        (value-only evaluations), and ``converged`` and ``message``.

    Example::

        >>> t = torch.linspace(0, 1, 20, dtype=torch.float64)
        >>> y = 2 * torch.exp(-3 * t)
        >>> system = piazzi.least_squares_system(lambda p: p[0] * torch.exp(-p[1] * t) - y)
        >>> res = piazzi.gauss_newton(system, torch.tensor([1.0, 1.0], dtype=torch.float64))
        >>> res.x
        tensor([2.0000, 3.0000], dtype=torch.float64)
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
            if not gtd < 0:  # an indefinite H or an inexact custom solver: fall back to steepest descent
                d = -gc * min(1.0, max_step / gc.abs().max().item())
                gtd = torch.dot(gc, d).item()
            t = 1.0
            dx = transfer(d, x.device, x.dtype)
            x_new = retract(manifold, x, dx)
            f_new, aux_new, g_new, H_new = eval_sys(x_new)
            n_bt = 0

            while not armijo_ok(f_new.item(), f_val, t, gtd, d) and n_bt < max_backtracks:
                t *= 0.5
                x_new = retract(manifold, x, t * dx)
                f_new, g_new = eval_value(x_new), None
                n_bt += 1
            if not armijo_ok(f_new.item(), f_val, t, gtd, d):
                notify(it, f_start, False)
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
            if below_noise(pred_first, fn, f_val):
                return result(it + 1, True, "predicted decrease below rounding noise")
            if rejections >= max_rejections:
                return result(it + 1, False, "too many rejected steps in a row")
            if stop:
                return result(it + 1, False, "stopped by callback")
    return result(max_iter, False, "max_iter reached")


def levenberg_marquardt(system: Callable, x0: torch.Tensor, **kwargs) -> OptimizeResult:
    """Levenberg-Marquardt: :func:`piazzi.gauss_newton` with ``method="lm"``.

    Args:
        system (Callable): ``system(x) -> (loss, g, H)``, as for :func:`piazzi.gauss_newton`.
        x0 (Tensor): starting point.
        **kwargs: any other keyword argument of :func:`piazzi.gauss_newton`.

    Returns:
        OptimizeResult: as for :func:`piazzi.gauss_newton`.
    """
    return gauss_newton(system, x0, method="lm", **kwargs)
