"""Gauss-Newton and Levenberg-Marquardt on user-supplied normal equations."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

from ._common import Callback, IterationInfo, OptimizeResult, check_x0, detach_tree, retract, transfer, unmoved
from ._linear_solvers import LinearSolver, cg_solver, dense_solver, diagonal, is_sparse, matvec
from ._manifolds import Manifold
from ._progress import with_progress

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
    verbose: bool | int = False,
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
    plays the role of Ceres Solver's ``function_tolerance``. Near a zero residual the loss's
    rounding noise is absolute and that test cannot see it, so a failed step also counts as
    converged when the Gauss-Newton step, which estimates the distance to the minimiser,
    leaves ``x`` unmoved: every entry within 64 units in the last place of its scale
    (``|x|``, or the manifold's :meth:`~piazzi.Manifold.entry_scale`). The loss cannot
    change measurably along such a step. For Levenberg-Marquardt, whose first trial from a
    point is damped, the undamped Gauss-Newton step is checked as well (one extra linear
    solve), and a trial with a non-finite value never counts. A trial point with a
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
        step_tol (float, optional): stop, as converged, once an accepted step's max-norm is
            below this. The step must estimate the distance to the minimiser: the full
            Gauss-Newton step before any backtracking, and for Levenberg-Marquardt, when the
            damped step is that short, also the undamped Gauss-Newton step. A step shortened by
            backtracking or damping alone does not count, and a step that does not change ``x``
            at all is never accepted. Default: ``1e-5``
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
        verbose (bool or int, optional): print a table of the run's progress with
            :func:`piazzi.print_progress` before calling ``callback``: ``True`` prints every
            iteration and an integer ``k`` every ``k``-th one. Default: ``False``

    Returns:
        OptimizeResult: the solution ``x`` (shaped like ``x0``), the final loss ``fun`` and
        ``aux``, ``n_iter``, the counts ``n_system`` (full system evaluations) and ``n_fun``
        (value-only evaluations), and ``converged`` and ``message``.

    See Also:
        - :func:`piazzi.least_squares`: the high-level interface, which builds ``system`` from
          residuals with :func:`piazzi.least_squares_system` and passes its other keyword arguments
          here.
        - :func:`piazzi.dogleg`: a trust-region method on the same systems.

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
    callback = with_progress(callback, verbose)
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
        # A non-finite trial value (NaN, or -inf, which would pass `<=`) is never accepted. The
        # noise-floor acceptance looks at the full step d, not t * d: backtracking shortens the
        # step without the minimiser being closer, so a flat backtracked trial is no evidence
        # of a minimiser (accepting it made the run repeat the same search until max_iter).
        return math.isfinite(fn) and (fn <= f_val + c1 * t * gtd or at_noise_floor(fn, f_val, d.abs().max().item()))

    solvers = {}

    def damping_vector(Hc, lam):
        diag = diagonal(Hc)
        if damping == "identity":
            return torch.full_like(diag, lam)
        return lam * diag.clamp(min=1e-12 * diag.max().item())

    def predicted_decrease(Hc, gc, d):
        """Decrease of the loss that the quadratic model predicts for the step ``d``."""
        return -(torch.dot(gc, d) + 0.5 * torch.dot(d, matvec(Hc, d))).item()

    # The Gauss-Newton step estimates the distance to the minimiser only when its ridge is light:
    # the distance checks use their own (at most the default) ridge, and the Gauss-Newton
    # method's shortcuts below are skipped when a large gn_ridge damps its step.
    check_ridge = min(gn_ridge, 1e-6)
    gn_step_is_distance = gn_ridge <= 1e-3

    def descent_or_none(gc, d):
        """``d`` if it is a descent direction, else None: no estimate of the distance."""
        return d if torch.dot(gc, d).item() < 0 else None

    def gauss_newton_step(Hc, gc):
        """The lightly ridged Gauss-Newton step, or None if it is not a descent direction."""
        return descent_or_none(
            gc, solve(Hc, gc, check_ridge * (diagonal(Hc).mean().item() if damping == "identity" else 1.0))
        )

    def gauss_newton_unmoved(Hc, gc, x):
        """Whether the Gauss-Newton step from ``x`` leaves it unmoved."""
        d = gauss_newton_step(Hc, gc)
        return d is not None and unmoved(manifold, x, retract(manifold, x, transfer(d, x.device, x.dtype)))

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
    # Levenberg-Marquardt state carried across rejected trials. It is set whenever rejections == 0,
    # which holds at the first iteration and after every accepted step, before it is read.
    pred_first, first_unmoved, gn_checked, gn_pred = math.inf, False, None, None
    aux_new, g_new, H_new = aux, g, H
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
            distance = newton and gn_step_is_distance  # d estimates the distance to the minimiser
            if torch.equal(x_new, x):  # the step does not change x at all: nothing to evaluate
                notify(it, f_start, False)
                if distance:
                    return result(it + 1, True, "step below the resolution of x")
                return result(it + 1, False, "line search failed")
            f_new, aux_new, g_new, H_new = eval_sys(x_new)
            full_unmoved = distance and math.isfinite(f_new.item()) and unmoved(manifold, x, x_new)
            n_bt = 0

            while not armijo_ok(f_new.item(), f_val, t, gtd, d) and n_bt < max_backtracks:
                t *= 0.5
                x_new = retract(manifold, x, t * dx)
                if torch.equal(x_new, x):  # backtracked to a step that does not change x: failure
                    break
                f_new, g_new = eval_value(x_new), None
                n_bt += 1
            if torch.equal(x_new, x) or not armijo_ok(f_new.item(), f_val, t, gtd, d):
                notify(it, f_start, False)
                if full_unmoved:
                    return result(it + 1, True, "step below the resolution of x")
                if below_noise(-gtd, f_new.item(), f_val):
                    return result(it + 1, True, "predicted decrease below rounding noise")
                return result(it + 1, False, "line search failed")
            if g_new is None:
                f_new, aux_new, g_new, H_new = eval_sys(x_new)
            # The full step, not the backtracked t * d: backtracking shortens the step without the
            # minimiser being any closer, so only the full step estimates the distance to it. A
            # large gn_ridge damps the step itself: then the lightly ridged one is judged instead.
            step = d.abs().max().item()
            if step < step_tol and newton and not gn_step_is_distance:
                d_gn = gauss_newton_step(Hc, gc)
                step = math.inf if d_gn is None else d_gn.abs().max().item()
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
        # With light damping (at most 1e-3 of the curvature on every coordinate), the damped step
        # is already the Gauss-Newton step to within 0.1 %: no separate Gauss-Newton solve needed.
        lightly_damped = bool((damping_vector(Hc, lam) <= 1e-3 * diagonal(Hc).abs()).all())
        pred = predicted_decrease(Hc, gc, d)
        if rejections == 0:
            # The prediction of the first trial from this point: rejections raise lam and shrink
            # later predictions, which must not pass for convergence far from a minimiser.
            pred_first = pred
        x_new = retract(manifold, x, transfer(d, x.device, x.dtype))
        if rejections == 0:
            # The first trial is damped, so its size alone does not estimate the distance to the
            # minimiser: when it leaves x unmoved, the Gauss-Newton step is checked as well, once.
            first_unmoved, gn_checked, gn_pred = unmoved(manifold, x, x_new), None, None
        # A trial that does not change x at all (heavy damping) is rejected without evaluating
        # it: accepting it would count as a step below step_tol anywhere.
        moved = not torch.equal(x_new, x)
        if not moved:
            f_new, have_sys = f, False
        elif rejections == 0:  # after an accepted step the trial is usually accepted too
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
        if moved and math.isfinite(fn) and accept:
            if not have_sys:
                f_new, aux_new, g_new, H_new = eval_sys(x_new)
            lam *= max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3)
            nu, rejections = 2.0, 0
            step = d.abs().max().item()
            converged = False
            if step < step_tol:
                # The damped step is short, which may only mean that lam is large: the undamped
                # Gauss-Newton step from the same point (one extra solve, no evaluation, unless
                # the damping is light) is what estimates the distance to the minimiser. Without
                # a Gauss-Newton descent step there is no such estimate, and no convergence.
                d_gn = descent_or_none(gc, d) if lightly_damped else gauss_newton_step(Hc, gc)
                converged = d_gn is not None and d_gn.abs().max().item() < step_tol
            x, f, aux = x_new, f_new, aux_new
            gc, Hc = host(g_new), host(H_new)
            stop = notify(it, f_start, True)
            if converged:
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
                # The damped first trial's prediction may be small only because lam is large:
                # the undamped Gauss-Newton step's prediction must be below the noise too.
                if gn_pred is None:  # without a Gauss-Newton descent step, no evidence (inf)
                    d_gn = gauss_newton_step(Hc, gc)
                    gn_pred = math.inf if d_gn is None else predicted_decrease(Hc, gc, d_gn)
                if below_noise(gn_pred, fn, f_val):
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

    See Also:
        - :func:`piazzi.least_squares`: the high-level interface, which runs this method by default.
    """
    return gauss_newton(system, x0, method="lm", **kwargs)
