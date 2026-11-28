"""Powell's dog-leg trust-region method on user-supplied normal equations.

Modelled on Ceres Solver's ``DoglegStrategy`` (``internal/ceres/dogleg_strategy.cc``). Everything
except the truncation of the dog-leg path to the trust region (:func:`_traditional_step`,
:func:`_subspace_step`) is independent of the region's shape, so that other regions (the
rectangular one of SciPy's ``dogbox``, for bounds) can share the loop.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

from ._common import Callback, IterationInfo, OptimizeResult, check_x0, detach_tree, retract, transfer, unmoved
from ._linear_solvers import LinearSolver, cg_solver, dense_solver, diagonal, is_sparse, matvec
from ._manifolds import Manifold
from ._progress import with_progress

__all__ = ["dogleg"]

# Ridge of the Gauss-Newton solve, relative to the curvature (Ceres: kMinMu, kMaxMu, and the factor 10).
MIN_MU = 1e-8
MAX_MU = 1.0
MU_FACTOR = 10.0
# Above this ridge the Gauss-Newton step no longer estimates the distance to the minimiser.
MAX_DISTANCE_MU = 1e-3


def _traditional_step(gs, gns, alpha, radius):
    """The point where the path origin -> Cauchy point -> Gauss-Newton point leaves the ball of ``radius``.

    All vectors are in scaled coordinates (the trust region is a ball there). ``gns`` is the
    Gauss-Newton step or ``None``, ``alpha`` the Cauchy step length (``inf`` for a direction
    of non-positive curvature).
    """
    g_norm = gs.norm().item()
    if gns is not None:
        gn_norm = gns.norm().item()
        if gn_norm <= radius:  # case 1: the Gauss-Newton step is inside
            return gns
    if alpha * g_norm >= radius:  # case 2: the Cauchy point is outside (or beyond the model's minimum)
        return -(radius / g_norm) * gs
    if gns is None:  # no Gauss-Newton step: the Cauchy point, which is inside
        return -alpha * gs
    # Case 3: the segment from the Cauchy point a to the Gauss-Newton point b crosses the boundary.
    a = -alpha * gs
    ba = gns - a
    ba2 = torch.dot(ba, ba).item()
    c = torch.dot(a, ba).item()
    a2 = torch.dot(a, a).item()
    d = math.sqrt(max(c * c + ba2 * (radius * radius - a2), 0.0))
    beta = (d - c) / ba2 if c <= 0 else (radius * radius - a2) / (d + c)
    return a + beta * ba


class _Subspace:
    """The two-dimensional model on the span of the scaled gradient and the Gauss-Newton step."""

    COSINE_THRESHOLD = 0.99  # Ceres: kCosineThreshold

    def __init__(self, gs, gns, hess_scaled):
        self.one_dimensional = False
        g_norm = gs.norm().item()
        u1 = gs / g_norm
        u2 = gns - torch.dot(u1, gns) * u1
        n2 = u2.norm().item()
        if not n2 > 1e-10 * gns.norm().item():  # gradient and Gauss-Newton step coincide
            self.one_dimensional = True
            return
        u2 = u2 / n2
        self.basis = (u1, u2)
        self.g = [torch.dot(u, gs).item() for u in self.basis]
        hu = [hess_scaled(u) for u in self.basis]
        self.B = [[torch.dot(u, h).item() for h in hu] for u in self.basis]

    def boundary_minimum(self, radius):
        """Minimiser of the model on the circle of ``radius``, or None if the root finding fails.

        Stationary points satisfy ``(B + y I) x = -g`` with ``|x| = radius``, which gives a
        polynomial of degree four in ``y`` (see Ceres's ``MakePolynomialForBoundaryConstrainedProblem``).
        """
        (b00, b01), (b10, b11) = self.B
        g0, g1 = self.g
        det_b, tr_b, r2 = b00 * b11 - b01 * b10, b00 + b11, radius * radius
        adj = ((b11, -b01), (-b10, b00))
        adj_g = (adj[0][0] * g0 + adj[0][1] * g1, adj[1][0] * g0 + adj[1][1] * g1)
        poly = [
            r2,
            2.0 * r2 * tr_b,
            r2 * (tr_b * tr_b + 2.0 * det_b) - (g0 * g0 + g1 * g1),
            -2.0 * (g0 * adj_g[0] + g1 * adj_g[1] - r2 * det_b * tr_b),
            r2 * det_b * det_b - (adj_g[0] ** 2 + adj_g[1] ** 2),
        ]
        if not all(math.isfinite(c) for c in poly):
            return None
        companion = torch.zeros(4, 4, dtype=torch.float64)
        companion[1:, :-1] = torch.eye(3, dtype=torch.float64)
        companion[:, -1] = -torch.tensor(poly[:0:-1], dtype=torch.float64) / poly[0]
        best, best_value = None, math.inf
        for y in torch.linalg.eigvals(companion).real.tolist():
            det = (b00 + y) * (b11 + y) - b01 * b10
            if det == 0 or not math.isfinite(det):
                continue
            x0 = -((b11 + y) * g0 - b01 * g1) / det
            x1 = -(-b10 * g0 + (b00 + y) * g1) / det
            n = math.hypot(x0, x1)
            if not (n > 0 and math.isfinite(n)):
                continue
            x0, x1 = x0 * radius / n, x1 * radius / n  # project onto the boundary
            value = 0.5 * (x0 * (b00 * x0 + b01 * x1) + x1 * (b10 * x0 + b11 * x1)) + g0 * x0 + g1 * x1
            if value < best_value:
                best, best_value = (x0, x1), value
        return best

    def step(self, gs, radius):
        """The subspace step on the boundary, or None when the caller should take a traditional one."""
        if self.one_dimensional:  # move along the gradient to the boundary
            return -(radius / gs.norm().item()) * gs
        x = self.boundary_minimum(radius)
        if x is None:
            return None
        # First-order optimality on the boundary: B x + g must point against x (Ceres checks the angle).
        (b00, b01), (b10, b11) = self.B
        grad = (b00 * x[0] + b01 * x[1] + self.g[0], b10 * x[0] + b11 * x[1] + self.g[1])
        denom = math.hypot(*x) * math.hypot(*grad)
        if not denom > 0 or -(x[0] * grad[0] + x[1] * grad[1]) / denom < self.COSINE_THRESHOLD:
            return None
        return x[0] * self.basis[0] + x[1] * self.basis[1]


def dogleg(
    system: Callable,
    x0: torch.Tensor,
    *,
    dogleg_type: str = "traditional",
    has_aux: bool = False,
    manifold: Manifold | None = None,
    value_fun: Callable | None = None,
    max_iter: int = 100,
    step_tol: float = 1e-5,
    initial_radius: float = 1.0,
    max_radius: float = 1e16,
    min_gain: float = 1e-3,
    max_rejections: int = 20,
    first_trial: str = "auto",
    damping: str = "identity",
    solve_dtype: torch.dtype = torch.float64,
    solve_device: str | torch.device = "cpu",
    linear_solver: LinearSolver | None = None,
    callback: Callback | None = None,
    verbose: bool | int = False,
) -> OptimizeResult:
    """Powell's dog-leg trust-region method on user-supplied normal equations.

    Minimises a loss given its gradient ``g`` and a curvature model ``H`` (usually
    ``J^T J``) at each point, like :func:`piazzi.gauss_newton`, but globalises with a trust
    region instead of damping or a line search. Each iteration solves the normal equations
    once, for the Gauss-Newton step, and computes the Cauchy step (the minimiser of the model
    along the steepest-descent direction). The trial step follows the path origin, Cauchy
    point, Gauss-Newton point as far as the trust region allows. A rejected trial costs no
    further linear solve: the radius is halved and the same path is truncated again, which
    is the method's advantage over Levenberg-Marquardt when the linear solve is expensive.

    The trust region is the ellipsoid ``|D s| <= radius``, with ``D = I``, or
    ``D = sqrt(diag(H))`` with ``damping="marquardt"`` (floored at ``1e-12 * max(diag H)``,
    as in Ceres Solver). The gain ratio ``rho`` (actual over predicted decrease) drives the
    radius, as in Ceres: a trial is accepted when ``rho > min_gain``, the radius is halved
    after an accepted step with ``rho < 0.25`` and raised to at least three times the step
    after one with ``rho > 0.75``, and halved after a rejected one.

    Evaluations: the first trial from a linearisation either evaluates the full system (the
    bet that it is accepted, which saves a second evaluation at that point) or only the
    value (when rejections are common, a rejected full evaluation wastes the gradient and
    ``H``). ``first_trial="auto"`` bets on the full system after a first trial that was
    accepted, and on the value after one that was rejected. Every later trial from the same
    linearisation is value-only, and an accepted value-only trial is followed by a full
    evaluation at that point.

    The Gauss-Newton step is solved with a ridge of ``mu * D^2`` (``mu * mean(diag H)`` for
    ``D = I``), starting from ``1e-8``: it is raised by a factor of 10 when the solve fails,
    returns a non-finite step or one that is not a descent direction (up to ``1``, beyond
    which the Cauchy step alone is used), and lowered after every accepted step. Directions
    of non-positive curvature take the steepest-descent step to the trust-region boundary.

    With ``dogleg_type="subspace"`` the trial step minimises the model over the
    two-dimensional subspace spanned by the gradient and the Gauss-Newton step, inside the
    trust region (Ceres's ``SUBSPACE_DOGLEG``): it needs two extra products with ``H`` and the
    roots of a quartic, and predicts a larger decrease than the traditional step at the same
    radius.

    The run stops, as converged, once the Gauss-Newton step from the current point is below
    ``step_tol`` (max-norm) and a step is accepted; when the gradient is exactly zero; or when
    a trial fails and either the Gauss-Newton step leaves ``x`` unmoved (every entry within 64
    units in the last place of its scale) or the model predicts less than the rounding noise,
    ``8 * eps * |loss|``, for it. These are the rules of :func:`piazzi.gauss_newton`, with the
    Gauss-Newton step, always available here, as the estimate of the distance to the minimiser;
    they end a float32 run at its minimiser. A Gauss-Newton step with a ridge above ``1e-3`` is
    no such estimate and ends no run. Where the Gauss-Newton step is no distance estimate at
    all (a singular ``H`` at the minimiser, for example ``r = x^2 + 5``), the trust region
    collapses instead, and a trial step below ``step_tol`` with a predicted decrease below the
    rounding noise, whose value did not measurably rise, ends the run as converged (message
    ``predicted decrease below rounding noise``): no step at that scale can improve the loss.
    A trial with a non-finite value is never accepted, and a
    non-finite value at ``x0`` stops the run at once.

    ``H`` may be a sparse COO or CSR tensor, as for :func:`piazzi.gauss_newton`.

    Args:
        system (Callable): ``system(x) -> (loss, g, H)``, or ``(loss, g, H, aux)`` with
            ``has_aux=True``, as for :func:`piazzi.gauss_newton`. ``H`` should be positive
            semi-definite for the trust-region guarantees; an indefinite ``H`` is tolerated.
        x0 (Tensor): starting point, of any shape. ``n`` is its number of elements, or the
            tangent size of ``manifold``.

    Keyword Args:
        dogleg_type (str, optional): ``"traditional"`` (the path through the Cauchy and
            Gauss-Newton points) or ``"subspace"`` (the minimiser of the model in their span).
            Default: ``"traditional"``
        has_aux (bool, optional): whether ``system`` also returns an auxiliary output (any
            tree of tensors), returned detached in ``result.aux`` and ``info.aux``.
            Default: ``False``
        manifold (Manifold, optional): keeps ``x`` on a manifold such as
            :class:`piazzi.SE3`; ``g`` and ``H`` are in tangent coordinates and the trust
            region bounds the tangent step. Default: ``None``
        value_fun (Callable, optional): a cheaper ``value_fun(x) -> loss`` (or
            ``(loss, aux)``) for value-only evaluations. Default: ``None``, which uses the
            loss that ``system`` returns.
        max_iter (int, optional): maximum number of iterations, accepted or rejected.
            Default: ``100``
        step_tol (float, optional): the Gauss-Newton step's max-norm below which an accepted
            step ends the run as converged. Default: ``1e-5``
        initial_radius (float, optional): initial trust-region radius, in the units of ``x``
            (of ``D s`` with ``damping="marquardt"``). Default: ``1.0``
        max_radius (float, optional): largest trust-region radius. Default: ``1e16``
        min_gain (float, optional): smallest gain ratio of an accepted step (Ceres's
            ``min_relative_decrease``). Default: ``1e-3``
        max_rejections (int, optional): stop after this many rejected trials in a row.
            Default: ``20``
        first_trial (str, optional): what the first trial from a linearisation evaluates:
            ``"system"`` (the full system), ``"value"`` (the value only) or ``"auto"`` (see
            above). Default: ``"auto"``
        damping (str, optional): ``"identity"`` (``D = I``) or ``"marquardt"``
            (``D = sqrt(diag(H))``). Default: ``"identity"``
        solve_dtype (torch.dtype, optional): dtype of the linear solve.
            Default: ``torch.float64``
        solve_device (str or torch.device, optional): device of the linear solve.
            Default: ``"cpu"``
        linear_solver (LinearSolver, optional): ``solver(H, g, damping) -> d``, as for
            :func:`piazzi.gauss_newton`. The Gauss-Newton step should be accurate, so use an
            exact solver (the default for a dense ``H``) or a tight ``tol`` for
            :func:`piazzi.cg_solver`. Default: ``None``
        callback (Callable, optional): called with an :class:`piazzi.IterationInfo` after
            every iteration, accepted or not; returning ``True`` stops the run.
            Default: ``None``
        verbose (bool or int, optional): print a table of the run's progress with
            :func:`piazzi.print_progress`. Default: ``False``

    Returns:
        OptimizeResult: as for :func:`piazzi.gauss_newton`.

    See Also:
        - :func:`piazzi.gauss_newton`: Gauss-Newton and Levenberg-Marquardt on the same systems.
        - :func:`piazzi.least_squares`: the high-level interface, which runs this method with
          ``method="dogleg"``.

    Example::

        >>> t = torch.linspace(0, 1, 20, dtype=torch.float64)
        >>> y = 2 * torch.exp(-3 * t)
        >>> system = piazzi.least_squares_system(lambda p: p[0] * torch.exp(-p[1] * t) - y)
        >>> res = piazzi.dogleg(system, torch.tensor([1.0, 1.0], dtype=torch.float64))
        >>> res.x
        tensor([2.0000, 3.0000], dtype=torch.float64)
    """
    if dogleg_type not in ("traditional", "subspace"):
        raise ValueError(f"dogleg_type must be 'traditional' or 'subspace', got {dogleg_type!r}")
    if damping not in ("identity", "marquardt"):
        raise ValueError(f"damping must be 'identity' or 'marquardt', got {damping!r}")
    if first_trial not in ("auto", "system", "value"):
        raise ValueError(f"first_trial must be 'auto', 'system' or 'value', got {first_trial!r}")
    if not (initial_radius > 0 and max_radius > 0):
        raise ValueError("initial_radius and max_radius must be positive")
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
        return loss.detach(), detach_tree(aux), g.detach().reshape(-1).clone(), H.detach().clone()

    def eval_value(xv):
        with torch.no_grad():
            loss = value_fun(xv.reshape(shape)) if value_fun is not None else system(xv.reshape(shape))[0]
        if isinstance(loss, tuple):
            loss = loss[0]
        counts["f"] += 1
        return loss.detach()

    def all_finite(t):
        values = t if not is_sparse(t) else (t.coalesce().values() if t.layout == torch.sparse_coo else t.values())
        return bool(torch.isfinite(values).all())

    def host(t):
        return transfer(t, solve_device, solve_dtype)

    def noise(f_val):
        return 8 * torch.finfo(f.dtype).eps * abs(f_val)

    def below_noise(pred, f_new, f_val):
        return pred <= noise(f_val) and math.isfinite(f_new) and f_new <= f_val + noise(f_val)

    def predicted_decrease(Hc, gc, d):
        return -(torch.dot(gc, d) + 0.5 * torch.dot(d, matvec(Hc, d))).item()

    solvers = {}

    def solve(Hc, gc, ridge):
        if linear_solver is not None:
            solver = linear_solver
        else:
            key = is_sparse(Hc)
            if key not in solvers:
                solvers[key] = cg_solver() if key else dense_solver
            solver = solvers[key]
        with torch.no_grad():
            return solver(Hc, gc, ridge)

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

    mu_state = [MIN_MU]  # the ridge: lowered on acceptance, raised on a failed solve

    class Model:
        """What one linearisation gives, reused by every trial of the iteration."""

        def __init__(self, Hc, gc):
            self.H, self.g = Hc, gc
            diag = diagonal(Hc)
            if damping == "identity":
                self.scale = torch.ones_like(gc)
                ridge_unit = torch.full_like(gc, max(diag.mean().item(), torch.finfo(gc.dtype).tiny))
            else:
                self.scale = diag.clamp(min=1e-12 * diag.max().item()).sqrt()
                ridge_unit = self.scale**2
            self.gs = gc / self.scale
            g_norm2 = torch.dot(self.gs, self.gs).item()
            u = self.gs / self.scale
            curvature = torch.dot(u, matvec(Hc, u)).item()
            self.alpha = g_norm2 / curvature if curvature > 0 else math.inf
            self.gn = self.gn_pred = self.gns = None
            self.mu_used = math.inf
            while mu_state[0] < MAX_MU:
                d = solve(Hc, gc, mu_state[0] * ridge_unit)
                if all_finite(d) and torch.dot(gc, d).item() < 0:
                    self.gn, self.mu_used = d, mu_state[0]
                    self.gns = d * self.scale
                    self.gn_pred = predicted_decrease(Hc, gc, d)
                    break
                mu_state[0] *= MU_FACTOR
            self.subspace = None

        def distance_step(self):
            """The Gauss-Newton step if its ridge is light enough to estimate the distance to the minimiser."""
            return self.gn if self.mu_used <= MAX_DISTANCE_MU else None

        def hess_scaled(self, v):
            return matvec(self.H, v / self.scale) / self.scale

        def step(self, radius):
            """Trial step in ``x`` coordinates and its norm in scaled coordinates."""
            ss = None
            if dogleg_type == "subspace" and self.gns is not None:
                if self.gns.norm().item() <= radius:
                    ss = self.gns
                else:
                    if self.subspace is None:
                        self.subspace = _Subspace(self.gs, self.gns, self.hess_scaled)
                    ss = self.subspace.step(self.gs, radius)
            if ss is None:
                ss = _traditional_step(self.gs, self.gns, self.alpha, radius)
            return ss / self.scale, ss.norm().item()

    f, aux, g, H = eval_sys(x)
    if not torch.isfinite(f):
        return result(0, False, "objective is not finite at x0")
    if g.numel() == 0:
        return result(0, True, "no free parameters")
    gc, Hc = host(g), host(H)
    radius = min(initial_radius, max_radius)
    new_model, rejections = True, 0
    model: Model | None = None
    gn_unmoved: bool | None = None
    # Whether the first trial from a linearisation evaluates the full system: a bet that it is accepted.
    full_first = first_trial != "value"
    aux_new, g_new, H_new = aux, g, H
    for it in range(max_iter):
        f_val, f_start = f.item(), f
        if new_model:
            if not (all_finite(gc) and all_finite(Hc)):
                return result(it, False, "gradient or curvature is not finite")
            if gc.abs().max().item() == 0:
                return result(it, True, "gradient is zero")
            model, gn_unmoved, new_model = Model(Hc, gc), None, False
        assert model is not None
        s, step_norm = model.step(radius)
        pred = predicted_decrease(Hc, gc, s)
        x_new = retract(manifold, x, transfer(s, x.device, x.dtype))
        moved = not torch.equal(x_new, x)
        if not moved:  # nothing to evaluate: a step that does not change x is never accepted
            f_new, have_sys = f, False
        elif rejections == 0 and full_first:
            f_new, aux_new, g_new, H_new = eval_sys(x_new)
            have_sys = True
        else:
            f_new, have_sys = eval_value(x_new), False
        fn = f_new.item()
        rho = (f_val - fn) / max(pred, 1e-20)
        small = s.abs().max().item() < step_tol and fn <= f_val + noise(f_val)
        # No measurable decrease from a trial step below step_tol: no step at that scale can improve
        # the loss. This ends runs whose Gauss-Newton step is no distance estimate (a singular H at the
        # minimiser), where the trust region collapses instead.
        stalled = small and math.isfinite(fn) and pred <= noise(f_val)
        accept = moved and math.isfinite(fn) and (rho > min_gain or (pred <= 0 and fn <= f_val) or small)
        if first_trial == "auto" and rejections == 0:
            full_first = accept  # bet on the full system after a first trial that was accepted, else value-only
        if accept:
            if not have_sys:
                f_new, aux_new, g_new, H_new = eval_sys(x_new)
            if rho < 0.25:
                radius *= 0.5
            elif rho > 0.75:
                radius = max(radius, 3.0 * step_norm)
            radius = min(radius, max_radius)
            mu_state[0] = max(MIN_MU, 2.0 * mu_state[0] / MU_FACTOR)
            # The Gauss-Newton step from the old point estimates the distance to the minimiser.
            d_gn = model.distance_step()
            converged = d_gn is not None and d_gn.abs().max().item() < step_tol
            x, f, aux = x_new, f_new, aux_new
            gc, Hc = host(g_new), host(H_new)
            new_model, rejections = True, 0
            stop = notify(it, f_start, True)
            if converged:
                return result(it + 1, True, "step below step_tol")
            if stalled:
                return result(it + 1, True, "predicted decrease below rounding noise")
            if stop:
                return result(it + 1, False, "stopped by callback")
        else:
            radius *= 0.5
            rejections += 1
            stop = notify(it, f_start, False)
            if stalled:
                return result(it + 1, True, "predicted decrease below rounding noise")
            d_gn = model.distance_step()
            if d_gn is not None and math.isfinite(fn):
                if gn_unmoved is None:
                    gn_x = retract(manifold, x, transfer(d_gn, x.device, x.dtype))
                    gn_unmoved = unmoved(manifold, x, gn_x)
                if gn_unmoved:
                    return result(it + 1, True, "step below the resolution of x")
                if below_noise(model.gn_pred, fn, f_val):
                    return result(it + 1, True, "predicted decrease below rounding noise")
            if rejections >= max_rejections:
                return result(it + 1, False, "too many rejected steps in a row")
            if stop:
                return result(it + 1, False, "stopped by callback")
    return result(max_iter, False, "max_iter reached")
