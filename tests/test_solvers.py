import pytest
import torch

import piazzi


def rosenbrock(x):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


def exp_residuals(theta, t, y):
    return theta[0] * torch.exp(-theta[1] * t) - y


@pytest.fixture
def exp_data():
    t = torch.linspace(0, 4, 40, dtype=torch.float64)
    y = 2.5 * torch.exp(-1.3 * t)
    return t, y


def test_lbfgs_armijo_rosenbrock():
    x0 = torch.tensor([-1.2, 1.0], dtype=torch.float64)
    res = piazzi.lbfgs_armijo(rosenbrock, x0, max_iter=500, step_tol=1e-10)
    assert torch.allclose(res.x, torch.ones(2, dtype=torch.float64), atol=1e-4)
    assert res.n_fun_grad > 0


def test_lbfgs_armijo_keeps_shape_aux_and_callback():
    target = torch.arange(6, dtype=torch.float64).reshape(2, 3)

    def fun(x):
        loss = ((x - target) ** 2).sum()
        return loss, {"half": 0.5 * loss}

    seen = []
    res = piazzi.lbfgs_armijo(
        fun,
        torch.zeros(2, 3, dtype=torch.float64),
        has_aux=True,
        max_step=10.0,
        callback=lambda info: seen.append(info.iteration),
    )
    assert res.x.shape == (2, 3)
    assert torch.allclose(res.x, target, atol=1e-6)
    assert torch.allclose(res.aux["half"], 0.5 * res.fun)
    assert seen == list(range(res.n_iter))


def test_lbfgs_callback_stop():
    res = piazzi.lbfgs_armijo(rosenbrock, torch.tensor([-1.2, 1.0]), max_iter=100, callback=lambda info: True)
    assert res.n_iter == 1 and res.message == "stopped by callback"


@pytest.mark.parametrize("method", ["gn", "lm"])
def test_gauss_newton_exponential_fit(method, exp_data):
    t, y = exp_data
    system = piazzi.least_squares_system(lambda th: exp_residuals(th, t, y))
    res = piazzi.gauss_newton(system, torch.tensor([1.0, 0.5], dtype=torch.float64), method=method, step_tol=1e-10)
    assert torch.allclose(res.x, torch.tensor([2.5, 1.3], dtype=torch.float64), atol=1e-6)
    assert res.converged, res.message


def test_gauss_newton_linear_one_step():
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 7.0]], dtype=torch.float64)
    b = torch.tensor([1.0, 0.0, 2.0], dtype=torch.float64)
    system = piazzi.least_squares_system(lambda x: A @ x - b)
    res = piazzi.gauss_newton(system, torch.zeros(2, dtype=torch.float64), method="gn", max_step=100.0, gn_ridge=0.0)
    expected = torch.linalg.lstsq(A, b).solution
    assert torch.allclose(res.x, expected, atol=1e-10)
    assert res.n_iter <= 2


def test_least_squares_system_matches_autograd(exp_data):
    t, y = exp_data
    theta = torch.tensor([1.7, 0.9], dtype=torch.float64)
    loss, g, H = piazzi.least_squares_system(lambda th: exp_residuals(th, t, y))(theta)
    th = theta.clone().requires_grad_(True)
    ref = 0.5 * (exp_residuals(th, t, y) ** 2).sum()
    (g_ref,) = torch.autograd.grad(ref, th)
    assert torch.allclose(loss, ref.detach())
    assert torch.allclose(g, g_ref)
    assert torch.allclose(H, H.T) and torch.linalg.eigvalsh(H).min() >= 0


def test_lm_float32_objective_value_fun(exp_data):
    t, y = (v.float() for v in exp_data)
    calls = {"value": 0}

    def value_fun(th):
        calls["value"] += 1
        return 0.5 * (exp_residuals(th, t, y) ** 2).sum()

    system = piazzi.least_squares_system(lambda th: exp_residuals(th, t, y))
    res = piazzi.levenberg_marquardt(system, torch.tensor([1.0, 0.5]), value_fun=value_fun)
    assert res.x.dtype == torch.float32
    assert torch.allclose(res.x, torch.tensor([2.5, 1.3]), atol=1e-3)
    assert calls["value"] == res.n_fun


def test_bad_method():
    with pytest.raises(ValueError, match="method must be"):
        piazzi.gauss_newton(lambda x: None, torch.zeros(2), method="dogleg")


# --- Regression tests from the code review of the solvers ----------------------------------


def rosen_nd(x):
    return (100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2).sum()


@pytest.mark.parametrize("scale", [1.0, 1e-3, 1e-6, 1e-9])
def test_lbfgs_armijo_is_scale_invariant(scale):
    """Absolute thresholds made tiny objectives stall (1e-6) or stop at x0 as converged (1e-9)."""
    x0 = torch.tensor([-1.2, 1.0], dtype=torch.float64)
    res = piazzi.lbfgs_armijo(lambda x: scale * rosen_nd(x), x0, max_iter=500, step_tol=1e-9)
    assert res.converged, res.message
    assert torch.allclose(res.x, torch.ones(2, dtype=torch.float64), atol=1e-5)


def test_lbfgs_armijo_tiny_gradient_is_not_converged_at_x0():
    target = torch.tensor([3.0, -2.0], dtype=torch.float64)
    res = piazzi.lbfgs_armijo(lambda x: 1e-7 * ((x - target) ** 2).sum(), torch.zeros(2, dtype=torch.float64))
    assert res.converged and torch.allclose(res.x, target, atol=1e-6), (res.x, res.message)


def test_lbfgs_armijo_stops_on_a_non_finite_gradient():
    # sqrt((x - 1)^2) has a NaN gradient exactly at its minimiser, which the first step hits.
    res = piazzi.lbfgs_armijo(lambda x: ((x - 1) ** 2).sqrt().sum(), torch.zeros(1, dtype=torch.float64))
    assert res.message == "gradient is not finite" and res.n_fun_grad == 2 and res.n_fun == 0
    assert res.x.item() == 1.0 and res.fun.item() == 0.0


def test_lbfgs_armijo_does_not_repeat_a_failed_steepest_descent_search():
    points = []

    def wrong_gradient(x):  # value 2 |x|^2, but gradient -2 x: every search direction goes uphill
        points.append(x.detach().clone())
        return 3 * (x.detach() ** 2).sum() - (x**2).sum()

    res = piazzi.lbfgs_armijo(wrong_gradient, torch.ones(2, dtype=torch.float64))
    assert not res.converged and res.message == "line search failed"
    assert len({tuple(p.tolist()) for p in points}) == len(points)


def test_lbfgs_armijo_at_the_minimiser_evaluates_once():
    res = piazzi.lbfgs_armijo(rosen_nd, torch.ones(3, dtype=torch.float64))
    assert res.converged and res.message == "gradient is zero" and res.n_fun_grad == 1


@pytest.mark.parametrize("n", [2, 5, 10])
def test_lbfgs_armijo_float32_reports_convergence(n):
    x0 = torch.tensor([-1.2, 1.0]).repeat(n // 2) if n % 2 == 0 else torch.zeros(n)
    res = piazzi.lbfgs_armijo(rosen_nd, x0, max_iter=1000)
    assert res.converged, res.message
    assert torch.allclose(res.x, torch.ones(n), atol=1e-4)


@pytest.mark.parametrize("solver", ["lbfgs", "gn"])
def test_value_fun_may_return_aux(solver, exp_data):
    t, y = exp_data

    def fun_aux(th):
        r = exp_residuals(th, t, y)
        return 0.5 * (r**2).sum(), {"r0": r[0]}

    x0 = torch.tensor([1.0, 0.5], dtype=torch.float64)
    if solver == "lbfgs":
        res = piazzi.lbfgs_armijo(fun_aux, x0, has_aux=True, value_fun=fun_aux, lr=50.0, max_step=100.0, max_iter=500)
    else:
        system = piazzi.least_squares_system(lambda th: (exp_residuals(th, t, y), {"r0": th[0]}), has_aux=True)
        x0 = torch.tensor([50.0, 10.0], dtype=torch.float64)  # far away: the line search backtracks
        res = piazzi.gauss_newton(system, x0, has_aux=True, method="gn", value_fun=fun_aux, max_step=10.0)
    assert res.n_fun > 0, "the line search never used value_fun"
    assert torch.allclose(res.x, torch.tensor([2.5, 1.3], dtype=torch.float64), atol=1e-4)


@pytest.mark.parametrize("method", ["gn", "lm"])
def test_gauss_newton_copes_with_a_system_that_reuses_its_buffers(method, exp_data):
    t, y = exp_data
    fresh = piazzi.least_squares_system(lambda th: exp_residuals(th, t, y))
    g_buf, H_buf = torch.empty(2, dtype=torch.float64), torch.empty(2, 2, dtype=torch.float64)

    def buffered(th):
        loss, g, H = fresh(th)
        g_buf.copy_(g)
        H_buf.copy_(H)
        return loss, g_buf, H_buf

    x0 = torch.tensor([50.0, 10.0], dtype=torch.float64)
    a = piazzi.gauss_newton(fresh, x0, method=method, max_step=10.0, step_tol=1e-10)
    b = piazzi.gauss_newton(buffered, x0, method=method, max_step=10.0, step_tol=1e-10)
    assert torch.equal(a.x, b.x) and a.n_iter == b.n_iter and a.message == b.message


@pytest.mark.parametrize("method", ["gn", "lm"])
def test_gauss_newton_stops_on_non_finite_curvature(method):
    def system(x):
        return (x**2).sum(), 2 * x, torch.full((2, 2), float("nan"), dtype=x.dtype)

    res = piazzi.gauss_newton(system, torch.ones(2, dtype=torch.float64), method=method)
    assert not res.converged and res.message == "gradient or curvature is not finite"
    assert res.n_system == 1 and res.n_fun == 0


def test_gauss_newton_never_steps_uphill_with_an_indefinite_curvature():
    def system(x):  # f = |x|^2 with a negative-definite "curvature"
        return (x**2).sum(), 2 * x, -torch.eye(2, dtype=x.dtype)

    x0 = torch.tensor([1.0, -2.0], dtype=torch.float64)
    values = []
    res = piazzi.gauss_newton(system, x0, method="gn", callback=lambda info: values.append(info.fun.item()))
    assert all(b <= a for a, b in zip([5.0, *values], values, strict=False)), values
    assert res.fun.item() < 5.0


def test_least_squares_aux_may_hold_non_tensors(exp_data):
    t, y = exp_data

    def fun(th):
        r = exp_residuals(th, t, y)
        return r, {"name": "fit", "n": 3, "worst": r.abs().max(), "nested": [r[0], None]}

    res = piazzi.least_squares(fun, torch.tensor([1.0, 0.5], dtype=torch.float64), has_aux=True)
    assert res.aux["name"] == "fit" and res.aux["n"] == 3 and res.aux["nested"][1] is None
    assert torch.equal(res.aux["worst"], exp_residuals(res.x, t, y).abs().max())


# --- Issue #19: float32 runs that reach a minimiser must report convergence ----------------

FLOAT32_ROSEN_STARTS = {
    # Found among random starts in [-2, 2]: each ended "line search failed twice in a row" at a
    # minimiser accurate to float32 precision. The values are exact float32 numbers.
    "2d": [-1.9106969833374023, -1.3245642185211182],
    "4d": [0.07636308670043945, 0.463409423828125, 1.240753173828125, 1.9203882217407227],
    "10d-local": [  # the local minimum near (-1, 1, ..., 1), f ~ 3.99
        -0.701124906539917,
        1.5627131462097168,
        -0.3598811626434326,
        0.7786250114440918,
        0.3552436828613281,
        0.8509364128112793,
        -0.6796550750732422,
        0.9751427173614502,
        -1.396941900253296,
        0.4517099857330322,
    ],
}


@pytest.mark.parametrize("start", FLOAT32_ROSEN_STARTS, ids=list(FLOAT32_ROSEN_STARTS))
def test_lbfgs_armijo_float32_converges_at_the_minimiser(start):
    x0 = torch.tensor(FLOAT32_ROSEN_STARTS[start])
    res = piazzi.lbfgs_armijo(rosen_nd, x0, max_iter=3000)
    ref = piazzi.lbfgs_armijo(rosen_nd, x0.double(), max_iter=3000, step_tol=1e-12)
    assert res.converged, res.message
    # A minimiser is determined to about sqrt(eps) of the dtype.
    assert torch.allclose(res.x.double(), ref.x, atol=3 * torch.finfo(torch.float32).eps ** 0.5)


def registration(seed, dtype):
    """Points ~100 from the origin; data perturbed by +-1e-5 (about an ulp) to mimic how float32
    rounding differs across platforms. The residual is zero up to that rounding."""
    g = torch.Generator().manual_seed(0)
    P = 5 * torch.randn(30, 3, generator=g, dtype=torch.float64) + torch.tensor([100.0, 80.0, -60.0])
    T_true = piazzi.se3_exp(torch.tensor([3.0, -2.0, 5.0, 0.9, -1.2, 0.6], dtype=torch.float64))
    P = P.float() + 1e-5 * torch.randint(-1, 2, P.shape, generator=torch.Generator().manual_seed(seed))
    Q = (P.double() @ T_true[:3, :3].T + T_true[:3, 3]).float()
    P, Q = P.to(dtype), Q.to(dtype)
    return (lambda T: (P @ T[:3, :3].T + T[:3, 3] - Q).reshape(-1)), piazzi.SE3(pivot=P.mean(0)), T_true


# Seeds 0-7, plus those where L-BFGS once ended on a steepest-descent failure after a
# steepest-descent step had succeeded on noise (58, 63, 113, 126, 189), or one step short of
# the resolution of x at 32 ulps (185).
@pytest.mark.parametrize("seed", [*range(8), 58, 63, 113, 126, 185, 189])
@pytest.mark.parametrize("solver", ["lm", "gn", "lbfgs"])
def test_float32_registration_converges_near_a_zero_residual(solver, seed):
    """Near a zero residual the loss's rounding noise is absolute: the relative noise test cannot
    see it, and LM rejected (L-BFGS failed) every step at the minimiser before #19."""
    residuals, manifold, T_true = registration(seed, torch.float32)
    T0 = torch.eye(4)
    if solver == "lbfgs":
        options = {"max_iter": 2000, "step_tol": 1e-6}
        res = piazzi.minimize(lambda T: 0.5 * residuals(T).square().sum(), T0, manifold=manifold, options=options)
    else:
        res = piazzi.least_squares(residuals, T0, manifold=manifold, method=solver, max_step=1e3)
    assert res.converged, res.message
    torch.testing.assert_close(res.x.double(), T_true, atol=1e-3, rtol=1e-3)


def pinned_quadratic(x0, minimum, penalty):
    """A (loss, g, H) system for 0.5 |x - minimum|^2, except that the loss gets ``penalty``
    added anywhere but at ``x0``: every step is then rejected, as at a noise floor."""

    def system(x):
        r = x - minimum
        loss = 0.5 * (r * r).sum() + (0.0 if torch.equal(x, x0) else penalty)
        return torch.as_tensor(loss, dtype=x.dtype), r, torch.eye(x.numel(), dtype=x.dtype)

    return system


@pytest.mark.parametrize("method", ["gn", "lm"])
def test_resolution_rule_measures_each_parameter_at_its_own_scale(method):
    """With a large parameter next to a small one, a step that is tiny next to max|x| can still
    move the small one: the run must not report convergence (review of #25)."""
    x0 = torch.tensor([1e4, 1.0])
    system = pinned_quadratic(x0, torch.tensor([1e4, 1.001]), penalty=1.0)
    res = piazzi.gauss_newton(system, x0, method=method)
    assert not res.converged, res.message


def test_resolution_rule_ignores_heavily_damped_lm_trials():
    """With heavy damping the first trial is tiny however far the minimiser is: the undamped
    Gauss-Newton step must be checked before reporting convergence (review of #25)."""
    x0 = torch.tensor([1.0, 1.0], dtype=torch.float64)
    system = pinned_quadratic(x0, torch.tensor([2.0, 2.0], dtype=torch.float64), penalty=1.0)
    # tau = 1e15: the first trial moves x by ~1e-15, a few ulps. Two rejections keep the trials
    # above half an ulp, so that x does move (a zero step is another matter: step_tol).
    res = piazzi.gauss_newton(system, x0, method="lm", tau=1e15, max_rejections=2)
    assert not res.converged and res.message == "too many rejected steps in a row", res.message


@pytest.mark.parametrize("method", ["gn", "lm"])
def test_resolution_rule_needs_a_finite_trial_value(method):
    """A step that leaves x unmoved but lands on a NaN must not count as convergence."""
    x0 = torch.tensor([1.0, 1.0], dtype=torch.float64)
    minimum = x0 + 2 * torch.finfo(torch.float64).eps
    system = pinned_quadratic(x0, minimum, penalty=float("nan"))
    # Judge the first (full) step alone: no backtracking, a single rejection.
    res = piazzi.gauss_newton(system, x0, method=method, max_backtracks=0, max_rejections=1)
    assert not res.converged, res.message


def test_float32_zero_valued_parameter_converges():
    """An offset whose true value is 0 has no scale of its own: without a floor on the entry
    scale, the resolution rule could never fire (review of #25)."""
    g = torch.Generator().manual_seed(1)
    t = torch.linspace(0, 4, 40)
    a, b = 1 + 3 * torch.rand(1, generator=g).item(), 0.5 + torch.rand(1, generator=g).item()
    y = a * torch.exp(-b * t)  # a zero residual at (a, b, 0)
    res = piazzi.least_squares(lambda th: th[0] * torch.exp(-th[1] * t) + th[2] - y, torch.tensor([1.0, 1.0, 0.5]))
    assert res.converged, res.message
    assert abs(res.x[0].item() - a) < 1e-3 * a and abs(res.x[2].item()) < 1e-3


def scripted(values, gradients):
    """An objective whose value and gradient at the k-th call are values[k] and gradients[k],
    wherever it is evaluated: it scripts the line search's decisions exactly."""
    calls = []

    def fun(x):
        k = min(len(calls), len(values) - 1)
        calls.append(k)
        g = torch.tensor(gradients[k], dtype=x.dtype)
        return values[k] + (g * (x - x.detach())).sum()

    return fun


def test_lbfgs_stall_verdict_does_not_survive_real_progress(monkeypatch):
    """A failed step from memory may show a minimiser, and that verdict is kept for a later
    steepest-descent failure, but only near where it was given (review of #25)."""
    monkeypatch.setattr("piazzi._lbfgs.unmoved", lambda *args: True)  # every step from memory "unmoved"
    values = [10.0, 5.0] + [6.0] * 11 + [1.0] + [2.0] * 11
    gradients = [[1.0, 0.0], [0.5, 0.0]] + [[0.5, 0.0]] * 11 + [[0.5, 1.0]] + [[0.5, 1.0]] * 11
    # 0: x0. 1: steepest step, accepted (a curvature pair). 2-12: a step from memory and its
    # backtracks fail: verdict. 13: the steepest-descent retry moves x by 0.5 and is accepted,
    # real progress; its pair is dropped. 14-24: the next steepest-descent search fails.
    res = piazzi.lbfgs_armijo(scripted(values, gradients), torch.zeros(2, dtype=torch.float64))
    assert res.message == "line search failed" and not res.converged, res.message
