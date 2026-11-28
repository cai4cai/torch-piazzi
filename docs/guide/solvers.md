# Solvers

piazzi has two layers of solvers:

- **High-level interfaces**, modelled on SciPy: {func}`~piazzi.least_squares` and {func}`~piazzi.minimize`. They take the objective in its most natural form (residuals or a scalar), choose a method by name, and pass the method's settings through. Start here.
- **Method-specific solvers**: {func}`~piazzi.gauss_newton`, {func}`~piazzi.levenberg_marquardt` and {func}`~piazzi.lbfgs`. These are the algorithms the high-level interfaces call, with every setting exposed. Call them directly when you build the curvature yourself, or want one method's full control.

## High-level interfaces

| Function | Objective | Methods |
|---|---|---|
| {func}`~piazzi.least_squares` | residuals `r(x)`, with an optional robust loss | Like {func}`scipy.optimize.least_squares`. `method="lm"` (default, Levenberg-Marquardt) or `"gn"` (Gauss-Newton), run by {func}`~piazzi.gauss_newton` on the normal equations that {func}`~piazzi.least_squares_system` builds by forward-mode autodiff. Other keyword arguments (`max_iter`, `step_tol`, `damping`, `linear_solver`, ...) go to {func}`~piazzi.gauss_newton`. |
| {func}`~piazzi.minimize` | scalar `f(x)`, gradient from autograd | Like {func}`scipy.optimize.minimize`. `method` names a solver in {data}`piazzi.MINIMIZE_METHODS` (currently `"lbfgs"`, the default), with its settings in `options`. `method` can also be a `torch.optim` optimiser class (see {doc}`torch-optim`). |

Both accept `manifold=` (see {doc}`manifolds`), `has_aux=` and `callback=`, and return an {class}`~piazzi.OptimizeResult`. {func}`piazzi.scipy.least_squares` and {func}`piazzi.scipy.minimize` have the same calling conventions and run SciPy's methods instead (see {doc}`backends`).

## Method-specific solvers

| Function | Needs | Method |
|---|---|---|
| {func}`~piazzi.gauss_newton` with `method="lm"`, or {func}`~piazzi.levenberg_marquardt` | `system(x) -> (loss, g, H)` | Levenberg-Marquardt with Nielsen's damping update. |
| {func}`~piazzi.gauss_newton` with `method="gn"` | `system(x) -> (loss, g, H)` | Gauss-Newton with Armijo backtracking. |
| {func}`~piazzi.lbfgs` | `f(x)`, gradient from autograd | L-BFGS with Armijo backtracking (the default) or a strong Wolfe line search (`line_search="strong_wolfe"`). With Armijo it usually costs one evaluation per iteration. Each trial point is evaluated with its autograd graph, and the backward pass runs only at the accepted point, so a rejected trial costs a forward pass. With strong Wolfe every trial needs its gradient, so a rejected trial costs a value-and-gradient evaluation. |

{func}`~piazzi.least_squares_system` is the building block between the two layers: it turns residuals into the `system(x) -> (loss, g, H)` that {func}`~piazzi.gauss_newton` needs, that is `(0.5‖r‖², Jᵀr, JᵀJ)` or its robust equivalent.

## Choosing a solver

- **You have residuals** (a model minus data, reprojection errors, point-to-point distances): use {func}`~piazzi.least_squares`. Gauss-Newton curvature usually converges in far fewer evaluations than a quasi-Newton method.
- **You have a scalar objective** that isn't a sum of squares (a likelihood, an image similarity measure): use {func}`~piazzi.minimize`.
- **You build the normal equations yourself** (for example per-observation Jacobians assembled into a sparse `JᵀJ`): pass a `system(x) -> (loss, g, H)` to {func}`~piazzi.gauss_newton`. See {doc}`sparse`.

`method="lm"` (the default) adapts its damping to how well the model predicts each step, which makes it the safer choice. `method="gn"` takes the undamped step with a backtracking line search; with SciPy's curvature model and mostly outlying residuals it copes better than Levenberg-Marquardt (see {doc}`least-squares`).

## L-BFGS line searches

{func}`~piazzi.lbfgs` offers three line searches through `line_search`:

- `"armijo"` (the default) backtracks until the sufficient-decrease condition holds. A rejected trial costs a forward pass only (`n_fun`), and the backward pass runs once, at the accepted point. This is the cheapest choice when the objective is expensive, but `sᵀy > 0` is not guaranteed, so pairs with non-positive curvature are skipped.
- `"strong_wolfe"` brackets and zooms with cubic interpolation until the strong Wolfe conditions hold (`c1` and `c2`), as in {class}`torch.optim.LBFGS` and Ceres. The curvature condition keeps every pair positive, but each trial point needs its gradient, so a rejected trial costs a full `n_fun_grad` and `value_fun` is ignored.

- `"hager_zhang"` is the line search of CG_DESCENT (Hager and Zhang, 2006): bracketing, then secant steps and bisection. It accepts a point that satisfies the Wolfe conditions or the *approximate* Wolfe conditions, which compare slopes instead of values, so they stay meaningful near a minimiser where the loss differences are lost in rounding noise (typically in float32, or near a zero residual). `hz_epsilon` (default `1e-6`) bounds how far such a step may raise the loss, relative to the loss; `hz_epsilon=0` keeps every step monotone. Like strong Wolfe, each trial costs an `n_fun_grad`, and `value_fun` is ignored.

On the Rosenbrock function from `(-1.2, 1)`, strong Wolfe needs fewer iterations than Armijo (36 against 42) but not fewer evaluations (45 value-and-gradient evaluations against 43 plus 7 forward passes), and it matches the counts of Ceres and of PyTorch's L-BFGS (see the {doc}`Rosenbrock notebook <../examples/rosenbrock>`). Hager-Zhang needs the fewest iterations (34) and the most evaluations (49). Try the Wolfe searches on non-convex problems where Armijo skips many pairs, and Hager-Zhang where rounding noise stops the other searches near the minimiser; keep Armijo when a rejected trial must stay cheap.

## Damping and step limits

`damping="marquardt"` scales the Levenberg-Marquardt damping (and the Gauss-Newton ridge) by `diag(H)`, as Ceres does. Use it when parameters have very different units; it plays the role of SciPy's `x_scale="jac"`.

Steps are capped so that no coordinate moves more than `max_step` (default 1). Set it for the units of your problem: a large translation in millimetres needs a larger cap.

## Results and stopping

Every solver returns an {class}`~piazzi.OptimizeResult` with `x`, `fun`, `aux`, evaluation counts, `converged` and `message`. `converged` means a minimiser was reached to working precision: the step fell below `step_tol`, or the gradient is zero. The step judged against `step_tol` is one that estimates the distance to the minimiser (the full Gauss-Newton or L-BFGS step, or Levenberg-Marquardt's undamped Gauss-Newton step), so heavy damping (a large `tau`) or a long backtracking line search cannot end a run early, and a step too small to change `x` at all is never accepted. In float32, Gauss-Newton, Levenberg-Marquardt and L-BFGS also recognise a minimiser once the predicted decrease falls below the rounding noise of the loss, or once a Newton-like step no longer moves `x` beyond a few units in the last place of each entry (message `step below the resolution of x`). The second rule catches minimisers near a zero residual, where the loss's rounding noise is absolute rather than relative to the loss.

The evaluation counts are split by kind, because their costs differ: `n_fun_grad` (value and gradient), `n_fun` (value only, as in a line search) and `n_system` (normal equations).

## Callbacks

`callback(info)` receives an {class}`~piazzi.IterationInfo` after every iteration, accepted or not: the current point and value, the value at the start of the iteration, whether the step was accepted, and the evaluation counts so far (`info.n_fun_grad`, `info.n_fun`, `info.n_system`). Plot progress against those counts rather than iterations, which hide the evaluations a line search spends. Return `True` to stop the run.

To watch a run, pass `verbose=True` to any solver or wrapper. It prints a table of the run, one row per iteration, in the spirit of Ceres Solver's per-iteration summary, and your own `callback` still runs after it:

```python
t = torch.linspace(0, 1, 20, dtype=torch.float64)
y = 2 * torch.exp(-3 * t)
x0 = torch.tensor([1.0, 1.0], dtype=torch.float64)
res = piazzi.least_squares(lambda p: p[0] * torch.exp(-p[1] * t) - y, x0, verbose=True)
```

```text
 iter            fun  rel_decrease      step  n_fun_grad   n_fun  n_system
    0   3.876612e-01     7.208e-01  accepted           0       0         2
    1   3.996416e-03     9.897e-01  accepted           0       0         3
    ...
```

Pass `verbose=10` to print every tenth iteration. The table comes from {func}`piazzi.print_progress`, a ready-made callback: use `callback=piazzi.print_progress(file=...)` to write it elsewhere than standard output.
