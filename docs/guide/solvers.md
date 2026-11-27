# Solvers

piazzi has two families of native solvers: L-BFGS for a scalar objective, and Gauss-Newton / Levenberg-Marquardt for least squares, where curvature comes from the normal equations.

| Function | Needs | Notes |
|---|---|---|
| {func}`~piazzi.lbfgs_armijo` | value + gradient (autograd) | L-BFGS with Armijo backtracking. It usually costs one evaluation per iteration. Each trial point is evaluated with its autograd graph, and the backward pass runs only at the accepted point, so a rejected trial costs a forward pass. |
| {func}`~piazzi.minimize` | value + gradient (autograd) | Like {func}`scipy.optimize.minimize`: picks a scalar minimiser by name ({data}`piazzi.MINIMIZE_METHODS`; currently only `"lbfgs_armijo"`) and passes it `options`. `method` can also be a `torch.optim` optimiser class (see {doc}`torch-optim`). |
| {func}`~piazzi.gauss_newton` with `method="gn"` | `(loss, g, H)` | Gauss-Newton with Armijo backtracking. |
| {func}`~piazzi.gauss_newton` with `method="lm"`, or {func}`~piazzi.levenberg_marquardt` | `(loss, g, H)` | Levenberg-Marquardt with Nielsen's damping update. |
| {func}`~piazzi.least_squares` | residuals `r(x)` | Like {func}`scipy.optimize.least_squares`: Levenberg-Marquardt (or `method="gn"`) on the normal equations from {func}`~piazzi.least_squares_system`, with an optional robust loss. |
| {func}`~piazzi.least_squares_system` | residuals `r(x)` | Builds `(F, g, H)` with forward-mode autodiff: `(0.5‖r‖², Jᵀr, JᵀJ)`, or the robust equivalents. |

## Choosing a solver

- **You have residuals** (a model minus data, reprojection errors, point-to-point distances): use {func}`~piazzi.least_squares`. Gauss-Newton curvature usually converges in far fewer evaluations than a quasi-Newton method.
- **You have a scalar objective** that isn't a sum of squares (a likelihood, an image similarity measure): use {func}`~piazzi.minimize` or {func}`~piazzi.lbfgs_armijo` directly.
- **You build the normal equations yourself** (for example per-observation Jacobians assembled into a sparse `JᵀJ`): pass a `system(x) -> (loss, g, H)` to {func}`~piazzi.gauss_newton`. See {doc}`sparse`.

`method="lm"` (the default) adapts its damping to how well the model predicts each step, which makes it the safer choice. `method="gn"` takes the undamped step with a backtracking line search; with SciPy's curvature model and mostly outlying residuals it copes better than Levenberg-Marquardt (see {doc}`least-squares`).

## Damping and step limits

`damping="marquardt"` scales the Levenberg-Marquardt damping (and the Gauss-Newton ridge) by `diag(H)`, as Ceres does. Use it when parameters have very different units; it plays the role of SciPy's `x_scale="jac"`.

Steps are capped so that no coordinate moves more than `max_step` (default 1). Set it for the units of your problem: a large translation in millimetres needs a larger cap.

## Results and stopping

Every solver returns an {class}`~piazzi.OptimizeResult` with `x`, `fun`, `aux`, evaluation counts, `converged` and `message`. `converged` means a minimiser was reached to working precision: the step fell below `step_tol`, or the gradient is zero. In float32, Gauss-Newton, Levenberg-Marquardt and L-BFGS also recognise a minimiser once the predicted decrease falls below the rounding noise of the loss.

The evaluation counts are split by kind, because their costs differ: `n_fun_grad` (value and gradient), `n_fun` (value only, as in a line search) and `n_system` (normal equations).

## Callbacks

`callback(info)` receives an {class}`~piazzi.IterationInfo` after every iteration, accepted or not: the current point and value, the value at the start of the iteration, whether the step was accepted, and the evaluation counts so far (`info.n_fun_grad`, `info.n_fun`, `info.n_system`). Plot progress against those counts rather than iterations, which hide the evaluations a line search spends. Return `True` to stop the run.
