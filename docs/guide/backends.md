# Third-party backends

piazzi also has thin wrappers around third-party solvers, so you can compare backends on the same problem. Each wrapper takes PyTorch functions (converting to the backend's arrays where it needs them) and returns the same {class}`~piazzi.OptimizeResult` as the native solvers. The backends are optional dependencies, and `import piazzi` does not load them.

```bash
pip install "piazzi[scipy] @ git+https://github.com/cai4cai/torch-piazzi"
```

```python
import piazzi.scipy

res = piazzi.scipy.least_squares(fun, x0, loss="cauchy", f_scale=0.1)  # same fun and x0 as piazzi.least_squares
res = piazzi.scipy.minimize(cost, x0, method="L-BFGS-B")  # same cost and x0 as piazzi.minimize
```

## SciPy

{func}`piazzi.scipy.least_squares` wraps {func}`scipy.optimize.least_squares`:

- `fun` and the result use tensors. The Jacobian comes from forward-mode autodiff by default, as in {func}`piazzi.least_squares`; `jac` can also be a PyTorch callable or SciPy's `"2-point"` / `"3-point"`.
- A PyTorch `loss` callable is supported, with derivatives from autograd.
- `bounds`, `x_scale` and `diff_step` may hold tensors.
- `callback(info)` works as for the native solvers (SciPy 1.16 or later).
- Other keyword arguments (`method`, `ftol`, ...) go to SciPy, and `result.raw` holds SciPy's own result.

{func}`piazzi.scipy.minimize` wraps {func}`scipy.optimize.minimize`, with the calling convention of {func}`piazzi.minimize` and any SciPy method:

- By default each evaluation returns the value and its autograd gradient together; gradient-free methods (Nelder-Mead, Powell, COBYLA, COBYQA) get no gradient.
- `hess="autodiff"` / `hessp="autodiff"` give exact Hessians or Hessian-vector products for Newton-CG and the trust-region methods.
- `callback(info)` gets an {class}`~piazzi.IterationInfo` and can stop the run by returning `True`.
- `bounds` may hold tensors, and other keyword arguments (`tol`, `constraints`, ...) go to SciPy, with SciPy's `options`.

The wrappers count the calls they make rather than relaying SciPy's `nfev` / `njev` / `nhev`, so finite-difference evaluations are included in `n_fun`, and Hessian (or Hessian-vector product) evaluations appear in `n_system`.

SciPy works in float64 on the CPU; the wrappers convert at each call, and `fun` runs in the dtype and on the device of `x0`. The SciPy wrappers do not support `manifold=`.

## PyPose

{func}`piazzi.pypose.least_squares` runs [PyPose](https://github.com/pypose/pypose)'s `pypose.optim.GaussNewton` (`method="gn"`) or `pypose.optim.LevenbergMarquardt` (`method="lm"`, the default) on a residual function, with the calling convention of {func}`piazzi.least_squares` (`pip install "piazzi[pypose]"`):

```python
import piazzi.pypose

res = piazzi.pypose.least_squares(fun, x0, method="lm", solver="cholesky", strategy="adaptive")
```

- `solver` takes `"pinv"`, `"lstsq"`, `"cholesky"` or `"cg"` (or a PyPose solver object), and `strategy` the damping strategy of Levenberg-Marquardt, `"constant"`, `"adaptive"` or `"trust_region"` (or a PyPose strategy object). `solver_options` and `strategy_options` hold their constructor arguments.
- `x0` can be a `pypose.LieTensor`: PyPose then updates it on the left, `X <- Exp(delta) X`, and `fun` receives a `LieTensor`. This takes the place of `manifold=`, which this wrapper does not accept: PyPose applies its own updates, so a {class}`piazzi.Manifold` cannot be used.
- Everything runs on the device and in the dtype of `x0`, including PyPose's linear solves (piazzi's native solvers solve in float64 on the CPU by default).
- The cost is `0.5 * sum(r^2)` as in {func}`piazzi.least_squares` (PyPose's own loss is twice that), and `callback(info)` works as for the native solvers. Robust kernels and weights are not wrapped.

PyPose's optimisers take one step per call and have no stopping rules or line search, and their `step` evaluates the model at the same point several times (for the residuals, inside the Jacobian and for the loss). The wrapper therefore adds what piazzi's guarantees need:

- **Stopping and convergence.** It runs up to `max_iter` steps and accepts a step only if the cost went down. The run converges when an accepted step is below `step_tol` or the cost is zero. A step that does not decrease the cost is undone and ends the run unconverged, unless it is below `step_tol` and within rounding noise of the old cost. Gauss-Newton takes full steps, so from a bad start it can diverge: the wrapper then returns the starting point and says so in `message`. A non-finite cost is never accepted, and `x` is never worse than `x0`.
- **Evaluation counts.** `n_system` counts the Jacobian evaluations (each also computes the residuals at its point), and `n_fun` the plain residual evaluations, every one at a point not evaluated before: the wrapper answers PyPose's repeated evaluations from a memo. Levenberg-Marquardt's rejected trials each cost one `n_fun`. `n_fun_grad` is always 0.

PyPose's Gauss-Newton and Levenberg-Marquardt are the same algorithms as {func}`piazzi.gauss_newton` and {func}`piazzi.levenberg_marquardt`, so this wrapper is mainly a benchmark reference (see the Rosenbrock notebook, where piazzi's Levenberg-Marquardt needs fewer evaluations and its Gauss-Newton has the line search PyPose's lacks). Its use is the `LieTensor` parameters, when the rest of a model is written with PyPose.
