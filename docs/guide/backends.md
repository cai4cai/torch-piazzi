# Third-party backends

piazzi also has thin wrappers around third-party solvers, so you can compare backends on the same problem. Each wrapper takes PyTorch functions, converts between PyTorch and the backend's arrays, and returns the same {class}`~piazzi.OptimizeResult` as the native solvers. The backends are optional dependencies, and `import piazzi` does not load them.

```bash
pip install "piazzi[scipy,torchmin] @ git+https://github.com/cai4cai/torch-piazzi"
```

```python
import piazzi.scipy

res = piazzi.scipy.least_squares(fun, x0, loss="cauchy", f_scale=0.1)  # same fun and x0 as piazzi.least_squares
res = piazzi.scipy.minimize(cost, x0, method="L-BFGS-B")  # same cost and x0 as piazzi.minimize

import piazzi.torchmin

res = piazzi.torchmin.minimize(cost, x0, "trust-exact")
res = piazzi.minimize(cost, x0, method="torchmin-trust-exact")  # the same, by name
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

## pytorch-minimize

{func}`piazzi.torchmin.minimize` wraps [pytorch-minimize](https://github.com/rfeinman/pytorch-minimize) (`torchmin`, MIT licence), which implements SciPy-style solvers in PyTorch: BFGS, L-BFGS, conjugate gradient, Newton-CG, Newton-exact and the trust-region methods (dogleg, trust-ncg, trust-exact, trust-krylov). It has the calling convention of {func}`piazzi.minimize`, and importing the module registers each method in {data}`piazzi.MINIMIZE_METHODS` as `"torchmin-bfgs"`, `"torchmin-trust-exact"`, ...

- Nothing leaves PyTorch: `fun` and the solver run on the device and in the dtype of `x0`, with gradients, Hessians and Hessian-vector products from autograd (the Newton-type and trust-region methods need a twice-differentiable objective).
- `has_aux`, and `callback(info)` with the value and `aux` at each iterate, work as for the native solvers. torchmin's own callback only receives `x`; the wrapper looks the value up among the evaluations it has made, so the callback costs no extra evaluation.
- `manifold=` raises {class}`ValueError`: torchmin has no retraction. Use {func}`piazzi.lbfgs_armijo` for manifolds.
- Every evaluation torchmin makes is a value-and-gradient one, so `n_fun_grad` is the number of calls to `fun`, and `n_fun` and `n_system` are zero: Hessian-vector products are backward passes through the graph kept from an evaluation, which the wrapper cannot see, so they are not counted.
- `result.converged` is torchmin's `success`. For BFGS, L-BFGS, Newton-CG and Newton-exact it includes stopping because a step or a decrease fell below `xtol`, which is weaker than a small gradient.

Things to know about torchmin 0.1.0 itself: the trust-region methods compute the gradient and Hessian at every trial point, rejected ones included; Newton-CG and Newton-exact evaluate the point their line search accepts a second time; `trust-krylov` raises `RuntimeError("reducible T matrix encountered.")` when its Lanczos process breaks down, which it does on Rosenbrock; and its tolerance options differ per method (`gtol` for the quasi-Newton, conjugate-gradient and trust-region methods, `xtol` for the Newton ones). `torchmin.least_squares` is not wrapped: it has no callback, supports only a one-dimensional `x`, and evaluates the residual twice at every accepted point; use {func}`piazzi.least_squares`.
