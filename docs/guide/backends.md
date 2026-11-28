# Third-party backends

piazzi also has thin wrappers around third-party solvers, so you can compare backends on the same problem. Each wrapper takes PyTorch functions, converts between PyTorch and the backend's arrays, and returns the same {class}`~piazzi.OptimizeResult` as the native solvers. The backends are optional dependencies, and `import piazzi` does not load them.

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
- `callback(info)` and `verbose=` work as for the native solvers (SciPy 1.16 or later); piazzi's `verbose` takes the place of SciPy's own.
- Other keyword arguments (`method`, `ftol`, ...) go to SciPy, and `result.raw` holds SciPy's own result.

{func}`piazzi.scipy.minimize` wraps {func}`scipy.optimize.minimize`, with the calling convention of {func}`piazzi.minimize` and any SciPy method:

- By default each evaluation returns the value and its autograd gradient together; gradient-free methods (Nelder-Mead, Powell, COBYLA, COBYQA) get no gradient.
- `hess="autodiff"` / `hessp="autodiff"` give exact Hessians or Hessian-vector products for Newton-CG and the trust-region methods.
- `callback(info)` gets an {class}`~piazzi.IterationInfo` and can stop the run by returning `True`, and `verbose=True` prints a table of the run as for the native solvers.
- `bounds` may hold tensors, and other keyword arguments (`tol`, `constraints`, ...) go to SciPy, with SciPy's `options`.

The wrappers count the calls they make rather than relaying SciPy's `nfev` / `njev` / `nhev`, so finite-difference evaluations are included in `n_fun`, and Hessian (or Hessian-vector product) evaluations appear in `n_system`.

SciPy works in float64 on the CPU; the wrappers convert at each call, and `fun` runs in the dtype and on the device of `x0`. The SciPy wrappers do not support `manifold=`.
