# Least squares and robust losses

{func}`piazzi.least_squares` minimises

$$F(x) = \tfrac12 \sum_i C^2 \rho\!\left(r_i(x)^2 / C^2\right),$$

where $C$ is `f_scale`, the soft margin between inlier and outlier residuals. It uses Levenberg-Marquardt (or Gauss-Newton with `method="gn"`) on normal equations built by {func}`piazzi.least_squares_system`. The Jacobian of the residuals comes from forward-mode autodiff (`torch.func.jacfwd`), so `fun` only has to return the residuals.

## Robust losses

As in SciPy, `loss` is `"linear"` (the default), `"huber"`, `"soft_l1"`, `"cauchy"`, `"arctan"`, or any elementwise PyTorch function `rho(z)`, whose derivatives come from autograd. {data}`piazzi.LOSSES` maps the names to their functions. {func}`piazzi.robust_cost` evaluates the cost differentiably, for use with {func}`piazzi.lbfgs`.

The gradient is `Jᵀ(ρ′ ⊙ r)`. Two curvature models are available:

- `curvature="irls"` (default): `Jᵀ diag(ρ′) J`, the iteratively reweighted least-squares model that Ceres Solver also uses for these losses.
- `curvature="scipy"`: SciPy's `max(ρ′ + 2ρ″z, ε)` weights. It is more accurate for inliers, but nearly singular when most residuals start out as outliers. Levenberg-Marquardt can then stall, and `method="gn"` copes better.

Both models converge to the same minimiser.

```python
res = piazzi.least_squares(residuals, x0, loss="cauchy", f_scale=0.1)
res = piazzi.least_squares(residuals, x0, loss=lambda z: torch.log1p(z), f_scale=0.1)  # custom rho, same as "cauchy"
```

The {doc}`robust regression notebook <../examples/robust_regression>` compares every loss with {func}`scipy.optimize.least_squares`.

## Auxiliary outputs

With `has_aux=True`, `fun(x)` returns `(residuals, aux)`. `aux` (any tree of tensors) is detached and returned in `res.aux` for the final point, and in `info.aux` for each callback, so you can keep a rendered image or intermediate quantities without evaluating the model again.

## Objectives that autograd cannot differentiate

If `fun` calls a non-differentiable renderer or simulator, a NumPy-based library or a measurement, wrap it with {func}`piazzi.finite_difference`. The wrapped function returns the same values and has finite-difference derivatives, in forward mode (used here) and in reverse mode (used by {func}`piazzi.minimize`), so every solver, loss and manifold works unchanged:

```python
fd_fun = piazzi.finite_difference(simulate)  # simulate(x) -> tensor, not differentiable by autograd
res = piazzi.least_squares(lambda x: fd_fun(x) - target, x0)
print(fd_fun.n_evals)  # every call of simulate, the differences included
```

This is a last resort, because it works against piazzi's aim of making each evaluation count: every Jacobian costs `n` (`scheme="forward"`) or `2n` (`scheme="central"`, the default) extra calls of `fun` for `n` parameters, and the solver's own counts (`n_system`, `n_fun`) do not include them, so read `n_evals` from the wrapper. The step is `eps * max(|x_i|, 1)` per coordinate, as in `torch.autograd.gradcheck`'s numerical Jacobian, with `eps` defaulting to the step that balances truncation and rounding error for the dtype (about `6e-6` in float64, `5e-3` in float32). Prefer float64: a central difference is accurate to about `1e-8` there, but only to about `1e-3` in float32. `fun` must be deterministic.

With a `manifold`, pass it to `finite_difference` as well as to the solver: `piazzi.finite_difference(fun, manifold=M)`. The wrapper then recovers the tangent coordinates of each direction and evaluates `fun(M.plus(x, ±h·delta))`, so `fun` only sees points on the manifold (valid rotation matrices, say) and a Jacobian costs `2 × tangent size` calls (12 for {class}`piazzi.SE3`, not 32). Without `manifold=`, the differences are taken in ambient space along the tangent directions, so `fun` is evaluated at points that leave the manifold by `O(eps²)`.
