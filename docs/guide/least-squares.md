# Least squares and robust losses

{func}`piazzi.least_squares` minimises

$$F(x) = \tfrac12 \sum_i C^2 \rho\!\left(r_i(x)^2 / C^2\right), \qquad C = \texttt{f\_scale},$$

with Levenberg-Marquardt (or Gauss-Newton with `method="gn"`) on normal equations built by {func}`piazzi.least_squares_system`. The Jacobian of the residuals comes from forward-mode autodiff (`torch.func.jacfwd`), so `fun` only has to return the residuals.

## Robust losses

As in SciPy, `loss` is `"linear"` (the default), `"huber"`, `"soft_l1"`, `"cauchy"`, `"arctan"`, or any elementwise PyTorch function `rho(z)`, whose derivatives come from autograd. {data}`piazzi.LOSSES` maps the names to their functions. {func}`piazzi.robust_cost` evaluates the cost differentiably, for use with {func}`piazzi.lbfgs_armijo`.

The gradient is `Jᵀ(ρ′ ⊙ r)`. Two curvature models are available:

- `curvature="irls"` (default): `Jᵀ diag(ρ′) J`, the iteratively reweighted least-squares model that Ceres Solver also uses for these losses.
- `curvature="scipy"`: SciPy's `max(ρ′ + 2ρ″z, ε)` weights. It is more accurate for inliers, but nearly singular when most residuals start out as outliers. Levenberg-Marquardt can then stall, and `method="gn"` copes better.

Both models converge to the same minimiser.

```python
res = piazzi.least_squares(residuals, x0, loss="cauchy", f_scale=0.1)
res = piazzi.least_squares(residuals, x0, loss=lambda z: torch.log1p(z))  # custom rho, same as "cauchy"
```

The {doc}`robust regression notebook <../examples/robust_regression>` compares every loss with {func}`scipy.optimize.least_squares`.

## Auxiliary outputs

With `has_aux=True`, `fun(x)` returns `(residuals, aux)`. `aux` (any tree of tensors) is detached and returned in `res.aux` for the final point, and in `info.aux` for each callback, so you can keep a rendered image or intermediate quantities without evaluating the model again.
