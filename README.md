# torch-piazzi

`piazzi` provides deterministic PyTorch solvers for small nonlinear fitting problems, such as pose or camera registration. In these problems there are few parameters, every objective evaluation is expensive (rendering, for example), and you want each evaluation to count. `torch.optim` targets stochastic training instead; `piazzi` targets full-batch, curvature-aware fitting.

> Named after Giuseppe Piazzi, who discovered the dwarf planet Ceres in 1801. Gauss recovered Ceres from Piazzi's observations by inventing least squares.

**Status:** pre-alpha, and the API may change. The package is not on PyPI yet.

## Solvers

| Function | Needs | Notes |
|---|---|---|
| `lbfgs_armijo(fun, x0)` | value + gradient (autograd) | L-BFGS with Armijo backtracking. It usually costs one evaluation per iteration, and the line search uses value-only evaluations. |
| `minimize(fun, x0, method="lbfgs_armijo", options=...)` | value + gradient (autograd) | Like `scipy.optimize.minimize`: picks a scalar minimiser by name (`piazzi.MINIMIZE_METHODS`; currently only `lbfgs_armijo`) and passes it `options`. |
| `gauss_newton(system, x0, method="gn")` | `(loss, g, H)` | Gauss-Newton with Armijo backtracking. |
| `gauss_newton(system, x0, method="lm")` / `levenberg_marquardt` | `(loss, g, H)` | Levenberg-Marquardt with Nielsen's damping update. |
| `least_squares(fun, x0, loss=..., f_scale=...)` | residuals `r(x)` | Like `scipy.optimize.least_squares`: Levenberg-Marquardt (or `method="gn"`) on the normal equations from `least_squares_system`, with an optional robust loss. |
| `least_squares_system(residual_fun, loss=..., f_scale=...)` | residuals `r(x)` | Builds `(F, g, H)` with forward-mode autodiff: `(0.5‖r‖², Jᵀr, JᵀJ)`, or the robust equivalents. |

### Robust losses

As in SciPy, `loss` is `"linear"` (the default), `"huber"`, `"soft_l1"`, `"cauchy"`, `"arctan"`, or any elementwise PyTorch function `rho(z)`, whose derivatives come from autograd. The cost is `F = 0.5 * Σ C² ρ(rᵢ² / C²)` with `C = f_scale`. `piazzi.robust_cost(r, loss, f_scale)` evaluates it differentiably, for use with `lbfgs_armijo`.

The gradient is `Jᵀ(ρ′ ⊙ r)`. By default the curvature model is `Jᵀ diag(ρ′) J` (`curvature="irls"`), the iteratively reweighted least-squares model that Ceres Solver also uses for these losses. `curvature="scipy"` uses SciPy's `max(ρ′ + 2ρ″z, ε)` weights instead. It is more accurate for inliers but nearly singular when most residuals start out as outliers. Both models converge to the same minimiser.

The normal equations are solved in float64 on the CPU by default, so the objective itself can run in float32 on a GPU or MPS device. `damping="marquardt"` scales the Levenberg-Marquardt damping (and the Gauss-Newton ridge) by `diag(H)`, as Ceres does. Use it when parameters have very different units; it plays the role of SciPy's `x_scale="jac"`. Every solver returns an `OptimizeResult` with `x`, `fun`, `aux`, evaluation counts, `converged` and `message`. You can pass `callback(info)` to record the iterates or to stop early by returning `True`.

### Manifolds: poses and rotations

Pass `manifold=` to `least_squares`, `gauss_newton`, `lbfgs_armijo` or `minimize` to keep `x` on a constrained set, as Ceres Solver's [`Manifold`](https://ceres-solver.readthedocs.io/latest/nnls_modeling.html#manifold) does. A manifold supplies `plus(x, delta)`, which moves `x` by a tangent vector. The solver takes its Jacobian or gradient with respect to `delta` at zero, by autodiff through `plus`, and applies each step as `x <- plus(x, step)`. A 4 x 4 pose therefore gives 6 Jacobian columns, and every iterate stays a rigid transform.

| Manifold | Stored as | Tangent | Notes |
|---|---|---|---|
| `Euclidean(shape)` | any tensor | same size | The default behaviour; a shape is only needed inside `Product`. |
| `Subset(size, constant)` | vector | free entries | Holds some entries fixed, like Ceres's `SubsetManifold`. |
| `SO3(side="right")` | 3 x 3 rotation | 3 | `R @ so3_exp(w)` (body frame), or `so3_exp(w) @ R` with `side="left"`. |
| `SE3(side="right", retraction="decoupled", pivot=None)` | 4 x 4 rigid transform | 6, `(rho, omega)` | `retraction="exp"` uses the true SE(3) exponential (a screw motion). `pivot` sets the point that rotations turn about. |
| `Product(*manifolds)` | flat vector | sum | Use `M.join(...)` to build `x0` and `M.split(x)` to unpack it, for example a pose plus camera intrinsics. |

The manifolds also provide `minus(y, x)`, which inverts `plus`. The solvers don't use it, but it measures errors between points. `so3_exp`, `so3_log`, `se3_exp`, `se3_log` and `so3_hat` work on batches.

```python
P, Q = ...  # matching 3D points, Q ~ R @ p + t
res = piazzi.least_squares(
    lambda T: P @ T[:3, :3].T + T[:3, 3] - Q, torch.eye(4), manifold=piazzi.SE3(pivot=P.mean(0)), max_step=1e3
)
res.x  # a 4 x 4 rigid transform
```

**Set the pivot for registration.** Put `pivot` at the centre of the object. If the object sits far from the origin, a rotation about the origin also moves it a long way. Rotation and translation are then strongly coupled and Gauss-Newton crawls. In the tests, a point set 140 units from the origin converges in 5 Gauss-Newton iterations with the pivot at its centre, and has not converged after 10 without it.

**Set `max_step` for the problem's units.** It caps every tangent component, so it applies to radians and translation units alike. Its default of 1 can slow down a large translation.

The [`_manifolds.py`](src/piazzi/_manifolds.py) docstring explains how this design compares with Ceres, [PyPose](https://github.com/pypose/pypose) and [GeoTorch](https://github.com/lezcano/geotorch). [`tests/test_ceres_manifolds.py`](tests/test_ceres_manifolds.py) ports Ceres's manifold and rotation tests.

### Large sparse problems

`system(x)` may return `H` as a sparse COO or CSR tensor, as in bundle adjustment, where `J^T J` has millions of entries but is 99.7 % zeros. `gauss_newton` then needs only the diagonal of `H` and products with it. The `linear_solver` argument chooses how the damped system `(H + diag(damping)) d = -g` is solved:

| Linear solver | Notes |
|---|---|
| `piazzi.dense_solver` | Direct `torch.linalg.solve`. This is the default for a dense `H`. |
| `piazzi.cg_solver(tol, max_iter, preconditioner="jacobi")` | Preconditioned conjugate gradients from [torchsparsegradutils](https://github.com/cai4cai/torchsparsegradutils). This is the default for a sparse `H`. It accepts your own preconditioner, for example block-Jacobi. It is an optional dependency: `pip install "piazzi[sparse]"`. |
| any `solver(H, g, damping) -> d` | For example a Schur-complement solver that eliminates the points in bundle adjustment. The [bundle adjustment notebook](notebooks/bundle_adjustment.ipynb) shows one. |

## Third-party backends

`piazzi` also has thin wrappers around third-party solvers, so you can compare backends on the same problem. Each wrapper takes PyTorch functions, converts between PyTorch and the backend's arrays, and returns the same `OptimizeResult` as the native solvers. The backends are optional dependencies, and `import piazzi` does not load them.

| Function | Backend | Notes |
|---|---|---|
| `piazzi.scipy.least_squares(fun, x0, loss=..., f_scale=...)` | [`scipy.optimize.least_squares`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html) | `fun` and the result use tensors. The Jacobian comes from forward-mode autodiff by default, as in `piazzi.least_squares`; `jac` can also be a PyTorch callable or SciPy's `"2-point"` / `"3-point"`. A PyTorch `loss` callable is supported, with derivatives from autograd. Other keyword arguments (`method`, `bounds`, `ftol`, ...) go to SciPy, and `result.raw` holds SciPy's own result. |
| `piazzi.scipy.minimize(fun, x0, method=..., options=...)` | [`scipy.optimize.minimize`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html) | Same calling convention as `piazzi.minimize` (except `manifold=`, which the SciPy wrappers do not support), with any SciPy method. By default each evaluation returns the value and its autograd gradient together. `hess="autodiff"` / `hessp="autodiff"` give exact Hessians or Hessian-vector products for Newton-CG and the trust-region methods. `bounds` may hold tensors, and other keyword arguments (`tol`, `constraints`, ...) go to SciPy. |

```bash
pip install "piazzi[scipy] @ git+https://github.com/cai4cai/torch-piazzi"
```

```python
import piazzi.scipy

res = piazzi.scipy.least_squares(fun, x0, loss="cauchy", f_scale=0.1)  # same fun and x0 as piazzi.least_squares
res = piazzi.scipy.minimize(cost, x0, method="L-BFGS-B")  # same cost and x0 as piazzi.minimize
```

## Example

```python
import torch
import piazzi

t = torch.linspace(0, 4, 40)
y = 2.5 * torch.exp(-1.3 * t)

y[::10] += 2.0  # outliers

res = piazzi.least_squares(
    lambda th: th[0] * torch.exp(-th[1] * t) - y, torch.tensor([1.0, 0.5]), loss="cauchy", f_scale=0.1
)
print(res.x, res.message)  # tensor([2.5026, 1.3005]) step below step_tol; loss="linear" gives [3.1057, 1.2745]
```

## Notebooks

The notebooks in [`notebooks/`](notebooks) run locally or in Google Colab. On Colab, the first cell installs `piazzi` from GitHub.

| Notebook | Description | |
|---|---|---|
| [Bundle adjustment](notebooks/bundle_adjustment.ipynb) | The [SciPy Cookbook bundle adjustment example](https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html) (49 cameras, 7776 points), with each camera pose on SE(3) (`piazzi.SE3` in a `piazzi.Product`), autodiff Jacobians through the manifold, sparse normal equations, and two linear solvers: torchsparsegradutils' CG and a Schur complement. It is compared with `scipy.optimize.least_squares`, then run again in float32 on a GPU (CUDA or Apple MPS), either with the solve on the CPU in float64 or with everything on the GPU. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cai4cai/torch-piazzi/blob/main/notebooks/bundle_adjustment.ipynb) |
| [Robust nonlinear regression](notebooks/robust_regression.ipynb) | The [SciPy Cookbook robust regression example](https://scipy-cookbook.readthedocs.io/items/robust_regression.html) with `piazzi.least_squares` and L-BFGS: robust losses (Huber, soft L1, Cauchy, arctan, custom), checked against `scipy.optimize.least_squares`. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cai4cai/torch-piazzi/blob/main/notebooks/robust_regression.ipynb) |

To run them locally:

```bash
uv sync --group notebooks
uv run jupyter lab notebooks/
```

## Related work

`piazzi` builds on ideas from these projects, and one of them may suit your problem better:

- [Ceres Solver](http://ceres-solver.org/): Google's C++ library for nonlinear least squares and general unconstrained optimisation. It offers trust-region (Levenberg-Marquardt, dog-leg) and line-search methods, robust loss functions and sparse solvers for large bundle-adjustment problems. The name `piazzi` is a nod to it.
- [Theseus](https://github.com/facebookresearch/theseus): a PyTorch library for differentiable nonlinear least squares (Gauss-Newton, Levenberg-Marquardt, dog-leg). It runs batched and sparse solves and backpropagates through the optimiser, so you can embed it as a layer in a network.
- [PyPose](https://github.com/pypose/pypose): a PyTorch library for robotics, with Lie-group types (SO3, SE3 and others) and second-order optimisers such as Gauss-Newton and Levenberg-Marquardt, including robust kernels.
- [`torch.optim.LBFGS`](https://pytorch.org/docs/stable/generated/torch.optim.LBFGS.html): PyTorch's built-in L-BFGS, with an optional strong-Wolfe line search. It uses the `torch.optim` closure interface.
- [nlesc-dirac/pytorch](https://github.com/nlesc-dirac/pytorch): improved L-BFGS and L-BFGS-B optimisers for PyTorch, with line searches and support for full-batch and stochastic use.
- [pytorch-minimize](https://github.com/rfeinman/pytorch-minimize) (`torchmin`): `scipy.optimize.minimize`-style solvers for PyTorch (BFGS, L-BFGS, conjugate gradient, Newton and trust-region methods), plus `least_squares`.
- [`scipy.optimize`](https://docs.scipy.org/doc/scipy/reference/optimize.html): the reference CPU implementations, including `least_squares` (trust-region reflective, dog-box, MINPACK Levenberg-Marquardt) and `minimize` (L-BFGS-B and others). The notebooks use it to check `piazzi`'s results.

## Development

```bash
uv sync
uv run pytest
uv run ruff check .
```

## Roadmap

- Dog-leg and other trust-region methods
- Batched solves (many independent small problems at once)
- Release on PyPI as `piazzi`

## Licence

Apache-2.0
