# torch-piazzi

`piazzi` provides deterministic PyTorch solvers for small nonlinear fitting problems, such as pose or camera registration. In these problems there are few parameters, every objective evaluation is expensive (rendering, for example), and you want each evaluation to count. `torch.optim` targets stochastic training instead; `piazzi` targets full-batch, curvature-aware fitting.

> Named after Giuseppe Piazzi, who discovered the dwarf planet Ceres in 1801. Gauss recovered Ceres from Piazzi's observations by inventing least squares.

**Status:** pre-alpha, and the API may change. The package is not on PyPI yet.

## Solvers

| Function | Needs | Notes |
|---|---|---|
| `lbfgs_armijo(fun, x0)` | value + gradient (autograd) | L-BFGS with Armijo backtracking. It usually costs one evaluation per iteration, and the line search uses value-only evaluations. |
| `gauss_newton(system, x0, method="gn")` | `(loss, g, H)` | Gauss-Newton with Armijo backtracking. |
| `gauss_newton(system, x0, method="lm")` / `levenberg_marquardt` | `(loss, g, H)` | Levenberg-Marquardt with Nielsen's damping update. |
| `least_squares_system(residual_fun)` | residuals `r(x)` | Builds `(0.5‖r‖², Jᵀr, JᵀJ)` with forward-mode autodiff. |

The normal equations are solved in float64 on the CPU by default, so the objective itself can run in float32 on a GPU or MPS device. Every solver returns an `OptimizeResult` with `x`, `fun`, `aux`, evaluation counts, `converged` and `message`. You can pass `callback(info)` to record the iterates or to stop early by returning `True`.

## Example

```python
import torch
import piazzi

t = torch.linspace(0, 4, 40)
y = 2.5 * torch.exp(-1.3 * t)

system = piazzi.least_squares_system(lambda th: th[0] * torch.exp(-th[1] * t) - y)
res = piazzi.levenberg_marquardt(system, torch.tensor([1.0, 0.5]))
print(res.x, res.message)
```

## Notebooks

The notebooks in [`notebooks/`](notebooks) run locally or in Google Colab. On Colab, the first cell installs `piazzi` from GitHub.

| Notebook | Description | |
|---|---|---|
| [Robust nonlinear regression](notebooks/robust_regression.ipynb) | The [SciPy Cookbook robust regression example](https://scipy-cookbook.readthedocs.io/items/robust_regression.html) with Levenberg-Marquardt and L-BFGS: robust losses (Huber, soft L1, Cauchy, arctan) through IRLS normal equations, checked against `scipy.optimize.least_squares`. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cai4cai/torch-piazzi/blob/main/notebooks/robust_regression.ipynb) |

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
