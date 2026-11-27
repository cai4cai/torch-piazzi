# Examples

Executed notebooks. Each runs locally or in Google Colab, where its first cell installs piazzi from GitHub. To run them locally:

```bash
uv sync --group notebooks
uv run jupyter lab notebooks/
```

| Notebook | Description | |
|---|---|---|
| {doc}`Robust nonlinear regression <robust_regression>` | The [SciPy Cookbook robust regression example](https://scipy-cookbook.readthedocs.io/items/robust_regression.html) with {func}`piazzi.least_squares` and L-BFGS: robust losses (Huber, soft L1, Cauchy, arctan, custom), checked against {func}`scipy.optimize.least_squares`. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cai4cai/torch-piazzi/blob/main/notebooks/robust_regression.ipynb) |
| {doc}`The Rosenbrock function <rosenbrock>` | The classic test problem, after the [Ceres gradient tutorial](https://ceres-solver.readthedocs.io/latest/gradient_tutorial.html) and the [pytorch-minimize example](https://github.com/rfeinman/pytorch-minimize/blob/main/examples/rosen_minimize.ipynb): piazzi's and PyTorch's L-BFGS on the scalar function versus Gauss-Newton and Levenberg-Marquardt on its residuals, compared with SciPy's methods, plus the N-dimensional version and float32. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cai4cai/torch-piazzi/blob/main/notebooks/rosenbrock.ipynb) |
| {doc}`Bundle adjustment <bundle_adjustment>` | The [SciPy Cookbook bundle adjustment example](https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html) (49 cameras, 7776 points), with camera poses on SE(3), autodiff Jacobians through the manifold, sparse normal equations, and two linear solvers (CG and a Schur complement), on the CPU and in float32 on a GPU. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cai4cai/torch-piazzi/blob/main/notebooks/bundle_adjustment.ipynb) |

```{toctree}
:hidden:

robust_regression
rosenbrock
bundle_adjustment
```
