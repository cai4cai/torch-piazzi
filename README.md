# torch-piazzi

[![tests](https://github.com/cai4cai/torch-piazzi/actions/workflows/tests.yml/badge.svg)](https://github.com/cai4cai/torch-piazzi/actions/workflows/tests.yml)
[![docs](https://github.com/cai4cai/torch-piazzi/actions/workflows/docs.yml/badge.svg)](https://cai4cai.github.io/torch-piazzi/)

`piazzi` provides deterministic PyTorch solvers for nonlinear fitting problems, such as pose or camera registration, curve fitting and bundle adjustment. It is built for problems where every objective evaluation is expensive (rendering, for example) and you want each evaluation to count. `torch.optim` targets stochastic training instead; `piazzi` targets full-batch, curvature-aware fitting.

> Named after Giuseppe Piazzi, who discovered the dwarf planet Ceres in 1801. Gauss recovered Ceres from Piazzi's observations by inventing least squares.

**Status:** pre-alpha, and the API may change. The package is not on PyPI yet.

**Documentation:** <https://cai4cai.github.io/torch-piazzi/>

## Installation

```bash
pip install "piazzi @ git+https://github.com/cai4cai/torch-piazzi"
pip install "piazzi[scipy,sparse] @ git+https://github.com/cai4cai/torch-piazzi"  # optional backends and sparse CG
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
print(res.x, res.message)  # tensor([2.5026, 1.3005]) step below step_tol; loss="linear" gives [3.1053, 1.2741]
```

## Features

- **[Solvers](https://cai4cai.github.io/torch-piazzi/guide/solvers.html):** L-BFGS with an Armijo line search that counts every evaluation, Gauss-Newton, Levenberg-Marquardt and Powell's dog-leg trust region (Ceres's traditional and subspace variants), and SciPy-style `least_squares` and `minimize`.
- **[Robust losses](https://cai4cai.github.io/torch-piazzi/guide/least-squares.html):** SciPy's losses or any PyTorch `rho(z)`, with IRLS or SciPy curvature.
- **[Black-box objectives](https://cai4cai.github.io/torch-piazzi/guide/least-squares.html):** `finite_difference` makes a function that autograd cannot see through usable by every solver.
- **[Manifolds](https://cai4cai.github.io/torch-piazzi/guide/manifolds.html):** keep rotations and poses on SO(3) and SE(3), as Ceres's `Manifold` does, with batched exp/log maps.
- **[Large sparse problems](https://cai4cai.github.io/torch-piazzi/guide/sparse.html):** sparse normal equations, conjugate gradients and custom linear solvers such as a Schur complement.
- **[Devices and precision](https://cai4cai.github.io/torch-piazzi/guide/devices-precision.html):** run the objective in float32 on CUDA or Apple MPS, and solve the normal equations in float64 on the CPU or on the GPU.
- **[`torch.optim` optimisers](https://cai4cai.github.io/torch-piazzi/guide/torch-optim.html)** and **[SciPy backends](https://cai4cai.github.io/torch-piazzi/guide/backends.html)** behind the same interface, to compare methods on the same problem.

## Notebooks

The notebooks in [`notebooks/`](notebooks) run in Google Colab, or locally after `uv sync --group notebooks`. They are also rendered in the [documentation](https://cai4cai.github.io/torch-piazzi/examples/index.html).

| Notebook | |
|---|---|
| [Robust nonlinear regression](notebooks/robust_regression.ipynb): robust losses, checked against `scipy.optimize.least_squares` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cai4cai/torch-piazzi/blob/main/notebooks/robust_regression.ipynb) |
| [The Rosenbrock function](notebooks/rosenbrock.ipynb): L-BFGS variants versus Gauss-Newton and Levenberg-Marquardt, and SciPy's methods | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cai4cai/torch-piazzi/blob/main/notebooks/rosenbrock.ipynb) |
| [Bundle adjustment](notebooks/bundle_adjustment.ipynb): 49 cameras on SE(3), sparse normal equations, CG and Schur solvers, on the CPU and a GPU | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cai4cai/torch-piazzi/blob/main/notebooks/bundle_adjustment.ipynb) |

## Related work

Ceres Solver, Theseus, PyPose, `torch.optim.LBFGS`, nlesc-dirac/pytorch, pytorch-minimize and `scipy.optimize` solve related problems; the [related work page](https://cai4cai.github.io/torch-piazzi/related-work.html) compares them.

## Development

```bash
uv sync --group dev
uv run pytest
uv run ruff check .
```

See the [contributing page](https://cai4cai.github.io/torch-piazzi/contributing.html) and [AGENTS.md](AGENTS.md) for the full checks and conventions. Planned work is tracked in the [issues](https://github.com/cai4cai/torch-piazzi/issues).

## Licence

Apache-2.0
