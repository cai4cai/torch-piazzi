# Getting started

## Installation

piazzi needs Python 3.11 or later and PyTorch 2.15 or later. Install the alpha pre-release from PyPI:

```bash
pip install piazzi
```

To install the latest development version from GitHub instead:

```bash
pip install "piazzi @ git+https://github.com/cai4cai/torch-piazzi"
```

Optional extras add third-party backends and solvers:

| Extra | Adds |
|---|---|
| `piazzi[scipy]` | {doc}`guide/backends`: `piazzi.scipy.least_squares` and `piazzi.scipy.minimize` |
| `piazzi[pypose]` | {doc}`guide/backends`: `piazzi.pypose.least_squares`, PyPose's Gauss-Newton and Levenberg-Marquardt, also on `LieTensor` parameters |
| `piazzi[torchmin]` | {doc}`guide/backends`: `piazzi.torchmin.minimize`, wrapping pytorch-minimize |
| `piazzi[sparse]` | {func}`piazzi.cg_solver` (conjugate gradients from torchsparsegradutils), the default linear solver for a sparse Hessian |

```bash
pip install "piazzi[scipy,pypose,torchmin,sparse]"
```

## A first fit

Fit an exponential decay to data with outliers. {func}`piazzi.least_squares` takes a residual function written in PyTorch and a starting point; the Jacobian comes from forward-mode autodiff.

```python
import torch
import piazzi

t = torch.linspace(0, 4, 40)
y = 2.5 * torch.exp(-1.3 * t)
y[::10] += 2.0  # outliers


def residuals(theta):
    return theta[0] * torch.exp(-theta[1] * t) - y


res = piazzi.least_squares(residuals, torch.tensor([1.0, 0.5]), loss="cauchy", f_scale=0.1)
print(res.x, res.message)  # tensor([2.5026, 1.3005]) step below step_tol
```

With the default `loss="linear"`, the outliers pull the fit to `[3.1053, 1.2741]`; see {doc}`guide/least-squares` for the robust losses.

For a scalar objective, use {func}`piazzi.minimize` (L-BFGS with an Armijo line search by default):

```python
def rosenbrock(x):
    return (100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2).sum()


res = piazzi.minimize(rosenbrock, torch.tensor([-1.2, 1.0], dtype=torch.float64))
```

## Reading the result

Every solver returns a {class}`piazzi.OptimizeResult`:

- `x`: the solution, with the shape, dtype and device of `x0`;
- `fun`: the final objective value, and `aux` its auxiliary output (with `has_aux=True`);
- `converged` and `message`: whether a minimiser was reached to working precision, and why the run stopped;
- `n_iter`, and the evaluation counts by kind: `n_fun_grad` (value and gradient), `n_fun` (value only) and `n_system` (normal equations).

## Watching progress

Pass `callback(info)` to record the iterates. It receives an {class}`piazzi.IterationInfo` after every iteration, accepted or not, with the evaluation counts so far. Return `True` to stop the run. To just watch a run, pass `verbose=True`, which prints one row per iteration.

```python
history = []
res = piazzi.least_squares(residuals, torch.tensor([1.0, 0.5]), callback=lambda info: history.append(info.fun.item()))
```

## Next steps

- {doc}`guide/solvers` explains the solvers and how to choose one.
- {doc}`examples/index` has complete, executed notebooks that run locally or in Google Colab.
