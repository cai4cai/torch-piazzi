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
