# piazzi

**Deterministic PyTorch solvers for small nonlinear fitting problems.**

`piazzi` targets fitting problems such as pose or camera registration, curve fitting and bundle adjustment, where there are few parameters (or a sparse structure), every objective evaluation is expensive (rendering, for example), and you want each evaluation to count. `torch.optim` targets stochastic training instead; `piazzi` targets full-batch, curvature-aware fitting.

> Named after Giuseppe Piazzi, who discovered the dwarf planet Ceres in 1801. Gauss recovered Ceres from Piazzi's observations by inventing least squares.

```python
import torch
import piazzi

t = torch.linspace(0, 4, 40)
y = 2.5 * torch.exp(-1.3 * t)
y[::10] += 2.0  # outliers

res = piazzi.least_squares(
    lambda th: th[0] * torch.exp(-th[1] * t) - y, torch.tensor([1.0, 0.5]), loss="cauchy", f_scale=0.1
)
print(res.x, res.message)  # tensor([2.5026, 1.3005]) step below step_tol
```

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Getting started
:link: getting-started
:link-type: doc
Install piazzi, run a first fit, and read the result.
:::

:::{grid-item-card} User guide
:link: guide/index
:link-type: doc
Solvers, robust losses, manifolds, sparse problems, devices and backends.
:::

:::{grid-item-card} Examples
:link: examples/index
:link-type: doc
Executed notebooks: robust regression, Rosenbrock, bundle adjustment.
:::

:::{grid-item-card} API reference
:link: api
:link-type: doc
Every public function and class.
:::
::::

**Status:** pre-alpha, and the API may change. The package is not on PyPI yet.

```{toctree}
:hidden:

getting-started
guide/index
examples/index
api
related-work
contributing
changelog
```

```{toctree}
:hidden:
:caption: Project links

GitHub repository <https://github.com/cai4cai/torch-piazzi>
Issues <https://github.com/cai4cai/torch-piazzi/issues>
```
