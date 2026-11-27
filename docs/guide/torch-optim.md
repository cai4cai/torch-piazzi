# `torch.optim` optimisers

{func}`piazzi.minimize` also runs any {class}`torch.optim.Optimizer` subclass: pass the class as `method` and its constructor arguments in `options`, alongside piazzi's `max_iter`, `step_tol` and, for LBFGS, `max_eval`. The result is the usual {class}`~piazzi.OptimizeResult`, with the optimiser itself in `result.raw`.

```python
res = piazzi.minimize(fun, x0, method=torch.optim.LBFGS, options={"line_search_fn": "strong_wolfe"})
res = piazzi.minimize(fun, x0, method=torch.optim.Adam, options={"lr": 1e-2, "max_iter": 1000})
```

Each iteration is one optimiser step, so `callback` sees every iterate. {class}`torch.optim.LBFGS` runs one L-BFGS iteration per step, with `max_iter`, `max_eval` and its `tolerance_grad` / `tolerance_change` tests applied to the whole run: it takes exactly the same path as a single upstream `step(closure)` with the same settings. Evaluations are memoised, so the point the strong-Wolfe line search accepts is not evaluated a second time at the start of the next step.

`manifold=` is not supported with `torch.optim` methods, and raises an error.

The {doc}`Rosenbrock notebook <../examples/rosenbrock>` compares piazzi's L-BFGS with PyTorch's, with and without the strong-Wolfe line search.
