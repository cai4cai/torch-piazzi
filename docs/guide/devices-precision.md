# Devices and precision

The objective runs on the device and in the dtype of `x0`. Only the small `n x n` normal equations of {func}`~piazzi.gauss_newton` (and {func}`~piazzi.least_squares`) move: they are solved in `solve_dtype` on `solve_device`, float64 on the CPU by default. That split lets the expensive part (rendering, residuals and Jacobians) run in float32 on a GPU, while the solve keeps float64 accuracy and stays cheap for small `n`.

```python
x0 = x0.to("cuda", torch.float32)  # or "mps"
res = piazzi.least_squares(residuals, x0)  # residuals and Jacobians on the GPU, solve in float64 on the CPU
res = piazzi.least_squares(residuals, x0, solve_device="cuda", solve_dtype=torch.float32)  # everything on the GPU
```

For large sparse problems the solve dominates each iteration, so keeping it on the GPU pays off; the {doc}`bundle adjustment notebook <../examples/bundle_adjustment>` measures both placements on CUDA and Apple MPS.

## float32

- Use tolerances that float32 can reach: `step_tol=1e-10` never triggers in float32.
- Gauss-Newton, Levenberg-Marquardt and L-BFGS recognise a minimiser once the predicted decrease falls below the rounding noise of the loss, so `converged` is reported correctly at a float32 minimiser.
- Finite differences (in the {doc}`SciPy backends <backends>`) need a float64 `x0`: in float32 their steps are lost to rounding.

## Apple MPS

MPS has no float64. piazzi never creates a float64 tensor on MPS: transfers to the default float64 CPU solve cast on the CPU side. Sparse CSR tensors can't be built on MPS, so use COO there.

## Testing on accelerators

`tests/test_devices.py` runs the solvers on every available device: always the CPU, plus CUDA or MPS when present. CI is CPU-only, so run `uv run pytest tests/test_devices.py` on a Mac or a CUDA machine after touching device or dtype handling.
