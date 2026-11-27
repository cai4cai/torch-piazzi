# Changelog

## Unreleased

piazzi is pre-alpha and not yet on PyPI. Development so far:

- Solvers: L-BFGS with an Armijo line search, Gauss-Newton and Levenberg-Marquardt, `least_squares` with robust losses, and `minimize` with `torch.optim` optimisers.
- `finite_difference`, a wrapper that gives black-box objectives finite-difference derivatives.
- Manifolds (`SO3`, `SE3`, `Product`, `Subset`) and batched Lie-group maps.
- Sparse normal equations, a `linear_solver` hook with dense and CG solvers, and `damping="marquardt"`.
- Optional SciPy wrappers (`piazzi.scipy`).
- Device tests on the CPU, CUDA and MPS, in float32 and float64.
- Notebooks: robust regression, Rosenbrock, bundle adjustment.
- This documentation site.
