# Changelog

## Unreleased

piazzi is pre-alpha and not yet on PyPI. Development so far:

- Solvers: L-BFGS with an Armijo line search, Gauss-Newton and Levenberg-Marquardt, `least_squares` with robust losses, and `minimize` with `torch.optim` optimisers.
- `finite_difference`, a wrapper that gives black-box objectives finite-difference derivatives.
- Manifolds (`SO3`, `SE3`, `Product`, `Subset`) and batched Lie-group maps.
- Sparse normal equations, a `linear_solver` hook with dense and CG solvers, and `damping="marquardt"`.
- Optional SciPy wrappers (`piazzi.scipy`).
- Optional PyPose wrapper (`piazzi.pypose`): its Gauss-Newton and Levenberg-Marquardt with piazzi's stopping rules and evaluation counts.
- Device tests on the CPU, CUDA and MPS, in float32 and float64.
- Float32 runs that reach a minimiser near a zero residual report convergence (`step below the resolution of x`), judged entry by entry at each point's scale (`Manifold.entry_scale`).
- `step_tol` is judged on a step that estimates the distance to the minimiser, so heavy Levenberg-Marquardt damping or a long backtracking line search no longer ends a run as converged far from a minimiser.
- Notebooks: robust regression, Rosenbrock, bundle adjustment.
- This documentation site.
- Requires Python 3.11 and PyTorch 2.15 or later, and CI tests the lowest allowed versions of every dependency.
- `print_progress`, a callback that prints a table of a run's progress, and `verbose=` on every solver and wrapper to use it.
- Dynamic PEP 440 versioning from git tags via `hatch-vcs`, and a Trusted Publishing GitHub Actions workflow for TestPyPI and PyPI.
