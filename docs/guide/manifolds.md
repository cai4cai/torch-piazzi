# Manifolds: poses and rotations

Pass `manifold=` to {func}`~piazzi.least_squares`, {func}`~piazzi.gauss_newton`, {func}`~piazzi.lbfgs_armijo` or {func}`~piazzi.minimize` to keep `x` on a constrained set, as Ceres Solver's [`Manifold`](https://ceres-solver.readthedocs.io/latest/nnls_modeling.html#manifold) does. A manifold supplies `plus(x, delta)`, which moves `x` by a tangent vector. The solver takes its Jacobian or gradient with respect to `delta` at zero, by autodiff through `plus`, and applies each step as `x <- plus(x, step)`. A 4 x 4 pose therefore gives 6 Jacobian columns, and every iterate stays a rigid transform.

| Manifold | Stored as | Tangent | Notes |
|---|---|---|---|
| {class}`~piazzi.Euclidean` | any tensor | same size | The default behaviour; a shape is only needed inside `Product`. |
| {class}`~piazzi.Subset` | vector | free entries | Holds some entries fixed, like Ceres's `SubsetManifold`. |
| {class}`~piazzi.SO3` | 3 x 3 rotation | 3 | `R @ so3_exp(w)` (body frame), or `so3_exp(w) @ R` with `side="left"`. |
| {class}`~piazzi.SE3` | 4 x 4 rigid transform | 6, `(rho, omega)` | `retraction="exp"` uses the true SE(3) exponential (a screw motion). `pivot` sets the point that rotations turn about. |
| {class}`~piazzi.Product` | flat vector | sum | Use `M.join(...)` to build `x0` and `M.split(x)` to unpack it, for example a pose plus camera intrinsics. |

The manifolds also provide `minus(y, x)`, which inverts `plus`. The solvers don't use it, but it measures errors between points. {func}`~piazzi.so3_exp`, {func}`~piazzi.so3_log`, {func}`~piazzi.se3_exp`, {func}`~piazzi.se3_log` and {func}`~piazzi.so3_hat` work on batches, with finite derivatives in forward and reverse mode, including at the identity and near a half-turn.

## Rigid registration

```python
P, Q = ...  # matching 3D points, Q ~ R @ p + t
res = piazzi.least_squares(
    lambda T: P @ T[:3, :3].T + T[:3, 3] - Q, torch.eye(4), manifold=piazzi.SE3(pivot=P.mean(0)), max_step=1e3
)
res.x  # a 4 x 4 rigid transform
```

**Set the pivot for registration.** Put `pivot` at the centre of the object. If the object sits far from the origin, a rotation about the origin also moves it a long way. Rotation and translation are then strongly coupled and Gauss-Newton crawls. In the tests, a point set 140 units from the origin converges in 5 Gauss-Newton iterations with the pivot at its centre, and has not converged after 10 without it.

**Set `max_step` for the problem's units.** It caps every tangent component, so it applies to radians and translation units alike. Its default of 1 can slow down a large translation.

## Design choices

Two choices matter most in practice for pose problems:

- **Which point rotations are about.** Ceres's quaternion manifold and PyPose perturb on the left (about the world origin); for an object far from the origin a small rotation then also moves it a long way, which couples rotation and translation. {class}`~piazzi.SE3` perturbs on the right by default and takes a `pivot`.
- **Decoupled vs screw retraction, and SE(3) vs `Product(SO3, R^3)`.** All of these describe the same set of poses, so they share minimisers. The decoupled and screw retractions have identical derivatives at `delta = 0`, so they give the same Gauss-Newton step and differ only in the curve along which it is applied. A product with a world-frame translation has a Jacobian that differs by an invertible linear map; Gauss-Newton steps are invariant to that, but Levenberg-Marquardt damping, L-BFGS scaling and `max_step` are not. The decoupled update (`rho` added as a body-frame translation) is the simpler default; the true SE(3) exponential is available as `retraction="exp"`.

The design follows Ceres's `Manifold`, which keeps parameters as plain tensors. The [`_manifolds.py`](https://github.com/cai4cai/torch-piazzi/blob/main/src/piazzi/_manifolds.py) module docstring compares it with Ceres, [PyPose](https://github.com/pypose/pypose) and [GeoTorch](https://github.com/lezcano/geotorch), and [`tests/test_ceres_manifolds.py`](https://github.com/cai4cai/torch-piazzi/blob/main/tests/test_ceres_manifolds.py) ports Ceres's manifold and rotation tests.

## Writing your own manifold

Subclass {class}`~piazzi.Manifold`, set `ambient_shape` (the shape a point is stored in, or `None` for any shape) and implement `tangent_size(x)` and `plus(x, delta)`; `minus(y, x)` is optional. The solvers only need `plus` and its derivative at `delta = 0`, which they take by autodiff, so `plus` must be differentiable in forward mode (`torch.func.jacfwd`).

Optionally override `entry_scale(x)`, the size each entry of a point is computed at. The solvers treat a change of a few units in the last place of it as rounding noise when they decide that a Newton-like step no longer moves `x`. The default, `|x|`, suits independent coordinates; {class}`~piazzi.SO3` and {class}`~piazzi.SE3` use 1 for rotation-matrix entries, whose rows have norm 1.
