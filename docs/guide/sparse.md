# Large sparse problems

`system(x)` may return `H` as a sparse COO or CSR tensor, as in bundle adjustment, where `JᵀJ` has millions of entries but is 99.7 % zeros. {func}`~piazzi.gauss_newton` then needs only the diagonal of `H` and products with it. The `linear_solver` argument chooses how the damped system `(H + diag(damping)) d = -g` is solved:

| Linear solver | Notes |
|---|---|
| {func}`piazzi.dense_solver` | Direct `torch.linalg.solve`. This is the default for a dense `H`. |
| {func}`piazzi.cg_solver` | Preconditioned conjugate gradients from [torchsparsegradutils](https://github.com/cai4cai/torchsparsegradutils). This is the default for a sparse `H`. It accepts your own preconditioner, for example block-Jacobi. It is an optional dependency: `pip install "piazzi[sparse]"`. |
| any `solver(H, g, damping) -> d` | See {data}`piazzi.LinearSolver`. For example a Schur-complement solver that eliminates the points in bundle adjustment. |

`damping="marquardt"` (see {doc}`solvers`) is usually needed on these problems, whose parameters have very different scales.

The {doc}`bundle adjustment notebook <../examples/bundle_adjustment>` builds per-observation Jacobians with `vmap(jacfwd)` through the camera poses' manifold, assembles a sparse `JᵀJ`, and compares the CG solver with a Schur complement written as a custom `linear_solver`, on the CPU and on a GPU.
