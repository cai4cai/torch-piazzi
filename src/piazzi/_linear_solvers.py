"""Linear solvers for the damped normal equations ``(H + diag(damping)) d = -g`` of :func:`gauss_newton`.

``H`` is a dense tensor or a sparse COO/CSR tensor and ``damping`` a vector (n,) added to
its diagonal (``lam`` everywhere for identity damping, ``lam * diag(H)`` for Marquardt
damping). A solver is any callable ``solver(H, g, damping) -> d``; :func:`gauss_newton`
picks :func:`dense_solver` for a dense ``H`` and :func:`cg_solver` for a sparse one, unless
``linear_solver`` says otherwise.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import torch

__all__ = ["LinearSolver", "cg_solver", "dense_solver"]

LinearSolver = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def is_sparse(H: torch.Tensor) -> bool:
    return H.layout != torch.strided


def diagonal(H: torch.Tensor) -> torch.Tensor:
    """Diagonal of a dense or sparse (COO/CSR) square matrix."""
    if not is_sparse(H):
        return H.diagonal()
    coo = (H if H.layout == torch.sparse_coo else H.to_sparse_coo()).coalesce()
    rows, cols = coo.indices()
    on_diag = rows == cols
    out = torch.zeros(H.shape[0], dtype=H.dtype, device=H.device)
    return out.index_add_(0, rows[on_diag], coo.values()[on_diag])


def matvec(H: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """``H @ v`` for a dense or sparse ``H`` and a vector ``v``."""
    return (H @ v[:, None])[:, 0] if is_sparse(H) else H @ v


def dense_solver(H: torch.Tensor, g: torch.Tensor, damping: torch.Tensor) -> torch.Tensor:
    """Direct solve with ``torch.linalg.solve`` (a sparse ``H`` is densified)."""
    if is_sparse(H):
        H = H.to_dense()
    return torch.linalg.solve(H + torch.diag(damping), -g)


def cg_solver(
    tol: float = 1e-6,
    max_iter: int = 500,
    preconditioner: str | Callable[[torch.Tensor, torch.Tensor], Callable] | None = "jacobi",
) -> LinearSolver:
    """Preconditioned conjugate gradients from `torchsparsegradutils`_, for large sparse ``H``.

    Returns a ``solver(H, g, damping)`` for the ``linear_solver`` argument of
    :func:`gauss_newton`. ``H`` is only used through its diagonal and matrix-vector
    products, so it can be any sparse layout ``H @ v`` supports. CG stops at a relative
    residual of ``tol`` (> 0) or after ``max_iter`` iterations, without a warning: an
    inexact step is fine for Gauss-Newton and Levenberg-Marquardt, whose line search or gain
    ratio checks it. If CG breaks down (a numerically singular damped system), the solver
    returns the preconditioned gradient step ``-M^{-1} g`` instead, which is still a descent
    direction.

    Args:
        preconditioner: ``"jacobi"`` (default) divides by ``diag(H) + damping``. A callable
            ``preconditioner(H, damping)`` returning ``v -> M^{-1} v`` supplies your own
            (for example block-Jacobi, inverting each parameter block's diagonal block).
            None disables preconditioning.

    Needs the optional dependency ``torchsparsegradutils`` (``pip install "piazzi[sparse]"``).

    .. _torchsparsegradutils: https://github.com/cai4cai/torchsparsegradutils
    """
    try:
        from torchsparsegradutils.utils import linear_cg
    except ImportError as err:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            'cg_solver needs torchsparsegradutils: pip install "piazzi[sparse]", '
            "or pass another linear_solver (for example piazzi.dense_solver)"
        ) from err
    if not tol > 0:
        raise ValueError(f"tol must be positive, got {tol!r}")
    if not (preconditioner is None or preconditioner == "jacobi" or callable(preconditioner)):
        raise ValueError(f"preconditioner must be 'jacobi', a callable or None, got {preconditioner!r}")

    def solve(H, g, damping):
        if preconditioner == "jacobi":
            inv_diag = 1.0 / (diagonal(H) + damping).clamp(min=torch.finfo(H.dtype).tiny)

            def precond(v):
                return v * (inv_diag if v.dim() == 1 else inv_diag[:, None])

        elif preconditioner is None:
            precond = None
        else:
            precond = preconditioner(H, damping)

        def matmul(v):
            return (H @ v if v.dim() == 2 else matvec(H, v)) + (damping * v if v.dim() == 1 else damping[:, None] * v)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "CG terminated", UserWarning)
            try:
                return linear_cg(
                    matmul,
                    -g,
                    tolerance=tol,
                    max_iter=max_iter,
                    max_tridiag_iter=min(max_iter, 20),
                    preconditioner=precond,
                )
            except RuntimeError as err:
                if "breakdown" not in str(err):
                    raise
                d = -(precond(g) if precond is not None else g)
                return d if torch.isfinite(d).all() else -g

    return solve
