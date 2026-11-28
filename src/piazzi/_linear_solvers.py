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
from typing import cast

import torch

__all__ = ["LinearSolver", "cg_solver", "dense_solver"]

#: Signature of a linear solver for :func:`piazzi.gauss_newton`: ``solver(H, g, damping) -> d``
#: solves ``(H + diag(damping)) d = -g``, with ``H`` of shape ``(n, n)`` (dense, or sparse COO
#: or CSR) and ``g``, ``damping`` and ``d`` of shape ``(n,)``.
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
    """Solve the damped normal equations directly, with :func:`torch.linalg.solve`.

    The default :data:`piazzi.LinearSolver` of :func:`piazzi.gauss_newton` for a dense ``H``.
    A sparse ``H`` is densified first.

    Args:
        H (Tensor): curvature model of shape ``(n, n)``, dense or sparse.
        g (Tensor): gradient of shape ``(n,)``.
        damping (Tensor): diagonal damping of shape ``(n,)``.

    Returns:
        Tensor: the step ``d`` of shape ``(n,)`` that solves ``(H + diag(damping)) d = -g``.
    """
    if is_sparse(H):
        H = H.to_dense()
    return torch.linalg.solve(H + torch.diag(damping), -g)


def cg_solver(
    tol: float = 1e-6,
    max_iter: int = 500,
    preconditioner: str | Callable[[torch.Tensor, torch.Tensor], Callable] | None = "jacobi",
) -> LinearSolver:
    """Make a preconditioned conjugate-gradient solver for large sparse normal equations.

    The returned :data:`piazzi.LinearSolver` is the default of :func:`piazzi.gauss_newton`
    for a sparse ``H``. It uses CG from `torchsparsegradutils`_, which needs the optional
    dependency (``pip install "piazzi[sparse]"``). ``H`` is only used through its diagonal
    and matrix-vector products, so it can be any sparse layout that ``H @ v`` supports. CG
    stops at a relative residual of ``tol`` or after ``max_iter`` iterations, without a
    warning: an inexact step is fine for Gauss-Newton and Levenberg-Marquardt, whose line
    search or gain ratio checks it. If CG breaks down (a numerically singular damped
    system), the solver returns the preconditioned gradient step ``-M^{-1} g`` instead,
    which is still a descent direction.

    Args:
        tol (float, optional): relative residual at which CG stops; must be positive.
            Default: ``1e-6``
        max_iter (int, optional): maximum number of CG iterations per solve.
            Default: ``500``
        preconditioner (str or Callable, optional): ``"jacobi"`` divides by
            ``diag(H) + damping``. A callable ``preconditioner(H, damping)`` returning
            ``v -> M^{-1} v`` supplies your own (for example block-Jacobi, inverting each
            parameter block's diagonal block). ``None`` disables preconditioning.
            Default: ``"jacobi"``

    Returns:
        LinearSolver: ``solver(H, g, damping) -> d``, for the ``linear_solver`` argument of
        :func:`piazzi.gauss_newton`.

    .. _torchsparsegradutils: https://github.com/cai4cai/torchsparsegradutils
    """
    if not tol > 0:
        raise ValueError(f"tol must be positive, got {tol!r}")
    if not (preconditioner is None or preconditioner == "jacobi" or callable(preconditioner)):
        raise ValueError(f"preconditioner must be 'jacobi', a callable or None, got {preconditioner!r}")
    try:
        from torchsparsegradutils.utils import linear_cg
    except ImportError as err:
        raise ImportError(
            'cg_solver needs torchsparsegradutils: pip install "piazzi[sparse]", '
            "or pass another linear_solver (for example piazzi.dense_solver)"
        ) from err

    def solve(H, g, damping):
        precond: Callable | None
        if preconditioner == "jacobi":
            inv_diag = 1.0 / (diagonal(H) + damping).clamp(min=torch.finfo(H.dtype).tiny)

            def jacobi(v):
                return v * (inv_diag if v.dim() == 1 else inv_diag[:, None])

            precond = jacobi
        elif preconditioner is None:
            precond = None
        else:
            precond = preconditioner(H, damping)

        def matmul(v):
            return (H @ v if v.dim() == 2 else matvec(H, v)) + (damping * v if v.dim() == 1 else damping[:, None] * v)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "CG terminated", UserWarning)
            try:
                # Without the optional outputs, linear_cg returns the solution alone.
                return cast(
                    torch.Tensor,
                    linear_cg(
                        matmul,
                        -g,
                        tolerance=tol,
                        max_iter=max_iter,
                        max_tridiag_iter=min(max_iter, 20),
                        preconditioner=precond,
                    ),
                )
            except RuntimeError as err:
                if "breakdown" not in str(err):
                    raise
                d = -(precond(g) if precond is not None else g)
                return d if torch.isfinite(d).all() else -g

    return solve
