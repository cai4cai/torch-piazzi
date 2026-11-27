"""Nonlinear least squares, like ``scipy.optimize.least_squares``, on top of :func:`gauss_newton`."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.utils._pytree as pytree

from ._common import OptimizeResult, quiet_jvp_warnings
from ._gauss_newton import gauss_newton
from ._losses import Loss, check_loss, rho_and_derivatives, robust_cost
from ._manifolds import Manifold

__all__ = ["least_squares", "least_squares_system"]


def least_squares_system(
    residual_fun: Callable,
    has_aux: bool = False,
    *,
    loss: Loss = "linear",
    f_scale: float = 1.0,
    curvature: str = "irls",
    manifold: Manifold | None = None,
) -> Callable:
    """Build normal equations from residuals, for :func:`piazzi.gauss_newton`.

    The returned ``system(x)`` gives ``(loss, g, H)`` for the cost
    ``F = 0.5 * sum(C^2 * rho(z))`` with ``z = (r / C)^2`` and ``C = f_scale``, as in
    :func:`scipy.optimize.least_squares`. With the default ``loss="linear"`` this is
    ``F = 0.5 * ||r||^2``, ``g = J^T r`` and ``H = J^T J``. For a robust loss the gradient is
    ``g = J^T (rho'(z) * r)`` and ``H = J^T diag(w) J``, with weights ``w`` set by
    ``curvature``. The Jacobian ``J`` of shape ``(m, n)`` comes from forward-mode autodiff
    (:func:`torch.func.jacfwd`), which is cheap when there are few parameters. For large or
    structured problems, write ``system`` by hand and reduce ``J^T J`` without
    materialising ``J``.

    Args:
        residual_fun (Callable): ``residual_fun(x) -> r``, or ``(r, aux)`` with
            ``has_aux=True``, where ``r`` is a tensor of residuals of any shape (flattened
            to ``(m,)``) and ``x`` has the shape of the point passed to ``system``.
        has_aux (bool, optional): whether ``residual_fun`` also returns an auxiliary output
            (any tree of tensors), which ``system`` then returns as a fourth element.
            Default: ``False``

    Keyword Args:
        loss (str or Callable, optional): a name from :data:`piazzi.LOSSES`
            (``"linear"``, ``"huber"``, ``"soft_l1"``, ``"cauchy"``, ``"arctan"``), or a
            callable ``rho(z)`` written in PyTorch and applied elementwise, whose derivatives
            come from autograd. Default: ``"linear"``
        f_scale (float, optional): soft margin ``C`` between inlier and outlier residuals,
            in the units of ``r``. Default: ``1.0``
        curvature (str, optional): the weights ``w`` of ``H = J^T diag(w) J`` for a robust
            loss. The gradient does not depend on it, so the choice changes the path, not
            the minimiser. Default: ``"irls"``

            - ``"irls"``: ``w = rho'(z)``, as in iteratively reweighted least squares.
              Ceres Solver uses the same weights for losses with ``rho'' <= 0``, which
              includes every built-in loss.
            - ``"scipy"``: ``w = max(rho'(z) + 2 * rho''(z) * z, eps)``, the model of
              :func:`scipy.optimize.least_squares`. It is more accurate for inliers, but for
              outliers the bracket is negative and floored at ``eps``. When most residuals
              start as outliers, ``H`` is then nearly singular and Levenberg-Marquardt can
              stall; ``method="gn"`` copes better.
        manifold (Manifold, optional): keeps ``x`` on a manifold. ``J`` is then the
            Jacobian of ``delta -> r(manifold.plus(x, delta))`` at ``delta = 0``, so ``g``
            and ``H`` are in tangent coordinates (for example 6 columns for
            :class:`piazzi.SE3`). Pass the same manifold to :func:`piazzi.gauss_newton`.
            Default: ``None``

    Returns:
        Callable: ``system(x) -> (loss, g, H)``, or ``(loss, g, H, aux)`` with ``has_aux``,
        with ``g`` of shape ``(n,)`` and ``H`` of shape ``(n, n)``.

    See Also:
        - :func:`piazzi.least_squares`: the high-level interface, which builds and solves this system
          for you.
    """
    check_loss(loss, f_scale)
    if curvature not in ("irls", "scipy"):
        raise ValueError(f"curvature must be 'irls' or 'scipy', got {curvature!r}")
    robust = loss != "linear"

    def system(x):
        shape = x.shape
        # jacfwd passes only tensors through has_aux, so aux is split into its tensor leaves
        # (returned through jacfwd) and the rest (kept here), and put back together below.
        static = {}

        def flat(xf):
            out = residual_fun(xf.reshape(shape))
            if has_aux:
                r, aux = out
                leaves, static["spec"] = pytree.tree_flatten(aux)
                static["leaves"] = leaves
                tensors = [leaf for leaf in leaves if isinstance(leaf, torch.Tensor)]
                return r.reshape(-1), (r.reshape(-1), tensors)
            return out.reshape(-1), (out.reshape(-1),)

        def rebuild(tensors):
            it = iter(tensors)
            leaves = [next(it) if isinstance(leaf, torch.Tensor) else leaf for leaf in static["leaves"]]
            return pytree.tree_unflatten(leaves, static["spec"])

        with quiet_jvp_warnings():
            if manifold is None:
                J, extra = torch.func.jacfwd(flat, has_aux=True)(x.reshape(-1))
            else:
                delta = x.new_zeros(manifold.tangent_size(x))
                if delta.numel() == 0:  # every parameter fixed: J has no columns (jacfwd cannot do this)
                    _, extra = flat(manifold.plus(x, delta))
                    J = extra[0].new_zeros(extra[0].numel(), 0)
                else:
                    J, extra = torch.func.jacfwd(lambda d: flat(manifold.plus(x, d)), has_aux=True)(delta)
        r, aux = extra[0], (rebuild(extra[1]) if has_aux else None)
        if robust:
            z = (r / f_scale) ** 2
            rho, d1, d2 = rho_and_derivatives(z, loss)
            loss_val = 0.5 * f_scale**2 * rho.sum()
            w = d1 if curvature == "irls" else (d1 + 2 * d2 * z).clamp(min=torch.finfo(r.dtype).eps)
            g, H = J.T @ (d1 * r), J.T @ (w[:, None] * J)
        else:
            loss_val = 0.5 * torch.dot(r, r)
            g, H = J.T @ r, J.T @ J
        return (loss_val, g, H, aux) if has_aux else (loss_val, g, H)

    return system


def least_squares(
    fun: Callable,
    x0: torch.Tensor,
    *,
    loss: Loss = "linear",
    f_scale: float = 1.0,
    method: str = "lm",
    has_aux: bool = False,
    curvature: str = "irls",
    manifold: Manifold | None = None,
    **kwargs,
) -> OptimizeResult:
    """Minimise a sum of squared residuals, like :func:`scipy.optimize.least_squares`.

    Minimises ``0.5 * sum(f_scale^2 * rho((r / f_scale)^2))`` with ``r = fun(x)``. This
    builds the normal equations with :func:`piazzi.least_squares_system` and solves them
    with :func:`piazzi.gauss_newton`, using the residuals alone, without the Jacobian, for
    value-only evaluations. With a ``manifold`` such as :class:`piazzi.SE3`, ``x0`` is a
    point on it (a ``(4, 4)`` pose) and every iterate stays on it.

    Args:
        fun (Callable): ``fun(x) -> r``, or ``(r, aux)`` with ``has_aux=True``, where ``r``
            is a tensor of residuals of any shape and ``x`` has the shape of ``x0``.
        x0 (Tensor): starting point, of any shape, or a point on ``manifold``.

    Keyword Args:
        loss (str or Callable, optional): a name from :data:`piazzi.LOSSES` or a callable
            ``rho(z)``, as in :func:`piazzi.least_squares_system`. Default: ``"linear"``
        f_scale (float, optional): soft margin between inlier and outlier residuals, in the
            units of ``r``. Default: ``1.0``
        method (str, optional): ``"lm"`` (Levenberg-Marquardt) or ``"gn"`` (Gauss-Newton).
            Default: ``"lm"``
        has_aux (bool, optional): whether ``fun`` also returns an auxiliary output (any tree
            of tensors), returned detached in ``result.aux`` and ``info.aux``.
            Default: ``False``
        curvature (str, optional): ``"irls"`` or ``"scipy"``, as in
            :func:`piazzi.least_squares_system`. Default: ``"irls"``
        manifold (Manifold, optional): keeps ``x`` on a manifold (see
            :class:`piazzi.Manifold`). Default: ``None``
        **kwargs: other keyword arguments of :func:`piazzi.gauss_newton`, such as
            ``max_iter``, ``step_tol``, ``max_step``, ``damping``, ``linear_solver`` or
            ``callback``.

    Returns:
        OptimizeResult: as for :func:`piazzi.gauss_newton`. ``result.fun`` is the cost, not
        the residuals.

    See Also:
        - :func:`piazzi.gauss_newton`: the method this runs (``method="lm"`` or ``"gn"``), which
          documents the other keyword arguments.
        - :func:`piazzi.least_squares_system`: the normal equations this builds from ``fun``.
        - :func:`piazzi.scipy.least_squares`: the same problem with SciPy's solvers.

    Example::

        >>> t = torch.linspace(0, 4, 40)
        >>> y = 2.5 * torch.exp(-1.3 * t)
        >>> y[::10] += 2.0  # outliers
        >>> def residuals(theta):
        ...     return theta[0] * torch.exp(-theta[1] * t) - y
        >>> res = piazzi.least_squares(residuals, torch.tensor([1.0, 0.5]), loss="cauchy", f_scale=0.1)
        >>> res.x
        tensor([2.5026, 1.3005])
    """
    system = least_squares_system(fun, has_aux, loss=loss, f_scale=f_scale, curvature=curvature, manifold=manifold)

    def value_fun(x):
        out = fun(x)
        return robust_cost((out[0] if has_aux else out).reshape(-1), loss, f_scale)

    kwargs.setdefault("value_fun", value_fun)
    return gauss_newton(system, x0, method=method, has_aux=has_aux, manifold=manifold, **kwargs)
