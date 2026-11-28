"""``piazzi``-style front end for PyPose's Gauss-Newton and Levenberg-Marquardt optimisers."""

from __future__ import annotations

import contextlib
import io
import math
from collections.abc import Callable
from typing import Any

import pypose as pp
import torch
from torch import nn

from .._common import Callback, IterationInfo, OptimizeResult, detach_tree, split_aux

__all__ = ["least_squares"]

_SOLVERS = {"pinv": "PINV", "lstsq": "LSTSQ", "cholesky": "Cholesky", "cg": "CG"}
_STRATEGIES = {"constant": "Constant", "adaptive": "Adaptive", "trust_region": "TrustRegion"}


def _plain(x: torch.Tensor) -> torch.Tensor:
    """The stored numbers of ``x`` as a plain tensor (a ``LieTensor`` keeps them in a tensor subclass)."""
    return (x.tensor() if isinstance(x, pp.LieTensor) else x).detach()


class _Residual(nn.Module):
    """PyPose model wrapping ``fun``, which counts and memoises evaluations.

    PyPose's ``step`` evaluates the model at the current point to get the residuals, again
    inside the Jacobian computation, and again for the loss it compares with, so a
    plain evaluation at an already evaluated point is returned from a one-entry memo. An
    evaluation with gradients enabled is the Jacobian's (PyPose runs plain ones under
    ``torch.no_grad``) and always runs.
    """

    def __init__(self, fun: Callable, x0: torch.Tensor, has_aux: bool):
        super().__init__()
        self.fun, self.has_aux = fun, has_aux
        self.x = pp.Parameter(x0.detach().clone()) if isinstance(x0, pp.LieTensor) else nn.Parameter(x0.clone())
        self.n_fun = self.n_system = 0
        self.memo: tuple[torch.Tensor, torch.Tensor, Any] | None = None  # (x, r, aux) of the last plain evaluation

    def forward(self, _=None):
        if torch.is_grad_enabled():
            self.n_system += 1
            r, aux = split_aux(self.fun(self.x), self.has_aux)
            return self._flat(r)
        key = _plain(self.x)
        if self.memo is not None and torch.equal(self.memo[0], key):
            return self.memo[1]
        self.n_fun += 1
        r, aux = split_aux(self.fun(self.x.detach().clone()), self.has_aux)
        r = self._flat(r).detach().clone()  # fun may reuse its output buffer
        self.memo = (key.clone(), r, detach_tree(aux))
        return r

    @staticmethod
    def _flat(r):
        return (r.tensor() if isinstance(r, pp.LieTensor) else r).reshape(-1)  # keeps the graph


def _make(table: dict[str, str], module, kind: str, spec, options: dict | None):
    if spec is None or not isinstance(spec, str):
        if options:
            raise ValueError(f"{kind}_options only apply when {kind} is given by name")
        return spec
    key = spec.lower()
    if key not in table:
        raise ValueError(f"{kind} must be one of {sorted(table)} or a PyPose {kind} object, got {spec!r}")
    return getattr(module, table[key])(**(options or {}))


def least_squares(
    fun: Callable,
    x0: torch.Tensor,
    *,
    method: str = "lm",
    solver: str | Callable | None = None,
    strategy: str | Any | None = None,
    solver_options: dict | None = None,
    strategy_options: dict | None = None,
    reject: int | None = None,
    vectorize: bool = True,
    has_aux: bool = False,
    max_iter: int = 100,
    step_tol: float = 1e-5,
    callback: Callback | None = None,
) -> OptimizeResult:
    """Minimise a sum of squared residuals with PyPose's Gauss-Newton or Levenberg-Marquardt.

    Runs ``pypose.optim.GaussNewton`` (``method="gn"``) or
    ``pypose.optim.LevenbergMarquardt`` (``method="lm"``) on ``fun``, minimising
    ``0.5 * sum(r^2)`` with ``r = fun(x)``, the cost :func:`piazzi.least_squares` uses (PyPose's own
    loss is twice that). ``fun`` runs in the dtype and on the device of ``x0``, and so do PyPose's
    linear solves: unlike :func:`piazzi.gauss_newton`, which solves in float64 on the CPU, a float32
    ``x0`` is solved in float32.

    ``x0`` is either a plain tensor, updated by addition, or a ``pypose.LieTensor``, updated
    on the left by PyPose, ``X <- Exp(delta) X``, so ``fun`` receives a ``LieTensor`` and the
    result ``x`` is one. This replaces ``manifold=``: a :class:`piazzi.Manifold` cannot be used.

    PyPose has no stopping rules: its ``step`` takes one step and returns the loss. The wrapper
    runs up to ``max_iter`` steps and applies piazzi's rules to the loss each returns:

    - an iteration whose cost is below the previous one is accepted. The run converges when the
      step is below ``step_tol`` (max-norm, in the tangent space for a ``LieTensor``) or the cost
      is exactly zero;
    - an iteration whose cost is not below the previous one is rejected (the previous point is
      restored), except that a step below ``step_tol`` that raises the cost by no more than
      rounding noise (``8 * eps * cost``) is accepted and converged. Any other such step ends
      the run unconverged: Gauss-Newton takes full steps without a line search and can
      diverge, and Levenberg-Marquardt stops there after ``reject`` rejected trials. The run
      then returns the best point it accepted;
    - a non-finite cost ends the run unconverged, at the previous point.

    Evaluations are counted by the wrapper. PyPose evaluates the model at the current point for the
    residuals, inside the Jacobian computation and for its loss; plain evaluations at a point
    already evaluated are answered from a one-entry memo, so each trial point costs one
    ``n_fun`` and each iteration one ``n_system`` (the residuals and their Jacobian), plus one
    ``n_fun`` for ``x0`` at the start. The Jacobian evaluation is not memoised and repeats the
    residuals at its point, as PyPose's does. Levenberg-Marquardt counts every rejected trial in
    ``n_fun``. PyPose's reverse-mode ``jacobian`` needs ``fun`` to be differentiable by
    :mod:`torch.autograd`.

    Args:
        fun (Callable): ``fun(x) -> r``, or ``(r, aux)`` with ``has_aux=True``, where ``r`` is a
            tensor of residuals of any shape and ``x`` has the type and shape of ``x0``. ``fun``
            is called with a copy of the point (and with the parameter itself while PyPose
            differentiates, so it must not modify ``x`` in place).
        x0 (Tensor): starting point, a floating-point tensor of any shape or a
            ``pypose.LieTensor``.

    Keyword Args:
        method (str, optional): ``"gn"`` or ``"lm"``. Default: ``"lm"``
        solver (str or Callable, optional): the linear solver for the normal equations: ``"pinv"``,
            ``"lstsq"``, ``"cholesky"`` or ``"cg"``, or a PyPose solver such as
            ``pypose.optim.solver.PCG``. Default: PyPose's: ``"pinv"`` for ``"gn"``,
            ``"cholesky"`` for ``"lm"``
        strategy (str or Any, optional): ``"lm"`` only: the damping strategy, ``"constant"``,
            ``"adaptive"`` or ``"trust_region"``, or a PyPose strategy object. Default: PyPose's
            ``"trust_region"``
        solver_options (dict, optional): constructor arguments of the solver chosen by name.
            Default: ``None``
        strategy_options (dict, optional): constructor arguments of the strategy chosen by name,
            for example ``{"damping": 1e-3}``. Default: ``None``
        reject (int, optional): ``"lm"`` only: how many unsuccessful trial steps PyPose tries in
            one iteration before it accepts a worse one (which the wrapper then undoes).
            Default: PyPose's ``16``
        vectorize (bool, optional): whether PyPose vectorises its Jacobian computation.
            Default: ``True``
        has_aux (bool, optional): whether ``fun`` also returns an auxiliary output (any tree of
            tensors), returned detached in ``result.aux`` and ``info.aux``. Default: ``False``
        max_iter (int, optional): maximum number of PyPose steps. Default: ``100``
        step_tol (float, optional): convergence tolerance on the size of an accepted step.
            Default: ``1e-5``
        callback (Callable, optional): called after every iteration, accepted or not, with an
            :class:`piazzi.IterationInfo`; returning ``True`` stops the run. Default: ``None``

    Returns:
        OptimizeResult: ``x`` with the type, shape, dtype and device of ``x0``, and ``fun`` the
        cost. ``n_fun_grad`` is always ``0``. ``raw`` is the PyPose optimiser.

    See Also:
        - :func:`piazzi.least_squares`: the native solver with the same calling convention.
        - :func:`piazzi.levenberg_marquardt`: the native Levenberg-Marquardt.

    Example::

        >>> t = torch.linspace(0, 4, 40, dtype=torch.float64)
        >>> y = 2.5 * torch.exp(-1.3 * t)
        >>> x0 = torch.tensor([1.0, 0.5], dtype=torch.float64)
        >>> res = piazzi.pypose.least_squares(lambda p: p[0] * torch.exp(-p[1] * t) - y, x0)
        >>> res.x
        tensor([2.5000, 1.3000], dtype=torch.float64)
    """
    if not isinstance(x0, torch.Tensor) or not x0.is_floating_point():
        raise TypeError("x0 must be a floating-point tensor")
    method = method.lower()
    if method not in ("gn", "lm"):
        raise ValueError(f"method must be 'gn' or 'lm', got {method!r}")
    if method == "gn" and (strategy is not None or strategy_options or reject is not None):
        raise ValueError("strategy, strategy_options and reject apply to method='lm' only")
    if max_iter < 0:
        raise ValueError("max_iter must be non-negative")

    pp_solver = _make(_SOLVERS, pp.optim.solver, "solver", solver, solver_options)
    model = _Residual(fun, x0, has_aux)
    if method == "gn":
        opt = pp.optim.GaussNewton(model, solver=pp_solver, vectorize=vectorize)
    else:
        pp_strategy = _make(_STRATEGIES, pp.optim.strategy, "strategy", strategy, strategy_options)
        extra = {} if reject is None else {"reject": reject}
        opt = pp.optim.LevenbergMarquardt(model, solver=pp_solver, strategy=pp_strategy, vectorize=vectorize, **extra)

    is_lie = isinstance(x0, pp.LieTensor)
    eps = torch.finfo(x0.dtype).eps

    def current():
        return model.x.detach().clone()

    def tangent_step(x_new, x_old):
        if is_lie:
            return (x_new @ x_old.Inv()).Log().tensor().abs().max().item()
        return (x_new - x_old).abs().max().item()

    with torch.no_grad():
        r0 = model()  # the residuals at x0, memoised: PyPose's first step reuses them
    aux = model.memo[2]
    cost_t = 0.5 * r0.square().sum()
    cost = float(cost_t)
    n_iter, converged, message = 0, False, f"reached max_iter={max_iter}"
    if not math.isfinite(cost):
        return _result(model, opt, cost_t, aux, 0, False, "the cost at x0 is not finite", current())
    if cost == 0.0:
        return _result(model, opt, cost_t, aux, 0, True, "the cost is zero", current())

    for it in range(max_iter):
        x_old, cost_old, cost_old_t, aux_old = current(), cost, cost_t, aux
        out = io.StringIO()
        with contextlib.redirect_stdout(out):  # PyPose prints when its linear solver fails
            loss = opt.step(None)
        solver_failed = "Linear solver failed" in out.getvalue()
        cost_t = (0.5 * loss).detach().clone()
        cost = float(cost_t)
        x_new = current()
        n_iter = it + 1
        noise = 8 * eps * cost_old
        step = tangent_step(x_new, x_old) if math.isfinite(cost) else math.inf
        if not math.isfinite(cost):
            accepted, stop, converged, message = False, True, False, "the cost became non-finite"
        elif cost < cost_old or (step < step_tol and cost <= cost_old + noise):
            accepted, stop = True, False
            aux = model.memo[2] if model.memo is not None and torch.equal(model.memo[0], _plain(x_new)) else aux
            if cost == 0.0:
                stop, converged, message = True, True, "the cost is zero"
            elif step < step_tol:
                stop, converged, message = True, True, "the step is below step_tol"
        else:
            accepted, stop, converged = False, True, False
            message = (
                "PyPose's linear solver failed"
                if solver_failed
                else "PyPose's step did not decrease the cost"
                + (f" after {opt.reject} trial steps" if method == "lm" else " (no line search in Gauss-Newton)")
            )
        if not accepted:  # undo the step exactly (PyPose's own undo is only up to rounding)
            with torch.no_grad():
                model.x.copy_(x_old)
            if hasattr(opt, "loss"):
                opt.loss = 2 * cost_old_t
            cost, cost_t, aux = cost_old, cost_old_t, aux_old
        if callback is not None:
            info = IterationInfo(
                it, current(), cost_t.clone(), cost_old_t.clone(), aux, accepted, 0, model.n_fun, model.n_system
            )
            if callback(info):
                message = message if stop else "stopped by the callback"
                stop = True
        if stop:
            break
    return _result(model, opt, cost_t, aux, n_iter, converged, message, current())


def _result(model, opt, cost_t, aux, n_iter, converged, message, x):
    return OptimizeResult(
        x=x,
        fun=cost_t,
        aux=aux,
        n_iter=n_iter,
        n_fun_grad=0,
        n_fun=model.n_fun,
        n_system=model.n_system,
        converged=converged,
        message=message,
        raw=opt,
    )
