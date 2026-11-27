"""Conversions between PyTorch tensors and NumPy arrays."""

from __future__ import annotations

import numpy as np
import torch

__all__ = ["bounds_to_scipy", "iteration_callback", "n2t", "t2n", "to_scipy_array"]


def t2n(x: torch.Tensor) -> np.ndarray:
    """Convert a tensor to a float64 NumPy array, as the SciPy wrappers do.

    Always an owned copy: for a CPU float64 tensor ``.numpy()`` alone would alias its storage.
    SciPy's robust-loss scaling is written to work in place (``f *= ...``, ``copy=False``);
    current versions copy first, but a view of PyTorch storage should never depend on that.

    Args:
        x (Tensor): a tensor on any device, of any dtype and shape.

    Returns:
        numpy.ndarray: a detached float64 copy, with the shape of ``x``.
    """
    # Move first, then cast: casting on the source device fails where it lacks float64 (MPS).
    return x.detach().cpu().to(torch.float64).numpy().copy()


def n2t(a, like: torch.Tensor) -> torch.Tensor:
    """Convert an array-like to a tensor, as the SciPy wrappers do.

    Args:
        a (array_like): a NumPy array, scalar or nested sequence.
        like (Tensor): the tensor whose dtype and device the result takes.

    Returns:
        Tensor: a copy of ``a`` with the dtype and device of ``like``.
    """
    if np.iscomplexobj(a):
        raise TypeError("complex values are not supported (complex-step derivatives cannot pass through PyTorch)")
    # Cast on the CPU, then move, for the same reason as in t2n.
    return torch.from_numpy(np.array(a, dtype=np.float64)).to(like.dtype).to(like.device)


def to_scipy_array(v):
    """A tensor or array argument (bounds, ``x_scale``, ``diff_step``) for SciPy.

    Arrays and tensors become float64 NumPy arrays flattened to match the flat ``x0``; 0-dim
    ones become Python floats, which SciPy broadcasts. Anything else passes through.
    """
    if isinstance(v, torch.Tensor):
        v = t2n(v)
    elif isinstance(v, np.ndarray):
        v = v.astype(np.float64)
    else:
        return v
    return v.reshape(-1) if v.ndim else float(v)


def bounds_to_scipy(bounds):
    """``bounds`` for SciPy: a ``(lb, ub)`` pair, a ``Bounds``, or an (n, 2) tensor of pairs."""
    import scipy.optimize

    if isinstance(bounds, scipy.optimize.Bounds):
        keep = np.asarray(bounds.keep_feasible)
        return scipy.optimize.Bounds(
            to_scipy_array(np.asarray(t2n(bounds.lb) if isinstance(bounds.lb, torch.Tensor) else bounds.lb)),
            to_scipy_array(np.asarray(t2n(bounds.ub) if isinstance(bounds.ub, torch.Tensor) else bounds.ub)),
            keep_feasible=keep.reshape(-1) if keep.ndim else keep,
        )
    if isinstance(bounds, torch.Tensor):  # (n, 2) tensor of (min, max) pairs
        return t2n(bounds).reshape(-1, 2)
    if isinstance(bounds, tuple) and len(bounds) == 2:
        return tuple(to_scipy_array(b) for b in bounds)
    return bounds


def iteration_callback(callback, like, shape, last, counts, cost_of):
    """Adapt a piazzi ``callback(info)`` to SciPy's ``callback(intermediate_result)``.

    The callback gets an :class:`piazzi.IterationInfo` with tensors, and returning True stops
    SciPy (through ``StopIteration``). ``cost_of(intermediate_result)`` gives the value;
    ``last`` holds the most recent evaluation, whose ``fun`` and ``aux`` are used when it
    was at the same point. TNC passes only ``x`` and cannot be stopped.
    """
    if callback is None:
        return None
    from .._common import IterationInfo

    state = {"it": 0}

    def scipy_callback(intermediate_result):
        x_only = isinstance(intermediate_result, np.ndarray)
        xn = intermediate_result if x_only else intermediate_result.x
        if "x" in last and np.array_equal(last["x"], xn):
            fun, aux = last["fun"], last["aux"]
        else:
            fun = n2t(np.nan if x_only else cost_of(intermediate_result), like)
            aux = None
        info = IterationInfo(
            state["it"],
            n2t(xn, like).reshape(shape),
            fun,
            state.get("fun", last.get("start", fun)),
            aux,
            True,
            counts["fg"],
            counts["f"],
            counts["sys"],
        )
        state["it"] += 1
        state["fun"] = fun
        if callback(info) and not x_only:
            raise StopIteration

    return scipy_callback
