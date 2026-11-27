"""Conversions between PyTorch tensors and NumPy arrays."""

from __future__ import annotations

import numpy as np
import torch

__all__ = ["n2t", "t2n"]


def t2n(x: torch.Tensor) -> np.ndarray:
    """Detached float64 NumPy copy of a tensor from any device.

    Always an owned copy: for a CPU float64 tensor ``.numpy()`` alone would alias its storage.
    SciPy's robust-loss scaling is written to work in place (``f *= ...``, ``copy=False``);
    current versions copy first, but a view of PyTorch storage should never depend on that.
    """
    # Move first, then cast: casting on the source device fails where it lacks float64 (MPS).
    return x.detach().cpu().to(torch.float64).numpy().copy()


def n2t(a, like: torch.Tensor) -> torch.Tensor:
    """Copy of an array-like as a tensor with the dtype and device of ``like``."""
    if np.iscomplexobj(a):
        raise TypeError("complex values are not supported (complex-step derivatives cannot pass through PyTorch)")
    # Cast on the CPU, then move, for the same reason as in t2n.
    return torch.from_numpy(np.array(a, dtype=np.float64)).to(like.dtype).to(like.device)
