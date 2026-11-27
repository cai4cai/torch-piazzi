"""Thin wrappers around SciPy solvers, for comparing them with piazzi's native solvers.

SciPy is an optional dependency, so ``import piazzi`` does not load this module. Import it
explicitly, after installing SciPy (``pip install "piazzi[scipy]"``)::

    import piazzi.scipy

    res = piazzi.scipy.least_squares(fun, x0, loss="cauchy")

The wrappers convert between PyTorch and NumPy: ``fun`` takes and returns tensors, and the
result is a :class:`piazzi.OptimizeResult` like the native solvers return.
"""

try:
    import scipy.optimize  # noqa: F401
except ImportError as e:
    raise ImportError(
        'piazzi.scipy needs SciPy, an optional dependency: install it with pip install "piazzi[scipy]"'
    ) from e

from ._least_squares import least_squares, n2t, t2n

__all__ = ["least_squares", "n2t", "t2n"]
