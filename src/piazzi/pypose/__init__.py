"""A wrapper around PyPose's second-order optimisers, for comparing them with piazzi's native solvers.

PyPose is an optional dependency, so ``import piazzi`` does not load this module. Import it
explicitly, after installing PyPose (``pip install "piazzi[pypose]"``)::

    import piazzi.pypose

    res = piazzi.pypose.least_squares(fun, x0, method="lm", strategy="adaptive")

The wrapper gives ``pypose.optim.GaussNewton`` and ``pypose.optim.LevenbergMarquardt``
piazzi's calling convention: ``fun`` is a residual function of ``x`` (a plain tensor, or a
``pypose.LieTensor``), and the result is a :class:`piazzi.OptimizeResult`.
"""

import warnings

try:
    # Importing PyPose triggers two warnings of its own, which the test suite turns into errors:
    # torch's deprecation of torch.jit.script (PyPose uses it for its Lie group kernels; torch says "not
    # supported" on Python 3.14 and later), and an
    # invalid escape sequence in a docstring, reported only while compiling pypose/testing/comparison.py,
    # that is, on the first import after installation without byte-compiled files. Both are PyPose's
    # (https://github.com/pypose/pypose); drop the filters once a release no longer triggers them.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"`torch\.jit\.script` is (deprecated|not supported)", FutureWarning)
        # A DeprecationWarning before Python 3.12, a SyntaxWarning from then on.
        warnings.filterwarnings("ignore", r".*invalid escape sequence", DeprecationWarning)
        warnings.filterwarnings("ignore", r".*invalid escape sequence", SyntaxWarning)
        import pypose  # noqa: F401
except ImportError as e:
    raise ImportError(
        'piazzi.pypose needs PyPose, an optional dependency: install it with pip install "piazzi[pypose]"'
    ) from e

from ._least_squares import least_squares

__all__ = ["least_squares"]
