"""A wrapper around pytorch-minimize (``torchmin``), for comparing it with piazzi's native solvers.

pytorch-minimize is an optional dependency, so ``import piazzi`` does not load this module. Import it
explicitly, after installing it (``pip install "piazzi[torchmin]"``)::

    import piazzi.torchmin

    res = piazzi.torchmin.minimize(fun, x0, method="trust-krylov")

Importing it also registers its methods in :data:`piazzi.MINIMIZE_METHODS` under the prefixed names
``"torchmin-bfgs"``, ``"torchmin-l-bfgs"``, ..., so ``piazzi.minimize(fun, x0, method="torchmin-bfgs")``
works too (that call imports this module itself).
"""

import warnings

try:
    with warnings.catch_warnings():
        # torchmin decorates a class with torch.jit.script when it is imported. That warns that it is deprecated
        # and, on Python 3.14+, that it "is not supported in Python 3.14+". Remove this filter once a torchmin
        # release no longer does (upstream: https://github.com/rfeinman/pytorch-minimize, torchmin/function.py).
        warnings.filterwarnings(
            "ignore",
            r"`torch\.jit\.script` is (deprecated|not supported in Python 3\.14\+)",
            FutureWarning,
            r"torch\.jit",
        )
        import torchmin  # noqa: F401
except ImportError as e:
    raise ImportError(
        'piazzi.torchmin needs pytorch-minimize, an optional dependency: install it with pip install "piazzi[torchmin]"'
    ) from e

from .._minimize import MINIMIZE_METHODS
from ._minimize import METHODS, minimize

__all__ = ["METHODS", "minimize"]


def _register(name: str):
    def method(fun, x0, *, has_aux=False, manifold=None, callback=None, **options):
        return minimize(fun, x0, name, has_aux=has_aux, manifold=manifold, callback=callback, options=options)

    method.__name__ = method.__qualname__ = f"torchmin_{name.replace('-', '_')}"
    method.__doc__ = f"pytorch-minimize's ``{name}`` through :func:`piazzi.torchmin.minimize`."
    return method


for _name in METHODS:
    MINIMIZE_METHODS[f"torchmin-{_name}"] = _register(_name)
