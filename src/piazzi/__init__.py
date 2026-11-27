"""piazzi: deterministic PyTorch solvers for small nonlinear fitting problems.

Named after Giuseppe Piazzi, who discovered Ceres in 1801; Gauss recovered it from
Piazzi's observations with the method of least squares.
"""

from ._common import IterationInfo, OptimizeResult
from ._minimize import MINIMIZE_METHODS, minimize
from .gauss_newton import gauss_newton, least_squares, least_squares_system, levenberg_marquardt
from .lbfgs import lbfgs_armijo
from .losses import LOSSES, robust_cost

__version__ = "0.1.0.dev0"

__all__ = [
    "LOSSES",
    "MINIMIZE_METHODS",
    "IterationInfo",
    "OptimizeResult",
    "gauss_newton",
    "lbfgs_armijo",
    "least_squares",
    "least_squares_system",
    "levenberg_marquardt",
    "minimize",
    "robust_cost",
]
