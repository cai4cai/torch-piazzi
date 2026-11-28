"""piazzi: deterministic PyTorch solvers for nonlinear fitting problems.

Named after Giuseppe Piazzi, who discovered Ceres in 1801; Gauss recovered it from
Piazzi's observations with the method of least squares.
"""

from ._common import IterationInfo, OptimizeResult
from ._dogleg import dogleg
from ._finite_difference import finite_difference
from ._gauss_newton import gauss_newton, levenberg_marquardt
from ._lbfgs import lbfgs_armijo
from ._least_squares import least_squares, least_squares_system
from ._linear_solvers import LinearSolver, cg_solver, dense_solver
from ._losses import LOSSES, robust_cost
from ._manifolds import SE3, SO3, Euclidean, Manifold, Product, Subset, se3_exp, se3_log, so3_exp, so3_hat, so3_log
from ._minimize import MINIMIZE_METHODS, minimize
from ._progress import print_progress

try:
    from ._version import __version__
except ImportError:
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("piazzi")
    except PackageNotFoundError:
        __version__ = "0.0.0.dev0+unknown"

__all__ = [
    "LOSSES",
    "MINIMIZE_METHODS",
    "SE3",
    "SO3",
    "Euclidean",
    "IterationInfo",
    "LinearSolver",
    "Manifold",
    "OptimizeResult",
    "Product",
    "Subset",
    "cg_solver",
    "dense_solver",
    "dogleg",
    "finite_difference",
    "gauss_newton",
    "lbfgs_armijo",
    "least_squares",
    "least_squares_system",
    "levenberg_marquardt",
    "minimize",
    "print_progress",
    "robust_cost",
    "se3_exp",
    "se3_log",
    "so3_exp",
    "so3_hat",
    "so3_log",
]
