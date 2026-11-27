API reference
=============

.. currentmodule:: piazzi

Solvers
-------

.. autofunction:: least_squares
.. autofunction:: minimize
.. autofunction:: gauss_newton
.. autofunction:: levenberg_marquardt
.. autofunction:: lbfgs_armijo
.. autofunction:: least_squares_system

.. py:data:: MINIMIZE_METHODS
   :type: dict[str, Callable[..., OptimizeResult]]

   Scalar minimisers available to :func:`minimize` by name (currently ``"lbfgs_armijo"``).
   To add one, register a callable ``method(fun, x0, *, has_aux, callback, **options)``
   under a lower-case name. :func:`minimize` passes ``manifold=`` only when one is given, so a
   method that doesn't support manifolds need not accept it.

Results and callbacks
---------------------

.. autoclass:: OptimizeResult
   :no-members:

.. autoclass:: IterationInfo
   :no-members:

Robust losses
-------------

.. autofunction:: robust_cost

.. py:data:: LOSSES

   The built-in robust losses by name: ``"linear"``, ``"huber"``, ``"soft_l1"``, ``"cauchy"``
   and ``"arctan"``, as in :func:`scipy.optimize.least_squares`.

Manifolds
---------

.. autoclass:: Manifold
.. autoclass:: Euclidean
.. autoclass:: Subset
.. autoclass:: SO3
.. autoclass:: SE3
.. autoclass:: Product

Lie-group maps
--------------

.. autofunction:: so3_exp
.. autofunction:: so3_log
.. autofunction:: so3_hat
.. autofunction:: se3_exp
.. autofunction:: se3_log

Linear solvers
--------------

.. autofunction:: dense_solver
.. autofunction:: cg_solver

.. py:data:: LinearSolver

   The signature of a linear solver for :func:`gauss_newton`: a callable
   ``solver(H, g, damping) -> d`` that solves ``(H + diag(damping)) d = -g``. ``H`` may be
   dense or sparse (COO or CSR).

SciPy backend
-------------

.. currentmodule:: piazzi.scipy

Requires the ``piazzi[scipy]`` extra.

.. autofunction:: least_squares
.. autofunction:: minimize
.. autofunction:: t2n
.. autofunction:: n2t
