API reference
=============

.. currentmodule:: piazzi

High-level interfaces
---------------------

The SciPy-style entry points: residuals or a scalar objective, with the method chosen by name.

.. autofunction:: least_squares
.. autofunction:: minimize

.. py:data:: MINIMIZE_METHODS
   :type: dict[str, Callable[..., OptimizeResult]]

   Scalar minimisers available to :func:`minimize` by name (currently ``"lbfgs_armijo"``).
   To add one, register a callable ``method(fun, x0, *, has_aux, callback, **options)``
   under a lower-case name. :func:`minimize` passes ``manifold=`` only when one is given, so a
   method that doesn't support manifolds need not accept it.

Method-specific solvers
-----------------------

The algorithms behind the high-level interfaces, with every setting exposed.

.. autofunction:: gauss_newton
.. autofunction:: levenberg_marquardt
.. autofunction:: lbfgs_armijo
.. autofunction:: least_squares_system

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
   :type: dict[str, Callable]

   The built-in robust losses by name: ``"linear"``, ``"huber"``, ``"soft_l1"``, ``"cauchy"``
   and ``"arctan"``, as in :func:`scipy.optimize.least_squares`. Each maps ``z`` to
   ``(rho(z), rho'(z), rho''(z))`` in closed form, so the normal equations need no autograd.

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

   Signature of a linear solver for :func:`gauss_newton`: ``solver(H, g, damping) -> d``
   solves ``(H + diag(damping)) d = -g``, with ``H`` of shape ``(n, n)`` (dense, or sparse COO
   or CSR) and ``g``, ``damping`` and ``d`` of shape ``(n,)``.

SciPy backend
-------------

.. module:: piazzi.scipy

Requires the ``piazzi[scipy]`` extra.

.. autofunction:: least_squares
.. autofunction:: minimize
.. autofunction:: t2n
.. autofunction:: n2t
