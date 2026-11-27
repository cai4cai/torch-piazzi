API reference
=============

.. currentmodule:: piazzi

Everything below is importable from ``piazzi``, except the optional SciPy backend in
``piazzi.scipy``. Start with the high-level interfaces; the method-specific solvers are the
algorithms they run, for when you need a method's full control.

:doc:`High-level interfaces <interfaces>`
-----------------------------------------

SciPy-style entry points: give residuals or a scalar objective, choose the method by name.

.. autosummary::
   :nosignatures:

   least_squares
   minimize

The methods :func:`minimize` knows by name are in :data:`MINIMIZE_METHODS`.

:doc:`Method-specific solvers <methods>`
----------------------------------------

The algorithms behind the interfaces, with every setting exposed. Use them directly when you
build the normal equations yourself.

.. autosummary::
   :nosignatures:

   gauss_newton
   levenberg_marquardt
   lbfgs_armijo
   least_squares_system

:doc:`Results and callbacks <results>`
--------------------------------------

.. autosummary::
   :nosignatures:

   OptimizeResult
   IterationInfo
   print_progress

:doc:`Robust losses <losses>`
-----------------------------

.. autosummary::
   :nosignatures:

   robust_cost

The built-in losses by name are in :data:`LOSSES`.

:doc:`Manifolds and Lie groups <manifolds>`
-------------------------------------------

.. autosummary::
   :nosignatures:

   Manifold
   Euclidean
   Subset
   SO3
   SE3
   Product
   so3_exp
   so3_log
   so3_hat
   se3_exp
   se3_log

:doc:`Linear solvers <linear-solvers>`
--------------------------------------

.. autosummary::
   :nosignatures:

   dense_solver
   cg_solver

Custom solvers follow the :data:`LinearSolver` signature.

:doc:`SciPy backend <scipy>`
----------------------------

Requires the ``piazzi[scipy]`` extra.

.. currentmodule:: piazzi.scipy

.. autosummary::
   :nosignatures:

   least_squares
   minimize
   t2n
   n2t

.. toctree::
   :hidden:

   interfaces
   methods
   results
   losses
   manifolds
   linear-solvers
   scipy
