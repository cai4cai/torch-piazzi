High-level interfaces
=====================

.. currentmodule:: piazzi

The SciPy-style entry points, and where to start. They take the objective in its most natural
form (residuals, or a scalar), choose a method by name and pass that method's settings
through: :func:`least_squares` forwards its other keyword arguments, and :func:`minimize`
forwards ``options``. The :doc:`method-specific solvers <methods>` they run document every
setting.

.. autofunction:: least_squares
.. autofunction:: minimize
.. autofunction:: finite_difference

.. py:data:: MINIMIZE_METHODS
   :type: dict[str, Callable[..., OptimizeResult]]

   Scalar minimisers available to :func:`minimize` by name (currently ``"lbfgs_armijo"``).
   To add one, register a callable ``method(fun, x0, *, has_aux, callback, **options)``
   under a lower-case name. :func:`minimize` passes ``manifold=`` only when one is given, so a
   method that doesn't support manifolds need not accept it.
