Method-specific solvers
=======================

.. currentmodule:: piazzi

The algorithms behind the :doc:`high-level interfaces <interfaces>`, with every setting
exposed. :func:`least_squares` runs :func:`gauss_newton` and :func:`minimize` runs
:func:`lbfgs_armijo`, passing their extra arguments through. Call these directly when you
build the normal equations yourself (for example a sparse ``J^T J``), or to use one method
without going through the interface.

.. autofunction:: gauss_newton
.. autofunction:: levenberg_marquardt
.. autofunction:: lbfgs_armijo
.. autofunction:: least_squares_system
