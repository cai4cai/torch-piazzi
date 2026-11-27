Method-specific solvers
=======================

.. currentmodule:: piazzi

The algorithms behind the :doc:`high-level interfaces <interfaces>`, with every setting
exposed. :func:`least_squares` runs :func:`gauss_newton` and passes its other keyword
arguments through. :func:`minimize` runs :func:`lbfgs_armijo` by default and passes its
``options`` to it (with a :class:`torch.optim.Optimizer` class as ``method``, it runs that
optimiser instead). Call these directly when you build the normal equations yourself (for
example a sparse ``J^T J``), or to use one method without going through the interface.

.. autofunction:: gauss_newton
.. autofunction:: levenberg_marquardt
.. autofunction:: lbfgs_armijo
.. autofunction:: least_squares_system
