Robust losses
=============

.. currentmodule:: piazzi

The ``loss`` and ``f_scale`` arguments of :func:`least_squares` and
:func:`least_squares_system`, and the same cost as a differentiable function.

.. autofunction:: robust_cost

.. py:data:: LOSSES
   :type: dict[str, Callable]

   The built-in robust losses by name: ``"linear"``, ``"huber"``, ``"soft_l1"``, ``"cauchy"``
   and ``"arctan"``, as in :func:`scipy.optimize.least_squares`. Each maps ``z`` to
   ``(rho(z), rho'(z), rho''(z))`` in closed form, so the normal equations need no autograd.
