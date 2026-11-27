Linear solvers
==============

.. currentmodule:: piazzi

The ``linear_solver`` argument of :func:`gauss_newton` (and :func:`least_squares`): how each
damped normal-equations system is solved. See the :doc:`sparse problems guide
<../guide/sparse>`.

.. autofunction:: dense_solver
.. autofunction:: cg_solver

.. py:data:: LinearSolver

   Signature of a linear solver for :func:`gauss_newton`: ``solver(H, g, damping) -> d``
   solves ``(H + diag(damping)) d = -g``, with ``H`` of shape ``(n, n)`` (dense, or sparse COO
   or CSR) and ``g``, ``damping`` and ``d`` of shape ``(n,)``.
