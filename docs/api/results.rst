Results and callbacks
=====================

.. currentmodule:: piazzi

Every solver and wrapper returns an :class:`OptimizeResult`, and passes an
:class:`IterationInfo` to ``callback`` after every iteration. :func:`print_progress` is a
ready-made callback that prints them.

.. autoclass:: OptimizeResult
   :no-members:

.. autoclass:: IterationInfo
   :no-members:

.. autofunction:: print_progress
