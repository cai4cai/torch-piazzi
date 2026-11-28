pytorch-minimize backend
========================

.. module:: piazzi.torchmin

A wrapper that runs the solvers of `pytorch-minimize <https://github.com/rfeinman/pytorch-minimize>`_
(``torchmin``) on PyTorch functions, with the calling convention of :func:`piazzi.minimize`, so you
can compare backends on the same problem. Requires the ``piazzi[torchmin]`` extra; ``import piazzi``
does not load it, and importing this module registers ``"torchmin-<method>"`` in
:data:`piazzi.MINIMIZE_METHODS`. See the :doc:`backends guide <../guide/backends>`.

.. autofunction:: minimize

.. data:: METHODS

   The method names of ``torchmin.minimize``: ``"bfgs"``, ``"l-bfgs"``, ``"cg"``, ``"newton-cg"``,
   ``"newton-exact"``, ``"dogleg"``, ``"trust-ncg"``, ``"trust-exact"`` and ``"trust-krylov"``.
