SciPy backend
=============

.. module:: piazzi.scipy

Wrappers that run SciPy's solvers on PyTorch functions, with the calling conventions of
:func:`piazzi.least_squares` and :func:`piazzi.minimize`, so you can compare backends on the
same problem. Requires the ``piazzi[scipy]`` extra; ``import piazzi`` does not load it. See the
:doc:`backends guide <../guide/backends>`.

.. autofunction:: least_squares
.. autofunction:: minimize
.. autofunction:: t2n
.. autofunction:: n2t
