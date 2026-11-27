Manifolds and Lie groups
========================

.. currentmodule:: piazzi

Pass a manifold as ``manifold=`` to keep the parameters on a constrained set, such as
rotations or rigid transforms. See the :doc:`manifolds guide <../guide/manifolds>`.

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
