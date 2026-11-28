PyPose backend
==============

.. module:: piazzi.pypose

A wrapper that runs PyPose's ``pypose.optim.GaussNewton`` and
``pypose.optim.LevenbergMarquardt`` on a PyTorch residual function, with the calling
convention of :func:`piazzi.least_squares`, so you can compare them with the native solvers.
Requires the ``piazzi[pypose]`` extra; ``import piazzi`` does not load it. See the
:doc:`backends guide <../guide/backends>`.

.. autofunction:: least_squares
