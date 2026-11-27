"""Test problems ported to PyTorch from SciPy's test suite.

Least-squares problems come from ``scipy/optimize/tests/test_least_squares.py`` and scalar
problems from ``scipy/optimize/tests/test_optimize.py``. They are shared by the ported tests
and by the differential tests that run each problem through both piazzi and
``piazzi.scipy``.

Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are
permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of
   conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of
   conditions and the following disclaimer in the documentation and/or other materials
   provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors may be used
   to endorse or promote products derived from this software without specific prior
   written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS
OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from dataclasses import dataclass, field

import torch

F64 = torch.float64


# Least-squares residual functions (test_least_squares.py).


def fun_trivial(x, a=0.0):
    return (x - a) ** 2 + 5.0


def fun_2d_trivial(x):
    return torch.stack([x[0], x[1]])


def fun_rosenbrock(x):
    return torch.stack([10 * (x[1] - x[0] ** 2), 1 - x[0]])


def jac_rosenbrock(x):
    return torch.stack([torch.stack([-20 * x[0], torch.full_like(x[0], 10.0)]), torch.tensor([-1.0, 0.0]).to(x)])


def fun_rosenbrock_cropped(x):
    """One residual, two parameters (m < n): SciPy's ``method="lm"`` rejects it."""
    return fun_rosenbrock(x)[:1]


def fun_bvp(x):
    """Discrete boundary value problem (SciPy gh-5556)."""
    n = int(round(x.numel() ** 0.5))
    x = x.reshape(n, n)
    u = torch.nn.functional.pad(x, (1, 1, 1, 1))
    y = u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:] - 4 * x + x**3
    return y.reshape(-1)


def fun_broyden_tridiagonal(x):
    """``BroydenTridiagonal.fun``, used here without SciPy's bounds."""
    f = (3 - x) * x + 1
    f = f - torch.nn.functional.pad(x[:-1], (1, 0))
    f = f - 2 * torch.nn.functional.pad(x[1:], (0, 1))
    return f


@dataclass
class LsqProblem:
    name: str
    fun: object
    x0: torch.Tensor
    x_opt: torch.Tensor | None = None  # known minimiser, if unique
    max_cost: float | None = None  # otherwise, the cost the solver must reach
    tol: dict = field(default_factory=dict)  # extra solver options


LSQ_PROBLEMS = [
    LsqProblem("trivial", fun_trivial, torch.tensor([2.0], dtype=F64), x_opt=torch.zeros(1, dtype=F64)),
    LsqProblem("2d_trivial", fun_2d_trivial, torch.tensor([3.0, -4.0], dtype=F64), x_opt=torch.zeros(2, dtype=F64)),
    LsqProblem("rosenbrock", fun_rosenbrock, torch.tensor([-2.0, 1.0], dtype=F64), x_opt=torch.ones(2, dtype=F64)),
    LsqProblem("rosenbrock_cropped", fun_rosenbrock_cropped, torch.tensor([-2.0, 1.0], dtype=F64), max_cost=1e-14),
    LsqProblem("bvp", fun_bvp, torch.ones(100, dtype=F64), max_cost=0.5),
    LsqProblem("broyden_tridiagonal", fun_broyden_tridiagonal, -torch.ones(20, dtype=F64), max_cost=1e-14),
]


# Scalar functions (test_optimize.py).


def rosen(x):
    """``scipy.optimize.rosen``: the n-dimensional Rosenbrock function."""
    return (100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2).sum()


# CheckOptimize: the constrained entropy maximisation problem from the machine translation
# example of Berger et al., Computational Linguistics 22(1):39-72, 1996.
ENTROPY_F = torch.tensor([[1, 1, 1], [1, 1, 0], [1, 0, 1], [1, 0, 0], [1, 0, 0]], dtype=F64)
ENTROPY_K = torch.tensor([1.0, 0.3, 0.5], dtype=F64)
ENTROPY_SOLUTION = torch.tensor([0.0, -0.524869316, 0.487525860], dtype=F64)


def entropy(x):
    log_pdot = ENTROPY_F.to(x) @ x
    return torch.logsumexp(log_pdot, 0) - ENTROPY_K.to(x) @ x


@dataclass
class ScalarProblem:
    name: str
    fun: object
    x0: torch.Tensor
    x_opt: torch.Tensor


SCALAR_PROBLEMS = [
    ScalarProblem("entropy", entropy, torch.zeros(3, dtype=F64), ENTROPY_SOLUTION),
    ScalarProblem("rosen_2d", rosen, torch.tensor([-1.2, 1.0], dtype=F64), torch.ones(2, dtype=F64)),
    ScalarProblem("rosen_10d", rosen, torch.zeros(10, dtype=F64), torch.ones(10, dtype=F64)),
    ScalarProblem("square_1d", lambda x: (x * x).sum(), torch.tensor([0.1], dtype=F64), torch.zeros(1, dtype=F64)),
]
