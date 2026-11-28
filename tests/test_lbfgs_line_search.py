"""The line searches of :func:`piazzi.lbfgs`: Armijo backtracking and strong Wolfe.

Ported from Ceres Solver (``internal/ceres``, master), whose line-search minimiser offers the same
two searches (``ARMIJO`` and ``WOLFE``) with L-BFGS:

- ``gradient_problem_solver_test.cc``: ``SolvesRosenbrockWithDefaultOptions`` (start at
  ``(-1.2, 0)``, converged to ``1e-9``).
- ``line_search_minimizer_test.cc``: ``FinalCostIsZero`` (``x**2`` from ``2``).

The SE3 registration of ``test_manifolds.py`` also runs with both searches.

Compared with upstream: ``torch.optim.LBFGS(line_search_fn="strong_wolfe")``, whose search piazzi
ports, so both must reach the same minimiser in a comparable number of evaluations.

Differences from Ceres: Ceres' Wolfe search is Nocedal and Wright's with its own interpolation
safeguards, and its termination test is a gradient tolerance; piazzi stops on a step below
``step_tol`` or a failed search, so the tolerances here are those it can reach.

The Ceres tests are under this licence:

    Ceres Solver - A fast non-linear least squares minimizer
    Copyright 2023 Google Inc. All rights reserved.
    http://ceres-solver.org/

    Redistribution and use in source and binary forms, with or without modification, are
    permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this list of
      conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of
      conditions and the following disclaimer in the documentation and/or other materials
      provided with the distribution.
    * Neither the name of Google Inc. nor the names of its contributors may be used to endorse
      or promote products derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS
    OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
    MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
    COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
    EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
    SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
    HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
    TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import math

import pytest
import torch
from problems import F64, SCALAR_PROBLEMS, rosen

import piazzi

SEARCHES = ["armijo", "strong_wolfe"]
OPTIONS = {"max_iter": 500, "step_tol": 1e-12}


def ceres_rosenbrock(x):
    return (1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2


@pytest.mark.parametrize("line_search", SEARCHES)
def test_ceres_solves_rosenbrock(line_search):
    res = piazzi.lbfgs(ceres_rosenbrock, torch.tensor([-1.2, 0.0], dtype=F64), line_search=line_search, **OPTIONS)
    assert res.converged, res.message
    assert torch.allclose(res.x, torch.ones(2, dtype=F64), atol=1e-9, rtol=0)


@pytest.mark.parametrize("line_search", SEARCHES)
def test_ceres_final_cost_is_zero(line_search):
    res = piazzi.lbfgs(lambda x: (x**2).sum(), torch.tensor([2.0], dtype=F64), line_search=line_search)
    assert res.converged, res.message
    assert res.fun.item() <= torch.finfo(F64).eps


@pytest.mark.parametrize("line_search", SEARCHES)
def test_scalar_problems(line_search):
    for problem in SCALAR_PROBLEMS:
        res = piazzi.lbfgs(problem.fun, problem.x0, line_search=line_search, **OPTIONS)
        assert res.converged, (problem.name, res.message)
        assert torch.allclose(res.x, problem.x_opt, atol=1e-5), (problem.name, res.x)


@pytest.mark.parametrize(("c1", "c2"), [(1e-4, 1.0), (0.5, 0.1), (0.0, 0.9)])
def test_strong_wolfe_rejects_inconsistent_constants(c1, c2):
    with pytest.raises(ValueError, match="c1 < c2"):
        piazzi.lbfgs(rosen, torch.zeros(2, dtype=F64), line_search="strong_wolfe", c1=c1, c2=c2)


def test_unknown_line_search():
    with pytest.raises(ValueError, match="line_search"):
        piazzi.lbfgs(rosen, torch.zeros(2, dtype=F64), line_search="exact")


def test_strong_wolfe_counts_every_evaluation_as_value_and_gradient():
    calls = 0

    def fun(x):
        nonlocal calls
        calls += 1
        return rosen(x)

    res = piazzi.lbfgs(fun, torch.tensor([-1.2, 1.0], dtype=F64), line_search="strong_wolfe", **OPTIONS)
    assert res.n_fun == 0
    assert res.n_fun_grad == calls
    seen = []
    piazzi.lbfgs(
        rosen,
        torch.tensor([-1.2, 1.0], dtype=F64),
        line_search="strong_wolfe",
        callback=lambda info: seen.append((info.n_fun_grad, info.n_fun)) and False,
        max_iter=5,
    )
    assert all(n_f == 0 for _, n_f in seen) and [n for n, _ in seen] == sorted(n for n, _ in seen)


def test_strong_wolfe_ignores_value_fun():
    def value_fun(x):
        raise AssertionError("strong Wolfe needs gradients and must not call value_fun")

    res = piazzi.lbfgs(
        rosen, torch.tensor([-1.2, 1.0], dtype=F64), line_search="strong_wolfe", value_fun=value_fun, **OPTIONS
    )
    assert res.converged


def test_strong_wolfe_accepted_steps_satisfy_the_wolfe_conditions():
    # Every accepted step must satisfy sufficient decrease and, unless the search ran out of
    # budget, the strong curvature condition; so s^T y > 0 at every iteration.
    c1 = 1e-4
    x_prev, f_prev, g_prev = None, None, None
    checked = 0

    def fun(x):
        return rosen(x)

    def cb(info):
        nonlocal x_prev, f_prev, g_prev, checked
        x = info.x.clone().requires_grad_(True)
        f = rosen(x)
        (g,) = torch.autograd.grad(f, x)
        if x_prev is not None and info.accepted:
            s = x.detach() - x_prev
            assert f.item() <= f_prev + c1 * torch.dot(g_prev, s).item() + 1e-12
            assert torch.dot(s, g - g_prev).item() > 0
            checked += 1
        x_prev, f_prev, g_prev = x.detach(), f.item(), g
        return False

    piazzi.lbfgs(fun, torch.tensor([-1.2, 1.0], dtype=F64), line_search="strong_wolfe", callback=cb, **OPTIONS)
    assert checked > 10


def test_strong_wolfe_first_trial_failing_armijo_is_bracketed():
    # A steep, narrow valley: the unit step overshoots, so the very first trial fails sufficient
    # decrease and the search must bracket and zoom from the starting point.
    def fun(x):
        return 50.0 * (x**2).sum() + torch.cos(3 * x).sum()

    res = piazzi.lbfgs(fun, torch.tensor([1.0, -0.7], dtype=F64), line_search="strong_wolfe", lr=1.0, **OPTIONS)
    assert res.converged, res.message
    assert torch.allclose(res.x, torch.zeros(2, dtype=F64), atol=1e-8)


def test_strong_wolfe_never_accepts_a_zero_step():
    # With a budget of one extra trial the search often ends back at its starting point; that is a
    # failed search, never an accepted step of length zero.
    flat_steps = []
    res = piazzi.lbfgs(
        rosen,
        torch.tensor([-1.2, 1.0], dtype=F64),
        line_search="strong_wolfe",
        max_backtracks=1,
        callback=lambda info: flat_steps.append(info.accepted and info.fun_start.item() == info.fun.item()) and False,
        max_iter=60,
    )
    assert not any(flat_steps)
    assert res.fun.item() < rosen(torch.tensor([-1.2, 1.0], dtype=F64)).item()


def test_strong_wolfe_respects_max_step():
    xs = []
    piazzi.lbfgs(
        lambda x: 0.5 * ((x - 100.0) ** 2).sum(),
        torch.zeros(3, dtype=F64),
        line_search="strong_wolfe",
        max_step=0.5,
        callback=lambda info: xs.append(info.x.clone()) and False,
        max_iter=20,
    )
    steps = [(b - a).abs().max().item() for a, b in zip([torch.zeros(3, dtype=F64), *xs], xs, strict=False)]
    assert max(steps) <= 0.5 * (1 + 1e-12)


def test_strong_wolfe_non_finite_trial_never_passes():
    def fun(x):
        # Finite only on x < 0.5: the first trial from 0 lands outside, as nan or inf.
        return torch.where(x < 0.5, (x - 0.3) ** 2, torch.full_like(x, float("nan"))).sum()

    res = piazzi.lbfgs(fun, torch.tensor([0.0], dtype=F64), line_search="strong_wolfe", **OPTIONS)
    assert math.isfinite(res.fun.item())
    assert abs(res.x.item() - 0.3) < 1e-6


@pytest.mark.parametrize("scale", [1e-9, 1.0, 1e9])
def test_strong_wolfe_is_scale_invariant(scale):
    x0 = torch.tensor([-1.2, 1.0], dtype=F64)
    ref = piazzi.lbfgs(rosen, x0, line_search="strong_wolfe", **OPTIONS)
    res = piazzi.lbfgs(lambda x: scale * rosen(x), x0, line_search="strong_wolfe", **OPTIONS)
    assert res.converged, res.message
    assert torch.allclose(res.x, ref.x, atol=1e-6)


def test_strong_wolfe_float32():
    x0 = torch.tensor([-1.2, 1.0])
    res = piazzi.lbfgs(rosen, x0, line_search="strong_wolfe", max_iter=500)
    assert res.converged, res.message
    assert torch.allclose(res.x, torch.ones(2), atol=1e-3)


# ---- comparison with upstream torch.optim.LBFGS ----


def upstream(fun, x0, **options):
    x = x0.clone().requires_grad_(True)
    opt = torch.optim.LBFGS([x], line_search_fn="strong_wolfe", tolerance_grad=1e-12, tolerance_change=1e-14, **options)
    evals = 0

    def closure():
        nonlocal evals
        evals += 1
        opt.zero_grad()
        loss = fun(x)
        loss.backward()
        return loss

    opt.step(closure)
    return x.detach(), evals


@pytest.mark.parametrize("problem", SCALAR_PROBLEMS, ids=[p.name for p in SCALAR_PROBLEMS])
def test_strong_wolfe_matches_upstream_minimiser_and_evaluation_budget(problem):
    x_ref, evals_ref = upstream(problem.fun, problem.x0, max_iter=500, max_eval=5000)
    res = piazzi.lbfgs(problem.fun, problem.x0, line_search="strong_wolfe", **OPTIONS)
    assert res.converged, res.message
    assert torch.allclose(res.x, x_ref, atol=1e-4), (res.x, x_ref)
    # Same search, same memory, so the same order of cost (piazzi stops earlier on step_tol).
    assert res.n_fun_grad <= 2 * evals_ref + 10


def test_strong_wolfe_and_armijo_both_converge_on_rosenbrock():
    x0 = torch.tensor([-1.2, 1.0], dtype=F64)
    for line_search in SEARCHES:
        assert piazzi.lbfgs(rosen, x0, line_search=line_search, **OPTIONS).converged


def test_strong_wolfe_budget_exhausted_while_expanding_keeps_the_better_point():
    # One extra trial: the first trial (t=1) satisfies Armijo but not curvature; the expansion
    # trial goes back up. The search must return the first trial, not the worse last one.
    from piazzi._line_search import wolfe_search

    def f(t):
        return -0.2 if t == 1.0 else -0.05

    def eval_full(xv):
        t = xv.item()
        # Slope stays negative everywhere, so no bracket is found, only an uphill expansion.
        return torch.tensor(f(t), dtype=F64), None, torch.tensor([-0.5], dtype=F64)

    x = torch.zeros(1, dtype=F64)
    out = wolfe_search(
        x,
        0.0,
        torch.tensor([-1.0], dtype=F64),
        None,
        torch.tensor([1.0], dtype=F64),
        1.0,
        100.0,
        -1.0,
        False,
        None,
        1e-4,
        0.1,
        1,
        eval_full,
        lambda _m, x, delta: x + delta,
    )
    x_new, f_new, _, _, t, _, _, ok = out
    assert ok and t == 1.0 and f_new.item() == -0.2 and x_new.item() == 1.0
