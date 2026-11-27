"""Evaluation counts in ``IterationInfo``."""

import pytest
import torch
from problems import F64, fun_rosenbrock, rosen

import piazzi

X0 = torch.tensor([-1.2, 1.0], dtype=F64)

RUNS = {
    "lbfgs_armijo": (rosen, lambda f, cb: piazzi.minimize(f, X0, callback=cb, options={"max_iter": 200})),
    "torch_lbfgs": (
        rosen,
        lambda f, cb: piazzi.minimize(
            f, X0, method=torch.optim.LBFGS, callback=cb, options={"line_search_fn": "strong_wolfe"}
        ),
    ),
    "torch_adam": (rosen, lambda f, cb: piazzi.minimize(f, X0, method=torch.optim.Adam, callback=cb)),
    "gn": (fun_rosenbrock, lambda f, cb: piazzi.least_squares(f, X0, method="gn", callback=cb, step_tol=1e-10)),
    "lm": (fun_rosenbrock, lambda f, cb: piazzi.least_squares(f, X0, method="lm", callback=cb, step_tol=1e-10)),
}


@pytest.mark.parametrize("name", RUNS)
def test_counts_match_calls_so_far(name):
    fun, run = RUNS[name]
    calls = 0
    seen = []

    def counted(x):
        nonlocal calls
        calls += 1
        return fun(x)

    def callback(info):
        seen.append((calls, info.n_fun_grad + info.n_fun + info.n_system, info))

    res = run(counted, callback)
    assert len(seen) == res.n_iter > 1
    # Every call to fun is one evaluation (a Jacobian is one vectorised call), counted by kind.
    assert all(expected == counted_so_far for expected, counted_so_far, _ in seen)
    last = seen[-1][2]
    assert (last.n_fun_grad, last.n_fun, last.n_system) == (res.n_fun_grad, res.n_fun, res.n_system)
    totals = [total for _, total, _ in seen]
    assert totals == sorted(totals)
