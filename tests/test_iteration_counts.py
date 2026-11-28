"""Evaluation counts in ``IterationInfo``, and what ``lbfgs``'s line search evaluates."""

import itertools
import weakref

import pytest
import torch
from problems import F64, fun_rosenbrock, rosen

import piazzi

X0 = torch.tensor([-1.2, 1.0], dtype=F64)

RUNS = {
    "lbfgs": (rosen, lambda f, cb: piazzi.minimize(f, X0, callback=cb, options={"max_iter": 200})),
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


@pytest.mark.parametrize("x0", [X0, X0.repeat(5)], ids=["2d", "10d"])
def test_lbfgs_backward_only_at_accepted_points(x0):
    """The line search keeps each trial's graph and runs the backward pass only on acceptance."""
    points, backwards = [], []

    def fun(x):
        points.append(x.detach().clone())
        if x.requires_grad:
            x.register_hook(backwards.append)  # runs once per backward pass through x
        return rosen(x)

    res = piazzi.minimize(fun, x0, options={"max_iter": 500})
    assert res.converged and res.n_fun > 0  # the line search did backtrack
    assert len(backwards) == res.n_fun_grad == res.n_iter + 1  # x0, then one per accepted step
    assert len(points) == res.n_fun_grad + res.n_fun
    assert not any(torch.equal(a, b) for a, b in itertools.pairwise(points))


def test_lbfgs_value_fun_counts():
    """With a value_fun, backtracking uses it; an accepted backtracked point costs one more fun call."""
    calls = {"fun": 0, "value": 0}

    def fun(x):
        calls["fun"] += 1
        return rosen(x)

    def value_fun(x):
        calls["value"] += 1
        return rosen(x)

    res = piazzi.minimize(fun, X0, options={"max_iter": 500, "value_fun": value_fun})
    ref = piazzi.minimize(rosen, X0, options={"max_iter": 500})
    assert torch.equal(res.x, ref.x)
    # A rejected first trial is a call to fun counted as value-only, since its backward pass
    # never runs; backtracking calls value_fun, and an accepted backtracked point calls fun.
    assert calls["value"] > 0
    assert calls["fun"] + calls["value"] == res.n_fun_grad + res.n_fun


def test_lbfgs_frees_rejected_graphs():
    """No earlier trial's graph is still alive when the next evaluation starts."""
    losses, alive_at_call = [], []

    def fun(x):
        alive_at_call.append(sum(ref() is not None for ref in losses))
        loss = rosen(x)
        losses.append(weakref.ref(loss))  # the graph lives as long as the loss tensor
        return loss

    res = piazzi.minimize(fun, X0, options={"max_iter": 500})
    assert res.n_fun > 0 and max(alive_at_call) == 0
