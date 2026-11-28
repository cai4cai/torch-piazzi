# Contributing

Development uses [uv](https://docs.astral.sh/uv/):

```bash
uv sync --group dev --group notebooks
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest -q
uv run python scripts/execute_notebooks.py   # when the library or a notebook changed
```

To build this documentation:

```bash
uv sync --group docs
uv run sphinx-build -W --keep-going docs docs/_build/html
```

## Supported versions

piazzi requires Python 3.11 or later and PyTorch 2.15 or later; the optional extras need SciPy 1.11 (`piazzi[scipy]`; the `least_squares` callbacks need SciPy 1.16) and torchsparsegradutils 0.2.7 (`piazzi[sparse]`). CI tests the oldest and newest Pythons that PyTorch supports (3.11 and 3.15) with the latest dependencies, plus Python 3.11 with the lowest version of every direct dependency that `pyproject.toml` allows (`uv pip install --resolution lowest-direct`). Raise a floor in `pyproject.toml` when code needs a newer version, and the lowest-versions job then tests it.

## Workarounds for upstream bugs

Each workaround says, in a comment next to it, which upstream bug it avoids and when to remove it, and the test suite fails if it is removed while the bug is still there. When the PyTorch floor is raised, check this list and remove the workarounds that are no longer needed.

| Workaround | Upstream bug | Remove when | Test that fails without it |
|---|---|---|---|
| `quiet_jvp_warnings` in `src/piazzi/_common.py` filters a `FutureWarning` | Forward-mode AD loads PyTorch's JVP decompositions, which call `torch.jit.script` (PyTorch 2.15 and earlier). It warns that `torch.jit.script` is deprecated and, on Python 3.14+, that it "is not supported in Python 3.14+". Fixed by [pytorch/pytorch#203621](https://github.com/pytorch/pytorch/pull/203621), after 2.15.0 | The PyTorch floor includes that fix, and neither warning appears on the newest Python in CI | Any test that runs `jacfwd` (warnings are errors), on every Python |
| `keepdim=True` in `_exp_coeffs` in `src/piazzi/_manifolds.py` | In forward-mode AD, a 0-dim float32 tensor combined with a Python scalar gets a float64 tangent, which then breaks matmuls with float32 tensors (PyTorch 2.15) | A PyTorch release keeps the float32 tangent: remove `keepdim` and run the tests | `test_exp_derivative_at_zero_is_finite_and_exact[fwd]` and `test_least_squares_se3_float32` in `tests/test_manifolds.py` |
| `quiet_torchmin_warnings` and the import filter in `src/piazzi/torchmin/` | pytorch-minimize 0.1.0 decorates a class with `torch.jit.script` when imported (a `FutureWarning`: deprecated, and on Python 3.14+ not supported), and its exact trust-region solver calls the deprecated `torch.triangular_solve` | A torchmin release no longer does either | Any test of `piazzi.torchmin` (warnings are errors); `test_problems[*-trust-exact]` for `triangular_solve` |
| `loss.detach().clone()` and the gradient hook in `wrapped` in `src/piazzi/torchmin/_minimize.py` | torchmin updates the value it was given in place (`f[...] = f_new`), and copies gradients in place, which fails on an expanded view such as the gradient of a sum | torchmin copies values and accepts non-contiguous gradients | `test_callback_info_and_aux` (`fun_start` is corrupted without the clone); `test_problems[square_1d-*]` (the hook) |

## Conventions

The conventions contributors (human or AI) are expected to follow, from evaluation counting to float32 tolerances and notebook outputs, are in [AGENTS.md](https://github.com/cai4cai/torch-piazzi/blob/main/AGENTS.md). Planned work is tracked in the [GitHub issues](https://github.com/cai4cai/torch-piazzi/issues).
