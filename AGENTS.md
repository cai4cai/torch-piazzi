# AGENTS.md

Expectations for anyone changing this repository, human or AI agent. The [README](README.md) and the [documentation](https://cai4cai.github.io/torch-piazzi/) (sources in `docs/`) describe what the library does; this file describes how to work on it.

## What piazzi is for

Deterministic PyTorch solvers for nonlinear fitting problems (pose or camera registration, curve fitting, bundle adjustment) where every objective evaluation is expensive. Two consequences shape most decisions:

- **Every evaluation counts.** Never evaluate the objective, its gradient or the normal equations twice at the same point, never compute a gradient that will be thrown away, and count every evaluation by kind (`n_fun_grad`, `n_fun`, `n_system`) in both `OptimizeResult` and `IterationInfo`. A change that adds evaluations needs a reason in its commit message.
- **Results must be trustworthy.** `converged=True` means a minimiser was reached to working precision, in float32 as well as float64; `message` says why the run stopped. A solver must never accept a trial point with a non-finite value, step uphill, or report convergence at a point it has not checked.

## Layout

- `src/piazzi/`: the library. Implementation modules are private (`_lbfgs.py`, `_gauss_newton.py`, ...); the public API is what `__init__.py` exports, and the pages in `docs/api/` list it, grouped as in `docs/api/index.rst`.
- `src/piazzi/scipy/`: optional wrappers around `scipy.optimize`, with the same calling convention as the native solvers. `import piazzi` must not import SciPy.
- `tests/`: pytest suite. `problems.py` holds test problems shared between files; ported test files say where they come from and carry the original licence.
- `notebooks/`: executed examples that run locally and in Google Colab. The docs render them from their stored outputs (`docs/conf.py` copies them into `docs/examples/` at build time).
- `docs/`: the Sphinx documentation (MyST Markdown pages, the user guide in `docs/guide/`, the API reference in `docs/api/`), published to GitHub Pages from `main`.
- `scripts/execute_notebooks.py`: runs the notebooks and fails on any error, warning or stderr output.

## Before every commit

```bash
uv sync --group dev --group notebooks
uv run ruff check .
uv run ruff format --check .   # also formats notebooks and README code blocks
uv run pyright                 # type-checks src/ (standard mode)
uv run pytest -q               # tests and docstring examples; warnings are errors (filterwarnings = ["error"])
uv run python scripts/execute_notebooks.py   # when the library or a notebook changed
uv run --group docs sphinx-build -W --keep-going docs docs/_build/html   # when docs, docstrings or notebooks changed
```

CI runs the same checks on Python 3.10 and 3.12, a job with only the required dependencies (the tests of the optional extras must skip, not fail), the notebooks (with `--smoke`, which caps the iterations of the slow ones: CI checks that they run, the full run above is yours), and the documentation build (warnings are errors). All of it must pass before a PR is merged.

## Code

- **Types.** Annotate public functions; pyright checks `src/` in standard mode (not strict, not the tests), and piazzi ships `py.typed`, so users' type checkers read the annotations. Fix a pyright error with a precise annotation, a narrowing `assert` or `cast` with a comment saying why it holds, or a small restructuring; don't silence it with `# type: ignore`.
- **Warnings are bugs.** Fix the cause rather than silencing it. If a warning comes from a dependency, filter it as narrowly as possible (message, category and module), next to a comment linking the upstream issue and saying when to remove the filter, as `quiet_jvp_warnings` in `_common.py` does.
- **No aliasing.** Clone `x0` and anything handed back to the user or to a callback; give user functions their own copy of the point; copy tensors a user's function returns if the solver keeps them across calls (a `system` may reuse its output buffers). Detach `aux` with `detach_tree`.
- **Devices and dtypes.** The objective runs on the device and in the dtype of `x0`; the normal equations are solved in `solve_dtype` on `solve_device` (float64 on the CPU by default). MPS has no float64, so cast on the side that supports the wider dtype (`transfer` in `_common.py`, `t2n`/`n2t` in `scipy/_convert.py`) and never create a float64 tensor on MPS, even briefly.
- **Tolerances relative to scale.** Tests such as "is this a descent direction" or "is this curvature pair positive" must be relative to the norms involved, so that an objective multiplied by 1e-9 behaves like the original.
- **Callbacks** receive an `IterationInfo` (tensors, plus the evaluation counts so far) after every iteration, accepted or not, and stop the run by returning `True`. Every solver and wrapper follows this convention.
- **Docstrings and the docs are part of the change.** A docstring must describe what the code does now: defaults, stopping rules, counts and messages. Update the user guide in `docs/guide/` (tables, numbers and examples), `docs/api/` for a new public name (its group page and the summary table in `docs/api/index.rst`), and the README's feature list and example, in the same commit. Keep the README short: details belong in the docs.
- Prose (docstrings, comments, README, docs, notebooks) uses British spelling (optimise, minimiser, behaviour); API names follow SciPy and PyTorch (`minimize`, `OptimizeResult`). Lines are at most 120 characters (ruff).

## Docstrings

Public functions and classes use PyTorch-style docstrings: Google sections, which Sphinx renders with napoleon and ruff checks (`D` rules, `convention = "google"`).

- A one-line summary, a blank line, then the behaviour: defaults, stopping rules, counts and messages.
- `Args:` for positional parameters and `Keyword Args:` for keyword-only ones, each as `name (type): description`. Add `, optional` to the type of a parameter with a default, and end the entry with ``Default: ``value````. Use short type names (`Tensor`, `Callable`, `Manifold`, `str or Callable`, `tuple of int`); `docs/conf.py` maps them to links (`napoleon_type_aliases`), so add new ones there.
- `Returns:` as `Type: description`, and `Attributes:` for dataclass fields.
- Give the shape of every tensor argument and result as ``(n,)``, ``(n, n)`` or ``(..., 3, 3)``, never "n x n", and the units where they matter (radians, the units of `x` or `r`). Write transposes as `J^T`.
- Refer to other APIs with roles, so they link: `` :func:`piazzi.gauss_newton` ``, `` :class:`torch.Tensor` ``, `` :func:`scipy.optimize.least_squares` ``. Use the full `piazzi.` path for piazzi's own names.
- Link a high-level interface and the method-specific solver it runs both ways with a `See Also:` section, one `- :func:` bullet per entry (without bullets, Sphinx runs the entries together).
- Add an `Example::` with `>>>` lines to entry points and to anything whose use isn't obvious. Examples run as doctests (`pytest` collects `src/` with `--doctest-modules`; `torch` and `piazzi` are already imported), so their output must match exactly: print rounded values or use `torch.allclose`, and seed any randomness with a local `torch.Generator`.

The docs build is nitpicky with warnings as errors, so a reference or type name that doesn't resolve fails CI.

## Tests

- **Every bug fix comes with a regression test, and the test must fail without the fix.** Check this by running it against the previous version of the code before committing.
- Test float32 with float32-appropriate tolerances, not just float64. Tolerances must be reachable in the dtype: no `step_tol=1e-10` in float32.
- `tests/test_devices.py` parametrises over the CPU, CUDA and MPS (skipping what is unavailable). CI is CPU-only, so after touching device or dtype handling run `uv run pytest tests/test_devices.py` on a Mac (MPS) or a CUDA machine and say so in the PR.
- Tests of optional extras skip when the extra is missing (`pytest.importorskip`, or a `skipif` marker such as `needs_scipy` / `NEEDS_CG`).
- Compare with a reference where one exists: SciPy for least squares and minimisation, Ceres for manifolds, upstream `torch.optim` for the adapter. Port reference test suites with attribution rather than inventing weaker tests.

## Notebooks

- The first code cell installs piazzi from GitHub when running in Colab; local runs use `uv sync --group notebooks`.
- Commit notebooks executed, with their outputs, and without stderr output. Re-execute them (`uv run jupyter nbconvert --to notebook --execute --inplace notebooks/<name>.ipynb`) whenever the library change affects their output.
- Numbers quoted in the prose (iterations, evaluations, costs, errors) must match the outputs. Check them after re-executing.
- Data a notebook downloads is cached next to it and ignored by git.

## Git and pull requests

- One branch and one PR per change, against `main`. Do not commit to `main` directly, force-push a shared branch or rewrite published history.
- Commit messages explain why, not only what; a bug fix says what went wrong and how it showed.
- Keep commits focused; do not mix unrelated refactoring into a fix.
- PRs are merged with a merge commit once CI is green and review comments are addressed (replied to and resolved).
