"""Execute notebooks and fail on any error, warning or other stderr output.

Usage: python scripts/execute_notebooks.py [--smoke] [NOTEBOOK ...]  (default: notebooks/*.ipynb)

--smoke sets PIAZZI_NOTEBOOK_SMOKE=1 in the kernel. Notebooks that are slow read it and cap their iterations,
so that the run checks that the code works, not that the solvers converge (the numbers then differ from the
committed outputs). Without it, or in Colab, the notebooks run in full.

The kernel runs with PYTHONWARNINGS=default, so every warning is shown in the cell output
(including DeprecationWarnings raised inside libraries, which IPython hides by default).
A warning is printed to stderr, so any stderr output fails the run. The notebooks are not
modified: re-run them with ``jupyter nbconvert --execute --inplace`` to refresh their outputs.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = Path(__file__).resolve().parent.parent


def problems_in(path: Path) -> list[str]:
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(path.parent)}})
    try:
        client.execute()
    except CellExecutionError as err:
        return [f"error: {err}"]
    except Exception as err:  # a cell timeout, a dead kernel (out of memory), ...: report it, go on
        return [f"error: {type(err).__name__}: {err}"]
    problems = []
    for i, cell in enumerate(nb.cells):
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream" and out.get("name") == "stderr":
                problems.append(f"cell {i} wrote to stderr:\n{out['text'].rstrip()}")
    return problems


def main(argv: list[str]) -> int:
    os.environ["PYTHONWARNINGS"] = "default"  # inherited by the kernel
    if "--smoke" in argv:
        os.environ["PIAZZI_NOTEBOOK_SMOKE"] = "1"
    paths = [Path(a) for a in argv if a != "--smoke"] or sorted((ROOT / "notebooks").glob("*.ipynb"))
    failed = False
    for path in paths:
        problems = problems_in(path)
        print(f"{'FAIL' if problems else 'ok  '} {path}")
        for p in problems:
            print("  " + p.replace("\n", "\n  "))
        failed |= bool(problems)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
