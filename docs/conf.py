"""Sphinx configuration for the piazzi documentation."""

from __future__ import annotations

import shutil
from pathlib import Path

import piazzi

DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parent

project = "piazzi"
author = "CAI4CAI"
copyright = "CAI4CAI"
release = piazzi.__version__
version = release

extensions = [
    "myst_nb",  # Markdown pages (MyST) and the executed notebooks
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
]

exclude_patterns = ["_build", "jupyter_execute", "**.ipynb_checkpoints"]
# Every cross-reference must resolve (with -W, CI fails on a broken one).
nitpicky = True

# -- Markdown and notebooks -------------------------------------------------------------
myst_enable_extensions = ["colon_fence", "dollarmath", "deflist"]
myst_heading_anchors = 3
# The notebooks are committed with their outputs and executed by CI (scripts/execute_notebooks.py),
# so the docs render the stored outputs rather than running them again.
nb_execution_mode = "off"


def _copy_notebooks() -> None:
    """Copy ``notebooks/*.ipynb`` into ``docs/examples/``: Sphinx only reads files under ``docs/``.

    Old copies are removed first, so a renamed or deleted notebook leaves no stale page behind.
    """
    target = DOCS / "examples"
    for old in target.glob("*.ipynb"):
        old.unlink()
    for nb in sorted((ROOT / "notebooks").glob("*.ipynb")):
        shutil.copy2(nb, target / nb.name)


_copy_notebooks()

# -- API reference --------------------------------------------------------------------
autodoc_member_order = "bysource"
# The docstrings give each argument's type, PyTorch style, so autodoc doesn't repeat the annotations.
autodoc_typehints = "none"
autodoc_default_options = {"members": True, "show-inheritance": True}
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_rtype = False
# Link the short type names of PyTorch-style docstrings ("x (Tensor, optional): ...").
napoleon_preprocess_types = True
napoleon_type_aliases = {
    "Tensor": ":class:`torch.Tensor`",
    "Callable": ":class:`~collections.abc.Callable`",
    "Any": ":data:`~typing.Any`",
    "Manifold": ":class:`piazzi.Manifold`",
    "OptimizeResult": ":class:`piazzi.OptimizeResult`",
    "LinearSolver": ":data:`piazzi.LinearSolver`",
    "torch.dtype": ":class:`torch.dtype`",
    "torch.device": ":class:`torch.device`",
    "array_like": ":term:`numpy:array_like`",
    "sequence": ":term:`python:sequence`",
    "numpy.ndarray": ":class:`numpy.ndarray`",
    "scipy.optimize.Bounds": ":class:`scipy.optimize.Bounds`",
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "torch": ("https://docs.pytorch.org/docs/stable", None),
}

# -- HTML -----------------------------------------------------------------------------
# The GitHub mark (Octicons "mark-github", MIT licence), for the footer icon.
_GITHUB_MARK = (
    '<svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">'
    '<path fill-rule="evenodd" d="'
    "M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.4"
    "9-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.5"
    "8 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.3"
    "1-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36."
    "09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.8"
    "7 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013"
    " 0 0 0 16 8c0-4.42-3.58-8-8-8z"
    '"></path></svg>'
)

html_theme = "furo"
html_title = f"piazzi {release}"
html_static_path = ["_static"]
html_theme_options = {
    "source_repository": "https://github.com/cai4cai/torch-piazzi",
    "source_branch": "main",
    "source_directory": "docs/",
    # A GitHub icon in the footer of every page, next to the sidebar's "Project links".
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/cai4cai/torch-piazzi",
            "html": _GITHUB_MARK,
            "class": "",
        },
    ],
}
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
