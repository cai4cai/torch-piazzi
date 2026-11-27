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
nitpicky = False

# -- Markdown and notebooks -------------------------------------------------------------
myst_enable_extensions = ["colon_fence", "dollarmath", "deflist"]
myst_heading_anchors = 3
# The notebooks are committed with their outputs and executed by CI (scripts/execute_notebooks.py),
# so the docs render the stored outputs rather than running them again.
nb_execution_mode = "off"


def _copy_notebooks() -> None:
    """Copy ``notebooks/*.ipynb`` into ``docs/examples/``: Sphinx only reads files under ``docs/``."""
    target = DOCS / "examples"
    for nb in sorted((ROOT / "notebooks").glob("*.ipynb")):
        shutil.copy2(nb, target / nb.name)


_copy_notebooks()

# -- API reference --------------------------------------------------------------------
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented_params"
autodoc_default_options = {"members": True, "show-inheritance": True}
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_rtype = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "torch": ("https://docs.pytorch.org/docs/stable", None),
}

# -- HTML -----------------------------------------------------------------------------
html_theme = "furo"
html_title = f"piazzi {release}"
html_static_path = ["_static"]
html_theme_options = {
    "source_repository": "https://github.com/cai4cai/torch-piazzi",
    "source_branch": "main",
    "source_directory": "docs/",
}
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
