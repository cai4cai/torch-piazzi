"""Pytest setup shared by the test suite and the docstring examples (``--doctest-modules`` on ``src/``)."""

import importlib.util

import pytest
import torch

import piazzi

# The SciPy wrappers import SciPy at module level: skip their docstrings when the extra is missing.
collect_ignore_glob = [] if importlib.util.find_spec("scipy") else ["src/piazzi/scipy/*"]


@pytest.fixture(autouse=True)
def _doctest_namespace(doctest_namespace):
    """Docstring examples use ``torch`` and ``piazzi`` without importing them, as PyTorch's do."""
    doctest_namespace["torch"] = torch
    doctest_namespace["piazzi"] = piazzi
