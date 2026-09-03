"""Imports written inside function bodies, and the kernel one of them hid.

Deferred imports are used throughout this app to keep heavy dependencies off the
startup path. Nothing type-checks them: a module deleted out from under a
function-level `import` stays invisible until that line runs in production. The
int8 scoring kernel was reached that way, so removing its module broke every
search while the whole test suite still passed.
"""

from __future__ import annotations

import ast
import importlib.util
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
STDLIB_OK = {"__future__"}


def _deferred_imports(path: pathlib.Path):
    """(module, lineno) for every import that is not at module top level."""
    tree = ast.parse(path.read_text())
    top = {id(n) for n in tree.body}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)) or id(node) in top:
            continue
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0], node.lineno
        elif node.level == 0 and node.module:
            yield node.module.split(".")[0], node.lineno


@pytest.mark.parametrize("path", sorted(ROOT.glob("*.py")), ids=lambda p: p.name)
def test_every_deferred_import_resolves(path):
    missing = [
        f"{path.name}:{line} imports {mod!r}"
        for mod, line in _deferred_imports(path)
        if mod not in STDLIB_OK and importlib.util.find_spec(mod) is None
    ]
    assert not missing, "function-level imports of modules that do not exist: " + "; ".join(missing)


def test_the_int8_kernel_scores_a_row_as_a_plain_dot_product():
    """`select` -> `score` -> this kernel is the whole search path, so a broken
    kernel is a broken app rather than a slow one."""
    import full_corpus

    kernel = full_corpus._score_kernel()
    corpus_i8 = np.array([[1, 2], [3, 4], [-1, 0]], dtype=np.int8)
    weights = np.array([1.0, 0.5], dtype=np.float32)
    out = np.empty(corpus_i8.shape[0], dtype=np.float32)
    kernel(corpus_i8, weights, out)
    assert np.allclose(out, [2.0, 5.0, -1.0])
