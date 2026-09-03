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
import re

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
STDLIB_OK = {"__future__"}
# Gitignored proprietary artifacts (AGENTS.md): repopulated locally before a
# deploy and COPYed into the image, so they are absent in a clean checkout and
# in CI. Their absence is expected; a missing first-party module is not.
VENDORED = {"adp", "lilypad_py"}


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
        if mod not in STDLIB_OK and mod not in VENDORED and importlib.util.find_spec(mod) is None
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


WEB_APP_JS = ROOT / "web" / "app.js"
# Endpoints the browser calls, as they appear in app.js: plain strings and
# template literals. `${...}` becomes a path parameter so `/api/v1/tags/${tag}`
# matches the route `/api/v1/tags/{tag}`.
_FETCH_CALL = re.compile(r"""(?:apiFetch|_fullFetch|_issueFullVector|fetch)\(\s*(["'`])(/[^"'`]+)\1""")
_TEMPLATE_ARG = re.compile(r"\$\{[^}]*\}")


def _frontend_endpoints() -> set[str]:
    out = set()
    for _quote, raw in _FETCH_CALL.findall(WEB_APP_JS.read_text()):
        path = _TEMPLATE_ARG.sub("{}", raw.split("?", 1)[0]).rstrip("/")
        out.add(path or "/")
    return out


def _served_paths() -> set[str]:
    import api_v1
    import web_server

    paths = {getattr(r, "path", "") for r in web_server.app.routes}
    paths |= {r.path for r in api_v1.router.routes}
    # Normalise `{tag}` to `{}` so a template-literal argument matches it.
    return {re.sub(r"\{[^}]*\}", "{}", p) for p in paths if p}


def test_the_frontend_only_calls_routes_that_exist():
    """Nothing type-checks a URL in a string. Deleting the legacy surface left
    app.js still calling /api/search and /api/refine -- unreachable behind a
    `return true`, so no error surfaced, and the route-surface test only checked
    what the server exposes, never what the browser asks for."""
    missing = sorted(_frontend_endpoints() - _served_paths())
    assert not missing, f"web/app.js calls routes the app does not serve: {missing}"
