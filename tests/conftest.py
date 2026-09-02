"""Shared test setup: the repo root on sys.path so tests import modules directly.

Run the suite from the repo root: python -m pytest tests/
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
