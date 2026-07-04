"""pytest fixtures/setup shared by tests/*.

Run the whole suite from the repo root:
    python -m pytest tests/

The tests import repo-root modules directly (e.g. `import lance_writer`,
`import search_engine`) rather than as an installed package, so the repo
root must be on `sys.path`. `pythonpath = .` in `pytest.ini` (repo root)
already does this for normal `python -m pytest` invocations; the explicit
insert below is a fallback for runners that don't honor that ini option.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
