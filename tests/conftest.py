"""Pytest fixtures and rootdir marker for this package.

An empty conftest.py at tests/ is the canonical SciTeX
convention (audit-project PS208) — it pins the pytest
rootdir and gives downstream fixtures a home.

Also wires module-import-time coverage support for child
Python interpreters (subprocess.run, jupyter nbconvert
--execute, pytest-xdist workers). See
~/proj/scitex-dev/src/scitex_dev/_skills/general/05_development_06_subprocess-coverage.md
for the rationale — pytest-cov silently drops subprocess
coverage in its default setup, so we force-set
COVERAGE_PROCESS_START + COVERAGE_FILE at import time and
drop an idempotent .pth shim into site-packages.
"""

from __future__ import annotations

import os
import sysconfig
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Force-set (NOT setdefault — pytest-cov already populates
# COVERAGE_FILE before conftest.py runs, making setdefault a
# silent no-op).
os.environ["COVERAGE_PROCESS_START"] = str(_PROJECT_ROOT / "pyproject.toml")
os.environ["COVERAGE_FILE"] = str(_PROJECT_ROOT / ".coverage")


def _ensure_subprocess_coverage_shim() -> None:
    """Drop an idempotent `.pth` file in site-packages that
    auto-starts coverage in every child Python interpreter
    via `coverage.process_startup()`.
    """
    purelib = Path(sysconfig.get_paths()["purelib"])
    pth = purelib / "_scitex_linalg_subprocess_coverage.pth"
    shim = (
        "import os, coverage\n"
        "if os.environ.get('COVERAGE_PROCESS_START'):\n"
        "    coverage.process_startup()\n"
    )
    try:
        if not pth.exists() or pth.read_text() != shim:
            pth.write_text(shim)
    except OSError:
        # site-packages may be read-only (e.g. system Python);
        # silently skip — local dev venvs are writable and
        # that's where this matters.
        pass


_ensure_subprocess_coverage_shim()
