"""Smoke test for examples/quickstart.py (rule PS303 mate).

Runs the standalone script in a subprocess so that any import-time or
runtime error surfaces as a test failure. Mirrors the broader
``test_examples_smoke.py`` glob, but pinned to the canonical example
expected by ``scitex-dev`` (one ``test_<example>.py`` per example).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "quickstart.py"


def test_quickstart_example_file_exists_on_disk() -> None:
    # Arrange
    example = EXAMPLE
    # Act
    is_file = example.is_file()
    # Assert
    assert is_file, f"missing example: {example}"


def test_quickstart_runs_to_completion_with_zero_exit_code(tmp_path: Path) -> None:
    # Arrange
    cmd = [sys.executable, str(EXAMPLE)]
    # Act
    result = subprocess.run(
        cmd,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    # Assert
    assert result.returncode == 0, (
        f"quickstart.py failed (rc={result.returncode})\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
