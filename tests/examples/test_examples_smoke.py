"""Smoke tests: every example script must run to completion."""

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = list(Path(__file__).resolve().parents[2].joinpath("examples").glob("*.py"))


def test_examples_directory_contains_at_least_one_script():
    # Arrange
    examples = EXAMPLES
    # Act
    has_any = bool(examples)
    # Assert
    assert has_any, "no example scripts found"


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda p: p.name)
def test_example_script_exits_with_zero_return_code(example, tmp_path):
    # Arrange
    cmd = [sys.executable, str(example)]
    # Act
    result = subprocess.run(
        cmd,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    # Assert
    assert result.returncode == 0, f"{example.name} failed: {result.stderr}"
