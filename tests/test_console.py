from __future__ import annotations

import subprocess
import sys


def test_module_help_runs() -> None:
    result = subprocess.run([sys.executable, "-m", "ocxkcd", "--help"], check=True, text=True, capture_output=True)
    assert "process" in result.stdout
    assert "doctor" in result.stdout

