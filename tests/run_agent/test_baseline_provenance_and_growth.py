"""Tests for baseline runner provenance and silent growth refusal in run_tests.py."""

import subprocess
import sys
import pytest
from pathlib import Path

def test_unprovenanced_baseline_refusal(tmp_path):
    # Baseline file without # Provenance header
    unprovenanced = tmp_path / "unprovenanced_baseline.txt"
    unprovenanced.write_text("tests/agent/test_fake.py::test_foo\n", encoding="utf-8")

    cmd = [
        sys.executable, ".github/scripts/run_tests.py",
        "--selector", "tests/run_agent/test_seam_handshake.py",
        "--baseline-file", str(unprovenanced)
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode != 0
    assert "lacks runner provenance header" in proc.stdout or "lacks runner provenance header" in proc.stderr

def test_unprovenanced_baseline_opt_in_override(tmp_path):
    unprovenanced = tmp_path / "unprovenanced_baseline.txt"
    unprovenanced.write_text("tests/agent/test_fake.py::test_foo\n", encoding="utf-8")

    cmd = [
        sys.executable, ".github/scripts/run_tests.py",
        "--selector", "tests/run_agent/test_seam_handshake.py",
        "--baseline-file", str(unprovenanced),
        "--allow-unprovenanced-baseline"
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert "lacks runner provenance header" not in proc.stdout

def test_baseline_growth_refusal(tmp_path):
    provenanced = tmp_path / "provenanced_baseline.txt"
    provenanced.write_text(
        "# Provenance: github-runner\n"
        "tests/agent/test_fake.py::test_foo1\n"
        "tests/agent/test_fake.py::test_foo2\n",
        encoding="utf-8"
    )

    cmd = [
        sys.executable, ".github/scripts/run_tests.py",
        "--selector", "tests/run_agent/test_seam_handshake.py",
        "--baseline-file", str(provenanced),
        "--max-baseline-failures", "1"
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode != 0
    assert "exceeds maximum allowed ceiling" in proc.stdout or "exceeds maximum allowed ceiling" in proc.stderr
