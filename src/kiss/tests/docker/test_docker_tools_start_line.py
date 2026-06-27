# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for ``start_line`` on ``DockerTools.Read``.

The Docker Read tool synthesises a shell snippet that runs inside the
container. To avoid requiring a Docker daemon for the parameter contract,
these tests execute the very same shell snippet against the *host* bash
on a real temp file. This is still an end-to-end test of the shell-side
behaviour — the same code path the in-container Read uses.

The contract mirrors ``UsefulTools.Read``:

* default ``start_line=1`` is backward-compatible;
* an interior window is returned with a footer counting only the lines
  *after* the window;
* a ``start_line`` past EOF returns an explicit error sentinel and does
  not leak earlier content;
* ``start_line<=0`` is rejected.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from kiss.docker.docker_tools import DockerTools


def _host_bash(command: str, _description: str) -> str:
    """Run the synthesised shell snippet on the host (no Docker required)."""
    proc = subprocess.run(
        ["bash", "-c", command],
        capture_output=True,
        text=True,
        check=False,
    )
    out = proc.stdout
    if proc.returncode != 0:
        # Match DockerManager.Bash's error-tagging convention so the test
        # can assert on it; full stderr is appended for diagnosability.
        out += f"\n[exit code: {proc.returncode}]\n{proc.stderr}"
    return out


@pytest.fixture
def temp_dir() -> Iterator[Path]:
    d = Path(tempfile.mkdtemp()).resolve()
    cwd = Path.cwd()
    os.chdir(d)
    yield d
    os.chdir(cwd)
    shutil.rmtree(d, ignore_errors=True)


def _make_numbered(p: Path, n: int) -> None:
    p.write_text("".join(f"line-{i}\n" for i in range(1, n + 1)))


@pytest.mark.skipif(sys.platform == "win32", reason="needs POSIX bash")
def test_default_start_line_unchanged(temp_dir: Path) -> None:
    """Default ``start_line=1`` reads from the top — backward compatible."""
    p = temp_dir / "f.txt"
    _make_numbered(p, 5)
    tools = DockerTools(_host_bash)

    out = tools.Read(str(p))

    assert out.startswith("line-1\nline-2\nline-3\nline-4\nline-5\n")


@pytest.mark.skipif(sys.platform == "win32", reason="needs POSIX bash")
def test_start_line_skips_prefix(temp_dir: Path) -> None:
    p = temp_dir / "f.txt"
    _make_numbered(p, 5)
    tools = DockerTools(_host_bash)

    out = tools.Read(str(p), start_line=3)

    assert out.startswith("line-3\nline-4\nline-5\n")
    assert "line-1" not in out
    assert "line-2" not in out


@pytest.mark.skipif(sys.platform == "win32", reason="needs POSIX bash")
def test_window_truncation_counts_only_tail(temp_dir: Path) -> None:
    p = temp_dir / "f.txt"
    _make_numbered(p, 10)
    tools = DockerTools(_host_bash)

    out = tools.Read(str(p), start_line=4, max_lines=3)

    assert "line-4" in out
    assert "line-6" in out
    assert "line-7" not in out
    assert "[truncated: 4 more lines]" in out


@pytest.mark.skipif(sys.platform == "win32", reason="needs POSIX bash")
def test_start_line_past_eof_is_sentinel(temp_dir: Path) -> None:
    p = temp_dir / "f.txt"
    _make_numbered(p, 3)
    tools = DockerTools(_host_bash)

    out = tools.Read(str(p), start_line=10)

    assert "start_line" in out
    assert "line-1" not in out


@pytest.mark.skipif(sys.platform == "win32", reason="needs POSIX bash")
def test_start_line_zero_rejected(temp_dir: Path) -> None:
    p = temp_dir / "f.txt"
    _make_numbered(p, 3)
    tools = DockerTools(_host_bash)

    out = tools.Read(str(p), start_line=0)

    assert "start_line" in out.lower()
    assert "line-1" not in out


@pytest.mark.skipif(sys.platform == "win32", reason="needs POSIX bash")
def test_start_line_negative_rejected(temp_dir: Path) -> None:
    p = temp_dir / "f.txt"
    _make_numbered(p, 3)
    tools = DockerTools(_host_bash)

    out = tools.Read(str(p), start_line=-1)

    assert "start_line" in out.lower()
    assert "line-1" not in out
