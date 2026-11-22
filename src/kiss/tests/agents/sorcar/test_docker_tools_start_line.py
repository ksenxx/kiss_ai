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
* ``start_line<=0`` is rejected;
* the window and its line count come from one reading of the file.
"""

from __future__ import annotations

import functools
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from kiss.agents.sorcar.docker_tools import DockerTools


def _host_bash(command: str, _description: str, env: dict[str, str] | None = None) -> str:
    """Run the synthesised shell snippet on the host (no Docker required), in *env* if given."""
    proc = subprocess.run(
        ["bash", "-c", command],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    out = proc.stdout
    if proc.returncode != 0:
        out += f"\n[exit code: {proc.returncode}]\n{proc.stderr}"
    return out


def _awk_that_waits(bin_dir: Path, counted: Path, released: Path) -> None:
    """Put an ``awk`` in *bin_dir* that runs the real one, then waits for *released*.

    It makes *counted* once the real ``awk`` has exited (the line
    count is taken) and gives up waiting after 30 s.
    """
    real = shutil.which("awk")
    assert real is not None
    script = bin_dir / "awk"
    script.write_text(
        "#!/bin/sh\n"
        f'{shlex.quote(real)} "$@"\n'
        "status=$?\n"
        f": > {shlex.quote(str(counted))}\n"
        "n=0\n"
        f'while [ ! -e {shlex.quote(str(released))} ] && [ "$n" -lt 3000 ]; do\n'
        "  sleep 0.01; n=$((n + 1))\n"
        "done\n"
        'exit "$status"\n'
    )
    script.chmod(0o755)


def _read_into(results: list[str], tools: DockerTools, path: str, max_lines: int) -> None:
    """Thread body: ``Read`` the first *max_lines* lines of *path* and keep the result."""
    results.append(tools.Read(path, max_lines=max_lines))


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


@pytest.mark.skipif(sys.platform != "linux", reason="the Write script needs GNU mv -T")
def test_a_file_replaced_while_it_is_read_is_answered_as_one_version(temp_dir: Path) -> None:
    """The window and its line count come from one reading of the file (S5/F2).

    The count was one ``awk`` over the path and the window a ``tail``
    that opened the path again: a file another tool's Write replaced
    in between was answered with the new file's lines under the old
    file's count, which neither version gives.  The real ``awk`` is
    run by a wrapper first on ``PATH`` that waits once it has counted;
    a second tool instance replaces the file meanwhile.
    """
    p = temp_dir / "f.txt"
    p.write_text("old-1\nold-2\nold-3\n")
    bin_dir = temp_dir / "bin"
    bin_dir.mkdir()
    counted, released = temp_dir / "counted", temp_dir / "released"
    _awk_that_waits(bin_dir, counted, released)
    env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}
    reader = DockerTools(functools.partial(_host_bash, env=env))
    results: list[str] = []
    read = threading.Thread(target=_read_into, args=(results, reader, str(p), 2))
    read.start()
    try:
        deadline = time.monotonic() + 30
        while not counted.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert counted.exists(), "the Read never counted the file's lines"
        written = DockerTools(_host_bash).Write(str(p), "".join(f"new-{i}\n" for i in range(1, 7)))
        assert written.startswith("Successfully wrote"), written
    finally:
        released.touch()
        read.join(30)
    assert not read.is_alive()
    assert results[0] in (
        "old-1\nold-2\n[truncated: 1 more lines]\n",
        "new-1\nnew-2\n[truncated: 4 more lines]\n",
    ), results[0]
