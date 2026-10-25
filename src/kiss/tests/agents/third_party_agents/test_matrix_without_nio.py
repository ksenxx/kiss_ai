# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Matrix sign-in when ``matrix-nio`` is not installed.

``matrix-nio`` is a development dependency only, so an ordinary
installation may lack it.  ``matrix_sea`` imports ``nio`` lazily and
treats the ``ImportError`` as a supported condition: a browser sign-in
still stores the session and answers with a warning, a hand-supplied
token is refused with the import error.  The test process has nio
installed, so these paths run in a child interpreter whose ``sys.path``
starts with a shim directory holding a ``nio`` package that raises
``ImportError`` when imported: the real import machinery finds the
shim first, exactly as it would find nothing on a bare installation.
The child talks to this process's real OAuth emulator
(``_AuthServer`` from ``test_muse_connect_flows``) over loopback and
hands back what the tools answered as JSON.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.third_party_agents.matrix.matrix_sea import _config as mx_config
from kiss.tests.agents.third_party_agents.recording_http import recording_server
from kiss.tests.agents.third_party_agents.test_muse_connect_flows import (
    _AuthHandler,
    _AuthServer,
)

# Runs in the child.  argv: shim directory, homeserver URL.  Prints
# "started" once the device grant is pending, waits for a line on stdin
# (the parent approves the grant meanwhile), then prints one JSON line.
_CHILD = """
import json, sys, time
sys.path.insert(0, sys.argv[1])
try:
    import nio
except ImportError:
    pass
else:
    raise SystemExit("the shim did not hide nio")
from kiss.agents.third_party_agents.matrix.matrix_sea import MatrixAgent, _config
from kiss.tests.agents.third_party_agents.muse_test_utils import auth_tools

agent = MatrixAgent()
tools = auth_tools(agent)
homeserver = sys.argv[2]
legacy = json.loads(tools["authenticate_matrix"](homeserver, "tok"))
legacy_saved = _config.path.exists()
started = json.loads(tools["authenticate_matrix"](homeserver))
print("started", flush=True)
sys.stdin.readline()
deadline = time.monotonic() + 60
while True:
    done = json.loads(tools["finish_matrix_auth"]())
    if done.get("status") != "pending":
        break
    if time.monotonic() > deadline:
        raise SystemExit("finish_matrix_auth kept reporting pending")
    time.sleep(0.2)
print(json.dumps({
    "legacy": legacy,
    "legacy_saved": legacy_saved,
    "started": started,
    "done": done,
    "client_is_none": agent._backend._client is None,
    "status": tools["check_matrix_auth"](),
}), flush=True)
"""


def _write_nio_shim(root: Path) -> Path:
    """Create a ``nio`` package under *root* that fails to import; return *root*."""
    package = root / "nio"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "raise ImportError(\"No module named 'nio' (matrix-nio is not installed)\")\n"
    )
    return root


def _read_line(fd: int, timeout: float) -> str | None:
    """Read one line from *fd* within *timeout* seconds.

    Returns ``None`` on timeout and whatever was read (possibly ``""``) when
    the writer closes the pipe first.

    A daemon thread does the blocking reads (``select()`` does not watch
    pipes on Windows) and accumulates whatever arrives until a newline
    shows up, so a partial line stalls the thread, not the parent, which
    gives up at the deadline.
    """
    chunks: list[bytes] = []
    done = threading.Event()
    threading.Thread(target=_read_until_newline, args=(fd, chunks, done), daemon=True).start()
    if not done.wait(timeout):
        return None
    return b"".join(chunks).split(b"\n", 1)[0].decode()


def _read_until_newline(fd: int, chunks: list[bytes], done: threading.Event) -> None:
    """Append reads of *fd* to *chunks* until a newline or EOF, then set *done*."""
    try:
        while True:
            chunk = os.read(fd, 4096)
            chunks.append(chunk)
            if not chunk or b"\n" in chunk:
                break
    except OSError:  # the pipe was closed under a read that outlived its deadline
        pass
    done.set()


def _run_child(
    shim: Path, server: _AuthServer, script: str = _CHILD, start_timeout: float = 60
) -> dict[str, Any]:
    """Run the sign-in in a nio-less interpreter against *server*; return its answers.

    Both waits are bounded (pytest-timeout is disabled project-wide): a child
    that has not printed a full ``started`` line after *start_timeout* seconds
    is killed, as is one that has not finished 120 s after being approved.
    Its stderr goes to a file so a chatty child cannot fill a pipe nobody
    drains while the parent waits for ``started``.
    """
    stderr_path = shim.parent / "child.stderr"
    with (
        stderr_path.open("w") as stderr,
        subprocess.Popen(
            [sys.executable, "-c", script, str(shim), server.base_url],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
            text=True,
        ) as child,
    ):
        assert child.stdout is not None and child.stdin is not None
        try:
            line = _read_line(child.stdout.fileno(), start_timeout)
            assert line is not None, f"the child did not print 'started' within {start_timeout} s"
            assert line.strip() == "started", stderr_path.read_text()
            server.approve()
            child.stdin.write("approved\n")
            child.stdin.flush()
            out, _ = child.communicate(timeout=120)
        except BaseException:
            child.kill()
            raise
    assert child.returncode == 0, stderr_path.read_text()
    return dict(json.loads(out))


def test_run_child_is_bounded(tmp_path: Path) -> None:
    """A child that never prints ``started`` is killed at the deadline; a crash shows its stderr.

    A child that flushes only part of the line and then stalls must hit the
    same deadline instead of blocking the parent until the line completes.
    """
    partial = "import sys, time; sys.stdout.write('sta'); sys.stdout.flush(); time.sleep(1000)"
    with recording_server(_AuthHandler, _AuthServer) as server:
        with pytest.raises(AssertionError, match="did not print 'started' within 1 s"):
            _run_child(tmp_path, server, script="import time; time.sleep(1000)", start_timeout=1)
        with pytest.raises(AssertionError, match="boom"):
            _run_child(tmp_path, server, script="import sys; sys.exit('boom')")
        t0 = time.monotonic()
        with pytest.raises(AssertionError, match="did not print 'started' within 0.5 s"):
            _run_child(tmp_path, server, script=partial, start_timeout=0.5)
        assert time.monotonic() - t0 < 5, "a partial line stalled the parent past the deadline"


def test_matrix_sign_in_without_nio(isolated_kiss_home: Path, tmp_path: Path) -> None:
    """A token sign-in reports the missing module; a browser sign-in stores the session, warns."""
    shim = _write_nio_shim(tmp_path / "no_nio")
    with recording_server(_AuthHandler, _AuthServer) as server:
        server.expiring = True
        answers = _run_child(shim, server)

    legacy = answers["legacy"]
    assert legacy["ok"] is False and "nio" in legacy["error"]
    assert answers["legacy_saved"] is False
    assert answers["started"]["status"] == "consent_required"

    done = answers["done"]
    assert done["ok"] is True and done["user_id"] == "@alice:example.org"
    assert "matrix-nio is not installed" in done["warning"]
    assert answers["client_is_none"] is True
    cfg = json.loads(mx_config.path.read_text())
    assert cfg["access_token"] == "access-1" and cfg["refresh_token"] == "refresh-1"
    assert cfg["homeserver_url"] == server.base_url
    # Without a client the status tool reports "not authenticated" even
    # though the session is stored: it is picked up once nio is installed.
    assert answers["status"].startswith("Not authenticated with Matrix.")
