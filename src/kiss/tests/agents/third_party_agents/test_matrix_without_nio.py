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
import subprocess
import sys
from pathlib import Path
from typing import Any

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


def _run_child(shim: Path, server: _AuthServer) -> dict[str, Any]:
    """Run the sign-in in a nio-less interpreter against *server*; return its answers."""
    with subprocess.Popen(
        [sys.executable, "-c", _CHILD, str(shim), server.base_url],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ) as child:
        assert child.stdout is not None and child.stdin is not None
        assert child.stdout.readline().strip() == "started", child.stderr and child.stderr.read()
        server.approve()
        child.stdin.write("approved\n")
        child.stdin.flush()
        out, err = child.communicate(timeout=120)
    assert child.returncode == 0, err
    return dict(json.loads(out))


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
