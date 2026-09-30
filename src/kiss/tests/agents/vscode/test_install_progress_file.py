# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``install.sh`` publishes its current step for the VS Code extension.

While it runs, install.sh writes ``"<pid>\\n<step>\\n"`` to
``$KISS_HOME/.install-progress`` at every ``>>>`` banner and removes the
file when it exits.  The extension (``src/installProgress.ts``) mirrors
that file as a non-blocking progress notification, so the file must:

* honour ``$KISS_HOME`` (the extension resolves its state dir that way),
* name the live install.sh pid on line 1 and the step text on line 2,
* follow the steps as the run progresses,
* be gone once install.sh has exited, whether it succeeded or failed.

A sandboxed copy of install.sh runs for real with fake ``node``, ``npm``,
``npx`` and ``code`` CLIs on PATH.  The fakes snapshot the progress file
when they are called (step [2/5] calls ``node --version``; step [4/5]
calls ``npm ci``) and the fake ``npm ci`` fails, so the run ends with a
non-zero exit inside step [4/5].
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from kiss.tests.conftest import posix_only

pytestmark = posix_only("install.sh runs under bash")

REPO = Path(__file__).resolve().parents[5]
INSTALL_SCRIPT = REPO / "install.sh"

# ``snapshot NAME``: copy the progress file next to whether its pid is alive.
SNAPSHOT = """
snapshot() {
    cp "$KISS_HOME/.install-progress" "$SNAP_DIR/$1.txt"
    if kill -0 "$(head -n 1 "$KISS_HOME/.install-progress")" 2>/dev/null; then
        echo alive > "$SNAP_DIR/$1.alive"
    else
        echo dead > "$SNAP_DIR/$1.alive"
    fi
}
"""
FAKE_NODE = (
    "#!/bin/bash\n"
    + SNAPSHOT
    + '[ "$1" = "--version" ] && { snapshot node; echo v22.16.0; exit 0; }\nexit 0\n'
)
FAKE_NPM = (
    "#!/bin/bash\n"
    + SNAPSHOT
    + '[ "$1" = "--version" ] && { echo 10.9.0; exit 0; }\n'
    + '[ "$1" = "ci" ] && [ ! -e "$SNAP_DIR/npm.txt" ] && snapshot npm\nexit 1\n'
)
FAKE_NPX = "#!/bin/bash\nexit 0\n"
FAKE_CODE = '#!/bin/bash\n[ "$1" = "--version" ] && { echo 1.99.0; exit 0; }\nexit 0\n'


def run_sandboxed_install(tmp_path: Path) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    """Run a sandboxed copy of install.sh; return (proc, kiss_home, snap_dir)."""
    home = tmp_path / "home"
    kiss_home = tmp_path / "custom-kiss-home"
    snap_dir = tmp_path / "snapshots"
    for d in (home, snap_dir):
        d.mkdir()

    checkout = tmp_path / "checkout"
    (checkout / "src" / "kiss" / "agents" / "vscode").mkdir(parents=True)
    (checkout / "install.sh").write_text(INSTALL_SCRIPT.read_text())

    fakebin = tmp_path / "fakebin"
    fakebin.mkdir()
    for name, body in [
        ("node", FAKE_NODE),
        ("npm", FAKE_NPM),
        ("npx", FAKE_NPX),
        ("code", FAKE_CODE),
    ]:
        tool = fakebin / name
        tool.write_text(body)
        tool.chmod(0o755)

    env = dict(os.environ)
    env.update(
        HOME=str(home),
        KISS_HOME=str(kiss_home),
        SNAP_DIR=str(snap_dir),
        PATH=f"{fakebin}:{env['PATH']}",
        KISS_NONINTERACTIVE="1",
        KISS_HEARTBEAT_INTERVAL="1",
    )
    proc = subprocess.run(
        ["bash", str(checkout / "install.sh")],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    return proc, kiss_home, snap_dir


def test_install_publishes_live_step_to_kiss_home_and_removes_it_on_exit(tmp_path: Path) -> None:
    """The progress file follows the steps, names a live pid, and is removed at exit."""
    proc, kiss_home, snap_dir = run_sandboxed_install(tmp_path)
    out = proc.stdout + proc.stderr
    assert proc.returncode != 0, out  # the fake ``npm ci`` fails inside step [4/5]
    assert ">>> [4/5] Building VS Code extension..." in proc.stdout

    node_snapshot = (snap_dir / "node.txt").read_text().splitlines()
    npm_snapshot = (snap_dir / "npm.txt").read_text().splitlines()
    assert node_snapshot[1:] == ["[2/5] Checking Node.js..."], node_snapshot
    assert npm_snapshot[1:] == ["[4/5] Building VS Code extension..."], npm_snapshot
    assert node_snapshot[0].isdigit(), "line 1 is install.sh's pid"
    assert node_snapshot[0] == npm_snapshot[0], "the same pid through the whole run"
    assert (snap_dir / "node.alive").read_text().strip() == "alive"
    assert (snap_dir / "npm.alive").read_text().strip() == "alive"

    # The EXIT trap removes the file even on a failed run, closing the toast.
    assert not (kiss_home / ".install-progress").exists(), out
    assert not (kiss_home / ".install-progress.tmp").exists(), out  # write-then-rename scratch
    # The file lives in $KISS_HOME, never in a hard-coded $HOME/.kiss.
    assert not (tmp_path / "home" / ".kiss" / ".install-progress").exists()
