# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``install.sh`` publishes its current step while it runs.

While it runs, install.sh writes ``"<pid>\\n<step>\\n"`` to
``$KISS_HOME/.install-progress`` at every ``>>>`` banner and removes the
file when it exits, so anything watching an install (a detached run
started by the daemon's Update button, say) can tell where it is.  The
file must:

* honour ``$KISS_HOME``,
* name the live install.sh pid on line 1 and the step text on line 2,
* follow the steps as the run progresses,
* be gone once install.sh has exited, whether it succeeded or failed.

A sandboxed copy of install.sh runs for real with fake ``git`` and ``uv``
on PATH.  The fakes snapshot the progress file when they are called (step
[1/5] calls ``git rev-parse``; step [2/5] calls ``uv sync``) and the fake
``uv sync`` fails, so the run ends with a non-zero exit inside step [2/5].
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from kiss.tests.conftest import posix_only

pytestmark = posix_only("install.sh runs under bash")

REPO = Path(__file__).resolve().parents[4]
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
# ``git rev-parse`` is the first external command of step [1/5] (it decides
# how the app files are listed); failing it takes the ``find`` path.
FAKE_GIT = (
    "#!/bin/bash\n"
    + SNAPSHOT
    + '[ "$1" = "rev-parse" ] && [ ! -e "$SNAP_DIR/git.txt" ] && snapshot git\nexit 1\n'
)
# ``uv sync`` is step [2/5]; failing it ends the run there.
FAKE_UV = (
    "#!/bin/bash\n"
    + SNAPSHOT
    + '[ "$1" = "sync" ] && { snapshot uv; exit 1; }\necho "uv 0.9.0 (stub)"\nexit 0\n'
)


def run_sandboxed_install(tmp_path: Path) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    """Run a sandboxed copy of install.sh; return (proc, kiss_home, snap_dir)."""
    home = tmp_path / "home"
    kiss_home = tmp_path / "custom-kiss-home"
    snap_dir = tmp_path / "snapshots"
    for d in (home, snap_dir):
        d.mkdir()

    checkout = tmp_path / "checkout"
    media = checkout / "src" / "kiss" / "webapp" / "media"
    media.mkdir(parents=True)
    (checkout / "src" / "kiss" / "__init__.py").write_text("")
    (media / "chat.html").write_text("<html></html>\n")
    (checkout / "pyproject.toml").write_text('[project]\nname = "x"\n')
    (checkout / "install.sh").write_text(INSTALL_SCRIPT.read_text())

    fakebin = tmp_path / "fakebin"
    fakebin.mkdir()
    for name, body in [("git", FAKE_GIT), ("uv", FAKE_UV)]:
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
    assert proc.returncode != 0, out  # the fake ``uv sync`` fails inside step [2/5]
    assert ">>> [2/5] Building the Python environment..." in proc.stdout

    git_snapshot = (snap_dir / "git.txt").read_text().splitlines()
    uv_snapshot = (snap_dir / "uv.txt").read_text().splitlines()
    assert git_snapshot[1:] == [f"[1/5] Copying KISS Sorcar to {kiss_home}/kiss..."], git_snapshot
    assert uv_snapshot[1:] == ["[2/5] Building the Python environment..."], uv_snapshot
    assert git_snapshot[0].isdigit(), "line 1 is install.sh's pid"
    assert git_snapshot[0] == uv_snapshot[0], "the same pid through the whole run"
    assert (snap_dir / "git.alive").read_text().strip() == "alive"
    assert (snap_dir / "uv.alive").read_text().strip() == "alive"

    # The EXIT trap removes the file even on a failed run, closing the toast.
    assert not (kiss_home / ".install-progress").exists(), out
    assert not (kiss_home / ".install-progress.tmp").exists(), out  # write-then-rename scratch
    # The file lives in $KISS_HOME, never in a hard-coded $HOME/.kiss.
    assert not (tmp_path / "home" / ".kiss" / ".install-progress").exists()
