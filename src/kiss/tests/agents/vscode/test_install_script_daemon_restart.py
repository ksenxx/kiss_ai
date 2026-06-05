# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression: ``install.sh`` must stop the running kiss-web daemon by
**port** between ``code --install-extension`` and the ``.extension-updated``
marker write — otherwise the chat webview comes up blank when the script is
run while VS Code is already open.

Background — the bug
====================
When ``install.sh`` runs while VS Code is open, step [6/6] reinstalls the
extension with ``code --install-extension --force``.  That replaces the
extension directory tree the running daemon's bundled kiss_project
(``.venv/bin/kiss-web``) was loaded from, so the live daemon is technically
broken even while it is still listening on port 8787.

The script then writes ``~/.kiss/.extension-updated``.  The extension's
``fs.watchFile`` poll detects the marker and fires
``workbench.action.reloadWindow`` (~2 s later).  The freshly reloaded webview
reconnects to the still-listening-but-broken daemon over the UDS socket and
renders **blank**.

The fix mirrors ``scripts/build-extension.sh``: kill the daemon by listening
port (8787) AFTER ``--install-extension`` and BEFORE the marker write, then
remove the stale UDS socket, so the LaunchAgent / systemd ``KeepAlive``
respawns a clean daemon from the freshly-installed VSIX before the reload
fires.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
INSTALL_SCRIPT = REPO / "install.sh"

# The literal shell statement that *writes* the marker (the file path also
# appears earlier in explanatory comments, so match the redirection write).
MARKER_WRITE = '> "$HOME/.kiss/.extension-updated"'


def test_install_script_kills_daemon_after_install_before_marker() -> None:
    """``install.sh`` must kill kiss-web on port 8787 between
    ``--install-extension`` and the ``.extension-updated`` marker write.
    """
    src = INSTALL_SCRIPT.read_text(encoding="utf-8")
    install_idx = src.index('--install-extension "$VSIX"')
    kill_idx = src.index("lsof -ti :8787")
    marker_idx = src.index(MARKER_WRITE)
    assert install_idx < kill_idx < marker_idx, (
        "kiss-web kill block must appear AFTER --install-extension "
        "and BEFORE the .extension-updated marker write."
    )


def test_install_script_removes_stale_uds_socket() -> None:
    """``install.sh`` must remove ``~/.kiss/sorcar.sock`` after killing the
    daemon and before the marker write so the extension's reconnect does not
    hit a stale socket of the dead daemon.
    """
    src = INSTALL_SCRIPT.read_text(encoding="utf-8")
    assert 'rm -f "$HOME/.kiss/sorcar.sock"' in src, (
        "install.sh must remove the stale ~/.kiss/sorcar.sock"
    )
    sock_idx = src.index('rm -f "$HOME/.kiss/sorcar.sock"')
    kill_idx = src.index("lsof -ti :8787")
    marker_idx = src.index(MARKER_WRITE)
    assert kill_idx < sock_idx < marker_idx


def test_install_script_force_kills_survivors() -> None:
    """SIGTERM may not be enough — the script must SIGKILL stragglers."""
    src = INSTALL_SCRIPT.read_text(encoding="utf-8")
    assert "kill -9" in src, (
        "install.sh should force-kill (kill -9) survivors after the "
        "graceful-shutdown poll loop."
    )
