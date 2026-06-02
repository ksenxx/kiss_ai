"""Regression: ``scripts/build-extension.sh`` and ``DependencyInstaller.ts``
must kill the running kiss-web daemon by **port**, not by process name.

Background — the bug
====================
``kiss-web`` is a Python shebang script.  On macOS the kernel sets the
process's ``comm`` field to the (15-char-truncated) interpreter path,
NOT to ``kiss-web``, so ``pkill -x kiss-web`` is a silent no-op.

If the build script runs while VS Code is open:

1. ``code --uninstall-extension`` deletes the directory the running
   daemon's ``.venv/bin/kiss-web`` was loaded from.
2. ``code --install-extension`` recreates the directory.
3. The script writes ``~/.kiss/.extension-updated``; the extension's
   ``fs.watchFile`` poll triggers ``workbench.action.reloadWindow``.
4. Activation runs ``restartKissWebDaemon`` which historically did
   ``pkill -x kiss-web`` — a no-op — leaving the broken-files daemon
   alive across the launchctl bootout/bootstrap race window.

The fix kills the daemon by listening port (8787) in BOTH places:

* ``DependencyInstaller.ts::restartKissWebDaemon`` — via the
  ``killProcessOnPort(8787)`` helper.
* ``scripts/build-extension.sh`` — proactively, AFTER
  ``code --install-extension`` and BEFORE writing the marker, so
  by the time the extension auto-reloads (~2 s after marker)
  the LaunchAgent / systemd ``KeepAlive`` has already respawned
  a clean daemon from the freshly-installed VSIX.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
DEP_INSTALLER = REPO / "src/kiss/agents/vscode/src/DependencyInstaller.ts"
BUILD_SCRIPT = REPO / "scripts/build-extension.sh"








def test_build_script_kills_daemon_after_install_before_marker() -> None:
    """``build-extension.sh`` must kill kiss-web on port 8787 between
    ``code --install-extension`` and the ``.extension-updated`` marker
    write.  Ordering matters: the marker triggers an auto-reload of the
    extension within ~2 s, and we want the LaunchAgent / systemd
    ``KeepAlive`` to have respawned a clean daemon before that fires.
    """
    src = BUILD_SCRIPT.read_text(encoding="utf-8")
    install_idx = src.index("--install-extension kiss-sorcar.vsix")
    marker_idx = src.index(".extension-updated")
    kill_idx = src.index("lsof -ti :8787")
    assert install_idx < kill_idx < marker_idx, (
        "kiss-web kill block must appear AFTER --install-extension "
        "and BEFORE the .extension-updated marker write."
    )


def test_build_script_removes_stale_uds_socket() -> None:
    """``build-extension.sh`` must remove ``~/.kiss/sorcar.sock`` after
    killing the daemon — defensive cleanup so the extension's reconnect
    doesn't log ENOENT/ECONNREFUSED noise during the brief launchd
    respawn window.
    """
    src = BUILD_SCRIPT.read_text(encoding="utf-8")
    assert 'rm -f "$HOME/.kiss/sorcar.sock"' in src, (
        "build-extension.sh must remove the stale ~/.kiss/sorcar.sock"
    )
    # Ordering: socket cleanup after kill, before marker.
    sock_idx = src.index('rm -f "$HOME/.kiss/sorcar.sock"')
    kill_idx = src.index("lsof -ti :8787")
    marker_idx = src.index(".extension-updated")
    assert kill_idx < sock_idx < marker_idx


def test_build_script_force_kills_survivors() -> None:
    """SIGTERM may not be enough — the script must SIGKILL stragglers."""
    src = BUILD_SCRIPT.read_text(encoding="utf-8")
    assert "kill -9" in src, (
        "build-extension.sh should force-kill (kill -9) survivors after "
        "the graceful-shutdown poll loop."
    )


def test_dependency_installer_kickstarts_macos_launchagent() -> None:
    """Loading the plist is not enough; activation must force-start it."""
    src = DEP_INSTALLER.read_text(encoding="utf-8")
    assert "'kickstart', '-k'" in src, (
        "DependencyInstaller.ts must kickstart the macOS LaunchAgent after "
        "bootstrap/load so kiss-web actually starts."
    )


def test_dependency_installer_health_requires_uds_socket() -> None:
    """The VS Code extension uses ~/.kiss/sorcar.sock, not only port 8787."""
    src = DEP_INSTALLER.read_text(encoding="utf-8")
    assert "path.join(LOG_DIR, 'sorcar.sock')" in src, (
        "Daemon health must require the UDS socket used by AgentClient."
    )
