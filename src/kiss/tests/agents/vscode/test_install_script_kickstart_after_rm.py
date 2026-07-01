# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression test for the "KISS Sorcar Server is starting ..." hang after
the Update button.

Bug
===
Clicking the chat panel's Update button spawns ``install.sh`` detached from
the running kiss-web daemon (see ``_handle_run_update`` /
``_spawn_update_script`` in ``web_server.py``).  The script then:

  1. ``lsof -ti :8787 | xargs kill`` — sends SIGTERM and waits up to ~3 s.
  2. ``rm -f "$HOME/.kiss/sorcar.sock"`` — pre-emptively removes the stale
     UDS socket.
  3. Writes ``~/.kiss/.extension-updated`` so the VS Code extension reloads
     the window.

Steps 1 and 2 race against the supervisor (``KeepAlive`` on macOS,
``Restart=always`` on Linux): a freshly-respawned daemon may bind a NEW
``~/.kiss/sorcar.sock`` DURING the wait loop, only for step 2 to delete
the new file out from under it.  The daemon's open listening socket
survives the unlink (the kernel-level fd is independent of the directory
entry), so the daemon is "alive on port 8787" but its UDS file is gone —
every ``connect("$HOME/.kiss/sorcar.sock")`` from the extension's
``AgentClient`` returns ENOENT until *something* kills the daemon again.

That something used to require the user to fully restart VS Code, and
even then ``restartKissWebDaemon``'s old skip logic could leave them
stranded.

Defense-in-depth
================
This test pins a deterministic synchronous restart that ``install.sh``
issues immediately AFTER the ``rm -f``: ``launchctl kickstart -k`` on
macOS, ``systemctl --user restart`` on Linux.  Both commands force the
supervisor to (re-)start the unit RIGHT NOW, so the fresh daemon's
``_setup_server`` re-creates the UDS file BEFORE the marker write
triggers the VS Code reload.

The companion in-extension fix
(``decideRestart``'s ``unreachable-uds`` branch in
``src/kiss/agents/vscode/src/daemonHealth.js``) is a fallback for hosts
where ``launchctl`` / ``systemctl`` is unavailable.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
INSTALL_SCRIPT = REPO / "install.sh"

# The literal shell statement that *writes* the marker.
MARKER_WRITE = '> "$HOME/.kiss/.extension-updated"'


def test_install_script_kickstarts_macos_launchagent_after_rm() -> None:
    """On macOS, ``install.sh`` must explicitly ``launchctl kickstart -k``
    the kiss-web LaunchAgent AFTER removing the stale UDS file and
    BEFORE writing the update marker, so a fresh daemon respawn re-creates
    ``~/.kiss/sorcar.sock`` before the extension reload fires.
    """
    src = INSTALL_SCRIPT.read_text(encoding="utf-8")
    assert "launchctl kickstart -k" in src, (
        "install.sh must invoke ``launchctl kickstart -k`` so the kiss-web "
        "LaunchAgent is restarted synchronously after the rm — without this, "
        "a launchd KeepAlive respawn that races with ``rm -f sorcar.sock`` "
        "leaves the daemon listening on port 8787 with no UDS file, and the "
        "webview hangs on 'KISS Sorcar Server is starting ...' until VS Code "
        "is restarted."
    )
    assert "com.kiss.web-server" in src, (
        "the launchctl kickstart must target the kiss-web LaunchAgent label "
        "``com.kiss.web-server``"
    )

    rm_idx = src.index('rm -f "$HOME/.kiss/sorcar.sock"')
    kickstart_idx = src.index("launchctl kickstart -k")
    marker_idx = src.index(MARKER_WRITE)
    assert rm_idx < kickstart_idx < marker_idx, (
        "launchctl kickstart must run AFTER the rm (so the supervisor "
        "respawns a daemon that finds no stale UDS to fight with) and "
        "BEFORE the marker write (so the extension reload sees a fully-"
        "rebound UDS)"
    )


def test_install_script_restarts_systemd_user_unit_after_rm() -> None:
    """On Linux, ``install.sh`` must explicitly ``systemctl --user restart
    kiss-web`` AFTER removing the stale UDS file and BEFORE writing the
    marker, mirroring the macOS kickstart defense.
    """
    src = INSTALL_SCRIPT.read_text(encoding="utf-8")
    assert "systemctl --user restart kiss-web" in src, (
        "install.sh must invoke ``systemctl --user restart kiss-web`` so "
        "the systemd unit is restarted synchronously after the rm — without "
        "this, a Restart=always respawn that races with ``rm -f sorcar.sock`` "
        "leaves the daemon listening with no UDS file and the webview hangs."
    )

    rm_idx = src.index('rm -f "$HOME/.kiss/sorcar.sock"')
    restart_idx = src.index("systemctl --user restart kiss-web")
    marker_idx = src.index(MARKER_WRITE)
    assert rm_idx < restart_idx < marker_idx, (
        "systemctl --user restart must run AFTER the rm and BEFORE the "
        "marker write — same ordering rationale as the macOS kickstart."
    )


def test_install_script_kickstart_failures_do_not_abort() -> None:
    """The kickstart / systemctl-restart is BEST-EFFORT — a missing
    ``launchctl`` (CI / non-macOS) or a not-yet-loaded unit must NOT
    abort install.sh.  The in-extension ``decideRestart`` fallback handles
    those cases.  This pins the ``|| true`` suffix so future edits cannot
    silently turn the kickstart into a hard failure.
    """
    src = INSTALL_SCRIPT.read_text(encoding="utf-8")
    kickstart_block_start = src.index("launchctl kickstart -k")
    kickstart_block = src[kickstart_block_start:kickstart_block_start + 400]
    assert "|| true" in kickstart_block, (
        "launchctl kickstart must be best-effort (``|| true``) so a missing "
        "or unloaded LaunchAgent does not abort install.sh"
    )

    systemctl_idx = src.index("systemctl --user restart kiss-web")
    systemctl_block = src[systemctl_idx:systemctl_idx + 200]
    assert "|| true" in systemctl_block, (
        "systemctl --user restart must be best-effort (``|| true``) so a "
        "missing or unloaded unit does not abort install.sh on a CI host "
        "that has no systemd user instance"
    )
