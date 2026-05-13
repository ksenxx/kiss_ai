"""Integration tests for cloudflared cross-restart adoption.

Exercises the helpers added for the "stable quick-tunnel URL" feature
in :mod:`kiss.agents.vscode.web_server`:

* :func:`_is_pid_alive` — process liveness check using ``os.kill(pid, 0)``.
* :func:`_save_cloudflared_pidfile` / :func:`_load_cloudflared_pidfile`
  — atomic JSON round-trip for ``~/.kiss/cloudflared.pid``.
* :func:`_try_adopt_existing_cloudflared` — refuses to adopt a dead
  pid or a pid with no healthy metrics endpoint.
* :func:`_wait_for_remote_password` — polls the real
  ``~/.kiss/config.json`` until a non-empty password appears, falling
  back to ``""`` after *timeout* seconds (this is what eliminates the
  boot-time password race that was burning through quick-tunnel URLs).

Per the project test rules these are integration tests using real
subprocesses, real ``save_config``/``load_config`` against the
per-process ``KISS_HOME`` tempdir set up in
``src/kiss/tests/conftest.py``, and a real ``threading.Thread`` for
the "late write" scenario.  Only the module-level path constant
``web_server._CLOUDFLARED_PIDFILE`` is redirected via
``monkeypatch.setattr`` so that pidfile tests do not clobber the
user's real ``~/.kiss/cloudflared.pid``.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time

from kiss.agents.vscode import web_server as ws
from kiss.agents.vscode.vscode_config import (
    CONFIG_PATH,
    load_config,
    save_config,
)

# ---------------------------------------------------------------------------
# _is_pid_alive
# ---------------------------------------------------------------------------


def test_is_pid_alive_self() -> None:
    assert ws._is_pid_alive(os.getpid()) is True


def test_is_pid_alive_dead_pid() -> None:
    # Spawn and reap a child to get a pid that is guaranteed gone.
    proc = subprocess.Popen(["true"])
    proc.wait()
    time.sleep(0.05)  # let the kernel finalize the exit.
    assert ws._is_pid_alive(proc.pid) is False


def test_is_pid_alive_zero() -> None:
    assert ws._is_pid_alive(0) is False
    assert ws._is_pid_alive(-1) is False


# ---------------------------------------------------------------------------
# _save_cloudflared_pidfile / _load_cloudflared_pidfile
# ---------------------------------------------------------------------------


def test_save_and_load_pidfile_roundtrip(tmp_path, monkeypatch) -> None:
    pidfile = tmp_path / "cloudflared.pid"
    monkeypatch.setattr(ws, "_CLOUDFLARED_PIDFILE", pidfile)
    ws._save_cloudflared_pidfile(
        12345, 20240, "https://example.trycloudflare.com",
    )
    loaded = ws._load_cloudflared_pidfile()
    assert loaded is not None
    assert loaded["pid"] == 12345
    assert loaded["metrics_port"] == 20240
    assert loaded["url"] == "https://example.trycloudflare.com"


def test_save_pidfile_without_url(tmp_path, monkeypatch) -> None:
    pidfile = tmp_path / "cloudflared.pid"
    monkeypatch.setattr(ws, "_CLOUDFLARED_PIDFILE", pidfile)
    ws._save_cloudflared_pidfile(54321, 20241, None)
    loaded = ws._load_cloudflared_pidfile()
    assert loaded is not None
    assert loaded["pid"] == 54321
    assert loaded["metrics_port"] == 20241
    assert "url" not in loaded


def test_load_pidfile_missing_returns_none(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(ws, "_CLOUDFLARED_PIDFILE", tmp_path / "absent.pid")
    assert ws._load_cloudflared_pidfile() is None


def test_load_pidfile_malformed_returns_none(tmp_path, monkeypatch) -> None:
    pidfile = tmp_path / "bad.pid"
    pidfile.write_text("not json{")
    monkeypatch.setattr(ws, "_CLOUDFLARED_PIDFILE", pidfile)
    assert ws._load_cloudflared_pidfile() is None


def test_load_pidfile_no_pid_returns_none(tmp_path, monkeypatch) -> None:
    pidfile = tmp_path / "nopid.pid"
    pidfile.write_text(json.dumps({"metrics_port": 1234}))
    monkeypatch.setattr(ws, "_CLOUDFLARED_PIDFILE", pidfile)
    assert ws._load_cloudflared_pidfile() is None


def test_load_pidfile_non_dict_returns_none(tmp_path, monkeypatch) -> None:
    pidfile = tmp_path / "nondict.pid"
    pidfile.write_text(json.dumps([1, 2, 3]))
    monkeypatch.setattr(ws, "_CLOUDFLARED_PIDFILE", pidfile)
    assert ws._load_cloudflared_pidfile() is None


# ---------------------------------------------------------------------------
# _try_adopt_existing_cloudflared
# ---------------------------------------------------------------------------


def test_adopt_returns_none_when_pidfile_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(ws, "_CLOUDFLARED_PIDFILE", tmp_path / "missing.pid")
    assert ws._try_adopt_existing_cloudflared() is None


def test_adopt_returns_none_when_pid_dead(tmp_path, monkeypatch) -> None:
    pidfile = tmp_path / "cloudflared.pid"
    monkeypatch.setattr(ws, "_CLOUDFLARED_PIDFILE", pidfile)
    proc = subprocess.Popen(["true"])
    proc.wait()
    time.sleep(0.05)
    ws._save_cloudflared_pidfile(proc.pid, 20240, "https://x.example.com")
    assert ws._try_adopt_existing_cloudflared() is None


def test_adopt_returns_none_when_metrics_unhealthy(
    tmp_path, monkeypatch,
) -> None:
    """A live pid with no healthy metrics endpoint is not adopted.

    Uses a real long-running ``sleep`` subprocess so the pid is alive,
    but the saved ``metrics_port=1`` is a port nothing is listening on,
    so :func:`_probe_tunnel_ready` correctly returns ``False`` and the
    adoption attempt is refused.
    """
    pidfile = tmp_path / "cloudflared.pid"
    monkeypatch.setattr(ws, "_CLOUDFLARED_PIDFILE", pidfile)
    proc = subprocess.Popen(["sleep", "30"])
    try:
        ws._save_cloudflared_pidfile(proc.pid, 1, "https://x.example.com")
        assert ws._try_adopt_existing_cloudflared() is None
    finally:
        proc.terminate()
        proc.wait()


# ---------------------------------------------------------------------------
# _wait_for_remote_password (uses real config under KISS_HOME tempdir)
# ---------------------------------------------------------------------------


def _snapshot_config() -> dict | None:
    """Read the current config file (or ``None`` if it does not exist)."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    return None


def _restore_config(snapshot: dict | None) -> None:
    """Restore the config file to a previous snapshot."""
    if snapshot is None:
        if CONFIG_PATH.exists():
            CONFIG_PATH.unlink()
    else:
        CONFIG_PATH.write_text(json.dumps(snapshot, indent=2))


def test_wait_for_remote_password_immediate() -> None:
    """When the password is already on disk, return almost instantly."""
    snapshot = _snapshot_config()
    try:
        save_config({"remote_password": "secret"})
        assert load_config()["remote_password"] == "secret"
        t0 = time.monotonic()
        pw = ws._wait_for_remote_password(timeout=5.0)
        elapsed = time.monotonic() - t0
        assert pw == "secret"
        assert elapsed < 0.5
    finally:
        _restore_config(snapshot)


def test_wait_for_remote_password_times_out() -> None:
    """When no password ever appears, return ``""`` after ``timeout``."""
    snapshot = _snapshot_config()
    try:
        save_config({"remote_password": ""})
        t0 = time.monotonic()
        pw = ws._wait_for_remote_password(timeout=1.0)
        elapsed = time.monotonic() - t0
        assert pw == ""
        # Allow a generous upper bound for slow CI machines, but the
        # function must have waited at least the full timeout.
        assert 1.0 <= elapsed < 3.0
    finally:
        _restore_config(snapshot)


def test_wait_for_remote_password_picks_up_late_write() -> None:
    """A password written ~1s after the call begins must be returned."""
    snapshot = _snapshot_config()
    try:
        save_config({"remote_password": ""})

        def write_later() -> None:
            time.sleep(1.0)
            save_config({"remote_password": "set-late"})

        thread = threading.Thread(target=write_later, daemon=True)
        thread.start()
        try:
            t0 = time.monotonic()
            pw = ws._wait_for_remote_password(timeout=5.0)
            elapsed = time.monotonic() - t0
            assert pw == "set-late"
            # Must have waited until at least the late write fired (~1s),
            # but well under the 5s timeout.
            assert 0.9 < elapsed < 3.0
        finally:
            thread.join(timeout=2.0)
    finally:
        _restore_config(snapshot)
