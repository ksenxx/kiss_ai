# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: a stuck ``ps``, ``osascript`` or ``open`` cannot hang the daemon.

:mod:`kiss.server.standalone_app` runs those commands from the daemon's
watchdog sweep, from the popup's close handler and before it replaces
a stale brand bundle.  Each call is bounded by a timeout and answers
"cannot tell" when the command does not come back; a stub on ``PATH``
that sleeps stands in for the stuck command.  "Cannot tell" from ``ps``
is not "nothing runs": the bundle replacement, which deletes the
executables a running window may still use, is refused then.
"""

from __future__ import annotations

import json
import os
import signal
import stat
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from kiss.core.brand import BRAND
from kiss.core.processes import pid_alive
from kiss.server import standalone_app
from kiss.tests.conftest import posix_only

pytestmark = posix_only("stub commands are sh scripts")

TIMEOUT = 0.5
"""How long each probe gets before it gives up on its stuck stub."""


def _hanging_commands(bin_dir: Path, *names: str) -> None:
    """Put a stub named each of *names* into *bin_dir* that marks its start, then sleeps.

    The stub ``exec``s its sleep, so the sleep is the process the probe
    kills at its timeout (a sleep forked by the stub would outlive it);
    ``<stub>.started`` records that the stub did run, ``<stub>.pid`` its
    (and so the sleep's) pid.
    """
    bin_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        stub = bin_dir / name
        stub.write_text(f'#!/bin/sh\necho $$ > "{stub}.pid"\n: > "{stub}.started"\nexec sleep 60\n')
        stub.chmod(stub.stat().st_mode | stat.S_IXUSR)


def _stuck_probe(
    bin_dir: Path, name: str, probe: Callable[..., Any], *args: Any, **kwargs: Any
) -> Any:
    """Call ``probe(*args, **kwargs)`` against the stuck stub *name*; return its result.

    Asserts that the stub ran and that the probe waited its whole timeout
    for it, so a stub that failed to start (which also yields the
    "cannot tell" value) does not pass for a stuck one.
    """
    started = time.monotonic()
    result = probe(*args, **kwargs)
    elapsed = time.monotonic() - started
    assert (bin_dir / f"{name}.started").exists(), f"the {name} stub never ran"
    assert TIMEOUT <= elapsed < 5.0, f"{name}: waited {elapsed:.2f}s for a {TIMEOUT}s timeout"
    return result


@pytest.fixture
def stuck_bin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Stuck ``ps``, ``osascript`` and ``open`` first on ``PATH``."""
    bin_dir = tmp_path / "bin"
    _hanging_commands(bin_dir, "ps", "osascript", "open")
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return bin_dir


def test_stuck_probes_time_out(stuck_bin: Path, tmp_path: Path) -> None:
    """Each probe returns its "cannot tell" value at its timeout instead of hanging."""
    bundle = tmp_path / "Applications" / "Brand.app"
    processes = _stuck_probe(
        stuck_bin, "ps", standalone_app.macos_bundle_processes, bundle, timeout=TIMEOUT
    )
    assert processes is None
    windows = _stuck_probe(
        stuck_bin, "osascript", standalone_app.macos_windows_on_screen, 4242, timeout=TIMEOUT
    )
    assert windows is None
    activated = _stuck_probe(
        stuck_bin, "open", standalone_app.macos_activate_running, bundle, timeout=TIMEOUT
    )
    assert activated is False


def test_unanswering_ps_is_not_nothing_running(
    stuck_bin: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stale bundle is kept when ``ps`` cannot say whether a window runs from it.

    The watchdog treats the same answer as "no candidate to quit".
    Once ``ps`` answers again the bundle is replaced.
    """
    monkeypatch.setattr(standalone_app, "PROBE_TIMEOUT_SECS", TIMEOUT)
    apps = tmp_path / "Applications"
    bundle = apps / f"{BRAND['product_name']}.app"
    exe_copy = bundle / "Contents" / "MacOS" / "Chromium"
    exe_copy.parent.mkdir(parents=True)
    exe_copy.write_text("#!/bin/sh\n")
    (bundle / "Contents" / "Frameworks").mkdir()
    icon, exe, state = tmp_path / "icon.png", tmp_path / "kiss-app", tmp_path / "state"

    assert standalone_app.macos_bundle_in_use(bundle) is None
    assert standalone_app.macos_bundle_pid(bundle) is None
    started = time.monotonic()
    assert standalone_app.write_macos_launcher(apps, BRAND, icon, exe, state, None) is None
    assert time.monotonic() - started >= TIMEOUT
    assert exe_copy.exists() and (bundle / "Contents" / "Frameworks").is_dir()

    (stuck_bin / "ps").unlink()  # the real ``ps`` answers again
    assert standalone_app.macos_bundle_in_use(bundle) is False
    assert standalone_app.write_macos_launcher(apps, BRAND, icon, exe, state, None) is None
    assert not exe_copy.exists() and not (bundle / "Contents" / "Frameworks").exists()
    assert (bundle / "Contents" / "MacOS" / "launcher").exists()


def test_a_wait_that_never_elapses_is_a_usage_error(tmp_path: Path) -> None:
    """``--wait nan`` and ``--wait inf`` are refused instead of waited out.

    A NaN deadline is never reached and an infinite one never comes:
    with no URL file in the state directory ``kiss-app`` polled for it
    for good.  The command runs in a child, which this test's own
    deadline kills if it does.
    """
    env = {**os.environ, "KISS_HOME": str(tmp_path)}  # no remote-url.json there
    for wait in ("nan", "inf"):
        refused = subprocess.run(
            [sys.executable, "-m", "kiss.server.standalone_app", "--wait", wait],
            capture_output=True, text=True, env=env, timeout=120,
        )
        assert refused.returncode == 2, refused.stderr
        assert "--wait: SECS must be a finite number" in refused.stderr


def test_silent_playwright_driver_cannot_hang_the_browser_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``playwright_chromium`` is None at its timeout when the driver stays silent; the driver dies.

    Playwright's executable-path lookup starts its node driver and waits
    for the driver's handshake with no deadline of its own, and
    ``kiss-app`` asks it before opening any window (``find_browser``,
    ``sandboxed``), so a driver that starts and never answers would park
    the app for good. The real Node driver is stopped by a preload
    module before its handshake; the product timeout is unchanged. An
    invalid Node option then exercises an immediate driver failure.
    The lookup is cached per process:
    cleared before each question here and at the end, so the tests that
    follow ask the real driver again.
    """
    pid_file = tmp_path / "node.pid"
    preload = tmp_path / "stop-node.cjs"
    preload.write_text(
        f"require('node:fs').writeFileSync({json.dumps(str(pid_file))}, String(process.pid));\n"
        "process.kill(process.pid, 'SIGSTOP');\n", encoding="utf-8",
    )
    monkeypatch.delenv("PLAYWRIGHT_NODEJS_PATH", raising=False)
    monkeypatch.setenv("NODE_OPTIONS", f"--require={json.dumps(str(preload))}")
    timeout = standalone_app.PLAYWRIGHT_TIMEOUT_SECS
    lookup = standalone_app.playwright_chromium
    lookup.cache_clear()
    try:
        started = time.monotonic()
        assert lookup() is None
        elapsed = time.monotonic() - started
        assert pid_file.exists(), "the real Node driver never ran"
        assert timeout <= elapsed < timeout + 15.0, f"waited {elapsed:.2f}s for {timeout}s"
        # The stuck driver went with the lookup that gave up on it.
        pid = int(pid_file.read_text())
        deadline = time.monotonic() + 5.0
        while pid_alive(pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not pid_alive(pid), "the stuck driver outlived the lookup"

        monkeypatch.setenv("NODE_OPTIONS", "--invalid-kiss-test-option")
        lookup.cache_clear()
        started = time.monotonic()
        assert lookup() is None
        elapsed = time.monotonic() - started
        assert elapsed < timeout, f"a dead driver took {elapsed:.2f}s to give up on"
    finally:
        lookup.cache_clear()
        if pid_file.exists() and pid_alive(int(pid_file.read_text())):
            os.kill(int(pid_file.read_text()), signal.SIGKILL)
