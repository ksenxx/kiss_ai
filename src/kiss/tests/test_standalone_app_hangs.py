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

import os
import stat
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from kiss.core.brand import BRAND
from kiss.server import standalone_app
from kiss.tests.conftest import posix_only

pytestmark = posix_only("stub commands are sh scripts")

TIMEOUT = 0.5
"""How long each probe gets before it gives up on its stuck stub."""


def _hanging_commands(bin_dir: Path, *names: str) -> None:
    """Put a stub named each of *names* into *bin_dir* that marks its start, then sleeps.

    The stub ``exec``s its sleep, so the sleep is the process the probe
    kills at its timeout (a sleep forked by the stub would outlive it);
    ``<stub>.started`` records that the stub did run.
    """
    bin_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        stub = bin_dir / name
        stub.write_text(f'#!/bin/sh\n: > "{stub}.started"\nexec sleep 60\n')
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
        stuck_bin, "open", standalone_app.macos_activate_running, bundle, in_use=True,
        timeout=TIMEOUT,
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
