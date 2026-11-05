# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the 2026-11-01 audit of :mod:`kiss.server.standalone_app` (V7).

1. A ``ps`` that exits with an error is "cannot tell", like one that
   does not answer: it must not pass for "nothing runs from the
   bundle", which authorizes deleting the executables a window may
   still run from.
2. The commands that build the launcher (``sips``, ``iconutil``,
   ``cp``, ``codesign``, ``update-desktop-database``) are bounded: a
   stuck one is given up on after :data:`SETUP_TIMEOUT_SECS` instead of
   parking the launch for good.
3. Two icon conversions at once (two ``--install-launcher`` runs, a
   first build next to one) each get a scratch directory of their own:
   with the shared ``AppIcon.iconset`` one's cleanup deleted the
   other's inputs.
4. Two launcher writers at once (V9) are serialized: one that listed
   the processes before the other built the clone and a window started
   from it deleted that clone, executables and all, under the window.

The tools are stubs on ``PATH``; the files, threads and processes are
real.  The macOS paths run on Linux too: nothing in them is platform
specific but the tools' names.
"""

from __future__ import annotations

import os
import plistlib
import shutil
import stat
import subprocess
import threading
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
"""How long each setup command gets before the launcher gives up on it."""


def _stub(bin_dir: Path, name: str, body: str) -> Path:
    """Put an executable ``sh`` script *name* with *body* into *bin_dir*."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    stub = bin_dir / name
    stub.write_text("#!/bin/sh\n" + body)
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
    return stub


def _hanging_stub(bin_dir: Path, name: str) -> Path:
    """A stub that records its pid in ``<stub>.started`` and then sleeps in place."""
    stub = bin_dir / name
    return _stub(bin_dir, name, f'echo $$ > "{stub}.started"\nexec sleep 60\n')


def _alive(pid: int) -> bool:
    """Whether a process with *pid* exists (a zombie counts as alive)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_until(condition: Callable[[], bool], seconds: float = 10.0) -> bool:
    """Poll *condition* for up to *seconds*; True once it holds."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return condition()


def _call_bounded(stub: Path, fn: Callable[..., Any], *args: Any) -> Any:
    """Call ``fn(*args)`` on a thread against the stuck *stub*; return its result.

    Asserts that the stub ran and that the call came back within a few
    seconds of :data:`TIMEOUT`.  Whatever happens, the stuck stub is
    killed, so a call that never returns (the behaviour under test)
    leaves no sleeper behind.
    """
    results: list[Any] = []
    thread = threading.Thread(target=lambda: results.append(fn(*args)), daemon=True)
    started = time.monotonic()
    thread.start()
    try:
        thread.join(TIMEOUT + 5.0)
        elapsed = time.monotonic() - started
        marker = Path(f"{stub}.started")
        assert marker.exists(), f"the {stub.name} stub never ran"
        assert not thread.is_alive(), f"{stub.name}: still waiting after {elapsed:.1f}s"
        assert elapsed >= TIMEOUT, f"{stub.name}: came back after {elapsed:.2f}s"
        pid = int(marker.read_text().split()[0])
        assert _wait_until(lambda: not _alive(pid), 5.0), f"the {stub.name} stub outlived the call"
    finally:
        marker = Path(f"{stub}.started")
        if marker.exists():
            try:
                os.kill(int(marker.read_text().split()[0]), 9)
            except (OSError, ValueError):
                pass
    return results[0]


@pytest.fixture
def bin_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An empty directory first on ``PATH`` for the stubs."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return bin_dir


@pytest.fixture
def short_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each setup command gets :data:`TIMEOUT` instead of the real deadline.

    For the stuck-command tests only: the conversion race below holds
    a ``sips`` run until its coordinator has scheduled the other
    conversion, and on the real deadline a slow scheduler cannot time
    the held run out.
    """
    monkeypatch.setattr(standalone_app, "SETUP_TIMEOUT_SECS", TIMEOUT, raising=False)


def test_failed_ps_is_not_nothing_running(bin_dir: Path, tmp_path: Path) -> None:
    """A ``ps`` that exits with an error leaves a stale bundle alone, like one that hangs."""
    _stub(bin_dir, "ps", 'echo "ps: cannot enumerate processes" >&2\nexit 1\n')
    apps = tmp_path / "Applications"
    bundle = apps / f"{BRAND['product_name']}.app"
    exe_copy = bundle / "Contents" / "MacOS" / "Chromium"
    exe_copy.parent.mkdir(parents=True)
    exe_copy.write_text("#!/bin/sh\n")
    frameworks = bundle / "Contents" / "Frameworks"
    frameworks.mkdir()
    (frameworks / "helper").write_text("live app helper")
    icon, exe, state = tmp_path / "icon.png", tmp_path / "kiss-app", tmp_path / "state"

    assert standalone_app.macos_bundle_processes(bundle) is None
    assert standalone_app.macos_bundle_in_use(bundle) is None
    assert standalone_app.macos_bundle_pid(bundle) is None
    assert standalone_app.write_macos_launcher(apps, BRAND, icon, exe, state, None) is None
    assert exe_copy.exists() and (frameworks / "helper").exists()
    assert not (bundle / "Contents" / "MacOS" / "launcher").exists()

    (bin_dir / "ps").unlink()  # the real ``ps`` answers again: nothing runs from the bundle
    assert standalone_app.macos_bundle_in_use(bundle) is False
    assert standalone_app.write_macos_launcher(apps, BRAND, icon, exe, state, None) is None
    assert not exe_copy.exists() and not frameworks.exists()
    assert (bundle / "Contents" / "MacOS" / "launcher").exists()


@pytest.mark.usefixtures("short_deadline")
def test_stuck_icon_converter_times_out(bin_dir: Path, tmp_path: Path) -> None:
    """A ``sips`` that never returns makes the icon conversion fail instead of hang."""
    sips = _hanging_stub(bin_dir, "sips")
    _stub(bin_dir, "iconutil", "exit 0\n")
    icns = tmp_path / "Resources" / "AppIcon.icns"
    icns.parent.mkdir()
    assert _call_bounded(sips, standalone_app.write_icns, tmp_path / "icon.png", icns) is False
    assert not icns.exists()
    assert list(icns.parent.iterdir()) == [], "the conversion left its scratch directory behind"


@pytest.mark.usefixtures("short_deadline")
def test_stuck_browser_clone_times_out(bin_dir: Path, tmp_path: Path) -> None:
    """A ``cp`` that never returns makes the clone fail, with nothing left under the bundle."""
    cp = _hanging_stub(bin_dir, "cp")
    browser = tmp_path / "Chromium.app" / "Contents" / "MacOS" / "Chromium"
    browser.parent.mkdir(parents=True)
    browser.write_text("browser")
    (browser.parent.parent / "Frameworks").mkdir()
    contents = tmp_path / "Applications" / "Brand.app" / "Contents"
    clone = _call_bounded(
        cp, standalone_app.clone_macos_browser, contents, browser, "org.example.brand"
    )
    assert clone is None
    assert not (contents / "Frameworks").exists()
    assert not (contents / "MacOS" / "Chromium").exists()


@pytest.mark.usefixtures("short_deadline")
def test_stuck_desktop_database_refresh_times_out(bin_dir: Path, tmp_path: Path) -> None:
    """An ``update-desktop-database`` that never returns does not hold the Linux install."""
    refresh = _hanging_stub(bin_dir, "update-desktop-database")
    home = tmp_path / "home"
    entry = _call_bounded(
        refresh,
        lambda: standalone_app.install_launcher(
            home=home,
            exe=tmp_path / "kiss-app",
            state_dir=tmp_path / "state",
            browser=tmp_path / "chromium",
            icon=tmp_path / "icon.png",
            system="Linux",
        ),
    )
    assert entry == standalone_app.linux_launcher_path(home)
    assert entry.is_file()


# ``sips -z H W <png> --out <out>``: waits for ``<png>.gate`` (so the
# test schedules each conversion's steps), then writes the output;
# ``iconutil -c icns <iconset> -o <icns>``: needs all 12 sizes present.
_GATED_SIPS = (
    'for last; do :; done\n: > "$4.started"\n'
    'while [ ! -e "$4.gate" ]; do sleep 0.01; done\n'
    'printf icon > "$last" || exit 1\n'
)
_ICONUTIL = '[ "$(ls "$3" 2>/dev/null | wc -l)" -eq 12 ] || exit 1\nprintf icns > "$5"\n'


def test_concurrent_icon_conversions_keep_their_scratch_directories(
    bin_dir: Path, tmp_path: Path
) -> None:
    """Two conversions to the same ``.icns`` at once both succeed.

    The first is held at its first ``sips`` while the second runs to
    completion; the first then finishes.  With one shared
    ``AppIcon.iconset`` the second's cleanup deleted the directory the
    first was still converting into.
    """
    _stub(bin_dir, "sips", _GATED_SIPS)
    _stub(bin_dir, "iconutil", _ICONUTIL)
    icns = tmp_path / "Resources" / "AppIcon.icns"
    icns.parent.mkdir()
    first, second = tmp_path / "first.png", tmp_path / "second.png"
    results: dict[str, bool] = {}
    errors: list[BaseException] = []

    def convert(png: Path) -> None:
        try:
            results[png.name] = standalone_app.write_icns(png, icns)
        except BaseException as exc:  # noqa: BLE001 - reported by the assertions below
            errors.append(exc)

    threads = [threading.Thread(target=convert, args=(png,)) for png in (first, second)]
    try:
        threads[0].start()
        assert _wait_until(Path(f"{first}.started").exists), "the first conversion never ran sips"
        threads[1].start()
        assert _wait_until(Path(f"{second}.started").exists), "the second conversion never ran"
        Path(f"{second}.gate").touch()
        threads[1].join(10)
        assert not threads[1].is_alive()
        Path(f"{first}.gate").touch()
        threads[0].join(10)
        assert not threads[0].is_alive()
    finally:
        # A failed step leaves no stub waiting at its gate.
        for png in (first, second):
            Path(f"{png}.gate").touch()

    assert errors == []
    assert results == {"first.png": True, "second.png": True}
    assert icns.read_text() == "icns"
    assert list(icns.parent.iterdir()) == [icns], "a scratch directory was left behind"


# ``ps``: a slow one.  Lists the processes at once (the real ``ps``),
# then delivers the listing only once ``<stub>.gate`` exists.
_SLOW_PS = (
    'listing=$(PATH=/usr/bin:/bin ps "$@")\n: > "$0.started"\n'
    'while [ ! -e "$0.gate" ]; do sleep 0.01; done\n'
    'printf "%s\\n" "$listing"\n'
)


def _start_window(exe_copy: Path) -> tuple[subprocess.Popen[bytes], int]:
    """Run the clone's executable (a copy of ``sleep``); it and the executable's inode."""
    window = subprocess.Popen(
        [str(exe_copy), "60"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return window, exe_copy.stat().st_ino


def _browser_bundle(root: Path, version: str) -> Path:
    """A browser bundle of *version* under *root*: a copy of ``sleep`` and a versioned helper."""
    browser = root / f"Chromium-{version}.app" / "Contents" / "MacOS" / "Chromium"
    browser.parent.mkdir(parents=True)
    # copy, not copy2: macOS marks /bin/sleep "restricted", a file flag
    # copystat may not set (EPERM).
    shutil.copy(shutil.which("sleep") or "/bin/sleep", browser)
    (browser.parents[1] / "Frameworks").mkdir()
    (browser.parents[1] / "Frameworks" / f"helper-v{version}").write_text(f"helper {version}")
    with (browser.parents[1] / "Info.plist").open("wb") as f:
        plistlib.dump({"CFBundleVersion": version, "CFBundleShortVersionString": version}, f)
    return browser


def test_concurrent_launcher_writers_keep_the_clone_a_window_runs_from(
    bin_dir: Path, tmp_path: Path
) -> None:
    """Two writers at once: a stale process listing cannot delete the other's live clone.

    Writer B lists the processes (nothing runs from the bundle) and is
    held in that ``ps``; writer A, told nothing ran, starts meanwhile.
    Serialized, A waits for B, which builds the clone, and keeps it.
    Unserialized, A built the clone while B's listing was in flight, a
    window started from it, and B then deleted it under the window.
    """
    ps = _stub(bin_dir, "ps", _SLOW_PS)
    browser = _browser_bundle(tmp_path, "1")
    apps = tmp_path / "Applications"
    bundle = apps / f"{BRAND['product_name']}.app"
    exe_copy = bundle / "Contents" / "MacOS" / "Chromium"
    icon, exe, state = tmp_path / "icon.png", tmp_path / "kiss-app", tmp_path / "state"
    results: dict[str, Path | None] = {}
    errors: list[BaseException] = []

    def write(name: str, in_use: bool | None) -> None:
        try:
            results[name] = standalone_app.write_macos_launcher(
                apps, BRAND, icon, exe, state, browser, in_use
            )
        except BaseException as exc:  # noqa: BLE001 - reported by the assertions below
            errors.append(exc)

    writer_b = threading.Thread(target=write, args=("B", None))
    writer_a = threading.Thread(target=write, args=("A", False))
    window, inode = None, 0
    try:
        writer_b.start()
        assert _wait_until(Path(f"{ps}.started").exists), "writer B never listed the processes"
        writer_a.start()
        writer_a.join(1.0)
        if not writer_a.is_alive():  # not serialized: the window starts from A's clone
            window, inode = _start_window(exe_copy)
        Path(f"{ps}.gate").touch()
        for writer in (writer_b, writer_a):
            writer.join(10)
            assert not writer.is_alive()
        if window is None:
            window, inode = _start_window(exe_copy)
        assert errors == []
        assert results == {"A": exe_copy, "B": exe_copy}
        assert window.poll() is None
        assert exe_copy.stat().st_ino == inode, "the executable the window runs from was replaced"
        assert (bundle / "Contents" / "Frameworks" / "helper-v1").is_file()
    finally:
        Path(f"{ps}.gate").touch()
        if window is not None:
            window.kill()
            window.wait(10)


def test_a_stale_not_in_use_hint_cannot_delete_a_live_clone_of_another_browser(
    tmp_path: Path,
) -> None:
    """A "nothing runs" listing taken before the clone was built does not authorize deleting it.

    ``main()`` lists the processes once, before the writer's lock.  Writer
    A then builds the clone and a window starts from it; writer B, given
    that stale listing, targets an updated browser, so A's clone is not
    current for it.  The clone is nonetheless the one the window runs
    from: B must find that out again under the lock and leave it alone.
    """
    apps = tmp_path / "Applications"
    bundle = apps / f"{BRAND['product_name']}.app"
    icon, exe, state = tmp_path / "icon.png", tmp_path / "kiss-app", tmp_path / "state"
    stale_hint = standalone_app.macos_bundle_in_use(bundle)
    assert stale_hint is False
    clone = standalone_app.write_macos_launcher(
        apps, BRAND, icon, exe, state, _browser_bundle(tmp_path, "1")
    )
    assert clone is not None
    window, inode = _start_window(clone)
    try:
        assert _wait_until(lambda: standalone_app.macos_bundle_in_use(bundle) is True)
        updated = _browser_bundle(tmp_path, "2")
        assert standalone_app.write_macos_launcher(
            apps, BRAND, icon, exe, state, updated, in_use=stale_hint
        ) is None
        assert window.poll() is None
        assert clone.stat().st_ino == inode, "the executable the window runs from was replaced"
        assert (bundle / "Contents" / "Frameworks" / "helper-v1").is_file()
        assert standalone_app.macos_bundle_browser(bundle, updated) is None
    finally:
        window.kill()
        window.wait(10)
