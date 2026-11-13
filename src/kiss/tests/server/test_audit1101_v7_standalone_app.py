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
5. A clone is launched under the writers' lock, held shared: a writer
   that listed the processes just before the launch deleted it too.
   A launch waits for no launch, and starts nothing in a clone's place:
   not the browser itself, nor another browser's clone left at its path.
6. ``uninstall_launcher`` removes the macOS bundle as a writer replaces
   it: under that lock, and never while a window runs from it.
7. The watchdog's sweep counts its strikes anew once the popup has
   connected again: the strikes from before quit a minimized popup's
   browser at the first sweep after its connection dropped.

The tools are stubs on ``PATH`` (none in 5. and 6.: the real ``ps``, a writer
held with ``sys.setprofile``); the files, threads and processes are
real.  The macOS paths run on Linux too: nothing in them is platform
specific but the tools' names.  The clones are macOS code exercised on
Linux with copies of ``sleep`` for the browser: which executable a launch
starts shows here, the Dock icon macOS gives its window does not.
"""

from __future__ import annotations

import functools
import os
import platform
import plistlib
import shutil
import signal
import stat
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from types import FrameType
from typing import Any

import pytest

from kiss.core.brand import BRAND
from kiss.core.file_lock import exclusive_file_lock
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


def test_desktop_database_refresh_that_cannot_start_keeps_the_install(
    bin_dir: Path, tmp_path: Path
) -> None:
    """An ``update-desktop-database`` that is no program is skipped: the entry is written."""
    refresh = bin_dir / "update-desktop-database"
    refresh.write_bytes(b"\x00\x01 neither a binary nor a script")
    refresh.chmod(0o755)
    home = tmp_path / "home"
    entry = standalone_app.install_launcher(
        home=home,
        exe=tmp_path / "kiss-app",
        state_dir=tmp_path / "state",
        browser=tmp_path / "chromium",
        icon=tmp_path / "icon.png",
        system="Linux",
    )
    launcher = standalone_app.linux_launcher_path(home)
    assert entry == launcher
    assert launcher.is_file()


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
    """Run the clone's executable (a copy of ``sleep``); it and its inode."""
    window = subprocess.Popen(
        [str(exe_copy), "60"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return window, exe_copy.stat().st_ino


def _copy_sleep(executable: Path) -> None:
    """Write a runnable copy of ``sleep`` to *executable*."""
    shutil.copy(shutil.which("sleep") or "/bin/sleep", executable)
    if sys.platform == "darwin":
        # macOS kills a relocated Apple-signed executable unless it is signed for its new path.
        subprocess.run(["codesign", "--force", "--sign", "-", executable], check=True)


def _browser_bundle(root: Path, version: str, name: str = "Chromium") -> Path:
    """A browser bundle of *version* with a sleeping executable and versioned helper."""
    browser = root / f"{name}-{version}.app" / "Contents" / "MacOS" / name
    browser.parent.mkdir(parents=True)
    _copy_sleep(browser)
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


# A child that launches a clone: in the foreground (``exec`` replaces it with the clone), or,
# given a third argument, detached and stopped where it calls ``start_detached``, its lock
# held.  Its command line names no bundle: the process listing takes it for no window.
_LAUNCH = (
    "import os, signal, sys; from kiss.server import standalone_app as a\n"
    "home = a.Path(sys.argv[1]); browser, log = home / sys.argv[2], home / 'log'\n"
    "clone = a.macos_bundle_path(home) / 'Contents' / 'MacOS' / browser.name\n"
    "def stop(frame, event, arg):\n"
    "    if event == 'call' and frame.f_code is a.start_detached.__code__:\n"
    "        os.kill(os.getpid(), signal.SIGSTOP)\n"
    "sys.setprofile(stop if sys.argv[3:] else None)\n"
    "a.launch_app([str(clone), '300'], log, browser, foreground=not sys.argv[3:], home=home)\n"
)

# A child the kernel refuses flock(2), with ENOLCK as a file system without locks does (a
# seccomp filter; its numbers are those of x86-64 Linux): what launching the clone returned,
# and the descriptors it kept on the lock file.
_LAUNCH_WITHOUT_FLOCK = (
    "import ctypes, os, sys; from kiss.server import standalone_app as a\n"
    "home = a.Path(sys.argv[1]); browser, bundle = home / sys.argv[2], a.macos_bundle_path(home)\n"
    "class Rule(ctypes.Structure):\n"
    "    _fields_ = [('code', ctypes.c_ushort), ('jt', ctypes.c_ubyte), ('jf', ctypes.c_ubyte),\n"
    "                ('k', ctypes.c_uint)]\n"
    "class Program(ctypes.Structure):\n"
    "    _fields_ = [('len', ctypes.c_ushort), ('filter', ctypes.POINTER(Rule))]\n"
    "rules = (Rule * 4)(Rule(0x20, 0, 0, 0), Rule(0x15, 0, 1, 73),  # the call is flock:\n"
    "                   Rule(0x06, 0, 0, 0x00050000 | 37),  # fail with ENOLCK\n"
    "                   Rule(0x06, 0, 0, 0x7fff0000))  # any other call: allowed\n"
    "program, libc = Program(4, rules), ctypes.CDLL(None, use_errno=True)\n"
    "assert libc.prctl(38, 1, 0, 0, 0) == 0  # no new privileges, which a filter requires\n"
    "assert libc.prctl(22, 2, ctypes.byref(program), 0, 0) == 0\n"
    "clone = bundle / 'Contents' / 'MacOS' / browser.name\n"
    "started = a.launch_app(\n"
    "    [str(clone), '0'], home / 'log', browser, foreground=False, home=home)\n"
    "lock = str(bundle.parent / f'.{bundle.name}.lock')\n"
    "fds = [f'/proc/self/fd/{fd}' for fd in os.listdir('/proc/self/fd')]\n"
    "print(started, [fd for fd in fds if os.path.realpath(fd) == lock])\n"
)

# A child interrupted while it waits for a writer's lock: whether it kept no descriptor open.
_INTERRUPTED_WAIT = (
    "import os, signal, sys; from kiss.server import standalone_app as a\n"
    "signal.signal(signal.SIGALRM, signal.default_int_handler)\n"
    "kept = os.listdir('/dev/fd')\n"
    "try:\n"
    "    signal.setitimer(signal.ITIMER_REAL, 0.5)\n"
    "    a._lock_shared(a.Path(sys.argv[1]))\n"
    "except KeyboardInterrupt:\n"
    "    print(os.listdir('/dev/fd') == kept)\n"
)


def _clone_state(executable: Path) -> tuple[int, list[str]]:
    """The inode of a bundle's *executable* and the names of the helpers next to it."""
    return executable.stat().st_ino, os.listdir(executable.parents[1] / "Frameworks")


class _Steps:
    """``sys.setprofile`` hooks: the real functions run, the hooks wait and take notes.

    ``writer`` holds its thread where the process listing returns, under
    the writer's lock, until ``resume``.  ``launcher`` sets ``reached``
    once its thread asks for a file lock or has started its window, and
    keeps in ``started`` the :func:`_clone_state` it started that from.
    """

    def __init__(self) -> None:
        self.listed, self.resume, self.reached = (threading.Event() for _ in range(3))
        self.started: tuple[int, list[str]] | None = None

    def writer(self, frame: FrameType, event: str, arg: object) -> None:
        if event == "return" and frame.f_code is standalone_app.macos_bundle_in_use.__code__:
            self.listed.set()
            self.resume.wait(60)

    def launcher(self, frame: FrameType, event: str, arg: object) -> None:
        if event == "c_call" and getattr(arg, "__name__", "") == "flock":
            self.reached.set()
        elif event == "call" and frame.f_code is standalone_app.start_detached.__code__:
            self.started = _clone_state(Path(frame.f_locals["command"][0]))
        elif event == "return" and frame.f_code is standalone_app.start_detached.__code__:
            self.reached.set()


@pytest.mark.parametrize(
    "case", ["updated browser", "KISS_APP_BROWSER", "other browser", "read-only lock"]
)
def test_a_writers_process_listing_cannot_delete_the_clone_a_launch_starts(
    case: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A writer's list-then-delete and the launch of a clone exclude each other.

    Writer B (another browser) has listed the processes, none, under
    its lock when launch A starts the clone it chose earlier: B then
    deleted the executables under A's new window.  Now A waits for B
    and starts the clone of its own browser, which it rebuilds: what B
    left is another browser's, also under the clone's name ("updated
    browser": A started it there).  Where ``$KISS_APP_BROWSER`` names
    the clone itself (also behind a lock file that became read-only
    after B opened it), what B left in its place is A's to start.
    """
    apps = tmp_path / "Applications"
    bundle = apps / f"{BRAND['product_name']}.app"
    icon, exe, state = tmp_path / "icon.png", tmp_path / "kiss-app", tmp_path / "state"
    first = _browser_bundle(tmp_path, "1")
    other = _browser_bundle(tmp_path, "2", "Brave" if case == "other browser" else "Chromium")
    clone = standalone_app.write_macos_launcher(apps, BRAND, icon, exe, state, first)
    assert clone is not None
    named = case in ("KISS_APP_BROWSER", "read-only lock")  # the clone is the browser to start
    if named:
        monkeypatch.setenv("KISS_APP_BROWSER", str(clone))
    engine = standalone_app.find_browser() if named else first
    assert engine is not None
    assert standalone_app.macos_launch_browser(engine, home=tmp_path) == clone
    steps = _Steps()
    results: dict[str, object] = {}

    def write() -> None:
        sys.setprofile(steps.writer)
        results["B"] = standalone_app.write_macos_launcher(apps, BRAND, icon, exe, state, other)

    def launch() -> None:
        sys.setprofile(steps.launcher)
        results["A"] = standalone_app.launch_app(
            [str(clone), "60"], tmp_path / "kiss-app.log", engine, foreground=False, home=tmp_path
        )

    writer, launcher = threading.Thread(target=write), threading.Thread(target=launch)
    try:
        writer.start()
        assert steps.listed.wait(60), "writer B never listed the processes"
        if case == "read-only lock":
            (apps / f".{bundle.name}.lock").chmod(0o400)
        launcher.start()
        assert steps.reached.wait(60), "launch A neither waited for the lock nor started"
        steps.resume.set()
        writer.join(180)
        launcher.join(180)
        assert results == {"A": True, "B": clone.with_name(other.name)}
        assert standalone_app.macos_bundle_pid(bundle), "the window is not the brand bundle's"
        assert _clone_state(clone) == steps.started, "B replaced the clone under A's window"
        helper = "helper-v2" if named else "helper-v1"
        assert _clone_state(clone)[1] == [helper], "A started a browser it had not chosen"
    finally:
        steps.resume.set()
        subprocess.run(["pkill", "-f", f"^{tmp_path}/"], check=False)


# ``main()`` calls :func:`launch_app` on macOS only: its exit status for "nothing was
# started" cannot be reached on another platform without a test double for the platform.
def test_a_launch_that_chose_the_clone_starts_it_or_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A launch that chose the clone never starts the browser itself in its place.

    A writer's lock still held at the deadline ends the launch with a
    message (the browser itself, no file of the bundle, starts without
    the lock).  The clone starts as chosen, refreshed first when the
    browser was updated in place, and in the foreground ``exec`` releases
    the lock.  A clone that is another browser's by now, gone or still at
    its path, cannot be rebuilt while that is in use, or a stopped launch
    holds the lock: the launch ends too, as does an uninstall then.
    No launch waits for that lock, or for an unreadable or read-only file,
    and one the system refuses the lock starts the clone as before, unlocked.
    A wait for the lock that is interrupted keeps no descriptor of it.
    """
    apps = tmp_path / "Applications"
    bundle = apps / f"{BRAND['product_name']}.app"
    icon, exe, state = tmp_path / "icon.png", tmp_path / "kiss-app", tmp_path / "state"
    first, other = _browser_bundle(tmp_path, "1"), _browser_bundle(tmp_path, "2", "Brave")
    replaced = standalone_app.write_macos_launcher(apps, BRAND, icon, exe, state, other)
    clone = standalone_app.write_macos_launcher(apps, BRAND, icon, exe, state, first)
    assert replaced is not None and clone is not None
    log, lock = tmp_path / "kiss-app.log", apps / f".{bundle.name}.lock"
    launch = functools.partial(standalone_app.launch_app, foreground=False, home=tmp_path)
    child = [sys.executable, "-c", _LAUNCH, str(tmp_path), str(first.relative_to(tmp_path))]
    children: list[subprocess.Popen[bytes]] = []
    monkeypatch.setattr(standalone_app, "SETUP_TIMEOUT_SECS", TIMEOUT)  # first and last step wait
    try:
        with exclusive_file_lock(lock):
            assert launch([str(clone), "60"], log, first) is None
            assert "still locked" in capsys.readouterr().err
            assert launch([str(first), "0"], log, first) is True
            interrupted = subprocess.run(
                [sys.executable, "-c", _INTERRUPTED_WAIT, str(lock)],
                capture_output=True,
                text=True,
                timeout=60,
            )
            assert interrupted.stdout == "True\n", interrupted.stderr
        with (first.parents[1] / "Info.plist").open("wb") as plist:
            plistlib.dump({"CFBundleVersion": "1.1", "CFBundleShortVersionString": "1.1"}, plist)
        children.append(subprocess.Popen(child))
        assert _wait_until(lambda: standalone_app.macos_bundle_pid(bundle) == children[0].pid, 60)
        assert standalone_app.macos_bundle_browser(bundle, first) == clone, "a stale clone started"
        running = _clone_state(clone)
        with exclusive_file_lock(lock, blocking=False) as free:
            assert free, "the browser inherited the lock of its launch"
        for chosen, browser in ((replaced, other), (clone, _browser_bundle(tmp_path, "2"))):
            assert launch([str(chosen), "60"], log, browser) is None
            assert "no longer" in capsys.readouterr().err
        children.append(subprocess.Popen([*child, "stopped"]))
        assert os.WIFSTOPPED(os.waitpid(children[1].pid, os.WUNTRACED)[1])
        with exclusive_file_lock(lock, blocking=False) as free:
            assert not free, "a launch does not keep the writers out"
        with pytest.raises(TimeoutError, match="still locked"):
            standalone_app.uninstall_launcher(home=tmp_path, system="Darwin")
        for mode in (0, 0o400, 0o600):  # unreadable: not locked; read-only: locked like any other
            lock.chmod(mode)
            assert launch([str(clone), "300"], log, first) is True, "waited for another launch"
        (tmp_path / "link").symlink_to(clone)  # a browser path that only resolves into the bundle
        assert launch([str(tmp_path / "link"), "0"], log, tmp_path / "link") is True
        assert capsys.readouterr().err == "" and _clone_state(clone) == running
        if sys.platform == "linux" and platform.machine() == "x86_64":  # the filter's numbers
            refused = subprocess.run(
                [sys.executable, "-c", _LAUNCH_WITHOUT_FLOCK, *child[3:]],
                capture_output=True,
                text=True,
                timeout=60,
            )
            assert refused.stdout == "True []\n", refused.stderr
        assert launch([str(replaced), "60"], log, replaced) is None  # $KISS_APP_BROWSER named it
        assert "no longer" in capsys.readouterr().err
        ran = [standalone_app.macos_bundle_processes(b.parents[2]) for b in (clone, first, other)]
        assert len(ran[0] or []) == 4 and ran[1:] == [[], []], "not four windows, all the clone's"
    finally:
        subprocess.run(["pkill", "-f", f"^{tmp_path}/"], check=False)
        for started in children:
            started.kill()
            started.wait(10)


def test_uninstall_removes_the_bundle_as_a_writer_replaces_it(tmp_path: Path) -> None:
    """``uninstall_launcher`` leaves a bundle a window runs from and takes the writers' lock.

    It deleted the clone under the window, and next to a launch (the
    stopped one above) or another uninstall.  Of two at once, the second
    waits while the first lists the processes, and finds nothing left.
    """
    apps = tmp_path / "Applications"
    bundle = apps / f"{BRAND['product_name']}.app"
    icon, exe, state = tmp_path / "icon.png", tmp_path / "kiss-app", tmp_path / "state"
    clone = standalone_app.write_macos_launcher(
        apps, BRAND, icon, exe, state, _browser_bundle(tmp_path, "1")
    )
    assert clone is not None
    steps = _Steps()
    results: dict[str, Path | None] = {}

    def uninstall(name: str, hook: Callable[[FrameType, str, object], None]) -> None:
        sys.setprofile(hook)
        results[name] = standalone_app.uninstall_launcher(home=tmp_path, system="Darwin")

    window, inode = _start_window(clone)
    try:
        assert _wait_until(lambda: standalone_app.macos_bundle_in_use(bundle) is True)
        with pytest.raises(OSError, match="in use"):
            standalone_app.uninstall_launcher(home=tmp_path, system="Darwin")
        assert window.poll() is None and clone.stat().st_ino == inode
    finally:
        window.kill()
        window.wait(10)
    assert _wait_until(lambda: standalone_app.macos_bundle_in_use(bundle) is False)
    first = threading.Thread(target=uninstall, args=("first", steps.writer))
    second = threading.Thread(target=uninstall, args=("second", steps.launcher))
    first.start()
    try:
        assert steps.listed.wait(60), "the first uninstall never listed the processes"
        second.start()
        assert steps.reached.wait(60), "the second uninstall did not wait for the lock"
    finally:
        steps.resume.set()
    first.join(60)
    second.join(60)
    assert results == {"first": bundle, "second": None} and not bundle.exists()


def test_a_popup_that_connected_again_gets_the_sweeps_whole_count(
    bin_dir: Path, tmp_path: Path
) -> None:
    """Sweeps from before the popup last connected do not count against its browser.

    The browser (a copy of ``sleep`` in a bundle; the ``osascript`` stub
    counts no window on screen, as for a minimized popup) is quit by the
    third sweep in a row.  Two strikes, then the popup connects and its
    connection drops: the next sweep quit the browser.  Nor may a sweep
    quit it whose look at the processes spanned such a connection (it
    is held, by a profile hook, where ``windowless_app_pid`` returns).
    """
    _stub(bin_dir, "osascript", "echo 0\n")
    exe = tmp_path / "Brand.app" / "Contents" / "MacOS" / "Chromium"
    exe.parent.mkdir(parents=True)
    _copy_sleep(exe)
    sweep = functools.partial(standalone_app.sweep_windowless_app, exe.parents[2])
    looked, reconnected = threading.Event(), threading.Event()
    swept: list[bool] = []

    def reconnect() -> None:
        assert standalone_app.app_window_event("open")
        assert standalone_app.app_window_event("dropped")

    def held_sweep() -> None:
        sys.setprofile(hold_after_the_look)
        swept.append(sweep())

    def hold_after_the_look(frame: FrameType, event: str, arg: object) -> None:
        if event == "return" and frame.f_code is standalone_app.windowless_app_pid.__code__:
            looked.set()
            reconnected.wait(60)

    window, _ = _start_window(exe)
    try:
        assert _wait_until(lambda: standalone_app.macos_bundle_pid(exe.parents[2]) == window.pid)
        swept += [sweep(), sweep()]
        reconnect()
        swept += [sweep(), sweep()]
        assert swept == [False] * 4 and window.poll() is None, "strikes from before counted"
        sweeper = threading.Thread(target=held_sweep)
        sweeper.start()
        try:
            assert looked.wait(60), "the third sweep never looked at the processes"
            reconnect()
        finally:
            reconnected.set()
            sweeper.join(60)
        assert swept == [False] * 5 and window.poll() is None, "a sweep spanning a connection quit"
        assert [sweep(), sweep(), sweep()] == [False, False, True]
        assert window.wait(10) == -signal.SIGTERM and not sweep()
    finally:
        window.kill()
        window.wait(10)
