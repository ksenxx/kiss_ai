# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""What ``install.sh`` needs from Windows: the kiss-web daemon and the user's ``PATH``.

macOS hands kiss-web to launchd and Linux to systemd.  Windows has no
service manager a user can register a program with unless they are an
administrator, so ``install.sh`` (run by Git Bash) calls this module
with the interpreter of the app it installed:

* ``restart-daemon`` registers kiss-web to start at every logon of the
  user, stops the kiss-web that holds the port and starts the new one;
* ``start-daemon`` is the command registered for the logon;
* ``stop-daemon`` stops the kiss-web of one app directory, before an
  update replaces its files (Windows keeps the files of a running
  program locked);
* ``uninstall`` stops the daemon and its public tunnel and removes the
  logon start and the Start Menu shortcut (the setup program's
  uninstaller runs it);
* ``add-path`` puts directories on the user's ``PATH``.

The logon start is a value under the per-user ``Run`` registry key.  The
daemon runs without a console window, in the environment a fresh logon
of the user would have (so it is the same daemon whether Git Bash, the
setup program, an ssh session or the logon started it), with its output
appended to ``kiss-web-stdout.log`` and ``kiss-web-stderr.log`` in the
state directory, where launchd and systemd put it too.

Every function here is Windows-only and excluded from coverage on the
other platforms; ``kiss/tests/server/test_windows_install.py`` runs
them on Windows.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from kiss.core.config import kiss_home
from kiss.core.processes import pid_alive

RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
"""The per-user registry key whose values Windows runs at every logon."""
RUN_VALUE = "kiss-web"
ENVIRONMENT_KEY = "Environment"
"""The per-user registry key that holds the user's environment variables."""
SERVICE_FILE = "kiss-web-service.json"
"""In the state directory: the work directory the daemon serves."""
RUN_COMMAND_LIMIT = 260
"""Windows ignores a ``Run`` value whose command line is longer."""
STOP_TIMEOUT_SECS = 20.0


def venv_program(name: str) -> Path:
    """Return the path of program *name* next to the running interpreter (in ``Scripts``)."""
    return Path(sys.executable).with_name(name)


def registry() -> Any:  # pragma: no cover — Windows only
    """Return the ``winreg`` module, untyped: type checkers know its contents on Windows only."""
    import winreg

    return winreg


def system_library(name: str) -> Any:  # pragma: no cover — Windows only
    """Load the Windows library *name* (``kernel32``), untyped for the same reason."""
    assert sys.platform == "win32"
    return ctypes.WinDLL(name, use_last_error=True)  # type: ignore[attr-defined]


def run_command(home: Path | None) -> str:  # pragma: no cover — Windows only
    """Return the command line the logon runs: this module's ``start-daemon``.

    ``pythonw.exe`` runs it, so that no console window opens at logon.
    *home* is the state directory when the installer was given one
    (``KISS_HOME``); without it the daemon uses the app's default.
    """
    python = venv_program("pythonw.exe")
    if not python.exists():
        python = Path(sys.executable)
    parts = [str(python), "-m", "kiss.server.windows_install", "start-daemon"]
    if home is not None:
        parts += ["--home", str(home)]
    return subprocess.list2cmdline(parts)


def register(
    workdir: Path, home: Path | None, run_key: str = RUN_KEY
) -> str:  # pragma: no cover — Windows only
    """Register kiss-web to start at every logon, serving *workdir*; return the command line.

    Args:
        workdir: The directory the daemon serves.
        home: The state directory the installer was given, or ``None``
            for the app's default.
        run_key: The registry key under ``HKEY_CURRENT_USER`` to write
            the value to (the tests name a key of their own).

    Raises:
        OSError: The command line is longer than Windows accepts, or
            the registry or the state directory cannot be written.
    """
    winreg = registry()
    command = run_command(home)
    if len(command) > RUN_COMMAND_LIMIT:
        raise OSError(
            f"the logon command is {len(command)} characters, over the "
            f"{RUN_COMMAND_LIMIT} Windows runs: {command}"
        )
    state = home or kiss_home()
    state.mkdir(parents=True, exist_ok=True)
    (state / SERVICE_FILE).write_text(json.dumps({"workdir": str(workdir)}), encoding="utf-8")
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, run_key) as key:
        winreg.SetValueEx(key, RUN_VALUE, 0, winreg.REG_SZ, command)
    return command


def unregister(run_key: str = RUN_KEY) -> bool:  # pragma: no cover — Windows only
    """Remove the logon start :func:`register` wrote; True when there was one."""
    winreg = registry()
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, run_key, 0, winreg.KEY_SET_VALUE) as key:
            winreg.DeleteValue(key, RUN_VALUE)
    except FileNotFoundError:
        return False
    return True


def service_workdir(state: Path) -> Path:  # pragma: no cover — Windows only
    """Return the work directory :func:`register` recorded in *state*, else the home directory."""
    try:
        workdir = json.loads((state / SERVICE_FILE).read_text(encoding="utf-8"))["workdir"]
    except (OSError, ValueError, KeyError, TypeError):
        return Path.home()
    return Path(workdir) if isinstance(workdir, str) and os.path.isdir(workdir) else Path.home()


def logon_environment() -> dict[str, str]:  # pragma: no cover — Windows only
    """Return the environment a fresh logon of this user starts programs with.

    Windows builds it from the registry (``CreateEnvironmentBlock``),
    so it has the ``PATH`` the installer just extended and none of what
    the calling shell added (Git Bash's ``MSYSTEM`` and ``/usr/bin``, an
    ssh session's variables).  This process's own environment when
    Windows does not answer.
    """
    from ctypes import wintypes

    advapi32 = system_library("advapi32")
    userenv = system_library("userenv")
    kernel32 = system_library("kernel32")
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    token = wintypes.HANDLE()
    token_query = 0x0008
    if not advapi32.OpenProcessToken(
        wintypes.HANDLE(kernel32.GetCurrentProcess()), token_query, ctypes.byref(token)
    ):
        return dict(os.environ)
    block = ctypes.c_void_p()
    try:
        if not userenv.CreateEnvironmentBlock(ctypes.byref(block), token, False):
            return dict(os.environ)
    finally:
        kernel32.CloseHandle(token)
    try:
        environment: dict[str, str] = {}
        offset = 0
        while True:
            entry = ctypes.wstring_at((block.value or 0) + offset)
            if not entry:
                return environment
            offset += (len(entry) + 1) * ctypes.sizeof(ctypes.c_wchar)
            name, _, value = entry.partition("=")
            if name:  # "=C:=C:\dir" entries (per-drive directories) have none
                environment[name] = value
    finally:
        userenv.DestroyEnvironmentBlock(block)


def start_detached(
    command: list[str],
    cwd: Path,
    environment: dict[str, str],
    stdout: Path,
    stderr: Path,
) -> subprocess.Popen[bytes]:  # pragma: no cover — Windows only
    """Start *command* so that it outlives the caller and shows no window.

    The child gets a hidden console of its own (the console programs it
    starts in turn, git and bash, then open no window either), no
    stdin, and its output appended to *stdout* and *stderr*.  It leaves
    the caller's job object when that job allows it: an ssh session
    ends every process of its job when it closes.

    Raises:
        OSError: *command* cannot be started or a log cannot be opened.
    """
    assert sys.platform == "win32"
    hidden = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
    group = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    leave_job = subprocess.CREATE_BREAKAWAY_FROM_JOB  # type: ignore[attr-defined]
    with open(stdout, "ab") as out, open(stderr, "ab") as err:
        for breakaway in (leave_job, 0):
            try:
                return subprocess.Popen(
                    command,
                    cwd=cwd,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=out,
                    stderr=err,
                    creationflags=hidden | group | breakaway,
                )
            except PermissionError:  # the caller's job does not let a process leave it
                if not breakaway:
                    raise
    raise AssertionError("unreachable")


def start_daemon(home: Path | None) -> int:  # pragma: no cover — Windows only
    """Start kiss-web from this interpreter's venv; return its process id.

    Args:
        home: The state directory to run it for (its ``KISS_HOME``), or
            ``None`` for the app's default.

    Raises:
        OSError: kiss-web cannot be started or its logs cannot be opened.
    """
    environment = logon_environment()
    if home is not None:
        environment["KISS_HOME"] = str(home)
    else:
        environment.pop("KISS_HOME", None)
    state = home or kiss_home()
    state.mkdir(parents=True, exist_ok=True)
    return start_detached(
        [str(venv_program("kiss-web.exe"))],
        service_workdir(state),
        environment,
        state / "kiss-web-stdout.log",
        state / "kiss-web-stderr.log",
    ).pid


def listeners(port: int) -> list[int]:  # pragma: no cover — Windows only
    """Return the ids of the processes listening on TCP *port*, on any address.

    Read from ``netstat -ano``: a listening socket is the row whose
    remote address is ``0.0.0.0:0`` or ``[::]:0`` (the state column is
    translated on a non-English Windows).  Empty when ``netstat`` does
    not answer.
    """
    try:
        result = subprocess.run(
            ["netstat", "-ano", "-p", "TCP"],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=30,
            stdin=subprocess.DEVNULL,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    found: list[int] = []
    for line in result.stdout.splitlines():
        parts = line.split()
        if (
            len(parts) == 5
            and parts[0] == "TCP"
            and parts[1].rpartition(":")[2] == str(port)
            and parts[2] in ("0.0.0.0:0", "[::]:0")
            and parts[4].isdigit()
            and int(parts[4]) not in found
        ):
            found.append(int(parts[4]))
    return found


def command_line(pid: int) -> str:  # pragma: no cover — Windows only
    """Return the command line process *pid* was started with, or ``""`` when it cannot be read."""
    try:
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                f"(Get-CimInstance Win32_Process -Filter 'ProcessId={pid}').CommandLine",
            ],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=60,
            stdin=subprocess.DEVNULL,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout.strip()


def is_kiss_web(line: str, app: Path | None) -> bool:
    """Whether the command line *line* is a kiss-web daemon's, of app directory *app* if given."""
    line = line.lower()  # Windows names files in either case
    if "kiss-web" not in line and "kiss.server" not in line:
        return False
    return app is None or str(app).lower() + os.sep in line


def stop_daemon(
    port: int, app: Path | None = None, timeout: float = STOP_TIMEOUT_SECS
) -> list[int]:  # pragma: no cover — Windows only
    """End every kiss-web listening on *port*; return the process ids ended.

    Only the daemon itself is ended, not the processes it started: the
    installer the daemon's Update button runs is one of them.  A
    listener that is not kiss-web, or with *app* one that does not run
    from that app directory, is left alone.  Returns once the ended
    processes are gone, or after *timeout* seconds.
    """
    ended = [
        pid
        for pid in listeners(port)
        if pid != os.getpid() and is_kiss_web(command_line(pid), app)
    ]
    for pid in ended:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:  # it exited meanwhile
            pass
    deadline = time.monotonic() + timeout
    while any(pid_alive(pid) for pid in ended) and time.monotonic() < deadline:
        time.sleep(0.2)
    return ended


def add_to_user_path(
    directory: Path, key_name: str = ENVIRONMENT_KEY
) -> bool:  # pragma: no cover — Windows only
    """Append *directory* to the user's ``PATH``; False when it is there already.

    There already means in the user's value or in the ``PATH`` a logon
    gets (which starts with the machine's, where a Git installed for
    all users is).  The value is edited in the registry as it is stored
    (its ``%USERPROFILE%`` references are kept), and running programs
    are told that the environment changed, so a terminal opened
    afterwards has the new ``PATH`` without a new logon.

    Args:
        directory: The directory to add.
        key_name: The registry key under ``HKEY_CURRENT_USER`` that
            holds the ``Path`` value (the tests name a key of their own).

    Raises:
        OSError: The registry cannot be read or written.
    """
    winreg = registry()
    wanted = os.path.normcase(os.path.normpath(str(directory)))
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, key_name) as key:
        try:
            value, kind = winreg.QueryValueEx(key, "Path")
        except FileNotFoundError:
            value, kind = "", winreg.REG_EXPAND_SZ
        entries = [entry for entry in str(value).split(";") if entry]
        # Windows spells the variable ``Path`` or ``PATH``.
        logon = [
            entry
            for name, value in logon_environment().items()
            if name.upper() == "PATH"
            for entry in value.split(";")
        ]
        for entry in [*entries, *logon]:
            if os.path.normcase(os.path.normpath(os.path.expandvars(entry))) == wanted:
                return False
        winreg.SetValueEx(key, "Path", 0, kind, ";".join([*entries, str(directory)]))
    hwnd_broadcast, wm_settingchange, smto_abortifhung = 0xFFFF, 0x001A, 0x0002
    system_library("user32").SendMessageTimeoutW(
        hwnd_broadcast, wm_settingchange, 0, "Environment", smto_abortifhung, 5000, None
    )
    return True


def run(args: argparse.Namespace) -> None:  # pragma: no cover — Windows only
    """Carry out the parsed command line *args* of :func:`main`.

    Raises:
        OSError: The command failed; the text says why.
    """
    if args.command == "restart-daemon":
        # KISS_HOME of the installer, as it writes it into the launchd
        # plist and the systemd unit: only when it was given one.
        env_home = os.environ.get("KISS_HOME")
        home = Path(env_home) if env_home else None
        register(args.workdir.resolve(), home)
        for pid in stop_daemon(args.port):
            print(f"   Stopped kiss-web (process {pid}).")
        print(f"   Started kiss-web (process {start_daemon(home)}).")
    elif args.command == "start-daemon":
        start_daemon(args.home)
    elif args.command == "stop-daemon":
        for pid in stop_daemon(args.port, args.app.resolve()):
            print(f"   Stopped kiss-web (process {pid}).")
    elif args.command == "uninstall":
        # Imported here: the logon's start-daemon needs neither module.
        from kiss.server.standalone_app import uninstall_launcher
        from kiss.server.web_server import _terminate_orphan_cloudflared

        for pid in stop_daemon(args.port):
            print(f"Stopped kiss-web (process {pid}).")
        # The daemon leaves its tunnel running for the next daemon to adopt.
        _terminate_orphan_cloudflared()
        if unregister():
            print("Removed the start of kiss-web at logon.")
        shortcut = uninstall_launcher()
        if shortcut:
            print(f"Removed {shortcut}.")
    else:
        for directory in args.directories:
            if add_to_user_path(directory.resolve()):
                print(f"   Added {directory} to your PATH (new terminals have it).")


def main(argv: list[str] | None = None) -> int:
    """Run the command line (see the module documentation); return the exit status.

    2 on any platform but Windows, 1 when the command failed.
    """
    parser = argparse.ArgumentParser(prog="python -m kiss.server.windows_install")
    commands = parser.add_subparsers(dest="command", required=True)
    restart = commands.add_parser("restart-daemon", help="register, stop and start kiss-web")
    restart.add_argument("--workdir", type=Path, required=True)
    restart.add_argument("--port", type=int, required=True)
    start = commands.add_parser("start-daemon", help="start kiss-web (the logon command)")
    start.add_argument("--home", type=Path, default=None)
    stop = commands.add_parser("stop-daemon", help="stop the kiss-web of one app directory")
    stop.add_argument("--port", type=int, required=True)
    stop.add_argument("--app", type=Path, required=True)
    uninstall = commands.add_parser(
        "uninstall", help="stop kiss-web; remove its logon start and the Start Menu shortcut"
    )
    uninstall.add_argument("--port", type=int, required=True)
    path = commands.add_parser("add-path", help="add directories to the user's PATH")
    path.add_argument("directories", type=Path, nargs="+")
    args = parser.parse_args(argv)
    if sys.platform != "win32":
        print("kiss.server.windows_install runs on Windows only.", file=sys.stderr)
        return 2
    try:  # pragma: no cover — Windows only
        run(args)
    except OSError as exc:  # pragma: no cover — Windows only
        print(f"{args.command} failed: {exc}", file=sys.stderr)
        return 1
    return 0  # pragma: no cover — Windows only


if __name__ == "__main__":  # pragma: no cover — command line entry
    sys.exit(main())
