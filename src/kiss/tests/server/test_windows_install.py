# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of what ``install.sh`` needs from Windows.

:mod:`kiss.server.windows_install` starts kiss-web at logon through the
per-user ``Run`` registry key, starts and stops the daemon and edits the
user's ``PATH``; :mod:`kiss.server.standalone_app` writes the Start Menu
shortcut and :func:`kiss.core.processes.find_bash` finds Git Bash.

The Windows tests use the real thing: registry keys of their own under
``HKEY_CURRENT_USER\\Software\\KissSorcarTests`` (deleted afterwards),
real processes and listening sockets, ``netstat`` and PowerShell.  None
of them touches the user's ``Run`` key, ``PATH`` or Start Menu.  What
does not depend on Windows (which command line is a kiss-web daemon's,
the command line's refusals) runs everywhere.

Not run here: ``restart-daemon``, ``start-daemon`` and ``uninstall``
as commands.  They act on the user's own logon start and on the daemon
of port 8787; the functions they are made of are tested one by one.
"""

from __future__ import annotations

import contextlib
import json
import os
import socket
import subprocess
import sys
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

from kiss.core.processes import find_bash, pid_alive
from kiss.server import standalone_app, windows_install
from kiss.tests.conftest import find_free_port, posix_only

windows_only = pytest.mark.skipif(
    sys.platform != "win32", reason="Windows only: the registry, netstat, the logon environment"
)

_TESTS_KEY = r"Software\KissSorcarTests"
# A stand-in daemon: listens on the port given, says so and stays.
_LISTENER = """\
import socket, sys, time
server = socket.socket()
server.bind(("127.0.0.1", int(sys.argv[1])))
server.listen()
print("READY", flush=True)
time.sleep(600)
"""


def _command(*args: str, **env: str) -> subprocess.CompletedProcess[str]:
    """Run ``python -m kiss.server.windows_install`` with *args*."""
    return subprocess.run(
        [sys.executable, "-m", "kiss.server.windows_install", *args],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, **env},
        check=False,
    )


@pytest.fixture
def scratch_key() -> Iterator[str]:
    """A registry key of this test under ``HKEY_CURRENT_USER``, deleted afterwards."""
    assert sys.platform == "win32"
    winreg = windows_install.registry()

    name = rf"{_TESTS_KEY}\{uuid.uuid4().hex}"
    try:
        yield name
    finally:
        for key in (name, _TESTS_KEY):  # the parent goes with its last test
            with contextlib.suppress(OSError):
                winreg.DeleteKey(winreg.HKEY_CURRENT_USER, key)


# --- everywhere ---------------------------------------------------------------


def test_is_kiss_web_tells_the_daemon_of_an_app(tmp_path: Path) -> None:
    """A kiss-web command line is the program's or the module's, of the app directory asked for."""
    app = tmp_path / "kiss"
    program = f'"{app / ".venv" / "Scripts" / "kiss-web.exe"}"'
    module = f"{app / '.venv' / 'Scripts' / 'python.exe'} -m kiss.server.web_server"
    for line in (program, module):
        assert windows_install.is_kiss_web(line, None)
        assert windows_install.is_kiss_web(line, app)
        assert not windows_install.is_kiss_web(line, tmp_path / "other")
        # A directory whose name only starts like the app's is another app.
        assert not windows_install.is_kiss_web(line, tmp_path / "kis")
        # Windows spells one directory in any case.
        assert windows_install.is_kiss_web(line, Path(str(app).upper()))
    for line in ("", r"C:\Windows\System32\svchost.exe -k netsvcs", f"{app}/other-server.exe"):
        assert not windows_install.is_kiss_web(line, None)
    assert windows_install.venv_program("kiss-web.exe") == Path(sys.executable).with_name(
        "kiss-web.exe"
    )


@posix_only("on Windows the commands act on the user's logon start, daemon and PATH")
def test_commands_refuse_to_run_off_windows(tmp_path: Path) -> None:
    """Every command ends with status 2 and one line on stderr, having done nothing."""
    home, work = tmp_path / "home", tmp_path / "work"
    work.mkdir()
    for args in (
        ["restart-daemon", "--workdir", str(work), "--port", "1"],
        ["start-daemon", "--home", str(home)],
        ["stop-daemon", "--port", "1", "--app", str(work)],
        ["uninstall", "--port", "1"],
        ["add-path", str(work)],
    ):
        result = _command(*args, KISS_HOME=str(home))
        assert result.returncode == 2, args
        assert result.stderr == "kiss.server.windows_install runs on Windows only.\n"
        assert result.stdout == ""
    assert list(work.iterdir()) == []
    # Importing the app writes its model list there; a daemon writes these.
    assert not list(home.glob("kiss-web-*"))


@pytest.mark.parametrize(
    "args",
    [[], ["bogus"], ["add-path"], ["uninstall"], ["stop-daemon", "--port", "x", "--app", "."]],
)
def test_bad_command_line_is_refused(args: list[str]) -> None:
    """A command line the parser does not accept stops with the usage before anything runs."""
    result = _command(*args)
    assert result.returncode == 2
    assert "usage: python -m kiss.server.windows_install" in result.stderr
    assert result.stdout == ""


# --- Windows ------------------------------------------------------------------


@windows_only
def test_register_writes_the_logon_start_and_the_work_directory(
    tmp_path: Path, scratch_key: str
) -> None:
    """The ``Run`` value runs ``start-daemon``; the state directory records the work directory."""
    assert sys.platform == "win32"
    winreg = windows_install.registry()

    home, workdir = tmp_path / "h", tmp_path / "w"
    workdir.mkdir()
    command = windows_install.register(workdir, home, run_key=scratch_key)
    assert command.endswith(
        subprocess.list2cmdline(
            ["-m", "kiss.server.windows_install", "start-daemon", "--home", str(home)]
        )
    )
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, scratch_key) as key:
        assert winreg.QueryValueEx(key, windows_install.RUN_VALUE) == (command, winreg.REG_SZ)
    service = json.loads((home / windows_install.SERVICE_FILE).read_text(encoding="utf-8"))
    assert service == {"workdir": str(workdir)}
    assert windows_install.service_workdir(home) == workdir
    assert windows_install.service_workdir(tmp_path / "no-such-home") == Path.home()
    assert windows_install.unregister(scratch_key) is True
    assert windows_install.unregister(scratch_key) is False


@windows_only
def test_register_refuses_a_command_windows_would_not_run(
    tmp_path: Path, scratch_key: str
) -> None:
    """A logon command over 260 characters is an error, and nothing is written."""
    home = tmp_path / ("x" * windows_install.RUN_COMMAND_LIMIT)
    with pytest.raises(OSError, match="over the 260 Windows runs"):
        windows_install.register(tmp_path, home, run_key=scratch_key)
    assert not home.exists()
    assert windows_install.unregister(scratch_key) is False


@windows_only
def test_logon_environment_is_the_users_not_this_process_s(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """It has what Windows gives a fresh logon and nothing this process added."""
    monkeypatch.setenv("KISS_TEST_ONLY", "1")
    environment = {
        name.upper(): value for name, value in windows_install.logon_environment().items()
    }
    assert {"SYSTEMROOT", "USERPROFILE", "PATH"} <= environment.keys()
    assert os.path.isdir(environment["SYSTEMROOT"])
    assert "KISS_TEST_ONLY" not in environment


@windows_only
def test_start_detached_runs_a_program_with_its_output_appended(tmp_path: Path) -> None:
    """The program gets the environment and directory given; its output is added to the logs."""
    out, err = tmp_path / "out.log", tmp_path / "err.log"
    out.write_text("earlier\n", encoding="utf-8")
    script = "import os, sys; print(os.environ['KISS_TEST_ONLY']); sys.stderr.write(os.getcwd())"
    child = windows_install.start_detached(
        [sys.executable, "-c", script],
        tmp_path,
        {**os.environ, "KISS_TEST_ONLY": "from-the-test"},
        out,
        err,
    )
    assert child.pid > 0
    assert child.wait(timeout=120) == 0
    assert out.read_text(encoding="utf-8").splitlines() == ["earlier", "from-the-test"]
    assert os.path.samefile(err.read_text(encoding="utf-8").strip(), tmp_path)


@windows_only
def test_listeners_names_the_process_listening_on_a_port() -> None:
    """The listener of a port is found through ``netstat``, and gone once the socket is closed."""
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        port = server.getsockname()[1]
        assert windows_install.listeners(port) == [os.getpid()]
    assert windows_install.listeners(port) == []


@windows_only
def test_stop_daemon_ends_only_the_kiss_web_of_the_app(tmp_path: Path) -> None:
    """The listener is ended when it is a kiss-web of the app directory named, not otherwise."""
    app = tmp_path / "app"
    app.mkdir()
    script = app / "kiss-web-script.py"
    script.write_text(_LISTENER, encoding="utf-8")
    port = find_free_port()
    daemon = subprocess.Popen(
        [sys.executable, str(script), str(port)], stdout=subprocess.PIPE, text=True
    )
    try:
        assert daemon.stdout is not None
        assert daemon.stdout.readline().strip() == "READY"
        assert windows_install.stop_daemon(port, app=tmp_path / "other") == []
        assert daemon.poll() is None
        # The listener is the interpreter itself: a venv's python.exe may
        # only be the launcher that started it, so its id is not asked for.
        (pid,) = windows_install.stop_daemon(port, app=app)
        assert not pid_alive(pid)
        assert windows_install.listeners(port) == []
        daemon.wait(timeout=60)
    finally:
        daemon.kill()
        daemon.wait(timeout=60)


@windows_only
def test_add_to_user_path_appends_a_directory_once(tmp_path: Path, scratch_key: str) -> None:
    """A new directory goes to the end of the stored value; one a logon already has is left out."""
    assert sys.platform == "win32"
    winreg = windows_install.registry()

    first, second = tmp_path / "first", tmp_path / "second"
    assert windows_install.add_to_user_path(first, key_name=scratch_key) is True
    assert windows_install.add_to_user_path(first, key_name=scratch_key) is False
    assert windows_install.add_to_user_path(second, key_name=scratch_key) is True
    # On the PATH of every logon, through the machine's own value.
    system32 = Path(os.environ["SystemRoot"]) / "System32"
    assert windows_install.add_to_user_path(system32, key_name=scratch_key) is False
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, scratch_key) as key:
        assert winreg.QueryValueEx(key, "Path") == (f"{first};{second}", winreg.REG_EXPAND_SZ)


@windows_only
def test_find_bash_is_git_bash_not_the_wsl_launcher() -> None:
    """The bash found exists and is not the ``bash.exe`` of Windows, which starts WSL."""
    bash = find_bash()
    assert bash is not None and os.path.isfile(bash)
    system32 = os.path.join(os.environ["SystemRoot"], "System32")
    assert os.path.normcase(os.path.dirname(bash)) != os.path.normcase(system32)


@windows_only
def test_launcher_is_a_start_menu_shortcut(tmp_path: Path) -> None:
    """``install_launcher`` writes the shortcut with the state directory in its arguments."""
    state = tmp_path / "state"
    shortcut = standalone_app.install_launcher(
        system="Windows",
        home=tmp_path,
        state_dir=state,
        exe=Path(sys.executable).with_name("kiss-app.exe"),
    )
    assert shortcut is not None and shortcut == standalone_app.windows_launcher_path(tmp_path)
    assert shortcut.is_file()
    assert (state / f"{standalone_app.brand_slug()}.ico").is_file()
    arguments = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "(New-Object -ComObject WScript.Shell).CreateShortcut($env:KISS_SHORTCUT).Arguments",
        ],
        env={**os.environ, "KISS_SHORTCUT": str(shortcut)},
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    ).stdout.strip()
    assert arguments.endswith(subprocess.list2cmdline(["--home", str(state)]))
    assert standalone_app.uninstall_launcher(home=tmp_path, system="Windows") == shortcut
    assert not shortcut.exists()
    assert standalone_app.uninstall_launcher(home=tmp_path, system="Windows") is None
