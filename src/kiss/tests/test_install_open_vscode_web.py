# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: ``install.sh`` MUST end by opening VS Code in the browser.

The ``kiss-open-vscode-web`` block of ``install.sh`` starts
``code serve-web`` detached (pid file and log under ``$KISS_HOME``), waits
for the ``Web UI available at http://127.0.0.1:PORT?tkn=...`` line, and
opens that URL in the default browser, or prints it with an ``ssh -L``
hint when the machine is remote.  A re-run reuses a server that is still
alive and listening.

The block is extracted verbatim (with the shared browser helpers) and run
under ``bash -euo pipefail`` against a temp ``HOME`` with a stub ``code``
CLI that really listens on a TCP port, and stub ``xdg-open`` / ``open``
executables that log their arguments.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from kiss.tests.conftest import posix_only

_REPO = Path(__file__).resolve().parents[3]

pytestmark = posix_only("runs the open-vscode-web block of install.sh under bash")

_TOKEN = "4a085ab4-0d3a-44e5-b72d-01a56efbd173"

# A ``code`` stub whose ``serve-web`` really binds a port, announces the
# URL the way the VS Code CLI does (license banner first) and stays alive
# until killed.  Every invocation is logged with its arguments.
_SERVE_WEB_STUB = """#!{python}
import socket, sys, time
with open({log!r}, "a") as f:
    f.write(" ".join(sys.argv[1:]) + "\\n")
if sys.argv[1:2] != ["serve-web"]:
    sys.exit(0)
s = socket.socket()
s.bind(("127.0.0.1", 0))
s.listen(1)
print("*")
print("* Visual Studio Code Server")
print("*")
print("Web UI available at http://127.0.0.1:%d?tkn={token}" % s.getsockname()[1], flush=True)
while True:
    time.sleep(1)
"""


def _block(text: str, name: str) -> str:
    return text[text.index(f"# BEGIN: {name}"):text.index(f"# END: {name}")]


def _vscode_web_block() -> str:
    text = (_REPO / "install.sh").read_text(encoding="utf-8")
    return _block(text, "kiss-browser-helpers") + "\n" + _block(text, "kiss-open-vscode-web")


def _write_stub(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def _read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8").splitlines()


class _Sandbox:
    """A temp HOME with a stub ``code`` CLI and stub browser openers.

    The block runs with a PATH of two directories only: the stub
    directory and a directory of symlinks to the few system tools the
    block needs, so the host's own ``code`` / ``xdg-open`` / ``open`` can
    never be picked up (and no real browser or server is ever launched).
    """

    _TOOLS = ("bash", "sed", "head", "tail", "sleep", "nohup", "cat", "mkdir", "hostname", "id")

    def __init__(self, tmp_path: Path) -> None:
        self.home = tmp_path / "home"
        self.kiss_home = self.home / ".kiss"
        self.kiss_home.mkdir(parents=True)
        self.workspace = tmp_path / "workspace"
        self.workspace.mkdir()
        self.bin = tmp_path / "bin"
        self.bin.mkdir()
        self.tools = tmp_path / "tools"
        self.tools.mkdir()
        for name in self._TOOLS:
            real = shutil.which(name)
            assert real is not None, f"{name} is required to run install.sh"
            (self.tools / name).symlink_to(real)
        self.code_cli = self.bin / "code"
        self.code_log = tmp_path / "code.log"
        self.browser_log = tmp_path / "browser.log"
        self.pid_file = self.kiss_home / "vscode-web.pid"
        self.log_file = self.kiss_home / "vscode-web.log"

    def stub_code(self, body: str | None = None) -> None:
        _write_stub(
            self.code_cli,
            body
            or _SERVE_WEB_STUB.format(python=sys.executable, log=str(self.code_log), token=_TOKEN),
        )

    def stub_browser(self, name: str) -> None:
        _write_stub(self.bin / name, f'#!/bin/bash\nprintf "%s\\n" "$*" >> "{self.browser_log}"\n')

    def code_calls(self) -> list[str]:
        return _read_lines(self.code_log)

    def browser_calls(self, expected: int = 1) -> list[str]:
        # xdg-open is launched detached, so its log lines can land just
        # after bash has exited: wait for *expected* of them.
        for _ in range(50):
            if len(_read_lines(self.browser_log)) >= expected:
                break
            time.sleep(0.1)
        return _read_lines(self.browser_log)

    def server_pid(self) -> int:
        return int(self.pid_file.read_text(encoding="utf-8").strip())

    def served_url(self) -> str:
        for line in _read_lines(self.log_file):
            if line.startswith("Web UI available at "):
                return line.split(" ", 4)[4]
        raise AssertionError(f"no URL in {self.log_file}")

    def kill_server(self) -> None:
        if not self.pid_file.exists():
            return
        try:
            os.kill(self.server_pid(), signal.SIGTERM)
        except ProcessLookupError:
            pass

    def run(
        self,
        *,
        os_name: str = "Linux",
        env: dict[str, str] | None = None,
        wait_secs: int = 30,
        prelude: str = "",
    ) -> subprocess.CompletedProcess[str]:
        script = (
            f'OS="{os_name}"\nUSER_PWD="{self.workspace}"\nCODE_CLI="{self.code_cli}"\n'
            f"{prelude}\n{_vscode_web_block()}\n"
        )
        run_env = {
            "PATH": f"{self.bin}:{self.tools}",
            "HOME": str(self.home),
            "KISS_VSCODE_WEB_WAIT_SECS": str(wait_secs),
        }
        if env:
            run_env.update(env)
        return subprocess.run(
            ["bash", "-euo", "pipefail", "-c", script],
            capture_output=True,
            text=True,
            env=run_env,
            timeout=120,
        )


def _assert_serve_web_call(call: str, workspace: Path, port: str = "0") -> None:
    assert call == (
        f"serve-web --port {port} --accept-server-license-terms --default-folder {workspace}"
    )


def test_local_linux_starts_serve_web_and_opens_its_url(tmp_path: Path) -> None:
    """Local Linux desktop: start ``code serve-web`` detached, xdg-open its URL."""
    sb = _Sandbox(tmp_path)
    sb.stub_code()
    sb.stub_browser("xdg-open")
    try:
        proc = sb.run(env={"DISPLAY": ":0"})
        assert proc.returncode == 0, proc.stderr
        calls = sb.code_calls()
        assert len(calls) == 1
        _assert_serve_web_call(calls[0], sb.workspace)
        url = sb.served_url()
        assert url.startswith("http://127.0.0.1:") and url.endswith(f"?tkn={_TOKEN}")
        assert sb.browser_calls() == [url]
        assert f"Opening VS Code in the browser at {url}" in proc.stdout
        assert "aka.ms/vscode-server-license" in proc.stdout
        # The server outlives the block.
        os.kill(sb.server_pid(), 0)
    finally:
        sb.kill_server()


def test_macos_uses_open(tmp_path: Path) -> None:
    """macOS opens the URL with ``open``."""
    sb = _Sandbox(tmp_path)
    sb.stub_code()
    sb.stub_browser("open")
    try:
        proc = sb.run(os_name="Darwin")
        assert proc.returncode == 0, proc.stderr
        assert sb.browser_calls() == [sb.served_url()]
    finally:
        sb.kill_server()


def test_rerun_reuses_the_running_server(tmp_path: Path) -> None:
    """A second run finds the live server (pid alive, port open) and does not start another."""
    sb = _Sandbox(tmp_path)
    sb.stub_code()
    sb.stub_browser("xdg-open")
    try:
        first = sb.run(env={"DISPLAY": ":0"})
        assert first.returncode == 0, first.stderr
        url = sb.served_url()
        pid = sb.server_pid()
        second = sb.run(env={"DISPLAY": ":0"})
        assert second.returncode == 0, second.stderr
        assert f"already served in the browser by 'code serve-web' (pid {pid})" in second.stdout
        assert len(sb.code_calls()) == 1
        assert sb.browser_calls(expected=2) == [url, url]
    finally:
        sb.kill_server()


def test_dead_server_from_an_earlier_run_is_replaced(tmp_path: Path) -> None:
    """A stale pid file and log (server gone) lead to a fresh ``serve-web``."""
    sb = _Sandbox(tmp_path)
    sb.stub_code()
    sb.stub_browser("xdg-open")
    try:
        first = sb.run(env={"DISPLAY": ":0"})
        assert first.returncode == 0, first.stderr
        old_url = sb.served_url()
        sb.kill_server()
        for _ in range(50):
            try:
                os.kill(sb.server_pid(), 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)
        second = sb.run(env={"DISPLAY": ":0"})
        assert second.returncode == 0, second.stderr
        assert "Starting VS Code in the browser" in second.stdout
        assert len(sb.code_calls()) == 2
        new_url = sb.served_url()
        assert new_url != old_url
        assert sb.browser_calls(expected=2) == [old_url, new_url]
    finally:
        sb.kill_server()


def test_live_pid_with_closed_port_is_not_reused(tmp_path: Path) -> None:
    """A pid file naming a live process that no longer listens is not trusted."""
    sb = _Sandbox(tmp_path)
    sb.stub_code()
    sb.stub_browser("xdg-open")
    sleeper = subprocess.Popen(["sleep", "60"])
    try:
        sb.pid_file.write_text(f"{sleeper.pid}\n", encoding="utf-8")
        sb.log_file.write_text(
            f"Web UI available at http://127.0.0.1:1?tkn={_TOKEN}\n", encoding="utf-8"
        )
        proc = sb.run(env={"DISPLAY": ":0"})
        assert proc.returncode == 0, proc.stderr
        assert "Starting VS Code in the browser" in proc.stdout
        assert len(sb.code_calls()) == 1
        assert sb.server_pid() != sleeper.pid
    finally:
        sleeper.kill()
        sb.kill_server()


def test_ssh_session_prints_port_forward_hint_and_opens_no_browser(tmp_path: Path) -> None:
    """Remote machine: print ``ssh -L`` hint plus URL; never call a browser."""
    sb = _Sandbox(tmp_path)
    sb.stub_code()
    sb.stub_browser("xdg-open")
    try:
        proc = sb.run(env={"SSH_CONNECTION": "1.2.3.4 5 6.7.8.9 22", "DISPLAY": ":0"})
        assert proc.returncode == 0, proc.stderr
        url = sb.served_url()
        port = url[len("http://127.0.0.1:"):].split("?", 1)[0]
        assert f"ssh -L {port}:127.0.0.1:{port} " in proc.stdout
        assert f"    {url}" in proc.stdout
        assert sb.browser_calls() == []
    finally:
        sb.kill_server()


def test_cli_that_ignores_serve_web_is_reported(tmp_path: Path) -> None:
    """A ``code`` that exits without a URL (a shim) yields a hint, not a failure."""
    sb = _Sandbox(tmp_path)
    sb.stub_code(f'#!/bin/bash\nprintf "%s\\n" "$*" >> "{sb.code_log}"\nexit 0\n')
    sb.stub_browser("xdg-open")
    proc = sb.run(env={"DISPLAY": ":0"})
    assert proc.returncode == 0, proc.stderr
    assert "'code serve-web' did not announce a URL" in proc.stdout
    assert f"'{sb.code_cli}' serve-web --accept-server-license-terms" in proc.stdout
    assert sb.browser_calls() == []


def test_cli_that_never_announces_times_out(tmp_path: Path) -> None:
    """A ``code`` that stays silent is given KISS_VSCODE_WEB_WAIT_SECS, then reported."""
    sb = _Sandbox(tmp_path)
    sb.stub_code('#!/bin/bash\necho "still starting"\nsleep 60\n')
    sb.stub_browser("xdg-open")
    try:
        proc = sb.run(env={"DISPLAY": ":0"}, wait_secs=1)
        assert proc.returncode == 0, proc.stderr
        assert "'code serve-web' did not announce a URL" in proc.stdout
        assert "       still starting" in proc.stdout
        assert sb.browser_calls() == []
    finally:
        sb.kill_server()


def test_without_a_code_cli_the_step_is_skipped(tmp_path: Path) -> None:
    """No CODE_CLI and ``find_code_cli`` failing: say so and move on."""
    sb = _Sandbox(tmp_path)
    sb.stub_browser("xdg-open")
    proc = sb.run(
        env={"DISPLAY": ":0"},
        prelude='CODE_CLI=""\nfind_code_cli() { return 1; }',
    )
    assert proc.returncode == 0, proc.stderr
    assert "No VS Code CLI found; not starting VS Code in the browser." in proc.stdout
    assert sb.browser_calls() == []


def test_find_code_cli_supplies_a_missing_code_cli(tmp_path: Path) -> None:
    """An unset CODE_CLI is resolved through ``find_code_cli``."""
    sb = _Sandbox(tmp_path)
    sb.stub_code()
    sb.stub_browser("xdg-open")
    try:
        proc = sb.run(
            env={"DISPLAY": ":0"},
            prelude=f'CODE_CLI=""\nfind_code_cli() {{ CODE_CLI="{sb.code_cli}"; }}',
        )
        assert proc.returncode == 0, proc.stderr
        assert sb.browser_calls() == [sb.served_url()]
    finally:
        sb.kill_server()


def test_no_browser_opener_prints_url_for_the_user(tmp_path: Path) -> None:
    """Linux without xdg-open: the URL is printed for the user to open."""
    sb = _Sandbox(tmp_path)
    sb.stub_code()
    try:
        proc = sb.run(env={"DISPLAY": ":0"})
        assert proc.returncode == 0, proc.stderr
        assert f"Opening VS Code in the browser at {sb.served_url()}" in proc.stdout
        assert "Could not open a browser; open the URL above yourself." in proc.stdout
    finally:
        sb.kill_server()


def test_custom_port_is_passed_to_serve_web(tmp_path: Path) -> None:
    """KISS_VSCODE_WEB_PORT replaces the default ``--port 0``."""
    sb = _Sandbox(tmp_path)
    sb.stub_code()
    sb.stub_browser("xdg-open")
    try:
        proc = sb.run(env={"DISPLAY": ":0", "KISS_VSCODE_WEB_PORT": "8123"})
        assert proc.returncode == 0, proc.stderr
        _assert_serve_web_call(sb.code_calls()[0], sb.workspace, port="8123")
    finally:
        sb.kill_server()


def test_kiss_skip_launch_skips_the_step(tmp_path: Path) -> None:
    """KISS_SKIP_LAUNCH (Docker): neither ``code`` nor a browser is run."""
    sb = _Sandbox(tmp_path)
    sb.stub_code()
    sb.stub_browser("xdg-open")
    proc = sb.run(env={"DISPLAY": ":0", "KISS_SKIP_LAUNCH": "1"})
    assert proc.returncode == 0, proc.stderr
    assert sb.code_calls() == []
    assert sb.browser_calls() == []
    assert not sb.pid_file.exists()
