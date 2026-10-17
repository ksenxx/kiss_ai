# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: ``install.sh`` MUST end by opening the webapp.

The ``kiss-open-webapp`` block of ``install.sh`` waits for the kiss-web
daemon (started by step [5/5]) and then opens the Local URL
(``http://127.0.0.1:PORT``, plain HTTP on loopback) as a standalone app
window through ``kiss-app`` (next to ``kiss-web`` in the app's venv:
desktop launcher first, then the window), in the default browser when
there is no ``kiss-app`` or it finds no browser, or prints the
cloudflared URL when the machine is remote (an SSH session, or Linux
without a display).  ``kiss-web`` itself is only located (for the hint
printed on timeout), never run.

The block (together with the ``kiss-browser-helpers`` block it relies
on) is extracted verbatim and run under ``bash -euo pipefail`` against a
temp ``HOME`` / ``KISS_HOME`` with stub ``kiss-web``, ``kiss-app``,
``xdg-open`` and ``open`` executables that log their arguments.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from pathlib import Path

from kiss.tests.conftest import posix_only

_REPO = Path(__file__).resolve().parents[3]

pytestmark = posix_only("runs the open-webapp block of install.sh under bash")

_LOOPBACK = "http://127.0.0.1:8787"
_TUNNEL = "https://example-tunnel.trycloudflare.com"


def install_sh_block(name: str) -> str:
    """Return the ``# BEGIN: <name>`` ... ``# END: <name>`` block of install.sh."""
    text = (_REPO / "install.sh").read_text(encoding="utf-8")
    start = text.index(f"# BEGIN: {name}")
    end = text.index(f"# END: {name}")
    return text[start:end]


def install_sh_function(name: str) -> str:
    """Return the top-level shell function *name* of install.sh (``name() {`` ... ``}``)."""
    text = (_REPO / "install.sh").read_text(encoding="utf-8")
    match = re.search(rf"^{re.escape(name)}\(\) \{{\n.*?^\}}\n", text, re.MULTILINE | re.DOTALL)
    assert match is not None, f"install.sh has no function {name}"
    return match.group(0)


def _open_webapp_block() -> str:
    # The block uses machine_is_remote / open_in_browser from the shared
    # helpers block that precedes it in install.sh, app_dir_of (the app
    # of a state directory) and warn (kiss-install-problems: a warning
    # is printed and recorded for the web app's notification).
    return "\n".join(
        [
            install_sh_function("app_dir_of"),
            install_sh_block("kiss-install-problems"),
            install_sh_block("kiss-browser-helpers"),
            install_sh_block("kiss-open-webapp"),
        ]
    )


def _write_stub(path: Path, log: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'#!/bin/bash\nprintf "%s\\n" "$*" >> "{log}"\n',
        encoding="utf-8",
    )
    path.chmod(0o755)


def _url_json(**fields: str) -> str:
    return json.dumps(fields, indent=2)


class _Sandbox:
    """A temp HOME with stub binaries and a remote-url.json.

    The block runs with a PATH of two directories only: the stub
    directory and a directory of symlinks to the few system tools the
    block needs, so the host's own ``xdg-open`` / ``open`` / ``kiss-web``
    can never be picked up (and no real browser is ever launched).
    """

    _TOOLS = ("bash", "sed", "head", "sleep", "cp", "nohup")

    def __init__(self, tmp_path: Path) -> None:
        self.home = tmp_path / "home"
        self.kiss_home = self.home / ".kiss"
        self.kiss_home.mkdir(parents=True)
        self.app_dir = self.kiss_home / "kiss"
        self.bin = tmp_path / "bin"
        self.bin.mkdir()
        self.tools = tmp_path / "tools"
        self.tools.mkdir()
        for name in self._TOOLS:
            real = shutil.which(name)
            assert real is not None, f"{name} is required to run install.sh"
            (self.tools / name).symlink_to(real)
        self.kiss_web_log = tmp_path / "kiss-web.log"
        self.browser_log = tmp_path / "browser.log"
        self.url_file = self.kiss_home / "remote-url.json"

    def stub_kiss_web(self, where: Path | None = None) -> None:
        _write_stub(where or self.bin / "kiss-web", self.kiss_web_log)

    def stub_browser(self, name: str) -> None:
        _write_stub(self.bin / name, self.browser_log)

    def stub_kiss_app(self, *, fail_on: str = "", where: Path | None = None) -> Path:
        """A ``kiss-app`` (next to the stub kiss-web, or at *where*) that logs its calls.

        Each line of the log is ``<KISS_HOME>: <arguments>``.  It exits 1
        for the invocation whose first argument is *fail_on*
        (``--install-launcher`` or ``--url``), 0 otherwise.
        """
        path = where or self.bin / "kiss-app"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.kiss_app_log = self.home / "kiss-app.log"
        path.write_text(
            "#!/bin/bash\n"
            f'printf "%s: %s\\n" "${{KISS_HOME:-unset}}" "$*" >> "{self.kiss_app_log}"\n'
            f'[ "$1" != "{fail_on}" ] || exit 1\nexit 0\n',
            encoding="utf-8",
        )
        path.chmod(0o755)
        return path

    def expected_kiss_app_calls(
        self, kiss_home: Path | None = None, *, revealed: bool = True
    ) -> list[str]:
        """The launcher install, the window open and (*revealed*) the launcher's folder.

        All for *kiss_home* (default: ~/.kiss).  The folder is shown only
        after both the launcher and the window succeeded.
        """
        home = kiss_home or self.kiss_home
        calls = [f"{home}: --install-launcher", f"{home}: --url {_LOOPBACK} --wait 0"]
        if revealed:
            calls.append(f"{home}: --reveal-launcher")
        return calls

    def kiss_app_calls(self) -> list[str]:
        if not self.kiss_app_log.exists():
            return []
        return self.kiss_app_log.read_text(encoding="utf-8").splitlines()

    def kiss_web_calls(self) -> list[str]:
        if not self.kiss_web_log.exists():
            return []
        return self.kiss_web_log.read_text(encoding="utf-8").splitlines()

    def browser_calls(self) -> list[str]:
        # xdg-open is launched detached (a desktop may keep it running for
        # a while), so its log line can land just after bash has exited.
        for _ in range(50):
            if self.browser_log.exists():
                break
            time.sleep(0.1)
        if not self.browser_log.exists():
            return []
        return self.browser_log.read_text(encoding="utf-8").splitlines()

    def run(
        self,
        *,
        os_name: str = "Linux",
        env: dict[str, str] | None = None,
        wait_secs: int = 0,
        prelude: str = "",
    ) -> subprocess.CompletedProcess[str]:
        script = (
            f'OS="{os_name}"\nPROJECT_DIR="{self.home}/no-project"\nAPP_DIR="{self.app_dir}"\n'
            f"{prelude}\n{_open_webapp_block()}\n"
        )
        run_env = {
            "PATH": f"{self.bin}:{self.tools}",
            "HOME": str(self.home),
            "KISS_WEBAPP_WAIT_SECS": str(wait_secs),
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


def test_local_linux_opens_loopback_url(tmp_path: Path) -> None:
    """Local Linux desktop: xdg-open the plain-HTTP 127.0.0.1 URL."""
    sb = _Sandbox(tmp_path)
    sb.stub_kiss_web()
    sb.stub_browser("xdg-open")
    sb.url_file.write_text(_url_json(
        local="http://localhost:8787", tunnel=_TUNNEL, loopback=_LOOPBACK,
    ))
    proc = sb.run(env={"DISPLAY": ":0"})
    assert proc.returncode == 0, proc.stderr
    assert sb.kiss_web_calls() == []
    assert sb.browser_calls() == [_LOOPBACK]
    assert f"Opening the webapp at {_LOOPBACK}" in proc.stdout
    assert _TUNNEL not in proc.stdout


def test_macos_uses_open(tmp_path: Path) -> None:
    """On Darwin the Local URL is handed to ``open``."""
    sb = _Sandbox(tmp_path)
    sb.stub_kiss_web()
    sb.stub_browser("open")
    sb.url_file.write_text(_url_json(loopback=_LOOPBACK))
    proc = sb.run(os_name="Darwin")
    assert proc.returncode == 0, proc.stderr
    assert sb.kiss_web_calls() == []
    assert sb.browser_calls() == [_LOOPBACK]


def test_ssh_session_prints_tunnel_url_and_opens_no_browser(tmp_path: Path) -> None:
    """A remote machine gets the cloudflared URL printed instead."""
    sb = _Sandbox(tmp_path)
    sb.stub_kiss_web()
    sb.stub_browser("xdg-open")
    sb.url_file.write_text(_url_json(loopback=_LOOPBACK, tunnel=_TUNNEL))
    proc = sb.run(env={"DISPLAY": ":0", "SSH_CONNECTION": "1.2.3.4 1 5.6.7.8 22"})
    assert proc.returncode == 0, proc.stderr
    assert sb.kiss_web_calls() == []
    assert sb.browser_calls() == []
    assert "This machine is remote" in proc.stdout
    assert _TUNNEL in proc.stdout
    assert _LOOPBACK not in proc.stdout


def test_headless_linux_counts_as_remote(tmp_path: Path) -> None:
    """Linux without DISPLAY / WAYLAND_DISPLAY is treated as remote."""
    sb = _Sandbox(tmp_path)
    sb.stub_kiss_web()
    sb.stub_browser("xdg-open")
    sb.url_file.write_text(_url_json(loopback=_LOOPBACK, tunnel=_TUNNEL))
    proc = sb.run()
    assert proc.returncode == 0, proc.stderr
    assert sb.browser_calls() == []
    assert _TUNNEL in proc.stdout


def test_remote_waits_for_tunnel_url_then_gives_hint_on_timeout(tmp_path: Path) -> None:
    """Remote + URL file without a tunnel yet: wait, then print the hint."""
    sb = _Sandbox(tmp_path)
    sb.stub_kiss_web()
    sb.url_file.write_text(_url_json(loopback=_LOOPBACK))
    proc = sb.run(wait_secs=5)
    assert proc.returncode == 0, proc.stderr
    assert sb.kiss_web_calls() == []
    assert "did not come up within 5s" in proc.stdout
    # The hint names the binary that was found, since it is rarely on PATH.
    kiss_web = sb.bin / "kiss-web"
    assert f"'{kiss_web}' --url" in proc.stdout
    assert "--trust-ca" not in proc.stdout


def test_waits_for_daemon_that_comes_up_late(tmp_path: Path) -> None:
    """The URL file appearing after a few seconds is picked up by the loop."""
    sb = _Sandbox(tmp_path)
    sb.stub_kiss_web()
    sb.stub_browser("xdg-open")
    late = tmp_path / "late.json"
    late.write_text(_url_json(loopback=_LOOPBACK))
    prelude = f'(sleep 7; cp "{late}" "{sb.url_file}") >/dev/null 2>&1 &'
    proc = sb.run(env={"DISPLAY": ":0"}, wait_secs=60, prelude=prelude)
    assert proc.returncode == 0, proc.stderr
    assert "Waiting for the kiss-web daemon" in proc.stdout
    assert sb.kiss_web_calls() == []
    assert sb.browser_calls() == [_LOOPBACK]


def test_missing_kiss_web_binary_times_out(tmp_path: Path) -> None:
    """A URL file without any kiss-web binary is not enough."""
    sb = _Sandbox(tmp_path)
    sb.stub_browser("xdg-open")
    sb.url_file.write_text(_url_json(loopback=_LOOPBACK))
    proc = sb.run(env={"DISPLAY": ":0"})
    assert proc.returncode == 0, proc.stderr
    assert "did not come up within 0s" in proc.stdout
    assert "'kiss-web' --url" in proc.stdout
    assert "--trust-ca" not in proc.stdout
    assert sb.browser_calls() == []


def test_kiss_web_found_in_the_app_venv(tmp_path: Path) -> None:
    """Off PATH, kiss-web is located in the installed app's venv ($APP_DIR)."""
    sb = _Sandbox(tmp_path)
    sb.stub_kiss_web(where=sb.app_dir / ".venv" / "bin" / "kiss-web")
    sb.stub_browser("xdg-open")
    sb.url_file.write_text(_url_json(loopback=_LOOPBACK))
    proc = sb.run(env={"DISPLAY": ":0"})
    assert proc.returncode == 0, proc.stderr
    assert sb.kiss_web_calls() == []
    assert sb.browser_calls() == [_LOOPBACK]


def test_old_url_file_without_loopback_falls_back_to_127_0_0_1(tmp_path: Path) -> None:
    """A daemon that only wrote ``local`` still yields a 127.0.0.1 URL."""
    sb = _Sandbox(tmp_path)
    sb.stub_kiss_web()
    sb.stub_browser("xdg-open")
    sb.url_file.write_text(_url_json(local="http://localhost:9999"))
    proc = sb.run(env={"DISPLAY": ":0"})
    assert proc.returncode == 0, proc.stderr
    assert sb.browser_calls() == ["http://127.0.0.1:9999"]


def test_no_browser_opener_prints_url_for_the_user(tmp_path: Path) -> None:
    """Linux without xdg-open, and an unknown OS, print the URL instead."""
    sb = _Sandbox(tmp_path)
    sb.stub_kiss_web()
    sb.url_file.write_text(_url_json(loopback=_LOOPBACK))
    proc = sb.run(env={"DISPLAY": ":0"})
    assert proc.returncode == 0, proc.stderr
    assert "Could not open a browser" in proc.stdout
    assert f"Opening the webapp at {_LOOPBACK}" in proc.stdout

    proc = sb.run(os_name="FreeBSD")
    assert proc.returncode == 0, proc.stderr
    assert "Could not open a browser" in proc.stdout


def test_kiss_skip_launch_skips_the_step(tmp_path: Path) -> None:
    """Docker (``KISS_SKIP_LAUNCH``) neither waits nor opens anything."""
    sb = _Sandbox(tmp_path)
    sb.stub_kiss_web()
    sb.stub_browser("xdg-open")
    sb.url_file.write_text(_url_json(loopback=_LOOPBACK))
    proc = sb.run(env={"DISPLAY": ":0", "KISS_SKIP_LAUNCH": "1"})
    assert proc.returncode == 0, proc.stderr
    assert sb.kiss_web_calls() == []
    assert sb.browser_calls() == []
    assert "Waiting for the kiss-web daemon" not in proc.stdout


def test_kiss_app_installs_the_launcher_and_opens_the_window(tmp_path: Path) -> None:
    """With kiss-app beside kiss-web: launcher registered, window opened, no browser used."""
    sb = _Sandbox(tmp_path)
    sb.stub_kiss_web()
    kiss_app = sb.stub_kiss_app()
    sb.stub_browser("xdg-open")
    sb.url_file.write_text(
        _url_json(local="http://localhost:8787", tunnel=_TUNNEL, loopback=_LOOPBACK),
    )
    proc = sb.run(env={"DISPLAY": ":0"})
    assert proc.returncode == 0, proc.stderr
    assert sb.kiss_web_calls() == []
    assert sb.kiss_app_calls() == sb.expected_kiss_app_calls()
    assert sb.browser_calls() == []
    assert f"Opening the webapp in its own window at {_LOOPBACK}" in proc.stdout
    assert "Opening the webapp at" not in proc.stdout
    assert str(kiss_app) not in proc.stdout


def test_kiss_app_in_the_app_venv_on_macos(tmp_path: Path) -> None:
    """kiss-app is found next to the venv's kiss-web; ``open`` is not used."""
    sb = _Sandbox(tmp_path)
    venv_bin = sb.app_dir / ".venv" / "bin"
    sb.stub_kiss_web(where=venv_bin / "kiss-web")
    sb.bin = venv_bin
    sb.stub_kiss_app()
    sb.bin = tmp_path / "bin"
    sb.stub_browser("open")
    sb.url_file.write_text(_url_json(loopback=_LOOPBACK))
    proc = sb.run(os_name="Darwin")
    assert proc.returncode == 0, proc.stderr
    assert sb.kiss_app_calls() == sb.expected_kiss_app_calls()
    assert sb.browser_calls() == []


def test_kiss_app_without_a_browser_falls_back_to_the_default_browser(tmp_path: Path) -> None:
    """kiss-app exiting non-zero on ``--url`` means no window: xdg-open gets the URL."""
    sb = _Sandbox(tmp_path)
    sb.stub_kiss_web()
    sb.stub_kiss_app(fail_on="--url")
    sb.stub_browser("xdg-open")
    sb.url_file.write_text(_url_json(loopback=_LOOPBACK))
    proc = sb.run(env={"DISPLAY": ":0"})
    assert proc.returncode == 0, proc.stderr
    assert sb.kiss_app_calls() == sb.expected_kiss_app_calls(revealed=False)
    assert sb.browser_calls() == [_LOOPBACK]
    warning = "   WARNING: The standalone window did not open; using the browser instead."
    assert warning in proc.stdout
    assert f"Opening the webapp at {_LOOPBACK}" in proc.stdout


def test_launcher_failure_is_reported_but_the_window_still_opens(tmp_path: Path) -> None:
    """A failed ``--install-launcher`` prints how to start the app; the window opens anyway."""
    sb = _Sandbox(tmp_path)
    sb.stub_kiss_web()
    kiss_app = sb.stub_kiss_app(fail_on="--install-launcher")
    sb.stub_browser("xdg-open")
    sb.url_file.write_text(_url_json(loopback=_LOOPBACK))
    proc = sb.run(env={"DISPLAY": ":0"})
    assert proc.returncode == 0, proc.stderr
    assert sb.kiss_app_calls() == sb.expected_kiss_app_calls(revealed=False)
    assert sb.browser_calls() == []
    hint = (
        "   WARNING: The desktop launcher was not installed; start the app with "
        f"'KISS_HOME={sb.kiss_home} {kiss_app}'."
    )
    assert hint in proc.stdout


def test_reveal_failure_is_a_warning_only(tmp_path: Path) -> None:
    """A failed ``--reveal-launcher`` (no file manager) is reported; the install still succeeds."""
    sb = _Sandbox(tmp_path)
    sb.stub_kiss_web()
    sb.stub_kiss_app(fail_on="--reveal-launcher")
    sb.stub_browser("xdg-open")
    sb.url_file.write_text(_url_json(loopback=_LOOPBACK))
    proc = sb.run(env={"DISPLAY": ":0"})
    assert proc.returncode == 0, proc.stderr
    assert sb.kiss_app_calls() == sb.expected_kiss_app_calls()
    assert sb.browser_calls() == []
    assert "   WARNING: Could not show the desktop launcher's folder." in proc.stdout


def test_remote_machine_never_runs_kiss_app(tmp_path: Path) -> None:
    """Over SSH there is no display for a window: only the tunnel URL is printed."""
    sb = _Sandbox(tmp_path)
    sb.stub_kiss_web()
    sb.stub_kiss_app()
    sb.url_file.write_text(_url_json(loopback=_LOOPBACK, tunnel=_TUNNEL))
    proc = sb.run(env={"DISPLAY": ":0", "SSH_CONNECTION": "1.2.3.4 1 5.6.7.8 22"})
    assert proc.returncode == 0, proc.stderr
    assert sb.kiss_app_calls() == []
    assert _TUNNEL in proc.stdout


def test_restarted_brand_daemon_gets_its_own_kiss_app_and_home(tmp_path: Path) -> None:
    """A stock install that restarted a brand's daemon opens that brand's app for its home.

    ``RESTART_HOME`` names the brand home (``~/.s10s``): its URL file is
    read, the ``kiss-app`` of its app venv (``~/.s10s/s10s``) runs with
    ``KISS_HOME`` set to it, and the stock app next to kiss-web is left
    alone.
    """
    sb = _Sandbox(tmp_path)
    sb.stub_kiss_web()
    sb.stub_kiss_app()  # the stock one, next to kiss-web
    sb.stub_browser("xdg-open")
    brand_home = sb.home / ".s10s"
    brand_app = sb.stub_kiss_app(where=brand_home / "s10s" / ".venv" / "bin" / "kiss-app")
    # Make the two stubs distinguishable: the brand's logs its own path too.
    brand_app.write_text(
        brand_app.read_text(encoding="utf-8").replace('"$*"', '"brand $*"'), encoding="utf-8"
    )
    (brand_home / "remote-url.json").write_text(_url_json(loopback="http://127.0.0.1:9999"))
    sb.url_file.write_text(_url_json(loopback=_LOOPBACK))  # the stock daemon's, not used
    proc = sb.run(env={"DISPLAY": ":0"}, prelude=f'RESTART_HOME="{brand_home}"')
    assert proc.returncode == 0, proc.stderr
    assert sb.kiss_app_calls() == [
        f"{brand_home}: brand --install-launcher",
        f"{brand_home}: brand --url http://127.0.0.1:9999 --wait 0",
        f"{brand_home}: brand --reveal-launcher",
    ]
    assert sb.browser_calls() == []
    assert "Opening the webapp in its own window at http://127.0.0.1:9999" in proc.stdout


def test_custom_kiss_home_is_passed_to_kiss_app(tmp_path: Path) -> None:
    """``KISS_HOME`` of the install (not the brand default) is the home kiss-app runs for."""
    sb = _Sandbox(tmp_path)
    sb.stub_kiss_web()
    sb.stub_kiss_app()
    custom = tmp_path / "custom-home"
    custom.mkdir()
    (custom / "remote-url.json").write_text(_url_json(loopback=_LOOPBACK))
    proc = sb.run(env={"DISPLAY": ":0", "KISS_HOME": str(custom)})
    assert proc.returncode == 0, proc.stderr
    assert sb.kiss_app_calls() == sb.expected_kiss_app_calls(custom)


def test_failures_are_recorded_for_the_web_app_notification(tmp_path: Path) -> None:
    """Launcher and window failures are warnings: printed and appended to PROBLEMS_FILE."""
    sb = _Sandbox(tmp_path)
    sb.stub_kiss_web()
    sb.stub_kiss_app(fail_on="--url")
    sb.stub_browser("xdg-open")
    sb.url_file.write_text(_url_json(loopback=_LOOPBACK))
    problems = tmp_path / "problems.txt"
    proc = sb.run(env={"DISPLAY": ":0"}, prelude=f'PROBLEMS_FILE="{problems}"')
    assert proc.returncode == 0, proc.stderr
    assert problems.read_text(encoding="utf-8") == (
        "WARNING: The standalone window did not open; using the browser instead.\n"
    )
    assert sb.browser_calls() == [_LOOPBACK]
