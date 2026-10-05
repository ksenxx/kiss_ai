# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: ``install.sh`` MUST end by opening the webapp.

The ``kiss-open-webapp`` block of ``install.sh`` waits for the kiss-web
daemon (started by the extension inside VS Code), runs
``kiss-web --trust-ca`` and then opens the Local URL
(``https://127.0.0.1:PORT``) in the default browser, or prints the
cloudflared URL when the machine is remote (an SSH session, or Linux
without a display).

The block (together with the ``kiss-browser-helpers`` block it relies
on) is extracted verbatim and run under ``bash -euo pipefail`` against a
temp ``HOME`` / ``KISS_HOME`` with stub ``kiss-web``, ``xdg-open`` and
``open`` executables that log their arguments.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

from kiss.tests.conftest import posix_only

_REPO = Path(__file__).resolve().parents[3]

pytestmark = posix_only("runs the open-webapp block of install.sh under bash")

_LOOPBACK = "https://127.0.0.1:8787"
_TUNNEL = "https://example-tunnel.trycloudflare.com"


def install_sh_block(name: str) -> str:
    """Return the ``# BEGIN: <name>`` ... ``# END: <name>`` block of install.sh."""
    text = (_REPO / "install.sh").read_text(encoding="utf-8")
    start = text.index(f"# BEGIN: {name}")
    end = text.index(f"# END: {name}")
    return text[start:end]


def _open_webapp_block() -> str:
    # The block uses machine_is_remote / open_in_browser from the shared
    # helpers block that precedes it in install.sh.
    return install_sh_block("kiss-browser-helpers") + "\n" + install_sh_block("kiss-open-webapp")


def _write_stub(path: Path, log: Path, exit_code: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'#!/bin/bash\nprintf "%s\\n" "$*" >> "{log}"\nexit {exit_code}\n',
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

    def stub_kiss_web(self, exit_code: int = 0, where: Path | None = None) -> None:
        _write_stub(where or self.bin / "kiss-web", self.kiss_web_log, exit_code)

    def stub_browser(self, name: str) -> None:
        _write_stub(self.bin / name, self.browser_log)

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
            f'OS="{os_name}"\nPROJECT_DIR="{self.home}/no-project"\n'
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


def test_local_linux_trusts_ca_then_opens_loopback_url(tmp_path: Path) -> None:
    """Local Linux desktop: ``kiss-web --trust-ca`` then xdg-open 127.0.0.1."""
    sb = _Sandbox(tmp_path)
    sb.stub_kiss_web()
    sb.stub_browser("xdg-open")
    sb.url_file.write_text(_url_json(
        local="https://localhost:8787", tunnel=_TUNNEL, loopback=_LOOPBACK,
    ))
    proc = sb.run(env={"DISPLAY": ":0"})
    assert proc.returncode == 0, proc.stderr
    assert sb.kiss_web_calls() == ["--trust-ca"]
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
    assert sb.kiss_web_calls() == ["--trust-ca"]
    assert sb.browser_calls() == [_LOOPBACK]


def test_ssh_session_prints_tunnel_url_and_opens_no_browser(tmp_path: Path) -> None:
    """A remote machine gets the cloudflared URL printed instead."""
    sb = _Sandbox(tmp_path)
    sb.stub_kiss_web()
    sb.stub_browser("xdg-open")
    sb.url_file.write_text(_url_json(loopback=_LOOPBACK, tunnel=_TUNNEL))
    proc = sb.run(env={"DISPLAY": ":0", "SSH_CONNECTION": "1.2.3.4 1 5.6.7.8 22"})
    assert proc.returncode == 0, proc.stderr
    assert sb.kiss_web_calls() == ["--trust-ca"]
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
    assert f"'{kiss_web}' --trust-ca && '{kiss_web}' --url" in proc.stdout


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
    assert sb.kiss_web_calls() == ["--trust-ca"]
    assert sb.browser_calls() == [_LOOPBACK]


def test_missing_kiss_web_binary_times_out(tmp_path: Path) -> None:
    """A URL file without any kiss-web binary is not enough."""
    sb = _Sandbox(tmp_path)
    sb.stub_browser("xdg-open")
    sb.url_file.write_text(_url_json(loopback=_LOOPBACK))
    proc = sb.run(env={"DISPLAY": ":0"})
    assert proc.returncode == 0, proc.stderr
    assert "did not come up within 0s" in proc.stdout
    assert "'kiss-web' --trust-ca && 'kiss-web' --url" in proc.stdout
    assert sb.browser_calls() == []


def test_kiss_web_found_in_installed_extension_venv(tmp_path: Path) -> None:
    """Off PATH, kiss-web is located in the extension's kiss_project venv."""
    sb = _Sandbox(tmp_path)
    ext_bin = (
        sb.home / ".vscode" / "extensions" / "ksenxx.kiss-sorcar-2026.9.25"
        / "kiss_project" / ".venv" / "bin" / "kiss-web"
    )
    sb.stub_kiss_web(where=ext_bin)
    sb.stub_browser("xdg-open")
    sb.url_file.write_text(_url_json(loopback=_LOOPBACK))
    proc = sb.run(env={"DISPLAY": ":0"})
    assert proc.returncode == 0, proc.stderr
    assert sb.kiss_web_calls() == ["--trust-ca"]
    assert sb.browser_calls() == [_LOOPBACK]


def test_trust_ca_failure_warns_but_still_opens(tmp_path: Path) -> None:
    """A failing ``--trust-ca`` (no certutil, say) is not fatal."""
    sb = _Sandbox(tmp_path)
    sb.stub_kiss_web(exit_code=3)
    sb.stub_browser("xdg-open")
    sb.url_file.write_text(_url_json(loopback=_LOOPBACK))
    proc = sb.run(env={"DISPLAY": ":0"})
    assert proc.returncode == 0, proc.stderr
    assert "WARNING: 'kiss-web --trust-ca' failed" in proc.stdout
    assert sb.browser_calls() == [_LOOPBACK]


def test_old_url_file_without_loopback_falls_back_to_127_0_0_1(tmp_path: Path) -> None:
    """A daemon that only wrote ``local`` still yields a 127.0.0.1 URL."""
    sb = _Sandbox(tmp_path)
    sb.stub_kiss_web()
    sb.stub_browser("xdg-open")
    sb.url_file.write_text(_url_json(local="https://localhost:9999"))
    proc = sb.run(env={"DISPLAY": ":0"})
    assert proc.returncode == 0, proc.stderr
    assert sb.browser_calls() == ["https://127.0.0.1:9999"]


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
