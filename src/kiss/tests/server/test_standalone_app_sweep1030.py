# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``kiss-app`` after the 2026-10-30 sweep.

1. The daemon probes (``daemon_answers``, ``ask_app_window_open``) went
   through ``urllib.request.urlopen``, which honours ``http_proxy``: on a
   machine with a proxy and no matching ``no_proxy`` every probe of the
   daemon's 127.0.0.1 URL went to the proxy and failed, so the window
   sat on the "waiting for the daemon" page although the daemon was up.
2. ``--print-command`` printed ``" ".join(command)``: a browser or
   profile path with a space (``Google Chrome.app``, a home folder
   with a space) pasted into a shell split into several arguments.  It
   is now quoted with ``shlex.join`` (``subprocess.list2cmdline`` on
   Windows, whose branch this POSIX-only test cannot reach: the stub
   browser is a bash script).
"""

from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path

from kiss.server import standalone_app
from kiss.tests.conftest import find_free_port, posix_only
from kiss.tests.test_standalone_app import (
    _app_flag,
    _env,
    _Hits,
    _run,
    _site,
    _stub_calls,
    _write_stub_browser,
)

pytestmark = posix_only("runs the kiss-app console script and stub browsers under bash")


def _proxied_env(tmp_path: Path, **extra: str) -> dict[str, str]:
    """``_env`` plus a proxy that nothing listens on and no ``no_proxy`` exemption."""
    env = _env(tmp_path, **extra)
    for name in ("no_proxy", "NO_PROXY"):
        env.pop(name, None)
    dead_proxy = f"http://127.0.0.1:{find_free_port()}"
    for name in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY"):
        env[name] = dead_proxy
    return env


def test_a_proxy_in_the_environment_does_not_hide_the_running_daemon(tmp_path: Path) -> None:
    log = tmp_path / "browser.log"
    stub = _write_stub_browser(tmp_path / "bin" / "chrome", log)
    httpd, url = _site()
    _Hits.app_window_open = True
    try:
        env = _proxied_env(tmp_path, KISS_APP_BROWSER=str(stub))
        proc = _run(["--foreground", "--url", url, "--wait", "0"], env)
        assert proc.returncode == 0, proc.stderr
        assert _stub_calls(log)[0] == _app_flag(url), "the app itself, not the start page"
        asked = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; from kiss.server.standalone_app import ask_app_window_open; "
                "print(ask_app_window_open(sys.argv[1], timeout=10))",
                url,
            ],
            capture_output=True, text=True, env=env, timeout=60,
        )
        assert asked.stdout.strip() == "True", asked.stderr
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_print_command_quotes_paths_with_spaces_for_the_shell(tmp_path: Path) -> None:
    log = tmp_path / "browser log"
    stub = _write_stub_browser(tmp_path / "Google Chrome.app" / "chrome", log)
    url = "http://127.0.0.1:8787"
    env = _env(tmp_path, KISS_APP_BROWSER=str(stub), KISS_HOME=str(tmp_path / "kiss home"))
    proc = _run(["--print-command", "--url", url], env)
    assert proc.returncode == 0, proc.stderr
    argv = shlex.split(proc.stdout)
    assert argv[:2] == [str(stub), _app_flag(url)]
    profile = f"--user-data-dir={tmp_path / 'kiss home' / standalone_app.PROFILE_DIR_NAME}"
    assert profile in argv
    # Pasted into a shell, the printed line runs the browser with exactly those arguments.
    ran = subprocess.run(["/bin/sh", "-c", proc.stdout], capture_output=True, text=True, timeout=60)
    assert ran.returncode == 0, ran.stderr
    assert _stub_calls(log) == argv[1:]
