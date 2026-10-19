# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""``resolve_browser()`` promises never to raise, but an unbalanced quote in
``$BROWSER`` or in a ``.desktop`` ``Exec=`` line made ``shlex.split`` raise
``ValueError`` and every Browser-tab launch failed (finding conc1018 A2-F2).

Each case runs in a fresh interpreter with a controlled environment: the
module reads ``$BROWSER`` and shells out to ``xdg-settings``, so no
monkeypatching is needed."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

_PROBE = (
    "from kiss.core.default_browser import default_browser_executable, resolve_browser\n"
    "print(repr(default_browser_executable()))\n"
    "print(resolve_browser().name)\n"
)


def _run_probe(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", _PROBE],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_malformed_browser_env_falls_back_instead_of_raising() -> None:
    proc = _run_probe({**os.environ, "BROWSER": '"x', "KISS_BROWSER": ""})
    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.splitlines()
    assert lines[-1].strip() != ""


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason=".desktop lookup is Linux-only")
def test_malformed_desktop_exec_line_is_skipped(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    xdg_settings = bin_dir / "xdg-settings"
    xdg_settings.write_text("#!/bin/sh\necho conc1018-bad.desktop\n", encoding="utf-8")
    xdg_settings.chmod(0o755)
    apps = tmp_path / "share" / "applications"
    apps.mkdir(parents=True)
    (apps / "conc1018-bad.desktop").write_text('[Desktop Entry]\nExec="x %U\n', encoding="utf-8")
    env = {
        **os.environ,
        "BROWSER": "",
        "KISS_BROWSER": "",
        "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        "XDG_DATA_HOME": str(tmp_path / "share"),
        "XDG_DATA_DIRS": str(tmp_path / "empty"),
    }
    proc = _run_probe(env)
    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.splitlines()
    # The malformed Exec line yields no executable; the fallback name lookup
    # ("conc1018-bad") is not on PATH either, so no default browser.
    assert lines[0] == "None"
    assert lines[-1].strip() != ""
