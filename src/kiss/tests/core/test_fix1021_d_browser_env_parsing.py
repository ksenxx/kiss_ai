# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``$BROWSER`` means the same thing to the hand-off and the browser tab (audit 2026-10-21, D-F3).

``BROWSER=firefox:chromium`` (the :mod:`webbrowser` list form) used to be
launched entry by entry by ``open_in_default_browser`` while
``default_browser_executable`` looked up a single executable called
``firefox:chromium``, found nothing and reported that no default browser
was configured.  Both now go through :func:`browser_commands_from_env`.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import uuid
from pathlib import Path

import pytest

from kiss.core import browser_handoff, default_browser
from kiss.tests.conftest import install_cli_script

URL = "https://example.test/signin?a=1&b=2"

# A stand-in browser: records its arguments in $FIX1021_LAUNCH_LOG and
# exits 0.  It names this interpreter outright: the stand-ins sit alone
# on $PATH, where ``env`` would find no python3.
_RECORDING_BROWSER = f"""#!{Path(sys.executable).as_posix()}
import json, os, sys
with open(os.environ.get("FIX1021_LAUNCH_LOG", os.devnull), "a") as log:
    log.write(json.dumps(sys.argv[1:]) + "\\n")
"""


@pytest.fixture()
def stand_ins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Put stand-in ``firefox`` and ``chromium`` executables alone on ``$PATH``."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("firefox", "chromium"):
        install_cli_script(bin_dir / name, _RECORDING_BROWSER)
    monkeypatch.setenv("PATH", str(bin_dir))
    monkeypatch.delenv("KISS_BROWSER", raising=False)
    found = {name: shutil.which(name) for name in ("firefox", "chromium")}
    assert all(found.values()), found
    return {name: _resolved(exe) for name, exe in found.items()}


def _resolved(executable: str | None) -> str:
    """Canonical form of an executable path for comparison.

    ``shutil.which`` on Windows spells the extension as ``PATHEXT`` does
    (``firefox.CMD``); resolving restores the on-disk ``firefox.cmd``.
    """
    return str(Path(executable or "").resolve())


def _set_browser(monkeypatch: pytest.MonkeyPatch, *entries: str) -> None:
    monkeypatch.setenv("BROWSER", os.pathsep.join(entries))


def test_list_form_is_one_command_per_entry(
    stand_ins: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_browser(monkeypatch, "firefox", "chromium --new-window %s")
    assert default_browser.browser_commands_from_env() == [
        ["firefox"],
        ["chromium", "--new-window", "%s"],
    ]
    assert browser_handoff._launch_commands(URL) == [
        ["firefox", URL],
        ["chromium", "--new-window", URL],
    ]
    # The browser tab resolves to the first entry, the one a hand-off launches first.
    assert _resolved(default_browser.default_browser_executable()) == stand_ins["firefox"]


def test_first_installed_entry_wins(
    stand_ins: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_browser(monkeypatch, "no-such-browser-fix1021", "chromium")
    assert browser_handoff._launch_commands(URL) == [
        ["no-such-browser-fix1021", URL],
        ["chromium", URL],
    ]
    assert _resolved(default_browser.default_browser_executable()) == stand_ins["chromium"]
    resolved = default_browser.resolve_browser()
    assert (_resolved(resolved.executable), resolved.is_default) == (stand_ins["chromium"], True)


def test_nothing_installed_is_no_default_not_a_platform_probe(
    stand_ins: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_browser(monkeypatch, "no-such-browser-fix1021")
    assert browser_handoff._launch_commands(URL) == [["no-such-browser-fix1021", URL]]
    assert default_browser.default_browser_executable() is None


def test_malformed_and_blank_entries_are_dropped(
    stand_ins: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_browser(monkeypatch, '"unbalanced', "", "  ", "chromium")
    assert default_browser.browser_commands_from_env() == [["chromium"]]
    assert browser_handoff._launch_commands(URL) == [["chromium", URL]]
    assert _resolved(default_browser.default_browser_executable()) == stand_ins["chromium"]
    _set_browser(monkeypatch, '"unbalanced')
    assert default_browser.browser_commands_from_env() == []
    assert browser_handoff._launch_commands(URL) == []


def test_public_open_path_launches_the_first_entry_that_starts(
    stand_ins: dict[str, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``open_in_default_browser`` skips the missing entry and runs the next with the URL."""
    log = tmp_path / "launch-log"
    monkeypatch.setenv("FIX1021_LAUNCH_LOG", str(log))
    monkeypatch.setenv("KISS_HEADLESS", "0")
    # The full path (quoted: it may contain spaces) stands for the
    # executable itself, as Popen cannot run a Windows ``.cmd`` by name.
    _set_browser(monkeypatch, "no-such-browser-fix1021", f'"{shutil.which("chromium")}" %s')
    url = f"{URL}&attempt={uuid.uuid4()}"
    assert browser_handoff.open_in_default_browser(url) is True
    assert [json.loads(line) for line in log.read_text().splitlines()] == [[url]]
    # The same page is not opened again while it counts as recent.
    assert browser_handoff.open_in_default_browser(url) is True
    assert len(log.read_text().splitlines()) == 1


def test_env_wrapper_is_skipped_for_the_executable(
    stand_ins: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_browser(monkeypatch, "env MOZ_X=1 firefox %s")
    assert browser_handoff._launch_commands(URL) == [["env", "MOZ_X=1", "firefox", URL]]
    assert _resolved(default_browser.default_browser_executable()) == stand_ins["firefox"]
