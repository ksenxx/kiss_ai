# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The legacy-key migration spawns the user's shell at most once per store.

``_migrate_legacy_rc_keys`` used to ``source`` the user's RC on every
``load_api_keys()`` (every daemon or channel-agent start) as long as any
one of the sixteen ``API_KEY_ENV_VARS`` was absent from
``api_keys.env`` — which is every real install.  Each start therefore
ran arbitrary RC code for up to five seconds while holding the config
lock and the store's cross-process flock.

The migration now sources the RC only when some RC *mentions* a key
the textual scan could not import, and records in the store that it
did (a comment line), so later starts never spawn the shell again.

Every scenario runs ``load_api_keys()`` in a child Python process with
``HOME``/``KISS_HOME`` in a temp dir and a ``.bashrc`` that leaves a
``sourced-<pid>`` file behind each time it is sourced: the number of
those files is the number of shells spawned.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from kiss.core.user_config import API_KEY_ENV_VARS
from kiss.tests.conftest import posix_only

pytestmark = posix_only("legacy shell-RC key migration sources ~/.bashrc via bash")

_LOADER = r"""
import os, sys
from kiss.core.user_config import load_api_keys
for _ in range(int(sys.argv[1])):
    load_api_keys()
print(os.environ.get("GEMINI_API_KEY", "") + "|" + os.environ.get("OPENAI_API_KEY", ""))
"""

_TOUCH = 'touch "$HOME/sourced-$$"\n'


def _run_loader(tmp_path: Path, starts: int) -> str:
    """Run ``load_api_keys()`` *starts* times in one fresh process; return its stdout."""
    env = dict(os.environ)
    env["HOME"] = str(tmp_path / "home")
    env["KISS_HOME"] = str(tmp_path / "kiss")
    env["SHELL"] = "/bin/bash"
    for name in API_KEY_ENV_VARS:
        env.pop(name, None)
    done = subprocess.run(
        [sys.executable, "-c", _LOADER, str(starts)],
        env=env, capture_output=True, text=True, timeout=120, check=False,
    )
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


def _sourced_count(tmp_path: Path) -> int:
    return len(list((tmp_path / "home").glob("sourced-*")))


@pytest.fixture
def home(tmp_path: Path) -> Path:
    (tmp_path / "home").mkdir()
    (tmp_path / "kiss").mkdir()
    return tmp_path / "home"


def test_rc_that_names_no_key_is_never_sourced(tmp_path: Path, home: Path) -> None:
    (home / ".bashrc").write_text(_TOUCH)
    _run_loader(tmp_path, 2)
    assert _sourced_count(tmp_path) == 0


def test_key_the_textual_scan_imports_needs_no_shell(tmp_path: Path, home: Path) -> None:
    (home / ".bashrc").write_text(_TOUCH + "export GEMINI_API_KEY=plain\n")
    assert _run_loader(tmp_path, 2) == "plain|"
    assert _sourced_count(tmp_path) == 0
    assert "export GEMINI_API_KEY=plain" in (tmp_path / "kiss" / "api_keys.env").read_text()


def test_indirect_key_is_sourced_once_and_stays_migrated(tmp_path: Path, home: Path) -> None:
    (home / ".bashrc").write_text(_TOUCH + 'V=legacy\nexport GEMINI_API_KEY="${V}-value"\n')
    assert _run_loader(tmp_path, 2) == "legacy-value|"
    assert _run_loader(tmp_path, 1) == "legacy-value|"
    assert _sourced_count(tmp_path) == 1


def test_key_named_but_not_set_is_sourced_once_only(tmp_path: Path, home: Path) -> None:
    (home / ".bashrc").write_text(_TOUCH + 'if [ -z "$OPENAI_API_KEY" ]; then :; fi\n')
    assert _run_loader(tmp_path, 2) == "|"
    assert _run_loader(tmp_path, 1) == "|"
    assert _sourced_count(tmp_path) == 1


def test_marker_lands_on_its_own_line_in_a_store_without_final_newline(
    tmp_path: Path, home: Path,
) -> None:
    store = tmp_path / "kiss" / "api_keys.env"
    store.write_text("export OPENAI_API_KEY=kept")
    (home / ".bashrc").write_text(_TOUCH + "# GEMINI_API_KEY lives elsewhere\n")
    assert _run_loader(tmp_path, 2) == "|kept"
    assert _sourced_count(tmp_path) == 1
    lines = store.read_text().splitlines()
    assert lines[0] == "export OPENAI_API_KEY=kept"
    assert len(lines) == 2 and lines[1].startswith("#")
