"""``sea lint`` must not start daemons or write credential state.

Regression test for the 2026-10-12 audit finding: ``sea lint`` (run by
``uv run check --full``) and ``sea docs`` exercise every bundled SEA's
``tools()`` hook through ``sea_commands.check_sea``; the Muse-aware
channel SEAs used to mint a vault surrogate while *constructing* their
agent, which spawned a ``muse_auth.daemon`` under ``$KISS_HOME`` (or the
stale ``~/.kiss`` when ``KISS_HOME`` was unset) as a side effect of the
lint.  The lint runs as a subprocess against a fresh ``KISS_HOME`` with
Muse-auth at its platform default, and the test asserts that no
``muse_auth`` state and no daemon process appeared.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from kiss.agents.sorcar.sea_docs import TABLES
from kiss.tests.agents.third_party_agents.muse_test_utils import muse_daemons_of

skip_on_windows = pytest.mark.skipif(
    sys.platform == "win32", reason="the Muse daemon only runs on Unix"
)


@pytest.fixture
def fresh_home(tmp_path: Path) -> Iterator[Path]:
    """Yield an empty ``KISS_HOME`` and kill any daemon the lint left under it."""
    home = tmp_path / "home"
    home.mkdir()
    yield home
    for pid in muse_daemons_of(home):
        os.kill(pid, signal.SIGKILL)


def run_sea_cli(home: Path, *argv: str) -> subprocess.CompletedProcess[str]:
    """Run ``sea <argv>`` in a subprocess against *home*.

    ``KISS_MUSE_AUTH`` is removed from the environment so the platform
    default applies, as under ``uv run check --full``.

    Args:
        home: The ``KISS_HOME`` for the run.
        *argv: The ``sea`` sub-command and its arguments.

    Returns:
        The completed process (stdout and stderr captured).
    """
    env = {k: v for k, v in os.environ.items() if k != "KISS_MUSE_AUTH"}
    env["KISS_HOME"] = str(home)
    return subprocess.run(
        [sys.executable, "-m", "kiss.agents.sorcar.sea_cli", *argv],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )


def assert_ran_without_muse(home: Path, argv: tuple[str, ...], stdout: str) -> None:
    """Run ``sea <argv>`` against *home*; assert it left no daemon and no vault state.

    The command must really have run to completion: a Python that fails
    before importing Sorcar (a broken environment) exits non-zero with a
    traceback and an empty stdout, which would make the daemon/state
    check pass vacuously.  So it exited 0, raised nothing and printed
    exactly its own summary line (*stdout*, a pattern): a Muse-auth
    banner or warning on top of it would show the hooks did more than
    their job.
    """
    assert muse_daemons_of(home) == set()
    result = run_sea_cli(home, *argv)

    daemons = muse_daemons_of(home)
    state = sorted(str(p.relative_to(home)) for p in home.rglob("muse_auth*"))
    assert daemons == set(), f"sea {argv[0]} spawned muse_auth.daemon {daemons}; state: {state}"
    assert not (home / "muse_auth").exists(), f"sea {argv[0]} wrote {state}"
    output = result.stdout + result.stderr
    assert "Traceback" not in output, output
    assert result.returncode == 0, output
    assert re.fullmatch(stdout, result.stdout.strip()), output


@skip_on_windows
def test_sea_lint_starts_no_muse_daemon(fresh_home: Path) -> None:
    """``sea lint`` with Muse-auth at its default leaves no daemon and no vault state."""
    assert_ran_without_muse(fresh_home, ("lint",), r"sea lint: \d+ finding\(s\)")


@skip_on_windows
def test_sea_docs_starts_no_muse_daemon(fresh_home: Path, tmp_path: Path) -> None:
    """``sea docs`` with Muse-auth at its default leaves no daemon and no vault state.

    It renders a page of the test's own with one empty block per table
    (``commands`` loads every bundled SEA, and none may fail to load), so
    the run does not depend on whether the repository's generated pages
    are current.
    """
    page = tmp_path / "page.md"
    blocks = "".join(f"<!-- sea-docs: {name} -->\n<!-- /sea-docs -->\n" for name in TABLES)
    page.write_text(blocks, encoding="utf-8")
    assert_ran_without_muse(fresh_home, ("docs", str(page)), re.escape(f"updated: {page}"))
    rendered = page.read_text(encoding="utf-8")
    assert "| `/autorouter` | `agents/seas/autorouter/autorouter_sea.py` |" in rendered
    assert "(broken:" not in rendered, rendered
