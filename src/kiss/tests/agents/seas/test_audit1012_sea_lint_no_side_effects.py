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

from kiss.tests.agents.third_party_agents.muse_test_utils import muse_daemons_of


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


@pytest.mark.skipif(sys.platform == "win32", reason="the Muse daemon only runs on Unix")
@pytest.mark.parametrize(
    ("argv", "stdout"),
    [(("lint",), r"sea lint: \d+ finding\(s\)"), (("docs", "--check"), "")],
    ids=["lint", "docs"],
)
def test_sea_cli_starts_no_muse_daemon(
    fresh_home: Path, argv: tuple[str, ...], stdout: str
) -> None:
    """``sea lint`` / ``sea docs`` with Muse-auth at its default leave no daemon and no vault state.

    The command must really have run to completion: a Python that fails
    before importing Sorcar (a broken environment) exits non-zero with a
    traceback and an empty stdout, which would make the daemon/state
    check pass vacuously.  So it exited 0, raised nothing and printed
    exactly its own summary line (*stdout*, a pattern; empty for ``docs
    --check``, which is silent on success): a Muse-auth banner or
    warning on top of it would show the hooks did more than lint.
    """
    assert muse_daemons_of(fresh_home) == set()
    result = run_sea_cli(fresh_home, *argv)

    daemons = muse_daemons_of(fresh_home)
    state = sorted(str(p.relative_to(fresh_home)) for p in fresh_home.rglob("muse_auth*"))
    assert daemons == set(), f"sea {argv[0]} spawned muse_auth.daemon {daemons}; state: {state}"
    assert not (fresh_home / "muse_auth").exists(), f"sea {argv[0]} wrote {state}"
    output = result.stdout + result.stderr
    assert "Traceback" not in output, output
    assert result.returncode == 0, output
    assert re.fullmatch(stdout, result.stdout.strip()), output
