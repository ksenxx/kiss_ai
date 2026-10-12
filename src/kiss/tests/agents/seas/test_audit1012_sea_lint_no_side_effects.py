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
import signal
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

DAEMON_MODULE = "kiss.agents.third_party_agents.muse_auth.daemon"


def muse_daemons_of(home: Path) -> list[int]:
    """Return the pids of every ``muse_auth.daemon`` process whose ``KISS_HOME`` is *home*.

    Args:
        home: The ``KISS_HOME`` directory the lint ran against.

    Returns:
        The matching pids, from ``/proc``; always empty where there is
        no ``/proc`` (the on-disk assertions still apply there).
    """
    wanted = f"KISS_HOME={home}".encode()
    pids: list[int] = []
    if not Path("/proc").is_dir():
        return pids
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            cmdline = Path(f"/proc/{entry}/cmdline").read_bytes()
            environ = Path(f"/proc/{entry}/environ").read_bytes()
        except OSError:
            continue
        if DAEMON_MODULE.encode() in cmdline and wanted in environ.split(b"\0"):
            pids.append(int(entry))
    return pids


@pytest.fixture
def fresh_home(tmp_path: Path) -> Iterator[Path]:
    """Yield an empty ``KISS_HOME`` and kill any daemon the lint left under it."""
    home = tmp_path / "home"
    home.mkdir()
    yield home
    for pid in muse_daemons_of(home):
        os.kill(pid, signal.SIGKILL)


def run_sea(home: Path, *argv: str, muse_auth: str | None) -> subprocess.CompletedProcess[str]:
    """Run ``sea <argv>`` in a subprocess against *home*.

    Args:
        home: The ``KISS_HOME`` for the run.
        *argv: The ``sea`` sub-command and its arguments.
        muse_auth: Value for ``KISS_MUSE_AUTH``; ``None`` removes the
            variable so the platform default applies, as under
            ``uv run check --full``.

    Returns:
        The completed process (stdout and stderr captured).
    """
    env = {k: v for k, v in os.environ.items() if k != "KISS_MUSE_AUTH"}
    env["KISS_HOME"] = str(home)
    if muse_auth is not None:
        env["KISS_MUSE_AUTH"] = muse_auth
    return subprocess.run(
        [sys.executable, "-m", "kiss.agents.sorcar.sea_cli", *argv],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )


def assert_ran_clean(result: subprocess.CompletedProcess[str], expected_stdout: str) -> None:
    """Assert the ``sea`` subprocess really ran its command to completion.

    A Python that fails before importing Sorcar (a broken environment)
    exits non-zero with a traceback and an empty stdout, which would
    make a daemon/state check pass vacuously; so the command must have
    exited 0, printed its own summary line, and raised nothing.

    Args:
        result: The completed ``sea`` subprocess.
        expected_stdout: Text the command prints only after doing its
            work (empty for ``docs --check``, which is silent on success).
    """
    output = result.stdout + result.stderr
    assert "Traceback" not in output, output
    assert result.returncode == 0, output
    assert expected_stdout in result.stdout, output


@pytest.mark.skipif(sys.platform == "win32", reason="the Muse daemon only runs on Unix")
def test_sea_lint_starts_no_muse_daemon(fresh_home: Path) -> None:
    """``sea lint`` with Muse-auth at its default leaves no daemon and no vault state."""
    assert muse_daemons_of(fresh_home) == []
    baseline = run_sea(fresh_home, "lint", muse_auth="0")
    assert_ran_clean(baseline, "sea lint: 0 finding(s)")
    assert not (fresh_home / "muse_auth").exists(), "Muse-off lint must not touch muse_auth"

    result = run_sea(fresh_home, "lint", muse_auth=None)

    daemons = muse_daemons_of(fresh_home)
    state = sorted(str(p.relative_to(fresh_home)) for p in fresh_home.rglob("muse_auth*"))
    assert daemons == [], f"sea lint spawned muse_auth.daemon pid(s) {daemons}; state: {state}"
    assert not (fresh_home / "muse_auth").exists(), f"sea lint wrote {state}"
    assert_ran_clean(result, "sea lint: 0 finding(s)")
    assert result.stdout.strip() == baseline.stdout.strip()


@pytest.mark.skipif(sys.platform == "win32", reason="the Muse daemon only runs on Unix")
def test_sea_docs_starts_no_muse_daemon(fresh_home: Path) -> None:
    """``sea docs`` runs the same hooks as the lint and must stay side-effect free too."""
    baseline = run_sea(fresh_home, "docs", "--check", muse_auth="0")
    assert_ran_clean(baseline, "")
    result = run_sea(fresh_home, "docs", "--check", muse_auth=None)

    daemons = muse_daemons_of(fresh_home)
    assert daemons == [], f"sea docs spawned muse_auth.daemon pid(s) {daemons}"
    assert not (fresh_home / "muse_auth").exists()
    assert_ran_clean(result, "")
    assert result.stdout.strip() == baseline.stdout.strip()
