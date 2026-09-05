# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the ``sorcar`` console script entry point.

Regression suite for the ``uv run was recursively invoked 101 times``
bug: the ``sorcar`` REPL CLI was deleted (commit 1a782e0ab) together
with its ``[project.scripts]`` entry, but the VS Code extension kept
installing a ``~/.local/bin/sorcar`` wrapper running ``uv run …
sorcar``.  With no project script named ``sorcar``, uv fell back to
PATH, found the wrapper itself, and recursed until uv's limit.  The fix
adds :func:`kiss.agents.sorcar.sorcar_agent.main` and points the
``sorcar`` project script at it, so ``uv run … sorcar`` resolves inside
the project again.

Every test here launches a REAL subprocess running ``main()`` (the
exact code path of the installed console script) — no mocks, patches,
or fakes.  The two tests that reach ``agent.run`` use the cheap
``gpt-4o-mini`` stand-in model, like the other e2e suites in this
folder.

Branch-coverage notes for ``main()``:

* ``except Exception`` around ``yaml.safe_load(result)`` is unreachable
  without test doubles: ``RelentlessAgent.run`` always returns
  ``yaml.dump`` of a dict payload (its ``finish`` tool and every error
  path build the payload as a dict), so the parse can neither raise nor
  yield a non-dict.  Documented here instead of mocked, per the no-test-
  doubles policy.
* ``_ask_user_in_terminal``'s two branches (typed reply, EOF) are
  covered by real subprocesses with piped/closed stdin.
"""

from __future__ import annotations

import os
import pty
import subprocess
import sys
from pathlib import Path

import pytest

# Runs main() exactly as the installed `sorcar` console script does.
_BOOTSTRAP = "from kiss.agents.sorcar.sorcar_agent import main; main()"

_HAS_OPENAI_KEY = bool(os.getenv("OPENAI_API_KEY"))


def _run_cli(
    args: list[str],
    *,
    env: dict[str, str],
    cwd: str,
    stdin: int | None = subprocess.DEVNULL,
    input_text: str | None = None,
    timeout: float = 600.0,
) -> subprocess.CompletedProcess[str]:
    """Run ``main()`` in a real subprocess with the given CLI arguments.

    Args:
        args: Command-line arguments for the ``sorcar`` script.
        env: Full environment for the subprocess.
        cwd: Working directory for the subprocess.
        stdin: Stdin redirection when *input_text* is None.
        input_text: Text piped to the subprocess's stdin.
        timeout: Seconds to wait before failing the test.

    Returns:
        The completed process with captured text output.
    """
    return subprocess.run(
        [sys.executable, "-c", _BOOTSTRAP, *args],
        env=env,
        cwd=cwd,
        stdin=stdin if input_text is None else None,
        input=input_text,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _base_env(kiss_home: Path) -> dict[str, str]:
    """Return a copy of the environment isolated to a temp KISS_HOME.

    Args:
        kiss_home: Temp directory used as ``$KISS_HOME`` so the
            subprocess never touches the developer's ``~/.kiss``.

    Returns:
        The environment mapping for the subprocess.
    """
    env = dict(os.environ)
    env["KISS_HOME"] = str(kiss_home)
    env.pop("KISS_WORKDIR", None)
    return env


class TestArgumentHandling:
    """Cheap tests that never reach the LLM."""

    def test_help_exits_zero(self, tmp_path: Path) -> None:
        proc = _run_cli(["--help"], env=_base_env(tmp_path), cwd=str(tmp_path))
        assert proc.returncode == 0
        assert "Run the KISS SorcarAgent on a task." in proc.stdout
        assert "--max-budget" in proc.stdout

    def test_no_task_with_piped_empty_stdin_errors(self, tmp_path: Path) -> None:
        # stdin is not a tty, so main() reads it, finds nothing, and
        # exits with argparse's usage error (status 2).
        proc = _run_cli([], env=_base_env(tmp_path), cwd=str(tmp_path))
        assert proc.returncode == 2
        assert "no task given" in proc.stderr

    def test_no_task_with_tty_stdin_errors_without_reading(
        self, tmp_path: Path
    ) -> None:
        # A pty stdin makes isatty() True, so main() must NOT block
        # reading stdin and must exit with the usage error directly.
        master, slave = pty.openpty()
        try:
            proc = subprocess.run(
                [sys.executable, "-c", _BOOTSTRAP],
                env=_base_env(tmp_path),
                cwd=str(tmp_path),
                stdin=slave,
                capture_output=True,
                text=True,
                timeout=120,
            )
        finally:
            os.close(master)
            os.close(slave)
        assert proc.returncode == 2
        assert "no task given" in proc.stderr

    def test_no_model_available_exits_one(self, tmp_path: Path) -> None:
        # Scrub every provider credential and any claude/codex CLI from
        # PATH: get_default_model() returns "No model" and main() must
        # exit 1 with a clear message instead of starting an agent.
        env = _base_env(tmp_path)
        for key in list(env):
            if "API_KEY" in key:
                del env[key]
        env["PATH"] = "/usr/bin:/bin"
        env.pop("CLAUDE_CODE_OAUTH_TOKEN", None)
        proc = _run_cli(["do", "something"], env=env, cwd=str(tmp_path))
        assert proc.returncode == 1
        assert "no model available" in proc.stderr


class TestAskUserInTerminal:
    """Real-subprocess coverage of the terminal question callback."""

    _CALLBACK = (
        "import sys; "
        "from kiss.agents.sorcar.sorcar_agent import _ask_user_in_terminal; "
        "sys.stdout.write('reply=' + repr(_ask_user_in_terminal('Pick one?')))"
    )

    def test_returns_typed_line(self, tmp_path: Path) -> None:
        proc = subprocess.run(
            [sys.executable, "-c", self._CALLBACK],
            env=_base_env(tmp_path),
            cwd=str(tmp_path),
            input="blue\n",
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0
        assert "Pick one?" in proc.stdout
        assert "reply='blue'" in proc.stdout

    def test_returns_empty_on_eof(self, tmp_path: Path) -> None:
        proc = subprocess.run(
            [sys.executable, "-c", self._CALLBACK],
            env=_base_env(tmp_path),
            cwd=str(tmp_path),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0
        assert "reply=''" in proc.stdout


@pytest.mark.skipif(not _HAS_OPENAI_KEY, reason="needs OPENAI_API_KEY")
class TestRealAgentRuns:
    """Full CLI-to-agent runs with the cheap stand-in model."""

    def test_argv_task_success_exits_zero(self, tmp_path: Path) -> None:
        work = tmp_path / "work"
        work.mkdir()
        proc = _run_cli(
            [
                "-m",
                "gpt-4o-mini",
                "-b",
                "1.0",
                "--no-web",
                "--work-dir",
                str(work),
                "Do nothing else: immediately call the finish tool with"
                " success=true, is_continue=false, and summary_in_html"
                " '<p>ok</p>'.",
            ],
            env=_base_env(tmp_path),
            cwd=str(tmp_path),
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "success: true" in proc.stdout

    def test_stdin_task_failure_exits_one(self, tmp_path: Path) -> None:
        # The task arrives on piped stdin, the work dir via KISS_WORKDIR
        # (the variable the ~/.local/bin/sorcar wrapper exports), and
        # the agent is told to report failure — exit status must be 1.
        work = tmp_path / "work"
        work.mkdir()
        env = _base_env(tmp_path)
        env["KISS_WORKDIR"] = str(work)
        proc = _run_cli(
            ["-m", "gpt-4o-mini", "-b", "1.0", "--no-web"],
            env=env,
            cwd=str(tmp_path),
            input_text=(
                "Do nothing else: immediately call the finish tool with"
                " success=false, is_continue=false, and summary_in_html"
                " '<p>cannot</p>'."
            ),
        )
        assert proc.returncode == 1, proc.stdout + proc.stderr
        assert "success: false" in proc.stdout
