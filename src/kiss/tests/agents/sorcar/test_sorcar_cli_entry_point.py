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

The task is given via exactly one of two required, mutually exclusive
options: ``-t TASK`` (an inline task string) or ``-f FILE`` (the file's
content becomes the task).  Argparse enforces the exactly-one contract;
the CLI additionally rejects a blank ``-t`` value, an unreadable ``-f``
path, and an empty ``-f`` file.

Every test here launches a REAL subprocess — no mocks, patches, or
fakes.  ``test_console_script_is_wired`` runs the installed ``sorcar``
console script itself, so deleting or misspelling the
``[project.scripts]`` entry fails the suite (after the environment is
re-synced, which is how the script reaches PATH in the first place).
The tests that reach ``agent.run`` use the cheap ``gpt-4o-mini``
stand-in model, like the other e2e suites in this folder.

Isolation: subprocesses run with ``HOME`` and ``KISS_HOME`` pointed at
the test's temp dir (``kiss.core.models.model_info`` seeds
``~/.kiss/MY_MODELS.json`` under ``HOME``, not ``KISS_HOME``).  Saved
trajectories still land under the checkout's ``.kiss.artifacts`` —
that root is anchored at the package location by
``kiss.core.config._PROJECT_DIR`` for every agent in this suite, not
something the CLI can redirect.

Branch-coverage notes for ``main()``:

* ``except Exception`` around ``yaml.safe_load(result)`` is unreachable
  without test doubles: ``RelentlessAgent.run`` always returns
  ``yaml.dump`` of a dict payload (its ``finish`` tool and every error
  path build the payload as a dict), so the parse can neither raise nor
  yield a non-dict.  Documented here instead of mocked, per the no-test-
  doubles policy.
* The ``verbose = sys.stdout.isatty()`` True branch (interactive
  console printer, no raw-YAML reprint) would need a full agent run on
  a pty; the piped branch covers the exit-code and output contract that
  scripts rely on.
"""

from __future__ import annotations

import os
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


def _base_env(tmp: Path) -> dict[str, str]:
    """Return a copy of the environment isolated to a temp home.

    Args:
        tmp: Temp directory used as both ``$HOME`` and ``$KISS_HOME`` so
            the subprocess never reads or seeds the developer's
            ``~/.kiss`` (``MY_MODELS.json`` is resolved via ``HOME``).

    Returns:
        The environment mapping for the subprocess.
    """
    env = dict(os.environ)
    env["HOME"] = str(tmp)
    env["KISS_HOME"] = str(tmp / ".kiss")
    env.pop("KISS_WORKDIR", None)
    return env


class TestConsoleScriptWiring:
    """The [project.scripts] entry itself — the wiring whose loss caused
    the original recursion."""

    def test_console_script_is_wired(self, tmp_path: Path) -> None:
        # The environment's own `sorcar` script (generated from the
        # pyproject entry point at sync time) must exist next to the
        # interpreter and answer --help without recursing into uv.
        script = Path(sys.executable).parent / "sorcar"
        assert script.exists(), (
            "console script 'sorcar' missing from the venv — the "
            "[project.scripts] entry in pyproject.toml is gone"
        )
        proc = subprocess.run(
            [str(script), "--help"],
            env=_base_env(tmp_path),
            cwd=str(tmp_path),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0
        assert "Run the KISS SorcarAgent on a task." in proc.stdout


class TestArgumentHandling:
    """Cheap tests that never reach the LLM."""

    def test_help_exits_zero(self, tmp_path: Path) -> None:
        proc = _run_cli(["--help"], env=_base_env(tmp_path), cwd=str(tmp_path))
        assert proc.returncode == 0
        assert "Run the KISS SorcarAgent on a task." in proc.stdout
        assert "--max-budget" in proc.stdout
        assert "--task" in proc.stdout
        assert "--file" in proc.stdout
        assert "--no-web" not in proc.stdout

    def test_neither_task_nor_file_errors(self, tmp_path: Path) -> None:
        # -t and -f form a required mutually exclusive group: with
        # neither given, argparse exits with a usage error (status 2)
        # without touching stdin.
        proc = _run_cli([], env=_base_env(tmp_path), cwd=str(tmp_path))
        assert proc.returncode == 2
        assert "one of the arguments -t/--task -f/--file is required" in proc.stderr

    def test_task_and_file_together_error(self, tmp_path: Path) -> None:
        task_file = tmp_path / "task.txt"
        task_file.write_text("do something")
        proc = _run_cli(
            ["-t", "do something", "-f", str(task_file)],
            env=_base_env(tmp_path),
            cwd=str(tmp_path),
        )
        assert proc.returncode == 2
        assert "not allowed with argument" in proc.stderr

    def test_blank_task_errors(self, tmp_path: Path) -> None:
        # An explicit but whitespace-only -t value is rejected before
        # any model or agent setup.
        proc = _run_cli(["-t", "   "], env=_base_env(tmp_path), cwd=str(tmp_path))
        assert proc.returncode == 2
        assert "task must not be empty" in proc.stderr

    def test_unreadable_task_file_errors(self, tmp_path: Path) -> None:
        missing = tmp_path / "no_such_task.txt"
        proc = _run_cli(
            ["-f", str(missing)], env=_base_env(tmp_path), cwd=str(tmp_path)
        )
        assert proc.returncode == 2
        assert "cannot read task file" in proc.stderr

    def test_non_utf8_task_file_errors(self, tmp_path: Path) -> None:
        # A file that cannot be decoded as UTF-8 must produce the same
        # status-2 usage error as an unreadable one, not an uncaught
        # UnicodeDecodeError traceback with exit status 1.
        binary = tmp_path / "binary_task.bin"
        binary.write_bytes(b"\x89PNG\r\n\x1a\n\xff\xfe")
        proc = _run_cli(
            ["-f", str(binary)], env=_base_env(tmp_path), cwd=str(tmp_path)
        )
        assert proc.returncode == 2
        assert "cannot read task file" in proc.stderr

    def test_empty_task_file_errors(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty_task.txt"
        empty.write_text("  \n\t\n")
        proc = _run_cli(
            ["-f", str(empty)], env=_base_env(tmp_path), cwd=str(tmp_path)
        )
        assert proc.returncode == 2
        assert "is empty" in proc.stderr

    @pytest.mark.parametrize("bad", ["nan", "inf", "-inf", "0", "-5", "abc"])
    def test_invalid_budget_rejected(self, tmp_path: Path, bad: str) -> None:
        # float() accepts nan/inf, and a NaN cap makes every budget
        # comparison false — enforcement would be silently disabled, so
        # the CLI must reject these at parse time (argparse status 2).
        proc = _run_cli(
            ["-b", bad, "-t", "some task"],
            env=_base_env(tmp_path),
            cwd=str(tmp_path),
        )
        assert proc.returncode == 2
        assert "budget" in proc.stderr

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
        proc = _run_cli(["-t", "do something"], env=env, cwd=str(tmp_path))
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

    def test_task_option_success_exits_zero(self, tmp_path: Path) -> None:
        # Observable --work-dir verification: the task (which contains
        # literal {braces} that must survive prompt handling) makes the
        # agent write a file, asserted to appear in the --work-dir
        # directory — proving the flag actually reached agent.run.
        work = tmp_path / "work"
        work.mkdir()
        proc = _run_cli(
            [
                "-m",
                "gpt-4o-mini",
                "-b",
                "1.0",
                "--work-dir",
                str(work),
                "-t",
                "Use the Write tool to create a file named proof.txt in the"
                " current working directory with exact content {ok}. Then"
                " immediately call the finish tool with success=true,"
                " is_continue=false, and summary_in_html '<p>done</p>'."
                " Do nothing else.",
            ],
            env=_base_env(tmp_path),
            cwd=str(tmp_path),
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "success: true" in proc.stdout
        proof = work / "proof.txt"
        assert proof.exists(), "task file missing — --work-dir was not honored"
        assert "{ok}" in proof.read_text(), "braces were mangled in the prompt"

    def test_file_task_failure_exits_one(self, tmp_path: Path) -> None:
        # The task arrives via -f (the file's content IS the task, and
        # it too contains literal {braces} that must survive), the work
        # dir via KISS_WORKDIR (the variable the ~/.local/bin/sorcar
        # wrapper exports), and the agent is told to report failure —
        # exit status must be 1 and the marker file must land in
        # $KISS_WORKDIR with the exact braced content.
        work = tmp_path / "work"
        work.mkdir()
        env = _base_env(tmp_path)
        env["KISS_WORKDIR"] = str(work)
        task_file = tmp_path / "task.txt"
        task_file.write_text(
            "Use the Write tool to create a file named marker.txt in the"
            " current working directory with exact content {no}. Then"
            " immediately call the finish tool with success=false,"
            " is_continue=false, and summary_in_html '<p>cannot</p>'."
            " Do nothing else.\n"
        )
        proc = _run_cli(
            ["-m", "gpt-4o-mini", "-b", "1.0", "-f", str(task_file)],
            env=env,
            cwd=str(tmp_path),
        )
        assert proc.returncode == 1, proc.stdout + proc.stderr
        assert "success: false" in proc.stdout
        marker = work / "marker.txt"
        assert marker.exists(), "KISS_WORKDIR was not honored"
        assert "{no}" in marker.read_text(), (
            "braces were mangled in the file-sourced prompt"
        )

    def test_ask_user_question_reads_terminal_stdin(self, tmp_path: Path) -> None:
        # End-to-end wiring of ask_user_question_callback: the agent asks
        # a question, the answer arrives on the CLI's stdin (the task
        # itself came from -t, so stdin is free for answers), and the
        # answer must surface in the final result.
        work = tmp_path / "work"
        work.mkdir()
        proc = _run_cli(
            [
                "-m",
                "gpt-4o-mini",
                "-b",
                "1.0",
                "--work-dir",
                str(work),
                "-t",
                "First call the ask_user_question tool with the question"
                " 'What color?'. Then immediately call the finish tool with"
                " success=true, is_continue=false, and summary_in_html set to"
                " exactly the answer you received. Do nothing else.",
            ],
            env=_base_env(tmp_path),
            cwd=str(tmp_path),
            input_text="turquoise\n",
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "turquoise" in proc.stdout, (
            "the terminal answer never reached the agent — "
            "ask_user_question_callback is not wired"
        )
