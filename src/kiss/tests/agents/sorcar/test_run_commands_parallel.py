# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for ``UsefulTools.run_commands_parallel``.

The tool runs shell commands concurrently in threads with no LLM
sub-agent, so a test split (``uv run pytest`` per split) costs no
tokens.  Every test runs real shells; no mocks.

Branch not covered: the ``except Exception`` in ``_run_one_command``
(``_bash_streaming`` raising) — ``_spawn`` falls back rather than
raising on a vanished work_dir, so it needs a fault injection.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any

from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.agents.sorcar.useful_tools import UsefulTools
from kiss.core.tool_interrupt import (
    ToolCallInterrupted,
    begin_tool_call,
    end_tool_call,
    interrupt_tool_call,
    unregister_tool_call,
)


class TestReport:
    """The report lists every command, in order, with its exit status."""

    def test_mixed_outcomes_in_input_order(self, tmp_path: Path) -> None:
        tools = UsefulTools(work_dir=str(tmp_path))
        out = tools.run_commands_parallel(
            '["echo alpha", "echo beta >&2; exit 3", "sleep 5", "echo gamma"]',
            timeout_seconds=1,
        )
        lines = out.splitlines()
        assert lines[0] == "4 commands: 2 succeeded, 1 failed, 1 timed out."
        assert out.index("[1/4] exit 0") < out.index("[2/4] exit 3")
        assert out.index("[2/4] exit 3") < out.index("[3/4] TIMED OUT after")
        assert out.index("[3/4] TIMED OUT") < out.index("[4/4] exit 0")
        assert "$ echo alpha\nalpha" in out
        assert "$ echo beta >&2; exit 3\nbeta" in out
        assert "$ sleep 5\n(no output)" in out

    def test_commands_run_in_the_work_dir_with_kiss_workdir(self, tmp_path: Path) -> None:
        tools = UsefulTools(work_dir=str(tmp_path))
        out = tools.run_commands_parallel('["pwd", "echo $KISS_WORKDIR"]')
        real = os.path.realpath(tmp_path)
        assert out.count(real) == 2 or out.count(str(tmp_path)) == 2, out

    def test_output_truncated_per_command(self, tmp_path: Path) -> None:
        tools = UsefulTools(work_dir=str(tmp_path))
        out = tools.run_commands_parallel(
            '["seq 1 5000", "echo short"]', max_output_chars=300,
        )
        assert "truncated" in out
        assert "1\n2\n3" in out and "4999\n5000" in out
        assert "$ echo short\nshort" in out

    def test_bash_formats_are_unchanged(self, tmp_path: Path) -> None:
        """The shared engine refactor must not change ``Bash``'s outputs."""
        tools = UsefulTools(work_dir=str(tmp_path))
        assert tools.Bash("echo hi", "ok") == "hi\n"
        assert tools.Bash("echo hi; exit 2", "fail") == "Error (exit code 2):\nhi\n"
        assert tools.Bash("exit 4", "silent fail") == "Error (exit code 4):"
        assert (
            tools.Bash("sleep 5", "slow", timeout_seconds=0.5)
            == "Error: Command execution timeout"
        )


class TestArguments:
    """Invalid arguments are refused with an actionable ``Error:`` string."""

    def test_shell_substitution_is_rejected(self, tmp_path: Path) -> None:
        tools = UsefulTools(work_dir=str(tmp_path))
        out = tools.run_commands_parallel("$(cat ./tmp/cmds.json)")
        assert out.startswith("Error: commands must be a JSON array of strings")
        assert "Shell substitutions are not expanded" in out

    def test_non_array_and_empty_rejected(self, tmp_path: Path) -> None:
        tools = UsefulTools(work_dir=str(tmp_path))
        assert "got a JSON dict" in tools.run_commands_parallel('{"a": 1}')
        assert "empty array" in tools.run_commands_parallel("[]")
        assert "non-empty strings" in tools.run_commands_parallel('["ls", ""]')

    def test_negative_max_workers_rejected(self, tmp_path: Path) -> None:
        tools = UsefulTools(work_dir=str(tmp_path))
        out = tools.run_commands_parallel('["echo x"]', max_workers=-2)
        assert out == "Error: max_workers must be 0 or a positive integer, got -2."


class TestConcurrency:
    """``max_workers`` bounds concurrency; ``0`` runs everything at once."""

    def test_all_at_once_by_default(self, tmp_path: Path) -> None:
        tools = UsefulTools(work_dir=str(tmp_path))
        started = time.monotonic()
        out = tools.run_commands_parallel('["sleep 1", "sleep 1", "sleep 1", "sleep 1"]')
        elapsed = time.monotonic() - started
        assert out.startswith("4 commands: 4 succeeded, 0 failed, 0 timed out.")
        assert elapsed < 3.0, elapsed

    def test_max_workers_serializes(self, tmp_path: Path) -> None:
        tools = UsefulTools(work_dir=str(tmp_path))
        started = time.monotonic()
        out = tools.run_commands_parallel('["sleep 0.6", "sleep 0.6"]', max_workers=1)
        elapsed = time.monotonic() - started
        assert out.startswith("2 commands: 2 succeeded")
        assert elapsed >= 1.2, elapsed


class TestWorktreeGuard:
    """A command naming the parent repo of a live worktree is refused, per command."""

    def test_parent_repo_path_refused_others_run(self, tmp_path: Path) -> None:
        repo = Path(os.path.realpath(tmp_path)) / "repo"
        wt = repo / ".kiss-worktrees" / "kiss_wt-live"
        wt.mkdir(parents=True)
        (repo / "f.txt").write_text("main content\n")
        tools = UsefulTools(work_dir=str(wt))
        out = tools.run_commands_parallel(
            f'["echo hi > {repo}/f.txt", "echo inside-worktree"]'
        )
        assert out.startswith("2 commands: 1 succeeded, 1 failed, 0 timed out.")
        assert "[1/2] exit -1" in out
        assert "parent-repo path" in out
        assert "$ echo inside-worktree\ninside-worktree" in out
        assert (repo / "f.txt").read_text() == "main content\n"


class TestStop:
    """The task's Stop and the tool panel's Stop both kill every running command."""

    def test_task_stop_event_kills_all_commands(self, tmp_path: Path) -> None:
        stop = threading.Event()
        tools = UsefulTools(stop_event=stop, work_dir=str(tmp_path))
        threading.Timer(0.7, stop.set).start()
        started = time.monotonic()
        out = tools.run_commands_parallel('["sleep 30", "sleep 30"]')
        elapsed = time.monotonic() - started
        assert elapsed < 10, elapsed
        assert out.startswith("2 commands: 0 succeeded, 2 failed, 0 timed out.")
        assert out.count("killed by signal 9") == 2, out

    def test_task_stop_does_not_start_queued_commands(self, tmp_path: Path) -> None:
        stop = threading.Event()
        tools = UsefulTools(stop_event=stop, work_dir=str(tmp_path))
        marker = tmp_path / "queued-ran"
        threading.Timer(0.7, stop.set).start()
        out = tools.run_commands_parallel(
            f'["sleep 30", "touch {marker}", "touch {marker}"]', max_workers=1,
        )
        assert out.startswith("3 commands: 0 succeeded, 3 failed, 0 timed out.")
        assert out.count("killed by signal 9") == 1, out
        assert out.count("Not started: the task was stopped.") == 2, out
        assert not marker.exists()

    def test_tool_panel_interrupt_raises_and_kills(self, tmp_path: Path) -> None:
        tools = UsefulTools(work_dir=str(tmp_path))
        marker = tmp_path / "still-running"
        outcome: dict[str, Any] = {}

        def run() -> None:
            token = begin_tool_call("run_commands_parallel")
            try:
                try:
                    outcome["result"] = tools.run_commands_parallel(
                        f'["sleep 30; touch {marker}", "sleep 30; touch {marker}"]'
                    )
                    end_tool_call(token)
                except ToolCallInterrupted:
                    outcome["interrupted"] = True
            finally:
                unregister_tool_call(token)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        time.sleep(0.7)
        assert thread.ident is not None
        started = time.monotonic()
        assert interrupt_tool_call(thread.ident, "run_commands_parallel") is True
        thread.join(timeout=30)
        assert not thread.is_alive()
        assert time.monotonic() - started < 10
        assert outcome == {"interrupted": True}, outcome
        # The shells were killed, not left to finish in the background.
        time.sleep(0.5)
        assert not marker.exists()


class TestRegistration:
    """The tool is offered to the model next to ``Bash``."""

    def test_tool_in_sorcar_tool_list(self, tmp_path: Path) -> None:
        agent = SorcarAgent("tool-list")
        agent.work_dir = str(tmp_path)
        agent._use_web_tools = False
        names = [t.__name__ for t in agent._get_tools()]
        assert "Bash" in names
        assert "run_commands_parallel" in names
        assert names.index("Bash") < names.index("run_commands_parallel")
