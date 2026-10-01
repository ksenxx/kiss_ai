# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of ``/<command> help`` through the daemon.

``/xxx help`` is the one sub-task a slash command never relays to its
SEA: the task runner answers with the SEA's ``description()`` as the
task result, without creating a sub-agent or calling a model.  These
tests drive a real daemon (:class:`DaemonRunApiHarness`) with a stub
executor that records every ``KISSAgent.run`` call, so an executor
call would prove the relay ran when it must not.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from kiss.agents.seas.sh import sh_sea
from kiss.agents.sorcar import persistence, sea_commands
from kiss.core.config import kiss_home
from kiss.core.kiss_agent import KISSAgent
from kiss.server import sorcar
from kiss.tests.server.test_append_basic_tools import DaemonRunApiHarness


class SeaHelpCommandTest(DaemonRunApiHarness):
    """``/xxx help`` returns ``description()`` without running the SEA."""

    def _install_counting_stub(self, calls: list[dict[str, Any]]) -> None:
        """Replace the executor LLM loop with a stub that only records its calls."""

        def stub_run(self_agent: Any, **kwargs: Any) -> str:
            if kwargs.get("is_agentic") is False:
                return ""
            calls.append(dict(kwargs.get("arguments") or {}))
            self_agent.step_count = 1
            raw = "success: true\nis_continue: false\nsummary: agent ok\n"
            printer = kwargs.get("printer")
            if printer is not None:  # pragma: no branch
                printer.print(raw, type="result", step_count=1, total_tokens=1, cost="$0.0001")
            return raw

        cast(Any, KISSAgent).run = stub_run

    def _register_user_sea(self, name: str, source: str) -> Path:
        """Write ``<name>/<name>_sea.py`` under a SEAS.md folder and register it."""
        folder = Path(self.tmpdir) / "user-seas"
        sea = folder / name / f"{name}_sea.py"
        sea.parent.mkdir(parents=True)
        sea.write_text(source, encoding="utf-8")
        home = kiss_home()
        home.mkdir(parents=True, exist_ok=True)
        (home / "SEAS.md").write_text(str(folder) + "\n", encoding="utf-8")
        self.addCleanup(sea_commands._reset_for_tests)
        self.addCleanup((home / "SEAS.md").unlink)
        sea_commands.refresh_registry()
        assert sea_commands.get_command(name) == sea.resolve()
        return sea

    def test_bundled_sea_help_is_its_description(self) -> None:
        """``/sh help`` answers with ``sh_sea.description()`` and never runs the SEA."""
        calls: list[dict[str, Any]] = []
        self._install_counting_stub(calls)
        result = sorcar.run(
            "/sh help",
            work_dir=self.repo,
            use_worktree=True,
            endpoint_file=self.endpoint_file,
            timeout=60,
        )
        assert result.success is True, result
        assert result.text == sh_sea.description(), result
        assert "/sh" in result.text
        assert calls == []
        # The exchange is a real history row: prompt, result and end
        # event are persisted under the task id the result carried, so
        # a reload replays it and the sidebar lists it.
        assert result.task_id, result
        row = persistence._load_chat_events_by_task_id(result.task_id)
        assert row is not None
        assert row["task"] == "/sh help"
        assert row["chat_id"] == result.chat_id
        events = cast(list[dict[str, Any]], row["events"])
        types = [e.get("type") for e in events]
        assert types[0] == "prompt" and types[-1] == "task_done", types
        results = [e for e in events if e.get("type") == "result"]
        assert len(results) == 1 and results[0]["text"] == sh_sea.description(), results
        assert results[0]["success"] is True and "tabId" not in results[0]
        entry = next(e for e in persistence._load_history() if str(e["id"]) == result.task_id)
        assert entry["result"] == sh_sea.description()
        extra = json.loads(str(row["extra"] or "{}"))
        assert extra.get("tokens") == 0 and extra.get("steps") == 0
        assert extra.get("endTs", 0) >= extra.get("startTs", 1)

    def test_help_is_case_insensitive_and_needs_the_bare_word(self) -> None:
        """``/echo HELP`` is help; ``/echo help me`` is a relay of the sub-task ``help me``."""
        self._register_user_sea(
            "echo",
            'def description():\n    return "Echoes; use /echo <text>."\n\n\n'
            "def use_worktree():\n    return False\n",
        )
        calls: list[dict[str, Any]] = []
        self._install_counting_stub(calls)
        result = sorcar.run(
            "/echo HELP", work_dir=self.repo, endpoint_file=self.endpoint_file, timeout=60,
        )
        assert result.success is True, result
        assert result.text == "Echoes; use /echo <text>."
        assert calls == []

        first_chat = result.chat_id
        result = sorcar.run(
            "/echo help me",
            work_dir=self.repo,
            endpoint_file=self.endpoint_file,
            chat_id=first_chat,
            timeout=60,
        )
        assert result.success is True, result
        assert result.chat_id == first_chat
        assert len(calls) == 1, calls
        assert "run_agent" in calls[0]["task_description"]
        assert calls[0]["task_description"].endswith("help me")

    def test_help_on_a_sea_without_description_fails_with_a_diagnostic(self) -> None:
        """A SEA lacking ``description()`` makes ``/xxx help`` a failed task naming the file."""
        self._register_user_sea("nodesc", "def use_worktree():\n    return False\n")
        calls: list[dict[str, Any]] = []
        self._install_counting_stub(calls)
        result = sorcar.run(
            "/nodesc help", work_dir=self.repo, endpoint_file=self.endpoint_file, timeout=60,
        )
        assert result.success is False, result
        assert "nodesc_sea.py" in result.text and "description()" in result.text, result
        assert "description must be a zero-argument function" in result.text, result
        assert calls == []
        row = persistence._load_chat_events_by_task_id(result.task_id)
        assert row is not None
        types = [e.get("type") for e in cast(list[dict[str, Any]], row["events"])]
        assert types[-1] == "task_error", types
