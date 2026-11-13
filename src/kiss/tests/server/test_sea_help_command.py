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
import threading
import time
from pathlib import Path
from typing import Any, cast

from kiss.agents.seas.sh import sh_sea
from kiss.agents.sorcar import local_endpoint, persistence, sea_commands
from kiss.core.config import kiss_home
from kiss.core.kiss_agent import KISSAgent
from kiss.server import sorcar
from kiss.tests.agents.sorcar.conftest import reset_sea_registry
from kiss.tests.server.test_append_basic_tools import DaemonRunApiHarness


class SeaHelpCommandTest(DaemonRunApiHarness):
    """``/xxx help`` returns ``description()`` without running the SEA."""

    def _install_counting_stub(self, calls: list[dict[str, Any]]) -> None:
        """Replace the executor LLM loop with a stub that only records its calls."""

        def stub_run(self_agent: Any, **kwargs: Any) -> str:
            if kwargs.get("is_agentic") is False:
                return ""
            calls.append({
                **dict(kwargs.get("arguments") or {}),
                "system_prompt": str(kwargs.get("system_prompt") or ""),
            })
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
        self.addCleanup(reset_sea_registry)
        self.addCleanup((home / "SEAS.md").unlink)
        sea_commands.refresh_registry()
        assert sea_commands.get_command(name) == sea.resolve()
        return sea

    def test_bundled_sea_help_is_its_description(self) -> None:
        """``/sh help`` answers with ``sh_sea.ShSea().description()`` and never runs the SEA."""
        calls: list[dict[str, Any]] = []
        self._install_counting_stub(calls)
        result = sorcar.run(
            "/sh help",
            work_dir=self.repo,
            endpoint_file=self.endpoint_file,
            timeout=60,
        )
        assert result.success is True, result
        assert result.text == sh_sea.ShSea().description(), result
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
        assert len(results) == 1 and results[0]["text"] == sh_sea.ShSea().description(), results
        assert results[0]["success"] is True and "tabId" not in results[0]
        entry = next(e for e in persistence._load_history() if str(e["id"]) == result.task_id)
        assert entry["result"] == sh_sea.ShSea().description()
        extra = json.loads(str(row["extra"] or "{}"))
        assert extra.get("tokens") == 0 and extra.get("steps") == 0
        assert extra.get("endTs", 0) >= extra.get("startTs", 1)
        # A finished row like any other: its tags are derived and its
        # chat has the summary the CHATS panel lists it by.
        assert extra.get("tags"), extra
        summary = persistence._chat_summaries([result.chat_id])[result.chat_id]
        assert summary["summary"] and summary["last_finished"] == extra["endTs"], summary

    def test_help_is_case_insensitive_and_needs_the_bare_word(self) -> None:
        """``/echo HELP`` is help; ``/echo help me`` runs the echo SEA directly on ``help me``."""
        self._register_user_sea(
            "echo",
            """
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def description(self):
        return "Echoes; use /echo <text>."

    def settings(self, settings):
        return settings | {'use_web_tools': False}

    def system_prompt(self, system_prompt):
        return system_prompt + '\\n\\nECHO-PROTOCOL'
""",
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
        # One LLM run in the tab's own session: the SEA's settings apply
        # to it and its task is the trailing text, not a run_agent relay
        # (the chat history keeps the raw ``/echo HELP`` exchange).
        assert len(calls) == 1, calls
        task = calls[0]["task_description"]
        assert task.endswith("# Task (work on it now)\n\nhelp me"), task
        assert "run_agent" not in task
        assert "/echo HELP" in task and "Echoes; use /echo <text>." in task
        assert "ECHO-PROTOCOL" in calls[0]["system_prompt"]

    def test_help_on_a_sea_without_description_fails_with_a_diagnostic(self) -> None:
        """A SEA lacking ``description()`` makes ``/xxx help`` a failed task naming the file."""
        self._register_user_sea(
            "nodesc", """
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def settings(self, settings):
        return settings | {'use_web_tools': False}
""",
        )
        calls: list[dict[str, Any]] = []
        self._install_counting_stub(calls)
        result = sorcar.run(
            "/nodesc help", work_dir=self.repo, endpoint_file=self.endpoint_file, timeout=60,
        )
        assert result.success is False, result
        assert "nodesc_sea.py" in result.text and "description()" in result.text, result
        assert "description() must return a non-empty string" in result.text, result
        assert calls == []
        row = persistence._load_chat_events_by_task_id(result.task_id)
        assert row is not None
        types = [e.get("type") for e in cast(list[dict[str, Any]], row["events"])]
        assert types[-1] == "task_error", types
        # The history panel's ``failed`` filter lists the row, and its
        # chat has its CHATS-panel summary.
        failed = [str(e["id"]) for e in persistence._load_history(tag="failed")]
        assert failed == [result.task_id], failed
        assert result.chat_id in persistence._chat_summaries([result.chat_id])

    def test_a_viewer_attached_during_help_receives_the_answer(self) -> None:
        """A tab that resumes the chat while ``description()`` runs gets the help result too."""
        sea = self._register_user_sea(
            "slowhelp", """
import pathlib
import time

from kiss.agents.seas.base.base_sea import BaseSea

_DIR = pathlib.Path(__file__).resolve().parent


class Sea(BaseSea):
    def description(self):
        (_DIR / "entered").write_text("1", encoding="utf-8")
        deadline = time.time() + 60
        while time.time() < deadline and not (_DIR / "release").exists():
            time.sleep(0.02)
        return "slowhelp: the help text"
""",
        )
        calls: list[dict[str, Any]] = []
        self._install_counting_stub(calls)
        seed = sorcar.run(
            "/sh help", work_dir=self.repo, endpoint_file=self.endpoint_file, timeout=60,
        )
        assert seed.success is True and seed.chat_id, seed
        out: dict[str, Any] = {}

        def run_help() -> None:
            out["result"] = sorcar.run(
                "/slowhelp help",
                work_dir=self.repo,
                chat_id=seed.chat_id,
                endpoint_file=self.endpoint_file,
                timeout=60,
            )

        launcher = threading.Thread(target=run_help, daemon=True)
        launcher.start()
        viewer_events: list[dict[str, Any]] = []
        viewer = local_endpoint.connect(Path(self.endpoint_file), open_timeout=60)
        try:
            deadline = time.monotonic() + 30
            while not (sea.parent / "entered").exists():
                assert time.monotonic() < deadline, "description() never ran"
                time.sleep(0.02)
            # A second surface opens the chat while the help runs: it
            # attaches to the run and is told it is running.
            viewer.send(json.dumps({
                "type": "resumeSession", "chatId": seed.chat_id, "tabId": "viewer-tab",
            }))
            while True:
                event = json.loads(viewer.recv(timeout=60))
                if event.get("tabId") != "viewer-tab":
                    continue
                viewer_events.append(event)
                if event.get("type") == "status" and event.get("running"):
                    break
            (sea.parent / "release").write_text("1", encoding="utf-8")
            # The run's end reaches the viewer; the answer must have
            # reached it first.
            while True:
                event = json.loads(viewer.recv(timeout=60))
                if event.get("tabId") != "viewer-tab":
                    continue
                viewer_events.append(event)
                if event.get("type") == "status" and not event.get("running"):
                    break
        finally:
            (sea.parent / "release").write_text("1", encoding="utf-8")
            viewer.close()
            launcher.join(timeout=60)
        assert out["result"].success is True and out["result"].text == "slowhelp: the help text"
        results = [e for e in viewer_events if e.get("type") == "result"]
        assert [e["text"] for e in results] == ["slowhelp: the help text"], [
            (e.get("type"), e.get("text")) for e in viewer_events
        ]
        assert calls == []
