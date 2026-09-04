# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end regression tests for Wave3-Fixer-3 findings (real repos, no mocks).

B2  ``_run_task_inner`` initialised the per-subtask metric baselines to
    0 before the big ``try``.  ``tab.agent`` is REUSED across runs on
    the same tab, so its counters are cumulative — a failure before the
    first loop iteration attributed the agent's entire lifetime
    tokens/cost/steps to the failed task's row.

B3  ``_cmd_save_config`` propagated ``work_dir`` to ``self.work_dir`` /
    ``printer.work_dir`` OUTSIDE ``_save_config_lock``, so two racing
    saves with different folders could leave the live server pointed at
    a folder that does not match the persisted config.

D4  ``JsonPrinter._handle_message`` (message-object path) emitted
    ``tool_result`` events WITHOUT ``tool_name`` and bypassed the
    finish-suppression / streamed-dedup treatment of the primary
    ``print(type="tool_result")`` path.

No mocks/patches/fakes: real :class:`JsonPrinter` /
:class:`VSCodeServer` / :class:`WorktreeSorcarAgent` /
:class:`AgentState` subclasses (the same technique the wave-2
regression tests use), real threads, real git repos, and the real
sqlite persistence layer.
"""

from __future__ import annotations

import sqlite3
import subprocess
import threading
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar import persistence
from kiss.agents.sorcar.persistence import _add_task
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.core.models.model_info import get_available_models
from kiss.server import agent_state
from kiss.server.agent_state import AgentState
from kiss.server.json_printer import JsonPrinter
from kiss.server.server import VSCodeServer


def _run_git(repo: Path, *args: str) -> None:
    subprocess.run(
        [
            "git",
            "-c", "user.email=test@test",
            "-c", "user.name=test",
            "-c", "commit.gpgsign=false",
            *args,
        ],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def _make_repo(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    _run_git(repo, "init")
    (repo / "a.txt").write_text("hello\n")
    _run_git(repo, "add", "a.txt")
    _run_git(repo, "commit", "-m", "initial")


class _RecordingPrinter(JsonPrinter):
    """Real JsonPrinter subclass recording broadcast events in a list."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict[str, Any]] = []
        self._events_lock = threading.Lock()

    def broadcast(self, event: dict[str, Any]) -> None:
        """Record *event* in memory instead of persisting it."""
        with self._events_lock:
            self.events.append(event)

    def event_types(self) -> list[str]:
        """Return the recorded event ``type`` strings in order."""
        with self._events_lock:
            return [e.get("type", "") for e in self.events]


def _register_tab_state(task_id: str, tab_id: str) -> AgentState:
    """Register a server-owned agent state for a test tab."""
    state = AgentState(task_id, tab_id=tab_id, server_owned=True)
    agent_state.register(state)
    return state


def _pop_states(*task_ids: str) -> None:
    """Remove test-created entries from the global task registry."""
    with agent_state.STATE_LOCK:
        for task_id in task_ids:
            agent_state.agent_states.pop(task_id, None)



class _ScriptedAgent(WorktreeSorcarAgent):
    """Real agent subclass whose ``run`` allocates a real task row.

    Mirrors what the task_runner relies on from
    :meth:`ChatSorcarAgent.run` under ``_skip_persistence=True`` (same
    recorder as the wave-2 F2 tests): a fresh ``task_history`` row per
    call, ``_last_task_id`` updated, metric counters ACCUMULATED across
    calls — the agent object is reused across runs on the same tab.
    """

    def run(self, *args: Any, **kwargs: Any) -> str:
        """Allocate a task row, bump metrics, and return a YAML result."""
        prompt_template = kwargs.get("prompt_template", "")
        printer = kwargs.get("printer")
        task_id, self._chat_id = _add_task(
            prompt_template, chat_id=self._chat_id or "",
        )
        with self._task_id_lock:
            self._last_task_id = task_id
        if printer is not None:
            printer._thread_local.task_id = str(task_id)
        self.total_tokens_used = int(self.total_tokens_used or 0) + 100
        self.budget_used = float(self.budget_used or 0.0) + 0.25
        if printer is not None:
            printer._thread_local.task_id = ""
        return (
            "success: true\n"
            "is_continue: false\n"
            f"summary: done {prompt_template}\n"
        )


class _PreLoopFailTab(AgentState):
    """Real agent state whose armed prompt-recording write fails.

    ``_run_task_inner``'s subtask loop writes ``state.last_user_prompt``
    BEFORE the per-subtask work begins, so an armed failure here lands
    the run in the cleanup ``finally`` exactly as a transport /
    subtask-preparation error would: after the ``try:`` began, before
    the first ``agent.run`` call.
    """

    def __init__(self, task_id: str, tab_id: str) -> None:
        self._prompt_backing = ""
        self.fail_next_prompt_write = False
        super().__init__(task_id, tab_id=tab_id, server_owned=True)

    @property
    def last_user_prompt(self) -> str:  # pyright: ignore[reportIncompatibleVariableOverride]
        """Return the recorded prompt text."""
        return self._prompt_backing

    @last_user_prompt.setter
    def last_user_prompt(self, value: str) -> None:  # pyright: ignore[reportIncompatibleVariableOverride]
        """Record the prompt, or raise once when armed."""
        if self.fail_next_prompt_write:
            self.fail_next_prompt_write = False
            raise RuntimeError("simulated pre-subtask failure")
        self._prompt_backing = value


def _fetch_metrics(task_text: str) -> tuple[Any, ...] | None:
    """Fetch (result, tokens, cost, steps) for the row with *task_text*."""
    conn = sqlite3.connect(str(persistence._DB_PATH))
    try:
        row = conn.execute(
            "SELECT result, tokens, cost, steps FROM task_history "
            "WHERE task = ? ORDER BY timestamp DESC, rowid DESC LIMIT 1",
            (task_text,),
        ).fetchone()
    finally:
        conn.close()
    return tuple(row) if row is not None else None


def test_b2_preloop_failure_does_not_inherit_lifetime_metrics(
    tmp_path: Path,
) -> None:
    models = get_available_models()
    if not models:
        pytest.skip("no models configured in this environment")
    tab_id = "w3f3-b2-tab"
    warm_prompt = "w3f3-b2 warmup run"
    probe_prompt = "w3f3-b2 preloop-failure probe"
    printer = _RecordingPrinter()
    server = VSCodeServer(printer)
    tab = _PreLoopFailTab("w3f3-b2-task", tab_id)
    agent_state.register(tab)
    agent = _ScriptedAgent("Sorcar VS Code")
    tab.agent = agent
    try:
        server._run_task({
            "tabId": tab_id,
            "prompt": warm_prompt,
            "workDir": str(tmp_path),
            "model": models[0],
        })
        assert int(agent.total_tokens_used or 0) == 100
        _add_task(probe_prompt)
        tab.agent = agent
        tab.fail_next_prompt_write = True

        server._run_task({
            "tabId": tab_id,
            "prompt": probe_prompt,
            "workDir": str(tmp_path),
            "model": models[0],
        })

        row = _fetch_metrics(probe_prompt)
        assert row is not None
        result, tokens, cost, steps = row
        assert "simulated pre-subtask failure" in (result or "")
        assert tokens == 0, (
            "pre-loop failure inherited the reused agent's cumulative "
            f"lifetime tokens (got {tokens})"
        )
        assert cost == pytest.approx(0.0)
        assert steps == 0
    finally:
        _pop_states("w3f3-b2-task")


def test_b2_warm_agent_second_run_still_attributes_own_metrics(
    tmp_path: Path,
) -> None:
    """Regression guard: a SUCCESSFUL follow-up run keeps per-run deltas."""
    models = get_available_models()
    if not models:
        pytest.skip("no models configured in this environment")
    tab_id = "w3f3-b2b-tab"
    first_prompt = "w3f3-b2b first run"
    second_prompt = "w3f3-b2b second run"
    printer = _RecordingPrinter()
    server = VSCodeServer(printer)
    tab = _register_tab_state("w3f3-b2b-task", tab_id)
    agent = _ScriptedAgent("Sorcar VS Code")
    tab.agent = agent
    try:
        for prompt in (first_prompt, second_prompt):
            tab.agent = agent
            server._run_task({
                "tabId": tab_id,
                "prompt": prompt,
                "workDir": str(tmp_path),
                "model": models[0],
            })
        for prompt in (first_prompt, second_prompt):
            row = _fetch_metrics(prompt)
            assert row is not None
            result, tokens, cost, _steps = row
            assert result == f"done {prompt}"
            assert tokens == 100, prompt
            assert cost == pytest.approx(0.25), prompt
    finally:
        _pop_states("w3f3-b2b-task")



class _GatedWorkDirPrinter(_RecordingPrinter):
    """Printer (like WebPrinter) exposing ``work_dir`` — with a gate.

    The setter parks the writing thread when the armed value is first
    written, exposing the exact instant at which ``_cmd_save_config``
    propagates a new folder to the printer.  This is the real
    propagation step of the production code path — the gate merely
    freezes it so a second save can be interleaved deterministically.
    """

    def __init__(self) -> None:
        self._wd_backing = ""
        self.gate_value: str | None = None
        self.gate_reached = threading.Event()
        self.gate_release = threading.Event()
        super().__init__()
        self.work_dir = ""

    @property
    def work_dir(self) -> str:
        """Return the propagated working directory."""
        return self._wd_backing

    @work_dir.setter
    def work_dir(self, value: str) -> None:
        """Store *value*; park the writer once when it is the armed one."""
        if value and value == self.gate_value:
            self.gate_value = None
            self.gate_reached.set()
            self.gate_release.wait(timeout=10)
        self._wd_backing = value


def test_b3_racing_save_config_cannot_desync_live_and_persisted_work_dir(
    tmp_path: Path,
) -> None:
    """Two racing saveConfig with different folders must converge.

    Deterministic interleaving: save A is frozen at its
    ``printer.work_dir`` propagation step (the gated setter).  Pre-fix
    A had already released ``_save_config_lock`` by then, so save B
    runs to completion behind it (its on-disk write AND propagation),
    after which A's unfrozen propagation overwrites the printer with
    the stale folder — the live server no longer matches the persisted
    config.  Post-fix A still holds ``_save_config_lock`` across the
    propagation, so B is parked at the lock and the final state is
    consistent everywhere.
    """
    from kiss.core.vscode_config import load_config

    printer = _GatedWorkDirPrinter()
    server = VSCodeServer(printer=printer)
    original_work_dir = load_config().get("work_dir", "")
    dirs = []
    for name in ("proj-a", "proj-b"):
        d = tmp_path / name
        d.mkdir()
        dirs.append(str(d))

    def save(work_dir: str) -> None:
        server._cmd_save_config({"config": {"work_dir": work_dir}})

    try:
        printer.gate_value = dirs[0]
        t_a = threading.Thread(target=save, args=(dirs[0],), daemon=True)
        t_a.start()
        assert printer.gate_reached.wait(timeout=10), (
            "save A never reached the printer work_dir propagation"
        )
        b_done = threading.Event()

        def save_b() -> None:
            save(dirs[1])
            b_done.set()

        t_b = threading.Thread(target=save_b, daemon=True)
        t_b.start()
        b_done.wait(timeout=1.0)
        printer.gate_release.set()
        t_a.join(timeout=30)
        t_b.join(timeout=30)
        assert not t_a.is_alive() and not t_b.is_alive()

        persisted = load_config().get("work_dir")
        assert persisted in dirs
        assert server.work_dir == persisted, (
            f"live work_dir {server.work_dir!r} diverged from the "
            f"persisted config {persisted!r}"
        )
        assert printer.work_dir == persisted, (
            f"printer work_dir {printer.work_dir!r} diverged from the "
            f"persisted config {persisted!r}"
        )
    finally:
        printer.gate_value = None
        printer.gate_release.set()
        server._cmd_save_config({"config": {"work_dir": original_work_dir}})



class _ToolResultBlock:
    """Real content block shaped like a third-party agent's tool result."""

    def __init__(
        self, content: str, *, is_error: bool = False,
        tool_name: str | None = None,
    ) -> None:
        self.content = content
        self.is_error = is_error
        if tool_name is not None:
            self.tool_name = tool_name


class _TextBlock:
    """Content block WITHOUT is_error/content-pair (must be skipped)."""

    def __init__(self, text: str) -> None:
        self.text = text


class _ContentMessage:
    """Real message object carrying ``.content`` blocks."""

    def __init__(self, content: list[Any]) -> None:
        self.content = content


class _ResultMessage:
    """Real message object carrying a ``.result`` payload."""

    def __init__(self, result: str) -> None:
        self.result = result


class TestMessageObjectToolResults:
    def test_tool_name_stamped_from_block(self) -> None:
        printer = _RecordingPrinter()
        msg = _ContentMessage([
            _ToolResultBlock("bash output", tool_name="Bash"),
        ])

        printer.print(msg, type="message")

        results = [
            e for e in printer.events if e.get("type") == "tool_result"
        ]
        assert len(results) == 1
        assert results[0].get("tool_name") == "Bash", (
            "message-object tool_result lost its tool_name — downstream "
            "consumers key panel labels / highlighting on it"
        )
        assert results[0]["content"] == "bash output"
        assert results[0]["is_error"] is False

    def test_tool_name_falls_back_to_kwargs(self) -> None:
        printer = _RecordingPrinter()
        msg = _ContentMessage([_ToolResultBlock("read output")])

        printer.print(msg, type="message", tool_name="Read")

        results = [
            e for e in printer.events if e.get("type") == "tool_result"
        ]
        assert len(results) == 1
        assert results[0].get("tool_name") == "Read"

    def test_finish_result_suppressed_like_primary_path(self) -> None:
        printer = _RecordingPrinter()
        msg = _ContentMessage([
            _ToolResultBlock("final summary", tool_name="finish"),
        ])

        printer.print(msg, type="message")

        assert all(
            e.get("type") != "tool_result" for e in printer.events
        ), (
            "finish tool_result must be suppressed — the agentic loop "
            "renders it as a dedicated result panel right after"
        )

    def test_streamed_bash_output_deduplicated(self) -> None:
        printer = _RecordingPrinter()
        printer._thread_local.task_id = "w3f3-d4-task"
        printer.print("live streamed chunk", type="bash_stream")
        msg = _ContentMessage([
            _ToolResultBlock("live streamed chunk", tool_name="Bash"),
        ])

        printer.print(msg, type="message")

        results = [
            e for e in printer.events if e.get("type") == "tool_result"
        ]
        assert len(results) == 1
        assert results[0]["content"] == "", (
            "already-streamed output must not be duplicated into the "
            "tool_result event"
        )
        printer.print(
            _ContentMessage([_ToolResultBlock("fresh", tool_name="Bash")]),
            type="message",
        )
        results = [
            e for e in printer.events if e.get("type") == "tool_result"
        ]
        assert results[-1]["content"] == "fresh"

    def test_error_flag_and_mixed_blocks(self) -> None:
        printer = _RecordingPrinter()
        msg = _ContentMessage([
            _TextBlock("not a tool result"),
            _ToolResultBlock("boom", is_error=True, tool_name="Edit"),
        ])

        printer.print(msg, type="message")

        results = [
            e for e in printer.events if e.get("type") == "tool_result"
        ]
        assert len(results) == 1
        assert results[0]["is_error"] is True
        assert results[0]["tool_name"] == "Edit"

    def test_result_message_path_unchanged(self) -> None:
        printer = _RecordingPrinter()

        printer.print(
            _ResultMessage("all done"), type="message",
            total_tokens_used=5, budget_used=0.5,
        )

        results = [e for e in printer.events if e.get("type") == "result"]
        assert len(results) == 1
        assert results[0]["text"] == "all done"

    def test_primary_tool_result_path_unchanged(self) -> None:
        printer = _RecordingPrinter()

        printer.print(
            "file body", type="tool_result", tool_name="Read",
            tool_input={"file_path": "src/x.py", "start_line": 3},
        )

        results = [
            e for e in printer.events if e.get("type") == "tool_result"
        ]
        assert len(results) == 1
        assert results[0]["tool_name"] == "Read"
        assert results[0]["path"] == "src/x.py"
        assert results[0]["start_line"] == 3
        assert results[0]["content"] == "file body"
