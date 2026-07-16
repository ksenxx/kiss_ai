# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end regression tests for Wave3-Fixer-3 findings (real repos, no mocks).

B1  ``_MergeFlowMixin._finish_merge`` broadcast ``merge_ended`` FIRST
    while keeping ``tab.is_merging`` raised through the pending-worktree
    presentation and the autocommit dirty-file scan.  The frontend
    re-enables the input on ``merge_ended``, so a ``run`` submitted in
    that seconds-wide window was rejected ("Cannot run a task while
    merge review is in progress") and the prompt text was LOST.  The
    client-visible contract is now: by the time ``merge_ended`` is
    broadcast, the ``is_merging`` guard has already been cleared.

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
:class:`_RunningAgentState` subclasses (the same technique the wave-2
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
from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.core.models.model_info import get_available_models
from kiss.server.json_printer import JsonPrinter
from kiss.server.merge_flow import _MergeFlowMixin
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


def _pop_tabs(*tab_ids: str) -> None:
    """Remove test-created entries from the global tab registry."""
    for tab_id in tab_ids:
        _RunningAgentState.running_agent_states.pop(tab_id, None)


# ---------------------------------------------------------------------------
# B1 — merge_ended must be broadcast only after is_merging is cleared
# ---------------------------------------------------------------------------


class _MergeEndedFlagPrinter(_RecordingPrinter):
    """Recorder that snapshots ``tab.is_merging`` at broadcast time.

    The B1 contract: when the client receives ``merge_ended`` (and
    re-enables its input), the ``is_merging`` guard consulted by
    ``_run_task_inner`` must ALREADY be cleared — otherwise a prompt
    submitted right after the merge view closes is rejected and lost.
    """

    def __init__(self) -> None:
        super().__init__()
        self.tab: _RunningAgentState | None = None
        self.merging_at_merge_ended: bool | None = None
        self.merging_at_prompt: bool | None = None

    def broadcast(self, event: dict[str, Any]) -> None:
        """Record the event plus the tab's live ``is_merging`` flag."""
        if self.tab is not None:
            if event.get("type") == "merge_ended":
                self.merging_at_merge_ended = self.tab.is_merging
            elif event.get("type") == "autocommit_prompt":
                self.merging_at_prompt = self.tab.is_merging
        super().broadcast(event)


class _MergeHost(_MergeFlowMixin):
    """Concrete merge-flow host with the server state the mixin expects.

    Implements the same ``_get_tab`` / ``_any_non_wt_running``
    contracts as ``VSCodeServer`` against the real
    ``_RunningAgentState`` registry (mirrors the wave-2 harness).
    """

    def __init__(self, work_dir: str, printer: JsonPrinter | None = None) -> None:
        self.work_dir = work_dir
        self._state_lock = threading.RLock()
        self.printer = printer or _RecordingPrinter()

    def _get_tab(self, tab_id: str) -> _RunningAgentState:
        """Get or create per-tab state (mirrors ``VSCodeServer._get_tab``)."""
        with self._state_lock:
            tab = _RunningAgentState.running_agent_states.get(tab_id)
            if tab is None:
                tab = _RunningAgentState(tab_id, "test-model")
                _RunningAgentState.running_agent_states[tab_id] = tab
            return tab

    def _any_non_wt_running(self) -> bool:
        """True if any tab runs a non-worktree task (real server semantics)."""
        return any(
            t.is_running_non_wt
            for t in _RunningAgentState.running_agent_states.values()
        )

    def _dispose_if_closed(self, tab_id: str) -> None:
        """Mirror the server: pop only closed, fully-idle tabs."""
        with self._state_lock:
            tab = _RunningAgentState.running_agent_states.get(tab_id)
            if tab is not None and tab.frontend_closed and not (
                tab.is_task_active or tab.is_merging
            ):
                _RunningAgentState.running_agent_states.pop(tab_id, None)


class _RaisingPendingAgent(WorktreeSorcarAgent):
    """Real agent subclass whose pending-worktree probe fails.

    Simulates a mid-cleanup crash inside ``_finish_merge``'s body
    (``_present_pending_worktree`` reads ``agent._wt_pending``) so the
    exception path of the fix is exercised end-to-end.
    """

    @property
    def _wt_pending(self) -> bool:
        """Raise to simulate a crash during the pending-worktree probe."""
        raise RuntimeError("simulated pending-worktree probe failure")

    @_wt_pending.setter
    def _wt_pending(self, value: bool) -> None:
        """Ignore writes (base ``__init__`` initialises the flag)."""


class TestMergeEndedBroadcastAfterGuardCleared:
    def test_merge_ended_broadcast_with_guard_already_cleared(
        self, tmp_path: Path,
    ) -> None:
        """When the client learns the merge ended, the guard is down."""
        repo = tmp_path / "repo"
        _make_repo(repo)
        (repo / "dirty.txt").write_text("uncommitted\n")
        tab_id = "w3f3-b1-tab"
        printer = _MergeEndedFlagPrinter()
        host = _MergeHost(str(repo), printer)
        try:
            tab = host._get_tab(tab_id)
            tab.use_worktree = False
            tab.is_merging = True  # a merge review session is live
            printer.tab = tab

            host._finish_merge(tab_id, work_dir=str(repo))

            types = printer.event_types()
            assert types.count("merge_ended") == 1
            assert "autocommit_prompt" in types
            # The wave-2 F3 invariant is preserved: the dirty scan ran
            # under the claim.
            assert printer.merging_at_prompt is True
            # B1: the client-visible "merge is over, input re-enabled"
            # signal must come AFTER every guarded post-merge step —
            # a prompt submitted on receipt of merge_ended must not be
            # rejected by the is_merging task-start guard.
            assert printer.merging_at_merge_ended is False, (
                "merge_ended was broadcast while is_merging was still "
                "raised — a run submitted on merge-view close is "
                "rejected and the prompt text is lost"
            )
            assert types.index("merge_ended") > types.index(
                "autocommit_prompt",
            )
            assert tab.is_merging is False
        finally:
            _pop_tabs(tab_id)

    def test_clean_tree_still_ends_merge(self, tmp_path: Path) -> None:
        """No dirty files: merge_ended still fires, guard already down."""
        repo = tmp_path / "repo"
        _make_repo(repo)
        tab_id = "w3f3-b1b-tab"
        printer = _MergeEndedFlagPrinter()
        host = _MergeHost(str(repo), printer)
        try:
            tab = host._get_tab(tab_id)
            tab.is_merging = True
            printer.tab = tab

            host._finish_merge(tab_id, work_dir=str(repo))

            types = printer.event_types()
            assert types.count("merge_ended") == 1
            assert "autocommit_prompt" not in types
            assert printer.merging_at_merge_ended is False
            assert tab.is_merging is False
        finally:
            _pop_tabs(tab_id)

    def test_merge_ended_still_fires_when_cleanup_raises(
        self, tmp_path: Path,
    ) -> None:
        """A crash in the post-merge cleanup must not eat merge_ended.

        Otherwise the frontend's merge view (and its disabled input)
        would be stuck open forever.
        """
        repo = tmp_path / "repo"
        _make_repo(repo)
        tab_id = "w3f3-b1c-tab"
        printer = _MergeEndedFlagPrinter()
        host = _MergeHost(str(repo), printer)
        try:
            tab = host._get_tab(tab_id)
            tab.use_worktree = True
            tab.agent = _RaisingPendingAgent("w3f3-b1c")
            tab.is_merging = True
            printer.tab = tab

            with pytest.raises(RuntimeError):
                host._finish_merge(tab_id, work_dir=str(repo))

            types = printer.event_types()
            assert types.count("merge_ended") == 1
            assert printer.merging_at_merge_ended is False
            assert tab.is_merging is False
        finally:
            _pop_tabs(tab_id)

    def test_missing_tab_id_is_noop(self, tmp_path: Path) -> None:
        host = _MergeHost(str(tmp_path))
        host._finish_merge("", work_dir=str(tmp_path))
        assert isinstance(host.printer, _RecordingPrinter)
        assert host.printer.events == []


# ---------------------------------------------------------------------------
# B2 — a pre-loop failure must not attribute the agent's lifetime metrics
# ---------------------------------------------------------------------------


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


class _PreLoopFailTab(_RunningAgentState):
    """Real per-tab state whose armed prompt-recording write fails.

    ``_run_task_inner``'s subtask loop writes ``tab.last_user_prompt``
    BEFORE capturing the per-subtask metric baselines, so an armed
    failure here lands the run in the cleanup ``finally`` exactly as a
    transport / subtask-preparation error would: after the ``try:``
    began, before the first baseline capture.
    """

    def __init__(self, tab_id: str, model: str) -> None:
        self._prompt_backing = ""
        self.fail_next_prompt_write = False
        super().__init__(tab_id, model)

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


class _NoFollowupServer(VSCodeServer):
    """Server whose async followup generation records instead of
    dispatching a background LLM call."""

    def __init__(self, printer: JsonPrinter) -> None:
        super().__init__(printer=printer)
        self.followups: list[tuple[str, str, str | None]] = []

    def _generate_followup_async(
        self, task: str, result: str, task_id: str | None,
    ) -> None:
        """Record the followup request instead of calling an LLM."""
        self.followups.append((task, result, task_id))


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
    server = _NoFollowupServer(printer)
    tab = _PreLoopFailTab(tab_id, models[0])
    _RunningAgentState.running_agent_states[tab_id] = tab
    agent = _ScriptedAgent("Sorcar VS Code")
    tab.agent = agent
    try:
        # Warm the reused agent: one successful run accumulates
        # 100 tokens / $0.25 on its cumulative lifetime counters.
        server._run_task({
            "tabId": tab_id,
            "prompt": warm_prompt,
            "workDir": str(tmp_path),
            "model": models[0],
        })
        assert int(agent.total_tokens_used or 0) == 100
        # The failed run resolves its persistence writes against the
        # most recent row with the same task text — seed that row (a
        # previous run of the identical prompt in history).
        _add_task(probe_prompt)
        # ``_run_task`` disposes ``tab.agent`` in its outer finally;
        # re-attach the SAME warm agent, as tab reuse does.
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
        # Pre-fix: baselines were initialised to 0, so the failed run
        # (which consumed NOTHING) was attributed the warm agent's
        # entire lifetime: 100 tokens / $0.25.
        assert tokens == 0, (
            "pre-loop failure inherited the reused agent's cumulative "
            f"lifetime tokens (got {tokens})"
        )
        assert cost == pytest.approx(0.0)
        assert steps == 0
    finally:
        _pop_tabs(tab_id)


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
    server = _NoFollowupServer(printer)
    tab = _RunningAgentState(tab_id, models[0])
    _RunningAgentState.running_agent_states[tab_id] = tab
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
        _pop_tabs(tab_id)


# ---------------------------------------------------------------------------
# B3 — saveConfig work_dir propagation must be serialised with the save
# ---------------------------------------------------------------------------


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
    from kiss.server.vscode_config import load_config

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
        # Pre-fix: B completes fully while A's propagation is frozen
        # mid-flight (proving the propagation escaped the lock).
        # Post-fix: B is parked at _save_config_lock, so this wait
        # times out — releasing the gate lets A finish and B follow.
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
        # Leave the developer's real config as we found it.
        server._cmd_save_config({"config": {"work_dir": original_work_dir}})


# ---------------------------------------------------------------------------
# D4 — message-object tool_result path must match the primary path
# ---------------------------------------------------------------------------


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
        # A bash_stream print marks the state streamed=True: the
        # client already saw this output live.
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
        # The streamed flag is consumed: the next result carries content.
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
