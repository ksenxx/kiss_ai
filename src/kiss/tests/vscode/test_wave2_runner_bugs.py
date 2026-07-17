# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end regression tests for wave-2 fixer-6 findings (round 6).

Covers:

- F2: a multi-``<task>`` prompt persisted the result / ``extra``
  payload only under the LAST subtask's ``task_history`` row; the
  first N-1 rows stayed empty (no result, no start/end timestamps,
  no tokens) and metrics were attributed cumulatively to the last row.
- F5: ``_flush_bash`` / the inline ``bash_stream`` flush broadcast
  while holding the printer-global ``_bash_lock``, so one task's slow
  transport send blocked every other task's bash streaming.
- F6: the task-cleanup ``printer.reset()`` ran after the worker's
  thread-local ``task_id`` was already cleared to ``""``, so it
  mutated (generation-bumped and drained) the SHARED ``""`` fallback
  bash state belonging to task-less broadcasters.
- F9: ``_ask_user_question`` resolved the answer queue once (drain +
  pending-registry entry) while ``_await_user_response`` re-resolved
  it, so the agent could block on a DIFFERENT queue than the one it
  registered.
- F13: ``_cmd_save_config`` wrote ``self.work_dir`` without
  ``_state_lock`` and, unlike ``_cmd_set_work_dir``, neither cleared
  the ``@``-mention file cache nor synced ``printer.work_dir``.
- F12: ``_ensure_downloaded_model`` raced concurrent callers on a
  shared temp file and could publish a half-extracted model
  directory; now serialised by an ``fcntl`` lock with an atomic
  extract-to-temp + rename publish.

No mocks/patches/fakes: real :class:`JsonPrinter` /
:class:`VSCodeServer` / :class:`WorktreeSorcarAgent` subclasses (the
same technique the existing vscode regression tests use), real
threads, and the real sqlite persistence layer.
"""

from __future__ import annotations

import os
import queue
import random
import sqlite3
import threading
import time
import zipfile
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar import persistence
from kiss.agents.sorcar.persistence import _add_task
from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.core.models.model_info import get_available_models
from kiss.server.json_printer import JsonPrinter, _BashState
from kiss.server.server import VSCodeServer


class _CapturePrinter(JsonPrinter):
    """Real printer subclass that records every broadcast event."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict[str, Any]] = []

    def broadcast(self, event: dict[str, Any]) -> None:
        """Record *event* instead of writing it to a transport."""
        self.events.append(event)


def _pop_tabs(*tab_ids: str) -> None:
    """Remove test-created entries from the global tab registry."""
    for tab_id in tab_ids:
        _RunningAgentState.running_agent_states.pop(tab_id, None)


# ---------------------------------------------------------------------------
# F2 — multi-<task> prompts: every subtask row must be fully persisted
# ---------------------------------------------------------------------------


class _ScriptedAgent(WorktreeSorcarAgent):
    """Real agent subclass whose ``run`` allocates a real task row.

    Mirrors exactly what the task_runner relies on from
    :meth:`ChatSorcarAgent.run` under ``_skip_persistence=True``: a
    fresh ``task_history`` row per call (via the real ``_add_task``),
    ``_last_task_id`` updated under ``_task_id_lock``, the printer's
    thread-local ``task_id`` set for the duration of the run and
    cleared to ``""`` on exit, metric counters accumulated across
    calls — and NO result/extra save of its own.
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


def _run_scripted_task(
    tmp_path: Path, tab_id: str, prompt: str,
) -> tuple[_NoFollowupServer, _CapturePrinter, _ScriptedAgent]:
    """Run *prompt* through the real ``_run_task`` with a scripted agent."""
    models = get_available_models()
    if not models:
        pytest.skip("no models configured in this environment")
    printer = _CapturePrinter()
    server = _NoFollowupServer(printer)
    tab = _RunningAgentState(tab_id, models[0])
    _RunningAgentState.running_agent_states[tab_id] = tab
    agent = _ScriptedAgent("Sorcar VS Code")
    tab.agent = agent
    server._run_task({
        "tabId": tab_id,
        "prompt": prompt,
        "workDir": str(tmp_path),
        "model": models[0],
    })
    return server, printer, agent


def _fetch_rows(tasks: list[str]) -> dict[str, tuple[Any, ...]]:
    """Fetch (result, start_ts, end_ts, tokens, cost) per task text."""
    conn = sqlite3.connect(str(persistence._DB_PATH))
    try:
        rows = conn.execute(
            "SELECT task, result, start_ts, end_ts, tokens, cost "
            "FROM task_history WHERE task IN ({})".format(
                ",".join("?" * len(tasks)),
            ),
            tasks,
        ).fetchall()
    finally:
        conn.close()
    return {r[0]: r[1:] for r in rows}


def test_f2_every_subtask_row_gets_result_and_extra(tmp_path: Path) -> None:
    tab_id = "w2f2-tab"
    try:
        _run_scripted_task(
            tmp_path,
            tab_id,
            "<task>w2f2 alpha one</task>\n<task>w2f2 beta two</task>",
        )
        rows = _fetch_rows(["w2f2 alpha one", "w2f2 beta two"])
        assert set(rows) == {"w2f2 alpha one", "w2f2 beta two"}
        for task_text, (result, start_ts, end_ts, tokens, cost) in rows.items():
            # Pre-fix: the FIRST subtask's row had result == "" and
            # start_ts == end_ts == 0 (no extra payload at all).
            assert result == f"done {task_text}", task_text
            assert start_ts > 0, task_text
            assert end_ts >= start_ts, task_text
            # Pre-fix: tokens were 0 on the first row and the agent's
            # CUMULATIVE 200 on the last row.  Each subtask consumed
            # exactly 100 tokens / $0.25.
            assert tokens == 100, task_text
            assert cost == pytest.approx(0.25), task_text
    finally:
        _pop_tabs(tab_id)


def test_f2_single_task_persistence_unchanged(tmp_path: Path) -> None:
    tab_id = "w2f2s-tab"
    try:
        server, _, _ = _run_scripted_task(
            tmp_path, tab_id, "w2f2 single gamma",
        )
        rows = _fetch_rows(["w2f2 single gamma"])
        (result, start_ts, end_ts, tokens, cost) = rows["w2f2 single gamma"]
        assert result == "done w2f2 single gamma"
        assert start_ts > 0
        assert end_ts >= start_ts
        assert tokens == 100
        assert cost == pytest.approx(0.25)
        # The followup hook still fires exactly once for the task.
        assert len(server.followups) == 1
    finally:
        _pop_tabs(tab_id)


# ---------------------------------------------------------------------------
# F6 — cleanup reset() must not clobber the shared "" fallback bash state
# ---------------------------------------------------------------------------


def test_f6_task_cleanup_leaves_fallback_bash_state_alone(
    tmp_path: Path,
) -> None:
    tab_id = "w2f6-tab"
    try:
        models = get_available_models()
        if not models:
            pytest.skip("no models configured in this environment")
        printer = _CapturePrinter()
        server = _NoFollowupServer(printer)
        # Seed the shared fallback ("" key) bash state as a task-less
        # broadcaster would: buffered output awaiting its timer flush.
        fallback = _BashState()
        fallback.generation = 7
        fallback.buffer.append("pending task-less output")
        printer._bash_states[""] = fallback
        tab = _RunningAgentState(tab_id, models[0])
        _RunningAgentState.running_agent_states[tab_id] = tab
        tab.agent = _ScriptedAgent("Sorcar VS Code")
        server._run_task({
            "tabId": tab_id,
            "prompt": "w2f6 reset probe",
            "workDir": str(tmp_path),
            "model": models[0],
        })
        # Pre-fix: the cleanup's printer.reset() ran on the worker
        # thread AFTER its thread-local task_id was cleared to "", so
        # it bumped the fallback state's generation and discarded its
        # buffered text.
        assert fallback.generation == 7
        assert fallback.buffer == ["pending task-less output"]
    finally:
        _pop_tabs(tab_id)


# ---------------------------------------------------------------------------
# F5 — a slow broadcast must not block other tasks' bash streaming
# ---------------------------------------------------------------------------


class _BlockingBroadcastPrinter(JsonPrinter):
    """Printer whose transport stalls on one specific bash flush."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict[str, Any]] = []
        self.a_in_broadcast = threading.Event()
        self.release_a = threading.Event()

    def broadcast(self, event: dict[str, Any]) -> None:
        """Stall the system_output send carrying AAA (slow client)."""
        if event.get("type") == "system_output" and "AAA" in event.get(
            "text", "",
        ):
            self.a_in_broadcast.set()
            self.release_a.wait(timeout=10)
        self.events.append(event)


def test_f5_slow_broadcast_does_not_block_other_tasks() -> None:
    printer = _BlockingBroadcastPrinter()

    def task_a() -> None:
        printer._thread_local.task_id = "w2f5-a"
        # ``last_flush`` starts at 0.0 so the first chunk takes the
        # inline (immediate) flush path straight into broadcast.
        printer.print("AAA", type="bash_stream")

    thread_a = threading.Thread(target=task_a, daemon=True)
    thread_a.start()
    assert printer.a_in_broadcast.wait(timeout=5)

    done_b = threading.Event()

    def task_b() -> None:
        printer._thread_local.task_id = "w2f5-b"
        printer.print("BBB", type="bash_stream")
        done_b.set()

    thread_b = threading.Thread(target=task_b, daemon=True)
    thread_b.start()
    try:
        # Pre-fix: task A's stalled broadcast held the printer-global
        # ``_bash_lock``, so task B's print() wedged behind it.
        assert done_b.wait(timeout=1.0), (
            "another task's bash_stream blocked behind a slow broadcast"
        )
    finally:
        printer.release_a.set()
        thread_a.join(timeout=5)
        thread_b.join(timeout=5)
    assert not thread_a.is_alive()
    # Task A's flush is still delivered once released.
    assert any(
        e.get("type") == "system_output" and "AAA" in e.get("text", "")
        for e in printer.events
    )


def test_f5_reset_during_inflight_flush_still_discards_stale_text() -> None:
    """Regression guard: the reset-vs-flush TOCTOU stays closed.

    A ``reset()`` that starts BEFORE the flush drains the buffer must
    win (text discarded); one that starts while the flush is already
    broadcasting must block until the broadcast finishes (text was
    still part of the old turn).  Here we exercise the first ordering
    deterministically: reset, then flush — nothing may be emitted.
    """
    printer = _CapturePrinter()
    printer._thread_local.task_id = "w2f5-reset"
    with printer._bash_lock:
        bs = printer._bash_state
        bs.buffer.append("stale text")
        bs.last_flush = time.monotonic()
    printer.reset()
    printer._flush_bash()
    assert all(e.get("type") != "system_output" for e in printer.events)


# ---------------------------------------------------------------------------
# F9 — the agent must wait on the SAME queue it drained and registered
# ---------------------------------------------------------------------------


class _AskUserGatePrinter(_CapturePrinter):
    """Printer that gates the askUser broadcast so the test can act
    between the queue registration and the blocking wait."""

    def __init__(self) -> None:
        super().__init__()
        self.ask_user_sent = threading.Event()
        self.release_ask_user = threading.Event()

    def broadcast(self, event: dict[str, Any]) -> None:
        """Record *event*; hold the askUser send until released."""
        super().broadcast(event)
        if event.get("type") == "askUser":
            self.ask_user_sent.set()
            self.release_ask_user.wait(timeout=10)


def test_f9_ask_user_question_waits_on_registered_queue() -> None:
    printer = _AskUserGatePrinter()
    server = VSCodeServer(printer=printer)
    tab_id = "w2f9-tab"
    task_key = "990901"
    try:
        tab = _RunningAgentState(tab_id, "test-model")
        _RunningAgentState.running_agent_states[tab_id] = tab
        q1: queue.Queue[str] = queue.Queue(maxsize=1)
        tab.user_answer_queue = q1
        printer.subscribe_tab(task_key, tab_id)

        outcome: dict[str, Any] = {}

        def agent_thread() -> None:
            printer._thread_local.task_id = task_key
            printer._thread_local.stop_event = threading.Event()
            try:
                outcome["answer"] = server._ask_user_question("Proceed?")
            except BaseException as exc:  # pragma: no cover — fail path
                outcome["error"] = exc

        worker = threading.Thread(target=agent_thread, daemon=True)
        worker.start()
        assert printer.ask_user_sent.wait(timeout=5)
        # The question was drained + registered against q1.
        with server._state_lock:
            assert server._pending_user_answer_tasks.get(id(q1)) == task_key
        # While the askUser broadcast is still in flight, the owner tab
        # is disposed and re-created (closeTab + reopen): its state now
        # carries a brand-new queue.  Pre-fix,
        # ``_await_user_response``'s RE-resolution picked up this new
        # queue, so the answer delivered to the registered queue never
        # woke the agent.
        q2: queue.Queue[str] = queue.Queue(maxsize=1)
        tab.user_answer_queue = q2
        printer.release_ask_user.set()
        q1.put("yes-go-ahead", timeout=1)
        worker.join(timeout=2)
        assert not worker.is_alive(), (
            "agent blocked on a re-resolved queue instead of the one it "
            "drained and registered"
        )
        assert outcome.get("answer") == "yes-go-ahead"
        # The registration was cleaned up on exit.
        with server._state_lock:
            assert id(q1) not in server._pending_user_answer_tasks
    finally:
        _pop_tabs(tab_id)


# ---------------------------------------------------------------------------
# F13 — saveConfig work_dir change must mirror setWorkDir's discipline
# ---------------------------------------------------------------------------


class _WorkDirPrinter(_CapturePrinter):
    """Printer that (like WebPrinter) exposes a ``work_dir`` attribute."""

    def __init__(self) -> None:
        super().__init__()
        self.work_dir = ""


def test_f13_save_config_work_dir_syncs_cache_and_printer(
    tmp_path: Path,
) -> None:
    printer = _WorkDirPrinter()
    server = VSCodeServer(printer=printer)
    new_dir = str(tmp_path / "proj")
    os.makedirs(new_dir)
    server._file_cache["stale"] = ["old.py"]
    server._cmd_save_config({"config": {"work_dir": new_dir}})
    assert server.work_dir == new_dir
    # Pre-fix: the printer kept reporting the old folder and the stale
    # @-mention cache survived the switch.
    assert printer.work_dir == new_dir
    assert server._file_cache == {}


def test_f13_concurrent_save_config_is_serialised(tmp_path: Path) -> None:
    """Concurrent saveConfig commands must not lose each other's keys
    nor corrupt ``work_dir`` state (lock-protected read-modify-write)."""
    from kiss.core.vscode_config import load_config

    printer = _WorkDirPrinter()
    server = VSCodeServer(printer=printer)
    dirs = []
    for i in range(2):
        d = str(tmp_path / f"proj{i}")
        os.makedirs(d)
        dirs.append(d)

    def save(i: int) -> None:
        time.sleep(random.random() * 0.05)
        server._cmd_save_config({
            "config": {"work_dir": dirs[i], "use_web_browser": bool(i)},
        })

    threads = [
        threading.Thread(target=save, args=(i,), daemon=True)
        for i in range(2)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert all(not t.is_alive() for t in threads)
    # Whichever save won last, server / printer / config agree.
    cfg = load_config()
    assert server.work_dir in dirs
    assert printer.work_dir == server.work_dir
    assert cfg.get("work_dir") == server.work_dir


# ---------------------------------------------------------------------------
# F12 — concurrent model downloads must be serialised and atomic
# ---------------------------------------------------------------------------


def _make_model_zip(src_dir: Path, model_name: str) -> Path:
    """Build a tiny zip archive shaped like a Vosk model download."""
    inner = src_dir / model_name
    (inner / "conf").mkdir(parents=True)
    (inner / "conf" / "model.conf").write_text("ok")
    (inner / "README").write_text("marker")
    zip_path = src_dir / f"{model_name}.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for p in sorted(inner.rglob("*")):
            zf.write(p, p.relative_to(src_dir))
    return zip_path


def test_f12_concurrent_model_download_is_safe(tmp_path: Path) -> None:
    from kiss.server.voice_wake import _ensure_downloaded_model

    model_name = "vosk-model-test-w2f12"
    zip_path = _make_model_zip(tmp_path / "src", model_name)
    url = zip_path.as_uri()
    models_dir = tmp_path / "models"
    errors: list[BaseException] = []
    results: list[Path] = []

    def worker() -> None:
        try:
            time.sleep(random.random() * 0.05)
            results.append(
                _ensure_downloaded_model(models_dir, model_name, url),
            )
        except BaseException as exc:  # pragma: no cover — fail path
            errors.append(exc)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert all(not t.is_alive() for t in threads)
    assert errors == []
    assert len(results) == 6
    model_dir = models_dir / model_name
    assert all(r == model_dir for r in results)
    # The published model is complete.
    assert (model_dir / "conf" / "model.conf").read_text() == "ok"
    assert (model_dir / "README").read_text() == "marker"
    # No temp zips, temp extract dirs, or leftover archives remain.
    leftovers = [
        p.name
        for p in models_dir.iterdir()
        if p.name not in (model_name, f".{model_name}.lock")
    ]
    assert leftovers == []
