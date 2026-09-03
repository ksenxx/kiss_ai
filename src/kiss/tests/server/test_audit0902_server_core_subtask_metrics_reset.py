# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-09-02 (server-core): per-subtask metrics vs the agent's reset.

``RelentlessAgent.run`` starts every run with ``_reset``, which zeroes
``budget_used`` / ``total_tokens_used`` / ``total_steps``.  The task
runner nevertheless captured those counters BEFORE each ``agent.run``
of a multi-``<task>`` prompt as "baselines" and persisted the run's
metrics as ``max(0, counter - baseline)``.  For the second and later
subtasks the baseline is the PREVIOUS subtask's total while the counter
holds only THIS subtask's usage, so a subtask that spent less than its
predecessor was recorded — in ``task_history.extra`` and in its
failure banner — as having cost nothing, and one that spent more was
recorded with the difference instead of its own usage.

The scripted agent below follows the real ``run`` contract by calling
the real ``_reset`` (the same code path production runs take) before
recording the subtask's usage, so the runner sees exactly the counter
values a live agent produces.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
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

_USAGE: dict[str, tuple[int, float, int]] = {
    "a0902 first": (300, 0.30, 6),
    "a0902 second": (100, 0.10, 2),
    "a0902 third fails": (50, 0.05, 1),
}


class _CapturePrinter(JsonPrinter):
    """Real printer subclass that records every broadcast event."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict[str, Any]] = []

    def broadcast(self, event: dict[str, Any]) -> None:
        """Record *event* instead of writing it to a transport."""
        self.events.append(event)


class _ResettingAgent(WorktreeSorcarAgent):
    """Real agent whose ``run`` resets its counters exactly like production.

    Mirrors what ``ChatSorcarAgent.run`` does under
    ``_skip_persistence=True`` — a fresh ``task_history`` row via the
    real ``_add_task``, ``_last_task_id`` published under
    ``_task_id_lock``, the printer's thread-local ``task_id`` bound for
    the run — and, crucially, the real ``_reset`` at the top, which is
    where the live agent zeroes its usage counters.
    """

    def run(self, *args: Any, **kwargs: Any) -> str:
        prompt_template = kwargs.get("prompt_template", "")
        printer = kwargs.get("printer")
        self._reset(
            kwargs.get("model_name"), None, None, None,
            kwargs.get("work_dir"), None, printer, False,
        )
        task_id, self._chat_id = _add_task(
            prompt_template, chat_id=self._chat_id or "",
        )
        with self._task_id_lock:
            self._last_task_id = task_id
        if printer is not None:
            printer._thread_local.task_id = str(task_id)
        tokens, cost, steps = _USAGE[prompt_template]
        self.total_tokens_used += tokens
        self.budget_used += cost
        if prompt_template.endswith("fails"):
            # Plain-agent shape: steps land in ``step_count`` while
            # ``total_steps`` stays 0, exercising the runner's fallback.
            self.step_count = steps
        else:
            self.total_steps += steps
        if printer is not None:
            printer._thread_local.task_id = ""
        if prompt_template.endswith("fails"):
            raise RuntimeError(f"boom {prompt_template}")
        return (
            "success: true\n"
            "is_continue: false\n"
            f"summary: done {prompt_template}\n"
        )


def _fetch_rows(tasks: list[str]) -> dict[str, tuple[int, float, int]]:
    """Fetch ``(tokens, cost, steps)`` per task text from the history DB."""
    conn = sqlite3.connect(str(persistence._DB_PATH))
    try:
        rows = conn.execute(
            "SELECT task, tokens, cost, steps FROM task_history "
            "WHERE task IN ({})".format(",".join("?" * len(tasks))),
            tasks,
        ).fetchall()
    finally:
        conn.close()
    return {r[0]: (r[1], r[2], r[3]) for r in rows}


@pytest.fixture
def server_and_printer(
    tmp_path: Path,
) -> Iterator[tuple[VSCodeServer, _CapturePrinter, str]]:
    models = get_available_models()
    if not models:
        pytest.skip("no models configured in this environment")
    printer = _CapturePrinter()
    server = VSCodeServer(printer)
    yield server, printer, models[0]
    with agent_state.STATE_LOCK:
        for st in list(agent_state.agent_states.values()):
            if st.tab_id.startswith("a0902"):
                agent_state.agent_states.pop(st.task_id, None)


def _run(
    server: VSCodeServer, model: str, tmp_path: Path, tab_id: str, prompt: str,
) -> _ResettingAgent:
    agent = _ResettingAgent("Sorcar VS Code")
    # A previous task on the tab left the agent (and thus its counters)
    # behind, exactly like a pending worktree carry-over does.
    agent.total_tokens_used = 5000
    agent.budget_used = 5.0
    agent.total_steps = 90
    agent.step_count = 7
    state = AgentState(
        f"pre-{tab_id}", agent=agent, tab_id=tab_id, server_owned=True,
    )
    agent_state.register(state)
    server._run_task({
        "tabId": tab_id,
        "prompt": prompt,
        "workDir": str(tmp_path),
        "model": model,
        "useWorktree": False,
        "autoCommit": False,
        "_state_key": state.task_id,
    })
    return agent


def test_each_subtask_row_records_its_own_usage(
    server_and_printer: tuple[VSCodeServer, _CapturePrinter, str],
    tmp_path: Path,
) -> None:
    server, _printer, model = server_and_printer
    _run(
        server, model, tmp_path, "a0902-rows",
        "<task>a0902 first</task>\n<task>a0902 second</task>",
    )
    rows = _fetch_rows(["a0902 first", "a0902 second"])
    assert set(rows) == {"a0902 first", "a0902 second"}
    for task_text, (tokens, cost, steps) in rows.items():
        want_tokens, want_cost, want_steps = _USAGE[task_text]
        assert tokens == want_tokens, (
            f"{task_text!r}: persisted {tokens} tokens, spent {want_tokens} "
            "— the runner subtracted a baseline the agent's _reset had "
            "already discarded"
        )
        assert cost == pytest.approx(want_cost), (task_text, cost)
        assert steps == want_steps, (task_text, steps)


def test_failure_banner_reports_the_failed_subtasks_own_usage(
    server_and_printer: tuple[VSCodeServer, _CapturePrinter, str],
    tmp_path: Path,
) -> None:
    server, printer, model = server_and_printer
    _run(
        server, model, tmp_path, "a0902-banner",
        "<task>a0902 first</task>\n<task>a0902 third fails</task>",
    )
    banners = [
        e for e in printer.events
        if e.get("type") == "result" and e.get("success") is False
    ]
    assert banners, "missing failure result broadcast"
    banner = banners[-1]
    assert "boom a0902 third fails" in str(banner.get("text", ""))
    want_tokens, want_cost, want_steps = _USAGE["a0902 third fails"]
    assert banner.get("total_tokens") == want_tokens, banner
    assert banner.get("cost") == f"${want_cost:.4f}", banner
    assert banner.get("step_count") == want_steps, banner
    rows = _fetch_rows(["a0902 third fails"])
    assert rows["a0902 third fails"] == (
        want_tokens, pytest.approx(want_cost), want_steps,
    )
