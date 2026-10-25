# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Periodic task updates for the task-info panel.

While a chat page shows a running task, its task-info panel asks the
daemon (``getTaskUpdate``) for a short progress update.  The update is
rendered from what the task already reports (:func:`progress_digest`):
its latest ``summary`` — the model's own checkpoint or the executor's
ten-step recap — and the tool calls since, read from the persisted
events on each poll, so it costs no model call and stays live during
an hour-long wait.  Only the panel's refresh button runs the ``/ask``
agent (:mod:`~kiss.agents.seas.ask.ask_sea`) on :data:`UPDATE_QUESTION`,
the same answer a user gets by typing ``/ask`` into the task's chat;
:class:`TaskUpdateRunner` keeps one update per task and never doubles a
run.  (Before the 2026-10-24 trajectory audit a timer ran the agent a
minute after the task started and every ten minutes after: 207 probes
and $62 in a week, many fired while the task slept in a wait.)

:func:`run_task_update_sea` runs the agent in-process the way
:func:`~kiss.agents.sorcar.merge_conflict_resolver.run_merge_sea` runs the merge
agent: as a sub-agent of the task, in the task's own chat, so it opens as
a nested tab under the task's tab and its history row nests under the
task's; its spend is attributed to the task.
"""

from __future__ import annotations

import dataclasses
import hashlib
import html
import json
import logging
import re
import threading
import time
from collections.abc import Callable
from typing import Any

from kiss.agents.seas.ask.ask_sea import AskSea
from kiss.agents.sorcar._concurrency import _race_delay, locked_call
from kiss.agents.sorcar.persistence import TASK_USAGE_LOCK
from kiss.agents.sorcar.sea_commands import evaluate_sea
from kiss.core.step_recap import describe_args
from kiss.server.json_printer import stamp_event_ts

log = logging.getLogger(__name__)

# The question the panel asks the ``/ask`` agent about the task (refresh only).
UPDATE_QUESTION = "What has this task done so far, and what are its partial results?"
# USD cap of one run.
UPDATE_BUDGET_USD = 1.0
# Updates of tasks that stopped being polled are dropped after this long.
_PRUNE_AFTER_S = 3600.0
# A poll re-reads the task's events for the digest at most this often.
DIGEST_REFRESH_S = 15.0
# Tool calls listed after the latest summary (older ones are counted).
DIGEST_MAX_CALLS = 25
# Characters of a tool call's title kept in the digest.
_DIGEST_TITLE_CHARS = 120

# ``(parent_agent, task_id) -> (answer_html, cost_usd)``.
TaskUpdateSeaRunner = Callable[[Any, str], tuple[str, float]]


@dataclasses.dataclass
class TaskUpdate:
    """The task-info panel's update for one task.

    Attributes:
        text: The ``/ask`` agent's answer (HTML) to the last refresh,
            ``""`` until the user asked.  A failed run keeps the
            previous answer.
        digest: The HTML digest of the task's own events (its latest
            ``summary`` and the tool calls since), refreshed by polls.
        digest_at: ``time.time()`` when the digest was last computed.
        error: The failure of the last run, ``""`` when it succeeded.
        finished_at: ``time.time()`` when the last run ended (0 = never).
        cost: USD spent by the last run.
        running: Whether a run is in flight.
        polled_at: ``time.time()`` of the last poll (for pruning).
    """

    text: str = ""
    digest: str = ""
    digest_at: float = 0.0
    error: str = ""
    finished_at: float = 0.0
    cost: float = 0.0
    running: bool = False
    polled_at: float = 0.0

    @property
    def sig(self) -> str:
        """Return a fingerprint that changes whenever the panel must repaint."""
        digest_hash = hashlib.sha1(self.digest.encode("utf-8")).hexdigest()[:10]
        return f"{self.finished_at:.3f}:{int(self.running)}:{digest_hash}"

    @property
    def content(self) -> str:
        """Return the panel body: the agent's answer, when there is one, above the digest."""
        if not self.text:
            return self.digest
        return f'<div class="task-update-answer">{self.text}</div>\n{self.digest}'

    def payload(self) -> dict[str, Any]:
        """Return the wire fields of a ``taskUpdate`` reply.

        ``exists`` is true from the first poll on.  No run is ever
        scheduled: the digest keeps the panel current and the agent
        runs only on the panel's refresh button.
        """
        return {
            "exists": True,
            "content": self.content,
            "error": self.error,
            "running": self.running,
            "cost": self.cost,
            "updatedAt": int(self.finished_at * 1000),
            "sig": self.sig,
        }


_INLINE_CODE = re.compile(r"`([^`]+)`")
_BOLD = re.compile(r"\*\*([^*]+)\*\*")


def _inline_html(text: str) -> str:
    """Return *text* HTML-escaped with Markdown backticks and ``**bold**`` rendered."""
    escaped = html.escape(text, quote=False)
    escaped = _INLINE_CODE.sub(r"<code>\1</code>", escaped)
    return _BOLD.sub(r"<strong>\1</strong>", escaped)


def _summary_html(markdown: str) -> str:
    """Return the HTML of a ``summary`` description: bullets as a list, other lines as ``<p>``."""
    parts: list[str] = []
    items: list[str] = []
    for raw in markdown.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line[:2] in ("- ", "* "):
            items.append(f"<li>{_inline_html(line[2:])}</li>")
            continue
        if items:
            parts.append("<ul>" + "".join(items) + "</ul>")
            items = []
        parts.append(f"<p>{_inline_html(line)}</p>")
    if items:
        parts.append("<ul>" + "".join(items) + "</ul>")
    return "".join(parts)


def _call_title(entry: Any) -> str:
    """Return the one-line title of a TOOL CALL digest entry from its JSON arguments."""
    try:
        args = json.loads(entry.text) if entry.text else {}
    except ValueError:
        args = {}
    if not isinstance(args, dict):
        args = {}
    return describe_args(args)[:_DIGEST_TITLE_CHARS]


def progress_digest(task_id: str) -> str:
    """Return the HTML digest of what *task_id* has done: its latest summary and the calls since.

    Rendered from the task's persisted events, so it costs no model
    call and stays live during a long wait: the ``summary`` events are
    the model's own checkpoints or the executor's ten-step recaps
    (:mod:`kiss.core.step_recap`), the tool log is every call since.

    Args:
        task_id: The task's persisted id.

    Returns:
        The HTML, ``""`` for a task without a history row.
    """
    from kiss.agents.sorcar import task_digest

    task = task_digest.load_task(task_id)
    if task is None:
        return ""
    entries, _spend = task_digest.digest_events(task["events"])
    last_summary = -1
    for index, entry in enumerate(entries):
        if entry.kind == "SUMMARY":
            last_summary = index
    calls_before = sum(1 for e in entries[:last_summary] if e.kind == "TOOL CALL")
    calls_since = [e for e in entries[last_summary + 1:] if e.kind == "TOOL CALL"]
    parts = ['<section class="task-update-digest">']
    if last_summary >= 0:
        parts.append(f"<h4>Latest summary (after {calls_before} tool calls)</h4>")
        parts.append(_summary_html(entries[last_summary].text))
    else:
        parts.append("<h4>No summary yet</h4>")
    if calls_since:
        parts.append(f"<h4>Since then ({len(calls_since)} tool calls)</h4>")
        shown = calls_since[-DIGEST_MAX_CALLS:]
        if len(calls_since) > len(shown):
            parts.append(f"<p>… {len(calls_since) - len(shown)} earlier calls</p>")
        items = "".join(
            f"<li><code>{html.escape(e.name)}</code> {_inline_html(_call_title(e))}</li>"
            for e in shown
        )
        parts.append(f'<ol class="task-update-log">{items}</ol>')
    else:
        parts.append("<p>No tool calls since.</p>")
    parts.append("</section>")
    return "\n".join(parts)


def mark_legacy_updates_as_side_channels() -> int:
    """Stamp task-update children persisted before the side-channel flag.

    Rows written by earlier releases of :func:`run_task_update_sea`
    (which ran the ``/task_update`` agent with its prompt template)
    carry ``is_side_channel = 0``, so every reload of a chat re-opened
    each finished update as a dead sub-agent tab.  The daemon calls this
    once at startup; it is idempotent.

    Returns:
        The number of rows newly stamped.
    """
    from kiss.agents.seas.task_update import task_update_sea
    from kiss.agents.sorcar.persistence import _mark_legacy_side_channel_rows

    return _mark_legacy_side_channel_rows(task_update_sea.PROMPT_TEMPLATE)


def charge_side_channel_usage(
    printer: Any,
    task_agent: Any,
    task_id: str,
    budget: float,
    tokens: int,
    steps: int,
    epoch: Any = None,
) -> None:
    """Charge a side channel's spend to the task it worked for.

    A side channel (a task-update run, an ``/ask`` answerer) may end
    before or after its task.  While the task runs, the spend is banked
    on *task_agent*'s live counters (bound to *epoch*), so the task's
    final save and, for a sub-agent task, its parent's fold include
    it.  Once the task has finished, the spend is added to its row and
    to each finished ancestor's row
    (:func:`~kiss.agents.sorcar.persistence._add_late_task_usage`),
    every updated task's tabs get its new totals as a persisted
    ``usage_info``, and a still-running ancestor, when there is one,
    banks the spend on its live agent.  A running task's new totals
    are broadcast as its own ``usage_info`` (recorded and persisted),
    so its replayed transcript ends with the cost its row will store.  A task
    with no row at all (a run that persists none) is banked on its
    live agent like an unfinished one.  A task that finishes in the
    moment between the row check and the bank, after its final save
    read its counters, loses the spend (the same narrow window every
    live-agent bank has).

    Args:
        printer: The printer that shows the task's tabs, or ``None``.
        task_agent: The live agent of the task *task_id*.
        task_id: The task's persisted ``task_history`` row id.
        budget: USD spent.
        tokens: Tokens spent.
        steps: Steps taken.
        epoch: *task_agent*'s usage epoch captured when the side channel
            started (see ``RelentlessAgent._attribute_usage``), or
            ``None`` for its current one.
    """
    if not task_id or (budget <= 0 and tokens <= 0 and steps <= 0):
        return
    locked_call(
        TASK_USAGE_LOCK, _charge_locked,
        printer, task_agent, task_id, budget, tokens, steps, epoch,
    )


def _charge_locked(
    printer: Any,
    task_agent: Any,
    task_id: str,
    budget: float,
    tokens: int,
    steps: int,
    epoch: Any,
) -> None:
    """Charge the spend to the row or the live agent (``TASK_USAGE_LOCK`` held).

    See :func:`charge_side_channel_usage` for the arguments.
    """
    from kiss.agents.sorcar.persistence import (
        _add_late_task_usage,
        _append_chat_event,
    )
    from kiss.agents.sorcar.sorcar_agent import _attribute_sub_usage, _live_agent_usage
    from kiss.server import agent_state

    try:
        updated, running = _add_late_task_usage(task_id, tokens, budget, steps)
    except Exception:
        log.warning(
            "could not add side-channel spend to task %s", task_id,
            exc_info=True,
        )
        return
    _race_delay()  # test hook: widens the row-check-to-bank window
    if not updated and not running:
        # No row for the task (its run persists none, or has not
        # saved one yet): the live agent is the only place the
        # spend can go, exactly as for an unfinished row.
        running = task_id
    banked = False
    if running:
        # Banked first: the task can finish (saving its counters) at
        # any moment after the transaction above released it.
        if running == task_id:
            agent = task_agent
        else:
            state = agent_state.get(running)
            agent = state.agent if state is not None else None
            epoch = None
        if agent is None:
            log.warning(
                "side-channel spend of task %s not charged to running "
                "task %s: no live agent", task_id, running,
            )
        else:
            _attribute_sub_usage(agent, budget, tokens, steps, epoch=epoch)
            banked = True
            if printer is not None:
                # Publish the new totals (banked plus the in-flight
                # session's) as one of the running task's own
                # events, recorded and persisted like its other
                # usage events: the task may already be past its
                # last one, and both its replayed transcript and a
                # ``run_agent`` caller waiting on it
                # (daemon_client.run) take the cost from the latest.
                live_budget, live_tokens, live_steps = _live_agent_usage(agent)
                printer.broadcast({
                    "type": "usage_info",
                    "text": "",
                    "taskId": running,
                    "total_tokens": live_tokens,
                    "cost": f"${live_budget:.4f}",
                    "cost_usd": live_budget,
                    "total_steps": live_steps,
                })
    for row_id, row_tokens, row_cost, row_steps in updated:
        # ``taskId``: the tab that showed this row may already run a
        # newer task, whose cost this late event must not overwrite
        # (the chat page drops usage events of another task).
        event: dict[str, Any] = {
            "type": "usage_info",
            "text": "",
            "taskId": row_id,
            "total_tokens": row_tokens,
            "cost": f"${row_cost:.4f}",
            "cost_usd": row_cost,
            "total_steps": row_steps,
        }
        if banked:
            # The spend was banked on the running ancestor above,
            # so a ``run_agent`` caller still waiting on this row's
            # task must not fold it again (daemon_client._net_totals
            # subtracts it).
            event["ancestor_charged"] = {
                "cost": budget, "tokens": tokens, "steps": steps,
            }
        stamp_event_ts(event)
        if printer is not None:
            printer.broadcast_transient(event, task_id=row_id)
        _append_chat_event(dict(event), task_id=row_id)
    if updated and printer is not None:
        printer.broadcast({"type": "tasks_updated"})


def run_task_update_sea(parent_agent: Any, task_id: str) -> tuple[str, float]:
    """Ask the ``/ask`` agent :data:`UPDATE_QUESTION` about *task_id*, in-process.

    The child is a :class:`~kiss.agents.sorcar.chat_sorcar_agent.ChatSorcarAgent`
    stamped like a ``run_parallel`` child (``_tab_id`` /
    ``_subagent_info``) and resumed on *parent_agent*'s chat, so the
    frontend opens it as a nested tab of the task's tab and its history
    row nests under the task's row in the same chat.  The stamp marks it
    a side channel: its tab is open only while it runs — a reload never
    re-opens the finished tab, because its answer lives in the task-info
    panel, not in the tab.  It is configured exactly like a ``/ask``
    typed into the task's chat (:mod:`~kiss.agents.seas.ask.ask_sea`'s
    getters: its base system prompt and playbook suffix, the
    ``task_context`` tool and ``finish`` only, no memory, browser or
    sub-agents), capped at :data:`UPDATE_BUDGET_USD`, with the parent's
    model unless the script defines ``model()``, in the parent's work
    dir.

    Its spend is charged to the task whatever way the run ends.  While
    the task still runs it is banked on *parent_agent*'s live counters,
    bound to the usage epoch captured at the start (a parent that has
    since reset for its next task discards it instead of billing that
    task); the task's final save then writes those totals to its row.
    When the task finished while the update ran (its row already holds
    the final totals) the spend is added to the row and its finished
    ancestors instead (see :func:`charge_side_channel_usage`), so it is
    never counted twice.  Only an update that ends in the few
    milliseconds between the final save reading the counters and
    writing the row can lose its spend.

    Args:
        parent_agent: The agent of the running task to report on.
        task_id: That task's persisted ``task_history`` row id.

    Returns:
        ``(answer, cost)``: the child's ``finish`` summary and the USD
        it spent.

    Raises:
        Exception: Whatever the child's run raised (after the spend was
            attributed).
    """
    from kiss.agents.sorcar.chat_sorcar_agent import (
        ChatSorcarAgent,
        _extract_result_summary,
    )
    from kiss.agents.sorcar.sorcar_agent import (
        _live_agent_usage,
        _notify_subagent_done,
        _persisted_task_id,
        subagent_parent_tab_id_of,
    )

    printer = getattr(parent_agent, "printer", None)
    # The tab the chat pages show the parent under — for a sub-agent
    # parent its ``{parent}__sub_{task}`` tab, never its synthetic id.
    parent_tab_id = subagent_parent_tab_id_of(parent_agent)
    sub_tab_id = f"task-{task_id}__update-{int(time.time() * 1000)}"
    ask = evaluate_sea([AskSea()], UPDATE_QUESTION, task_id)
    ask_settings = ask.settings
    model_name = str(ask_settings.get("model") or parent_agent.model_name)
    agent = ChatSorcarAgent("Task update")
    agent._tab_id = sub_tab_id
    # A side channel like the ``/ask`` answerer: its answer lands in the
    # task-info panel, so the finished child has no tab worth reopening.
    # The daemon replays a finished side channel as ``subagentDone``
    # instead of ``openSubagentTab`` (see ``server._is_side_channel_row``);
    # without the stamp every reload re-opened one finished tab per
    # periodic update.
    agent._subagent_info = {
        "parent_task_id": task_id,
        "parent_tab_id": parent_tab_id,
        "reviewer": False,
        "side_channel": True,
    }
    agent.resume_chat_by_id(str(getattr(parent_agent, "chat_id", "") or ""))
    epoch_getter = getattr(parent_agent, "_usage_epoch", None)
    epoch = epoch_getter() if callable(epoch_getter) else None
    result = ""
    try:
        result = agent.run(
            prompt_template=ask.prompt,
            model_name=model_name,
            work_dir=str(getattr(parent_agent, "work_dir", "") or "."),
            printer=printer,
            # The ask SEA's ``none`` tool profile: its ``tools()`` and
            # ``finish`` are the whole tool set, no built-in tools.
            tools_hook=ask.tools_hook,
            llm_call_hook=ask.llm_call_hook,
            tool_call_hook=ask.tool_call_hook,
            append_basic_tools=False,
            tool_profile=ask_settings["tool_profile"],
            max_budget=UPDATE_BUDGET_USD,
            model_config=(
                getattr(parent_agent, "model_config", None)
                if model_name == parent_agent.model_name else None
            ),
            system_prompt_hook=ask.system_prompt_hook,
            web_tools=ask_settings["use_web_tools"],
            use_memory=ask_settings["use_memory"],
        )
    finally:
        budget, tokens, steps = _live_agent_usage(agent)
        charge_side_channel_usage(
            printer, parent_agent, task_id, budget, tokens, steps, epoch=epoch,
        )
        if printer is not None:
            _notify_subagent_done(
                printer, _persisted_task_id(agent), sub_tab_id, model_name,
                parent_tab_id,
            )
    return _extract_result_summary(result), budget


class TaskUpdateRunner:
    """Keeps one :class:`TaskUpdate` per task: the live digest, plus the agent's answer on refresh.

    Args:
        run_sea: Runs the agent for ``(parent_agent, task_id)`` and
            returns ``(answer, cost)``; :func:`run_task_update_sea` by
            default.
    """

    def __init__(self, run_sea: TaskUpdateSeaRunner = run_task_update_sea) -> None:
        self._run_sea = run_sea
        self._updates: dict[str, TaskUpdate] = {}
        self._lock = threading.Lock()

    def poll(self, task_id: str, parent_agent: Any, force: bool = False) -> TaskUpdate:
        """Return the update for *task_id*; refresh its digest, start a run only on *force*.

        The digest is re-read from the task's events at most every
        :data:`DIGEST_REFRESH_S` (a history-database read: call this off
        the event loop, ``asyncio.to_thread``).  No timer ever starts
        the agent: only the panel's refresh button (*force*) does, and
        a run already in flight is never doubled.

        Args:
            task_id: The running task's persisted id.
            parent_agent: The running task's agent (the run's parent).
            force: Run the agent now (the panel's refresh button).

        Returns:
            A snapshot of the task's update state.
        """
        now = time.time()
        upd = self._update_for(task_id, now)
        digest = progress_digest(task_id) if now - upd.digest_at >= DIGEST_REFRESH_S else None
        with self._lock:
            if digest is not None:
                upd.digest, upd.digest_at = digest, now
            if force and not upd.running:
                # ``running`` is set only once the thread exists (a
                # failed ``start()`` under thread exhaustion would
                # otherwise pin the update on "running" for good);
                # ``_run``'s finishing write waits for this lock, so
                # it cannot be overtaken.
                threading.Thread(
                    target=self._run,
                    args=(task_id, parent_agent),
                    name=f"task-update-{task_id[:8]}",
                    daemon=True,
                ).start()
                upd.running = True
            return dataclasses.replace(upd)

    def _update_for(self, task_id: str, now: float) -> TaskUpdate:
        """Return the update of *task_id* polled at *now*, created on its first poll.

        Pruning runs first, then the polled update is touched so no
        concurrent poll prunes it.
        """
        with self._lock:
            self._prune(now)
            upd = self._updates.get(task_id)
            if upd is None:
                upd = TaskUpdate()
                self._updates[task_id] = upd
            upd.polled_at = now
            return upd

    def _prune(self, now: float) -> None:
        """Drop idle updates nobody polled for :data:`_PRUNE_AFTER_S`."""
        stale = [
            tid for tid, upd in self._updates.items()
            if not upd.running and now - upd.polled_at >= _PRUNE_AFTER_S
        ]
        for tid in stale:
            del self._updates[tid]

    def _run(self, task_id: str, parent_agent: Any) -> None:
        """Thread body: run the agent and record its answer."""
        text, cost, error = "", 0.0, ""
        try:
            text, cost = self._run_sea(parent_agent, task_id)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            log.warning("task update for %s failed", task_id, exc_info=True)
        finally:
            # Also on a BaseException: an update pinned on ``running``
            # would never be restarted by ``poll`` nor pruned.
            with self._lock:
                upd = self._updates.setdefault(task_id, TaskUpdate())
                if text:
                    upd.text = text
                upd.error = error
                upd.cost = cost
                upd.running = False
                upd.finished_at = time.time()
