# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``/recommendations``: ten suggested next tasks from the user's recent work.

The agent reads the tasks the user ran (``recent_tasks``: the last
:data:`RECENT_HOURS` hours and, further back, enough older tasks to
review at least :data:`MIN_COMPLETED` completed ones), checks which
third-party apps are signed in (``connected_apps``), recalls from memory
what the user did in those apps, picks :data:`RECOMMENDATION_COUNT`
concrete tasks, of which at least :data:`APP_RECOMMENDATION_COUNT`
combine the connected apps, and stores them with ``save_recommendations``
in :func:`recommendations_file` (``<home>/recommendations.json``).  The
right sidebar's RECOMMENDATIONS subpanel shows that file
(:func:`load_recommendations`); a 6-hourly cron prompt job
``/recommendations now`` keeps it fresh.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from kiss.agents.seas.base.base_sea import WorkerSea
from kiss.agents.sorcar import persistence
from kiss.agents.sorcar.cron_agent import PROMPT_PREAMBLE
from kiss.agents.third_party_agents.auth_status import probe_statuses
from kiss.core.config import kiss_home
from kiss.core.memoryfield.tools import MemoryTools
from kiss.core.utils import atomic_write_text

RECENT_HOURS = 24
"""Every task of the last this many hours is listed."""

MIN_COMPLETED = 20
"""``recent_tasks`` reaches back past :data:`RECENT_HOURS` until this many completed tasks show."""

MAX_DAYS = 30
"""How far back ``recent_tasks`` looks for those completed tasks."""

RECOMMENDATION_COUNT = 10
"""How many suggestions one run stores."""

APP_RECOMMENDATION_COUNT = 3
"""How many of them must combine the signed-in third-party apps (when any is signed in)."""

MAX_TASKS = 100
"""Most tasks ``recent_tasks`` lists (the newest), after the automations are dropped."""

MAX_TASK_CHARS = 300
MAX_RESULT_CHARS = 200

OWN_SEA_PREFIXES = ("recommendations", "cron_prompt")
"""``task_history.sea`` prefixes of automations, not of the user's own tasks."""

PROMPT = f"Suggest {RECOMMENDATION_COUNT} tasks from my recent work and my connected apps."
"""Every run's task text (a run outside the daemon has no ``sea`` column to drop it by)."""

SYSTEM_PROMPT = (
    "You suggest the user's next tasks for their AI assistant.\n\n"
    "1. Call `recent_tasks()` once. It lists the tasks the user ran, newest first, with "
    f"directory and outcome: everything from the last {RECENT_HOURS} hours and, below "
    f"that, older tasks until at least {MIN_COMPLETED} completed ones are shown. Review "
    "every completed task (ok or failed) in the list before suggesting anything.\n"
    "2. Call `connected_apps()` once. It names the third-party apps the user is signed in "
    "to (Gmail, Google Calendar, Slack, GitHub, ...). When it names any, call "
    "`memory_pull` once or twice (e.g. `memory_pull(\"tasks done with <app>\")`) to learn "
    "what the user actually did in those apps before; a memory tool error or an empty "
    "memory just means there is no such record.\n"
    f"3. Choose {RECOMMENDATION_COUNT} suggestions grounded in that list: finish what "
    "failed or was stopped, the next step after a success, a check the changed area "
    "now needs, a repeated chore that could be automated. Each must name the file, "
    "feature, repository or topic it comes from. No generic advice, no repeat of a "
    "task that already succeeded, no two suggestions about the same thing. Do not "
    "echo a 'Suggested next' or follow-up line from a task's result: derive every "
    "suggestion from what the tasks did and what they left undone.\n"
    f"   When apps are connected, at least {APP_RECOMMENDATION_COUNT} of the "
    f"{RECOMMENDATION_COUNT} must use them: each combining two or more of the connected "
    "apps (for example mail plus calendar, a document plus a chat channel) when several "
    "are connected, or building on the one app when only one is, in a way that "
    "continues how the user used them per memory and the task list; the rest come from "
    "the task list. With no connected app, all come from the task list.\n"
    "4. Write each suggestion as one imperative sentence of at most 160 characters, "
    "ready to submit as a prompt.\n"
    f"5. Call `save_recommendations(tasks)` with the {RECOMMENDATION_COUNT} sentences, "
    "then `finish` with success=true and the same sentences as an HTML list.\n\n"
    f"If the task list is empty, save {RECOMMENDATION_COUNT} first tasks for a new "
    "software project (summarise the README, run the test suite, list the open TODOs, "
    "...), still with the app-combining ones when apps are connected. Never ask the "
    "user anything."
)

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def recommendations_file() -> Path:
    """Path of the JSON file the sidebar reads: ``<home>/recommendations.json``."""
    return kiss_home() / "recommendations.json"


def load_recommendations() -> dict[str, Any]:
    """Return the stored recommendations.

    Returns:
        ``{"generatedAt": float, "tasks": list[str]}`` (epoch seconds of
        the last run and its sentences), or ``{"generatedAt": 0.0,
        "tasks": []}`` when no run has stored any yet or the file is
        unreadable.
    """
    try:
        data = json.loads(recommendations_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"generatedAt": 0.0, "tasks": []}
    tasks = data.get("tasks") if isinstance(data, dict) else None
    if not isinstance(tasks, list):
        return {"generatedAt": 0.0, "tasks": []}
    generated = data.get("generatedAt", 0.0)
    return {
        "generatedAt": float(generated) if isinstance(generated, (int, float)) else 0.0,
        "tasks": [str(task) for task in tasks if str(task).strip()],
    }


def _plain(text: str, limit: int) -> str:
    """Collapse *text* (possibly HTML) to one line of at most *limit* chars."""
    flat = _WS_RE.sub(" ", _TAG_RE.sub(" ", text)).strip()
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"


def _age(seconds: float) -> str:
    """Render an age in seconds as ``5m``, ``3h`` or ``2d``."""
    if seconds < 3600:
        return f"{max(1, int(seconds // 60))}m"
    if seconds < 86400:
        return f"{int(seconds // 3600)}h"
    return f"{int(seconds // 86400)}d"


def _is_automation(row: dict[str, object]) -> bool:
    """True for a history row no user typed: a cron run or a run of this agent.

    A cron prompt job's task opens with the unattended preamble; a cron
    job without its own work directory runs in a scratch directory
    under ``<home>/cron/`` (the ``in_chat`` morning greeting among
    them); and this agent's own runs carry its ``sea`` or its fixed
    prompt.
    """
    task = str(row["task"])
    cron_dir = str(kiss_home() / "cron") + os.sep
    return (
        str(row["sea"]).startswith(OWN_SEA_PREFIXES)
        or task == PROMPT
        or task.startswith(PROMPT_PREAMBLE)
        or str(row["workDir"]).startswith(cron_dir)
    )


def recent_tasks() -> str:
    """List the tasks the user ran recently, newest first.

    Every top-level task of the last :data:`RECENT_HOURS` hours, then
    older ones (up to :data:`MAX_DAYS` days back) until at least
    :data:`MIN_COMPLETED` completed tasks are listed, so a quiet day
    still gives the model a day's worth of finished work to review.
    One line per task: how long ago it started, the directory it ran
    in, its outcome (ok / failed / unfinished), the request and the
    first line of the result.  Sub-agents, side channels, cron runs and
    earlier runs of this agent are left out; at most the newest
    :data:`MAX_TASKS` remaining tasks are listed.

    Returns:
        The lines, or ``"No tasks in the last 30 days."``.
    """
    now = time.time()
    rows = persistence.user_tasks_since(now - MAX_DAYS * 86400)
    recent_cutoff = now - RECENT_HOURS * 3600
    lines: list[str] = []
    completed = 0
    for row in rows:
        if _is_automation(row):
            continue
        started = float(row["timestamp"])  # type: ignore[arg-type]
        if len(lines) == MAX_TASKS or (completed >= MIN_COMPLETED and started < recent_cutoff):
            break
        success = row["success"]
        completed += success is not None
        outcome = "unfinished" if success is None else ("ok" if success else "failed")
        task = _plain(str(row["task"]), MAX_TASK_CHARS)
        line = f"- {_age(now - started)} ago | {row['workDir']} | {outcome} | {task}"
        # An unfinished row holds the "Agent Failed Abruptly" placeholder, not a result.
        result = "" if success is None else _plain(str(row["result"]), MAX_RESULT_CHARS)
        if result:
            line += f" -> {result}"
        lines.append(line)
    if not lines:
        return f"No tasks in the last {MAX_DAYS} days."
    return "\n".join(lines)


def connected_apps() -> str:
    """Name the third-party apps the user is signed in to.

    Probes every installed channel agent in a subprocess (up to about
    45 seconds); an app whose probe failed or timed out counts as not
    connected.

    Returns:
        ``"Connected apps: Gmail, Google Calendar."`` or ``"No
        third-party app is connected."``.
    """
    labels = [
        str(app.get("label") or app.get("name"))
        for app in probe_statuses()
        if app.get("authenticated") is True
    ]
    if not labels:
        return "No third-party app is connected."
    return "Connected apps: " + ", ".join(labels) + "."


def memory_read_tools() -> list[Callable[..., Any]]:
    """Return read-only tools over the user's general memory, or none when memory is off.

    ``memory_search``, ``memory_pull`` and ``memory_read`` of a
    :class:`MemoryTools` on the configured memory root (see
    :func:`kiss.agents.sorcar.sorcar_agent._memory_settings`); the
    general memory is where the user's work in third-party apps is
    recorded, so the domain memories of repositories are not attached.
    """
    from kiss.agents.sorcar.sorcar_agent import _memory_settings

    enabled, root = _memory_settings()
    if not enabled:
        return []
    memory = MemoryTools(root)
    return [memory.memory_search, memory.memory_pull, memory.memory_read]


def save_recommendations(tasks: list[str]) -> str:
    """Store the suggested tasks for the sidebar's RECOMMENDATIONS subpanel.

    Args:
        tasks: Exactly ten non-empty one-sentence task prompts.

    Returns:
        ``"Saved 10 recommendations."`` or an error describing what to fix.
    """
    if not isinstance(tasks, list) or not all(isinstance(task, str) for task in tasks):
        return f"Error: pass a list of {RECOMMENDATION_COUNT} strings."
    cleaned = [_WS_RE.sub(" ", task).strip() for task in tasks]
    if len(cleaned) != RECOMMENDATION_COUNT or any(not task for task in cleaned):
        return f"Error: pass exactly {RECOMMENDATION_COUNT} non-empty tasks (got {len(cleaned)})."
    # Staged in a per-writer temp file: a scheduled run and a manual one
    # may overlap, and the sidebar must never read a half-written file.
    atomic_write_text(
        recommendations_file(),
        json.dumps({"generatedAt": time.time(), "tasks": cleaned}, indent=2),
        create_mode=0o644,
    )
    return f"Saved {RECOMMENDATION_COUNT} recommendations."


class RecommendationsSea(WorkerSea):
    """``/recommendations``: suggest ten next tasks from recent history and connected apps."""

    def description(self) -> str:
        """One line for ``/help`` and the slash-command menu."""
        return (
            "Reviews your recent tasks (the last 24 hours and at least 20 completed ones), "
            "checks which third-party apps you are signed in to and what you did in them, "
            "and stores ten suggested tasks, at least three combining those apps, in "
            "<home>/recommendations.json, shown in the sidebar's RECOMMENDATIONS subpanel. "
            "`/recommendations now` runs it once; a 6-hourly default cron job keeps the "
            "panel fresh."
        )

    def system_prompt(self, system_prompt: str) -> str:
        """Replace the assistant's system prompt with the suggestion procedure."""
        return SYSTEM_PROMPT

    def prompt(self, task: str) -> str:
        """Ignore *task*: every run suggests from the history, the apps and memory alone."""
        return PROMPT

    def tools(self, tools: list[Callable[..., Any]]) -> list[Callable[..., Any]]:
        """The history reader, the app probe, read-only memory, the store and ``finish``.

        ``tool_profile`` is ``none``, so *tools* holds no built-in tool.
        """
        return [
            *super().tools(tools),
            recent_tasks,
            connected_apps,
            *memory_read_tools(),
            save_recommendations,
        ]

    def settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        """A short run: no built-in tools beyond ``finish``, $2 budget."""
        return settings | {
            "tool_profile": "none",
            "max_budget": 2.0,
            "timeout": 900,
            "locked": ["tool_profile"],
        }
