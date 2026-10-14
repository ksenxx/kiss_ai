# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``/recommendations``: five suggested next tasks from the last day's work.

The agent reads the tasks the user ran in the last
:data:`RECENT_HOURS` hours (``recent_tasks``), picks
:data:`RECOMMENDATION_COUNT` concrete follow-ups and stores them with
``save_recommendations`` in :func:`recommendations_file`
(``<home>/recommendations.json``).  The right sidebar's RECOMMENDATIONS
subpanel shows that file (:func:`load_recommendations`); a 6-hourly cron
prompt job ``/recommendations now`` keeps it fresh.
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
from kiss.core.config import kiss_home
from kiss.core.utils import atomic_write_text

RECENT_HOURS = 24
"""How far back ``recent_tasks`` looks."""

RECOMMENDATION_COUNT = 5
"""How many suggestions one run stores."""

MAX_TASKS = 100
"""Most tasks ``recent_tasks`` lists (the newest), after the automations are dropped."""

MAX_TASK_CHARS = 300
MAX_RESULT_CHARS = 200

OWN_SEA_PREFIXES = ("recommendations", "cron_prompt")
"""``task_history.sea`` prefixes of automations, not of the user's own tasks."""

PROMPT = f"Suggest {RECOMMENDATION_COUNT} tasks from the last {RECENT_HOURS} hours."
"""Every run's task text (a run outside the daemon has no ``sea`` column to drop it by)."""

SYSTEM_PROMPT = (
    "You suggest the user's next tasks for their AI assistant.\n\n"
    f"1. Call `recent_tasks()` once. It lists the tasks the user ran in the last "
    f"{RECENT_HOURS} hours, newest first, with directory and outcome.\n"
    f"2. Choose {RECOMMENDATION_COUNT} suggestions grounded in that list: finish what "
    "failed or was stopped, the next step after a success, a check the changed area "
    "now needs, a repeated chore that could be automated. Each must name the file, "
    "feature, repository or topic it comes from. No generic advice, no repeat of a "
    "task that already succeeded, no two suggestions about the same thing.\n"
    "3. Write each suggestion as one imperative sentence of at most 160 characters, "
    "ready to submit as a prompt.\n"
    f"4. Call `save_recommendations(tasks)` with the {RECOMMENDATION_COUNT} sentences, "
    "then `finish` with success=true and the same sentences as an HTML list.\n\n"
    f"If the list is empty, save {RECOMMENDATION_COUNT} first tasks for a new software "
    "project (summarise the README, run the test suite, list the open TODOs, ...). "
    "Never ask the user anything."
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
    """List the tasks the user ran in the last 24 hours, newest first.

    One line per task: how long ago it started, the directory it ran
    in, its outcome (ok / failed / unfinished), the request and the
    first line of the result.  Sub-agents, side channels, cron runs and
    earlier runs of this agent are left out; at most the newest
    :data:`MAX_TASKS` remaining tasks are listed.

    Returns:
        The lines, or ``"No tasks in the last 24 hours."``.
    """
    now = time.time()
    rows = persistence.user_tasks_since(now - RECENT_HOURS * 3600)
    lines: list[str] = []
    for row in rows:
        if _is_automation(row):
            continue
        if len(lines) == MAX_TASKS:
            break
        success = row["success"]
        outcome = "unfinished" if success is None else ("ok" if success else "failed")
        age = _age(now - float(row["timestamp"]))  # type: ignore[arg-type]
        task = _plain(str(row["task"]), MAX_TASK_CHARS)
        line = f"- {age} ago | {row['workDir']} | {outcome} | {task}"
        # An unfinished row holds the "Agent Failed Abruptly" placeholder, not a result.
        result = "" if success is None else _plain(str(row["result"]), MAX_RESULT_CHARS)
        if result:
            line += f" -> {result}"
        lines.append(line)
    if not lines:
        return f"No tasks in the last {RECENT_HOURS} hours."
    return "\n".join(lines)


def save_recommendations(tasks: list[str]) -> str:
    """Store the suggested tasks for the sidebar's RECOMMENDATIONS subpanel.

    Args:
        tasks: Exactly five non-empty one-sentence task prompts.

    Returns:
        ``"Saved 5 recommendations."`` or an error describing what to fix.
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
    """``/recommendations``: suggest five next tasks from the last day's history."""

    def description(self) -> str:
        """One line for ``/help`` and the slash-command menu."""
        return (
            "Reads the tasks you ran in the last 24 hours and stores five suggested next "
            "tasks in <home>/recommendations.json, shown in the sidebar's RECOMMENDATIONS "
            "subpanel. `/recommendations now` runs it once; a 6-hourly default cron job "
            "keeps the panel fresh."
        )

    def system_prompt(self, system_prompt: str) -> str:
        """Replace the assistant's system prompt with the suggestion procedure."""
        return SYSTEM_PROMPT

    def prompt(self, task: str) -> str:
        """Ignore *task*: every run suggests from the history alone."""
        return PROMPT

    def tools(self, tools: list[Callable[..., Any]]) -> list[Callable[..., Any]]:
        """Only the history reader, the store and ``finish`` (``tool_profile`` is ``none``)."""
        return [*super().tools(tools), recent_tasks, save_recommendations]

    def settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        """A short, tool-free run: no built-in tools beyond ``finish``, $1 budget."""
        return settings | {
            "tool_profile": "none",
            "max_budget": 1.0,
            "timeout": 600,
            "locked": ["tool_profile"],
        }
