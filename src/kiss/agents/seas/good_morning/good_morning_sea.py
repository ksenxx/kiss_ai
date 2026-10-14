# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``/good_morning``: greet the user, show the stored recommendations, ask what to do.

The morning default cron job (:data:`kiss.agents.sorcar.cron_agent.DEFAULT_JOBS`,
``in_chat``) runs it in a new chat every surface opens.  The
recommendations shown are the ones the 6-hourly ``/recommendations``
run stored (:func:`kiss.agents.seas.recommendations.recommendations_sea.load_recommendations`);
nothing is generated on the fly.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from kiss.agents.seas.base.base_sea import WorkerSea
from kiss.agents.seas.recommendations.recommendations_sea import load_recommendations

SYSTEM_PROMPT = (
    "You open the user's day in a fresh chat.\n\n"
    "1. Call `stored_recommendations()` once. It returns the tasks the last scheduled "
    "/recommendations run suggested, and when. Never invent, change or regenerate them.\n"
    "2. Call `finish` with success=true and this HTML: one short greeting; the stored "
    "suggestions as an ordered list, word for word, with when they were suggested; one "
    "closing question asking which of them, or what else, the user wants done today. "
    "Nothing else.\n\n"
    "If none are stored, greet, say the first suggestions arrive after the next scheduled "
    "/recommendations run, and ask what to do today."
)

PROMPT = "Good morning: greet me, show the stored recommendations and ask what I want done."


def stored_recommendations() -> str:
    """Return the suggestions the last scheduled ``/recommendations`` run stored.

    Returns:
        One numbered line per suggestion preceded by how long ago they
        were stored, or ``"No recommendations stored yet."``.
    """
    stored = load_recommendations()
    if not stored["tasks"]:
        return "No recommendations stored yet."
    minutes = int((time.time() - float(stored["generatedAt"])) // 60)
    amount, unit = (minutes, "minute") if minutes < 60 else (minutes // 60, "hour")
    lines = [f"Suggested {amount} {unit}{'' if amount == 1 else 's'} ago:"]
    lines += [f"{i}. {task}" for i, task in enumerate(stored["tasks"], 1)]
    return "\n".join(lines)


class GoodMorningSea(WorkerSea):
    """``/good_morning``: greet, list the stored recommendations, ask what to do."""

    def description(self) -> str:
        """One line for ``/help`` and the slash-command menu."""
        return (
            "Greets you, lists the recommendations the last scheduled /recommendations run "
            "stored (never generated on the fly) and asks what you want done today; the "
            "default morning cron job runs `/good_morning now` in a new chat."
        )

    def system_prompt(self, system_prompt: str) -> str:
        """Replace the assistant's system prompt with the greeting procedure."""
        return SYSTEM_PROMPT

    def prompt(self, task: str) -> str:
        """Ignore *task*: every run greets the same way."""
        return PROMPT

    def tools(self, tools: list[Callable[..., Any]]) -> list[Callable[..., Any]]:
        """Only the stored-recommendations reader and ``finish`` (``tool_profile`` is ``none``)."""
        return [*super().tools(tools), stored_recommendations]

    def settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        """A one-step run: no built-in tools beyond ``finish``, 50 cents."""
        return settings | {
            "tool_profile": "none",
            "max_budget": 0.5,
            "timeout": 300,
            "locked": ["tool_profile"],
        }
