# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Task-update agent — reports what a running task has done so far.

``/task_update <task_id>`` in the chat runs it as a sub-task through
``run_agent`` (see :mod:`kiss.agents.sorcar.sea_commands`).  (The
task-info panel's periodic task update is the ``/ask`` agent's answer,
see :mod:`kiss.server.task_update`; :data:`PROMPT_TEMPLATE` still
identifies the rows of the releases in which this agent produced it.)

The agent reads the task's persisted transcript (``$KISS_HOME/history.db``)
through the :func:`task_transcript` tool defined here (a thin wrapper
over :func:`kiss.agents.sorcar.task_digest.transcript_page`) and
answers :data:`PROMPT_TEMPLATE` with a short markdown progress report.
"""

from __future__ import annotations

from typing import Any

from kiss.agents.seas.base.base_sea import WorkerSea
from kiss.core.brand import HOME_DIR

PROMPT_TEMPLATE = (
    "What have the task with {task_id} done so far and what are the partial results?"
)

SYSTEM_PROMPT = """You report the progress of another Sorcar task.

The user's prompt names a task id (a 32-character hex string). If the prompt
is nothing but a task id, treat it as: "What have the task with <task_id>
done so far and what are the partial results?".

Procedure:
1. Call `task_transcript(task_id)` to read the task's persisted transcript
   (its prompt, tool calls, tool results, periodic `SUMMARY` entries, spend). It
   returns entries in pages of 150. When the transcript has up to 300 entries,
   read the second page with `start=150`; when it is longer, read only the last
   page next (`start = total - 150`) and say which entry range you skipped.
2. Call `finish` with `success=true` and, as `summary_in_html`, a concise
   progress report (under 400 words) in compact HTML (`<h4>` headings,
   `<ul><li>` bullets, `<p>`; no `<html>`/`<body>` wrapper) with these
   sections, each a short bullet list:
   - **Goal**: the task's request in one or two sentences.
   - **Done so far**: the concrete steps taken, in order (files read or
     changed, commands run, findings). Prefer the `SUMMARY` entries the task
     wrote itself; use the raw entries for the period after the last summary.
   - **Partial results**: results already produced (numbers, files, answers,
     decisions), or "None yet" when nothing is produced.
   - **Current activity**: what the task is doing in its latest entries.
   - **Spend**: steps, tokens and cost so far, from the transcript header.

Rules: report only what the transcript shows; never guess or embellish. Do
not quote long transcript excerpts. If the task id is missing or unknown,
finish with a one-line message saying so. If the task has already finished,
say so and report its final result.
""" """\


## Lessons from recent runs (rsi7d)

- `task_transcript` is your tool even though the tool profile lists only Bash. Never use
  Bash here: do not query the task database, tail log files, run status scripts, `cd` into
  other checkouts or `sleep`. The transcript is the only source; if it lacks something,
  say so in the report.
- Keep `count` at the default 150 or lower and call `task_transcript` at most twice
  (first page, then the last page for a long transcript, as step 1 says). Never ask for
  `count=400`: one such page exceeds the run's $1 budget before `finish`.
"""


class TaskUpdateSea(WorkerSea):
    """The ``/task_update`` SEA."""

    def description(self) -> str:
        """Return the one-sentence help text shown by ``/task_update help``."""
        return (
            "Reports what a running or finished Sorcar task has done so far and its partial "
            f"results by reading its persisted transcript from ~/{HOME_DIR}/history.db; use it as "
            '`/task_update <task_id>` in the chat or `run_agent(agent="task_update", '
            'task="<task_id>")`.'
        )

    def system_prompt(self, system_prompt: str) -> str:
        """Return the agent's base system prompt (:data:`SYSTEM_PROMPT`)."""
        return SYSTEM_PROMPT

    def settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        """A $1 worker with Bash only, running :data:`SYSTEM_PROMPT`."""
        return settings | {
            "tool_profile": "bash",
            "max_budget": 1.0,
        }

    def tools(self, tools: list[Any]) -> list[Any]:
        """Return the agent's tools: :func:`task_transcript`."""
        return tools + [task_transcript]


def build_prompt(task_id: str) -> str:
    """Return the agent's prompt for *task_id* (:data:`PROMPT_TEMPLATE`).

    Args:
        task_id: The ``task_history`` row id of the task to report on.

    Returns:
        The filled-in prompt text.
    """
    return PROMPT_TEMPLATE.format(task_id=task_id)


def task_transcript(task_id: str, start: int = 0, count: int = 150) -> str:
    """Return a page of the digested transcript of a Sorcar task.

    The digest is built from the task's persisted events: every tool
    call and clipped tool result, the task's own periodic ``SUMMARY``
    entries, coalesced assistant text, thoughts and shell output,
    messages the user sent while it ran, and the final result when
    the task has finished.  A header gives the task's prompt, status,
    model, work dir, start time, elapsed time and spend.

    Args:
        task_id: The ``task_history`` row id (32 hex characters).
        start: Index of the first digest entry to return (0-based).
        count: Number of entries to return (at most 400).

    Returns:
        The header followed by the numbered entries ``start`` to
        ``start + count - 1``, and a line telling how many entries
        remain; or an error line when no task has that id.
    """
    from kiss.agents.sorcar.task_digest import transcript_page

    return transcript_page(task_id, start, count)


