# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Ask agent — answers questions about the currently-running task.

Dispatched by the ``/ask <question>`` slash command.  The command
rewriter in :mod:`kiss.agents.sorcar.sea_commands` (idle tab) and the
side channel in :mod:`kiss.server.commands` (running tab) hand this
script:

* the user's question as the sub-task prompt,
* ``prompt(question)`` = the question followed by :data:`ADD_TO_PROMPT`,
  whose ``{task_id}`` the daemon substitutes with the calling (parent)
  task's id, so the answering agent knows which task the user is
  asking about,
* ``add_to_system_prompt()`` — the no-internet directive plus the
  answering playbook.

The answering session is a two-step Q&A, modelled on Guv's "Ask about
this chat" side channel: one call of :func:`task_context` returns the
whole context (the task's status, spend and sub-agents, the tail of
its progress log on disk, and as many of its newest transcript entries
as fit 60k characters, built by :func:`kiss.agents.sorcar.task_digest.context`),
and ``finish`` carries a two-or-three-sentence answer.  An analysis of
155 earlier ``/ask`` runs showed every one of them wasting its first
steps on the missing ``sqlite3`` CLI, a schema dump and raw event JSON
(median 9 steps, $0.91 and 99 s per answer); with no other tool to
reach for, the answer comes straight from the context.

Configuration: :func:`settings` picks the ``worker`` kind with the
``none`` tool profile (no built-in tool besides ``finish`` — the parent
task is still running in the same working tree, so the answerer must
never edit files or run commands), the SYSTEM_LITE ablation prompt as
the base system prompt; :func:`prompt` appends :data:`ADD_TO_PROMPT`,
:func:`add_to_system_prompt` supplies the playbook and
:func:`add_to_tools` makes :func:`task_context` the only tool.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kiss.core.brand import render_brand

# The /ask base prompt: the SYSTEM_LITE ablation prompt with the brand
# identity as a ``{{IDENTITY}}`` placeholder (see ``kiss.core.brand``).
# It is packaged next to this file so wheel installs (which exclude
# ``papers/`` per ``pyproject.toml``) work; the frozen copy under
# ``papers/kisssorcar/ablation/prompts/`` records what the ablation
# study actually ran and is not read by the product.
_SYSTEM_LITE_PATH = Path(__file__).resolve().parent / "_ask_system_lite.md"

# The prompt suffix of every ``/ask`` dispatch (appended by ``prompt()``):
# the daemon fills ``{task_id}`` with the calling task's id, for the
# chat command and the side channel (:mod:`kiss.server.commands`) alike.
ADD_TO_PROMPT = (
    "The question above is about the task with id {task_id}. "
    "Call task_context with that task id, then answer the question."
)

_PLAYBOOK = """**MUST FOLLOW: You MUST NOT USE internet or internet search \
at any point. You must answer quickly because the user is waiting.**

## How to answer (read-only side channel of a running task)
You have exactly two tools: `task_context` and `finish`. There is no shell, \
no file access, no memory and no browser; do not look for them and never try \
to read ~/{{HOME_DIR}}/history.db yourself.
1. Call `task_context(task_id)` once with the task id named in the prompt. It \
returns everything you can know: status, elapsed time, spend, the sub-agent \
tasks, the tail of the task's own progress log, and the newest transcript \
entries (TOOL CALL, RESULT, OUTPUT, ASSISTANT, THOUGHT, SUMMARY, USER, ASK \
ANSWER, FINISH, TASK RESULT), oldest first.
2. Call `finish` with the answer as a single HTML paragraph (`<p>…</p>`).

## How to write the answer
- Two or three sentences, like a colleague who watched the task answering \
over your shoulder. Plain words, short sentences, contractions are fine.
- Say what is happening and name the concrete fact that shows it: a file \
path, a command, a number, a timestamp. Give times in UTC and local time.
- Answer only from the context. If it does not show the answer, say so in one \
sentence ("The transcript doesn't show that yet"); never guess or fill in \
from general knowledge.
- No preamble, no bullet lists, no headings, no restating the question, no \
"Based on the transcript", no hedging filler, no emoji.
- "Status: running" means the task hasn't finished; much of the work may \
happen in the sub-agent tasks listed. USER entries are messages the user \
sent later; ASK ANSWER entries are earlier /ask answers, so for a repeated \
question say what changed since then.
- If the question asks you to change code or files, you can't: say the \
instruction must be typed into the running task's chat without /ask, and \
answer what you can."""


def description() -> str:
    """Return the one-sentence help text shown by ``/ask help``."""
    return (
        "Answers a question about the currently running task in two or three plain "
        "sentences, from its status, progress log and latest transcript entries, "
        "without editing files, running commands or using the internet; "
        "type `/ask <question>` into the task's chat tab."
    )


def system_prompt() -> str:
    """Return the SYSTEM_LITE ablation prompt as the base system prompt.

    Reads the bundled ``_ask_system_lite.md`` next to this module and
    fills the brand placeholders.
    """
    return render_brand(_SYSTEM_LITE_PATH.read_text(encoding="utf-8"))


def settings() -> dict[str, Any]:
    """Configure the answering session: a worker with no built-in tools.

    ``worker``: no worktree, auto-commit, classifier, fan-out, browser or
    memory — the answer comes from ``task_context`` alone, quickly.
    ``tool_profile: "none"`` keeps even the built-in toolset out, so the
    answerer (which shares the running task's tree) cannot run commands
    or touch files; it is ``locked``, so a ``run_agent(agent="ask",
    tool_profile=...)`` that asks for tools is refused instead of
    running an answerer with them.  ``system_prompt`` is the
    SYSTEM_LITE ablation prompt (``_ask_system_lite.md`` next to this
    module, brand placeholders filled).
    """
    return {"kind": "worker", "tool_profile": "none", "locked": ["tool_profile"]}


def prompt(task: str) -> str:
    """Return the question followed by :data:`ADD_TO_PROMPT`.

    ``{task_id}`` in the result is the calling task's id, filled in by
    the daemon: it names the task the question is about.
    """
    return task + "\n\n" + ADD_TO_PROMPT


def add_to_system_prompt() -> str:
    """Return the fixed suffix appended to the answering agent's system prompt.

    The no-internet and answer-quickly directives followed by the
    answering playbook (:data:`_PLAYBOOK`): the two-call recipe
    (``task_context`` then ``finish``) and the style of the answer.
    The daemon appends it for both dispatch paths (the ``/ask`` chat
    command and the running tab's side channel), so there is exactly
    one copy of the text.  The ``{{HOME_DIR}}`` placeholder names the
    brand's state directory (``~/.kiss`` for stock KISS Sorcar).
    """
    return render_brand(_PLAYBOOK)


def task_context(task_id: str) -> str:
    """Return everything known about a Sorcar task, in one call.

    Sections: header (prompt, running/finished status, model, work
    dir, start time, elapsed time, spend, last event time, current
    UTC and local time), the sub-agent tasks it dispatched (id,
    status, steps, cost, task text), the newest part of the progress
    log the task keeps in its work dir, and its transcript entries
    oldest first — numbered TOOL CALL, RESULT, OUTPUT, ASSISTANT,
    THOUGHT, SUMMARY, USER, ASK ANSWER, FINISH and TASK RESULT lines.
    The whole text is capped at 60k characters; when the transcript
    is longer, the oldest entries are dropped and the cut is marked.

    Args:
        task_id: The ``task_history`` row id (32 hex characters) of
            the task to inspect; a sub-agent id works too.

    Returns:
        The context text, or an error line when no task has that id.
    """
    from kiss.agents.sorcar.task_digest import context

    return context(task_id)


def add_to_tools() -> list[Any]:
    """Return the only tool: :func:`task_context`.

    With the ``none`` tool profile (see :func:`settings`) the built-in
    toolset is never built: the parent task is still running in the
    same working tree, so the answerer must not run commands or touch
    files; and every extra tool schema is a temptation to take a step
    the user has to wait for.
    """
    return [task_context]


