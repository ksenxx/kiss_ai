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
* ``append_to_prompt`` = :data:`APPEND_TO_PROMPT` with ``<task_id>``
  substituted by the calling (parent) task's id, so the answering
  agent knows which task the user is asking about,
* ``append_to_system_prompt`` = :func:`append_to_system_prompt` — the
  no-internet directive plus the answering playbook.

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

Overrides: :func:`system_prompt` swaps the base system prompt for the
SYSTEM_LITE ablation prompt, :func:`append_to_system_prompt` supplies
the fixed suffix (a getter defined in this file wins over the wire
value, so it is the single source of truth for both dispatch paths),
:func:`tools` adds :func:`task_context`, :func:`if_append_basic_tools`
returns ``False`` so the session has no built-in tool besides
``finish`` (the parent task is still running in the same working tree,
so the answerer must never edit files or run commands), and
:func:`is_parallel`, :func:`use_web_tools`, :func:`use_memory` return
``False`` so the answer comes from the context alone.
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

# The prompt suffix both dispatch paths append to the user's question.
# ``<task_id>`` is a literal placeholder: the idle-tab rewrite keeps it
# (the calling task's id is only known at dispatch time and
# :func:`kiss.agents.sorcar.agent_dispatch._dispatch` substitutes it);
# the side channel formats it directly.
APPEND_TO_PROMPT = (
    "The question above is about the task with id <task_id>. "
    "Call task_context with that task id, then answer the question."
)

_PLAYBOOK = """**MUST FOLLOW: You MUST NOT USE internet or internet search \
at any point. You must answer quickly because the user is waiting.**

## How to answer (read-only side channel of a running task)
You have exactly two tools: `task_context` and `finish`. There is no shell, \
no file access, no memory and no browser; do not look for them and never try \
to read ~/.kiss/history.db yourself.
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
    fills the brand placeholders, so the same text is served from a
    source checkout and from a wheel install.
    """
    return render_brand(_SYSTEM_LITE_PATH.read_text(encoding="utf-8"))


def append_to_system_prompt() -> str:
    """Return the fixed suffix appended to the answering agent's system prompt.

    The no-internet and answer-quickly directives followed by the
    answering playbook (:data:`_PLAYBOOK`): the two-call recipe
    (``task_context`` then ``finish``) and the style of the answer.
    Both dispatch paths read the string from here, and the daemon
    applies this getter over the wire value as well, so there is
    exactly one copy of the text.
    """
    return _PLAYBOOK


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


def tools() -> list[Any]:
    """Return the single context tool, :func:`task_context`."""
    return [task_context]


def if_append_basic_tools() -> bool:
    """Never build the built-in toolset: the session has ``task_context`` and ``finish`` only.

    The parent task is still running in the same working tree, so the
    answerer must not run commands or touch files; and every extra
    tool schema is a temptation to take a step the user has to wait for.
    """
    return False


def is_parallel() -> bool:
    """Never fan out the answering session.

    A ``/ask`` invocation is a single read-only Q&A over one task's
    persisted events; a parallel run would only fragment the answer.
    """
    return False


def use_web_tools() -> bool:
    """Never enable browser tools for the answering session.

    The answer is derived solely from the task's own events in
    ``~/.kiss/history.db``; internet access would let the answering
    agent drift off the local trajectory the user is asking about.
    """
    return False


def use_memory() -> bool:
    """Never give the answering session the memory tools.

    Earlier runs spent their first steps searching memory and their
    last steps writing pages while the user waited; the answer must
    come from the context tool alone.
    """
    return False
