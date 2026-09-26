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

The answering agent does not read ``~/.kiss/sorcar.db`` by hand.  An
analysis of 155 earlier ``/ask`` runs showed every one of them wasting
its first steps on the missing ``sqlite3`` CLI, a schema dump and raw
event JSON (median 9 steps, $0.91 and 99 s per answer), so the task's
trajectory is served pre-digested through three tools defined here
(:func:`task_overview`, :func:`task_transcript`, :func:`task_step`,
all thin wrappers over :mod:`kiss.agents.sorcar.task_digest`).

Overrides: :func:`system_prompt` swaps the base system prompt for the
SYSTEM_LITE ablation prompt, :func:`append_to_system_prompt` supplies
the fixed suffix (a getter defined in this file wins over the wire
value, so it is the single source of truth for both dispatch paths),
:func:`tools` adds the three trajectory tools, :func:`tool_profile`
cuts the built-in set down to the read-only ``review`` profile (the
parent task is still running in the same working tree, so the
answerer must never edit files), and :func:`is_parallel`,
:func:`use_web_tools`, :func:`use_memory` return ``False`` so the
answer comes from the trajectory alone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kiss.core.brand import render_brand

# ``src/kiss/agents/third_party_agents/ask_sea.py`` → repo root is
# ``parents[4]`` (third_party_agents → agents → kiss → src → repo).
# The authoritative SYSTEM_LITE ablation prompt lives under
# ``papers/`` at the repo root; a byte-identical copy is packaged
# next to this file as ``_ask_system_lite.md`` so wheel installs
# (which exclude ``papers/`` per ``pyproject.toml``) still work.
# The tests pin that the two files are byte-identical.
_SYSTEM_LITE_PATH = (
    Path(__file__).resolve().parents[4]
    / "papers"
    / "kisssorcar"
    / "ablation"
    / "prompts"
    / "SYSTEM_LITE.md"
)
_BUNDLED_SYSTEM_LITE_PATH = Path(__file__).resolve().parent / "_ask_system_lite.md"

# The prompt suffix both dispatch paths append to the user's question.
# ``<task_id>`` is a literal placeholder: the idle-tab rewrite keeps it
# (the calling task's id is only known at dispatch time and
# :func:`kiss.agents.sorcar.agent_dispatch._dispatch` substitutes it);
# the side channel formats it directly.
APPEND_TO_PROMPT = (
    "The question above is about the task with id <task_id>. "
    "Call task_overview with that task id first, then answer the question."
)

_PLAYBOOK = """**MUST FOLLOW: You MUST NOT USE internet or internet search \
at any point. You must answer quickly because the user is waiting.**

## How to answer (read-only side channel of a running task)
The trajectory of the task named in the prompt is already digested for you. \
Never read ~/.kiss/sorcar.db by hand: the `sqlite3` CLI is not installed, the \
schema is irrelevant, and raw event JSON is 10-100x larger than the digest.
1. Call `task_overview(task_id)` FIRST, in one call: status, spend, sub-agent \
tasks, the user's later messages, previous /ask answers, the task's own \
progress summaries and its latest steps.
2. Only if the question needs more: `task_transcript(task_id, start, count, \
contains)` pages or filters the whole transcript by literal terms (e.g. \
contains="paper.tex|pytest" matches entries containing either); \
`task_step(task_id, index)` returns one entry in full (a complete command \
output, diff or tool argument). The same tools work on a sub-agent task id.
3. For live state the transcript cannot show (result files, background jobs, \
`git diff` in the task's work dir), run ONE Bash command with a short timeout, \
reusing the exact status command or path the task itself used. Read only; \
the task is still running in that directory.
4. `finish` with the answer in HTML. Aim for at most 4 tool calls in total.

## Facts and pitfalls
- Indices in [brackets] are transcript entry numbers shared by all three tools.
- "Status: running" means the task has not finished. Much of the work happens \
inside sub-agent tasks (run_parallel/run_agent); the overview lists them, pass \
a child id to the same tools to inspect one.
- USER entries are messages the user sent to the task after it started; ASK \
ANSWER entries are earlier /ask answers. For a repeated status question report \
what changed since the previous answer and state the measurement time.
- Give times in UTC and in the local time printed by the overview; derive \
rates and ETAs from transcript timestamps, never guess.
- If the question asks you to change code or files, do not: /ask is \
read-only. Say the instruction must be typed into the running task's chat \
without /ask, and answer what you can.
- Do not use memory tools, do not write notes or files, do not narrate the \
whole trajectory; answer the question asked and cite entry indices or file \
paths for the key facts."""


def system_prompt() -> str:
    """Return the SYSTEM_LITE ablation prompt as the base system prompt.

    Prefers the repo copy at
    ``./papers/kisssorcar/ablation/prompts/SYSTEM_LITE.md`` so an
    ablation-time edit is picked up immediately; falls back to the
    bundled copy (``_ask_system_lite.md``, kept byte-identical by
    ``test_bundled_system_lite_is_byte_identical``) so a wheel
    install without the ``papers/`` tree still works.
    """
    src = _SYSTEM_LITE_PATH if _SYSTEM_LITE_PATH.is_file() else _BUNDLED_SYSTEM_LITE_PATH
    return render_brand(src.read_text(encoding="utf-8"))


def append_to_system_prompt() -> str:
    """Return the fixed suffix appended to the answering agent's system prompt.

    The no-internet and answer-quickly directives followed by the
    answering playbook (:data:`_PLAYBOOK`): the tool order that gets
    to an answer in the fewest steps and the pitfalls seen in earlier
    ``/ask`` runs.  Both dispatch paths read the string from here, and
    the daemon applies this getter over the wire value as well, so
    there is exactly one copy of the text.
    """
    return _PLAYBOOK


def task_overview(task_id: str) -> str:
    """Return the one-call orientation digest of a Sorcar task.

    Sections: header (prompt, running/finished status, model, work
    dir, start time, elapsed time, spend, last event time, current
    UTC and local time), sub-agent tasks it dispatched (id, status,
    steps, cost, task text), messages the user sent after the initial
    prompt, the latest /ask answers, every progress summary the task
    wrote about itself, and its last 30 transcript entries.

    Args:
        task_id: The ``task_history`` row id (32 hex characters) of
            the task to inspect; a sub-agent id works too.

    Returns:
        The digest text, or an error line when no task has that id.
    """
    from kiss.agents.sorcar.task_digest import overview

    return overview(task_id)


def task_transcript(task_id: str, start: int = 0, count: int = 150, contains: str = "") -> str:
    """Return a page of a task's digested transcript, optionally filtered.

    Entries are numbered from 0 in transcript order: TOOL CALL (name
    and arguments), RESULT (tool result), OUTPUT (shell output),
    ASSISTANT (the agent's prose), THOUGHT, SUMMARY (the task's own
    progress log), USER (a later user message), ASK ANSWER, FINISH and
    TASK RESULT.  Long texts are clipped; ``task_step`` returns one
    entry in full.

    Args:
        task_id: The ``task_history`` row id (32 hex characters).
        start: Index of the first entry to consider (0-based).
        count: Maximum number of entries to return (at most 400).
        contains: Optional filter: literal search terms separated by
            ``|`` (not a regex); only entries whose full text contains
            at least one term, case-insensitively, are returned with
            their indices (e.g. ``"paper.tex"``, ``"pytest|FAILED"``).

    Returns:
        The header, the numbered entries and how many more remain; or
        an error line when the task id is unknown.
    """
    from kiss.agents.sorcar.task_digest import transcript_page

    return transcript_page(task_id, start, count, contains)


def task_step(task_id: str, index: int, max_chars: int = 20000) -> str:
    """Return one transcript entry of a task in full.

    Use it to read a complete command output, diff, file content or
    tool argument that the overview or transcript page clipped.

    Args:
        task_id: The ``task_history`` row id (32 hex characters).
        index: The entry number shown in [brackets] by the other tools.
        max_chars: Clip length for the entry text (at most 100000).

    Returns:
        ``[index] KIND: full text``, or an error line when the task or
        index does not exist.
    """
    from kiss.agents.sorcar.task_digest import entry_detail

    return entry_detail(task_id, index, max_chars)


def tools() -> list[Any]:
    """Return the trajectory tools: task_overview, task_transcript and task_step."""
    return [task_overview, task_transcript, task_step]


def tool_profile() -> str:
    """Return the built-in tool profile: ``"review"`` (inspect and run, never edit)."""
    return "review"


def is_parallel() -> bool:
    """Never fan out the answering session.

    A ``/ask`` invocation is a single read-only Q&A over one task's
    persisted events; a parallel run would only fragment the answer.
    """
    return False


def use_web_tools() -> bool:
    """Never enable browser tools for the answering session.

    The answer is derived solely from the task's own events in
    ``~/.kiss/sorcar.db``; internet access would let the answering
    agent drift off the local trajectory the user is asking about.
    """
    return False


def use_memory() -> bool:
    """Never give the answering session the memory tools.

    Earlier runs spent their first steps searching memory and their
    last steps writing pages while the user waited; the answer must
    come from the trajectory tools alone.
    """
    return False
