# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Ask agent — answers questions about the currently-running task.

Dispatched by the ``/ask <question>`` slash command.  The command
rewriter in :mod:`kiss.agents.sorcar.sea_commands` turns
``/ask <question>`` into an immediate ``run_agent`` call that hands
this script:

* the user's question as the sub-task prompt,
* ``append_to_prompt`` = ``"Read the events of the task <task_id>
  from ~/.kiss/sorcar.db and answer the user question above."`` — the
  literal ``<task_id>`` placeholder is substituted with the calling
  (parent) task's id inside :func:`_dispatch_reserved` before the
  daemon round trip, so the answering agent reads the events of the
  task the user is asking about,
* ``append_to_system_prompt`` = :func:`append_to_system_prompt` — the
  no-internet directive plus a reminder to answer quickly.

Four overrides are wired here: :func:`system_prompt` swaps the base
system prompt for the SYSTEM_LITE ablation prompt,
:func:`append_to_system_prompt` supplies the fixed suffix above (a
getter defined in this file wins over the wire value, so it is the
single source of truth for both dispatch paths), and
:func:`is_parallel` and :func:`use_web_tools` both return ``False`` so
the answering session runs as a single offline agent that only queries
``~/.kiss/sorcar.db``.
"""

from __future__ import annotations

from pathlib import Path

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
    return src.read_text(encoding="utf-8")


def append_to_system_prompt() -> str:
    """Return the fixed suffix appended to the answering agent's system prompt.

    Two directives: never touch the internet (the answer must come
    from the task's own persisted events), and answer quickly (the
    user typed ``/ask`` into a live task and is waiting on the reply).
    Both dispatch paths — the idle-tab ``run_agent`` rewrite in
    :mod:`kiss.agents.sorcar.sea_commands` and the running-tab side
    channel in :mod:`kiss.server.commands` — read the string from
    here, and the daemon applies this getter over the wire value as
    well, so there is exactly one copy of the text.
    """
    return (
        "**MUST FOLLOW: You MUST NOT USE internet or internet search "
        "at any point. You must answer quickly because the user is "
        "waiting."
    )


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
