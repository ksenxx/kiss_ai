# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The classes every Sorcar Extension Agent (SEA) derives from.

A SEA is a Python file ``xxx/xxx_sea.py`` that defines exactly one
subclass of :class:`BaseSea`.  The subclass overrides the methods it
needs; every method here passes its input through (the root layer's
one rule of its own is the ``summary`` cadence, see :class:`BaseSea`),
so a SEA defines only what it changes::

    from kiss.agents.seas.base.base_sea import WorkerSea

    class ShellSea(WorkerSea):
        def description(self):
            return "Runs the command in the prompt."

        def settings(self, settings):
            return settings | {"tool_profile": "bash"}

        def prompt(self, task):
            return f"Run `{task}` and report its output."

        def system_prompt(self, system_prompt):
            return system_prompt + "\\n\\nAnswer in one line."

What a SEA *is* is its base class: :class:`BaseSea` is an ordinary
Sorcar session with the caller's or the user's settings;
:class:`WorkerSea` a focused tool-bound run on the caller's tree as it
is (no worktree tools, no classifier, no browser, no memory);
:class:`ChannelSea` a worker that serves an external service (Slack,
email, cron) from a scratch directory, inherits nothing from the
calling task and holds a channel workspace (every behaviour is listed
in :data:`kiss.agents.sorcar.sea_settings.CHANNEL_BEHAVIOURS`).

The launcher (:mod:`kiss.agents.sorcar.sea_commands`) instantiates the
class and threads the run's configuration through the methods of
every class of its inheritance chain, base first, each receiving
what the previous one returned: ``settings`` starts from ``{}``,
``prompt`` from the task text, ``system_prompt`` from the run's
assembled system prompt, ``tools`` from the run's built-in toolset,
``llm_call_hook`` from the messages of the LLM call, and
``tool_call_hook`` stops at the first refusing verdict (``refuse(text)``;
``ALLOW`` allows; ``end_run(result)`` ends the run — all three
importable from this module), as ``llm_response_hook`` stops at the
first ``end_run``.  The chaining is the launcher's job: a method must NOT call
``super()`` (the base's method runs anyway), and it must not
expect to be called in a particular order relative to another.

A SEA extends another by plain Python inheritance: import the base
class (``from kiss.agents.seas.bestrouter.bestrouter_sea import
BestrouterSea``) or load it by command name or path with
:func:`kiss.agents.sorcar.sea_commands.sea_class`, and derive from it.
A class may define further helper methods; only the ones below are
part of the contract.

:class:`BaseSea` is itself the root layer of EVERY run made from the
chat page: a plain prompt runs the bare
``BaseSea``, a ``/xxx`` command or a model-picker tab runs its SEA on
top of it, and the sub-agents of ``run_agent`` / ``run_parallel`` go
through it the same way.  So editing the methods of this file
customizes every such run at once — append a house rule in
``system_prompt``, add a tool in ``tools``, refuse a tool call in
``tool_call_hook``, pin a setting in ``settings`` (subject to
:data:`kiss.agents.sorcar.sea_settings.PRECEDENCE_RULE`: a value the
caller passed explicitly ranks above it).  The daemon
imports this module once, so restart it after editing.  A rule
``system_prompt`` appends is stated once per run: a sub-agent's prompt
is built by its own layers, not copied from its parent's.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from kiss.core.config import kiss_home
from kiss.core.llm_reply import LLMReply
from kiss.core.tool_schema import tool_schema
from kiss.core.tool_verdict import ALLOW, Verdict, allow_with_note, end_run, refuse

__all__ = [
    "ALLOW",
    "BaseSea",
    "ChannelSea",
    "LLMReply",
    "Verdict",
    "WorkerSea",
    "allow_with_note",
    "end_run",
    "refuse",
    "tool_schema",
]

WORKER_DEFAULTS: dict[str, Any] = {
    "auto_classify": False,
    "use_web_tools": False,
    "use_memory": False,
}
"""The settings :class:`WorkerSea` lays under its subclass's own."""

WORKTREE_TOOL_NAMES: frozenset[str] = frozenset({"create_worktree", "commit_and_merge"})
"""The tools :class:`WorkerSea` drops: a worker edits the caller's tree in place."""

SUMMARY_EVERY_STEPS = 10
"""A run with the ``summary`` tool must call it at every step that is a multiple of this."""

SUMMARY_DUE_NOTE = (
    "\n\nA summary is due (one is owed at every step that is a multiple of {every}): make your "
    "next tool call summary(description=...), recapping your steps since the last summary."
)
"""Appended to the result of every tool call but ``summary`` and ``finish`` while a summary is due.

The call runs as asked (a refusal cost a whole model step to re-issue
it, and the 2026-11-01 audit counted 246 of them in one task): the note
is what the model reads at the very step it chooses what to do next.
"""


def channel_work_dir() -> str:
    """Return the shared scratch directory a channel runs in: ``<home>/channel_work``.

    Computed on every call so a redirected ``$KISS_HOME`` is honoured.
    """
    return str(kiss_home() / "channel_work")


class BaseSea:
    """The SEA contract and the root layer of every run; each method returns its input unchanged.

    The one rule the root layer enforces is the ``summary`` cadence: a
    run whose toolset holds the ``summary`` tool must call it at every
    :data:`SUMMARY_EVERY_STEPS`-th step (the system prompt's "Periodic
    Activity Summaries" rule).  ``tools`` notes whether the tool is
    there, ``llm_response_hook`` reads the step of every reply (the
    ``Steps:`` count of the tool-result footer, which the rule is
    stated in), and ``tool_call_hook`` lets every other tool call run
    but ends its result with :data:`SUMMARY_DUE_NOTE` from the 10th,
    20th, ... step on until ``summary`` runs (``finish`` excepted).
    """

    path: Path | None = None
    """The file the SEA was loaded from (the launcher sets it; else the class's module file)."""

    step = 0
    """The run's step so far: the ``Steps:`` count of the footer, read from each reply.

    A continuation session restarts the footer's count at 1, as it
    restarts the model's context, so the cadence restarts with it.
    """

    has_summary_tool = False
    """Whether the run's toolset holds the ``summary`` tool (set by ``tools``)."""

    summary_due = False
    """Whether a ``summary`` call is owed: a 10th step began and none has run since."""

    def __init__(self) -> None:
        module = sys.modules.get(type(self).__module__)
        file = getattr(module, "__file__", None)
        if file:
            self.path = Path(file)

    def description(self) -> str:
        """Return one sentence saying what the SEA does and how to use it (``/xxx help``).

        A SEA registered as a slash command must return a non-empty
        string; the default is empty.
        """
        return ""

    def settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        """Return the SEA's settings, laid over *settings* (what the base classes declared).

        The keys are those of
        :data:`kiss.agents.sorcar.sea_settings.SETTING_TYPES`: per-run
        parameters (``model``, ``tool_profile``, ``max_budget``, ...),
        ``timeout``, ``locked`` and ``hidden``.  The launcher
        type-checks every value; a relative ``work_dir`` is a path
        under the calling task's directory.  What the run *is* (a
        session, a worker, a channel) is the base class, not a key.
        Return *settings* with the SEA's keys added (``settings |
        {...}``); a value of ``None`` means "no override".
        """
        return settings

    def prompt(self, task: str) -> str:
        """Return the prompt the run gets for *task* (the user's text, or a base's result).

        Every ``{task_id}`` of the final text (yours or the task's own)
        is replaced by the calling task's id, ``""`` without one.  The
        result must be a non-empty string.
        """
        return task

    def system_prompt(self, system_prompt: str) -> str:
        """Return the run's system prompt given the assembled *system_prompt*.

        Return *system_prompt* with text appended to add rules, or a
        different string to replace the whole prompt; the returned
        string is the run's system prompt, as ``prompt``'s is the
        run's prompt.  Called once per run, after the launcher has
        assembled the base prompt and the caller's additions.
        """
        return system_prompt

    def tools(self, tools: list[Callable[..., Any]]) -> list[Callable[..., Any]]:
        """Return the run's tool callables given the *tools* the run has so far.

        *tools* are the built-in tools of the run's tool profile plus
        those the caller passed on; return ``tools + [...]`` to add
        tools, a new list to fix the whole toolset (with
        ``tool_profile: "none"`` the list starts empty).  A tool is a
        function with a docstring; its ``__name__`` is the name the
        model calls.

        The root layer arms the summary cadence guardrail here: the
        toolset is built once per run (also for each ``<task>`` block
        of a multi-task prompt, which the daemon runs on the same
        hooks), so the step count and any owed summary start afresh,
        and ``has_summary_tool`` records whether the built-in toolset
        holds ``summary``.  The root runs first in the chain, so a SEA
        whose own ``tools`` drops ``summary`` should set
        ``self.has_summary_tool = False`` too (every bundled SEA only
        appends tools).
        """
        self.step = 0
        self.summary_due = False
        self.has_summary_tool = any(getattr(t, "__name__", "") == "summary" for t in tools)
        return tools

    def tool_call_hook(self, name: str, args: dict[str, Any]) -> Verdict:
        """Return ``ALLOW`` to let the tool call *name*(*args*) run, else ``refuse(text)``.

        Called before every tool call of the run; the verdict is a
        :class:`~kiss.core.tool_verdict.Verdict` (import ``ALLOW``,
        ``refuse`` and ``end_run`` from this module).  A refusal's text
        is returned to the model as the tool's result.  ``end_run(result,
        success=...)`` ends the run instead: the call does not run, no
        later call of the same turn runs, and *result* becomes the
        run's result (the ``summary`` of Sorcar's ``finish`` contract)
        — for a rule of the SEA's own that says the task is over (the
        report is accepted, the episode's turn budget is spent) without
        waiting for the model to call ``finish``.  The agent also probes
        this hook with ``("finish", {})`` before an implicit finish of a
        model that only talks.

        The root layer's own rule: while a ``summary`` is due (see
        :class:`BaseSea`), every call but ``summary`` and ``finish`` runs
        with :data:`SUMMARY_DUE_NOTE` appended to its result
        (:func:`~kiss.core.tool_verdict.allow_with_note`); the
        ``summary`` call clears the debt.  ``finish`` is exempt because
        its own summary ends the run (and the agent probes it for an
        implicit finish).
        """
        if name == "summary":
            self.summary_due = False
        elif self.summary_due and name != "finish":
            return allow_with_note(SUMMARY_DUE_NOTE.format(every=SUMMARY_EVERY_STEPS))
        return ALLOW

    def llm_call_hook(self, new_messages: list[Any]) -> list[Any]:
        """Return the messages to send given the *new_messages* of the next LLM call.

        The root layer passes them on unchanged.
        """
        return new_messages

    def llm_response_hook(self, reply: LLMReply) -> Verdict:
        """Return ``ALLOW`` to act on the model's *reply*, or ``end_run(result)`` to end the run.

        Called after every model reply of the run, before any tool call
        of the reply runs, with an :class:`~kiss.core.llm_reply.LLMReply`
        (import it from this module): the reply's text, its tool calls
        (``{"name", "arguments"}`` each), its token usage and cost
        (``reply.usage["output_tokens"]``, ...) and the run's totals so
        far.  ``end_run(result, success=...)`` ends the run with
        *result*; none of the reply's tool calls run.  Together with
        ``llm_call_hook``, which rewrites the next request, this is
        every exchange with the model: what a domain rule needs to cap
        the tokens of an episode or to end it on a reply that settles
        the task.  A tool Codex ran natively during the turn has run
        already.

        The root layer allows every reply and reads its ``step`` (the
        footer's count after the call): at every
        :data:`SUMMARY_EVERY_STEPS`-th step of a run that has the
        ``summary`` tool a summary comes due.  A step that does not grow
        is a continuation session restarting the count: a debt of the
        previous session, whose context is gone, is dropped with it.
        """
        if reply.step <= self.step:
            self.summary_due = False
        self.step = reply.step
        if self.has_summary_tool and self.step % SUMMARY_EVERY_STEPS == 0:
            self.summary_due = True
        return ALLOW

    def register_as_model(self) -> bool:
        """Return whether the SEA is offered in the model picker next to the real models."""
        return False

    def on_picked_as_model(self, work_dir: str) -> str:
        """Act on being picked as a tab's model (``register_as_model`` SEAs); return a log note.

        Args:
            work_dir: The tab's or run's work directory.
        """
        return ""


class WorkerSea(BaseSea):
    """A focused tool-bound run on the caller's tree.

    Lays :data:`~kiss.agents.sorcar.sea_settings.WORKER_DEFAULTS` under
    the subclass's own settings: no classifier, no browser, no memory
    (``/sh``, ``/ask``, ``/merge``, ``/remember``, ``/forget``,
    ``/task_update``).  A subclass that writes one of those keys
    overrides the default, as with any base.  A worker edits the
    caller's tree as it is: :meth:`tools` drops the ``create_worktree``
    and ``commit_and_merge`` tools, so the run never moves into a git
    worktree of its own (``/merge`` resolves conflicts in the checkout
    that holds them).
    """

    def settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        """Return *settings* with the worker defaults laid under the subclass's keys."""
        return settings | WORKER_DEFAULTS

    def tools(self, tools: list[Callable[..., Any]]) -> list[Callable[..., Any]]:
        """Return *tools* without the worktree tools (the worker edits the tree in place)."""
        return [
            tool for tool in tools
            if getattr(tool, "__name__", "") not in WORKTREE_TOOL_NAMES
        ]


class ChannelSea(WorkerSea):
    """A worker that serves an external service (Slack, email, cron), not the caller's project.

    Runs in the shared ``<home>/channel_work`` scratch directory unless
    its own ``work_dir`` says otherwise, takes nothing from the calling
    task, holds a channel workspace, gets the channel preamble, and
    has ``work_dir`` and the worker keys locked.  The loader reads the
    class (``isinstance(sea, ChannelSea)``), and the command registry
    reads the base name from the source, so a channel must derive
    from this class by name — see
    :data:`kiss.agents.sorcar.sea_settings.CHANNEL_BEHAVIOURS`.
    """

    def settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        """Return *settings* with the channel scratch directory as the default ``work_dir``."""
        return settings | {"work_dir": channel_work_dir()}
