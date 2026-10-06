# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The class every Sorcar Extension Agent (SEA) derives from.

A SEA is a Python file ``xxx/xxx_sea.py`` that defines exactly one
subclass of :class:`BaseSea`.  The subclass overrides the methods it
needs; every method has a do-nothing default here, so a SEA defines
only what it changes::

    from kiss.agents.seas.base.base_sea import BaseSea

    class ShellSea(BaseSea):
        def description(self):
            return "Runs the command in the prompt."

        def settings(self, settings):
            return settings | {"kind": "worker", "tool_profile": "bash"}

        def prompt(self, task):
            return f"Run `{task}` and report its output."

        def system_prompt(self, system_prompt):
            return system_prompt + "\\n\\nAnswer in one line."

The launcher (:mod:`kiss.agents.sorcar.sea_commands`) instantiates the
class and threads the run's configuration through the methods of
every class of its inheritance chain, base first, each receiving
what the previous one returned: ``settings`` starts from ``{}``,
``prompt`` from the task text, ``system_prompt`` from the run's
assembled system prompt, ``tools`` from the run's built-in toolset,
``llm_call_hook`` from the messages of the LLM call, and
``tool_call_hook`` stops at the first verdict other than ``"OK"``.
The chaining is the launcher's job: a method must NOT call
``super()`` (the base's method runs anyway), and it must not
expect to be called in a particular order relative to another.

A SEA extends another by plain Python inheritance: import the base
class (``from kiss.agents.seas.bestrouter.bestrouter_sea import
BestrouterSea``) or load it by command name or path with
:func:`kiss.agents.sorcar.sea_commands.sea_class`, and derive from it.
A class may define further helper methods; only the ones below are
part of the contract.

:class:`BaseSea` is itself the root layer of EVERY run made from the
chat (webapp or VS Code extension): a plain prompt runs the bare
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


class BaseSea:
    """The SEA contract and the root layer of every run; each method returns its input unchanged."""

    path: Path | None = None
    """The file the SEA was loaded from (the launcher sets it; else the class's module file)."""

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
        :data:`kiss.agents.sorcar.sea_settings.SETTING_TYPES`: a
        ``kind`` (``session``, ``worker`` or ``channel``), per-run
        parameters (``model``, ``tool_profile``, ``max_budget``, ...),
        ``timeout``, ``locked`` and ``hidden``.  The launcher lays the
        kind's defaults under the result and type-checks every value.
        Return *settings* with the SEA's keys added (``settings |
        {...}``); a value of ``None`` means "no override".
        """
        return settings

    def prompt(self, task: str) -> str:
        """Return the prompt the run gets for *task* (the user's text, or a base's result).

        ``{task_id}`` in the result is replaced by the calling task's
        id.  The result must be a non-empty string.
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
        """
        return tools

    def tool_call_hook(self, name: str, args: dict[str, Any]) -> str:
        """Return ``"OK"`` to let the tool call *name*(*args*) run, else the text to refuse it with.

        Called before every tool call of the run; a refusal is returned
        to the model as the tool's result.
        """
        return "OK"

    def llm_call_hook(self, new_messages: list[Any]) -> list[Any]:
        """Return the messages to send given the *new_messages* of the next LLM call."""
        return new_messages

    def register_as_model(self) -> bool:
        """Return whether the SEA is offered in the model picker next to the real models."""
        return False

    def on_picked_as_model(self, work_dir: str) -> str:
        """Act on being picked as a tab's model (``register_as_model`` SEAs); return a log note.

        Args:
            work_dir: The tab's or run's work directory.
        """
        return ""
