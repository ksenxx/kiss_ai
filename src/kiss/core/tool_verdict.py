# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The typed answer of a hook: ``ALLOW``, ``refuse(text)`` or ``end_run(result)``.

A :class:`~kiss.core.kiss_agent.KISSAgent` asks its ``tool_call_hook``
before every tool call and its ``llm_response_hook`` after every model
reply; a SEA's methods of those names are such hooks.  A hook returns a
:class:`Verdict` and nothing else — no ``None`` (the former "allow") and
no bare string (the former "refuse"), so an allow and a refusal can
never be confused.  :func:`end_run` is the verdict that ends the run
itself, with its text as the run's result: what a hook returns when a
rule of its own says the task is over (the report is accepted, the
episode's budget is spent), instead of waiting for the model to call
``finish``.

:class:`ToolRedirect` is the other way a tool answers the agent: raised
by a tool, it asks the agent to make a different call in its place.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Verdict:
    """What a hook decided about one tool call or one model reply.

    Attributes:
        allowed: Whether the call runs (the reply is acted on).
        text: The tool result the model reads instead when the call is
            refused, or the run's result when the verdict ends the run;
            when the call is allowed, a note its result ends with
            (:func:`allow_with_note`), empty for a plain ``ALLOW``.
        ends_run: Whether the run ends now, with *text* as its result
            (:func:`end_run`); the refused call does not run.
        success: Whether a run ended by this verdict counts as a success.
    """

    allowed: bool
    text: str = ""
    ends_run: bool = False
    success: bool = True


ALLOW = Verdict(True)
"""The verdict that lets the tool call run."""


def _non_blank(text: object, needed: str) -> str:
    """Return *text* when it is a non-blank string, else raise ``ValueError(needed)``."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError(needed)
    return text


def allow_with_note(text: str) -> Verdict:
    """Return the verdict that lets the tool call run and appends *text* to its result.

    For a rule the model should act on at its next step without losing
    this one: the call runs as asked, and the model reads *text* at the
    end of the result (the agent appends it after its result hook has
    read the plain result).

    Args:
        text: The note the result ends with; must not be blank.

    Raises:
        ValueError: When *text* is not a non-blank string.
    """
    return Verdict(
        True, _non_blank(text, "allow_with_note() needs the non-blank text appended to the result")
    )


class ToolRedirect(Exception):  # noqa: N818 - a control-flow signal, not an error
    """Raised by a tool to have the agent make another call in its place.

    A ``Bash`` that recognises a shell habit standing for a ``Write`` or
    an ``Edit`` (a here-document copy, a ``sed -i``) raises it instead
    of making that call itself: the agent then runs the call named here
    through its own tool path, so the tool-call hook, the guard and the
    event log see the ``Write``/``Edit`` exactly as if the model had
    issued it.  The original call's result is :attr:`note`, a newline,
    and the redirected call's result (a refusal when a hook or guard
    stopped it), then the :attr:`followup` call's result under a line
    naming that call, or the line saying it was not made because the
    redirected call did not go through (:func:`followup_result`).

    Attributes:
        name: The tool to call instead.
        arguments: Its arguments (``args`` is ``BaseException``'s tuple,
            so the dict cannot go by that name).
        note: The line the original call's result opens with: what the
            call was made as.
        followup: A second call ``(name, arguments)`` the agent makes
            through the same path once the redirected call went through:
            the rest of a compound command after the rewrite it stands
            for (``cd DIR && <rewrite> && <rest>``); ``None`` without one.
    """

    def __init__(
        self,
        name: str,
        arguments: dict[str, Any],
        note: str,
        followup: tuple[str, dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(note)
        self.name = name
        self.arguments = arguments
        self.note = note
        self.followup = followup


def call_text(name: str, arguments: dict[str, Any]) -> str:
    """Render the tool call *name* of *arguments* as a model writes it: ``name(key=value, ...)``.

    Values are JSON, so a command's newlines and quotes stay readable.
    """
    rendered = ", ".join(
        f"{key}={json.dumps(value, ensure_ascii=False)}" for key, value in arguments.items()
    )
    return f"{name}({rendered})"


def followup_result(
    name: str, arguments: dict[str, Any], result: str | None, redirected: str,
) -> str:
    """Report the :attr:`ToolRedirect.followup` call *name* of *arguments*.

    Args:
        name: The follow-up call's tool.
        arguments: Its arguments.
        result: Its result; ``None`` when it was not made.
        redirected: The redirected call's tool, which did not go through
            when the follow-up was not made.

    Returns:
        The line naming the call and its result, or the line saying it
        was not made.
    """
    call = call_text(name, arguments)
    if result is None:
        return f"{call} was not made: the {redirected} did not go through."
    return f"Then made {call}, whose result follows:\n{result}"


def refuse(text: str) -> Verdict:
    """Return the verdict that stops the tool call; the model reads *text* as the tool's result.

    Args:
        text: Why the call is refused, or what to do instead; must not
            be blank, since the model needs to read something.

    Raises:
        ValueError: When *text* is not a non-blank string.
    """
    return Verdict(
        False,
        _non_blank(text, "refuse() needs the non-blank text the model reads as the tool result"),
    )


def end_run(result: str, *, success: bool = True) -> Verdict:
    """Return the verdict that ends the run now, with *result* as the run's result.

    From a ``tool_call_hook`` the call it answers does not run and the
    run ends with *result*, rendered through the registered ``finish``
    tool's contract (Sorcar's ``finish(success, summary_in_html)`` YAML,
    a plain ``finish(result)``'s text) so the caller reads it like a
    result the model finished with; from an ``llm_response_hook`` the
    reply's tool calls do not run either.  The remaining tool calls of
    the same turn do not run.

    Args:
        result: The run's result; must not be blank.
        success: Whether the run counts as a success (``finish``'s
            ``success`` where the contract has one).

    Raises:
        ValueError: When *result* is not a non-blank string.
    """
    return Verdict(
        False,
        _non_blank(result, "end_run() needs the non-blank text that becomes the run's result"),
        ends_run=True,
        success=success,
    )


REFUSED_PREFIX = "REFUSED: the {name} call did NOT run and had no effect. "
"""The start of every refused call's tool result, before the refusal's own text."""

REISSUE_REMINDER = (
    "\n\nReminder: your earlier {name} call was REFUSED and never ran; "
    "re-issue it now if it is still needed."
)
"""Appended to the first allowed result after a refusal (not the refused tool's, not finish's)."""


def refusal_result(name: str, text: str) -> str:
    """Return the tool result the model reads for a refused call of *name*.

    Every refusal (a hook's ``refuse(text)`` or a guard's string)
    reaches the model through this, so a refusal can never read like
    a success: the result opens with :data:`REFUSED_PREFIX`, naming
    the tool and stating that it did not run, and the refusal's own
    *text* (the reason and what to do instead) follows.

    Args:
        name: The refused tool's name.
        text: The refusal's text.
    """
    return REFUSED_PREFIX.format(name=name) + text
