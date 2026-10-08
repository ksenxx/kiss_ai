# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The typed answer of a tool-call hook: ``ALLOW`` the call, or ``refuse(text)`` it.

A :class:`~kiss.core.kiss_agent.KISSAgent` asks its ``tool_call_hook``
before every tool call; a SEA's ``tool_call_hook`` method is such a
hook.  The hook returns a :class:`Verdict` and nothing else — no
``None`` (the former "allow") and no bare string (the former "refuse"),
so an allow and a refusal can never be confused.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Verdict:
    """What a tool-call hook decided about one tool call.

    Attributes:
        allowed: Whether the call runs.
        text: The tool result the model reads instead when the call is
            refused; empty when allowed.
    """

    allowed: bool
    text: str = ""


ALLOW = Verdict(True)
"""The verdict that lets the tool call run."""


def refuse(text: str) -> Verdict:
    """Return the verdict that stops the tool call; the model reads *text* as the tool's result.

    Args:
        text: Why the call is refused, or what to do instead; must not
            be blank, since the model needs to read something.

    Raises:
        ValueError: When *text* is not a non-blank string.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("refuse() needs the non-blank text the model reads as the tool result")
    return Verdict(False, text)
