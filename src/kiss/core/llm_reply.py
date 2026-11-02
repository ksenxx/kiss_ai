# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""What an ``llm_response_hook`` receives: one model reply with its usage.

A :class:`~kiss.core.kiss_agent.KISSAgent` builds an :class:`LLMReply`
after every billed model call of its agentic loop and hands it to the
hook before any tool call of the reply runs.  The hook answers with a
:class:`~kiss.core.tool_verdict.Verdict`: ``ALLOW`` lets the reply be
acted on, ``end_run(result)`` ends the run with *result*.  Together
with ``llm_call_hook``, which rewrites the next request, the two hooks
see and shape every exchange with the model, which is what an
OpenAI-compatible proxy between the agent and the model used to be
needed for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

USAGE_KEYS = ("input_tokens", "output_tokens", "cache_read", "cache_write", "cost", "stop_reason")
"""The keys of :attr:`LLMReply.usage`, always all present."""


def zero_usage() -> dict[str, Any]:
    """Return a usage dict of all :data:`USAGE_KEYS` at zero (``stop_reason`` ``None``).

    The default of :attr:`LLMReply.usage`, so a reply built without
    one (the launcher's contract check) reads like a real reply.
    """
    return {"input_tokens": 0, "output_tokens": 0, "cache_read": 0, "cache_write": 0,
            "cost": 0.0, "stop_reason": None}


@dataclass(frozen=True)
class LLMReply:
    """One model reply of the agentic loop, as the ``llm_response_hook`` sees it.

    Attributes:
        text: The reply's text (empty when the model only called tools).
        tool_calls: The calls the reply asks for, each
            ``{"name": <tool>, "arguments": <dict>}``, in the order the
            agent would run them.
        usage: The call's own accounting, the agent's ``last_call_usage``:
            ``input_tokens``, ``output_tokens``, ``cache_read``,
            ``cache_write``, ``cost`` (USD) and ``stop_reason`` (the
            provider's, ``None`` when it gave none) — every key always
            present (:data:`USAGE_KEYS`).
        step: The agent's step count after this call (1 for the first).
        budget_used: USD spent by the agent so far, this call included.
        total_tokens_used: Tokens the agent has used so far, this call
            included.
    """

    text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=zero_usage)
    step: int = 0
    budget_used: float = 0.0
    total_tokens_used: int = 0
