# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Batched compaction of old tool outputs in a model conversation.

Every step re-sends the whole conversation, so a 10k-token file listing
read at step 5 is paid for again at step 50, 100 and 150 although the
model never looks at it again.  The 7-day audit of 2026-09-19 measured
steps above 200k context at 12 % of all steps but 19 % of all spend.

:func:`compact_tool_results` replaces the text of *old, large* tool
results with a short stub that says how to get the text back (re-run the
call).  It edits the conversation in place and never changes its length,
so message indices held elsewhere (``KISSAgent._llm_hook_conversation_index``)
stay valid.  The agent's own trajectory (``KISSAgent.messages``) and the
persisted events keep the full text.

Under provider prompt caching a compaction is not free: every request
after it re-writes the whole prefix from the first edited message (a
cache write costs 1.25x the input price, a cache read 0.1x, or 0.025x on
Claude Fable 5.1), so the 72 h audit of 2026-09-20 found 18 of 57
production compactions cost more than they saved.  The compaction is
therefore planned first (:func:`plan_compaction`) and applied only when
:func:`should_compact` says the drop pays for the cache miss: at least
:data:`MIN_DROP_FRACTION` of the context must go, and the session must
not already be within :data:`HANDOFF_PROXIMITY_FRACTION` of its hand-off
limit, where too few steps remain to amortise the miss.

Three conversation shapes carry tool results:

* generic Chat-Completions: ``{"role": "tool", "content": str}``,
  answering ``{"role": "assistant", "tool_calls": [{"id", "function": {"name"}}]}``;
* Anthropic: ``{"role": "user", "content": [{"type": "tool_result",
  "tool_use_id", "content": str | [blocks]}]}``, answering
  ``{"role": "assistant", "content": [{"type": "tool_use", "id", "name"}]}``;
* OpenAI Responses / Gemini: ``{"type": "function_call_output", "call_id",
  "output": str}``, answering ``{"type": "function_call", "call_id", "name"}``.
"""

from __future__ import annotations

from typing import Any

COMPACTION_START_TOKENS = 100_000
"""Context size (tokens) at which the first compaction is considered
(the default of ``Config.compaction_start_tokens``)."""
COMPACTION_STEP_TOKENS = 100_000
"""Context growth after a compaction before the next one is considered
(the default of ``Config.compaction_step_tokens``).
Production sessions grew 2-5k tokens a step, so 100k is 20-50 steps of
warm cache between two forced cache misses."""
KEEP_RECENT_TOOL_RESULTS = 6
"""The newest tool results are never compacted (nor is anything appended
after the last assistant turn, which the model has not seen yet).  The
model re-reads only its last few results; 20 kept too much (the median
first compaction dropped just 34 % of a 100k context)."""
MIN_COMPACT_CHARS = 500
"""Tool results shorter than this are left alone.  Below 500 characters
the stub is about as long as the text it replaces."""
MIN_DROP_FRACTION = 0.25
"""A compaction must remove at least this share of the context.  Dropping
25 % pays back the cache miss after ~48 further steps on 0.1x cache-read
pricing; smaller drops rarely do before the session hands off."""
HANDOFF_PROXIMITY_FRACTION = 0.8
"""No compaction once the context reaches this share of the hand-off limit:
the remaining headroom (20 % of the limit, ~70k tokens for a 350k limit)
is 15-35 steps, fewer than a compaction needs to pay for itself."""
CHARS_PER_TOKEN = 4
"""Rough characters-per-token ratio used to turn dropped text into tokens."""
STUB_PREVIEW_CHARS = 200
PROTECTED_TOOLS = frozenset({"Edit", "Write", "finish"})
STUB_PREFIX = "[compacted tool output:"


def _names_by_call_id(conversation: list[Any]) -> dict[str, str]:
    """Map every tool-call id in *conversation* to its tool name."""
    names: dict[str, str] = {}
    for message in conversation:
        if not isinstance(message, dict):
            continue
        if message.get("type") == "function_call":
            names[str(message.get("call_id"))] = str(message.get("name", ""))
            continue
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            if isinstance(call, dict):
                function = call.get("function") or {}
                names[str(call.get("id"))] = str(function.get("name", ""))
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    names[str(block.get("id"))] = str(block.get("name", ""))
    return names


def _is_assistant_turn(message: dict[str, Any]) -> bool:
    return message.get("role") == "assistant" or message.get("type") == "function_call"


def _result_slots(conversation: list[Any]) -> list[tuple[dict[str, Any], str, str]]:
    """Return ``(holder, key, call_id)`` for every tool-result text the model has seen.

    ``holder[key]`` is the ``str`` text of one tool result.  Results
    appended after the last assistant turn have not been sent to the
    model yet and are never candidates: compaction runs before a model
    call, and hiding fresh evidence before the model's first look at it
    would defeat the tool call that produced it.
    """
    last_turn = max(
        (i for i, m in enumerate(conversation) if isinstance(m, dict) and _is_assistant_turn(m)),
        default=-1,
    )
    slots: list[tuple[dict[str, Any], str, str]] = []
    for index, message in enumerate(conversation):
        if not isinstance(message, dict) or index > last_turn:
            continue
        if message.get("role") == "tool" and isinstance(message.get("content"), str):
            slots.append((message, "content", str(message.get("tool_call_id"))))
        elif message.get("type") == "function_call_output" and isinstance(
            message.get("output"), str
        ):
            slots.append((message, "output", str(message.get("call_id"))))
        elif message.get("role") == "user" and isinstance(message.get("content"), list):
            for block in message["content"]:
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    continue
                call_id = str(block.get("tool_use_id"))
                if isinstance(block.get("content"), str):
                    slots.append((block, "content", call_id))
                elif isinstance(block.get("content"), list):
                    for part in block["content"]:
                        if isinstance(part, dict) and isinstance(part.get("text"), str):
                            slots.append((part, "text", call_id))
    return slots


def make_stub(text: str, preview_chars: int = STUB_PREVIEW_CHARS) -> str:
    """Build the replacement for a compacted tool output.

    Args:
        text: The original tool output.
        preview_chars: How many leading characters to keep.

    Returns:
        A stub naming the original size, the way to get the output back,
        and the first *preview_chars* characters.
    """
    return (
        f"{STUB_PREFIX} {len(text):,} chars; the first {preview_chars} follow. "
        f"To see it again, Read the file or re-run a read-only command; do not "
        f"repeat a command that has side effects.]\n"
        + text[:preview_chars]
    )


def plan_compaction(
    conversation: list[Any],
    keep_recent: int = KEEP_RECENT_TOOL_RESULTS,
    min_chars: int = MIN_COMPACT_CHARS,
    protected_tools: frozenset[str] = PROTECTED_TOOLS,
) -> list[tuple[dict[str, Any], str]]:
    """Select the old, large tool outputs a compaction would stub, without editing.

    Args:
        conversation: The model conversation (any of the three shapes in
            the module docstring; unknown messages are ignored).
        keep_recent: Number of newest tool results left untouched.
        min_chars: Results shorter than this are left untouched.
        protected_tools: Tool names whose results are never compacted.

    Returns:
        ``(holder, key)`` pairs such that ``holder[key]`` is a tool-result
        text to replace; pass the list to :func:`apply_compaction`.
    """
    slots = _result_slots(conversation)
    candidates = slots[: max(0, len(slots) - keep_recent)]
    if not candidates:
        return []
    names = _names_by_call_id(conversation)
    plan: list[tuple[dict[str, Any], str]] = []
    for holder, key, call_id in candidates:
        text = holder[key]
        if len(text) <= min_chars or text.startswith(STUB_PREFIX):
            continue
        if names.get(call_id, "") in protected_tools:
            continue
        plan.append((holder, key))
    return plan


def dropped_chars(plan: list[tuple[dict[str, Any], str]]) -> int:
    """Return how many characters applying *plan* would remove from the conversation."""
    return sum(len(holder[key]) - len(make_stub(holder[key])) for holder, key in plan)


def should_compact(
    context_tokens: int, dropped_tokens: int, handoff_tokens: float | None
) -> bool:
    """Decide whether a planned compaction pays for the prompt-cache miss it causes.

    Args:
        context_tokens: Size of the last request, in tokens.
        dropped_tokens: Tokens the plan would remove (see :func:`dropped_chars`
            and :data:`CHARS_PER_TOKEN`).
        handoff_tokens: Context size at which the session hands off, or
            ``None`` when the model's window is unknown.

    Returns:
        ``True`` when the drop is at least :data:`MIN_DROP_FRACTION` of
        the context and the session is still below
        :data:`HANDOFF_PROXIMITY_FRACTION` of its hand-off limit.
    """
    if dropped_tokens < MIN_DROP_FRACTION * context_tokens:
        return False
    if handoff_tokens is not None and context_tokens >= HANDOFF_PROXIMITY_FRACTION * handoff_tokens:
        return False
    return True


def apply_compaction(plan: list[tuple[dict[str, Any], str]]) -> int:
    """Replace every planned tool output with its stub, in place.

    Args:
        plan: The result of :func:`plan_compaction`.

    Returns:
        The number of tool results that were replaced.
    """
    for holder, key in plan:
        holder[key] = make_stub(holder[key])
    return len(plan)


def compact_tool_results(
    conversation: list[Any],
    keep_recent: int = KEEP_RECENT_TOOL_RESULTS,
    min_chars: int = MIN_COMPACT_CHARS,
    protected_tools: frozenset[str] = PROTECTED_TOOLS,
) -> int:
    """Replace old, large tool outputs in *conversation* with stubs, in place.

    Unconditional (no cache-economics gate); the agent uses
    :func:`plan_compaction` + :func:`should_compact` + :func:`apply_compaction`.

    Args:
        conversation: The model conversation.
        keep_recent: Number of newest tool results left untouched.
        min_chars: Results shorter than this are left untouched.
        protected_tools: Tool names whose results are never compacted.

    Returns:
        The number of tool results that were replaced.
    """
    return apply_compaction(plan_compaction(conversation, keep_recent, min_chars, protected_tools))
