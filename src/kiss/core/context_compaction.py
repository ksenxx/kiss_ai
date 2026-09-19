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
"""Context size (tokens) at which the first compaction runs."""
COMPACTION_STEP_TOKENS = 50_000
"""Context growth after a compaction that triggers the next one."""
KEEP_RECENT_TOOL_RESULTS = 20
"""The newest tool results are never compacted (nor is anything appended
after the last assistant turn, which the model has not seen yet)."""
MIN_COMPACT_CHARS = 2_000
"""Tool results shorter than this are left alone (Edit/Write confirmations
and most command outputs are below it)."""
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


def compact_tool_results(
    conversation: list[Any],
    keep_recent: int = KEEP_RECENT_TOOL_RESULTS,
    min_chars: int = MIN_COMPACT_CHARS,
    protected_tools: frozenset[str] = PROTECTED_TOOLS,
) -> int:
    """Replace old, large tool outputs in *conversation* with stubs, in place.

    Args:
        conversation: The model conversation (any of the three shapes in
            the module docstring; unknown messages are ignored).
        keep_recent: Number of newest tool results left untouched.
        min_chars: Results shorter than this are left untouched.
        protected_tools: Tool names whose results are never compacted.

    Returns:
        The number of tool results that were replaced.
    """
    slots = _result_slots(conversation)
    candidates = slots[: max(0, len(slots) - keep_recent)]
    if not candidates:
        return 0
    names = _names_by_call_id(conversation)
    compacted = 0
    for holder, key, call_id in candidates:
        text = holder[key]
        if len(text) <= min_chars or text.startswith(STUB_PREFIX):
            continue
        if names.get(call_id, "") in protected_tools:
            continue
        holder[key] = make_stub(text)
        compacted += 1
    return compacted
