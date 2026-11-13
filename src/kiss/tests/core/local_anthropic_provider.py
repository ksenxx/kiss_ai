# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""What the Anthropic API does with a request where the cost of a session depends on it.

Three behaviours of the provider, computed from the request body alone, for
the local stand-in (:mod:`kiss.tests.core.local_anthropic_server`):

* The prompt cache (:class:`PromptCache`): a request reads the longest
  prefix an earlier request left at one of its breakpoints and writes the
  rest, so the usage a session is billed follows from the bytes it sends.
* Thinking bound to its conversation (:func:`signatures`,
  :func:`stale_thinking`): Claude 5.1 and later sign each thinking block
  over the system prompt, the tools and the messages before it, and
  Claude Fable 5.2 answers 400 to a request that sends one back after
  any of those changed (measured on 2026-11-11 by stubbing the first
  tool result of a three-call loop).
* The server-side ``clear_tool_uses`` edit (:func:`clear_tool_uses`),
  applied as the provider applied it on 2026-11-11: in bulk and with a
  boundary that holds between two clears, except that on some calls it
  stopped one result short (229 of the 311 moves recorded that day).
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

CHARS_PER_TOKEN = 4
LOOKBACK_BLOCKS = 20
"""Blocks before a breakpoint among which the provider looks for a cached prefix."""
THINKING_TYPES = ("thinking", "redacted_thinking")
PREFIX_CHECKED_MODELS = (
    "claude-fable-5-1", "claude-fable-5-2", "claude-mythos-5-1", "claude-mythos-5-2",
    "claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5",
)
"""The models Anthropic documents as checking the prefix of a thinking block."""
CLEARED_RESULT = "[Old tool result content cleared]"
SHORT_BOUNDARY_PERIOD = 5
"""Every this many calls the provider's cleared set is one result short."""


def _content(message: dict[str, Any]) -> list[dict[str, Any]]:
    content = message["content"]
    return [{"type": "text", "text": content}] if isinstance(content, str) else content


def _text(block: dict[str, Any]) -> str:
    return json.dumps({k: v for k, v in block.items() if k != "cache_control"}, sort_keys=True)


def prompt_blocks(request: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the blocks of *request* in the order the cache reads them: tools, system, messages."""
    system = request.get("system") or []
    if isinstance(system, str):
        system = [{"type": "text", "text": system}]
    blocks = [*(request.get("tools") or []), *system]
    for message in request["messages"]:
        blocks.extend({"role": message["role"], **block} for block in _content(message))
    return blocks


def prompt_tokens(request: dict[str, Any]) -> int:
    """Return the size of *request*'s prompt, at :data:`CHARS_PER_TOKEN` characters a token."""
    return sum(len(_text(block)) // CHARS_PER_TOKEN for block in prompt_blocks(request))


class PromptCache:
    """The prompt cache of one session: the prefixes earlier requests left at their breakpoints."""

    def __init__(self) -> None:
        self.entries: set[str] = set()

    def usage(self, request: dict[str, Any]) -> dict[str, int]:
        """Return the usage the provider reports for *request*, and store its prefixes.

        A breakpoint is a block that carries ``cache_control`` or, under the
        top-level ``cache_control`` (automatic caching), the last block.
        The request reads the longest stored prefix that ends within
        :data:`LOOKBACK_BLOCKS` blocks of a breakpoint, writes the rest up
        to its last breakpoint and pays the full price for what follows.

        Args:
            request: The request body as the model reads it.
        """
        blocks = prompt_blocks(request)
        digest = hashlib.sha256()
        prefixes: list[tuple[str, int]] = []
        tokens = 0
        for block in blocks:
            text = _text(block)
            digest.update(text.encode())
            tokens += len(text) // CHARS_PER_TOKEN
            prefixes.append((digest.hexdigest(), tokens))
        points = [index for index, block in enumerate(blocks) if "cache_control" in block]
        if "cache_control" in request and blocks:
            points.append(len(blocks) - 1)
        read = max(
            (prefixes[index][1] for point in points
             for index in range(max(0, point - LOOKBACK_BLOCKS + 1), point + 1)
             if prefixes[index][0] in self.entries),
            default=0,
        )
        cached = max((prefixes[point][1] for point in points), default=0)
        self.entries.update(prefixes[point][0] for point in points)
        return {
            "input_tokens": tokens - cached,
            "output_tokens": 1,
            "cache_read_input_tokens": read,
            "cache_creation_input_tokens": cached - read,
        }


def signatures(request: dict[str, Any]) -> list[str]:
    """Return the signature of a thinking block at each message position of *request*.

    Entry ``i`` is a digest of the system prompt, the tools and the
    messages before message ``i``, their thinking blocks left out (a block
    is not bound to earlier thinking); the last entry is the one the reply
    to *request* gets.
    """
    digest = hashlib.sha256(
        json.dumps([request.get("system"), request.get("tools")], sort_keys=True).encode()
    )
    marks = [digest.hexdigest()]
    for message in request["messages"]:
        kept = [block for block in _content(message) if block.get("type") not in THINKING_TYPES]
        digest.update(json.dumps([message["role"], kept], sort_keys=True).encode())
        marks.append(digest.hexdigest())
    return marks


def stale_thinking(request: dict[str, Any]) -> str | None:
    """Return the provider's error for the first thinking block bound to a different conversation.

    ``None`` when every thinking block of *request* still follows the
    messages it was written after, or when the model is not one of the
    :data:`PREFIX_CHECKED_MODELS`.
    """
    if not request["model"].startswith(PREFIX_CHECKED_MODELS):
        return None
    marks = signatures(request)
    for index, message in enumerate(request["messages"]):
        if message["role"] != "assistant":
            continue
        for position, block in enumerate(_content(message)):
            if block.get("type") == "thinking" and block.get("signature") != marks[index]:
                return (
                    f"messages.{index}.content.{position}: Invalid `signature` in `thinking` "
                    "block. The block is bound to a different conversation. Remove the block, "
                    "or set `thinking.block_binding.prefix_mismatch_behavior` to \"drop_block\"."
                )
    return None


def _cleared_by_rule(request: dict[str, Any], edit: dict[str, Any]) -> int:
    """Return how many of the oldest tool results the ``clear_tool_uses`` *edit* clears.

    The rule replayed over the conversation as it grew: once the context,
    less what is already cleared, passes the trigger, every result but the
    last ``keep`` goes, provided that removes ``clear_at_least`` tokens.
    The count depends on the request alone, so it holds between two clears.
    """
    keep = edit["keep"]["value"]
    at_least = edit.get("clear_at_least", {}).get("value", 0)
    sizes: list[int] = []
    context = cleared = 0
    for block in prompt_blocks(request):
        size = len(_text(block)) // CHARS_PER_TOKEN
        context += size
        if block.get("type") != "tool_result":
            continue
        sizes.append(size)
        oldest = len(sizes) - keep
        freed = sum(sizes[cleared:max(cleared, oldest)])
        if oldest > cleared and context > edit["trigger"]["value"] and freed >= at_least:
            context -= freed
            cleared = oldest
    return cleared


def clear_tool_uses(
    request: dict[str, Any], calls: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Apply the ``clear_tool_uses`` edit of *request* as the provider did on 2026-11-11.

    Args:
        request: The request body the client sent.
        calls: How many requests the session has made, this one included;
            on every :data:`SHORT_BOUNDARY_PERIOD`-th the cleared set is
            one result short of the rule's.

    Returns:
        The request the model reads (*request* itself when nothing is
        cleared) and the ``applied_edits`` the response reports.
    """
    edits = (request.get("context_management") or {}).get("edits") or []
    edit = next((e for e in edits if e["type"] == "clear_tool_uses_20250919"), None)
    count = _cleared_by_rule(request, edit) if edit else 0
    if count and calls % SHORT_BOUNDARY_PERIOD == 0:
        count -= 1
    if not count:
        return request, []
    edited = copy.deepcopy(request)
    results = [block for message in edited["messages"] for block in _content(message)
               if block.get("type") == "tool_result"]
    for block in results[:count]:
        block["content"] = CLEARED_RESULT
    return edited, [{
        "type": "clear_tool_uses_20250919",
        "cleared_tool_uses": count,
        "cleared_input_tokens": prompt_tokens(request) - prompt_tokens(edited),
    }]
