# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Batched compaction of old tool outputs in a model conversation.

Every step re-sends the whole conversation, so a 10k-token file listing
read at step 5 is paid for again at step 50, 100 and 150 although the
model never looks at it again; the few steps that run at a large
context take a share of the spend well above their share of the steps.

:func:`apply_compaction` replaces the text of *old, large* tool results
(selected by :func:`plan_compaction`) with a short stub that says how to
get the text back (re-run the call), and the large *arguments* of old
tool calls (the content of a ``Write``, the strings of an ``Edit``, a
long shell command) with a stub naming the file they went to:
arguments are a large share of the tool text in context, mostly
whole-file ``Write`` contents repeated as the file was rewritten.  A
stubbed ``Write`` says that the current file is one ``Read`` away (the
agent's ``context_reset_hook`` forgets the Read tool's dedupe so that
Read returns the content again).  It edits the conversation in place
and never changes its length,
so message indices held elsewhere (``KISSAgent._llm_hook_conversation_index``)
stay valid.  The agent's own trajectory (``KISSAgent.messages``) and the
persisted events keep the full text.

Under provider prompt caching a compaction is not free: every request
after it re-writes the whole prefix from the first edited message (a
cache write costs 1.25x the input price, a cache read 0.1x, or 0.025x on
Claude Fable 5.1), so a compaction can cost more than it saves.  The
compaction is therefore planned first (:func:`plan_compaction`) and
applied only when :func:`should_compact` says the drop pays for the
cache miss: at least :data:`MIN_DROP_FRACTION` of the context must go,
and the session must not already be within
:data:`HANDOFF_PROXIMITY_FRACTION` of its hand-off limit, where too few
steps remain to amortise the miss.

Compaction is scheduled against a steady-state target
(:func:`compaction_target_tokens`, ``Config.compaction_target_fraction``
of the model's window, 150k of 500k by default): once the context is
past the target every step is a candidate, so the per-step cost stays
near the target's instead of climbing to the hand-off limit (a step at
300k context costs several times a step at 50k, and the steps above
150k take a large share of a long session's spend).

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

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

COMPACTION_START_TOKENS = 100_000
"""Context size (tokens) at which the first compaction is considered
(the default of ``Config.compaction_start_tokens``)."""
COMPACTION_STEP_TOKENS = 100_000
"""Context growth after a compaction before the next one is considered
(the default of ``Config.compaction_step_tokens``).  Every compaction
forces a cache miss, so the gap keeps the cache warm for many steps
between two of them."""
COMPACTION_MIN_STEP_GAP = 10
"""Steps that must pass after a compaction before the next one is
considered, whatever the context did meanwhile.  Every compaction is a
full cache miss, so two of them a few steps apart pay twice for one
batch of stale outputs; waiting lets more results age out of the kept
window, so the next compaction is larger and rarer.  Ten steps (was
30): with arguments compacted too, a compaction above the target drops
more per step of waiting, and the per-step cost climbs steeply enough
above 150k that a quarter of a 150k context pays for its miss in about
ten steps."""
COMPACTION_TARGET_FRACTION = 0.3
"""Share of the model's context window the context is kept near once
compaction can hold it there (the default of
``Config.compaction_target_fraction``): 150k tokens of a 500k window."""
MIN_DROP_FRACTION_ABOVE_TARGET = 0.1
"""Above the target a compaction needs to remove only this share of the
context (more when the model's cache discount makes the miss dearer,
see :func:`payback_floor`): the miss it causes is paid back by the steps
that follow at the lower per-step cost, which is what the target exists
for."""
PAYBACK_STEPS = 50
"""Steps a compaction gets to pay for the prompt-cache miss it causes.
Dropping a share ``f`` of a context of ``C`` tokens saves ``f*C`` tokens
at the cache-read price on every later step and costs the ``(1-f)*C``
kept tokens at the input price once, so it pays back after
``(1/f - 1) * r`` steps, ``r`` being the input price over the cache-read
price minus one (19 for Anthropic's 5 % cache reads, 9 for OpenAI's
10 %).  :func:`payback_floor` turns this budget into the smallest share
worth dropping: 28 % for ``r = 19``, 15 % for ``r = 9``, nothing extra
for a model without cache pricing.  Fifty is on the order of the steps
a session has left once it reaches the target, with a margin for the
drop estimate, which under-counts; a fixed floor as low as 10 % compacts
so often that on Anthropic pricing it costs more than no compaction at
all."""
KEEP_RECENT_TOOL_RESULTS = 6
"""The newest tool results are never compacted (nor is anything appended
after the last assistant turn, which the model has not seen yet).  The
model re-reads only its last few results; 20 kept too much for the
first compaction of a session to drop a worthwhile share."""
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
STUB_PREFIX = "[compacted tool output"
ARGUMENT_STUB_PREFIX = "[compacted argument"
"""Opens the stub of a compacted tool-call argument."""
STUB_HEAD = re.compile(r"\[compacted (?:tool output|argument)(?: #(\d+))?:[^\n]*\](?:\n|$)")
"""A stub's whole head line: its kind, the id under which the table keeps the
original (none in the oldest format), and metadata up to the ``]`` that ends
the line (a file path in the metadata may itself carry brackets).  Only such a
whole line is a stub; the opening words alone are ordinary text."""
PROTECTED_ARGUMENT_TOOLS = frozenset({"finish"})
"""Tools whose arguments are never compacted (a ``finish`` ends the run)."""
FILE_ARGUMENT_TOOLS = frozenset({"Edit", "Write"})
"""Tools whose ``file_path`` argument names the file their other arguments
went to; the stub of such an argument points the model at the file."""
ARGUMENT_PREVIEW_CHARS = 120
"""Leading characters of a compacted argument kept in its stub."""


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


class CompactionTable:
    """The originals of the stubs one session's compactions made, by stub id.

    A stub is a reference the harness expands, never text the model can
    copy: a tool call whose argument is a stub gets the original back
    (:func:`expand_stubs`) before the tool runs.  Equal texts share one id,
    so re-planning a skipped compaction adds nothing.

    Attributes:
        originals: The original text of each stub, by id (from 1).
    """

    def __init__(self) -> None:
        self.originals: dict[int, str] = {}
        self._ids: dict[str, int] = {}

    def add(self, text: str) -> int:
        """Return the id of *text*, keeping it (one id per distinct text)."""
        digest = hashlib.sha256(text.encode("utf-8", errors="surrogatepass")).hexdigest()
        stub_id = self._ids.get(digest)
        if stub_id is None:
            stub_id = len(self.originals) + 1
            self._ids[digest] = stub_id
            self.originals[stub_id] = text
        return stub_id


def make_stub(
    text: str, preview_chars: int = STUB_PREVIEW_CHARS, table: CompactionTable | None = None,
) -> str:
    """Build the replacement for a compacted tool output.

    Args:
        text: The original tool output.
        preview_chars: How many leading characters to keep.
        table: Where the original is kept under the stub's id (a table of
            the call's own when ``None``: the stub then has an id nothing
            expands).

    Returns:
        A stub naming its id, the original size, the way to get the
        output back, and the first *preview_chars* characters.
    """
    stub_id = (table or CompactionTable()).add(text)
    return (
        f"{STUB_PREFIX} #{stub_id}: {len(text):,} chars; the first {preview_chars} follow. "
        f"To see it again, Read the file or re-run a read-only command; do not "
        f"repeat a command that has side effects.]\n" + text[:preview_chars]
    )


def expand_stubs(
    args: dict[str, Any], table: CompactionTable,
) -> tuple[dict[str, Any], list[str], str | None]:
    """Replace the stub arguments of a tool call by their originals.

    Args:
        args: The call's arguments.
        table: The session's :class:`CompactionTable`.

    Returns:
        ``(args, notes, refusal)``: the arguments with every string that is
        a stub of the table replaced by its original, one note per
        restored argument, and the refusal when an argument is a stub the
        table does not hold or carries a stub inside a longer text (then
        *args* is returned unchanged: nothing that is or contains a stub is
        ever written to disk, run as a command or used as an Edit string).
        With nothing to restore, *args* itself is returned, so an edit a
        tool hook makes to it in place is the one the tool runs on.
    """
    expanded = args
    notes: list[str] = []
    for key, value in args.items():
        if not isinstance(value, str):
            continue
        # A stub is a whole head line (:data:`STUB_HEAD`); text that merely
        # names a stub's opening words (a source line of this module, a grep
        # for them, prose about them) is content.
        head = STUB_HEAD.search(value)
        if head is None:
            continue
        if head.start() != 0:
            return args, [], (
                f"Argument {key!r} carries a compaction stub inside a longer text; a stub is "
                "harness bookkeeping, not content.  Pass the text itself (the original of a "
                "stub is restored only when the whole argument is that stub).")
        original = table.originals.get(int(head.group(1))) if head.group(1) else None
        if original is None:
            return args, [], (
                f"Argument {key!r} is a compaction stub that this session's compaction table "
                "does not hold: pass the text itself.")
        # What follows the stub's head line is its preview, the original's start.
        preview = value[head.end():]
        if not original.startswith(preview):
            return args, [], (
                f"Argument {key!r} is the compaction stub #{head.group(1)} followed by text that "
                "is not its preview; a stub stands for its whole original.  Pass the text itself.")
        if expanded is args:
            expanded = dict(args)
        expanded[key] = original
        notes.append(f"argument {key!r} (#{head.group(1)}) restored from the compaction table")
    return expanded, notes, None


@dataclass
class Replacement:
    """One planned edit of the conversation: ``holder[key]`` becomes ``value``.

    Attributes:
        holder: The dict (a message, a block, a tool-use input) holding the text.
        key: The key of the text in *holder*.
        value: What replaces it: a stub string, or for a tool-use ``input``
            dict the new dict with its large values stubbed.
        dropped: Characters the replacement removes.
        kind: ``"result"`` for a tool output, ``"argument"`` for a call's arguments.
    """

    holder: dict[str, Any]
    key: str
    value: Any
    dropped: int
    kind: str


def _call_slots(conversation: list[Any]) -> list[tuple[str, str, dict[str, Any], str]]:
    """Return ``(call_id, tool name, holder, key)`` for every tool call's arguments.

    ``holder[key]`` is the arguments as the provider shape keeps them: a
    JSON string (Chat Completions ``function.arguments``, Responses
    ``arguments``) or a dict (Anthropic ``tool_use.input``).
    """
    slots: list[tuple[str, str, dict[str, Any], str]] = []
    for message in conversation:
        if not isinstance(message, dict):
            continue
        if message.get("type") == "function_call" and isinstance(message.get("arguments"), str):
            slots.append((str(message.get("call_id")), str(message.get("name", "")),
                          message, "arguments"))
            continue
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            function = call.get("function") if isinstance(call, dict) else None
            if isinstance(function, dict) and isinstance(function.get("arguments"), str):
                slots.append((str(call.get("id")), str(function.get("name", "")),
                              function, "arguments"))
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if (isinstance(block, dict) and block.get("type") == "tool_use"
                        and isinstance(block.get("input"), dict)):
                    slots.append((str(block.get("id")), str(block.get("name", "")),
                                  block, "input"))
    return slots


def make_argument_stub(
    name: str, key: str, text: str, file_path: str, superseded: bool,
    table: CompactionTable | None = None,
) -> str:
    """Build the replacement for a compacted tool-call argument.

    Args:
        name: The tool that was called.
        key: The argument's name.
        text: The argument's original text.
        file_path: The file the call went to (``""`` when the tool has none).
        superseded: Whether a later ``Edit``/``Write`` changed that file.
        table: Where the original is kept under the stub's id (see :func:`make_stub`).

    Returns:
        A stub naming its id, the size, where the text went and how to see
        the current state, and the first :data:`ARGUMENT_PREVIEW_CHARS` characters.
    """
    if file_path:
        state = ("the file was changed again later" if superseded
                 else "this call made the file's current content")
        where = f"{name}.{key} to {file_path}; {state}; Read {file_path} to see it as it is now"
    else:
        where = f"{name}.{key}"
    # The head is the stub's first line (:func:`expand_stubs` parses it so):
    # a newline in the path is spelled out.
    where = where.replace("\n", "\\n")
    stub_id = (table or CompactionTable()).add(text)
    return (f"{ARGUMENT_STUB_PREFIX} #{stub_id}: {len(text):,} chars of {where}; "
            f"the first {ARGUMENT_PREVIEW_CHARS} follow]\n{text[:ARGUMENT_PREVIEW_CHARS]}")


def _stubbed_arguments(
    name: str, args: dict[str, Any], superseded: bool, table: CompactionTable | None,
) -> tuple[dict[str, Any], int]:
    """Return *args* with its large string values stubbed, and the characters removed."""
    file_path = str(args.get("file_path", "")) if name in FILE_ARGUMENT_TOOLS else ""
    new_args: dict[str, Any] = {}
    dropped = 0
    for key, value in args.items():
        if (isinstance(value, str) and len(value) > MIN_COMPACT_CHARS
                and not value.startswith(ARGUMENT_STUB_PREFIX) and key != "file_path"):
            stub = make_argument_stub(name, key, value, file_path, superseded, table)
            dropped += len(value) - len(stub)
            new_args[key] = stub
        else:
            new_args[key] = value
    return new_args, dropped


def _argument_plan(
    conversation: list[Any], old_call_ids: set[str], table: CompactionTable | None,
) -> list[Replacement]:
    """Plan the argument stubs of the calls in *old_call_ids* (see :func:`plan_compaction`)."""
    slots = _call_slots(conversation)
    # The last call that touched each file: every earlier Edit/Write of it is superseded.
    last_touch: dict[str, int] = {}
    parsed: list[dict[str, Any] | None] = []
    for index, (_call_id, name, holder, key) in enumerate(slots):
        raw = holder[key]
        args: Any = raw
        if isinstance(raw, str):
            try:
                args = json.loads(raw)
            except ValueError:
                args = None
        args = args if isinstance(args, dict) else None
        parsed.append(args)
        if args is not None and name in FILE_ARGUMENT_TOOLS and args.get("file_path"):
            last_touch[str(args["file_path"])] = index
    plan: list[Replacement] = []
    for index, (call_id, name, holder, key) in enumerate(slots):
        args = parsed[index]
        if args is None or call_id not in old_call_ids or name in PROTECTED_ARGUMENT_TOOLS:
            continue
        superseded = (name in FILE_ARGUMENT_TOOLS and bool(args.get("file_path"))
                      and last_touch.get(str(args["file_path"])) != index)
        new_args, dropped = _stubbed_arguments(name, args, superseded, table)
        if dropped <= 0:
            continue
        value: Any = new_args if key == "input" else json.dumps(new_args)
        plan.append(Replacement(holder, key, value, dropped, "argument"))
    return plan


def plan_compaction(
    conversation: list[Any],
    keep_recent: int = KEEP_RECENT_TOOL_RESULTS,
    arguments: bool = True,
    table: CompactionTable | None = None,
) -> list[Replacement]:
    """Select the old, large tool outputs and arguments a compaction would stub, without editing.

    Args:
        conversation: The model conversation (any of the three shapes in
            the module docstring; unknown messages are ignored).
        keep_recent: Number of newest tool results left untouched, with
            the arguments of their calls.  Results of at most
            :data:`MIN_COMPACT_CHARS` characters and those of the
            :data:`PROTECTED_TOOLS` are never compacted; arguments of at
            most :data:`MIN_COMPACT_CHARS` characters and those of the
            :data:`PROTECTED_ARGUMENT_TOOLS` neither.
        arguments: Whether the arguments of the old calls are planned too.
        table: The session's :class:`CompactionTable`, which keeps the
            originals under the ids the stubs carry (``None``: the stubs
            get ids nothing expands).

    Returns:
        The :class:`Replacement` list to pass to :func:`apply_compaction`.
    """
    slots = _result_slots(conversation)
    candidates = slots[: max(0, len(slots) - keep_recent)]
    if not candidates:
        return []
    names = _names_by_call_id(conversation)
    plan: list[Replacement] = []
    for holder, key, call_id in candidates:
        text = holder[key]
        if len(text) <= MIN_COMPACT_CHARS or text.startswith(STUB_PREFIX):
            continue
        if names.get(call_id, "") in PROTECTED_TOOLS:
            continue
        stub = make_stub(text, table=table)
        plan.append(Replacement(holder, key, stub, len(text) - len(stub), "result"))
    if arguments:
        plan.extend(_argument_plan(conversation, {call_id for _, _, call_id in candidates}, table))
    return plan


def dropped_chars(plan: list[Replacement]) -> int:
    """Return how many characters applying *plan* would remove from the conversation."""
    return sum(replacement.dropped for replacement in plan)


KEEP_RECENT_TURNS = 10
"""Newest model turns a whole-turn compaction keeps (:func:`plan_turn_removal`)."""
TURN_REMOVAL_NOTE = (
    "[Harness: the first {turns} turns of this session were removed from the context to "
    "keep it small. Their effects are on disk and in your progress and ledger files; Read "
    "a file again rather than recall it.]"
)
"""The user note that stands where the removed turns were."""
TURN_REMOVAL_HEAD = re.compile(r"^\[Harness: the first (\d+) turns of this session were removed")
"""Recognises an earlier removal's note, which the next removal absorbs."""


def _is_model_item(message: Any) -> bool:
    """Whether *message* is part of a model turn (its text, reasoning or tool calls)."""
    return isinstance(message, dict) and (
        message.get("role") == "assistant" or message.get("type") in ("function_call", "reasoning")
    )


@dataclass
class TurnRemoval:
    """One planned removal of whole old turns from a conversation.

    Attributes:
        start: Index of the first removed item (the first model turn).
        stop: Index after the last removed item (where the kept turns start).
        turns: Model turns removed.
        dropped: Characters the removal takes out, the note it leaves counted.
        note: The user message left in place of the removed turns.
    """

    start: int
    stop: int
    turns: int
    dropped: int
    note: str


def plan_turn_removal(
    conversation: list[Any], keep_recent: int = KEEP_RECENT_TURNS,
) -> TurnRemoval | None:
    """Plan the removal of every model turn but the newest *keep_recent*, without editing.

    For the providers that require the reasoning of every earlier turn
    to be passed back whenever tools are present (DeepSeek), where a
    stubbed argument or result is the only edit allowed inside a turn,
    old turns go whole: the model's reasoning, text and tool calls with
    their results and whatever user notes sat between them.  The
    messages before the first model turn (the task) stay.

    Args:
        conversation: The model conversation (any of the three shapes).
        keep_recent: Newest model turns kept, with their results.

    Returns:
        The :class:`TurnRemoval` to pass to :func:`apply_turn_removal`,
        or ``None`` when no more than *keep_recent* turns exist.
    """
    starts: list[int] = []
    previous = False
    for index, message in enumerate(conversation):
        current = _is_model_item(message)
        if current and not previous:
            starts.append(index)
        previous = current
    if len(starts) <= max(keep_recent, 0):
        return None
    start = starts[0]
    stop = starts[len(starts) - keep_recent] if keep_recent > 0 else len(conversation)
    turns = len(starts) - max(keep_recent, 0)
    # An earlier removal's note just before the first turn is absorbed, so
    # one note tells how many turns are gone in all.
    earlier = conversation[start - 1] if start > 0 else None
    head = (
        TURN_REMOVAL_HEAD.match(str(earlier.get("content", "")))
        if isinstance(earlier, dict) else None
    )
    if head is not None and isinstance(earlier, dict) and earlier.get("role") == "user":
        start -= 1
        turns += int(head.group(1))
    note = TURN_REMOVAL_NOTE.format(turns=turns)
    removed = sum(len(json.dumps(item, default=str)) for item in conversation[start:stop])
    return TurnRemoval(start, stop, turns, removed - len(note), note)


def apply_turn_removal(conversation: list[Any], removal: TurnRemoval) -> int:
    """Remove the planned turns from *conversation* in place, leaving the note.

    Args:
        conversation: The list :func:`plan_turn_removal` was planned on.
        removal: The plan.

    Returns:
        The number of items removed (the note not subtracted).
    """
    removed = removal.stop - removal.start
    conversation[removal.start:removal.stop] = [{"role": "user", "content": removal.note}]
    return removed


def compaction_target_tokens(
    window_tokens: int | None, start_tokens: int, target_fraction: float,
) -> int:
    """Return the context size (tokens) past which every step considers a compaction.

    Args:
        window_tokens: The model's context window, ``None`` when unknown.
        start_tokens: ``Config.compaction_start_tokens``, the floor.
        target_fraction: ``Config.compaction_target_fraction`` of the window.

    Returns:
        The larger of *start_tokens* and the fraction of the window
        (*start_tokens* when the window is unknown).
    """
    if window_tokens is None:
        return start_tokens
    return max(start_tokens, int(window_tokens * target_fraction))


def payback_floor(cache_discount: float | None) -> float:
    """Return the share of the context a compaction must drop to pay back in :data:`PAYBACK_STEPS`.

    Args:
        cache_discount: The model's cache-read price divided by its input
            price (0.05 for Anthropic, 0.1 for OpenAI); ``None``, zero or
            one and above mean the model has no cache discount, so a
            compaction costs no miss and any drop pays at once.

    Returns:
        ``r / (r + PAYBACK_STEPS)`` with ``r = 1 / cache_discount - 1``,
        or ``0.0`` without a cache discount.
    """
    if cache_discount is None or cache_discount <= 0.0 or cache_discount >= 1.0:
        return 0.0
    ratio = 1.0 / cache_discount - 1.0
    return ratio / (ratio + PAYBACK_STEPS)


def should_compact(
    context_tokens: int,
    dropped_tokens: int,
    handoff_tokens: float | None,
    target_tokens: int | None = None,
    cache_discount: float | None = None,
) -> bool:
    """Decide whether a planned compaction pays for the prompt-cache miss it causes.

    Args:
        context_tokens: Size of the last request, in tokens.
        dropped_tokens: Tokens the plan would remove (see :func:`dropped_chars`
            and :data:`CHARS_PER_TOKEN`).
        handoff_tokens: Context size at which the session hands off, or
            ``None`` when the model's window is unknown.
        target_tokens: The steady-state target (:func:`compaction_target_tokens`);
            ``None`` applies the floor of a context below it.
        cache_discount: The model's cache-read price over its input price
            (see :func:`payback_floor`); ``None`` when unknown.

    Returns:
        ``True`` when the drop is at least :data:`MIN_DROP_FRACTION` of
        the context (:data:`MIN_DROP_FRACTION_ABOVE_TARGET` once the
        context is past *target_tokens*), at least the
        :func:`payback_floor` of the model's cache discount, and the
        session is still below :data:`HANDOFF_PROXIMITY_FRACTION` of its
        hand-off limit.
    """
    floor = MIN_DROP_FRACTION
    if target_tokens is not None and context_tokens >= target_tokens:
        floor = MIN_DROP_FRACTION_ABOVE_TARGET
    floor = max(floor, payback_floor(cache_discount))
    if dropped_tokens < floor * context_tokens:
        return False
    if handoff_tokens is not None and context_tokens >= HANDOFF_PROXIMITY_FRACTION * handoff_tokens:
        return False
    return True


def apply_compaction(plan: list[Replacement]) -> int:
    """Replace every planned tool output and argument with its stub, in place.

    Args:
        plan: The result of :func:`plan_compaction`.

    Returns:
        The number of replacements made (results and arguments).
    """
    for replacement in plan:
        replacement.holder[replacement.key] = replacement.value
    return len(plan)
