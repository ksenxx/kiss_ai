# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Abstract base class for LLM provider model implementations.

Also contains shared text-based tool calling helpers used by models that
lack native function calling support (e.g. DeepSeek R1, Claude Code CLI),
and — beside :class:`CLITextModel`, the base class of both CLI-backed
transports — the subprocess supervision those two share:
:class:`_CLIProcess` and :class:`_ToolCallFilteredStream`.
"""

import base64
import contextlib
import dataclasses
import importlib
import inspect
import json
import logging
import os
import queue
import re
import shutil
import signal
import subprocess
import threading
import time
import types as types_module
import typing
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any, Union, get_args, get_origin

from kiss.core import stop_signal
from kiss.core.kiss_error import KISSError
from kiss.core.locked import locked_call
from kiss.core.models.heif import (
    HEIF_MIME_TYPES,
    heif_to_jpeg,
    is_heif,
)
from kiss.core.models.stream_abort import (
    DEFAULT_STREAM_STALL_TIMEOUT,
    stop_aware_events,
    stop_error,
)
from kiss.core.processes import (
    _REAP_GRACE_SECONDS,
    SIGKILL,
    kill_process_group,
    popen_process_group,
)

logger = logging.getLogger(__name__)

TokenCallback = Callable[[str], None]

ThinkingCallback = Callable[[bool], None]

SUPPORTED_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
    "application/pdf",
    "audio/mpeg",
    "audio/wav",
    "audio/x-wav",
    "audio/ogg",
    "audio/webm",
    "audio/flac",
    "audio/aac",
    "audio/mp4",
    "video/mp4",
    "video/webm",
    "video/ogg",
    "video/mpeg",
    "video/quicktime",
}

_RESPONSES_PART_TYPES = {
    "input_text",
    "output_text",
    "input_image",
    "input_file",
    "input_audio",
    "refusal",
}


def _responses_parts_to_chat_parts(parts: list[Any]) -> list[dict[str, Any]]:
    """Convert Responses-API message content parts to Chat-Completions parts.

    ``input_text`` / ``output_text`` become ``text`` parts, ``refusal``
    becomes a ``text`` part, ``input_image`` becomes an ``image_url`` part,
    ``input_file`` becomes a ``file`` part and ``input_audio`` passes
    through (the shapes match).  Parts already in Chat-Completions shape
    are passed through unchanged.

    Args:
        parts: Content parts from a Responses-API message item.

    Returns:
        Chat-Completions-format content parts.
    """
    out: list[dict[str, Any]] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        ptype = part.get("type")
        if ptype in ("input_text", "output_text"):
            out.append({"type": "text", "text": str(part.get("text", ""))})
        elif ptype == "refusal":
            text = str(part.get("refusal", ""))
            if text.strip():
                out.append({"type": "text", "text": text})
        elif ptype == "input_image":
            url = part.get("image_url", "")
            if isinstance(url, str) and url:
                out.append({"type": "image_url", "image_url": {"url": url}})
        elif ptype == "input_file":
            out.append(
                {
                    "type": "file",
                    "file": {
                        "filename": part.get("filename", "attachment.pdf"),
                        "file_data": part.get("file_data", ""),
                    },
                }
            )
        else:
            out.append(part)
    return out


def responses_items_to_chat_messages(conversation: list[Any]) -> list[Any]:
    """Convert OpenAI Responses-API input items to Chat-Completions messages.

    :class:`~kiss.core.models.openai_compatible_model2.OpenAICompatibleModel2`
    stores its conversation as Responses-API ``input`` items (``message``
    items with ``input_text`` / ``output_text`` parts, standalone
    ``function_call`` / ``function_call_output`` / ``reasoning`` items).
    When such a conversation is handed off to another provider's model
    (e.g. via the Sorcar ``set_model`` tool, which does
    ``new_model.conversation = old_model.conversation``), those items must
    first be translated to the OpenAI Chat Completions format that every
    other model's normalizer understands:

    * ``function_call`` items become ``tool_calls`` entries merged into the
      immediately preceding assistant message (or a fresh assistant message
      when none precedes),
    * ``function_call_output`` items become ``role="tool"`` messages,
    * ``reasoning`` items (hidden provider state) are dropped,
    * ``message`` items keep their role and have their ``input_*`` /
      ``output_*`` content parts converted to Chat-Completions parts.

    Messages already in Chat-Completions (or another provider's) format are
    passed through unchanged, so the conversion is safe to apply to any
    conversation.  Shared message dicts are never mutated (copy-on-write).

    Args:
        conversation: The conversation, possibly containing Responses items.

    Returns:
        The conversation with all Responses-API items converted to
        Chat-Completions format.
    """
    out: list[Any] = []
    for item in conversation:
        if not isinstance(item, dict):
            out.append(item)
            continue
        itype = item.get("type")
        role = item.get("role")
        if itype == "function_call":
            args = item.get("arguments")
            tc = _chat_tool_call_entry(
                item.get("call_id", ""),
                item.get("name", ""),
                args if isinstance(args, str) else json.dumps(args or {}),
            )
            prev = out[-1] if out else None
            if isinstance(prev, dict) and prev.get("role") == "assistant":
                merged = dict(prev)
                merged["tool_calls"] = list(prev.get("tool_calls") or []) + [tc]
                out[-1] = merged
            else:
                out.append({"role": "assistant", "content": "", "tool_calls": [tc]})
            continue
        if itype == "function_call_output":
            output = item.get("output", "")
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": item.get("call_id", ""),
                    "content": output if isinstance(output, str) else json.dumps(output),
                }
            )
            continue
        if role is None:
            continue
        content = item.get("content")
        if isinstance(content, list) and any(
            isinstance(p, dict) and p.get("type") in _RESPONSES_PART_TYPES for p in content
        ):
            converted: dict[str, Any] = {
                "role": role,
                "content": _responses_parts_to_chat_parts(content),
            }
            if item.get("tool_calls"):
                converted["tool_calls"] = item["tool_calls"]
            out.append(converted)
            continue
        if itype == "message":
            out.append({"role": role, "content": content})
            continue
        out.append(item)
    return out


@dataclasses.dataclass
class Attachment:
    """A file attachment (image, document, audio, or video) to include in a prompt.

    Attributes:
        data: Raw file bytes.
        mime_type: MIME type string (e.g. "image/jpeg", "application/pdf",
            "audio/mpeg", "video/mp4").
    """

    data: bytes
    mime_type: str

    @staticmethod
    def from_bytes(data: bytes, mime_type: str) -> "Attachment":
        """Create an Attachment from raw bytes, transcoding HEIF photos.

        iPhone camera photos arrive as HEIC, which the OpenAI and Anthropic
        vision APIs reject.  Such bytes are transcoded to JPEG here so that
        every model sees a format it understands; the declared MIME type is
        ignored in favour of the file header, because iOS and the browsers
        disagree about (and sometimes omit) it.

        Args:
            data: Raw file bytes.
            mime_type: MIME type reported by the sender, if any.

        Returns:
            An Attachment holding either the original bytes or, for a HEIF
            photo on a host with a HEIF decoder, its JPEG rendition.
        """
        if not is_heif(data):
            return Attachment(data=data, mime_type=mime_type)
        jpeg = heif_to_jpeg(data)
        if jpeg is not None:
            return Attachment(data=jpeg, mime_type="image/jpeg")
        # Without a decoder the photo still reaches Gemini, the one provider
        # that accepts HEIF natively; a correct label beats a wrong one.
        return Attachment(data=data, mime_type="image/heic")

    def to_base64(self) -> str:
        """Return the file data as a base64-encoded string."""
        return base64.b64encode(self.data).decode("ascii")

    def to_data_url(self) -> str:
        """Return a data: URL suitable for OpenAI image_url fields."""
        return f"data:{self.mime_type};base64,{self.to_base64()}"


BINARY_ATTACHMENT_OPEN = "<<KISS_BINARY_ATTACHMENT mime_type={mime_type}>>"
BINARY_ATTACHMENT_OPEN_RE = re.compile(
    re.escape(BINARY_ATTACHMENT_OPEN).replace(re.escape("{mime_type}"), r"([^>\s]+)")
)
BINARY_ATTACHMENT_CLOSE = "<</KISS_BINARY_ATTACHMENT>>"

# The Read tool may also embed a HEIF photo: parse_binary_attachments() runs
# it through Attachment.from_bytes(), which transcodes it to JPEG.
READ_TOOL_BINARY_MIME_TYPES = set(SUPPORTED_MIME_TYPES) | set(HEIF_MIME_TYPES)


def encode_binary_attachment(mime_type: str, data: bytes) -> str:
    """Encode ``data`` for inline transport inside a tool result string.

    Args:
        mime_type: MIME type label (e.g. ``"image/png"``).
        data: Raw file bytes.

    Returns:
        A sentinel-wrapped base64 payload that
        :func:`parse_binary_attachments` will later decode back into an
        :class:`Attachment`.
    """
    opener = BINARY_ATTACHMENT_OPEN.format(mime_type=mime_type)
    return f"{opener}{Attachment(data, mime_type).to_base64()}{BINARY_ATTACHMENT_CLOSE}"


def parse_binary_attachments(text: str) -> tuple[str, list[Attachment]]:
    """Split a tool result string into plain text + binary attachments.

    Scans ``text`` for ``<<KISS_BINARY_ATTACHMENT ...>>...<<...>>`` blocks
    produced by :func:`encode_binary_attachment` and returns the residual
    plain text (with each block replaced by a short
    ``[attached image/png, N bytes]`` placeholder) along with the decoded
    :class:`Attachment` list.

    Args:
        text: Raw tool result content possibly containing sentinel blocks.

    Returns:
        ``(plain_text, attachments)``.  If the input has no sentinels,
        returns ``(text, [])`` unchanged.
    """
    attachments: list[Attachment] = []
    out_parts: list[str] = []
    cursor = 0
    for match in BINARY_ATTACHMENT_OPEN_RE.finditer(text):
        open_start, open_end = match.span()
        if open_start < cursor:
            # An opener inside the block just consumed (only possible in
            # text that quotes the sentinel, e.g. a Read of this file).
            # Skipping it keeps the scan linear: without this, every
            # such opener re-searched for the same closer and re-decoded
            # the overlapping payload, which is quadratic.
            continue
        close_idx = text.find(BINARY_ATTACHMENT_CLOSE, open_end)
        if close_idx == -1:  # pragma: no cover – defensive: malformed sentinel
            break
        mime_type = match.group(1)
        b64_payload = text[open_end:close_idx]
        out_parts.append(text[cursor:open_start])
        cursor = close_idx + len(BINARY_ATTACHMENT_CLOSE)
        try:
            data = base64.b64decode(b64_payload, validate=False)
        except Exception:
            # validate=False still rejects a payload of the wrong length
            # (incorrect padding); keep the surrounding text, drop the block.
            logger.debug("Failed to decode binary attachment", exc_info=True)
            out_parts.append(f"[undecodable {mime_type} attachment dropped]")
            continue
        out_parts.append(f"[attached {mime_type}, {len(data)} bytes]")
        attachments.append(Attachment.from_bytes(data, mime_type))
    out_parts.append(text[cursor:])
    return "".join(out_parts), attachments


def _tool_result_to_string(result_dict: dict[str, Any]) -> str:
    """Return a provider-safe string for a tool result payload.

    Structured values are JSON encoded consistently across the base,
    Chat-Completions, and Responses implementations.  Values unsupported by
    JSON fall back to ``str`` so attachment parsing always receives text.
    """
    raw_result = result_dict["result"] if "result" in result_dict else str(result_dict)
    if isinstance(raw_result, str):
        return raw_result
    try:
        return json.dumps(raw_result, ensure_ascii=False)
    except TypeError:
        return str(raw_result)


def parse_tool_arguments(arguments: Any) -> dict[str, Any]:
    """Return a tool call's ``arguments`` as a dict.

    Adapters store arguments as dicts, but a conversation handed off from
    an OpenAI-schema model (e.g. via the Sorcar ``set_model`` tool) stores
    them as JSON strings.  A blank string, unparseable JSON, JSON that is
    not an object, or any other value degrades to ``{}``.

    Args:
        arguments: The arguments value (dict, JSON string, or anything else).

    Returns:
        The arguments as a dict.
    """
    if isinstance(arguments, dict):
        return arguments
    if not isinstance(arguments, str) or not arguments.strip():
        return {}
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        logger.debug("Exception caught", exc_info=True)
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _chat_tool_call_entry(call_id: str, name: str, args_json: str) -> dict[str, Any]:
    """Return the Chat Completions ``tool_calls`` entry for one call.

    Args:
        call_id: The provider's call id.
        name: The tool name.
        args_json: The model's argument string, stored verbatim so the
            conversation replays exactly what the model wrote.
    """
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": args_json}}


def split_data_url(
    url: str, default_mime: str = "application/octet-stream"
) -> tuple[str, str] | None:
    """Split a base64 ``data:`` URL into its media type and base64 payload.

    Args:
        url: A URL that may be a ``data:<media_type>;base64,<data>`` URL.
        default_mime: The media type to report when the URL omits one.

    Returns:
        A ``(media_type, base64_data)`` tuple, or ``None`` when *url* is
        not a base64 data URL with a non-empty payload.
    """
    if not url.startswith("data:"):
        return None
    header, _, data = url.partition(",")
    if ";base64" not in header or not data:
        return None
    media_type = header[len("data:") :].split(";", 1)[0]
    return media_type or default_mime, data


def pop_max_tokens_and_stop(params: dict[str, Any]) -> tuple[Any, list[Any] | None]:
    """Pop the OpenAI-style output-cap and stop aliases out of *params*.

    ``max_tokens`` and ``max_completion_tokens`` name the same cap; both
    are removed and the first one set wins.  ``stop`` is removed and
    returned as a list (a bare string becomes a one-item list) so the
    caller can store it under the provider's ``stop_sequences`` key.

    Args:
        params: Request parameters, mutated in place.

    Returns:
        ``(max_tokens, stop_sequences)``, each ``None`` when not set.
    """
    max_tokens = params.pop("max_tokens", None)
    max_completion_tokens = params.pop("max_completion_tokens", None)
    if max_tokens is None:
        max_tokens = max_completion_tokens
    stop = params.pop("stop", None)
    if isinstance(stop, str):
        stop = [stop]
    elif not isinstance(stop, list):
        stop = None
    return max_tokens, stop


def iter_assistant_tool_calls(message: dict[str, Any]) -> Iterator[tuple[str, str]]:
    """Yield ``(function_name, tool_call_id)`` for every tool call in *message*.

    Recognises both message shapes a conversation can carry — OpenAI
    ``tool_calls`` entries and Anthropic ``tool_use`` content blocks — so
    a conversation handed off between providers (e.g. via the Sorcar
    ``set_model`` tool) is read the same way everywhere.  A missing name
    or id yields ``""``.

    Args:
        message: One assistant message.

    Yields:
        ``(function_name, tool_call_id)`` per tool call, in message order.
    """
    for tc in message.get("tool_calls") or []:
        if isinstance(tc, dict):
            yield (tc.get("function") or {}).get("name", ""), tc.get("id", "")
    content = message.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                yield block.get("name", ""), block.get("id", "")


# The one canonical audio table: MIME type -> the short format string the
# OpenAI SDKs use (``input_audio.format``).  The two other spellings of the
# same knowledge — format -> MIME (Anthropic's Whisper fallback) and
# MIME -> file extension (the Whisper upload filename) — are derived below,
# so a new audio format is added in exactly one place.
_AUDIO_MIME_TO_FORMAT: dict[str, str] = {
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/ogg": "ogg",
    "audio/webm": "webm",
    "audio/flac": "flac",
    "audio/aac": "aac",
    "audio/mp4": "mp4",
}


# format -> MIME, keeping the FIRST (canonical) MIME for formats that have
# aliases: mp3 -> audio/mpeg (not audio/mp3), wav -> audio/wav.
def _invert_audio_table(mime_to_format: dict[str, str]) -> dict[str, str]:
    """Invert a MIME->format table, keeping the first (canonical) MIME per format."""
    inverted: dict[str, str] = {}
    for mime, fmt in mime_to_format.items():
        inverted.setdefault(fmt, mime)
    return inverted


_AUDIO_FORMAT_TO_MIME: dict[str, str] = _invert_audio_table(_AUDIO_MIME_TO_FORMAT)

# MIME -> filename extension; audio/mp4 bytes conventionally use ``.m4a``.
_AUDIO_MIME_TO_EXT: dict[str, str] = {
    mime: (".m4a" if fmt == "mp4" else f".{fmt}") for mime, fmt in _AUDIO_MIME_TO_FORMAT.items()
}


def _audio_mime_to_format(mime_type: str) -> str:
    """Map an audio MIME type to the short format string expected by OpenAI.

    Args:
        mime_type: An audio MIME type (e.g. "audio/mpeg").

    Returns:
        The short format string (e.g. "mp3"). Falls back to the MIME subtype
        if no explicit mapping exists.
    """
    fallback = mime_type.split("/", 1)[1] if "/" in mime_type else mime_type
    return _AUDIO_MIME_TO_FORMAT.get(mime_type, fallback)


#: Serializes the lazy imports of the provider modules and SDKs.  The
#: ``openai`` package's import graph is cyclic (``openai.resources.chat``,
#: ``openai.resources.responses`` and ``openai.resources.audio`` reach
#: each other through ``openai.types``), which Python's per-module import
#: locks resolve only when ONE thread walks the cycle: two threads
#: entering it from different modules (a task thread building its model
#: while the daemon's prewarm thread imports the other adapter, or a
#: transcription importing the audio resource) trip the import system's
#: deadlock detector and the import fails with ``_DeadlockError``.
#: Re-entrant so an import done under the lock may import another one.
_IMPORT_LOCK = threading.RLock()


def import_provider_module(name: str) -> types_module.ModuleType:
    """Import module *name* under the process-wide provider import lock.

    Every lazy import of a ``kiss.core.models`` provider module or of a
    provider SDK submodule (the package's lazy class attributes, the
    daemon's prewarm, the Responses-API delegate, the audio resource of
    :func:`transcribe_audio`) goes through here so that no two threads
    walk the SDK's import graph at the same time.  Interrupt-safe on a
    task thread (see :func:`kiss.core.locked.locked_call`).

    Args:
        name: Dotted module name, e.g. ``"kiss.core.models.gemini_model"``.

    Returns:
        The imported module.
    """
    return locked_call(_IMPORT_LOCK, importlib.import_module, name)


def transcribe_audio(data: bytes, mime_type: str, api_key: str | None = None) -> str:
    """Transcribe audio bytes to text using OpenAI's Whisper API.

    This is used as a fallback for model providers that do not support audio
    attachments natively (e.g. Anthropic).  The call runs synchronously while
    a request is being built and is not stop-aware, so its wait is bounded
    tightly: 10 s to connect, 120 s per attempt, and the SDK's own retry of
    transient failures (connection errors, timeouts, 408/409/429 and 5xx)
    at most twice.  That is the only retry policy; there is no loop here.

    Args:
        data: Raw audio file bytes.
        mime_type: MIME type of the audio (e.g. ``"audio/mpeg"``).
        api_key: OpenAI API key.  Falls back to the ``OPENAI_API_KEY``
            environment variable when *None*.

    Returns:
        The transcribed text.

    Raises:
        ValueError: If no API key is available.
        RuntimeError: If the transcription API call fails.
    """
    import httpx

    openai = import_provider_module("openai")
    # ``client.audio`` imports this lazily; done here, under the lock.
    import_provider_module("openai.resources.audio")

    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise ValueError("OpenAI API key is required for audio transcription")

    ext = _AUDIO_MIME_TO_EXT.get(mime_type, ".mp3")
    client = openai.OpenAI(api_key=key, timeout=httpx.Timeout(120.0, connect=10.0), max_retries=2)
    try:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=(f"audio{ext}", data, mime_type),
            response_format="text",
        )
    except Exception as exc:
        raise RuntimeError(f"Audio transcription failed: {exc}") from exc
    return str(transcript).strip()


def _get_attr_or_key(obj: Any, key: str, default: Any = None) -> Any:
    """Return ``obj.key`` or ``obj[key]``, falling back to *default*.

    Provider responses arrive either as pydantic SDK objects (attribute
    access; OpenRouter's extra usage fields such as ``cost`` are pydantic
    extras readable the same way) or as plain dicts (gateway payloads,
    recorded JSON, the Responses-delegate path).

    Args:
        obj: A pydantic object, a dict, or ``None``.
        key: The attribute or mapping key to read.
        default: Returned when *obj* is ``None`` or lacks *key*.

    Returns:
        The field value, or *default*.
    """
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _iter_jsonl(lines: Iterable[str]) -> Iterator[Any]:
    """Yield the decoded JSON values of a JSONL event stream, one per line.

    Blank lines and lines that are not valid JSON are skipped: the CLI
    transports interleave their ``--json`` event stream with progress
    noise, and one undecodable line must not end the turn.

    Args:
        lines: The stream's lines (one event per line).

    Yields:
        Each decoded line — an event dict in a well-formed stream.
    """
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def normalize_content_blocks(
    content: Any,
    foreign_types: tuple[str, ...],
    convert_foreign: Callable[[dict[str, Any]], dict[str, Any] | None],
    keep_thinking: bool,
) -> list[dict[str, Any]]:
    """Normalize response content blocks to JSON-serializable dicts.

    Text blocks whose text is empty or whitespace-only are dropped: the
    APIs reject them (``text content blocks must contain non-whitespace
    text``).  A dict block whose ``type`` is one of *foreign_types* (the
    other provider's media parts, present after a conversation hand-off)
    is replaced by ``convert_foreign(block)`` or dropped when that returns
    ``None``; every other dict passes through.  SDK objects become dicts:
    ``text`` and ``tool_use`` by their fields, anything else by
    ``model_dump(exclude_none=True)`` or ``str()``.  ``thinking`` and
    ``redacted_thinking`` blocks are hidden provider state that must not
    be replayed to a different provider, so they are dropped unless
    *keep_thinking* (an SDK ``thinking`` object then keeps its
    ``thinking`` text and ``signature``).

    Args:
        content: The content blocks of a response or message (``None``
            for none).
        foreign_types: Block ``type`` values routed through *convert_foreign*.
        convert_foreign: Converts one foreign dict block, ``None`` to drop it.
        keep_thinking: Keep thinking blocks (same-provider replay).

    Returns:
        list[dict[str, Any]]: Normalized content blocks as dictionaries.
    """
    blocks: list[dict[str, Any]] = []
    if content is None:
        return blocks
    for block in content:
        if isinstance(block, dict):
            dict_block_type = block.get("type")
            if dict_block_type in ("thinking", "redacted_thinking") and not keep_thinking:
                continue
            if dict_block_type == "text" and not block.get("text", "").strip():
                continue
            if dict_block_type in foreign_types:
                converted = convert_foreign(block)
                if converted is not None:
                    blocks.append(converted)
                continue
            blocks.append(block)
            continue
        block_type = getattr(block, "type", None)
        if block_type in ("thinking", "redacted_thinking") and not keep_thinking:
            continue
        if block_type == "text":
            text = getattr(block, "text", "")
            if not text.strip():
                continue
            blocks.append({"type": "text", "text": text})
        elif block_type == "tool_use":
            blocks.append(
                {
                    "type": "tool_use",
                    "id": getattr(block, "id", ""),
                    "name": getattr(block, "name", ""),
                    "input": getattr(block, "input", {}) or {},
                }
            )
        elif block_type == "thinking":
            thinking_block: dict[str, Any] = {
                "type": "thinking",
                "thinking": getattr(block, "thinking", ""),
            }
            signature = getattr(block, "signature", None)
            if signature is not None:
                thinking_block["signature"] = signature
            blocks.append(thinking_block)
        elif hasattr(block, "model_dump"):
            blocks.append(block.model_dump(exclude_none=True))
        else:
            text = str(block)
            if not text.strip():
                continue
            blocks.append({"type": "text", "text": text})
    return blocks


def checked_nonempty(normalized: list[Any]) -> list[Any]:
    """Return the normalized conversation, raising when nothing is left to send.

    Every transport drops whitespace-only messages before a request; when
    that leaves no message at all the API would reject the request with an
    opaque error, so the transports raise this one instead.

    Args:
        normalized: The conversation after the transport's normalization.

    Returns:
        *normalized* unchanged.

    Raises:
        KISSError: When *normalized* is empty.
    """
    if not normalized:
        raise KISSError(
            "Cannot generate response: all messages have whitespace-only content "
            "that was filtered out. At least one message with non-whitespace "
            "content is required."
        )
    return normalized


def text_of_parts(parts: list[Any]) -> str:
    """Join the ``text`` of the ``text``-typed parts of a content array.

    Serves both Chat Completions content parts and normalized Anthropic
    content blocks: the ``text`` of every ``{"type": "text", ...}`` dict
    is concatenated and anything else is skipped.

    Args:
        parts: Content parts (dicts with a ``type``).

    Returns:
        The concatenated text.
    """
    return "".join(
        str(part.get("text", "") or "")
        for part in parts
        if isinstance(part, dict) and part.get("type") == "text"
    )


def flatten_content_to_text(content: Any) -> str:
    """Flatten a message ``content`` value into plain text.

    Used by the stateless CLI-backed models (Claude Code, Codex) whose
    prompt is a single text block.  Native content is a plain string, but a
    conversation handed off from another provider's model (e.g. via the
    Sorcar ``set_model`` tool) may carry Anthropic content-block lists or
    OpenAI content-part lists; interpolating those verbatim would dump raw
    dict reprs (including base64 payloads) into the prompt.

    ``text`` block text is kept, ``tool_use`` blocks are rendered as
    ``[Tool Call] name(args)``, ``tool_result`` blocks as their text
    payload, and hidden/binary blocks (``thinking``, images, documents,
    audio) are dropped or replaced by a short placeholder.

    Args:
        content: The message content (string, block list, or other).

    Returns:
        str: The flattened text.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return "" if content is None else str(content)
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            parts.append(str(block))
            continue
        block_type = block.get("type")
        if block_type in ("thinking", "redacted_thinking"):
            continue
        if block_type == "text":
            parts.append(block.get("text", ""))
        elif block_type == "tool_use":
            parts.append(
                f"[Tool Call] {block.get('name', '')}({json.dumps(block.get('input') or {})})"
            )
        elif block_type == "tool_result":
            parts.append(flatten_content_to_text(block.get("content")))
        elif block_type in ("image", "image_url", "document", "file", "input_audio"):
            parts.append(f"[{block_type} attachment omitted]")
    return "\n".join(p for p in parts if p)


def merge_system_texts(configured: str | None, messages: list[dict[str, Any]]) -> str | None:
    """Merge a configured system instruction with conversation system messages.

    Providers whose request carries the system prompt out-of-band (the
    Anthropic ``system`` parameter, Gemini's ``system_instruction`` config
    field) cannot send OpenAI-style ``role="system"`` messages — which
    enter the conversation when it is handed off from an OpenAI-schema
    model, e.g. via the Sorcar ``set_model`` tool — so their text is
    hoisted here instead.  The configured instruction comes first,
    conversation system messages follow in order, duplicates of an
    already-collected text are skipped (compared with any
    :data:`SYSTEM_CACHE_BREAK` marker removed, since a conversation
    handed off from another adapter holds the stripped copy of the
    configured instruction), and content-part lists contribute only
    their ``text`` parts.

    Args:
        configured: The ``model_config["system_instruction"]`` value, if any.
        messages: The conversation messages to scan for ``role="system"``.

    Returns:
        The merged instruction joined with blank lines, or ``None`` when
        neither source contributed any text.
    """
    system_texts: list[str] = [configured] if configured else []
    seen = {strip_system_cache_break(text) for text in system_texts}
    for msg in messages:
        if msg.get("role") != "system":
            continue
        content = msg.get("content")
        if isinstance(content, list):
            content = text_of_parts(content)
        if not isinstance(content, str) or not content.strip():
            continue
        stripped = strip_system_cache_break(content)
        if stripped not in seen:
            seen.add(stripped)
            system_texts.append(content)
    return "\n\n".join(system_texts) if system_texts else None


# Boundary, inside a system instruction, between the prefix that is
# byte-identical across runs (framework prompt, tool guidance, repository
# conventions) and a per-run tail (work directory, task settings,
# deadlines).  The Anthropic adapter sends the two halves as separate
# ``system`` blocks with an explicit prompt-cache breakpoint on the prefix,
# so every task and sub-agent sharing that prefix reads it from the cache
# instead of paying a full cache write on its first call; every other
# adapter strips the marker.  Callers put it on its own line:
# ``prefix + "\n" + SYSTEM_CACHE_BREAK + "\n" + tail``.
SYSTEM_CACHE_BREAK = "<!-- kiss:system-cache-break -->"


def split_system_cache_break(text: str) -> tuple[str, str]:
    """Split a system instruction at its first :data:`SYSTEM_CACHE_BREAK`.

    Args:
        text: The system instruction, with or without markers.

    Returns:
        ``(prefix, tail)``: the text before the first marker and the
        text after it with any further markers removed; ``tail`` is
        ``""`` when the text carries no marker.
    """
    prefix, _sep, tail = text.partition(SYSTEM_CACHE_BREAK)
    return prefix, tail.replace(SYSTEM_CACHE_BREAK, "")


def strip_system_cache_break(text: str) -> str:
    """Return *text* with every :data:`SYSTEM_CACHE_BREAK` marker removed.

    Args:
        text: The system instruction, with or without markers.

    Returns:
        The instruction as a single provider-neutral string.
    """
    return text.replace(SYSTEM_CACHE_BREAK, "")


# The text of the follow-up user message that carries attachments a tool
# returned, so the model knows where the bytes came from.
TOOL_RESULT_ATTACHMENT_NOTE = "[attachments from previous tool result(s)]"

# model_config keys the framework itself consumes: they configure this
# framework's behaviour (routing, watchdogs, caching policy, the system
# prompt) and are never request parameters of any provider, so every
# adapter strips them before shaping a request.
FRAMEWORK_ONLY_CONFIG_KEYS = frozenset(
    {
        "system_instruction",
        "use_responses_api",
        "stream_stall_timeout",
        "enable_cache",
        # Whether a turn streams follows from whether a token callback is
        # attached, so every adapter sets this itself.
        "stream",
        # Working directory for CLI-backed agent models (cc/*, codex/*):
        # their subprocess runs with this cwd so native tools act on the
        # task's work tree, not the daemon's.  API adapters ignore it.
        "work_dir",
    }
)


def accepted_request_params(method: Any) -> frozenset[str]:
    """Return the keyword parameter names an SDK request method declares.

    The provider SDKs are generated from the vendors' OpenAPI specs and
    declare every request parameter explicitly with **no** ``**kwargs``,
    so their signatures are the authoritative list of what a request may
    carry — and stay correct across SDK upgrades, unlike a hand-written
    table.

    Args:
        method: The unbound request function of an SDK resource class
            (e.g. ``openai.resources.responses.Responses.create``), so the
            answer does not depend on a client having been built yet.

    Returns:
        The declared parameter names, without ``self``.
    """
    return frozenset(inspect.signature(method).parameters) - {"self"}


def environ_without(names: Collection[str]) -> dict[str, str]:
    """Return a copy of ``os.environ`` without the variables in *names*.

    Copied key by key with ``get``: a settings save on another thread
    may delete a variable between the key listing and its lookup, which
    would make ``dict(os.environ)`` raise.

    Args:
        names: The variable names to leave out.

    Returns:
        The environment copy, for a CLI child.
    """
    env: dict[str, str] = {}
    for name in list(os.environ):
        value = os.environ.get(name)
        if value is not None and name not in names:
            env[name] = value
    return env


class Model(ABC):
    """Abstract base class for LLM provider implementations."""

    #: Whether :meth:`native_tools_exposed` serves the tools to the model
    #: (Codex, over MCP).  The agent builds the native tool set and its
    #: executor only for a model that does.
    serves_tools_natively = False

    @contextlib.contextmanager
    def native_tools_exposed(
        self,
        tools_schema: list[dict[str, Any]],
        execute: Callable[[dict[str, Any]], tuple[str, str]],
    ) -> Iterator[None]:
        """Make the KISS tools natively callable by the model within the block.

        The default does nothing: most models receive the tools in the
        request itself.  A model whose backend is an agent that accepts
        external tool servers (Codex, via MCP) overrides this to serve
        *tools_schema* through *execute* for the duration of the block;
        :class:`~kiss.core.kiss_agent.KISSAgent` wraps every
        ``generate_and_process_with_tools`` step in it.

        Args:
            tools_schema: The tools in OpenAI function-calling layout
                (``_build_openai_tools_schema``).
            execute: Called with ``{"name": ..., "arguments": {...}}``
                for each tool call; returns ``(name, result_text)``.
        """
        yield

    def __init__(
        self,
        model_name: str,
        model_config: dict[str, Any] | None = None,
        token_callback: TokenCallback | None = None,
        thinking_callback: ThinkingCallback | None = None,
    ):
        """Initialize a Model instance.

        Args:
            model_name: The name/identifier of the model.
            model_config: Optional dictionary of model configuration parameters.
            token_callback: Optional callback invoked with each streamed text token.
            thinking_callback: Optional callback invoked with ``True`` when a
                thinking block starts and ``False`` when it ends.  Used by
                printers to switch between thinking and text display modes.
        """
        self.model_name = model_name
        self.model_config = model_config or {}
        self.token_callback = token_callback
        self.thinking_callback = thinking_callback
        self.usage_info_for_messages: str = ""
        self.conversation: list[Any] = []
        self.client: Any = None
        # The inputs ``client`` was built from; see _reuse_client.
        self._client_inputs: tuple[Any, ...] | None = None
        # A complete, billed response the adapter refused to use
        # (truncated, incomplete, refused); see take_partial_usage_response.
        self._rejected_response: Any = None
        # Seconds of event-level silence tolerated before a streamed
        # request is aborted with a retryable TimeoutError.  Read here
        # once for every streaming transport (the key is in
        # FRAMEWORK_ONLY_CONFIG_KEYS, so no adapter forwards it).
        self._stream_stall_timeout = float(
            self.model_config.get("stream_stall_timeout", DEFAULT_STREAM_STALL_TIMEOUT)
        )
        # Whether a thinking block is currently open, so an aborted
        # stream can close it (see _close_thinking_if_open).
        self._thinking_open = False
        # model_config keys already reported as unsupported, so a long run
        # is told once rather than on every step.
        self._reported_unsupported_config_keys: set[str] = set()

    def _reuse_client(self, inputs: tuple[Any, ...], build: Callable[[], Any]) -> None:
        """Keep the SDK client unless the inputs it was built from changed.

        Every SDK client owns an httpx connection pool, and ``KISSAgent``
        reuses the adapter and calls ``initialize()`` per run, so the
        client is built once per set of *inputs* and reused: rebuilding
        it every time would abandon a pool (and its TLS handshakes) to
        the garbage collector.  When *inputs* changed (a new key, endpoint
        or header between turns) the client is rebuilt with *build* and
        the replaced one closed so its pool is released at once.

        Args:
            inputs: Everything *build* reads (API key, endpoint, headers…).
            build: The zero-argument SDK client constructor.
        """
        if self.client is not None and inputs == self._client_inputs:
            return
        replaced = self.client
        self.client = build()
        self._client_inputs = inputs
        if replaced is not None:
            replaced.close()

    def system_instruction_text(self) -> str | None:
        """Return ``model_config["system_instruction"]`` as provider-neutral text.

        Strips every :data:`SYSTEM_CACHE_BREAK` marker; only the
        Anthropic adapter, which turns the marker into a prompt-cache
        breakpoint, reads the raw value.

        Returns:
            The system instruction, or ``None`` when none is configured.
        """
        system_instruction = self.model_config.get("system_instruction")
        if not system_instruction:
            return None
        return strip_system_cache_break(system_instruction)

    def _keep_supported_request_params(
        self,
        params: dict[str, Any],
        supported: Collection[str],
        provider: str,
    ) -> dict[str, Any]:
        """Return the ``model_config`` params *provider* accepts, reporting the rest.

        The one policy every adapter applies to the public
        :attr:`model_config` dict: keys the framework consumes are popped
        by the adapter, portable keys are translated to the provider's own
        spelling, and whatever is left is forwarded *iff* the provider
        declares it.  A key the provider does not declare is dropped with
        a warning naming both — never forwarded into a keyword-only SDK
        call (a ``TypeError`` that is neither retryable nor actionable),
        and never discarded in silence.

        The warning is emitted once per key per model instance.

        Args:
            params: The candidate request parameters, already translated.
            supported: The parameter names the provider accepts, taken
                from the SDK itself (see :func:`accepted_request_params`).
            provider: Human-readable transport name used in the warning.

        Returns:
            The subset of *params* the provider accepts.
        """
        kept = {key: value for key, value in params.items() if key in supported}
        dropped = sorted(
            key
            for key in params
            if key not in kept and key not in self._reported_unsupported_config_keys
        )
        if dropped:
            self._reported_unsupported_config_keys.update(dropped)
            logger.warning(
                "%s does not support model_config key(s) %s; ignoring them.",
                provider,
                ", ".join(dropped),
            )
        return kept

    def _invoke_token_callback(self, token: str) -> None:
        """Invoke the token callback synchronously."""
        if self.token_callback is not None:
            self.token_callback(token)

    def _invoke_thinking_callback(self, is_start: bool) -> None:
        """Invoke the thinking callback synchronously.

        Args:
            is_start: ``True`` when a thinking block starts, ``False`` when it ends.
        """
        self._thinking_open = is_start
        if self.thinking_callback is not None:
            self.thinking_callback(is_start)

    def _close_thinking_if_open(self) -> None:
        """Close an open thinking block, if there is one.

        A stream that is aborted mid-thinking — a user stop, or a stall —
        never reaches the loop's own closing callback, and the printer
        would then render everything afterwards as "thinking" forever.
        Safe to call at any time: it is a no-op when no block is open.
        """
        if self._thinking_open:
            self._invoke_thinking_callback(False)

    def _open_thinking_if_closed(self) -> None:
        """Open a thinking block unless one is already open.

        The streaming loops call this on every reasoning delta, so a
        run of deltas produces exactly one ``thinking_callback(True)``.
        """
        if not self._thinking_open:
            self._invoke_thinking_callback(True)

    @contextlib.contextmanager
    def _watched_events(self, stream: Any, name: str) -> Iterator[Iterator[Any]]:
        """Iterate *stream* under the stop/stall watchdog for one turn.

        Yields the :func:`~kiss.core.models.stream_abort.stop_aware_events`
        generator over *stream* (stall tolerance ``stream_stall_timeout``),
        and however the block ends — normally, by a stop, a stall, a
        transport error or a token callback that raised — closes that
        generator and then any open thinking bracket.

        Closing the generator explicitly is what makes the watchdog safe:
        an abandoned generator runs its cleanup only when the traceback
        holding its frame is released, and until then its daemon thread
        stays armed over a connection that never returns to the pool.

        The bracket is closed here, not after the loop, because the
        ``finally`` is the one exit every outcome shares: a stop, a stall
        and every transport failure all leave the ``with`` body through
        it, and ``KISSAgent`` retries transport failures in the SAME run
        without resetting the printer — which would otherwise render the
        retry's answer as reasoning.  A no-op when the turn ended outside
        a reasoning block.

        Args:
            stream: The SDK stream object to iterate and, if needed, abort.
            name: Watchdog thread name, for readable stack dumps.

        Yields:
            The events of *stream*.
        """
        events = stop_aware_events(stream, stall_timeout=self._stream_stall_timeout, name=name)
        try:
            yield events
        finally:
            # Nested so the bracket closes even when closing the generator
            # itself raises (a Stop delivered during teardown).
            try:
                events.close()
            finally:
                self._close_thinking_if_open()

    def reset_conversation(self) -> None:
        """Reset conversation state for reuse across sub-sessions.

        Clears the conversation history, the usage info and the thinking
        bracket, while keeping the HTTP client and model configuration
        intact.

        The bracket has to go with them: ``KISSAgent._reset`` reuses the
        same adapter instance across runs and binds a *new* printer to
        it, so a block left open by a stream that was aborted
        mid-thinking would make the next task's first thought skip its
        opening ``thinking_callback(True)`` — rendering reasoning as
        ordinary assistant text — or emit a spurious closing ``False``
        into a printer that never opened one.
        """
        self.conversation = []
        self.usage_info_for_messages = ""
        self._thinking_open = False

    def rebind_callbacks(
        self,
        token_callback: TokenCallback | None,
        thinking_callback: ThinkingCallback | None,
    ) -> None:
        """Point this model's streaming callbacks at a new printer.

        The hook ``KISSAgent._reset`` calls when it reuses an existing
        adapter instance for a new run.  Only the two attributes are
        rebound; a transport that delegates a turn to a cached sub-model
        (``OpenAICompatibleModel``'s Responses delegate) copies them to
        that delegate at the start of every delegated turn, so the
        delegate never streams into the *previous* run's printer.

        Args:
            token_callback: The new printer's token callback, or ``None``.
            thinking_callback: The new printer's thinking callback, or
                ``None``.
        """
        self.token_callback = token_callback
        self.thinking_callback = thinking_callback

    def _replace_last_assistant_with_tool_calls(
        self, content: str, function_calls: list[dict[str, Any]]
    ) -> None:
        """Replace the last assistant message with one that includes tool calls.

        Used by the text-based tool calling of
        :meth:`generate_and_process_with_tools` (the CLI-backed models),
        where ``generate()`` already appended a plain assistant message and it
        needs to be upgraded to include parsed tool call metadata.

        Args:
            content: The full text content of the assistant message.
            function_calls: Parsed tool calls, each with ``id``, ``name``,
                and ``arguments`` (dict).
        """
        self.conversation[-1] = {
            "role": "assistant",
            "content": content,
            "tool_calls": [
                _chat_tool_call_entry(fc["id"], fc["name"], json.dumps(fc["arguments"]))
                for fc in function_calls
            ],
        }

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self.model_name})"

    __repr__ = __str__

    @abstractmethod
    def initialize(self, prompt: str, attachments: list[Attachment] | None = None) -> None:
        """Initializes the conversation with an initial user prompt.

        Args:
            prompt: The initial user prompt to start the conversation.
            attachments: Optional list of file attachments (images, PDFs, audio,
                video) to include. Provider support varies — unsupported types
                are skipped with a warning.
        """
        pass  # pragma: no cover

    def generate(self) -> tuple[str, Any]:
        """Generates content from the current conversation without tools.

        The default is a tool-less :meth:`generate_and_process_with_tools`
        turn; adapters whose tool-less request differs override it.

        Returns:
            tuple[str, Any]: A tuple of (generated_text, raw_response).
        """
        _, content, response = self.generate_and_process_with_tools({})
        return content, response

    @abstractmethod
    def generate_and_process_with_tools(
        self,
        function_map: dict[str, Callable[..., Any]],
        tools_schema: list[dict[str, Any]] | None = None,
    ) -> tuple[list[dict[str, Any]], str, Any]:
        """Generates content with tools, processes the response, and adds it to conversation.

        Args:
            function_map: Dictionary mapping function names to callable functions.
            tools_schema: Optional pre-built tool schema list. When provided,
                skips schema rebuilding from function_map (performance optimization).

        Returns:
            tuple[list[dict[str, Any]], str, Any]: A tuple of
                (function_calls, response_text, raw_response).
        """
        pass  # pragma: no cover

    def keep_prompt_cache_warm(
        self,
        function_map: dict[str, Callable[..., Any]],
        tools_schema: list[dict[str, Any]] | None = None,
    ) -> Any | None:
        """Touch the provider's prompt cache so it survives a long tool call.

        Called from a background thread while a tool call that may outlive
        the cache TTL is running (see
        :mod:`kiss.core.prompt_cache_keepalive`).  The conversation ends
        with the assistant turn whose tool calls are in flight and must
        not be modified.  Providers whose cache outlives the longest tool
        call, or that have none, return ``None`` (this default).

        Args:
            function_map: The agent's tools, exactly as passed to
                :meth:`generate_and_process_with_tools`.
            tools_schema: The pre-built tool schema list, exactly as passed
                to :meth:`generate_and_process_with_tools`.

        Returns:
            The raw provider response of the ping (a billed request the
            caller accounts for), or ``None`` when no request was sent.
        """
        return None

    def _find_tool_call_ids_from_last_assistant(self) -> list[tuple[str, str]]:
        """Find tool call (name, id) pairs from the last assistant message.

        Searches backwards through the conversation for the most recent
        assistant message containing tool calls and extracts their IDs.
        Also recognises trailing OpenAI Responses-API ``function_call``
        items (present when the conversation was handed off from an
        :class:`~kiss.core.models.openai_compatible_model2.OpenAICompatibleModel2`,
        e.g. via the Sorcar ``set_model`` tool), skipping any that already
        received a ``function_call_output``.

        Returns:
            list[tuple[str, str]]: A list of (function_name, tool_call_id) tuples,
                or an empty list if no assistant message with tool calls is found.
        """
        answered: set[str] = set()
        native: list[tuple[str, str]] = []
        for msg in reversed(self.conversation):
            if not isinstance(msg, dict):
                continue
            itype = msg.get("type")
            if itype == "function_call_output":
                answered.add(str(msg.get("call_id", "")))
                continue
            if itype == "function_call":
                call_id = str(msg.get("call_id", ""))
                if call_id not in answered:
                    native.append((str(msg.get("name", "")), call_id))
                continue
            if itype == "reasoning":
                continue
            if native and msg.get("role") is not None:
                break
            if msg.get("role") == "assistant":
                ids = list(iter_assistant_tool_calls(msg))
                if ids:
                    return ids
                break
        return list(reversed(native))

    def tool_result_text_and_attachments(
        self, result_dict: dict[str, Any]
    ) -> tuple[str, list[Attachment]]:
        """Split one tool result into provider-safe text and its attachments.

        The three steps every adapter needs, in the one order that works:
        JSON-encode a structured result (a regex cannot be run over a dict),
        lift any binary payload out of the sentinel the tool embedded it in,
        and append the usage line the framework shows the model.

        Args:
            result_dict: One tool's result payload, as passed to
                :meth:`add_function_results_to_conversation_and_return`.

        Returns:
            The text to put in the tool message, and the attachments the
            tool produced (empty when it produced none).
        """
        text, attachments = parse_binary_attachments(_tool_result_to_string(result_dict))
        if self.usage_info_for_messages:
            text = f"{text}\n\n{self.usage_info_for_messages}"
        return text, attachments

    @staticmethod
    def tool_result_call_id(
        result_dict: dict[str, Any],
        index: int,
        tool_calls: list[tuple[str, str]],
        func_name: str,
        prefix: str = "call",
    ) -> str:
        """Return the id of the tool call one result answers.

        An explicit ``tool_use_id`` in the result wins — it is part of the
        public signature of
        :meth:`add_function_results_to_conversation_and_return`, so it is
        honoured by every adapter rather than by one.  Otherwise results are
        matched to the last assistant message's tool calls by position, and
        an id is synthesised when there is nothing to match.

        Args:
            result_dict: One tool's result payload.
            index: Its position in the result list.
            tool_calls: ``(name, id)`` pairs of the calls being answered.
            func_name: The tool's name, used in a synthesised id.
            prefix: Prefix of a synthesised id (Anthropic uses ``toolu``).

        Returns:
            The tool-call id to answer.
        """
        explicit = result_dict.get("tool_use_id")
        if explicit:
            return str(explicit)
        if index < len(tool_calls):
            return tool_calls[index][1]
        return f"{prefix}_{func_name}_{index}"

    def _deliver_tool_result_attachments(self, attachments: list[Attachment]) -> None:
        """Make bytes a tool returned visible to the model.

        A tool result is a string, so an image or audio clip a tool produced
        cannot live inside the tool message on most transports.  The default
        is to append a follow-up ``user`` message carrying the attachments,
        which every adapter that renders an ``attachments`` key turns into
        real image / file / audio content parts.

        A transport that cannot carry bytes at all must override this and
        *say so* — to the log and to the model — instead of dropping them.

        Args:
            attachments: The attachments lifted out of the tool results.
        """
        self.conversation.append(
            {
                "role": "user",
                "content": TOOL_RESULT_ATTACHMENT_NOTE,
                "attachments": list(attachments),
            }
        )

    def add_function_results_to_conversation_and_return(
        self, function_results: list[tuple[str, dict[str, Any]]]
    ) -> None:
        """Adds function results to the conversation state.

        Each result becomes one ``tool`` message, answering the tool call
        named by its ``tool_use_id`` or matched by position against the last
        assistant message.  Binary payloads are lifted out of the text and
        handed to :meth:`_deliver_tool_result_attachments`, so what a tool
        returned actually reaches the model.

        Args:
            function_results: List of tuples containing (function_name, result_dict).
        """
        tool_calls = self._find_tool_call_ids_from_last_assistant()
        pending: list[Attachment] = []

        for i, (func_name, result_dict) in enumerate(function_results):
            result_content, attachments = self.tool_result_text_and_attachments(result_dict)
            self.conversation.append(
                {
                    "role": "tool",
                    "tool_call_id": self.tool_result_call_id(result_dict, i, tool_calls, func_name),
                    "content": result_content,
                }
            )
            pending.extend(attachments)

        if pending:
            self._deliver_tool_result_attachments(pending)

    def add_message_to_conversation(
        self, role: str, content: str, attachments: list[Attachment] | None = None
    ) -> None:
        """Adds a message to the conversation state.

        Args:
            role: The role of the message sender (e.g., 'user', 'assistant').
            content: The message content.
            attachments: Files to carry with a ``user`` message; every
                adapter renders them as image / file / audio content
                parts, as it does for the files a tool result returns.
        """
        if role == "user" and self.usage_info_for_messages:  # pragma: no branch
            content = f"{content}\n\n{self.usage_info_for_messages}"
        message: dict[str, Any] = {"role": role, "content": content}
        if attachments:
            message["attachments"] = list(attachments)
        self.conversation.append(message)

    @abstractmethod
    def extract_input_output_token_counts_from_response(
        self, response: Any
    ) -> (
        tuple[int, int, int, int]
        | tuple[int, int, int, int, int]
        | tuple[int, int, int, int, int, int, int]
        | tuple[int, int, int, int, int, int, int, int]
        | tuple[int, int, int, int, int, int, int, int, int]
    ):
        """Extracts token counts from an API response.

        Args:
            response: The raw API response object.

        Returns:
            A 4-tuple ``(input_tokens, output_tokens, cache_read_tokens,
            cache_write_tokens)``, a 5-tuple whose last element is the
            Anthropic one-hour cache-write token count, a 7-tuple whose
            last two elements are the audio input/output token subsets
            of an OpenAI audio-chat response (billed at the model's
            separate audio rates), an 8-tuple whose last element is
            the audio subset of the cache reads (Gemini), or a 9-tuple
            whose last element is the IMAGE subset of the output
            (Gemini image models, billed at the model's image output
            rate).  Audio and image tokens are never also counted in
            the text counts.
        """
        pass  # pragma: no cover

    def extract_cost_from_response(self, response: Any) -> float | None:
        """Return the USD cost the provider itself reported for *response*.

        Providers that state what they charged for a call (OpenRouter's
        ``usage.cost``) override this; the agent bills that figure instead
        of the catalog estimate from
        :func:`kiss.core.models.model_info.calculate_cost`.  The base
        implementation knows of no such field.

        Args:
            response: The raw API response object.

        Returns:
            The billed cost in USD, or ``None`` when the response carries
            no provider-reported cost.
        """
        return None

    def cost_multiplier_for_response(self, response: Any) -> float:
        """Return the factor the served processing tier applies to catalog rates.

        Some providers price the same tokens differently by processing
        tier and state in the response which tier served the call
        (OpenAI's ``service_tier``: Flex at half the Standard rate, Fast
        at twice, Ultrafast at six times).  The agent multiplies the
        :func:`~kiss.core.models.model_info.calculate_cost` estimate by
        this factor; it is not applied to a provider-reported cost.  The
        base implementation knows of no tiers.

        Args:
            response: The raw API response object.

        Returns:
            ``1.0`` unless the provider served a differently priced tier.
        """
        return 1.0

    def take_partial_usage_response(self) -> Any:
        """Return (and consume) usage for a generation that raised, if known.

        A generation call can raise after the provider already billed it:
        adapters reject a complete response they cannot use (output
        truncated at the token limit, an ``incomplete`` Responses status,
        a safety refusal) and record it in ``self._rejected_response``
        just before raising.  Adapters that observe usage mid-stream (e.g.
        Claude Code's per-message ``message_delta`` events) override this.
        The returned object suits
        :meth:`extract_input_output_token_counts_from_response`.
        Consuming clears the stored value so the same usage is never
        counted twice.

        Returns:
            A response object carrying the billed usage, or ``None``.
        """
        response = self._rejected_response
        self._rejected_response = None
        return response

    @abstractmethod
    def get_embedding(self, text: str, embedding_model: str | None = None) -> list[float]:
        """Generates an embedding vector for the given text.

        Args:
            text: The text to generate an embedding for.
            embedding_model: Optional model name to use for embedding generation.

        Returns:
            list[float]: The embedding vector as a list of floats.
        """
        pass  # pragma: no cover

    def get_embeddings(
        self, texts: list[str], embedding_model: str | None = None
    ) -> list[list[float]]:
        """Generates one embedding vector per text.

        The base implementation calls :meth:`get_embedding` once per text;
        providers with a batch endpoint override it with a single request.

        Args:
            texts: The texts to embed.
            embedding_model: Optional model name to use for embedding generation.

        Returns:
            The embedding vectors, in the order of *texts*.
        """
        return [self.get_embedding(text, embedding_model) for text in texts]

    def set_usage_info_for_messages(self, usage_info: str) -> None:
        """Sets token information to append to messages sent to the LLM.

        Args:
            usage_info: The usage information string to append.
        """
        self.usage_info_for_messages = usage_info

    def _resolve_openai_tools_schema(
        self,
        function_map: dict[str, Callable[..., Any]],
        tools_schema: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        """Return pre-built tools_schema or build one from function_map.

        Args:
            function_map: Dictionary mapping function names to callable functions.
            tools_schema: Optional pre-built tool schema list. When provided,
                returned as-is (skips schema rebuilding for performance).

        Returns:
            list[dict[str, Any]]: The resolved OpenAI-format tool schema list.
        """
        return (
            tools_schema
            if tools_schema is not None
            else self._build_openai_tools_schema(function_map)
        )

    def _build_openai_tools_schema(
        self, function_map: dict[str, Callable[..., Any]]
    ) -> list[dict[str, Any]]:
        """Builds the OpenAI-compatible tools schema from a function map.

        Args:
            function_map: Dictionary mapping function names to callable functions.

        Returns:
            list[dict[str, Any]]: A list of tool schemas in OpenAI format.
        """
        return [self._function_to_openai_tool(func) for func in function_map.values()]

    def _function_to_openai_tool(self, func: Callable[..., Any]) -> dict[str, Any]:
        """Converts a Python function to an OpenAI tool schema.

        Args:
            func: The Python function to convert.

        Returns:
            dict[str, Any]: The tool schema in OpenAI format.
        """
        description, doc, parameters = _tool_summary_and_params(func, func.__name__)
        param_descriptions = _parse_docstring_params(doc)

        # Under ``from __future__ import annotations`` (PEP 563) the
        # signature's raw annotations are strings like ``"int"``, which
        # would fall through the type mapping to ``{"type": "string"}``.
        # Resolve them to real types so int/bool/float parameters keep
        # their proper JSON-schema types.
        try:
            hints = typing.get_type_hints(func)
        except Exception:
            hints = {}

        properties: dict[str, Any] = {}
        required: list[str] = []

        for param_name, param in parameters.items():
            param_type = hints.get(param_name, param.annotation)
            param_schema = _python_type_to_json_schema(param_type)

            if param_name in param_descriptions:
                param_schema["description"] = param_descriptions[param_name]

            properties[param_name] = param_schema

            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        return {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


def _parse_docstring_params(docstring: str) -> dict[str, str]:
    """Parses parameter descriptions from a Google-style ``Args:`` section.

    An entry is ``name: text`` or ``name (type): text`` at the
    section's base indentation; lines indented deeper continue the
    entry and are joined with single spaces, so a description that
    wraps over several lines reaches the model whole.  A blank line
    or a ``Returns:`` / ``Raises:`` / ``Example:`` heading ends the
    section.

    Args:
        docstring: The docstring to parse.

    Returns:
        dict[str, str]: A dictionary mapping parameter names to descriptions.
    """
    param_descriptions: dict[str, str] = {}
    in_args_section = False
    base_indent: int | None = None
    current: str | None = None

    for line in docstring.split("\n"):
        stripped = line.strip()
        if stripped.lower().startswith("args:"):
            in_args_section, base_indent, current = True, None, None
            continue
        if stripped.lower().startswith(("returns:", "raises:", "example:")) or (
            in_args_section and not stripped
        ):
            in_args_section = False
            continue
        if not in_args_section:
            continue
        indent = len(line) - len(line.lstrip())
        if base_indent is None:
            base_indent = indent
        if indent > base_indent and current is not None:
            param_descriptions[current] += " " + stripped
        elif ":" in stripped:
            param_part, desc_part = (part.strip() for part in stripped.split(":", 1))
            current = param_part.split("(")[0].strip()
            param_descriptions[current] = desc_part

    return param_descriptions


def _python_type_to_json_schema(python_type: Any) -> dict[str, Any]:
    """Converts a Python type annotation to a JSON schema type.

    Args:
        python_type: The Python type annotation to convert.

    Returns:
        dict[str, Any]: The JSON schema type definition.
    """
    if python_type is inspect.Parameter.empty:
        return {"type": "string"}

    origin = get_origin(python_type)
    args = get_args(python_type)

    if origin is Union or origin is types_module.UnionType:
        non_none_args = [a for a in args if a is not type(None)]
        if len(non_none_args) == 1:
            return _python_type_to_json_schema(non_none_args[0])
        return {"anyOf": [_python_type_to_json_schema(a) for a in non_none_args]}

    if origin is list:
        if args:
            return {"type": "array", "items": _python_type_to_json_schema(args[0])}
        return {"type": "array"}

    if origin is dict:
        return {"type": "object"}

    type_mapping: dict[type, dict[str, str]] = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        type(None): {"type": "null"},
    }

    if python_type in type_mapping:
        return type_mapping[python_type]

    return {"type": "string"}


class CLITextModel(Model):
    """Base class for the stateless CLI-backed models (Claude Code, Codex).

    Both CLIs are coding agents in their own right, but KISS drives them
    as plain language models: every turn runs the CLI with its native
    tools, MCP servers, hooks and project instructions switched off and
    KISS's own system prompt installed in place of the CLI's (``claude
    --tools "" --system-prompt`` and ``codex exec`` with its tool
    features disabled and ``model_instructions_file``), so the KISS tool
    loop — and only it — acts on the user's behalf.

    Both transports flatten the conversation into a single text prompt,
    support tool calling only via the text-based ``tool_calls`` JSON
    protocol, and cannot produce embeddings.  That shared plumbing lives
    here; subclasses implement the subprocess invocation and stream
    parsing.
    """

    _cli_model_name = "CLITextModel"
    _cli_logger = logger

    # Set for the duration of a tool-bearing turn by
    # :class:`_ToolCallFilteredStream`; adapters that can end such a turn
    # early at the first complete ``tool_calls`` block consult it.
    _tool_bearing_turn = False

    def _build_prompt(self) -> str:
        """Return the conversation as the prompt sent to the CLI.

        A one-message conversation is the task text itself; a multi-turn
        conversation is flattened into a ``[User]/[Assistant]/[Tool
        Result]`` transcript.  The system instruction never rides in the
        prompt: each adapter hands it to its CLI as the real system
        prompt (see :meth:`_build_cli_args` of the subclasses).

        Returns:
            The prompt text, without any system instruction.
        """
        if len(self.conversation) == 1:
            return flatten_content_to_text(self.conversation[0]["content"])
        return self._conversation_as_dialogue()

    def initialize(self, prompt: str, attachments: list[Attachment] | None = None) -> None:
        """Initialize the conversation with an initial user prompt.

        Args:
            prompt: The initial user prompt.
            attachments: Not supported — ignored with a warning if provided.
        """
        if attachments:  # pragma: no cover – attachments not used in practice
            self._cli_logger.warning(
                f"{self._cli_model_name} does not support attachments; they will be ignored."
            )
        self.conversation = [{"role": "user", "content": prompt}]

    def _deliver_tool_result_attachments(self, attachments: list[Attachment]) -> None:
        """State that this transport cannot show the bytes a tool returned.

        The CLI is driven by a single text prompt and has no content-part
        shape, so a screenshot or audio clip a tool produced cannot be
        handed to the model.  Saying so — in the log and in the prompt
        itself — is what keeps the model from reasoning about a picture it
        never received, which is what silently dropping the bytes did.

        Args:
            attachments: The attachments lifted out of the tool results.
        """
        described = ", ".join(f"{att.mime_type} ({len(att.data)} bytes)" for att in attachments)
        self._cli_logger.warning(
            "%s cannot show attachments to the model; %s from the tool result were not sent.",
            self._cli_model_name,
            described,
        )
        self.conversation.append(
            {
                "role": "user",
                "content": (
                    f"[{len(attachments)} attachment(s) in the previous tool "
                    f"result could not be shown to this model: {described}]"
                ),
            }
        )

    def _conversation_as_dialogue(self) -> str:
        """Render the conversation as a ``[User]/[Assistant]/[Tool Result]`` transcript.

        The CLI transports are stateless across invocations, so multi-turn
        conversations are flattened into a single text block.  Tool-result
        messages (``role == "tool"``) are rendered as ``[Tool Result]: …``.

        Returns:
            The assembled transcript string.
        """
        parts: list[str] = []
        for msg in self.conversation:
            role = msg["role"]
            content = flatten_content_to_text(msg.get("content", ""))
            if role == "user":
                parts.append(f"[User]: {content}")
            elif role == "assistant":
                parts.append(f"[Assistant]: {content}")
            elif role == "tool":
                parts.append(f"[Tool Result]: {content}")
        return "\n\n".join(parts)

    @contextlib.contextmanager
    def _tools_prompt_installed(
        self, function_map: dict[str, Callable[..., Any]]
    ) -> Iterator[None]:
        """Run one text-based tool turn with the tools prompt installed.

        A *copy* of ``model_config`` whose ``system_instruction`` carries
        the tools prompt is installed, the stream callbacks are wrapped by
        :class:`_ToolCallFilteredStream` (which streams the assistant text
        with the raw ``tool_calls`` block held out of it), and on every
        exit the callbacks are restored first and the original config
        last.

        Args:
            function_map: Dictionary mapping function names to callables.
        """
        tools_prompt = _build_text_based_tools_prompt(function_map)
        original_config = self.model_config
        config = dict(original_config)
        original_system = config.get("system_instruction", "")
        config["system_instruction"] = (original_system + "\n\n" + tools_prompt).strip()
        self.model_config = config
        try:
            with _ToolCallFilteredStream(self):
                yield
        finally:
            self.model_config = original_config

    def generate_and_process_with_tools(
        self,
        function_map: dict[str, Callable[..., Any]],
        tools_schema: list[dict[str, Any]] | None = None,
    ) -> tuple[list[dict[str, Any]], str, Any]:
        """Generate one turn with text-based tool calling.

        The tool descriptions are injected into the system prompt (the
        CLI's own tools are switched off, so these are the only tools the
        model can use), the turn runs through :meth:`generate`, and the
        model's text is parsed for JSON ``tool_calls`` blocks, which are
        returned to the framework for execution.  Thinking tokens and
        text stream to the callbacks as they arrive, minus the raw
        ``tool_calls`` JSON, so the block is never rendered in the chat
        panel.  Adapters shape the turn through :meth:`_text_tools` and
        :meth:`_tool_call_text`.

        Args:
            function_map: Dictionary mapping function names to callable functions.
            tools_schema: Ignored — text-based tool calling builds its own prompt.

        Returns:
            Tuple of ``(function_calls, content, response)``.
        """
        with self._tools_prompt_installed(self._text_tools(function_map)):
            content, response = self.generate()
            # Leaving the block flushes held text to the token callback,
            # which raises KeyboardInterrupt on Stop: the completed turn is
            # then billed through take_partial_usage_response().
            self._rejected_response = response
        self._rejected_response = None
        function_calls = _parse_text_based_tool_calls(self._tool_call_text(content))
        if function_calls:
            self._replace_last_assistant_with_tool_calls(content, function_calls)
        return function_calls, content, response

    def _text_tools(
        self, function_map: dict[str, Callable[..., Any]]
    ) -> dict[str, Callable[..., Any]]:
        """Return the tools the text-based prompt of a turn should describe.

        Args:
            function_map: Every tool the framework offers for the turn.
        """
        return function_map

    def _tool_call_text(self, content: str) -> str:
        """Return the text of a finished turn that is parsed for ``tool_calls``.

        Args:
            content: The turn's assistant text.
        """
        return content

    def _emit_as_thinking(self, text: str) -> None:
        """Forward *text* to the token callback wrapped in a thinking block.

        Args:
            text: The text to stream inside a thinking start/end pair.
        """
        self._invoke_thinking_callback(True)
        self._invoke_token_callback(text)
        self._invoke_thinking_callback(False)

    @contextlib.contextmanager
    def _cli_turn(self, args: list[str], label: str) -> Iterator["_CLIProcess"]:
        """Run one CLI turn: start *args*, bound it, and classify how it ended.

        The turn skeleton both adapters share.  Inside the block the
        caller sends the prompt, parses the child's stdout and checks its
        exit status; this wrapper owns the child's lifetime
        (:class:`_CLIProcess`) and the two ways a turn is cut short:

        * output silence longer than ``model_config["timeout"]`` (300 s by
          default) becomes the retryable :func:`_cli_stall_error`, never a
          ``KISSError`` that would abort the whole task;
        * a user Stop propagates as ``KeyboardInterrupt``.

        However the block ends — those two, a parser or callback error
        that ``KISSAgent`` will retry, or normally — an open thinking
        block is closed, so the printer never renders the retry, the
        "stopped" notice or the next turn's answer as reasoning.

        Args:
            args: The command line to run.
            label: Human-readable CLI name used in every error message.

        Yields:
            The running child.
        """
        timeout = self.model_config.get("timeout", 300)
        work_dir = self.model_config.get("work_dir") or None
        with _CLIProcess(
            args,
            label,
            timeout,
            cwd=work_dir,
            env=self._cli_env(),
            waiting_on_caller=self._tool_call_in_flight,
            caller_error=self._tool_call_error,
        ) as proc:
            try:
                yield proc
            except _StreamReadTimeoutError:
                raise _cli_stall_error(label, timeout) from None
            finally:
                self._close_thinking_if_open()

    def _tool_call_in_flight(self) -> bool:
        """Return whether the CLI is silent because it waits on a KISS tool.

        A CLI that calls KISS tools natively (Codex over the MCP bridge)
        produces no output while a tool runs, so its silence must not
        count as a stall.  The default — a CLI with no native tool
        channel — is never waiting on the caller.
        """
        return False

    def _tool_call_error(self) -> Exception | None:
        """Return the fatal error a KISS tool the CLI called natively raised, if any.

        The turn is ended and the error raised as soon as the wait loop
        sees it: the CLI itself would talk past the tool's error reply
        for the rest of its turn.  The default never errs.
        """
        return None

    def _cli_env(self) -> dict[str, str] | None:
        """Return the environment for the CLI child, or ``None`` to inherit.

        Adapters whose CLI reads credentials from the environment
        override this (Claude Code drops the API key so a turn bills the
        user's subscription instead).
        """
        return None

    def get_embedding(self, text: str, embedding_model: str | None = None) -> list[float]:
        """Not supported — the CLI transports do not provide embeddings.

        Raises:
            KISSError: Always.
        """
        raise KISSError(f"{self._cli_model_name} does not support embeddings.")


# Longest wait for a child that already closed stdout to exit on its own.
# Bounded by the run's own deadline, so a short timeout stays short.
_EXIT_GRACE_SECONDS = 10.0
# How often a blocked reader re-checks the thread's stop signal.
_STOP_POLL_SECONDS = 0.1


def _cli_stall_error(label: str, timeout: float) -> TimeoutError:
    """Build the retryable stall error for a CLI that stopped producing output.

    ``KISSAgent._run_agentic_loop`` re-raises every :class:`KISSError`
    immediately and retries everything else, so a transient stall must
    NOT be a ``KISSError``: that would abort the whole task where the
    equivalent Anthropic condition (``AnthropicModel._stall_error``) is
    simply retried.

    Args:
        label: Human-readable CLI name for the message.
        timeout: Seconds of output silence the run exceeded.

    Returns:
        The ``TimeoutError`` for the caller to raise.
    """
    return TimeoutError(
        f"{label} timed out: it produced no output for {timeout}s and was "
        f"killed; the run did not finish."
    )


class _StreamReadTimeoutError(Exception):
    """Internal sentinel: the CLI stdout stream stalled past the deadline."""


# The launch line of the ``<name>.cmd`` shim npm writes for a package's
# ``bin`` entry on Windows (the cmd-shim package): the target relative to
# the shim's directory, run directly or through ``node`` (``%_prog%``,
# followed by the flags of a ``#!/usr/bin/env -S node <flags>`` shebang).
_NPM_CMD_SHIM_TARGET = re.compile(r'(?:"%_prog%"((?:\s+-[^\s"]+)*)\s+)?"%dp0%\\([^"]+)"\s+%\*')


def _unwrap_npm_cmd_shim(args: list[str]) -> list[str]:
    """Replace an npm ``.cmd`` shim at ``args[0]`` with the program it runs.

    On Windows ``shutil.which("claude")`` / ``("codex")`` find the batch
    shims npm installs, and ``Popen`` runs a batch file through
    ``cmd.exe``, which re-parses the argument list by its own rules: every
    argument is cut at its first newline (so ``--append-system-prompt``
    kept only the first line of the system prompt and the arguments after
    it were dropped) and ``%VAR%`` references are expanded.  The shim's
    launch line names the real target, so start that instead and leave
    ``cmd.exe`` out of the picture.  Anything that is not such a shim is
    returned unchanged.

    Args:
        args: The command line to run.

    Returns:
        *args* with the shim replaced by ``[target]`` or
        ``[node, <flags>..., target]``.
    """
    exe = args[0]
    if not exe.lower().endswith((".cmd", ".bat")):
        return args
    try:
        match = _NPM_CMD_SHIM_TARGET.search(Path(exe).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError):
        return args
    if match is None:
        return args
    shim_dir = os.path.dirname(exe)
    target = os.path.join(shim_dir, match.group(2).replace("\\", os.sep))
    if not os.path.isfile(target):
        return args
    if match.group(1) is None:
        return [target, *args[1:]]
    # The shim prefers the node.exe installed beside it (a Node bundled
    # with npm) over the one on PATH.
    sibling = os.path.join(shim_dir, "node.exe")
    node = sibling if os.path.isfile(sibling) else shutil.which("node")
    if node is None:
        return args
    return [node, *match.group(1).split(), target, *args[1:]]


class _CLIProcess:
    """A CLI child process whose whole lifetime is bounded by a ``with`` block.

    Shared by both CLI-backed adapters, because both need four
    guarantees that :class:`subprocess.Popen` does not give for free:

    * **stderr is drained continuously.**  A CLI that fills the 64 KiB
      stderr pipe with progress or deprecation noise otherwise blocks in
      ``write(2)`` and stops writing stdout too, which is indistinguishable
      from a model stall and costs the entire timeout.
    * **writing the prompt cannot kill the caller.**  Agentic prompts are
      the whole flattened dialogue and routinely exceed the pipe buffer,
      so a child that rejects its arguments and exits breaks the pipe
      mid-write.  The useful diagnosis is the child's exit status and
      stderr, not ``BrokenPipeError``.
    * **stdout is read with a deadline and a stop signal.**  A parked
      reader cannot be rescued cooperatively, so the reader polls and
      raises instead of blocking forever (see :mod:`kiss.core.stop_signal`).
    * **the child is always reaped and its pipes released.**  Leaving
      that to ``Popen.__del__`` leaks a zombie and three descriptors per
      timed-out call, which a long-running daemon eventually pays for
      with ``OSError: [Errno 24] Too many open files``.  The prompt pipe
      is released by :meth:`close` unconditionally: its writer is a
      cancellable non-blocking loop, so a grandchild that inherited the
      pipe and never reads it cannot keep the thread — or the descriptor
      — alive.  An output pipe is released once its reader has seen EOF;
      a grandchild that inherited stdout or stderr delays that until it
      exits (a reader parked in ``read(2)`` cannot be cancelled without
      closing the descriptor under it).

    The reader threads deliberately do nothing but move bytes: parsing
    (and therefore every token/thinking callback) happens on the calling
    thread, so an abandoned reader can never emit a dead run's output
    into the next run's printer.
    """

    def __init__(
        self,
        args: list[str],
        label: str,
        timeout: float,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        waiting_on_caller: Callable[[], bool] | None = None,
        caller_error: Callable[[], Exception | None] | None = None,
    ) -> None:
        """Start *args* and begin draining both of its output pipes.

        Args:
            args: The command line to run.
            label: Human-readable CLI name used in error messages.
            timeout: Seconds of output silence tolerated before the run is
                aborted.  The deadline is refreshed every time the child
                produces a line of stdout (see :meth:`lines`), so a busy
                agentic run is never cut off mid-work while a stalled one
                is detected within *timeout* seconds.
            cwd: Working directory for the child, or ``None`` to inherit
                the caller's.
            env: Environment for the child, or ``None`` to inherit the
                caller's.
            waiting_on_caller: Returns ``True`` while the child is silent
                because it waits on the caller (a tool call it made to the
                caller is still running); an expired deadline is then
                re-armed instead of ending the run.  ``None`` never waits.
            caller_error: Returns the error that ends the run (a tool
                call the child made to the caller hit a fatal limit), or
                ``None``; polled with the stop signal by :meth:`lines`.
                ``None`` never errs.

        Raises:
            KISSError: If the executable could not be started.
        """
        self._label = label
        self._timeout = timeout
        self._deadline = time.monotonic() + timeout
        self._waiting_on_caller = waiting_on_caller
        self._caller_error = caller_error
        try:
            # errors="replace": a hook, plugin or shell command the CLI
            # runs can print bytes in any encoding.  With the strict
            # default the first undecodable byte raises inside a drain
            # thread, which then stops reading: on stderr the child
            # blocks in write(2) once the pipe is full and the turn is
            # killed as a stall; on stdout everything after the byte is
            # silently dropped and a truncated answer reported as done.
            # Its own process group (job object on Windows), so that
            # _signal_tree can take down the tool commands the CLI runs
            # as grandchildren together with it.
            self._proc = popen_process_group(
                _unwrap_npm_cmd_shim(args),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=cwd,
                env=env,
            )
        except OSError as e:
            raise KISSError(f"Failed to start {label}: {e}") from e
        self._stdout_lines: queue.Queue[str | None] = queue.Queue()
        self._stderr_chunks: list[str] = []
        readers: list[threading.Thread] = []
        try:
            readers.append(self._start_reader(self._drain_stdout, "stdout"))
            readers.append(self._start_reader(self._drain_stderr, "stderr"))
        except BaseException:
            self._abort_start(readers)
            raise
        self._readers = (readers[0], readers[1])
        # The stdin writer, once send_prompt() has created it, and the
        # flag that tells it to give up: close() (and a Stop or deadline
        # in send_prompt) set the flag, join the thread, then close stdin.
        # The lock is held by the writer for the whole write and by
        # close() while it closes stdin, so the descriptor can never be
        # closed (and its number reused) under an os.write: a Stop
        # delivered inside Thread.start() leaves a writer that is not yet
        # alive, so nothing to join, but it still runs.
        self._writer: threading.Thread | None = None
        self._writer_cancel = threading.Event()
        self._stdin_lock = threading.RLock()

    def __enter__(self) -> "_CLIProcess":
        """Return the running process supervisor."""
        return self

    def __exit__(self, *exc_info: Any) -> None:
        """End the child's lifetime, however the ``with`` block finished."""
        self.close()

    def _abort_start(self, readers: list[threading.Thread]) -> None:
        """Kill, reap and release a child whose reader threads failed to start.

        ``Thread.start()`` failing (thread exhaustion) inside ``__init__``
        means the ``with`` block is never entered and ``close()`` never
        runs, so the child would stay alive behind three open pipes.  Once
        the child is dead a reader that did start drains its pipe to EOF
        and exits; as in :meth:`close`, an output pipe is closed only
        after its reader is gone, stdin unconditionally.

        Args:
            readers: The reader threads that did start, stdout's first.
        """
        self._signal_tree(SIGKILL)
        self._reap(_REAP_GRACE_SECONDS)
        for thread in readers:
            thread.join(timeout=_REAP_GRACE_SECONDS)
        owners = dict(zip((self._proc.stdout, self._proc.stderr), readers, strict=False))
        for pipe in (self._proc.stdin, self._proc.stdout, self._proc.stderr):
            owner = owners.get(pipe)
            if owner is None or not owner.is_alive():
                assert pipe is not None  # every pipe is subprocess.PIPE
                with contextlib.suppress(OSError):
                    pipe.close()

    def _start_reader(self, target: Callable[[], None], suffix: str) -> threading.Thread:
        """Start a daemon thread draining one of the child's pipes."""
        thread = threading.Thread(target=target, daemon=True, name=f"cli-{suffix}-drain")
        thread.start()
        return thread

    def _drain_stdout(self) -> None:
        """Move every stdout line into the queue, then post the EOF sentinel."""
        try:
            for line in self._proc.stdout:  # type: ignore[union-attr]
                self._stdout_lines.put(line)
        except (OSError, ValueError):  # pragma: no cover – pipe closed under us
            logger.debug("%s stdout closed while being read", self._label)
        finally:
            self._stdout_lines.put(None)

    def _drain_stderr(self) -> None:
        """Keep the stderr pipe empty so the child never blocks writing to it."""
        try:
            for line in self._proc.stderr:  # type: ignore[union-attr]
                self._stderr_chunks.append(line)
        except (OSError, ValueError):  # pragma: no cover – pipe closed under us
            logger.debug("%s stderr closed while being read", self._label)

    def send_prompt(self, prompt: str) -> None:
        """Write *prompt* to the child's stdin and close it.

        A flattened agent conversation dwarfs the pipe buffer, so the
        write only completes as fast as the child drains it.  Doing it
        inline would park the agent thread inside ``write(2)`` — deaf to
        both the turn deadline and Stop, which are polled only while
        reading the child's *output* — for as long as a CLI stuck in
        start-up, auth or plugin loading leaves stdin unread.  The write
        therefore runs on its own thread and is bounded by exactly the
        same two conditions :meth:`lines` obeys.

        A child that exits before reading breaks the pipe mid-write.
        That is swallowed: the caller reports the child's exit status
        and stderr, which is the failure the user can act on.  The wait
        also ends as soon as the child has exited, whether or not the
        write has: a grandchild that inherited the prompt pipe keeps it
        from breaking, and the reply the child already wrote must not
        wait on a prompt nobody will ever read.  A Stop or the deadline
        cancels the writer before raising, so no thread is left behind.

        Args:
            prompt: The complete prompt text.

        Raises:
            KeyboardInterrupt: The user pressed Stop.
            _StreamReadTimeoutError: The turn outlived its deadline
                before the child accepted the whole prompt.
            Exception: What ``caller_error()`` returned.
        """
        writer = threading.Thread(
            target=self._write_prompt,
            args=(prompt.encode("utf-8"),),
            daemon=True,
            name="cli-stdin-write",
        )
        # Owned before start(): a Stop raised right after start() returns
        # must find the running thread in self._writer so close() joins
        # it before closing stdin.  close() skips the join when start()
        # raised (the thread is not alive), so this cannot abort cleanup.
        self._writer = writer
        writer.start()
        while True:
            try:
                remaining = self._remaining()
            except BaseException:
                self._writer_cancel.set()
                raise
            writer.join(timeout=min(remaining, _STOP_POLL_SECONDS))
            if not writer.is_alive() or self._proc.poll() is not None:
                return

    def _raise_for_caller(self) -> None:
        """Raise when the caller's side ends the run: a Stop, or the error it reports.

        The one check of every wait loop of this class (:meth:`send_prompt`,
        :meth:`lines`, :meth:`wait_for_exit`): a tool call the child made
        to the caller can hit a fatal limit while the caller is blocked
        writing the prompt or waiting for the child to wind down, not
        only while it is reading output.

        Raises:
            KeyboardInterrupt: The user pressed Stop.
            Exception: What ``caller_error()`` returned.
        """
        if stop_signal.stop_requested():
            raise stop_error()
        error = self._caller_error() if self._caller_error is not None else None
        if error is not None:
            raise error

    def _remaining(self) -> float:
        """Return the seconds left before the inactivity deadline.

        A passed deadline is reported first; otherwise
        :meth:`_raise_for_caller` gets to raise before the value is
        returned.

        Raises:
            _StreamReadTimeoutError: The deadline has passed.
            KeyboardInterrupt: The user pressed Stop.
            Exception: What ``caller_error()`` returned.
        """
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            raise _StreamReadTimeoutError()
        self._raise_for_caller()
        return remaining

    def _write_prompt(self, data: bytes) -> None:
        """Push *data* into the child's stdin, then close it.

        Runs on the writer thread, where nothing can be reported: a
        broken or already-closed pipe means the child died or
        :meth:`close` tore the turn down, and both are diagnosed by the
        caller from the child's exit status.

        The pipe is written through its raw descriptor in non-blocking
        mode, one ``os.write`` at a time, re-checking the cancel flag,
        the child's exit and the turn deadline between writes.  A
        blocking buffered ``write()`` could be cancelled by nothing
        short of the reader going away — and a grandchild that inherited
        the pipe never reads it — which pinned this thread and the
        descriptor for as long as that grandchild lived.  NO ``select``
        readiness API guards the writes: ``select.select`` raises
        ``ValueError`` for any descriptor >= ``FD_SETSIZE`` (1024),
        which a long-running daemon with many concurrent tasks exceeds
        routinely, and ``select.poll`` does not exist on native Windows
        — either failure mode ends with the "child exited" handler below
        (or the thread dying outside it) silently sending the child an
        empty prompt.  Instead, a full pipe surfaces as
        ``BlockingIOError`` (or, on Windows pipes, a zero-byte write),
        and the writer waits up to :data:`_STOP_POLL_SECONDS` on the
        cancel event — waking early on cancellation — before retrying.

        ``_stdin_lock`` is held from before the descriptor is read until
        the pipe is closed, so :meth:`close` cannot close it between the
        cancel check and the ``os.write``; a writer that only gets the
        lock after :meth:`close` finds the file closed and exits.
        """
        stdin = self._proc.stdin
        assert stdin is not None
        view = memoryview(data)
        with self._stdin_lock:
            try:
                fd = stdin.fileno()
                os.set_blocking(fd, False)
                while view:
                    if (
                        self._writer_cancel.is_set()
                        or self._proc.poll() is not None
                        or time.monotonic() >= self._deadline
                    ):
                        break
                    try:
                        written = os.write(fd, view)
                    except BlockingIOError:
                        self._writer_cancel.wait(_STOP_POLL_SECONDS)
                        continue
                    if written == 0:  # pragma: no cover
                        # Not reachable on POSIX: a non-blocking pipe write
                        # either transfers at least one byte or raises
                        # BlockingIOError.  Windows pipes can report a full
                        # buffer as a zero-byte write instead; treat it the
                        # same so the loop never spins hot or stalls.
                        self._writer_cancel.wait(_STOP_POLL_SECONDS)
                        continue
                    view = view[written:]
            except (OSError, ValueError):
                logger.debug("%s exited before reading its prompt", self._label)
            finally:
                with contextlib.suppress(OSError):
                    stdin.close()

    def lines(self) -> Iterator[str]:
        """Yield the child's stdout lines until EOF.

        Each received line pushes the inactivity deadline out by the full
        configured timeout: the timeout bounds *silence*, not total run
        time, so a CLI agent productively streaming events for hours is
        never cut off mid-work while one that stops producing output is
        detected within ``timeout`` seconds.  Silence while the child
        waits on the caller (``waiting_on_caller()`` is true) re-arms the
        deadline instead of ending the run; an error the caller reports
        (``caller_error()``) ends the run at once.

        Yields:
            One line of stdout at a time.

        Raises:
            KeyboardInterrupt: The user pressed Stop.
            _StreamReadTimeoutError: The child produced no output for
                ``timeout`` seconds.
            Exception: What ``caller_error()`` returned.
        """
        while True:
            if self._waiting_on_caller is not None and self._waiting_on_caller():
                # Re-armed on every poll, not only once expired: the
                # child must get the full timeout of silence AFTER the
                # tool returned, not whatever was left of a deadline
                # re-armed while the tool ran.
                self._deadline = time.monotonic() + self._timeout
            remaining = self._remaining()
            try:
                line = self._stdout_lines.get(timeout=min(remaining, _STOP_POLL_SECONDS))
            except queue.Empty:
                continue
            if line is None:
                return
            self._deadline = time.monotonic() + self._timeout
            yield line

    def _signal_tree(self, sig: int) -> None:
        """Send *sig* to the child and to every process it started.

        The child may be a launcher (``node codex.js``, or the ``cmd.exe``
        behind a shim :func:`_unwrap_npm_cmd_shim` did not recognise)
        whose grandchild is the real CLI, and the CLI runs its tool
        commands (shells, tests, servers) as grandchildren.  Signalling
        the child alone would leave those running with no owner, holding
        the pipes: the whole process group (job object on Windows) is
        signalled instead, falling back to the child alone when the OS
        could not deliver to the group.

        Args:
            sig: The signal to send (ignored on Windows, which has no
                signals: the tree is terminated outright).
        """
        try:
            kill_process_group(self._proc.pid, sig)
        except ProcessLookupError:
            pass
        except OSError:
            self._proc.send_signal(sig)

    def _reap(self, grace: float) -> int | None:
        """Wait up to *grace* seconds for the child to exit, never raising.

        Args:
            grace: Seconds to wait.

        Returns:
            The exit status, or ``None`` when the child is still running.
        """
        try:
            return self._proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            return None

    def wait_for_exit(self) -> int | None:
        """Return the child's exit status, or ``None`` if it is still running.

        A CLI that lingers after closing stdout (flushing telemetry,
        waiting on its own children) must not turn a complete turn into
        an unclassified ``subprocess.TimeoutExpired``.  The wait is
        bounded by whatever is left of the run's deadline, so a short
        configured timeout stays short, and it polls the stop signal and
        the caller's error in :data:`_STOP_POLL_SECONDS` slices like
        :meth:`lines` does: neither a Stop pressed while the child winds
        down nor a tool call of the child that hit a fatal limit
        meanwhile must be answered with a successful turn.

        Returns:
            The exit status, or ``None`` when the child outlived the wait.

        Raises:
            KeyboardInterrupt: The user pressed Stop during the wait.
            Exception: What ``caller_error()`` returned.
        """
        remaining = max(self._deadline - time.monotonic(), 0.0)
        wait_until = time.monotonic() + min(remaining, _EXIT_GRACE_SECONDS)
        while True:
            self._raise_for_caller()
            slice_ = min(wait_until - time.monotonic(), _STOP_POLL_SECONDS)
            status = self._reap(max(slice_, 0.0))
            if status is not None or slice_ <= 0:
                return status

    def stderr_text(self) -> str:
        """Return everything the child wrote to stderr.

        Returns:
            The accumulated stderr text, after giving its reader a
            moment to finish.
        """
        self._readers[1].join(timeout=_REAP_GRACE_SECONDS)
        return "".join(self._stderr_chunks)

    def raise_for_exit(self) -> None:
        """Raise when the child ended its turn with a failure status.

        The one exit-status policy of both CLI adapters: a clean exit and
        a child that is still winding down (see :meth:`wait_for_exit`)
        are fine; anything else — including death by signal, which is
        reported as a negative status — means the reply the caller
        collected is at best truncated and must not be passed off as a
        finished turn.

        Raises:
            KISSError: With the exit status and the child's stderr.
            KeyboardInterrupt: The user pressed Stop while the child was
                winding down (see :meth:`wait_for_exit`).
            Exception: What ``caller_error()`` returned meanwhile.
        """
        status = self.wait_for_exit()
        if status not in (0, None):
            raise KISSError(f"{self._label} failed (exit {status}): {self.stderr_text().strip()}")

    def _close_stdin(self) -> None:
        """Close the prompt pipe; runs under ``_stdin_lock`` from :meth:`close`."""
        assert self._proc.stdin is not None  # every pipe is subprocess.PIPE
        with contextlib.suppress(OSError):
            self._proc.stdin.close()

    def close(self) -> None:
        """Terminate the child, reap it, and release its pipes.

        The prompt pipe is always released: its writer is told to stop,
        joined if it is alive (it re-checks the flag every
        :data:`_STOP_POLL_SECONDS`; a writer whose ``start()`` raised is
        not alive and has nothing to join), and stdin is closed under
        ``_stdin_lock`` so a writer still inside ``Thread.start()`` when
        the Stop landed cannot write through the closed descriptor
        afterwards (through :func:`locked_call`: this runs on the task
        thread, where a second Stop landing right after the acquire
        would leak a plain ``with``).  An output pipe is closed only once the
        thread moving bytes through it has finished.  A grandchild that
        inherited stdout or stderr keeps it open after the child is
        gone, leaving that reader parked inside ``read(2)`` holding the
        buffer's lock — closing the pipe from here would then block this
        thread forever, so such a pipe is left to the abandoned daemon
        thread until the grandchild exits.

        The whole group is signalled whatever the child's own state: a
        grandchild may outlive a child that exited on its own or on the
        SIGTERM, and it may ignore SIGTERM, so the grace period is always
        followed by a SIGKILL of the group (a group that is already gone
        only raises ``ProcessLookupError``, which :meth:`_signal_tree`
        swallows).
        """
        proc = self._proc
        self._signal_tree(signal.SIGTERM)
        if proc.poll() is None:
            self._reap(_REAP_GRACE_SECONDS)
        self._signal_tree(SIGKILL)
        if proc.poll() is None:
            self._reap(_REAP_GRACE_SECONDS)
        self._writer_cancel.set()
        if self._writer is not None and self._writer.is_alive():
            self._writer.join()
        locked_call(self._stdin_lock, self._close_stdin)
        attached = (
            (proc.stdout, self._readers[0]),
            (proc.stderr, self._readers[1]),
        )
        for _pipe, thread in attached:
            thread.join(timeout=_REAP_GRACE_SECONDS)
        for pipe, thread in attached:
            if thread.is_alive():
                logger.warning("%s left %s attached to its pipe", self._label, thread.name)
                continue
            assert pipe is not None
            with contextlib.suppress(OSError):
                pipe.close()


# A complete markdown fence opener (``` plus an optional language tag,
# which may be preceded by spaces) separated from the opening brace by
# nothing but whitespace — blank lines and CRLF endings included.  That
# is the shape a model uses to wrap its tool-call JSON in a code block,
# and it matches the whitespace tolerance of the batch stripper this
# streaming filter replaced.
_FENCE_OPENER_RE = re.compile(r"`{3}[ \t]*[A-Za-z0-9_+~.-]{0,20}[ \t\r\n]*\Z")
# A trailing run that could still grow into a fence opener followed by a
# brace once the next streaming chunk arrives.
_FENCE_PREFIX_RE = re.compile(r"(?:`{1,2}|`{3}[ \t]*[A-Za-z0-9_+~.-]{0,20}[ \t\r\n]*)\Z")

# What the next significant character of a JSON prefix may be.
_KEY_OR_CLOSE, _KEY, _COLON, _VALUE, _VALUE_OR_CLOSE, _SCALAR, _NEXT = range(7)
# Characters a number or a literal is made of: ``true``, ``false``,
# ``null`` and the ``NaN``/``Infinity``/``-Infinity`` the decoder accepts.
_SCALAR_CHARS = frozenset("0123456789+-.EIaefilnrstuyN")
_CLOSER = {"{": "}", "[": "]"}
# Nesting depth of the deepest envelope key: ``{"tool_calls": [{"name":
# ..., "arguments": {``.  Keys at that depth or above are the envelope's
# identifiers; keys below it are argument data and may hold anything.
_ENVELOPE_DEPTH = 3


class _JsonPrefix:
    """Check, one character at a time, that text is still the prefix of a JSON value.

    The stream filter holds text back only while it can still become a
    ``tool_calls`` object; the moment a character rules that out the held
    text is prose and must stream.  The check is structural (``{`` wants
    a key or ``}``, a key wants ``:``, a value wants ``,`` or a closer)
    and deliberately loose about scalars: any run of the characters
    numbers and the three literals are made of passes, and the parser
    that runs on the finished object judges those.

    Strings are the one place the check is looser than JSON and the one
    place it is stricter.  A value string may contain raw control
    characters, because the models this filter serves emit literal
    newlines inside long arguments and the scanner parses them with
    ``strict=False``.  A key string of the envelope (nested no deeper
    than :data:`_ENVELOPE_DEPTH`) may not: those keys are identifiers,
    and a prose quote after a stray brace (``print("{")``) would
    otherwise hold the rest of the turn inside an unclosed "key".  Keys
    below the envelope are argument data, which the decoder accepts with
    raw control characters, so they pass here too: a prefix released as
    prose cannot be taken back once the object turns out to be a call.
    """

    def __init__(self) -> None:
        self.open: list[str] = []
        self._expect = _VALUE
        self._in_string = False
        self._in_key = False
        self._escaped = False

    def feed(self, char: str) -> bool:
        """Consume *char*.

        Args:
            char: The next character of the text.

        Returns:
            False once the text so far cannot be the prefix of a JSON
            value; the state is meaningless after that.  ``open`` is
            empty again when the value has closed.
        """
        if self._in_string:
            return self._feed_string_char(char)
        if char in " \t\r\n":
            if self._expect == _SCALAR:
                self._expect = _NEXT
            return True
        if self._expect == _SCALAR:
            if char in _SCALAR_CHARS:
                return True
            self._expect = _NEXT
        if self._expect == _NEXT:
            if char == "," and self.open:
                self._expect = _KEY if self.open[-1] == "{" else _VALUE
                return True
            return self._close(char)
        if self._expect in (_KEY, _KEY_OR_CLOSE):
            if char == '"':
                self._in_string = self._in_key = True
                return True
            return self._expect == _KEY_OR_CLOSE and self._close(char)
        if self._expect == _COLON:
            self._expect = _VALUE
            return char == ":"
        if char == '"':
            self._in_string = True
            self._in_key = False
        elif char in _CLOSER:
            self.open.append(char)
            self._expect = _KEY_OR_CLOSE if char == "{" else _VALUE_OR_CLOSE
        elif char in _SCALAR_CHARS:
            self._expect = _SCALAR
        else:
            return self._expect == _VALUE_OR_CLOSE and self._close(char)
        return True

    def _feed_string_char(self, char: str) -> bool:
        """Consume one character inside a string; see :meth:`feed`."""
        if self._escaped:
            self._escaped = False
        elif char == "\\":
            self._escaped = True
        elif char == '"':
            self._in_string = False
            self._expect = _COLON if self._in_key else _NEXT
        elif self._in_key and char < " " and len(self.open) <= _ENVELOPE_DEPTH:
            return False
        return True

    def _close(self, char: str) -> bool:
        """Close the innermost container if *char* is its closer."""
        if not self.open or char != _CLOSER[self.open[-1]]:
            return False
        self.open.pop()
        self._expect = _NEXT
        return True


class _ToolCallFilteredStream:
    """Stream a turn to the UI with its tool-call JSON held back.

    Text-based tool calling makes the assistant's *visible* text carry
    the ``{"tool_calls": [...]}`` block the framework parses out of it,
    and rendering that blob in the chat panel is noise the user cannot
    act on.  Both CLI adapters therefore install this filter around the
    turn, which keeps streaming text as it arrives — the panel must
    update incrementally, not in one dump when the turn ends — but
    buffers each top-level JSON object until it closes, then drops it if
    it is a tool-call block and emits it otherwise.  The buffering lasts
    only while the text can still become a JSON object (see
    :class:`_JsonPrefix`): a brace in prose releases what it held the
    moment the next character rules JSON out, so one unmatched ``{``
    never parks the rest of the turn.

    A markdown code fence wrapping a suppressed tool call is suppressed
    with it: a fence opener immediately preceding a ``{`` is held back
    until the object resolves, and the matching closing fence is consumed
    after a suppressed call, so the panel never shows the empty wrapper.
    A fence around anything that is *not* a validated tool call streams
    untouched.

    Thinking tokens are never filtered: they are already displayed apart
    from the answer, and a tool call quoted inside reasoning is content.
    """

    def __init__(self, model: CLITextModel) -> None:
        """Capture *model*'s callbacks so they can be wrapped and restored.

        Args:
            model: The adapter whose stream is being filtered.
        """
        self._model = model
        self._token_callback = model.token_callback
        self._thinking_callback = model.thinking_callback
        self._in_thinking = False
        self._pending = ""
        self._json: _JsonPrefix | None = None
        self._held_prefix = ""
        self._fence_open = ""
        self._eat_close = False
        self._close_buf = ""

    def __enter__(self) -> "_ToolCallFilteredStream":
        """Mark the turn as tool-bearing and install the filtering callbacks.

        The mark is what lets an adapter end the turn at the first complete
        ``tool_calls`` block, so that capability is reachable through the
        plain ``generate()`` contract instead of a widened signature only
        one adapter has.  It is set even when nobody is streaming, because
        the early stop matters regardless of who is watching.
        """
        self._model._tool_bearing_turn = True
        if self._token_callback is not None:
            self._model.token_callback = self._on_token
            self._model.thinking_callback = self._on_thinking
        return self

    def __exit__(self, *exc_info: Any) -> None:
        """Release buffered text, restore the callbacks and clear the mark.

        The flush hands held text to the original token callback, which
        in production raises ``KeyboardInterrupt`` once the user has
        pressed Stop; the restoration must still happen, or the reused
        adapter keeps streaming through this dead filter and treats its
        next ordinary turn as tool-bearing.
        """
        try:
            self._flush()
        finally:
            self._model.token_callback = self._token_callback
            self._model.thinking_callback = self._thinking_callback
            self._model._tool_bearing_turn = False

    def _emit(self, text: str) -> None:
        """Send *text* onward, if there is any and anyone is listening."""
        if text and self._token_callback is not None:
            self._token_callback(text)

    def _on_thinking(self, is_start: bool) -> None:
        """Track and forward the thinking bracket."""
        self._in_thinking = is_start
        if self._thinking_callback is not None:
            self._thinking_callback(is_start)

    def _on_token(self, token: str) -> None:
        """Forward a thinking token as-is, or filter a text token."""
        if self._in_thinking:
            self._emit(token)
        else:
            self._feed(token)

    def _feed(self, text: str) -> None:
        """Emit the plain text of *text*, buffering any JSON object in it.

        A ``{`` opens a candidate that is held while :class:`_JsonPrefix`
        still accepts it and released by :meth:`_release` when it closes
        or when a character proves it prose; that character is then
        rescanned as plain text.  Rescans and the closing-fence eater
        continue in this same loop, so a chunk with thousands of false
        candidates costs no recursion.
        """
        if self._eat_close:
            text = self._eat_closing_fence(text)
        start = index = 0
        while index < len(text):
            char = text[index]
            if self._json is None:
                if char == "{":
                    self._begin_object(text[start:index])
                    start = index
                    self._json = _JsonPrefix()
                    self._json.feed(char)
            elif not self._json.feed(char):
                self._pending += text[start:index]
                self._release()
                start = index
                continue
            elif not self._json.open:
                self._pending += text[start : index + 1]
                self._release()
                start = index + 1
                if self._eat_close:
                    text = self._eat_closing_fence(text[start:])
                    start = index = 0
                    continue
            index += 1
        if self._json is None:
            self._hold_tail(text[start:])
        else:
            self._pending += text[start:]

    def _begin_object(self, before: str) -> None:
        """Emit the text preceding a ``{``, holding back its fence opener.

        A fence opener directly against the brace may be the wrapper of a
        tool call, so it is parked in ``_fence_open`` until the object
        resolves; everything else streams immediately.

        Args:
            before: The depth-0 text between the previous emission point
                and the opening brace.
        """
        combined = self._held_prefix + before
        self._held_prefix = ""
        match = _FENCE_OPENER_RE.search(combined)
        if match:
            self._fence_open = match.group(0)
            combined = combined[: match.start()]
        else:
            self._fence_open = ""
        self._emit(combined)

    def _hold_tail(self, tail: str) -> None:
        """Emit depth-0 text, holding back a possible partial fence opener.

        The next chunk may complete the opener and its ``{``, so a
        trailing run that could still become a fence opener waits in
        ``_held_prefix`` instead of streaming; it is released the moment
        anything breaks the pattern, or by :meth:`_flush`.

        Args:
            tail: The depth-0 text ending the current chunk.
        """
        combined = self._held_prefix + tail
        match = _FENCE_PREFIX_RE.search(combined)
        self._held_prefix = match.group(0) if match else ""
        self._emit(combined[: len(combined) - len(self._held_prefix)])

    def _eat_closing_fence(self, text: str) -> str:
        """Consume the closing fence of a suppressed fenced tool call.

        Args:
            text: The next chunk of depth-0 text.

        Returns:
            The text left over after the fence — empty while the fence is
            still being consumed, or everything buffered plus the rest of
            *text* when it turns out not to be a closing fence.
        """
        for index, char in enumerate(text):
            ticks = self._close_buf.count("`")
            if ticks == 3:
                if char in " \t\r":
                    self._close_buf += char
                    continue
                self._eat_close = False
                self._close_buf = ""
                return text[index + 1 :] if char == "\n" else text[index:]
            if char == "`":
                self._close_buf += char
            elif ticks == 0 and char in " \t\r\n":
                self._close_buf += char
            else:
                leftover = self._close_buf + text[index:]
                self._eat_close = False
                self._close_buf = ""
                return leftover
        return ""

    def _release(self) -> None:
        """Let go of the held candidate: drop a validated tool call, emit the rest.

        Called when the object closes, when a character proves the held
        text is not JSON, and at the end of the turn.  The held text is
        dropped only when it parse-validates as a tool call; malformed
        JSON and ordinary text that merely contains the substring
        ``tool_calls`` are emitted, so no ordinary text is ever
        swallowed.  A suppressed tool call takes its held fence opener
        with it and arms the closing-fence eater; anything else streams
        together with the opener that was held for it.
        """
        block, self._pending = self._pending, ""
        fence, self._fence_open = self._fence_open, ""
        self._json = None
        if _parse_text_based_tool_calls(block):
            self._eat_close = bool(fence)
        else:
            self._emit(fence + block)

    def _flush(self) -> None:
        """Release text the turn ended in the middle of.

        A held possible fence opener is plain text once nothing follows
        it; the pending candidate goes through :meth:`_release`, which
        drops it only when it is a tool call — the early-stop path cuts
        the stream off exactly there.  A partially consumed closing fence
        belongs to the already-suppressed wrapper and is discarded with
        it.
        """
        held, self._held_prefix = self._held_prefix, ""
        self._emit(held)
        self._release()
        self._eat_close = False
        self._close_buf = ""


def _tool_summary_and_params(
    func: Callable[..., Any], name: str
) -> tuple[str, str, Mapping[str, inspect.Parameter]]:
    """Return the one-line summary, full docstring and parameters of a tool.

    The summary is the first docstring line, or ``Function <name>`` when
    *func* has no docstring; the docstring is ``""`` in that case.  This
    is the single reader of a tool's signature and docstring, shared by
    the OpenAI schema builder and the text-based tools prompt.

    Args:
        func: The tool function.
        name: The name to use in the fallback summary.

    Returns:
        ``(summary, docstring, parameters)``.
    """
    doc = inspect.getdoc(func) or ""
    summary = doc.split("\n")[0] if doc else f"Function {name}"
    return summary, doc, inspect.signature(func).parameters


def _annotation_name(annotation: Any) -> str:
    """Return the name shown for a parameter annotation in the text tools prompt.

    A class shows its ``__name__``; anything else (a ``list[str]`` alias,
    or the verbatim string of a module under ``from __future__ import
    annotations``) shows its ``str``; an unannotated parameter is ``any``.
    """
    if annotation is inspect.Parameter.empty:
        return "any"
    return getattr(annotation, "__name__", str(annotation))


def _build_text_based_tools_prompt(function_map: dict[str, Callable[..., Any]]) -> str:
    """Build a text-based tools description for models without native function calling.

    Args:
        function_map: Dictionary mapping function names to callable functions.

    Returns:
        A formatted prompt string describing available tools and how to call them,
        or an empty string if no functions are provided.
    """
    if not function_map:
        return ""

    tools_desc = []
    for func_name, func in function_map.items():
        summary, _doc, parameters = _tool_summary_and_params(func, func_name)
        params = [
            f"    - {param_name} ({_annotation_name(param.annotation)})"
            for param_name, param in parameters.items()
        ]
        params_str = "\n".join(params) if params else "    (no parameters)"
        tools_desc.append(f"- **{func_name}**: {summary}\n  Parameters:\n{params_str}")

    return f"""
## Available Tools

To call a tool, output a JSON object in the following format:

```json
{{"tool_calls": [{{"name": "tool_name", "arguments": {{"arg1": "value1", "arg2": "value2"}}}}]}}
```

You can call multiple tools at once by including multiple objects in the tool_calls array.

### Tools:
{chr(10).join(tools_desc)}

CRITICAL RULES:
1. When you want to call a tool, output ONLY the JSON object with tool_calls.
2. STOP IMMEDIATELY after outputting the tool_calls JSON. Do NOT continue generating.
3. Do NOT predict, simulate, or hallucinate what the tool results will be.
4. The system will execute the tools and provide actual results in the next message.
5. When you have the final answer, call the `finish` tool with your result.
"""


def _brace_matches(content: str) -> dict[int, tuple[int, bool]]:
    """Pair every ``{`` in *content* with the ``}`` a scan started at it would find.

    A scan that starts at a ``{`` is outside any JSON string, so for a
    brace that a single left-to-right scan of the whole text sees *inside*
    a string, the answer is what the opposite string parity sees from
    there on.  Both parities are therefore tracked at once, each matching
    the braces it sees outside its strings on its own stack.  A backslash
    escapes the quote after it on both (JSON has that only inside
    strings; giving prose the same rule keeps the two parities exact
    complements).  Braces never affect the parity, so an escaped brace
    still counts for the parity that sees it outside a string: a ``\\{``
    in prose is a candidate like any other.

    Args:
        content: The text to scan.

    Returns:
        Mapping of the index of each matched ``{`` to ``(end, quoted)``:
        the index just past its ``}`` and whether the whole-text scan
        sees that ``{`` inside a string.  Two candidates with the same
        ``quoted`` flag see each other's text with the same string parity.
    """
    matches: dict[int, tuple[int, bool]] = {}
    stacks: tuple[list[int], list[int]] = ([], [])
    in_string = False
    escaped = False
    for index, char in enumerate(content):
        if char == "\\" and not escaped:
            escaped = True
            continue
        if char == '"':
            if not escaped:
                in_string = not in_string
        elif char == "{":
            stacks[in_string].append(index)
        elif char == "}" and stacks[in_string]:
            matches[stacks[in_string].pop()] = (index + 1, in_string)
        escaped = False
    return matches


def _iter_balanced_json_objects(
    content: str,
) -> "list[tuple[int, int, Any]]":
    """Find every balanced ``{...}`` substring that parses as valid JSON.

    Every ``{`` is a candidate whose span runs to the ``}`` a scan started
    at that brace pairs it with (see :func:`_brace_matches`), and the
    candidates are tried in text order: one that parses is reported and
    swallows the candidates inside it, one that does not is skipped, so a
    stray ``{`` in prose, an unclosed quote or an object that is not JSON
    never hides a later object.  Nested objects, arrays and braces inside
    string values are handled by the pairing.

    A candidate that fails to decode has a valid prefix up to the error
    position: every object the decoder passed in that prefix is valid and
    is reported when its turn comes, while a candidate that starts in the
    prefix and reaches past the error is decoded along the same path and
    fails at the same place, so it is skipped without a decode.  That
    keeps deeply nested malformed text at one decode per failure instead
    of one per nesting level.  The shortcut applies only to candidates
    that see the failed one's text with the same string parity; one that
    starts inside one of its strings is a different text and is tried.

    Args:
        content: The text to scan.

    Returns:
        A list of ``(start, end_exclusive, parsed)`` tuples in left-to-right
        order, one per successfully parsed top-level balanced object.
    """
    matches = _brace_matches(content)
    results: list[tuple[int, int, Any]] = []
    skip_to = 0
    failed_at = [0, 0]
    for start in sorted(matches):
        end, quoted = matches[start]
        if start < skip_to or (start < failed_at[quoted] < end):
            continue
        try:
            parsed = json.loads(content[start:end], strict=False)
        except json.JSONDecodeError as exc:
            failed_at[quoted] = start + exc.pos
            continue
        except RecursionError:
            # Nested deeper than the parser can follow (tens of thousands
            # of levels).  The candidates inside are nearly as deep and
            # each costs a walk to the recursion limit, so they are
            # skipped with it.
            skip_to = end
            continue
        results.append((start, end, parsed))
        skip_to = end
    return results


def _iter_tool_calls_lists(obj: Any) -> "list[list[Any]]":
    """Recursively collect every ``tool_calls`` list inside *obj*.

    Walks dicts and lists looking for any ``"tool_calls"`` key whose value
    is a list.  This supports both top-level ``{"tool_calls": [...]}``
    objects and tool_calls nested inside an outer wrapper such as
    ``{"outer": {"tool_calls": [...]}}``.

    Args:
        obj: A parsed JSON value (dict, list, scalar).

    Returns:
        Lists of tool-call entries in encounter order.
    """
    out: list[list[Any]] = []
    if isinstance(obj, dict):
        tcs = obj.get("tool_calls")
        if isinstance(tcs, list):
            out.append(tcs)
        for v in obj.values():
            out.extend(_iter_tool_calls_lists(v))
    elif isinstance(obj, list):
        for item in obj:
            out.extend(_iter_tool_calls_lists(item))
    return out


def _parse_text_based_tool_calls(content: str) -> list[dict[str, Any]]:
    """Parse tool calls from text-based model output.

    Uses a brace-balanced JSON scanner (see
    :func:`_iter_balanced_json_objects`) to find every valid JSON object in
    *content*, then collects every nested ``tool_calls`` list.  Duplicates
    (same ``name`` + same ``arguments``) are removed so a single logical
    call emitted multiple times collapses to one entry.

    This robustly handles reasoning models (e.g. ``cc/opus``) that:

    - emit bare ``{"tool_calls": [...]}`` JSON without code fences,
    - include arguments containing nested objects, arrays, or strings with
      ``[]``/``{}`` characters (e.g. markdown links inside a finish summary),
    - emit several distinct ``tool_calls`` blocks in one response.

    Args:
        content: The text content to parse for tool calls.

    Returns:
        A list of function call dictionaries, each containing ``id``,
        ``name``, and ``arguments`` keys.  Returns an empty list if no
        valid tool calls are found.
    """
    function_calls: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for _start, _end, parsed in _iter_balanced_json_objects(content):
        for tcs in _iter_tool_calls_lists(parsed):
            for tc in tcs:
                if not (isinstance(tc, dict) and "name" in tc):
                    continue
                arguments = tc.get("arguments", {})
                key = (tc["name"], json.dumps(arguments, sort_keys=True))
                if key in seen:
                    continue
                seen.add(key)
                function_calls.append(
                    {
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "name": tc["name"],
                        "arguments": arguments,
                    }
                )

    return function_calls
