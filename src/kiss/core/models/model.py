# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Abstract base class for LLM provider model implementations.

Also contains shared text-based tool calling helpers used by models that
lack native function calling support (e.g. DeepSeek R1, Claude Code CLI).
"""

import base64
import dataclasses
import inspect
import json
import logging
import mimetypes
import os
import re
import time
import types as types_module
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any, Union, get_args, get_origin

from kiss.core.kiss_error import KISSError

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

# Content-part types produced by the OpenAI Responses API
# (:class:`OpenAICompatibleModel2` stores its conversation in this shape).
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
            # ``input_audio`` has the same shape in both APIs; anything
            # else is already Chat-Completions-shaped (text / image_url /
            # file / tool_use / ...) and passes through unchanged.
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
            tc = {
                "id": item.get("call_id", ""),
                "type": "function",
                "function": {
                    "name": item.get("name", ""),
                    "arguments": args if isinstance(args, str) else json.dumps(args or {}),
                },
            }
            prev = out[-1] if out else None
            if (
                isinstance(prev, dict)
                and prev.get("role") == "assistant"
                and prev.get("tool_call_id") is None
            ):
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
            # Standalone non-message items (``reasoning``, the internal
            # ``_kiss_pending_tool_result_attachment`` sentinel, ...) are
            # hidden provider state with no Chat-Completions equivalent.
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
            # Responses ``message`` item whose content is already
            # chat-compatible: keep only the Chat-Completions keys
            # (``type`` / ``id`` / ``status`` are Responses-only).
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
    def from_file(path: str) -> "Attachment":
        """Create an Attachment from a file path.

        Args:
            path: Path to the file to attach.

        Returns:
            An Attachment with the file's bytes and detected MIME type.

        Raises:
            ValueError: If the MIME type is not supported.
            FileNotFoundError: If the file does not exist.
        """
        file_path = Path(path)
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if mime_type is None:  # pragma: no cover – mimetypes knows all supported extensions
            suffix = file_path.suffix.lower()
            mime_map = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".gif": "image/gif",
                ".webp": "image/webp",
                ".pdf": "application/pdf",
                ".mp3": "audio/mpeg",
                ".wav": "audio/wav",
                ".ogg": "audio/ogg",
                ".webm": "video/webm",
                ".flac": "audio/flac",
                ".aac": "audio/aac",
                ".m4a": "audio/mp4",
                ".mp4": "video/mp4",
                ".mpeg": "video/mpeg",
                ".mov": "video/quicktime",
            }
            mime_type = mime_map.get(suffix, "")
        if mime_type not in SUPPORTED_MIME_TYPES:
            raise ValueError(
                f"Unsupported MIME type '{mime_type}' for file '{path}'. "
                f"Supported: {sorted(SUPPORTED_MIME_TYPES)}"
            )
        return Attachment(data=file_path.read_bytes(), mime_type=mime_type)

    def to_base64(self) -> str:
        """Return the file data as a base64-encoded string."""
        return base64.b64encode(self.data).decode("ascii")

    def to_data_url(self) -> str:
        """Return a data: URL suitable for OpenAI image_url fields."""
        return f"data:{self.mime_type};base64,{self.to_base64()}"


# Tool results that include binary file contents (e.g. ``Read`` on a PNG)
# embed each file inside these sentinel markers so that model implementations
# can lift the payload into a real image/document content block instead of
# shipping kilobytes of base64 as plain text.
BINARY_ATTACHMENT_OPEN_RE = re.compile(
    r"<<KISS_BINARY_ATTACHMENT mime_type=([^>\s]+)>>"
)
BINARY_ATTACHMENT_CLOSE = "<</KISS_BINARY_ATTACHMENT>>"

# MIME types the ``Read`` tool is willing to embed inline in its return
# value.  Set equal to :data:`SUPPORTED_MIME_TYPES` so audio/video are
# also encoded as sentinel-wrapped base64; each model backend then decides
# whether it can actually ingest the bytes (e.g. OpenAI Chat Completions
# accepts ``input_audio``; Gemini accepts any ``inline_data`` MIME;
# Anthropic transcribes audio to text and drops video; text-CLI backends
# drop the bytes after lifting the placeholder text).
READ_TOOL_BINARY_MIME_TYPES = set(SUPPORTED_MIME_TYPES)


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
    payload = base64.b64encode(data).decode("ascii")
    return (
        f"<<KISS_BINARY_ATTACHMENT mime_type={mime_type}>>"
        f"{payload}{BINARY_ATTACHMENT_CLOSE}"
    )


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
        close_idx = text.find(BINARY_ATTACHMENT_CLOSE, open_end)
        if close_idx == -1:  # pragma: no cover – defensive: malformed sentinel
            break
        mime_type = match.group(1)
        b64_payload = text[open_end:close_idx]
        try:
            data = base64.b64decode(b64_payload, validate=False)
        except Exception:  # pragma: no cover – validate=False is permissive
            logger.debug("Failed to decode binary attachment", exc_info=True)
            cursor = close_idx + len(BINARY_ATTACHMENT_CLOSE)
            continue
        out_parts.append(text[cursor:open_start])
        out_parts.append(f"[attached {mime_type}, {len(data)} bytes]")
        attachments.append(Attachment(data=data, mime_type=mime_type))
        cursor = close_idx + len(BINARY_ATTACHMENT_CLOSE)
    out_parts.append(text[cursor:])
    return "".join(out_parts), attachments


def _tool_result_to_string(result_dict: dict[str, Any]) -> str:
    """Return a provider-safe string for a tool result payload.

    Structured values are JSON encoded consistently across the base,
    Chat-Completions, and Responses implementations.  Values unsupported by
    JSON fall back to ``str`` so attachment parsing always receives text.
    """
    raw_result = result_dict.get("result", str(result_dict))
    if isinstance(raw_result, str):
        return raw_result
    try:
        return json.dumps(raw_result, ensure_ascii=False)
    except TypeError:
        return str(raw_result)


_AUDIO_MIME_TO_EXT: dict[str, str] = {
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/ogg": ".ogg",
    "audio/webm": ".webm",
    "audio/flac": ".flac",
    "audio/aac": ".aac",
    "audio/mp4": ".m4a",
}


def _is_transient_transcription_error(exc: Exception) -> bool:
    """Return True when a transcription failure is worth retrying.

    Retryable: rate limits (429), connection/timeout errors, and
    server-side 5xx responses.  Not retryable: authentication,
    invalid-request, and other deterministic client errors.

    Args:
        exc: The exception raised by the OpenAI transcription call.

    Returns:
        True when the error is transient and the call may be retried.
    """
    import openai

    if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError)):
        return True
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code == 429 or exc.status_code >= 500
    return False


def transcribe_audio(data: bytes, mime_type: str, api_key: str | None = None) -> str:
    """Transcribe audio bytes to text using OpenAI's Whisper API.

    This is used as a fallback for model providers that do not support audio
    attachments natively (e.g. Anthropic).  Transient API failures (rate
    limits, connection errors, 5xx responses) are retried a couple of times
    with a short backoff before giving up.

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
    from openai import OpenAI

    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise ValueError("OpenAI API key is required for audio transcription")

    ext = _AUDIO_MIME_TO_EXT.get(mime_type, ".mp3")
    client = OpenAI(api_key=key)
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=(f"audio{ext}", data, mime_type),
                response_format="text",
            )
            return str(transcript).strip()
        except Exception as exc:
            if attempt < max_attempts and _is_transient_transcription_error(exc):
                time.sleep(0.5 * attempt)
                continue
            raise RuntimeError(f"Audio transcription failed: {exc}") from exc
    raise RuntimeError("Audio transcription failed")  # pragma: no cover - unreachable


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


class Model(ABC):
    """Abstract base class for LLM provider implementations."""

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

    def _invoke_token_callback(self, token: str) -> None:
        """Invoke the token callback synchronously."""
        if self.token_callback is not None:
            self.token_callback(token)

    def _invoke_thinking_callback(self, is_start: bool) -> None:
        """Invoke the thinking callback synchronously.

        Args:
            is_start: ``True`` when a thinking block starts, ``False`` when it ends.
        """
        if self.thinking_callback is not None:
            self.thinking_callback(is_start)

    def reset_conversation(self) -> None:
        """Reset conversation state for reuse across sub-sessions.

        Clears the conversation history and usage info while keeping the
        HTTP client and model configuration intact.
        """
        self.conversation = []
        self.usage_info_for_messages = ""

    def _replace_last_assistant_with_tool_calls(
        self, content: str, function_calls: list[dict[str, Any]]
    ) -> None:
        """Replace the last assistant message with one that includes tool calls.

        Used by text-based tool calling paths (ClaudeCodeModel, CodexModel,
        OpenAICompatibleModel)
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
                {
                    "id": fc["id"],
                    "type": "function",
                    "function": {
                        "name": fc["name"],
                        "arguments": json.dumps(fc["arguments"]),
                    },
                }
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

    @abstractmethod
    def generate(self) -> tuple[str, Any]:
        """Generates content from prompt.

        Returns:
            tuple[str, Any]: A tuple of (generated_text, raw_response).
        """
        pass  # pragma: no cover

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
                # The trailing Responses-API ``function_call`` run ended
                # at this role-bearing message.
                break
            if msg.get("role") == "assistant":
                if msg.get("tool_calls"):
                    return [
                        (tc["function"]["name"], tc["id"]) for tc in msg["tool_calls"]
                    ]
                content = msg.get("content")
                if isinstance(content, list):
                    ids = [
                        (b.get("name", ""), b.get("id", ""))
                        for b in content
                        if b.get("type") == "tool_use"
                    ]
                    if ids:
                        return ids
                break
        return list(reversed(native))

    def add_function_results_to_conversation_and_return(
        self, function_results: list[tuple[str, dict[str, Any]]]
    ) -> None:
        """Adds function results to the conversation state.

        Matches results to tool calls by index from the last assistant message.

        Args:
            function_results: List of tuples containing (function_name, result_dict).
        """
        tool_calls = self._find_tool_call_ids_from_last_assistant()

        for i, (func_name, result_dict) in enumerate(function_results):
            result_content = _tool_result_to_string(result_dict)
            # Strip binary attachment payloads — the default OpenAI-style
            # ``role: tool`` message does not accept image content blocks,
            # so we drop the base64 bytes and keep only the placeholder
            # text so the conversation does not balloon to megabytes.
            result_content, _ = parse_binary_attachments(result_content)
            if self.usage_info_for_messages:
                result_content = f"{result_content}\n\n{self.usage_info_for_messages}"

            if i < len(tool_calls):
                tool_call_id = tool_calls[i][1]
            else:
                tool_call_id = f"call_{func_name}_{i}"

            self.conversation.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": result_content,
                }
            )

    def add_message_to_conversation(self, role: str, content: str) -> None:
        """Adds a message to the conversation state.

        Args:
            role: The role of the message sender (e.g., 'user', 'assistant').
            content: The message content.
        """
        if role == "user" and self.usage_info_for_messages:  # pragma: no branch
            content = f"{content}\n\n{self.usage_info_for_messages}"
        self.conversation.append({"role": role, "content": content})

    @abstractmethod
    def extract_input_output_token_counts_from_response(
        self, response: Any
    ) -> (
        tuple[int, int, int, int]
        | tuple[int, int, int, int, int]
        | tuple[int, int, int, int, int, int, int]
    ):
        """Extracts token counts from an API response.

        Args:
            response: The raw API response object.

        Returns:
            A 4-tuple ``(input_tokens, output_tokens, cache_read_tokens,
            cache_write_tokens)``, a 5-tuple whose last element is the
            Anthropic one-hour cache-write token count, or a 7-tuple
            whose last two elements are the audio input/output token
            subsets of an OpenAI audio-chat response (billed at the
            model's separate audio rates).
        """
        pass  # pragma: no cover

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
        return tools_schema if tools_schema is not None else self._build_openai_tools_schema(
            function_map
        )

    # =========================================================================
    # Helper methods for building tool schemas (shared across implementations)
    # ========================================================================

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
        sig = inspect.signature(func)
        doc = inspect.getdoc(func) or ""

        param_descriptions = self._parse_docstring_params(doc)

        properties: dict[str, Any] = {}
        required: list[str] = []

        for param_name, param in sig.parameters.items():
            param_type = param.annotation
            param_schema = self._python_type_to_json_schema(param_type)

            if param_name in param_descriptions:
                param_schema["description"] = param_descriptions[param_name]

            properties[param_name] = param_schema

            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        description = doc.split("\n")[0] if doc else f"Function {func.__name__}"

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

    def _parse_docstring_params(self, docstring: str) -> dict[str, str]:
        """Parses parameter descriptions from a docstring.

        Args:
            docstring: The docstring to parse.

        Returns:
            dict[str, str]: A dictionary mapping parameter names to descriptions.
        """
        param_descriptions: dict[str, str] = {}
        lines = docstring.split("\n")
        in_args_section = False

        for line in lines:
            stripped = line.strip()
            if stripped.lower().startswith("args:"):
                in_args_section = True
                continue
            elif stripped.lower().startswith(("returns:", "raises:", "example:")):
                in_args_section = False
                continue

            if in_args_section and ":" in stripped:
                parts = stripped.split(":", 1)
                if len(parts) == 2:  # pragma: no branch
                    param_part = parts[0].strip()
                    desc_part = parts[1].strip()
                    if "(" in param_part:
                        param_name = param_part.split("(")[0].strip()
                    else:
                        param_name = param_part
                    param_descriptions[param_name] = desc_part

        return param_descriptions

    def _python_type_to_json_schema(self, python_type: Any) -> dict[str, Any]:
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
                return self._python_type_to_json_schema(non_none_args[0])
            return {"anyOf": [self._python_type_to_json_schema(a) for a in non_none_args]}

        if origin is list:
            if args:
                return {
                    "type": "array",
                    "items": self._python_type_to_json_schema(args[0]),
                }
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

    Both transports flatten the conversation into a single text prompt,
    support tool calling only via the text-based ``tool_calls`` JSON
    protocol, and cannot produce embeddings.  That shared plumbing lives
    here; subclasses implement the subprocess invocation and stream
    parsing.
    """

    # Concrete transports override both attributes so inherited methods
    # preserve the exact class label and provider-module logger used before
    # this plumbing moved into the shared base.  In particular, subclasses
    # of ClaudeCodeModel/CodexModel must retain the concrete transport name
    # rather than exposing their own subclass name in warnings/errors.
    _cli_model_name = "CLITextModel"
    _cli_logger = logger

    def initialize(self, prompt: str, attachments: list[Attachment] | None = None) -> None:
        """Initialize the conversation with an initial user prompt.

        Args:
            prompt: The initial user prompt.
            attachments: Not supported — ignored with a warning if provided.
        """
        if attachments:  # pragma: no cover – attachments not used in practice
            self._cli_logger.warning(
                f"{self._cli_model_name} does not support attachments; "
                "they will be ignored."
            )
        self.conversation = [{"role": "user", "content": prompt}]

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

    def _install_tools_prompt_in_system_instruction(
        self, function_map: dict[str, Callable[..., Any]]
    ) -> dict[str, Any]:
        """Install a copied config containing the text-based tools prompt.

        Returns the original config so callers can restore it in their own
        ``finally`` block.  Keeping restoration at the call site is
        intentional: Codex historically installed the config *before*
        reading or replacing its stream callbacks, then restored the config
        *before* those callbacks.  A context manager necessarily changes one
        of those exception/order semantics because nested contexts unwind in
        reverse order.

        Args:
            function_map: Dictionary mapping function names to callables.

        Returns:
            The original ``model_config`` object, unchanged.
        """
        tools_prompt = _build_text_based_tools_prompt(function_map)
        original_config = self.model_config
        config = dict(original_config)
        original_system = config.get("system_instruction", "")
        config["system_instruction"] = (original_system + "\n\n" + tools_prompt).strip()
        self.model_config = config
        return original_config

    def get_embedding(self, text: str, embedding_model: str | None = None) -> list[float]:
        """Not supported — the CLI transports do not provide embeddings.

        Raises:
            KISSError: Always.
        """
        raise KISSError(f"{self._cli_model_name} does not support embeddings.")


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
        sig = inspect.signature(func)
        doc = inspect.getdoc(func) or f"Function {func_name}"

        params = []
        for param_name, param in sig.parameters.items():
            param_type = param.annotation
            type_name = getattr(param_type, "__name__", str(param_type))
            if type_name == "_empty":
                type_name = "any"
            params.append(f"    - {param_name} ({type_name})")

        params_str = "\n".join(params) if params else "    (no parameters)"
        first_line = doc.split("\n")[0]
        tools_desc.append(f"- **{func_name}**: {first_line}\n  Parameters:\n{params_str}")

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


def _iter_balanced_json_objects(
    content: str,
) -> "list[tuple[int, int, Any]]":
    """Find every balanced ``{...}`` substring that parses as valid JSON.

    Scans *content* character by character.  At every ``{`` that is not
    inside a JSON string, walks forward tracking brace depth and string
    state (with ``\\`` escape handling) until the matching ``}`` is found,
    then attempts to parse the substring with :func:`json.loads`.  This
    correctly handles nested objects, arrays, and brackets/braces embedded
    inside JSON string values — cases the previous regex-based approach
    silently dropped.

    Args:
        content: The text to scan.

    Returns:
        A list of ``(start, end_exclusive, parsed)`` tuples in left-to-right
        order, one per successfully parsed top-level balanced object.
    """
    results: list[tuple[int, int, Any]] = []
    i, n = 0, len(content)
    while i < n:
        if content[i] != "{":
            i += 1
            continue
        depth = 0
        j = i
        in_str = False
        esc = False
        end = -1
        while j < n:
            ch = content[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = j + 1
                    break
            j += 1
        if end == -1:
            # Unbalanced — this "{" (e.g. a stray brace in prose) never
            # closes, but a later "{" may still start a valid object.
            # Resume the scan at the next "{" instead of aborting.
            nxt = content.find("{", i + 1)
            if nxt == -1:
                break
            i = nxt
            continue
        try:
            # ``strict=False`` permits raw control characters (e.g. literal
            # newlines, tabs) inside JSON string values.  Reasoning models
            # such as ``cc/opus`` routinely emit unescaped newlines inside
            # ``summary`` arguments, which strict JSON would reject.
            parsed = json.loads(content[i:end], strict=False)
        except json.JSONDecodeError:
            i += 1
            continue
        results.append((i, end, parsed))
        i = end
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
                try:
                    key = (tc["name"], json.dumps(arguments, sort_keys=True))
                except TypeError:
                    # Non-JSON-serializable args — fall back to repr for keying.
                    key = (tc["name"], repr(arguments))
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


# Matches an empty fenced code block left behind after the JSON inside it
# has been stripped — e.g. ``"```json\n\n```"`` or ``"```\n\n```"``.
_EMPTY_FENCE_PATTERN = re.compile(r"```(?:json)?\s*```", re.DOTALL)


def _strip_text_based_tool_calls(content: str) -> str:
    """Remove tool_calls JSON blocks from *content*, keeping surrounding text.

    Uses :func:`_iter_balanced_json_objects` to find every balanced JSON
    object and strips those that contain a ``tool_calls`` list.  Empty
    fenced code blocks left behind (e.g. ``"```json\\n\\n```"``) are also
    cleaned up so the visible Thoughts panel does not show stray fences.

    Args:
        content: The full model response text.

    Returns:
        The text with tool_calls JSON removed, stripped of leading/trailing
        whitespace.  Returns ``""`` if the entire content was a tool_calls
        wrapper.
    """
    spans = [
        (start, end)
        for start, end, parsed in _iter_balanced_json_objects(content)
        if _iter_tool_calls_lists(parsed)
    ]
    if not spans:
        return content.strip()
    parts: list[str] = []
    cursor = 0
    for start, end in spans:
        parts.append(content[cursor:start])
        cursor = end
    parts.append(content[cursor:])
    return _EMPTY_FENCE_PATTERN.sub("", "".join(parts)).strip()
