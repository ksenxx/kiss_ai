# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Anthropic model implementation for Claude models."""

import base64
import json
import logging
from collections.abc import Callable
from typing import Any

import httpx
from anthropic import Anthropic, APITimeoutError
from anthropic.resources.messages import Messages

from kiss.core import stop_signal
from kiss.core.kiss_error import KISSError, ModelRefusalError
from kiss.core.models.model import (
    FRAMEWORK_ONLY_CONFIG_KEYS,
    Attachment,
    Model,
    ThinkingCallback,
    TokenCallback,
    accepted_request_params,
    responses_items_to_chat_messages,
    transcribe_audio,
)
from kiss.core.models.stream_abort import (
    DEFAULT_STREAM_STALL_TIMEOUT,
    StreamAbortWatchdog,
    stall_error,
)

logger = logging.getLogger(__name__)

_CONNECT_TIMEOUT = 10.0
_MAX_RETRIES = 1

# The request parameters an Anthropic message call accepts, taken from the
# SDK's own signature (keyword-only, no **kwargs), so an unsupported
# model_config key is reported instead of raising TypeError.
_ANTHROPIC_REQUEST_PARAMS = accepted_request_params(Messages.stream)


def cache_creation_tokens(usage: Any, get: Callable[[Any, str], Any]) -> tuple[int, int]:
    """Return the 5-minute and 1-hour cache-creation token counts.

    Anthropic reports cache writes either split by TTL under
    ``cache_creation`` or as a single aggregate.  An aggregate is
    attributed to the one-hour bucket, the more expensive of the two, so
    an unknown TTL is never under-billed.  The Claude Code CLI re-emits
    exactly this shape as JSON, so :mod:`kiss.core.models.claude_code_model`
    shares this parser and differs only in *get* — otherwise a change to
    Anthropic's cache tiers would have to be made twice.

    Args:
        usage: The provider's usage record — an SDK object for the API
            transport, a decoded JSON dict for the CLI transport.
        get: Reads a named field off *usage* or a nested record,
            returning ``None`` when the field is absent.

    Returns:
        ``(cache_write_5m_tokens, cache_write_1h_tokens)``.
    """
    cache_creation = get(usage, "cache_creation")
    if cache_creation is not None:
        return (
            get(cache_creation, "ephemeral_5m_input_tokens") or 0,
            get(cache_creation, "ephemeral_1h_input_tokens") or 0,
        )
    return 0, get(usage, "cache_creation_input_tokens") or 0


def _attribute_field(record: Any, name: str) -> Any:
    """Return attribute *name* of *record*, or ``None`` when it is absent."""
    return getattr(record, name, None)


_THINKING_FAMILIES = ("opus", "sonnet", "haiku", "fable")


def _parse_claude_version(model_name: str) -> tuple[str, int, int | None] | None:
    """Parse ``claude-<family>-<major>[-<minor>][-<date>]`` model names.

    The accepted shapes are exactly ``claude-<family>-<major>``,
    ``claude-<family>-<major>-<date>``, ``claude-<family>-<major>-<minor>``,
    and ``claude-<family>-<major>-<minor>-<date>``, where ``<family>`` is
    alphabetic, ``<major>`` and ``<minor>`` are non-zero-padded runs of
    digits, ``<minor>`` is not 8 digits long, and ``<date>`` is exactly
    8 digits and may only be the final segment.

    Args:
        model_name: The model name to parse (e.g. ``claude-opus-4-8``,
            ``claude-opus-5``, ``claude-haiku-4-5-20251001``).

    Returns:
        A ``(family, major, minor)`` tuple, where ``minor`` is ``None``
        when the name has no minor version (e.g. ``claude-opus-4`` or a
        dated snapshot of a bare major like ``claude-opus-4-20250514``,
        whose 8-digit date segment is not a minor version).  Returns
        ``None`` for every name outside the grammar above: non-Claude
        models, legacy ``claude-3-5-sonnet-*`` names whose family slot
        holds a digit, zero-padded or non-numeric majors or minors
        (``claude-opus-04``, ``claude-opus-x``, ``claude-opus-5-04``),
        empty or non-numeric trailing segments (``claude-opus-5-``,
        ``claude-opus-5-junk``), extra segments beyond
        ``<minor>-<date>`` (``claude-opus-5-1-2``,
        ``claude-opus-5-1-20260301-777``), and non-final 8-digit date
        segments (``claude-opus-4-20250514-1``,
        ``claude-opus-5-20260301-99``).
    """
    parts = model_name.split("-")
    if len(parts) < 3 or len(parts) > 5 or parts[0] != "claude":
        return None
    family = parts[1]
    if not family.isalpha():
        return None
    major_str = parts[2]
    if not major_str.isdigit() or (len(major_str) > 1 and major_str[0] == "0"):
        return None
    tail = parts[3:]
    if tail and len(tail[-1]) == 8 and tail[-1].isdigit():
        tail = tail[:-1]  # a trailing 8-digit segment is a date, not a minor
    if len(tail) > 1:
        return None
    minor: int | None = None
    if tail:
        minor_str = tail[0]
        if (
            not minor_str.isdigit()
            or len(minor_str) == 8
            or (len(minor_str) > 1 and minor_str[0] == "0")
        ):
            return None
        minor = int(minor_str)
    return family, int(major_str), minor


def _uses_adaptive_thinking(model_name: str) -> bool:
    """Return True if the Claude model requires ``thinking.type=adaptive``.

    Consultation order:

    1. ``MODEL_INFO[model_name].adaptive_thinking`` when explicitly set
       (``True`` / ``False``) — the highest-priority source of truth.
       This is loaded from ``MODEL_INFO.json`` at import time so a JSON
       edit reconfigures the adapter without a code change.
    2. A version-aware heuristic on the model name, parsed as
       ``claude-<family>-<major>[-<minor>]``:

       * every modern-family model (opus/sonnet/haiku/fable) with major
         version >= 5 uses adaptive thinking (``claude-opus-5``,
         ``claude-fable-5``, future ``claude-*-6`` ... included), since
         Anthropic rejects ``thinking.type=enabled`` for these models;
       * within the 4.x generation only Opus 4.6 and later use
         adaptive; older Opus 4.x (4, 4.1, 4.5) and all Sonnet/Haiku
         4.x still use ``enabled``. An 8-digit date segment (e.g.
         ``claude-opus-4-20250514``) is not treated as a minor version.
    """
    from kiss.core.models.model_info import MODEL_INFO

    info = MODEL_INFO.get(model_name)
    if info is not None and info.adaptive_thinking is not None:
        return info.adaptive_thinking

    version = _parse_claude_version(model_name)
    if version is None:
        return False
    family, major, minor = version
    if family not in _THINKING_FAMILIES:
        return False
    if major >= 5:
        return True
    return family == "opus" and major == 4 and minor is not None and minor >= 6


def _supports_extended_thinking(model_name: str) -> bool:
    """Return True if the Claude model should send the ``thinking`` param.

    Consultation order:

    1. ``MODEL_INFO[model_name].extended_thinking`` when explicitly set —
       the highest-priority source of truth. Setting the flag to
       ``False`` lets ``MODEL_INFO.json`` opt a specific model out of
       extended thinking without a code change.
    2. A version-aware heuristic on the model name, parsed as
       ``claude-<family>-<major>``: every modern-family model
       (opus/sonnet/haiku/fable) with major version >= 4 supports
       extended thinking. Legacy ``claude-3*`` names (whose family slot
       holds a digit) and non-Claude names never match, so they stay
       non-thinking.

    The paper-analysed ``claude-fable-5`` failure — and the identical
    ``claude-opus-5`` regression after it — lived in the gap between an
    unset JSON flag and the previous hardcoded ``claude-*-4`` prefix
    allowlist: the adapter never sent the ``thinking`` param, so the
    model's reasoning stayed invisible (or came back encrypted-only and
    ``KISSAgent`` misread it as "empty response").
    """
    from kiss.core.models.model_info import MODEL_INFO

    info = MODEL_INFO.get(model_name)
    if info is not None and info.extended_thinking is not None:
        return info.extended_thinking

    version = _parse_claude_version(model_name)
    if version is None:
        return False
    family, major, _minor = version
    return family in _THINKING_FAMILIES and major >= 4


_AUDIO_FORMAT_TO_MIME: dict[str, str] = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "ogg": "audio/ogg",
    "webm": "audio/webm",
    "flac": "audio/flac",
    "aac": "audio/aac",
    "mp4": "audio/mp4",
}


def _parse_data_url(url: str) -> tuple[str, str] | None:
    """Split a base64 data URL into its media type and base64 payload.

    Args:
        url: A URL that may be a ``data:<media_type>;base64,<data>`` URL.

    Returns:
        A ``(media_type, base64_data)`` tuple, or ``None`` when *url* is not
        a base64 data URL.
    """
    if not url.startswith("data:"):
        return None
    header, _, data = url.partition(",")
    if ";base64" not in header or not data:
        return None
    media_type = header[len("data:"):].split(";", 1)[0]
    return media_type or "application/octet-stream", data


def _openai_part_to_anthropic_block(part: dict[str, Any]) -> dict[str, Any] | None:
    """Convert an OpenAI content part to the equivalent Anthropic block.

    OpenAI Chat Completions content parts (``image_url`` / ``file`` /
    ``input_audio``) enter the conversation when it is handed off from an
    OpenAI-schema model (e.g. via the Sorcar ``set_model`` tool).  The
    Anthropic Messages API instead expects ``image`` / ``document`` blocks;
    audio has no Anthropic equivalent and is transcribed via Whisper when
    possible.

    Args:
        part: The OpenAI content-part dict.

    Returns:
        The equivalent Anthropic block dict, or ``None`` when the part
        cannot be represented (in which case it is dropped with a warning).
    """
    part_type = part.get("type")
    if part_type == "image_url":
        url = (part.get("image_url") or {}).get("url", "")
        parsed = _parse_data_url(url)
        if parsed is not None:
            media_type, data = parsed
            return {
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": data},
            }
        if url:
            return {"type": "image", "source": {"type": "url", "url": url}}
        logger.warning("Dropping unconvertible OpenAI image_url part.")
        return None
    if part_type == "file":
        file_data = (part.get("file") or {}).get("file_data", "")
        parsed = _parse_data_url(file_data)
        if parsed is not None:
            media_type, data = parsed
            return {
                "type": "document",
                "source": {"type": "base64", "media_type": media_type, "data": data},
            }
        logger.warning("Dropping unconvertible OpenAI file part.")
        return None
    if part_type == "input_audio":
        audio = part.get("input_audio") or {}
        fmt = audio.get("format", "")
        mime_type = _AUDIO_FORMAT_TO_MIME.get(fmt, f"audio/{fmt}" if fmt else "audio/mpeg")
        try:
            text = transcribe_audio(base64.b64decode(audio.get("data", "")), mime_type)
            return {"type": "text", "text": f"[Audio transcription]\n{text}"}
        except Exception:
            logger.warning(
                "Anthropic does not support input_audio content parts and "
                "automatic transcription failed; dropping.",
            )
            return None
    logger.warning("Dropping unconvertible OpenAI %s content part.", part_type)
    return None


def _tool_calls_to_tool_use_blocks(tool_calls: list[Any]) -> list[dict[str, Any]]:
    """Convert OpenAI ``tool_calls`` entries into Anthropic ``tool_use`` blocks.

    Args:
        tool_calls: The ``tool_calls`` list of an OpenAI-format assistant
            message (dicts or SDK objects with ``id`` / ``function`` attrs).

    Returns:
        The equivalent list of Anthropic ``tool_use`` block dicts.
    """
    blocks: list[dict[str, Any]] = []
    for tc in tool_calls:
        if isinstance(tc, dict):
            fn = tc.get("function") or {}
            call_id = tc.get("id", "")
            name = fn.get("name", "")
            arguments = fn.get("arguments", "")
        else:
            fn = getattr(tc, "function", None)
            call_id = getattr(tc, "id", "")
            name = getattr(fn, "name", "") if fn is not None else ""
            arguments = getattr(fn, "arguments", "") if fn is not None else ""
        if isinstance(arguments, str):
            try:
                input_dict = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError:
                logger.debug("Exception caught", exc_info=True)
                input_dict = {}
        else:
            input_dict = arguments or {}
        if not isinstance(input_dict, dict):
            input_dict = {}
        blocks.append(
            {"type": "tool_use", "id": call_id, "name": name, "input": input_dict}
        )
    return blocks


def _attachments_to_blocks(attachments: list[Attachment]) -> list[dict[str, Any]]:
    """Convert :class:`Attachment` objects into Anthropic content blocks.

    Images become ``image`` blocks, PDFs become ``document`` blocks, and
    audio is transcribed to text via Whisper when possible.  Unsupported
    MIME types (e.g. video) are dropped with a warning.

    Args:
        attachments: The attachments to convert.

    Returns:
        list[dict[str, Any]]: The equivalent Anthropic content blocks.
    """
    blocks: list[dict[str, Any]] = []
    for att in attachments:
        source = {
            "type": "base64",
            "media_type": att.mime_type,
            "data": att.to_base64(),
        }
        if att.mime_type.startswith("image/"):
            blocks.append({"type": "image", "source": source})
        elif att.mime_type == "application/pdf":
            blocks.append({"type": "document", "source": source})
        elif att.mime_type.startswith("audio/"):
            try:
                text = transcribe_audio(att.data, att.mime_type)
                blocks.append(
                    {"type": "text", "text": f"[Audio transcription]\n{text}"}
                )
            except Exception as exc:
                logger.warning(
                    "Anthropic does not support %s attachments and "
                    "automatic transcription failed (%s); skipping.",
                    att.mime_type,
                    exc,
                )
        else:
            logger.warning(
                "Anthropic does not support %s attachments; skipping.",
                att.mime_type,
            )
    return blocks


def _content_as_block_list(content: Any) -> list[dict[str, Any]]:
    """Return message content as a list of Anthropic content blocks.

    Args:
        content: A message ``content`` value (string or block list).

    Returns:
        The content as a block list, wrapping strings in a text block.
    """
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return list(content)


class AnthropicModel(Model):
    """A model that uses Anthropic's Messages API (Claude)."""

    def __init__(
        self,
        model_name: str,
        api_key: str,
        model_config: dict[str, Any] | None = None,
        token_callback: TokenCallback | None = None,
        thinking_callback: ThinkingCallback | None = None,
    ):
        """Initialize an AnthropicModel instance.

        Args:
            model_name: The name of the Claude model to use.
            api_key: The Anthropic API key for authentication.
            model_config: Optional dictionary of model configuration parameters.
            token_callback: Optional callback invoked with each streamed text token.
            thinking_callback: Optional callback invoked with ``True`` when a
                thinking block starts and ``False`` when it ends.
        """
        super().__init__(
            model_name,
            model_config=model_config,
            token_callback=token_callback,
            thinking_callback=thinking_callback,
        )
        self.api_key = api_key
        self._stream_stall_timeout = float(
            self.model_config.get("stream_stall_timeout", DEFAULT_STREAM_STALL_TIMEOUT)
        )

    def initialize(self, prompt: str, attachments: list[Attachment] | None = None) -> None:
        """Initializes the conversation with an initial user prompt.

        Args:
            prompt: The initial user prompt to start the conversation.
            attachments: Optional list of file attachments (images, PDFs, audio,
                video) to include. Audio attachments are automatically
                transcribed to text via OpenAI Whisper when an ``OPENAI_API_KEY``
                is available; otherwise they are skipped with a warning.  Video
                attachments are always skipped.
        """
        self.client = Anthropic(
            api_key=self.api_key,
            timeout=httpx.Timeout(self._stream_stall_timeout, connect=_CONNECT_TIMEOUT),
            max_retries=_MAX_RETRIES,
        )
        content: str | list[dict[str, Any]] = prompt
        if attachments:
            blocks = _attachments_to_blocks(attachments)
            blocks.append({"type": "text", "text": prompt})
            content = blocks
        self.conversation = [{"role": "user", "content": content}]

    def _normalize_content_blocks(self, content: Any) -> list[dict[str, Any]]:
        """Normalize Anthropic content blocks to JSON-serializable dicts.

        Drops text blocks whose text is empty or whitespace-only, because
        the Anthropic API rejects them with ``invalid_request_error:
        messages: text content blocks must contain non-whitespace text``.

        Args:
            content: The content blocks from an Anthropic response.

        Returns:
            list[dict[str, Any]]: Normalized content blocks as dictionaries.
        """
        blocks: list[dict[str, Any]] = []
        if content is None:
            return blocks
        for block in content:
            if isinstance(block, dict):
                dict_block_type = block.get("type")
                if dict_block_type == "text" and not block.get("text", "").strip():
                    continue
                if dict_block_type in ("image_url", "file", "input_audio"):
                    converted = _openai_part_to_anthropic_block(block)
                    if converted is not None:
                        blocks.append(converted)
                    continue
                blocks.append(block)
                continue
            block_type = getattr(block, "type", None)
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
                dumped = block.model_dump(exclude_none=True)
                if dumped.get("type") == "text" and not dumped.get("text", "").strip():
                    continue
                blocks.append(dumped)
            else:
                text = str(block)
                if not text.strip():
                    continue
                blocks.append({"type": "text", "text": text})
        return blocks

    def _extract_text_from_blocks(self, blocks: list[dict[str, Any]]) -> str:
        """Extract text content from normalized content blocks.

        Args:
            blocks: List of normalized content blocks.

        Returns:
            str: Concatenated text from all text blocks.
        """
        return "".join(b.get("text", "") for b in blocks if b.get("type") == "text")

    def _build_anthropic_tools_schema(
        self,
        openai_schema: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Convert an OpenAI-format tools schema to Anthropic format.

        Args:
            openai_schema: Tool schema list in OpenAI format.

        Returns:
            list[dict[str, Any]]: A list of tool schemas in Anthropic format.
        """
        tools = []
        for tool in openai_schema:
            fn = tool.get("function", {})
            tools.append(
                {
                    "name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
                }
            )
        return tools

    def _normalize_conversation_for_api(
        self,
        conversation: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Normalize all messages in a conversation before sending to the API.

        Ensures that all text content blocks are non-whitespace and that no
        messages contain only whitespace-only text blocks.  Also converts
        OpenAI Chat Completions-format entries (assistant ``tool_calls``
        arrays, ``role="tool"`` messages, ``image_url`` / ``file`` /
        ``input_audio`` content parts) — which enter the conversation when it
        is handed off from an OpenAI-schema model, e.g. via the Sorcar
        ``set_model`` tool — into the Anthropic Messages equivalents
        (``tool_use`` / ``tool_result`` / ``image`` / ``document`` blocks) so
        the API does not reject them.  OpenAI Responses-API items (handed
        off from an ``OpenAICompatibleModel2``) are converted first via
        :func:`responses_items_to_chat_messages`.  Consecutive user turns are
        merged so that ``tool_result`` blocks land in the message immediately
        following their ``tool_use`` turn, as the Anthropic API requires.

        Args:
            conversation: The conversation to normalize.

        Returns:
            list[dict[str, Any]]: The normalized conversation.
        """
        normalized: list[dict[str, Any]] = []
        for msg in responses_items_to_chat_messages(conversation):
            for converted in self._normalize_message_for_api(msg):
                prev = normalized[-1] if normalized else None
                if (
                    prev is not None
                    and prev.get("role") == "user"
                    and converted.get("role") == "user"
                ):
                    prev["content"] = _content_as_block_list(
                        prev["content"]
                    ) + _content_as_block_list(converted["content"])
                else:
                    normalized.append(converted)
        return normalized

    def _normalize_message_for_api(self, msg: dict[str, Any]) -> list[dict[str, Any]]:
        """Normalize a single message into Anthropic Messages-format messages.

        A message may expand to zero messages (all content filtered out) or
        one message.  OpenAI Chat Completions-format entries — which enter
        the conversation when it is handed off from an OpenAI-schema model,
        e.g. via the Sorcar ``set_model`` tool — are converted to their
        Anthropic equivalents:

        * ``role="system"`` messages are dropped here;
          ``_build_create_kwargs`` hoists their text into the top-level
          ``system`` parameter (the Messages API rejects the "system" role).
        * ``role="tool"`` messages become user messages carrying a
          ``tool_result`` block.
        * assistant messages with ``tool_calls`` become assistant messages
          whose content is text + ``tool_use`` blocks.

        Args:
            msg: The conversation message to normalize.

        Returns:
            list[dict[str, Any]]: Anthropic Messages-format messages.
        """
        role = msg.get("role")
        if role == "system":
            return []
        if role == "tool":
            block: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id", ""),
            }
            content = msg.get("content")
            if isinstance(content, list):
                nested = self._normalize_content_blocks(content)
                if nested:
                    block["content"] = nested
            elif content is not None and str(content).strip():
                block["content"] = str(content)
            return [{"role": "user", "content": [block]}]

        msg_copy = msg.copy()
        attachments = msg_copy.pop("attachments", None)
        if attachments:
            att_blocks = _attachments_to_blocks(attachments)
            prior = msg_copy.get("content")
            if isinstance(prior, str):
                if prior.strip():
                    att_blocks.append({"type": "text", "text": prior})
            elif isinstance(prior, list):
                att_blocks.extend(prior)
            msg_copy["content"] = att_blocks
        content = msg_copy.get("content")
        tool_calls = msg_copy.pop("tool_calls", None)
        if tool_calls:
            blocks: list[dict[str, Any]] = []
            if isinstance(content, str):
                if content.strip():
                    blocks.append({"type": "text", "text": content})
            elif isinstance(content, list):
                blocks.extend(self._normalize_content_blocks(content))
            blocks.extend(_tool_calls_to_tool_use_blocks(tool_calls))
            return [{"role": msg_copy.get("role", "assistant"), "content": blocks}]

        if isinstance(content, str):
            if content.strip():
                return [msg_copy]
            return []
        if isinstance(content, list):
            normalized_blocks = self._normalize_content_blocks(content)
            if normalized_blocks:
                msg_copy["content"] = normalized_blocks
                return [msg_copy]
            return []
        return []

    def _build_create_kwargs(self, tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """Build keyword arguments for the Anthropic API create call.

        Args:
            tools: Optional list of tool schemas to include.

        Returns:
            dict[str, Any]: The keyword arguments for the API call.
        """
        kwargs = {
            key: value
            for key, value in self.model_config.items()
            if key not in FRAMEWORK_ONLY_CONFIG_KEYS
        }
        enable_cache = self.model_config.get("enable_cache", True)
        system_instruction = self.model_config.get("system_instruction")

        system_texts: list[str] = [system_instruction] if system_instruction else []
        for msg in self.conversation:
            if msg.get("role") != "system":
                continue
            content = msg.get("content")
            if isinstance(content, list):
                content = "".join(
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                )
            if isinstance(content, str) and content.strip() and content not in system_texts:
                system_texts.append(content)
        if system_texts:
            system_instruction = "\n\n".join(system_texts)

        max_tokens = kwargs.pop("max_tokens", None)
        if max_tokens is None:
            max_tokens = kwargs.pop("max_completion_tokens", None)
        user_set_max_tokens = max_tokens is not None
        if max_tokens is None:
            max_tokens = 16384

        if "stop" in kwargs and "stop_sequences" not in kwargs:
            stop_val = kwargs.pop("stop")
            if isinstance(stop_val, str):
                kwargs["stop_sequences"] = [stop_val]
            elif isinstance(stop_val, list):
                kwargs["stop_sequences"] = stop_val

        kwargs = self._keep_supported_request_params(
            kwargs,
            _ANTHROPIC_REQUEST_PARAMS,
            "Anthropic Messages",
        )

        if "thinking" not in kwargs and _supports_extended_thinking(self.model_name):
            if not user_set_max_tokens:
                version = _parse_claude_version(self.model_name)
                is_opus = version is not None and version[0] == "opus"
                max_tokens = 65536 if is_opus else 64000
            if _uses_adaptive_thinking(self.model_name):
                kwargs["thinking"] = {"type": "adaptive", "display": "summarized"}
            else:
                budget = min(10000, max_tokens - 1)
                if budget >= 1024:
                    kwargs["thinking"] = {
                        "type": "enabled",
                        "budget_tokens": budget,
                    }

        if "thinking" in kwargs:
            existing_headers = kwargs.get("extra_headers") or {}
            beta_header = existing_headers.get("anthropic-beta", "")
            beta_token = "interleaved-thinking-2025-05-14"
            if beta_token not in beta_header:
                merged_beta = (
                    f"{beta_header},{beta_token}" if beta_header else beta_token
                )
                kwargs["extra_headers"] = {
                    **existing_headers,
                    "anthropic-beta": merged_beta,
                }

        normalized_messages = self._normalize_conversation_for_api(self.conversation)
        if not normalized_messages:
            raise KISSError(
                "Cannot build API request: all messages are whitespace-only. "
                "At least one message with non-whitespace text content is required."
            )
        kwargs.update(
            {
                "model": self.model_name,
                "messages": normalized_messages,
                "max_tokens": max_tokens,
            }
        )
        if system_instruction:
            kwargs["system"] = system_instruction
        if tools:
            kwargs["tools"] = tools
            if "tool_choice" not in kwargs and "thinking" not in kwargs:
                kwargs["tool_choice"] = {"type": "any"}

        if enable_cache:
            kwargs["cache_control"] = {"type": "ephemeral"}

        return kwargs

    def _append_assistant_message(self, blocks: list[dict[str, Any]], content: str) -> None:
        """Append the assistant response to the conversation when non-empty.

        Prefers the normalized *blocks* over the plain *content* string and
        skips the append entirely when both are empty.

        Args:
            blocks: Normalized content blocks from the response.
            content: Text extracted from the response blocks.
        """
        msg_content: list[dict[str, Any]] | str = blocks if blocks else content
        if msg_content:
            self.conversation.append({"role": "assistant", "content": msg_content})

    def _create_message(self, kwargs: dict[str, Any]) -> Any:
        """Create a message, streaming tokens to the callback when set.

        Aborts with a clear, retryable :class:`TimeoutError` when the
        stream stalls for :attr:`_stream_stall_timeout` seconds at either
        level:

        * **byte level** — the client's httpx read timeout raises
          ``httpx.ReadTimeout`` mid-iteration (SDK does not retry it, and
          its message is often empty) or ``anthropic.APITimeoutError``
          when the response headers never arrive (SDK retries
          ``_MAX_RETRIES`` times first);
        * **event level** — :class:`StreamAbortWatchdog` closes the
          response when no SSE event is yielded in time, catching wedged
          requests that keep the connection alive with ``ping`` events
          (which the SDK filters out before yielding).

        ``KISSAgent._run_agentic_loop`` treats ``TimeoutError`` as
        retryable and re-asks the model instead of hanging forever.  An
        open thinking bracket is closed before raising so the UI does not
        stay in "thinking" mode across the retry.

        Args:
            kwargs: Keyword arguments for the Anthropic API call.

        Returns:
            The raw Anthropic response message.

        Raises:
            TimeoutError: When the streaming connection delivers no data
                (or no events) for ``stream_stall_timeout`` seconds.
        """
        watchdog: StreamAbortWatchdog | None = None
        in_thinking = False
        thinking_started = False
        try:
            with self.client.messages.stream(**kwargs) as stream:
                watchdog = StreamAbortWatchdog(
                    stream,
                    stall_timeout=self._stream_stall_timeout,
                    stop_event=stop_signal.get_thread_stop_event(),
                    name="anthropic-stream-abort-watchdog",
                )
                try:
                    for event in stream:
                        watchdog.beat()
                        if self.token_callback is None:
                            continue
                        if event.type == "content_block_start":
                            block = getattr(event, "content_block", None)
                            if block and getattr(block, "type", "") == "thinking":
                                in_thinking = True
                                thinking_started = False
                        elif event.type == "content_block_delta":
                            delta = event.delta
                            delta_type = getattr(delta, "type", "")
                            if delta_type == "thinking_delta":
                                text = getattr(delta, "thinking", "")
                                if text:
                                    if in_thinking and not thinking_started:
                                        self._invoke_thinking_callback(True)
                                        thinking_started = True
                                    self._invoke_token_callback(text)
                            elif delta_type == "text_delta":
                                self._invoke_token_callback(getattr(delta, "text", ""))
                        elif event.type == "content_block_stop":
                            if in_thinking:
                                in_thinking = False
                                if thinking_started:
                                    self._invoke_thinking_callback(False)
                                    thinking_started = False
                    # An aborted socket ends the iterator at EOF instead
                    # of raising, so the abort has to be reported here
                    # too — otherwise get_final_message() would surface
                    # it as a confusing "incomplete message" error.
                    if watchdog.stopped:
                        raise self._stop_error(thinking_started)
                    if watchdog.stalled:
                        raise self._stall_error(thinking_started)
                    return stream.get_final_message()
                finally:
                    watchdog.stop()
        except (httpx.TimeoutException, APITimeoutError) as exc:
            if self._stream_was_stopped(watchdog):
                raise self._stop_error(thinking_started) from exc
            raise self._stall_error(thinking_started) from exc
        except TimeoutError:
            # Already the stall error raised after the loop below; its
            # thinking bracket is closed, so re-wrapping it would emit a
            # second thinking_callback(False).
            raise
        except Exception as exc:
            if self._stream_was_stopped(watchdog):
                raise self._stop_error(thinking_started) from exc
            if watchdog is not None and watchdog.stalled:
                raise self._stall_error(thinking_started) from exc
            raise

    @staticmethod
    def _stream_was_stopped(watchdog: StreamAbortWatchdog | None) -> bool:
        """Return whether the user stopped the task during this request.

        The watchdog reports a stop it acted on, but it only exists once
        the response headers have arrived: a request that is silent
        BEFORE that (the SDK's own connect/read timeout territory) fails
        with ``watchdog is None``, and reporting that as a retryable
        stall would make the agentic loop re-ask the model on behalf of
        a task the user already stopped.  Asking the thread's stop
        signal directly covers both windows.

        Args:
            watchdog: The stall watchdog for this request, or ``None``
                when the stream never opened.

        Returns:
            ``True`` when the request must unwind as a user stop.
        """
        if watchdog is not None and watchdog.stopped:
            return True
        return stop_signal.stop_requested()

    def _stop_error(self, thinking_started: bool) -> KeyboardInterrupt:
        """Build the stop error for a stream the user aborted.

        A user stop must NOT surface as the retryable
        :class:`TimeoutError` a stall produces — the agentic loop would
        re-ask the model and the task would keep running.
        ``KeyboardInterrupt`` is the same signal ``_check_stop`` raises,
        so the whole stack unwinds into the normal "Task stopped by
        user" path.

        Args:
            thinking_started: Whether ``thinking_callback(True)`` was
                emitted without its matching ``False``.

        Returns:
            The ``KeyboardInterrupt`` for the caller to raise.
        """
        if thinking_started:
            self._invoke_thinking_callback(False)
        return KeyboardInterrupt("Agent stop requested")

    def _stall_error(self, thinking_started: bool) -> TimeoutError:
        """Build the retryable stall error, closing any open thinking bracket.

        The message itself comes from
        :func:`~kiss.core.models.stream_abort.stall_error`, so this
        transport's hand-run watchdog loop reports a stall in exactly the
        words the wrapped transports do; only the offending model is
        added, because a stall is usually diagnosed from a log line that
        does not say which model was being asked.

        The extra side effect is the closing ``thinking_callback(False)``:
        a stall can strike mid-thinking, and without it the printer/UI
        would render everything after the retry as "thinking" forever.

        Args:
            thinking_started: Whether ``thinking_callback(True)`` was
                emitted without its matching ``False``.

        Returns:
            The ``TimeoutError`` for the caller to raise.
        """
        if thinking_started:
            self._invoke_thinking_callback(False)
        return TimeoutError(
            f"Anthropic model {self.model_name}: "
            f"{stall_error(self._stream_stall_timeout)}"
        )

    def _raise_on_refusal(self, response: Any) -> None:
        """Raise :class:`ModelRefusalError` when the model refused the request.

        Adaptive-thinking Claude models (fable-5 in production, task
        ``daa89a7e``/``c3cd9c95`` in ``~/.kiss/sorcar.db``) can return
        ``stop_reason="refusal"`` with an EMPTY ``content`` list when their
        safety layer declines an otherwise benign prompt (observed on
        security-research text that opus-4-8 answers normally).  Without
        this check the empty turn propagated as ``("", [])``, KISSAgent
        burned a useless "MUST have at least one function call" retry (a
        refusal is deterministic for identical content), and the eventual
        fallback swap was misreported as "repeated empty responses" — a
        misleading adapter-bug diagnosis.

        Args:
            response: The raw Anthropic response message.

        Raises:
            ModelRefusalError: When ``response.stop_reason`` is ``"refusal"``.
        """
        if getattr(response, "stop_reason", None) == "refusal":
            raise ModelRefusalError(
                f"Model {self.model_name} refused the request for safety "
                f'reasons (stop_reason="refusal", empty response). Retrying '
                f"the identical request will keep failing; rephrase the "
                f"prompt or use a different model."
            )

    def generate(self) -> tuple[str, Any]:  # pragma: no cover – API call
        """Generates content from the current conversation.

        Returns:
            tuple[str, Any]: A tuple of (generated_text, raw_response).
        """
        kwargs = self._build_create_kwargs()
        response = self._create_message(kwargs)
        self._raise_on_refusal(response)

        blocks = self._normalize_content_blocks(getattr(response, "content", None))
        content = self._extract_text_from_blocks(blocks)
        self._append_assistant_message(blocks, content)
        return content, response

    def generate_and_process_with_tools(  # pragma: no cover – API call
        self,
        function_map: dict[str, Callable[..., Any]],
        tools_schema: list[dict[str, Any]] | None = None,
    ) -> tuple[list[dict[str, Any]], str, Any]:
        """Generates content with tools and processes the response.

        Args:
            function_map: Dictionary mapping function names to callable functions.
            tools_schema: Optional pre-built OpenAI-format tool schema list.

        Returns:
            tuple[list[dict[str, Any]], str, Any]: A tuple of
                (function_calls, response_text, raw_response).
        """
        resolved = self._resolve_openai_tools_schema(function_map, tools_schema)
        tools = self._build_anthropic_tools_schema(resolved)
        kwargs = self._build_create_kwargs(tools=tools or None)
        response = self._create_message(kwargs)
        self._raise_on_refusal(response)

        stop_reason = getattr(response, "stop_reason", None)
        blocks = self._normalize_content_blocks(getattr(response, "content", None))

        if stop_reason == "max_tokens":
            blocks = [b for b in blocks if b.get("type") != "tool_use"]

        content = self._extract_text_from_blocks(blocks)

        function_calls: list[dict[str, Any]] = []
        for b in blocks:
            if b.get("type") == "tool_use":
                function_calls.append(
                    {
                        "id": b.get("id", ""),
                        "name": b.get("name", ""),
                        "arguments": b.get("input", {}) or {},
                    }
                )

        self._append_assistant_message(blocks, content)
        return function_calls, content, response

    def add_function_results_to_conversation_and_return(
        self, function_results: list[tuple[str, dict[str, Any]]]
    ) -> None:
        """Add tool results to the conversation as ``tool_result`` blocks.

        Anthropic is the one provider whose tool result may carry the bytes
        directly, so binary payloads a tool produced become image blocks
        inside the ``tool_result`` instead of a follow-up user message.

        Args:
            function_results: List of (func_name, result_dict) tuples.
                result_dict can contain:
                - "result": The result content (a string, or any
                  JSON-encodable value)
                - "tool_use_id": Optional explicit tool_use_id to use
        """
        tool_call_ids = self._find_tool_call_ids_from_last_assistant()

        tool_results_blocks: list[dict[str, Any]] = []
        for i, (func_name, result_dict) in enumerate(function_results):
            text, attachments = self.tool_result_text_and_attachments(result_dict)
            tool_use_id = self.tool_result_call_id(
                result_dict, i, tool_call_ids, func_name, prefix="toolu"
            )
            content: str | list[dict[str, Any]] = text
            if attachments:
                blocks: list[dict[str, Any]] = []
                if text.strip():
                    blocks.append({"type": "text", "text": text})
                blocks.extend(_attachments_to_blocks(attachments))
                content = blocks or text
            tool_results_blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": content,
                }
            )

        self.conversation.append({"role": "user", "content": tool_results_blocks})

    def extract_input_output_token_counts_from_response(
        self, response: Any
    ) -> tuple[int, int, int, int, int]:
        """Extracts token counts from an Anthropic API response.

        Returns:
            (input_tokens, output_tokens, cache_read_tokens,
            cache_write_5m_tokens, cache_write_1h_tokens).
        """
        if hasattr(response, "usage") and response.usage:
            cache_write_5m, cache_write_1h = cache_creation_tokens(
                response.usage, _attribute_field
            )
            return (
                getattr(response.usage, "input_tokens", 0) or 0,
                getattr(response.usage, "output_tokens", 0) or 0,
                getattr(response.usage, "cache_read_input_tokens", 0) or 0,
                cache_write_5m,
                cache_write_1h,
            )
        return 0, 0, 0, 0, 0

    def get_embedding(self, text: str, embedding_model: str | None = None) -> list[float]:
        """Generates an embedding vector for the given text.

        Args:
            text: The text to generate an embedding for.
            embedding_model: Optional model name (not used by Anthropic).

        Raises:
            KISSError: Anthropic does not provide an embeddings API.
        """
        raise KISSError("Anthropic does not provide an embeddings API.")
