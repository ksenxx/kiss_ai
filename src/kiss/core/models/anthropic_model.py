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

from anthropic import Anthropic

from kiss.core.kiss_error import KISSError, ModelRefusalError
from kiss.core.models.model import (
    Attachment,
    Model,
    ThinkingCallback,
    TokenCallback,
    parse_binary_attachments,
    responses_items_to_chat_messages,
    transcribe_audio,
)

logger = logging.getLogger(__name__)


def _anthropic_cache_creation_tokens(usage: Any) -> tuple[int, int]:
    """Return Anthropic 5-minute and 1-hour cache-creation token counts."""
    cache_creation = getattr(usage, "cache_creation", None)
    if cache_creation is not None:
        five_minute = getattr(cache_creation, "ephemeral_5m_input_tokens", 0) or 0
        one_hour = getattr(cache_creation, "ephemeral_1h_input_tokens", 0) or 0
        return five_minute, one_hour
    aggregate = getattr(usage, "cache_creation_input_tokens", 0) or 0
    return 0, aggregate


def _uses_adaptive_thinking(model_name: str) -> bool:
    """Return True if the Claude model requires ``thinking.type=adaptive``.

    Consultation order:

    1. ``MODEL_INFO[model_name].adaptive_thinking`` when explicitly set
       (``True`` / ``False``) — the source of truth for
       ``claude-fable-*``, ``claude-sonnet-5`` and any other new-family
       Claude models whose name does not fit the legacy prefix
       heuristic.  This is loaded from ``MODEL_INFO.json`` at import
       time so a JSON edit reconfigures the adapter without a code
       change.
    2. The legacy prefix heuristic: newer Claude Opus models (4.6 and
       later) no longer support ``thinking.type=enabled`` and must use
       ``adaptive`` instead. Older Opus 4.x models (4, 4.1, 4.5) still
       use ``enabled``. Sonnet/Haiku 4 models continue to use
       ``enabled`` as before.
    """
    # Deferred import to avoid a cycle between ``model_info`` (which
    # imports the model classes lazily) and this module.
    from kiss.core.models.model_info import MODEL_INFO

    info = MODEL_INFO.get(model_name)
    if info is not None and info.adaptive_thinking is not None:
        return info.adaptive_thinking

    prefix = "claude-opus-4-"
    if not model_name.startswith(prefix):
        return False
    suffix = model_name[len(prefix):]
    minor_str = suffix.split("-", 1)[0]
    if len(minor_str) == 8 and minor_str.isdigit():
        # Date-stamped official id (e.g. ``claude-opus-4-20250514``): the
        # token is a release date, not a minor version — this is Opus 4.0,
        # which only supports ``thinking.type=enabled``.
        return False
    try:
        minor = int(minor_str)
    except ValueError:
        return False
    return minor >= 6


def _supports_extended_thinking(model_name: str) -> bool:
    """Return True if the Claude model should send the ``thinking`` param.

    Consultation order:

    1. ``MODEL_INFO[model_name].extended_thinking`` when explicitly set —
       the source of truth for new-family Claude models whose name is
       not covered by the legacy prefix allowlist below (e.g.
       ``claude-fable-5``, ``claude-sonnet-5``). Setting the flag to
       ``False`` also lets ``MODEL_INFO.json`` opt a specific model out
       of extended thinking without a code change.
    2. Legacy prefix allowlist: every ``claude-{opus,sonnet,haiku}-4``
       model supports extended thinking.

    The paper-analysed ``claude-fable-5`` failure lived in the gap
    between these two rules: its name does not match the prefix
    allowlist, so before this helper existed the adapter never sent the
    ``thinking`` param and the model returned encrypted-only reasoning
    turns that ``KISSAgent`` misread as "empty response".
    """
    from kiss.core.models.model_info import MODEL_INFO

    info = MODEL_INFO.get(model_name)
    if info is not None and info.extended_thinking is not None:
        return info.extended_thinking
    return model_name.startswith(
        ("claude-opus-4", "claude-sonnet-4", "claude-haiku-4")
    )


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
        self.client = Anthropic(api_key=self.api_key)
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
                # Drop pre-existing whitespace-only text dicts too.
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
            # Gemini hand-off: lift the Attachment objects into Anthropic
            # content blocks (the API rejects unknown message fields).
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

        # If content is a string, ensure it's non-whitespace
        if isinstance(content, str):
            if content.strip():
                return [msg_copy]
            # Skip messages with whitespace-only string content
            return []
        # If content is a list of blocks, normalize them
        if isinstance(content, list):
            normalized_blocks = self._normalize_content_blocks(content)
            if normalized_blocks:
                msg_copy["content"] = normalized_blocks
                return [msg_copy]
            # Skip messages where all blocks were dropped
            return []
        return []

    def _build_create_kwargs(self, tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """Build keyword arguments for the Anthropic API create call.

        Args:
            tools: Optional list of tool schemas to include.

        Returns:
            dict[str, Any]: The keyword arguments for the API call.
        """
        kwargs = self.model_config.copy()
        enable_cache = kwargs.pop("enable_cache", True)
        system_instruction = kwargs.pop("system_instruction", None)
        # ``reasoning_effort`` and ``use_responses_api`` are OpenAI-specific
        # knobs (the factory auto-defaults the former into ``model_config``
        # for gpt-5.x reasoning models; the latter forces/disables the
        # /v1/responses delegation); they arrive here verbatim when a live
        # conversation (and its config) is handed over by the Sorcar
        # ``set_model`` tool.  The Anthropic API rejects unknown kwargs, so
        # drop them — the extended-thinking default below already enables
        # native reasoning for Claude 4+ models.
        kwargs.pop("reasoning_effort", None)
        kwargs.pop("use_responses_api", None)

        # Hoist OpenAI-style ``role="system"`` messages (present when the
        # conversation was handed off from an OpenAI-schema model, e.g. via
        # the Sorcar ``set_model`` tool) into the top-level ``system``
        # parameter; the Anthropic Messages API rejects the "system" role.
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

        if "thinking" not in kwargs and _supports_extended_thinking(self.model_name):
            if not user_set_max_tokens:
                max_tokens = 65536 if self.model_name.startswith("claude-opus-4") else 64000
            if _uses_adaptive_thinking(self.model_name):
                # ``display`` defaults to "omitted" on adaptive-thinking
                # models (fable-5, mythos-5, sonnet-5, opus-4-7/4-8): the
                # API then returns thinking blocks with an EMPTY ``thinking``
                # field (encrypted signature only) and emits no
                # ``thinking_delta`` stream events, so no thinking tokens
                # are ever revealed to the user.  Request the readable
                # summary explicitly.
                kwargs["thinking"] = {"type": "adaptive", "display": "summarized"}
            else:
                # The API requires ``max_tokens > budget_tokens`` and
                # ``budget_tokens >= 1024``.  Cap the budget below the
                # user's max_tokens and skip thinking entirely when there
                # is no room for the minimum budget.
                budget = min(10000, max_tokens - 1)
                if budget >= 1024:
                    kwargs["thinking"] = {
                        "type": "enabled",
                        "budget_tokens": budget,
                    }

        # When extended thinking is enabled, request interleaved thinking via
        # the anthropic-beta header so the model emits between-tool-call
        # reasoning as ``thinking`` content blocks (routed to the Thoughts
        # panel) instead of as plain ``text`` blocks (which would render in
        # the main response area).
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
                # KISSAgent's ReAct loop requires every agentic turn to
                # produce a tool call (``finish`` is always present), so
                # non-thinking models force ``tool_choice=any`` to prevent
                # tool-less turns.  When thinking is active (``enabled`` or
                # ``adaptive``) tool use only supports ``tool_choice``
                # ``auto``/``none``: ``enabled`` rejects forced tool use with
                # a 400, and adaptive models (fable-5, sonnet-5, opus-4-7/4-8)
                # silently DISABLE thinking for the request ("graceful
                # thinking degradation") — the response then contains only
                # ``tool_use`` blocks and no thinking is ever revealed.
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

    def _create_message(self, kwargs: dict[str, Any]) -> Any:  # pragma: no cover – API call
        """Create a message, streaming tokens to the callback when set.

        Args:
            kwargs: Keyword arguments for the Anthropic API call.

        Returns:
            The raw Anthropic response message.
        """
        with self.client.messages.stream(**kwargs) as stream:
            if self.token_callback is not None:
                in_thinking = False
                thinking_started = False
                for event in stream:
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
            return stream.get_final_message()

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
        """Add tool results to the conversation.

        Args:
            function_results: List of (func_name, result_dict) tuples.
                result_dict can contain:
                - "result": The result content string
                - "tool_use_id": Optional explicit tool_use_id to use
        """
        tool_call_ids = self._find_tool_call_ids_from_last_assistant()

        tool_results_blocks: list[dict[str, Any]] = []
        for i, (func_name, result_dict) in enumerate(function_results):
            result_content = result_dict.get("result", str(result_dict))
            if self.usage_info_for_messages:
                result_content = f"{result_content}\n\n{self.usage_info_for_messages}"

            tool_use_id = result_dict.get("tool_use_id")
            if tool_use_id is None and i < len(tool_call_ids):
                tool_use_id = tool_call_ids[i][1]
            if tool_use_id is None:
                tool_use_id = f"toolu_{func_name}_{i}"

            plain_text, attachments = parse_binary_attachments(result_content)
            if attachments:
                content_blocks: list[dict[str, Any]] = []
                if plain_text.strip():
                    content_blocks.append({"type": "text", "text": plain_text})
                content_blocks.extend(_attachments_to_blocks(attachments))
                if not content_blocks:
                    content_blocks.append(
                        {"type": "text", "text": plain_text or result_content}
                    )
                tool_results_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": content_blocks,
                    }
                )
            else:
                tool_results_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": result_content,
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
            cache_write_5m, cache_write_1h = _anthropic_cache_creation_tokens(response.usage)
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
