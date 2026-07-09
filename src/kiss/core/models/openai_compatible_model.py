# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""OpenAI-compatible model implementation for custom endpoints."""

import json
import logging
import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from openai import BadRequestError, OpenAI

if TYPE_CHECKING:  # pragma: no cover – import cycle avoided at runtime
    from kiss.core.models.openai_compatible_model2 import OpenAICompatibleModel2

from kiss.core.kiss_error import KISSError
from kiss.core.models.model import (
    Attachment,
    Model,
    ThinkingCallback,
    TokenCallback,
    _build_text_based_tools_prompt,
    _parse_text_based_tool_calls,
    parse_binary_attachments,
    responses_items_to_chat_messages,
)

logger = logging.getLogger(__name__)

# Generated alias suffix produced by ``scripts/update_models.py`` to expose
# the uncapped ``reasoning_effort="xhigh"`` level as its own catalog entry.
# Both this module and ``OpenAICompatibleModel2`` strip it from the model
# name they send to the provider so the alias routes to the same upstream
# model id as the base entry while still carrying ``thinking="xhigh"`` in
# ``MODEL_INFO``.
_XHIGH_SUFFIX = "-xhigh"


def _provider_model_name(model_name: str) -> str:
    """Return the upstream provider id for a KISS catalog ``model_name``.

    Two transformations are applied in order:

    * An ``openrouter/`` routing prefix is removed (callers reach
      OpenRouter via the catalog key ``openrouter/<provider>/<id>`` but
      the OpenRouter API itself wants the bare ``<provider>/<id>``).
    * A trailing ``-xhigh`` is stripped so the synthetic xhigh alias
      maps back to its base model id.  ``MODEL_INFO`` carries the
      sibling entry purely so callers can select ``reasoning_effort=
      "xhigh"`` by model name; the provider's HTTP endpoint only knows
      the base name.

    Args:
        model_name: The catalog model name as passed in.

    Returns:
        The string to send as ``model=`` over the wire.
    """
    provider_name = (
        model_name[len("openrouter/") :]
        if model_name.startswith("openrouter/")
        else model_name
    )
    return provider_name.removesuffix(_XHIGH_SUFFIX)


def _model_thinking_level(model_name: str) -> str | None:
    """Return the default ``reasoning_effort`` level for *model_name*, if any.

    The level lives on ``MODEL_INFO[model_name].thinking`` and is set
    per-model in ``model_info.py`` via the ``thinking="<level>"`` argument of
    the ``_mi(...)`` helper (e.g. ``thinking="xhigh"`` for the gpt-5.5
    family).  Models not in ``MODEL_INFO`` (e.g. custom endpoints with
    arbitrary model names) return ``None`` so we never send an unsupported
    ``reasoning_effort`` to such providers.

    Args:
        model_name: The full model name as passed to
            :class:`OpenAICompatibleModel`, including any ``openrouter/``
            prefix.

    Returns:
        The thinking level string (e.g. ``"xhigh"``) if the matching
        ``MODEL_INFO`` entry sets one, otherwise ``None``.
    """
    # Lazy import: avoid a circular import between this module and
    # ``model_info`` (which imports ``OpenAICompatibleModel`` from this
    # package via ``_openai_compatible``).
    from kiss.core.models.model_info import MODEL_INFO
    info = MODEL_INFO.get(model_name)
    return info.thinking if info is not None else None


DEEPSEEK_REASONING_MODELS = {
    "deepseek/deepseek-r1",
    "deepseek/deepseek-r1-0528",
    "deepseek/deepseek-r1-turbo",
    "deepseek/deepseek-r1-distill-qwen-1.5b",
    "deepseek/deepseek-r1-distill-qwen-7b",
    "deepseek/deepseek-r1-distill-llama-8b",
    "deepseek/deepseek-r1-distill-qwen-14b",
    "deepseek/deepseek-r1-distill-qwen-32b",
    "deepseek/deepseek-r1-distill-llama-70b",
    "deepseek-ai/DeepSeek-R1",
    "deepseek-ai/DeepSeek-R1-0528-tput",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
}

# Per-process cache of live verdicts on whether an endpoint accepts
# ``tools`` + ``reasoning_effort`` on the same Chat Completions request,
# keyed by ``base_url``.  Only consulted for endpoints whose capability is
# not declared in ``model_info.OPENAI_COMPATIBLE_PROVIDERS`` (custom
# gateways, or registered vendors declared ``None`` = unverified): the
# first tool-bearing request keeps the effort optimistically and the
# vendor's actual response records the verdict here (adaptive probe), so
# future vendors work correctly without a manual host allowlist patch.
_ADAPTIVE_TOOL_EFFORT_VERDICTS: dict[str, bool] = {}


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


def _extract_deepseek_reasoning(content: str) -> tuple[str, str]:
    """Extract reasoning and final answer from DeepSeek R1 response.

    DeepSeek R1 models wrap their reasoning in <think>...</think> tags.

    Args:
        content: The raw response content from a DeepSeek R1 model.

    Returns:
        A tuple of (reasoning, final_answer) where reasoning is the content
        within <think> tags and final_answer is the remaining content.
    """
    think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    match = think_pattern.search(content)
    if match:
        reasoning = match.group(1).strip()
        final_answer = think_pattern.sub("", content).strip()
        return reasoning, final_answer
    return "", content


def _anthropic_media_block_to_openai_part(block: dict[str, Any]) -> dict[str, Any] | None:
    """Convert an Anthropic ``image``/``document`` block to an OpenAI content part.

    Anthropic media blocks carry a ``source`` dict (``{"type": "base64",
    "media_type": ..., "data": ...}`` or ``{"type": "url", "url": ...}``).
    OpenAI Chat Completions instead uses ``image_url`` / ``file`` parts.
    Such blocks enter the conversation when it is handed off from an
    :class:`AnthropicModel` (e.g. via the Sorcar ``set_model`` tool).

    Args:
        block: The Anthropic media block dict.

    Returns:
        The equivalent OpenAI content-part dict, or ``None`` when the block
        cannot be represented (in which case it is dropped with a warning).
    """
    source = block.get("source") or {}
    url = ""
    if source.get("type") == "base64":
        media_type = source.get("media_type", "application/octet-stream")
        url = f"data:{media_type};base64,{source.get('data', '')}"
    elif source.get("type") == "url":
        url = source.get("url", "")
    if not url:
        logger.warning("Dropping unconvertible Anthropic %s block.", block.get("type"))
        return None
    if block.get("type") == "image":
        return {"type": "image_url", "image_url": {"url": url}}
    return {"type": "file", "file": {"file_data": url}}


def _stringify_tool_call_arguments(tool_calls: list[Any]) -> list[Any]:
    """Ensure every tool call's ``function.arguments`` is a JSON string.

    GeminiModel stores tool-call arguments as dicts; the OpenAI Chat
    Completions API requires a JSON string.  Such entries enter the
    conversation when it is handed off from a :class:`GeminiModel`
    (e.g. via the Sorcar ``set_model`` tool).

    Args:
        tool_calls: The ``tool_calls`` list of an assistant message.

    Returns:
        The list with dict arguments replaced by their JSON encoding.
    """
    result: list[Any] = []
    for tc in tool_calls:
        if isinstance(tc, dict):
            fn = tc.get("function") or {}
            args = fn.get("arguments")
            if not isinstance(args, str):
                tc = {**tc, "function": {**fn, "arguments": json.dumps(args or {})}}
        result.append(tc)
    return result


def _tool_result_block_text(block: dict[str, Any]) -> str:
    """Extract the text payload of an Anthropic ``tool_result`` block.

    The block's ``content`` may be a plain string or a list of nested
    blocks; only the text of nested ``text`` blocks is kept because the
    OpenAI ``role="tool"`` message accepts string content only.

    Args:
        block: The Anthropic ``tool_result`` block dict.

    Returns:
        The concatenated text content of the block.
    """
    content = block.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return "" if content is None else str(content)


def _merge_tools_prompt_into_content(content: Any, tools_prompt: str) -> Any:
    """Append *tools_prompt* to a user message content of either shape.

    The first user message's ``content`` is a plain string in the common
    case, but a list of content parts when the prompt was initialized with
    attachments (images/PDFs/audio).

    Args:
        content: The existing message content (str or list of parts).
        tools_prompt: The text-based tools prompt to append.

    Returns:
        The merged content in the same shape as the input.
    """
    if isinstance(content, list):
        return [*content, {"type": "text", "text": tools_prompt}]
    return str(content) + "\n" + tools_prompt


class OpenAICompatibleModel(Model):
    """A model that uses an OpenAI-compatible API with a custom base URL.

    This model can be used with any API that implements the OpenAI chat completions
    format, such as local LLM servers (Ollama, vLLM, LM Studio), or third-party
    providers that offer OpenAI-compatible endpoints.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str,
        model_config: dict[str, Any] | None = None,
        token_callback: TokenCallback | None = None,
        thinking_callback: ThinkingCallback | None = None,
    ):
        """Initialize an OpenAI-compatible model.

        Args:
            model_name: The name/identifier of the model to use.
            base_url: The base URL for the API endpoint (e.g., "http://localhost:11434/v1").
            api_key: API key for authentication.
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
        self.base_url = base_url
        self.api_key = api_key
        self._api_model_name = _provider_model_name(model_name)
        # Lazily-created ``OpenAICompatibleModel2`` used to send tool-bearing
        # requests through ``/v1/responses`` so ``reasoning_effort`` (incl.
        # ``"xhigh"``) survives instead of being stripped (see
        # ``_generate_with_tools_via_responses``).
        self._responses_delegate: OpenAICompatibleModel2 | None = None
        # call_id -> raw Responses-API output items (reasoning / message /
        # function_call) captured for the assistant turn that produced the
        # call.  Replayed verbatim on later turns so OpenAI's "function_call
        # provided without its required reasoning item" constraint is
        # satisfied for reasoning models.
        self._delegate_raw_items: dict[str, list[dict[str, Any]]] = {}
        # Default ``reasoning_effort`` to the level declared on the model's
        # MODEL_INFO entry (e.g. ``"xhigh"`` for gpt-5.5).  Copy first so we
        # never mutate the caller's dict.  Caller-supplied values always win.
        thinking_level = _model_thinking_level(self.model_name)
        if thinking_level is not None and "reasoning_effort" not in self.model_config:
            self.model_config = dict(self.model_config)
            self.model_config["reasoning_effort"] = thinking_level
        # Base64-encoded audio payload of the most recent ``generate()``
        # response, populated when the request asked for audio output
        # (``model_config={"modalities": ["text", "audio"], "audio":
        # {...}}`` with a GPT audio-chat model).  ``None`` otherwise.
        # Lets callers (e.g. the ``talk`` tool's speech synthesis via
        # ``KISSAgent``) retrieve model-generated audio through the core
        # model layer instead of calling the OpenAI SDK directly.
        self.last_audio_data: str | None = None

    def __str__(self) -> str:
        """Return a string representation of the model.

        Returns:
            A string showing the class name, model name, and base URL.
        """
        return f"{self.__class__.__name__}(name={self.model_name}, base_url={self.base_url})"

    __repr__ = __str__

    def initialize(self, prompt: str, attachments: list[Attachment] | None = None) -> None:
        """Initialize the conversation with an initial user prompt.

        Args:
            prompt: The initial user prompt to start the conversation.
            attachments: Optional list of file attachments (images, PDFs) to include.
        """
        extra_headers = self.model_config.get("extra_headers") or {}
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=1800.0,
            default_headers=extra_headers,
        )
        self.conversation = []
        system_instruction = self.model_config.get("system_instruction")
        if system_instruction:
            self.conversation.append({"role": "system", "content": system_instruction})
        content: str | list[dict[str, Any]] = prompt
        if attachments:
            parts = self._attachments_to_content_parts(attachments)
            parts.append({"type": "text", "text": prompt})
            content = parts
        self.conversation.append({"role": "user", "content": content})

    @staticmethod
    def _attachment_to_content_part(att: Attachment) -> dict[str, Any] | None:
        """Convert a single :class:`Attachment` to an OpenAI content-part dict.

        Returns ``None`` for unsupported MIME types (e.g. video, which OpenAI
        Chat Completions does not accept), logging a warning so the caller
        knows the attachment was dropped.

        Args:
            att: The attachment to convert.

        Returns:
            A content-part dict suitable for the ``content`` array of a
            chat-completions message, or ``None`` if the MIME type is not
            supported by OpenAI Chat Completions.
        """
        if att.mime_type.startswith("image/"):
            return {"type": "image_url", "image_url": {"url": att.to_data_url()}}
        if att.mime_type == "application/pdf":
            return {"type": "file", "file": {"file_data": att.to_data_url()}}
        if att.mime_type.startswith("audio/"):
            fmt = _audio_mime_to_format(att.mime_type)
            return {
                "type": "input_audio",
                "input_audio": {"data": att.to_base64(), "format": fmt},
            }
        logger.warning(
            "OpenAI Chat Completions does not support %s attachments; skipping.",
            att.mime_type,
        )
        return None

    @classmethod
    def _attachments_to_content_parts(
        cls, attachments: list[Attachment]
    ) -> list[dict[str, Any]]:
        """Convert attachments to a list of OpenAI content-part dicts.

        Unsupported MIME types are silently dropped (with a warning).

        Args:
            attachments: The attachments to convert.

        Returns:
            A list of content-part dicts.  May be shorter than *attachments*
            if some MIME types were not supported.
        """
        parts: list[dict[str, Any]] = []
        for att in attachments:
            part = cls._attachment_to_content_part(att)
            if part is not None:
                parts.append(part)
        return parts

    def add_function_results_to_conversation_and_return(
        self, function_results: list[tuple[str, dict[str, Any]]]
    ) -> None:
        """Add tool results to the conversation, lifting binary attachments.

        The OpenAI Chat Completions ``tool`` role only accepts string
        content, so binary attachments produced by the ``Read`` tool (e.g. a
        PNG screenshot or an MP3 audio file) cannot live inside the tool
        message.  Each tool message is appended with the sentinel payload
        stripped to a short placeholder; if attachments were present, a
        follow-up ``user`` message carrying ``image_url`` / ``file`` /
        ``input_audio`` content parts is appended right after so the model
        can actually see the file.  Unsupported MIME types (e.g. video) are
        dropped with a warning.

        Args:
            function_results: List of ``(function_name, result_dict)`` tuples.
        """
        tool_calls = self._find_tool_call_ids_from_last_assistant()
        pending_attachments: list[Attachment] = []

        for i, (func_name, result_dict) in enumerate(function_results):
            result_content = result_dict.get("result", str(result_dict))
            result_content, attachments = parse_binary_attachments(result_content)
            if self.usage_info_for_messages:
                result_content = f"{result_content}\n\n{self.usage_info_for_messages}"

            tool_call_id = (
                tool_calls[i][1] if i < len(tool_calls) else f"call_{func_name}_{i}"
            )
            self.conversation.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": result_content,
                }
            )
            pending_attachments.extend(attachments)

        if pending_attachments:
            parts = self._attachments_to_content_parts(pending_attachments)
            if parts:
                parts.append(
                    {
                        "type": "text",
                        "text": "[attachments from previous tool result(s)]",
                    }
                )
                self.conversation.append({"role": "user", "content": parts})

    def _is_deepseek_reasoning_model(self) -> bool:
        """Check if this is a DeepSeek R1 reasoning model.

        Uses ``_api_model_name`` (which strips the ``openrouter/`` routing
        prefix) so that models accessed via OpenRouter are matched correctly.

        Returns:
            True if the API model name is in DEEPSEEK_REASONING_MODELS.
        """
        return self._api_model_name in DEEPSEEK_REASONING_MODELS

    def _is_openrouter_anthropic(self) -> bool:
        """Check if this is an OpenRouter Anthropic model (Claude via OpenRouter)."""
        return self.model_name.startswith("openrouter/anthropic/")

    @classmethod
    def _normalize_content_blocks(cls, content: Any) -> list[dict[str, Any]]:
        """Normalize content blocks to JSON-serializable dicts.

        Drops text blocks whose text is empty or whitespace-only, because
        many APIs reject them with invalid_request_error about non-whitespace text.

        Also drops Anthropic ``thinking`` / ``redacted_thinking`` blocks: the
        OpenAI Chat Completions API has no such content-part type and rejects
        them with ``invalid_value`` (thinking blocks are hidden provider state
        that must not be replayed to a different provider).  Such blocks enter
        the conversation when it is handed off from an :class:`AnthropicModel`
        (e.g. via the Sorcar ``set_model`` tool).

        Args:
            content: The content blocks from a response.

        Returns:
            list[dict[str, Any]]: Normalized content blocks as dictionaries.
        """
        blocks: list[dict[str, Any]] = []
        if content is None:
            return blocks
        for block in content:
            if isinstance(block, dict):
                block_type = block.get("type")
                if block_type in ("thinking", "redacted_thinking"):
                    continue
                # Drop pre-existing whitespace-only text dicts too.
                if block_type == "text" and not block.get("text", "").strip():
                    continue
                if block_type in ("image", "document"):
                    converted = _anthropic_media_block_to_openai_part(block)
                    if converted is not None:
                        blocks.append(converted)
                    continue
                blocks.append(block)
                continue
            block_type = getattr(block, "type", None)
            if block_type in ("thinking", "redacted_thinking"):
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

    def _normalize_conversation_for_api(
        self,
        conversation: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Normalize all messages in a conversation before sending to the API.

        Ensures that all text content blocks are non-whitespace and that no
        messages contain only whitespace-only text blocks.  Also converts
        Anthropic Messages-format entries (assistant ``tool_use`` blocks,
        user ``tool_result`` blocks, ``thinking`` blocks) — which enter the
        conversation when it is handed off from an :class:`AnthropicModel`,
        e.g. via the Sorcar ``set_model`` tool — into the OpenAI Chat
        Completions equivalents (``tool_calls`` arrays, ``role="tool"``
        messages) so the API does not reject them with ``invalid_value``.
        OpenAI Responses-API items (handed off from an
        :class:`OpenAICompatibleModel2`) are converted first via
        :func:`responses_items_to_chat_messages`.

        Args:
            conversation: The conversation to normalize.

        Returns:
            list[dict[str, Any]]: The normalized conversation.
        """
        normalized: list[dict[str, Any]] = []
        for msg in responses_items_to_chat_messages(conversation):
            normalized.extend(self._normalize_message_for_api(msg))
        return normalized

    @classmethod
    def _normalize_message_for_api(cls, msg: dict[str, Any]) -> list[dict[str, Any]]:
        """Normalize a single conversation message into OpenAI-format messages.

        A message may expand to zero messages (all content filtered out),
        one message (the common case), or several messages (an Anthropic
        user message carrying multiple ``tool_result`` blocks becomes one
        ``role="tool"`` message per block).

        Args:
            msg: The conversation message to normalize.

        Returns:
            list[dict[str, Any]]: OpenAI Chat Completions-format messages.
        """
        msg_copy = msg.copy()
        if msg_copy.get("tool_calls"):
            # Gemini hand-off: arguments may be dicts; OpenAI wants JSON strings.
            msg_copy["tool_calls"] = _stringify_tool_call_arguments(msg_copy["tool_calls"])
        attachments = msg_copy.pop("attachments", None)
        if attachments:
            # Gemini hand-off: lift the Attachment objects into OpenAI
            # content parts (the API rejects unknown message fields).
            parts = cls._attachments_to_content_parts(attachments)
            prior = msg_copy.get("content")
            if isinstance(prior, str) and prior.strip():
                parts.append({"type": "text", "text": prior})
            elif isinstance(prior, list):
                parts.extend(prior)
            msg_copy["content"] = parts
        content = msg_copy.get("content")
        has_tool_calls = bool(msg_copy.get("tool_calls"))

        if isinstance(content, str):
            # Keep whitespace-only content when tool_calls are attached
            # (OpenAI allows empty assistant content alongside tool_calls).
            if content.strip() or has_tool_calls:
                return [msg_copy]
            return []
        if not isinstance(content, list):
            return [msg_copy] if content is not None or has_tool_calls else []

        blocks = cls._normalize_content_blocks(content)
        tool_results = [b for b in blocks if b.get("type") == "tool_result"]
        tool_uses = [b for b in blocks if b.get("type") == "tool_use"]
        rest = [b for b in blocks if b.get("type") not in ("tool_result", "tool_use")]

        if tool_uses:
            # Anthropic assistant message: text blocks + tool_use blocks.
            text = "".join(b.get("text", "") for b in rest if b.get("type") == "text")
            tool_calls = list(msg_copy.get("tool_calls") or [])
            for b in tool_uses:
                tool_calls.append(
                    {
                        "id": b.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": b.get("name", ""),
                            "arguments": json.dumps(b.get("input") or {}),
                        },
                    }
                )
            return [
                {
                    "role": msg_copy.get("role", "assistant"),
                    "content": text,
                    "tool_calls": tool_calls,
                }
            ]
        if tool_results:
            # Anthropic user message carrying tool results: one OpenAI
            # ``role="tool"`` message per tool_result block.
            converted = [
                {
                    "role": "tool",
                    "tool_call_id": b.get("tool_use_id", ""),
                    "content": _tool_result_block_text(b),
                }
                for b in tool_results
            ]
            if rest:
                msg_copy["content"] = rest
                converted.append(msg_copy)
            return converted
        if not blocks:
            return [msg_copy] if has_tool_calls else []
        msg_copy["content"] = blocks
        return [msg_copy]

    def _apply_cache_control_for_openrouter_anthropic(self, kwargs: dict[str, Any]) -> None:
        """Add top-level cache_control for OpenRouter Anthropic prompt caching.

        Uses the same approach as AnthropicModel: a single top-level cache_control
        that lets OpenRouter automatically place the breakpoint at the last cacheable
        block and move it forward as the conversation grows.
        """
        if not self._is_openrouter_anthropic():
            return
        if not self.model_config.get("enable_cache", True):
            return
        kwargs.setdefault("extra_body", {})["cache_control"] = {"type": "ephemeral"}

    @staticmethod
    def _build_tool_call_lists(
        entries: list[tuple[str, str, str]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Build function_calls and raw_tool_calls from (id, name, arguments_json) triples.

        Args:
            entries: List of (call_id, function_name, arguments_json_string) tuples.

        Returns:
            A tuple of (function_calls, raw_tool_calls) for conversation storage.
        """
        function_calls: list[dict[str, Any]] = []
        raw_tool_calls: list[dict[str, Any]] = []
        for call_id, name, args_json in entries:
            try:
                arguments = json.loads(args_json)
            except json.JSONDecodeError:
                logger.debug("Exception caught", exc_info=True)
                arguments = {}
            function_calls.append({"id": call_id, "name": name, "arguments": arguments})
            raw_tool_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": args_json},
                }
            )
        return function_calls, raw_tool_calls

    @staticmethod
    def _parse_tool_call_accum(
        accum: dict[int, dict[str, str]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Parse accumulated streaming tool-call deltas into structured lists.

        Args:
            accum: Mapping of tool-call index to accumulated id/name/arguments strings.

        Returns:
            A tuple of (function_calls, raw_tool_calls) for conversation storage.
        """
        entries = [
            (accum[idx]["id"], accum[idx]["name"], accum[idx]["arguments"])
            for idx in sorted(accum)
        ]
        return OpenAICompatibleModel._build_tool_call_lists(entries)

    @staticmethod
    def _parse_tool_calls_from_message(
        message: Any,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Extract tool calls from a non-streamed OpenAI message.

        Args:
            message: The message object from a chat completion response.

        Returns:
            A tuple of (function_calls, raw_tool_calls) for conversation storage.
        """
        if not message.tool_calls:
            return [], []
        entries = [
            (tc.id, tc.function.name, tc.function.arguments)
            for tc in message.tool_calls
        ]
        return OpenAICompatibleModel._build_tool_call_lists(entries)

    @staticmethod
    def _finalize_stream_response(response: Any | None, last_chunk: Any | None) -> Any:
        """Pick the best response object from a stream.

        Args:
            response: The chunk containing usage info, if seen.
            last_chunk: The last chunk seen in the stream.

        Returns:
            A response-like object with usage info when available.
        """
        if response is not None:
            return response
        if last_chunk is not None:
            return last_chunk
        raise KISSError("Streaming response was empty.")

    @staticmethod
    def _extract_output_audio(response: Any) -> Any | None:
        """Return the output-audio object of a chat completion, if any.

        Args:
            response: A chat-completions response (or final stream chunk).

        Returns:
            The ``message.audio`` object of the first choice — carrying
            ``data`` (base64 audio) and ``transcript`` — or ``None`` when
            the response has no audio output (the common text-only case,
            and always for streamed chunks, which carry deltas instead of
            a full message).
        """
        choices = getattr(response, "choices", None)
        if not choices:
            return None
        message = getattr(choices[0], "message", None)
        return getattr(message, "audio", None)

    def _stream_text(  # pragma: no cover – API streaming
        self, kwargs: dict[str, Any],
    ) -> tuple[str, Any]:
        """Stream a chat completion, invoking the token callback for each text delta.

        When no callback is set, falls back to a normal (non-streaming) call.

        Args:
            kwargs: Keyword arguments for the OpenAI chat completions API.

        Returns:
            A tuple of (content, response).
        """
        if self.token_callback is None:
            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content or "", response

        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}
        content = ""
        response = None
        last_chunk = None
        in_reasoning = False
        for chunk in self.client.chat.completions.create(**kwargs):
            last_chunk = chunk
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta:
                    reasoning = getattr(delta, "reasoning_content", None)
                    if reasoning:
                        if not in_reasoning:
                            in_reasoning = True
                            self._invoke_thinking_callback(True)
                        self._invoke_token_callback(reasoning)
                    if delta.content:
                        if in_reasoning:
                            in_reasoning = False
                            self._invoke_thinking_callback(False)
                        content += delta.content
                        self._invoke_token_callback(delta.content)
            if chunk.usage is not None:
                response = chunk
        if in_reasoning:
            self._invoke_thinking_callback(False)
        response = self._finalize_stream_response(response, last_chunk)
        return content, response

    def _build_chat_kwargs(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Build the base kwargs dict for a Chat Completions request.

        Copies ``model_config`` (dropping keys that are not Chat Completions
        parameters), attaches the provider model name and ``messages``, and
        applies OpenRouter Anthropic cache control.

        Args:
            messages: The normalized messages to send.

        Returns:
            The kwargs dict for ``client.chat.completions.create(...)``.
        """
        kwargs = self.model_config.copy()
        kwargs.pop("system_instruction", None)
        kwargs.pop("use_responses_api", None)
        kwargs.update({"model": self._api_model_name, "messages": messages})
        self._apply_cache_control_for_openrouter_anthropic(kwargs)
        return kwargs

    def _normalized_conversation_checked(self) -> list[dict[str, Any]]:
        """Normalize the conversation for the API, raising when nothing remains.

        Returns:
            The normalized conversation messages.

        Raises:
            KISSError: When every message was filtered out as whitespace-only.
        """
        normalized = self._normalize_conversation_for_api(self.conversation)
        if not normalized:
            raise KISSError(
                "Cannot generate response: all messages have whitespace-only "
                "content that was filtered out. At least one message with "
                "non-whitespace content is required."
            )
        return normalized

    def generate(self) -> tuple[str, Any]:
        """Generate content from prompt without tools.

        Returns:
            A tuple of (content, response) where content is the generated text
            and response is the raw API response object.
        """
        kwargs = self._build_chat_kwargs(self._normalized_conversation_checked())

        self.last_audio_data = None
        content, response = self._stream_text(kwargs)
        audio = self._extract_output_audio(response)
        if audio is not None:
            self.last_audio_data = getattr(audio, "data", None)
            if not content:
                # Audio-output models leave ``message.content`` empty and
                # put the spoken text in ``message.audio.transcript``.
                content = getattr(audio, "transcript", None) or ""

        if self._is_deepseek_reasoning_model():
            _, content = _extract_deepseek_reasoning(content)

        self.conversation.append({"role": "assistant", "content": content})
        return content, response

    def generate_and_process_with_tools(
        self,
        function_map: dict[str, Callable[..., Any]],
        tools_schema: list[dict[str, Any]] | None = None,
    ) -> tuple[list[dict[str, Any]], str, Any]:
        """Generate content with tools, process the response, and add it to conversation.

        Args:
            function_map: Dictionary mapping function names to callable functions.
            tools_schema: Optional pre-built tool schema list.

        Returns:
            A tuple of (function_calls, content, response) where function_calls is a list
            of dictionaries containing tool call information, content is the text response,
            and response is the raw API response object.
        """
        if self._is_deepseek_reasoning_model():
            return self._generate_with_text_based_tools(function_map)

        tools = self._resolve_openai_tools_schema(function_map, tools_schema)

        # OpenAI's /v1/chat/completions endpoint rejects the combination of
        # ``tools`` + ``reasoning_effort`` for GPT-5.x / o-series reasoning
        # models with: "Function tools with reasoning_effort are not supported
        # ... in /v1/chat/completions. Please use /v1/responses instead."
        # Tool-bearing requests to endpoints that implement ``/v1/responses``
        # are therefore delegated to the Responses-API transport
        # (:class:`OpenAICompatibleModel2`), which supports the combination
        # natively — including ``"xhigh"``.
        if (
            tools
            and "reasoning_effort" in self.model_config
            and self._should_delegate_to_responses()
        ):
            return self._generate_with_tools_via_responses(function_map, tools)

        kwargs = self._build_chat_kwargs(
            self._normalize_conversation_for_api(self.conversation)
        )
        kwargs["tools"] = tools or None

        # Whether ``tools`` + ``reasoning_effort`` survive on the same Chat
        # Completions request is a per-vendor capability declared in
        # ``model_info.OPENAI_COMPATIBLE_PROVIDERS`` (e.g. OpenRouter and
        # Together accept the combination — verified live; api.openai.com
        # rejects it, hence the Responses delegation above).  Endpoints
        # with an unknown capability (custom gateways, or vendors declared
        # ``None``) keep the effort optimistically —
        # ``_create_chat_completion_adaptive`` learns the verdict from the
        # vendor's actual response and caches it per endpoint, so no
        # manual allowlist patch is needed for future vendors.  The
        # no-tools ``generate()`` path always keeps the effort.
        if (
            tools
            and "reasoning_effort" in kwargs
            and self._tools_reasoning_effort_capability() is False
        ):
            dropped = kwargs.pop("reasoning_effort")
            logger.debug(
                "Dropping reasoning_effort=%r because tools are attached and "
                "endpoint %s is known to reject the combination.",
                dropped,
                self.base_url,
            )

        if self.token_callback is not None:  # pragma: no cover – API streaming
            kwargs["stream"] = True
            kwargs["stream_options"] = {"include_usage": True}
            content = ""
            tool_calls_accum: dict[int, dict[str, str]] = {}
            response = None
            last_chunk = None
            in_reasoning = False
            for chunk in self._create_chat_completion_adaptive(kwargs):
                last_chunk = chunk
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta:
                        reasoning = getattr(delta, "reasoning_content", None)
                        if reasoning:
                            if not in_reasoning:
                                in_reasoning = True
                                self._invoke_thinking_callback(True)
                            self._invoke_token_callback(reasoning)
                        if delta.content:
                            if in_reasoning:
                                in_reasoning = False
                                self._invoke_thinking_callback(False)
                            content += delta.content
                            self._invoke_token_callback(delta.content)
                        if delta.tool_calls:
                            if in_reasoning:
                                in_reasoning = False
                                self._invoke_thinking_callback(False)
                            for tc_delta in delta.tool_calls:
                                idx = tc_delta.index
                                if idx not in tool_calls_accum:
                                    tool_calls_accum[idx] = {
                                        "id": "",
                                        "name": "",
                                        "arguments": "",
                                    }
                                if tc_delta.id:
                                    tool_calls_accum[idx]["id"] = tc_delta.id
                                if tc_delta.function:
                                    if tc_delta.function.name:
                                        tool_calls_accum[idx]["name"] = tc_delta.function.name
                                    if tc_delta.function.arguments:
                                        tool_calls_accum[idx]["arguments"] += (
                                            tc_delta.function.arguments
                                        )
                if chunk.usage is not None:
                    response = chunk
            if in_reasoning:
                self._invoke_thinking_callback(False)
            response = self._finalize_stream_response(response, last_chunk)
            function_calls, raw_tool_calls = self._parse_tool_call_accum(tool_calls_accum)
        else:
            response = self._create_chat_completion_adaptive(kwargs)
            message = response.choices[0].message
            content = message.content or ""
            function_calls, raw_tool_calls = self._parse_tool_calls_from_message(message)

        self._append_assistant_turn(content, function_calls, raw_tool_calls)
        return function_calls, content, response

    def _append_assistant_turn(
        self,
        content: str,
        function_calls: list[dict[str, Any]],
        raw_tool_calls: list[dict[str, Any]],
    ) -> None:
        """Append the assistant turn, attaching ``tool_calls`` when calls were made.

        Args:
            content: The assistant text content.
            function_calls: Parsed function calls (may be empty).
            raw_tool_calls: Raw ``tool_calls`` entries to store when
                ``function_calls`` is non-empty.
        """
        msg: dict[str, Any] = {"role": "assistant", "content": content}
        if function_calls:
            msg["tool_calls"] = raw_tool_calls
        self.conversation.append(msg)

    def _should_delegate_to_responses(self) -> bool:
        """Decide whether tool-bearing requests should use ``/v1/responses``.

        The ``use_responses_api`` model-config flag forces the decision in
        either direction (``True`` = always delegate, ``False`` = never).
        When the flag is absent, the decision is the vendor's declared
        ``delegate_tools_to_responses`` capability from
        ``model_info.OPENAI_COMPATIBLE_PROVIDERS`` (currently only
        ``api.openai.com``, the provider that both implements
        ``/v1/responses`` and rejects ``tools`` + ``reasoning_effort`` on
        Chat Completions); unknown endpoints never delegate.

        Returns:
            True when the tool-calling transport should be the Responses API.
        """
        flag = self.model_config.get("use_responses_api")
        if flag is not None:
            return bool(flag)
        # Imported lazily: ``model_info`` imports this module at load time.
        from kiss.core.models.model_info import openai_compatible_provider_for_base_url

        provider = openai_compatible_provider_for_base_url(self.base_url)
        return provider is not None and provider.delegate_tools_to_responses

    def _tools_reasoning_effort_capability(self) -> bool | None:
        """Return whether this endpoint accepts ``tools`` + ``reasoning_effort``.

        The verdict is the vendor's declared
        ``tools_accept_reasoning_effort`` capability from
        ``model_info.OPENAI_COMPATIBLE_PROVIDERS`` when the endpoint is
        registered and verified; otherwise the cached adaptive verdict
        learned from the endpoint's actual responses, if any.

        Returns:
            True when the combination is known to be accepted, False when
            known to be rejected, None when unknown (send optimistically
            and let ``_create_chat_completion_adaptive`` learn the verdict).
        """
        # Imported lazily: ``model_info`` imports this module at load time.
        from kiss.core.models.model_info import openai_compatible_provider_for_base_url

        provider = openai_compatible_provider_for_base_url(self.base_url)
        declared = provider.tools_accept_reasoning_effort if provider else None
        if declared is not None:
            return declared
        return _ADAPTIVE_TOOL_EFFORT_VERDICTS.get(self.base_url)

    def _create_chat_completion_adaptive(self, kwargs: dict[str, Any]) -> Any:
        """Create a chat completion, learning the effort capability if unknown.

        For endpoints whose ``tools`` + ``reasoning_effort`` capability is
        unknown (not declared in the vendor registry and not yet probed),
        the request is sent optimistically with the effort attached. If the
        endpoint rejects it with a 400 that mentions ``reasoning_effort``,
        the verdict is cached as False and the request is retried once
        without the effort; on success the verdict is cached as True. Known
        endpoints and requests without tools or effort pass straight
        through.

        Args:
            kwargs: Fully-built Chat Completions request arguments; mutated
                (``reasoning_effort`` removed) when the endpoint rejects it.

        Returns:
            The API response (or stream iterator when ``stream=True``).
        """
        unverified = (
            bool(kwargs.get("tools"))
            and "reasoning_effort" in kwargs
            and self._tools_reasoning_effort_capability() is None
        )
        try:
            response = self.client.chat.completions.create(**kwargs)
        except BadRequestError as e:
            if unverified and "reasoning_effort" in str(e):
                _ADAPTIVE_TOOL_EFFORT_VERDICTS[self.base_url] = False
                dropped = kwargs.pop("reasoning_effort")
                logger.debug(
                    "Endpoint %s rejected tools + reasoning_effort=%r; "
                    "retrying without it and caching the verdict.",
                    self.base_url,
                    dropped,
                )
                return self.client.chat.completions.create(**kwargs)
            raise
        if unverified:
            _ADAPTIVE_TOOL_EFFORT_VERDICTS[self.base_url] = True
        return response

    @staticmethod
    def _chat_parts_to_responses_parts(
        parts: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Convert Chat-Completions user content parts to Responses parts.

        ``text`` becomes ``input_text``, ``image_url`` becomes
        ``input_image``, ``file`` becomes ``input_file`` and ``input_audio``
        passes through.  Unknown part types are dropped with a warning.

        Args:
            parts: Chat-Completions content parts.

        Returns:
            Responses-API ``input_*`` content parts.
        """
        converted: list[dict[str, Any]] = []
        for part in parts:
            ptype = part.get("type")
            if ptype == "text":
                if str(part.get("text", "")).strip():
                    converted.append({"type": "input_text", "text": part["text"]})
            elif ptype == "image_url":
                url = (part.get("image_url") or {}).get("url", "")
                if url:
                    converted.append(
                        {"type": "input_image", "image_url": url, "detail": "auto"}
                    )
            elif ptype == "file":
                file_info = part.get("file") or {}
                file_data = file_info.get("file_data", "")
                if file_data:
                    converted.append(
                        {
                            "type": "input_file",
                            "filename": file_info.get("filename", "attachment.pdf"),
                            "file_data": file_data,
                        }
                    )
            elif ptype == "input_audio":
                converted.append(part)
            else:
                logger.warning(
                    "Dropping unconvertible content part %r for the "
                    "Responses API.",
                    ptype,
                )
        return converted

    def _chat_conversation_to_responses_input(self) -> list[dict[str, Any]]:
        """Convert the chat-format conversation to Responses ``input`` items.

        The conversation stays in OpenAI Chat Completions format as the
        single source of truth (so hand-off between models keeps working);
        this method derives the equivalent Responses-API ``input`` array
        on demand for the delegated tool-calling transport:

        * ``role="tool"`` messages become ``function_call_output`` items.
        * Assistant messages with ``tool_calls`` are replaced by the raw
          Responses output items cached for that turn (reasoning items
          included) when available, and otherwise reconstructed as
          ``function_call`` items.
        * Content-part lists are mapped to ``input_*`` parts.

        Returns:
            The Responses-API ``input`` array.
        """
        items: list[dict[str, Any]] = []
        for msg in self._normalize_conversation_for_api(self.conversation):
            tool_calls = msg.get("tool_calls") or []
            if msg.get("role") == "assistant" and tool_calls:
                cached = self._delegate_raw_items.get(tool_calls[0].get("id", ""))
                if cached is not None:
                    items.extend(cached)
                    continue
            items.extend(self._chat_message_to_responses_items(msg))
        return items

    @classmethod
    def _chat_message_to_responses_items(
        cls, msg: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Convert one Chat-Completions message to Responses ``input`` items.

        * ``role="tool"`` messages become ``function_call_output`` items.
        * Assistant messages with ``tool_calls`` become an optional
          assistant text message plus one ``function_call`` item per call
          (dict arguments are JSON-encoded).
        * Content-part lists are mapped to ``input_*`` parts (assistant
          part lists collapse to their concatenated text).

        Args:
            msg: A Chat-Completions-format message.

        Returns:
            Responses-API ``input`` items (possibly empty).
        """
        role = msg.get("role")
        content = msg.get("content")
        items: list[dict[str, Any]] = []
        if role == "tool":
            output = content if isinstance(content, str) else json.dumps(content)
            return [
                {
                    "type": "function_call_output",
                    "call_id": msg.get("tool_call_id", ""),
                    "output": output,
                }
            ]
        tool_calls = msg.get("tool_calls") or []
        if role == "assistant" and tool_calls:
            if isinstance(content, str) and content.strip():
                items.append({"role": "assistant", "content": content})
            elif isinstance(content, list):
                text = "".join(
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                )
                if text.strip():
                    items.append({"role": "assistant", "content": text})
            for tc in tool_calls:
                fn = tc.get("function") or {}
                args = fn.get("arguments")
                items.append(
                    {
                        "type": "function_call",
                        "call_id": tc.get("id", ""),
                        "name": fn.get("name", ""),
                        "arguments": args
                        if isinstance(args, str)
                        else json.dumps(args or {}),
                    }
                )
            return items
        if isinstance(content, list):
            if role == "assistant":
                text = "".join(
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                )
                if text.strip():
                    items.append({"role": "assistant", "content": text})
                return items
            parts = cls._chat_parts_to_responses_parts(content)
            if parts:
                items.append({"role": role, "content": parts})
            return items
        if isinstance(content, str) and content.strip():
            items.append({"role": role, "content": content})
        return items

    def _generate_with_tools_via_responses(
        self,
        function_map: dict[str, Callable[..., Any]],
        tools: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], str, Any]:
        """Run one tool-bearing turn through the ``/v1/responses`` endpoint.

        Delegates the request to a lazily-created
        :class:`~kiss.core.models.openai_compatible_model2.OpenAICompatibleModel2`
        so ``reasoning_effort`` (including ``"xhigh"``) is preserved instead
        of being stripped.  The delegate is stateless across turns: its
        conversation is rebuilt from this model's chat-format conversation
        before every request, keeping the chat conversation the single
        source of truth (and hand-off compatible).  The raw Responses output
        items of the turn (reasoning items included) are cached per
        ``call_id`` so follow-up turns replay them verbatim.

        Args:
            function_map: Mapping of tool name to callable.
            tools: Chat-Completions-style tools schema (the delegate
                flattens it to the Responses shape).

        Returns:
            ``(function_calls, content, response)`` matching
            :meth:`generate_and_process_with_tools`.
        """
        from kiss.core.models.openai_compatible_model2 import OpenAICompatibleModel2

        delegate = self._responses_delegate
        if delegate is None:
            delegate_config = {
                k: v
                for k, v in self.model_config.items()
                if k not in ("system_instruction", "use_responses_api")
            }
            delegate = OpenAICompatibleModel2(
                model_name=self.model_name,
                base_url=self.base_url,
                api_key=self.api_key,
                model_config=delegate_config,
                token_callback=self.token_callback,
                thinking_callback=self.thinking_callback,
            )
            self._responses_delegate = delegate
        delegate.initialize("_")
        input_items = self._chat_conversation_to_responses_input()
        delegate.conversation = list(input_items)

        function_calls, content, response = delegate.generate_and_process_with_tools(
            function_map, tools
        )

        new_items = [
            item
            for item in delegate.conversation[len(input_items) :]
            if isinstance(item, dict)
            and item.get("type") != "_kiss_pending_tool_result_attachment"
        ]
        raw_tool_calls = [
            {
                "id": fc["id"],
                "type": "function",
                "function": {
                    "name": fc["name"],
                    "arguments": json.dumps(fc["arguments"]),
                },
            }
            for fc in function_calls
        ]
        for fc in function_calls:
            self._delegate_raw_items[fc["id"]] = new_items
        self._append_assistant_turn(content, function_calls, raw_tool_calls)
        return function_calls, content, response

    def _generate_with_text_based_tools(
        self, function_map: dict[str, Callable[..., Any]]
    ) -> tuple[list[dict[str, Any]], str, Any]:
        """Generate with text-based tool calling for models without native function calling.

        This method injects tool descriptions into the conversation and parses
        tool calls from the model's text output.

        Args:
            function_map: Dictionary mapping function names to callable functions.

        Returns:
            A tuple of (function_calls, content, response) where function_calls is a list
            of dictionaries containing parsed tool call information, content is the raw
            text response, and response is the raw API response object.
        """
        tools_prompt = _build_text_based_tools_prompt(function_map)

        modified_conversation = self._normalized_conversation_checked()
        if modified_conversation[0]["role"] == "user":
            modified_conversation[0] = {
                "role": "user",
                "content": _merge_tools_prompt_into_content(
                    modified_conversation[0]["content"], tools_prompt
                ),
            }
        else:
            modified_conversation.insert(0, {"role": "system", "content": tools_prompt})

        kwargs = self._build_chat_kwargs(modified_conversation)

        content, response = self._stream_text(kwargs)

        _, content_clean = _extract_deepseek_reasoning(content)

        function_calls = _parse_text_based_tool_calls(content_clean)

        self.conversation.append({"role": "assistant", "content": content})
        if function_calls:
            self._replace_last_assistant_with_tool_calls(content, function_calls)

        return function_calls, content, response

    def extract_input_output_token_counts_from_response(
        self, response: Any
    ) -> tuple[int, int, int, int]:
        """Extract token counts from an API response.

        Returns:
            (input_tokens, output_tokens, cache_read_tokens, cache_write_tokens).
            For OpenAI, cached_tokens is a subset of prompt_tokens; input_tokens
            is reported as (prompt_tokens - cached_tokens) so costs apply correctly.
            OpenRouter returns cache_write_tokens in prompt_tokens_details.
            OpenAI reasoning models may report reasoning tokens in
            completion_tokens_details.reasoning_tokens; those are counted as output
            tokens so Sorcar shows thinking-token usage.
            Responses-API responses (produced by the delegated tool-calling
            transport) report ``usage.input_tokens`` / ``usage.output_tokens``
            instead; those are routed to the delegate's extractor.
        """
        if self._responses_delegate is not None:
            # Alias so isinstance narrowing does not affect ``response``,
            # which the Chat-Completions path below still treats as ``Any``.
            response_alias: Any = response
            usage_obj: Any = (
                response_alias.get("usage")
                if isinstance(response_alias, dict)
                else getattr(response_alias, "usage", None)
            )
            has_responses_usage = (
                usage_obj.get("input_tokens") if isinstance(usage_obj, dict)
                else getattr(usage_obj, "input_tokens", None)
            ) is not None
            if has_responses_usage:
                return (
                    self._responses_delegate
                    .extract_input_output_token_counts_from_response(response)
                )
        if hasattr(response, "usage") and response.usage:
            usage = response.usage
            prompt_tokens = getattr(usage, "prompt_tokens", None) or 0
            completion_tokens = getattr(usage, "completion_tokens", None) or 0
            cached_tokens = 0
            cache_write_tokens = 0
            details = getattr(usage, "prompt_tokens_details", None)
            if details is not None:
                cached_tokens = getattr(details, "cached_tokens", 0) or 0
                cache_write_tokens = getattr(details, "cache_write_tokens", 0) or 0
            return (
                max(0, prompt_tokens - cached_tokens - cache_write_tokens),
                completion_tokens,
                cached_tokens,
                cache_write_tokens,
            )
        return 0, 0, 0, 0

    def get_embedding(self, text: str, embedding_model: str | None = None) -> list[float]:
        """Generate an embedding vector for the given text.

        Args:
            text: The text to generate an embedding for.
            embedding_model: Optional model name for embedding generation. Uses the
                model's name if not specified.

        Returns:
            A list of floating point numbers representing the embedding vector.

        Raises:
            KISSError: If the embedding generation fails.
        """
        model_to_use = embedding_model or self.model_name
        try:
            response = self.client.embeddings.create(model=model_to_use, input=text)
            return list(response.data[0].embedding)
        except Exception as e:
            logger.debug("Exception caught", exc_info=True)
            raise KISSError(f"Embedding generation failed for model {model_to_use}: {e}") from e
