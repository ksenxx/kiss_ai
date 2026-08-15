# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""OpenAI-compatible model implementation for custom endpoints."""

import json
import logging
import re
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from openai import BadRequestError, OpenAI
from openai.resources.chat.completions import Completions

if TYPE_CHECKING:  # pragma: no cover – import cycle avoided at runtime
    from kiss.core.models.openai_compatible_model2 import OpenAICompatibleModel2

from kiss.core.kiss_error import KISSError
from kiss.core.models.model import (
    FRAMEWORK_ONLY_CONFIG_KEYS,
    TOOL_RESULT_ATTACHMENT_NOTE,
    Attachment,
    Model,
    ThinkingCallback,
    TokenCallback,
    _build_text_based_tools_prompt,
    _parse_text_based_tool_calls,
    accepted_request_params,
    responses_items_to_chat_messages,
)
from kiss.core.models.stream_abort import (
    DEFAULT_STREAM_STALL_TIMEOUT,
    stop_aware_events,
)

logger = logging.getLogger(__name__)

# The request parameters a Chat Completions call accepts, taken from the
# SDK's own signature (keyword-only, no **kwargs).
_CHAT_REQUEST_PARAMS = accepted_request_params(Completions.create)

def _provider_model_name(model_name: str) -> str:
    """Return the upstream provider id for a KISS catalog ``model_name``.

    Two transformations are applied in order:

    * A trailing thinking-level alias suffix (``-xhigh``, or a
      marker-verified ``-high`` / ``-medium`` / ``-low``) is stripped so
      the synthetic alias maps back to its base model id.  The alias is
      resolved against the full catalog key BEFORE prefix removal so only
      an exact ``alias_of``-marked entry can rewrite the name (an
      unrelated catalog alias can never rewrite a similarly-named custom
      model).  ``MODEL_INFO`` carries the sibling entries purely so
      callers can select a ``reasoning_effort`` level by model name; the
      provider's HTTP endpoint only knows the base name.
    * An ``openrouter/`` routing prefix is removed (callers reach
      OpenRouter via the catalog key ``openrouter/<provider>/<id>`` but
      the OpenRouter API itself wants the bare ``<provider>/<id>``).

    Args:
        model_name: The catalog model name as passed in.

    Returns:
        The string to send as ``model=`` over the wire.
    """
    from kiss.core.models.model_info import _strip_thinking_alias

    base_name = _strip_thinking_alias(model_name)
    if base_name.startswith("openrouter/"):
        return base_name[len("openrouter/") :]
    return base_name


def _model_thinking_level(model_name: str) -> str | None:
    """Return the default ``reasoning_effort`` level for *model_name*, if any.

    The level lives on ``MODEL_INFO[model_name].thinking`` and is set
    per-model via the ``thinking`` key in ``MODEL_INFO.json`` /
    ``~/.kiss/MY_MODELS.json`` (e.g. ``thinking="xhigh"`` for the gpt-5.5
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

_ADAPTIVE_TOOL_EFFORT_VERDICTS: dict[tuple[str, str], bool] = {}
_ADAPTIVE_TOOL_EFFORT_LOCK = threading.Lock()


def _tool_effort_verdict(key: tuple[str, str]) -> bool | None:
    """Return the learned ``tools`` + ``reasoning_effort`` verdict, if any.

    Args:
        key: The ``(base_url, api_model_name)`` the verdict was learned
            for.  The capability is a property of the *pair*: a
            reasoning model can reject the combination while another
            model on the same host accepts it.

    Returns:
        ``True``/``False`` when the pair has been probed, ``None``
        otherwise.
    """
    with _ADAPTIVE_TOOL_EFFORT_LOCK:
        return _ADAPTIVE_TOOL_EFFORT_VERDICTS.get(key)


def _record_tool_effort_verdict(key: tuple[str, str], accepted: bool) -> None:
    """Record what one ``(endpoint, model)`` did with the combination.

    Parallel subagents probe concurrently, so the check and the write
    straddle an HTTP round trip.  A rejection is definitive and always
    wins; a success is only recorded when nothing is known yet, so a
    probe that raced with a rejection can never erase it and the outcome
    no longer depends on which response happened to arrive last.

    Args:
        key: The ``(base_url, api_model_name)`` that was probed.
        accepted: Whether the endpoint accepted the combination.
    """
    with _ADAPTIVE_TOOL_EFFORT_LOCK:
        if accepted:
            _ADAPTIVE_TOOL_EFFORT_VERDICTS.setdefault(key, True)
        else:
            _ADAPTIVE_TOOL_EFFORT_VERDICTS[key] = False


# The attachment formats OpenAI accepts as input.  Images: PNG, JPEG,
# WEBP and non-animated GIF ("Image input requirements",
# https://developers.openai.com/api/docs/guides/images-vision, which states
# the behaviour "is the same in both the Responses API and the Chat
# Completions API").  Audio: mp3 and wav, the only values the SDK's own
# ``input_audio.format`` literal allows — in the Chat Completions param
# (``chat_completion_content_part_input_audio_param.py``) as well as the
# Responses one (``response_input_audio_param.py``).
#
# Both transports apply this one set: whether a turn goes to Chat
# Completions or the Responses API is a routing decision the caller never
# made, so it must not change which attachments the model gets to see.
OPENAI_INPUT_IMAGE_MIME_TYPES = frozenset(
    {"image/png", "image/jpeg", "image/jpg", "image/webp", "image/gif"}
)
OPENAI_INPUT_AUDIO_FORMATS = frozenset({"mp3", "wav"})

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


def _delta_reasoning_text(delta: Any) -> str | None:
    """Extract reasoning text from a Chat Completions streaming delta.

    Providers expose reasoning on different delta fields:

    * ``reasoning_content`` — DeepSeek-native (and vLLM-style) servers.
    * ``reasoning`` — OpenRouter's normalized plain-string field (also
      newer vLLM versions).
    * ``reasoning_details`` — OpenRouter's structured list of
      ``reasoning.text`` / ``reasoning.summary`` entries.

    The plain string fields are preferred; ``reasoning_details`` is
    consulted ONLY when both string fields are empty because OpenRouter
    sends the same text in ``reasoning`` AND ``reasoning_details``
    simultaneously — reading both would double-emit every thinking token.

    Args:
        delta: A streaming chunk's ``choices[0].delta`` object.

    Returns:
        The reasoning text for this delta, or None when the delta carries
        no reasoning.
    """
    reasoning_content = getattr(delta, "reasoning_content", None)
    if isinstance(reasoning_content, str) and reasoning_content:
        return reasoning_content
    reasoning = getattr(delta, "reasoning", None)
    if isinstance(reasoning, str) and reasoning:
        return reasoning
    details = getattr(delta, "reasoning_details", None)
    if not isinstance(details, list):
        return None
    texts: list[str] = []
    for entry in details:
        if isinstance(entry, dict):
            entry_type = entry.get("type")
            text = entry.get("text")
            summary = entry.get("summary")
        else:
            entry_type = getattr(entry, "type", None)
            text = getattr(entry, "text", None)
            summary = getattr(entry, "summary", None)
        value = text if entry_type == "reasoning.text" else summary
        if isinstance(value, str) and value:
            texts.append(value)
    return "".join(texts) or None


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


def _accumulate_tool_call_deltas(
    accum: dict[int, dict[str, str]], tool_call_deltas: list[Any]
) -> None:
    """Merge streamed ``tool_calls`` deltas into the per-index accumulator.

    Args:
        accum: Mapping of tool-call index to its accumulated
            id / name / arguments strings.  Mutated in place.
        tool_call_deltas: The ``delta.tool_calls`` entries of one chunk.
    """
    for tc_delta in tool_call_deltas:
        slot = accum.setdefault(
            tc_delta.index, {"id": "", "name": "", "arguments": ""}
        )
        if tc_delta.id:
            slot["id"] = tc_delta.id
        if tc_delta.function:
            if tc_delta.function.name:
                slot["name"] = tc_delta.function.name
            if tc_delta.function.arguments:
                slot["arguments"] += tc_delta.function.arguments


class OpenAICompatibleBase(Model):
    """Shared vendor-detection helpers for the OpenAI-compatible transports.

    Both :class:`OpenAICompatibleModel` (Chat Completions) and
    :class:`~kiss.core.models.openai_compatible_model2.OpenAICompatibleModel2`
    (Responses) route the same model names to the same vendors, so the
    DeepSeek-R1 / OpenRouter-Anthropic detection and the OpenRouter
    prompt-cache request marker live here once.
    """

    _api_model_name: str

    def __init__(
        self,
        model_name: str,
        model_config: dict[str, Any] | None = None,
        token_callback: TokenCallback | None = None,
        thinking_callback: ThinkingCallback | None = None,
    ):
        """Initialize the shared OpenAI-compatible state.

        Args:
            model_name: The KISS catalog model name.
            model_config: Optional model parameters.  ``stream_stall_timeout``
                (seconds of event-level silence tolerated before a
                streamed request is aborted with a retryable
                ``TimeoutError``) is read here for both transports.
            token_callback: Optional callback for each streamed text token.
            thinking_callback: Optional callback bracketing thinking blocks.
        """
        super().__init__(
            model_name,
            model_config=model_config,
            token_callback=token_callback,
            thinking_callback=thinking_callback,
        )
        self._stream_stall_timeout = float(
            self.model_config.get(
                "stream_stall_timeout", DEFAULT_STREAM_STALL_TIMEOUT
            )
        )

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

    def _apply_cache_control_for_openrouter_anthropic(self, kwargs: dict[str, Any]) -> None:
        """Add top-level cache_control for OpenRouter Anthropic prompt caching.

        Uses the same approach as AnthropicModel: a single top-level cache_control
        that lets OpenRouter automatically place the breakpoint at the last cacheable
        block and move it forward as the conversation grows.

        Args:
            kwargs: The request kwargs dict (``chat.completions.create`` or
                ``responses.create``).  Mutated in place.
        """
        if not self._is_openrouter_anthropic():
            return
        if not self.model_config.get("enable_cache", True):
            return
        existing = kwargs.get("extra_body")
        extra_body = dict(existing) if isinstance(existing, dict) else {}
        extra_body["cache_control"] = {"type": "ephemeral"}
        kwargs["extra_body"] = extra_body


class OpenAICompatibleModel(OpenAICompatibleBase):
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
        self._effort_verdict_key = (base_url, self._api_model_name)
        self._responses_delegate: OpenAICompatibleModel2 | None = None
        self._delegate_raw_items: dict[str, list[dict[str, Any]]] = {}
        thinking_level = _model_thinking_level(self.model_name)
        reasoning_cfg = self.model_config.get("reasoning")
        has_native_effort = (
            isinstance(reasoning_cfg, dict) and "effort" in reasoning_cfg
        )
        if (
            thinking_level is not None
            and "reasoning_effort" not in self.model_config
            and not has_native_effort
        ):
            self.model_config = dict(self.model_config)
            self.model_config["reasoning_effort"] = thinking_level
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

        Only the formats OpenAI actually accepts are carried
        (:data:`OPENAI_INPUT_IMAGE_MIME_TYPES`,
        :data:`OPENAI_INPUT_AUDIO_FORMATS` and PDFs) — the same set
        :class:`~kiss.core.models.openai_compatible_model2.OpenAICompatibleModel2`
        applies, so a turn delegated to the Responses transport carries
        exactly the same attachments.  Anything else returns ``None`` with a
        warning naming the format, rather than being sent for the provider
        to reject or dropped in silence.

        Args:
            att: The attachment to convert.

        Returns:
            A content-part dict suitable for the ``content`` array of a
            chat-completions message, or ``None`` if OpenAI does not accept
            the format.
        """
        if att.mime_type == "application/pdf":
            return {"type": "file", "file": {"file_data": att.to_data_url()}}
        if att.mime_type in OPENAI_INPUT_IMAGE_MIME_TYPES:
            return {"type": "image_url", "image_url": {"url": att.to_data_url()}}
        if att.mime_type.startswith("audio/"):
            fmt = _audio_mime_to_format(att.mime_type)
            if fmt in OPENAI_INPUT_AUDIO_FORMATS:
                return {
                    "type": "input_audio",
                    "input_audio": {"data": att.to_base64(), "format": fmt},
                }
        logger.warning(
            "OpenAI does not accept %s attachments (images: %s; audio: %s; "
            "application/pdf); skipping.",
            att.mime_type,
            ", ".join(sorted(OPENAI_INPUT_IMAGE_MIME_TYPES)),
            ", ".join(sorted(OPENAI_INPUT_AUDIO_FORMATS)),
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

    def _deliver_tool_result_attachments(
        self, attachments: list[Attachment]
    ) -> None:
        """Re-attach tool-result bytes as Chat Completions content parts.

        The ``tool`` role only accepts string content, so binary payloads a
        tool produced (a PNG screenshot, an MP3) cannot live inside the tool
        message: they are carried by a follow-up ``user`` message holding
        ``image_url`` / ``file`` / ``input_audio`` parts.  Formats OpenAI
        does not accept are dropped by
        :meth:`_attachments_to_content_parts` with a warning, and when that
        leaves nothing to send no message is appended at all.

        Args:
            attachments: The attachments lifted out of the tool results.
        """
        parts = self._attachments_to_content_parts(attachments)
        if not parts:
            return
        parts.append({"type": "text", "text": TOOL_RESULT_ATTACHMENT_NOTE})
        self.conversation.append({"role": "user", "content": parts})

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
            msg_copy["tool_calls"] = _stringify_tool_call_arguments(msg_copy["tool_calls"])
        attachments = msg_copy.pop("attachments", None)
        if attachments:
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

    @staticmethod
    def _response_error_message(response: Any) -> str:
        """Return the gateway-level error message carried by a response.

        Several OpenAI-compatible gateways answer an upstream failure
        with ``200 {"choices": [], "error": {...}}`` rather than a 4xx,
        so the only diagnosis available is this extra field.

        Args:
            response: A Chat Completions response (SDK object or dict).

        Returns:
            The error message, or ``""`` when the response carries none.
        """
        err = (
            response.get("error")
            if isinstance(response, dict)
            else getattr(response, "error", None)
        )
        if err is None:
            return ""
        message = (
            err.get("message")
            if isinstance(err, dict)
            else getattr(err, "message", None)
        )
        return str(message or err)

    @staticmethod
    def _raise_for_finish_reason(finish_reason: str | None) -> None:
        """Reject a truncated completion instead of using its partial output.

        ``finish_reason="length"`` means the model hit its output-token
        budget mid-sentence — and, on a tool-bearing turn, mid-JSON.  The
        partial ``arguments`` string then fails to parse and
        :meth:`_build_tool_call_lists` turns it into ``{}``, so the agent
        calls the tool with no arguments and is told the tool failed for
        a reason unrelated to the real cause.  The Responses transport
        already refuses the equivalent ``status="incomplete"`` response.

        Args:
            finish_reason: The choice's ``finish_reason``, if any.

        Raises:
            KISSError: When the completion was truncated.
        """
        if finish_reason == "length":
            raise KISSError(
                "Chat Completions response was truncated "
                "(finish_reason='length'): the model hit its output-token "
                "budget, so its text and any tool-call arguments are "
                "incomplete. Raise max_tokens or lower reasoning_effort."
            )

    @classmethod
    def _raise_for_failed_completion(cls, response: Any) -> None:
        """Reject a non-streamed response that carries no usable choice.

        The counterpart of
        :meth:`~kiss.core.models.openai_compatible_model2.OpenAICompatibleModel2._raise_for_failed_response`
        for the Chat Completions transport.

        Args:
            response: The response returned by ``chat.completions.create``.

        Raises:
            KISSError: When ``choices`` is empty (surfacing the gateway's
                own error message) or the single choice was truncated.
        """
        choices = getattr(response, "choices", None) or []
        if not choices:
            message = cls._response_error_message(response)
            raise KISSError(
                "Chat Completions returned no choices"
                + (f": {message}" if message else "")
            )
        cls._raise_for_finish_reason(getattr(choices[0], "finish_reason", None))

    def _stream_chat_completion(
        self, kwargs: dict[str, Any], adaptive: bool
    ) -> tuple[str, dict[int, dict[str, str]], Any, str | None]:
        """Stream one chat completion, driving the token/thinking callbacks.

        The single streaming loop behind both streamed entry points —
        :meth:`_stream_text` (tool-less) and
        :meth:`generate_and_process_with_tools` (the agentic path) — which
        differ only in whether the request goes through the adaptive
        ``reasoning_effort`` probe.

        ``stop_aware_events`` is used instead of a bare ``for chunk in
        ...``: otherwise the thread sits in ``recv()`` on a quiet
        connection for the client's full 1800s timeout, deaf to Stop,
        because the flag is only read when the agent emits something and
        an injected ``KeyboardInterrupt`` cannot reach a thread inside C
        code (``reports/stop_button_delay_2026-08-05.html``).

        Args:
            kwargs: Fully built request arguments; ``stream`` and
                ``stream_options`` are set here.
            adaptive: Send through :meth:`_create_chat_completion_adaptive`
                (which may retry once without ``reasoning_effort``)
                instead of a plain create.

        Returns:
            ``(content, tool_call_accumulator, response, finish_reason)``.
        """
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}
        content = ""
        tool_calls_accum: dict[int, dict[str, str]] = {}
        response = None
        last_chunk = None
        finish_reason: str | None = None
        stream = (
            self._create_chat_completion_adaptive(kwargs)
            if adaptive
            else self.client.chat.completions.create(**kwargs)
        )
        # The bracket is closed in `finally`, not after the loop:
        # `stop_aware_events` runs `on_abort` for a stop and for a stall
        # but re-raises every other transport failure untouched, and
        # KISSAgent retries those in the SAME run without resetting the
        # model — so a provider that drops the connection mid-reasoning
        # would leave the printer rendering the retry's answer as
        # thinking.  `_close_thinking_if_open` is a no-op when the turn
        # ended outside a reasoning block.
        try:
            for chunk in stop_aware_events(
                stream,
                stall_timeout=self._stream_stall_timeout,
                on_abort=self._close_thinking_if_open,
                name=(
                    "openai-tools-stream-abort-watchdog"
                    if adaptive
                    else "openai-stream-abort-watchdog"
                ),
            ):
                last_chunk = chunk
                if chunk.choices:
                    choice = chunk.choices[0]
                    if getattr(choice, "finish_reason", None):
                        finish_reason = choice.finish_reason
                    delta = choice.delta
                    if delta:
                        reasoning = _delta_reasoning_text(delta)
                        if reasoning:
                            if not self._thinking_open:
                                self._invoke_thinking_callback(True)
                            self._invoke_token_callback(reasoning)
                        if delta.content:
                            self._close_thinking_if_open()
                            content += delta.content
                            self._invoke_token_callback(delta.content)
                        if delta.tool_calls:
                            self._close_thinking_if_open()
                            _accumulate_tool_call_deltas(
                                tool_calls_accum, delta.tool_calls
                            )
                if chunk.usage is not None:
                    response = chunk
        finally:
            self._close_thinking_if_open()
        response = self._finalize_stream_response(response, last_chunk)
        return content, tool_calls_accum, response, finish_reason

    def _stream_text(self, kwargs: dict[str, Any]) -> tuple[str, Any]:
        """Stream a chat completion, invoking the token callback for each text delta.

        When no callback is set, falls back to a normal (non-streaming) call.

        Args:
            kwargs: Keyword arguments for the OpenAI chat completions API.

        Returns:
            A tuple of (content, response).
        """
        if self.token_callback is None:
            response = self.client.chat.completions.create(**kwargs)
            self._raise_for_failed_completion(response)
            return response.choices[0].message.content or "", response

        content, _accum, response, finish_reason = self._stream_chat_completion(
            kwargs, adaptive=False
        )
        self._raise_for_finish_reason(finish_reason)
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
        kwargs = self._keep_supported_request_params(
            {
                key: value
                for key, value in self.model_config.items()
                if key not in FRAMEWORK_ONLY_CONFIG_KEYS
            },
            _CHAT_REQUEST_PARAMS,
            "OpenAI Chat Completions",
        )
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

        if self.token_callback is not None:
            (
                content,
                tool_calls_accum,
                response,
                finish_reason,
            ) = self._stream_chat_completion(kwargs, adaptive=True)
            self._raise_for_finish_reason(finish_reason)
            function_calls, raw_tool_calls = self._parse_tool_call_accum(tool_calls_accum)
        else:
            response = self._create_chat_completion_adaptive(kwargs)
            self._raise_for_failed_completion(response)
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
        from kiss.core.models.model_info import openai_compatible_provider_for_base_url

        provider = openai_compatible_provider_for_base_url(self.base_url)
        declared = provider.tools_accept_reasoning_effort if provider else None
        if declared is not None:
            return declared
        return _tool_effort_verdict(self._effort_verdict_key)

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
                _record_tool_effort_verdict(self._effort_verdict_key, False)
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
            _record_tool_effort_verdict(self._effort_verdict_key, True)
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
        # The delegate is cached for its connection pool, so its
        # callbacks have to be re-synced every turn: KISSAgent._reset
        # rebinds THIS model's callbacks to the new run's printer without
        # touching the delegate, which would otherwise keep streaming
        # every token and thinking bracket into a finished task's printer.
        delegate.token_callback = self.token_callback
        delegate.thinking_callback = self.thinking_callback
        delegate._ensure_client()
        delegate.reset_conversation()
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
        self._prune_delegate_raw_items()
        return function_calls, content, response

    def _prune_delegate_raw_items(self) -> None:
        """Drop cached Responses items whose tool call left the conversation.

        The cache exists so a follow-up turn can replay a delegated
        turn's raw output items (reasoning included) verbatim, keyed by
        call_id.  An entry whose call_id is no longer anywhere in the
        conversation can never be looked up again, so keeping it only
        grows the process's memory — one whole turn's payload per tool
        call, for the lifetime of a model instance that Sorcar reuses
        across sub-sessions.
        """
        live = {
            tc.get("id", "")
            for msg in self.conversation
            if isinstance(msg, dict)
            for tc in (msg.get("tool_calls") or [])
        }
        for call_id in [k for k in self._delegate_raw_items if k not in live]:
            del self._delegate_raw_items[call_id]

    def reset_conversation(self) -> None:
        """Reset the conversation and every cache derived from it.

        The base implementation clears only the conversation, but this
        transport also caches raw Responses items per call_id and owns a
        Responses delegate with its own per-turn state.  Both are dead
        weight once the conversation they describe is gone, and the
        delegate's leftover pending function calls would reject the very
        next generation on the fresh, empty conversation.
        """
        super().reset_conversation()
        self._delegate_raw_items.clear()
        if self._responses_delegate is not None:
            self._responses_delegate.reset_conversation()

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
    ) -> tuple[int, int, int, int] | tuple[int, int, int, int, int, int, int]:
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
            audio_input_tokens = 0
            audio_output_tokens = 0
            details = getattr(usage, "prompt_tokens_details", None)
            if details is not None:
                cached_tokens = getattr(details, "cached_tokens", 0) or 0
                cache_write_tokens = getattr(details, "cache_write_tokens", 0) or 0
                audio_input_tokens = getattr(details, "audio_tokens", 0) or 0
            completion_details = getattr(usage, "completion_tokens_details", None)
            if completion_details is not None:
                audio_output_tokens = getattr(completion_details, "audio_tokens", 0) or 0
            text_input_tokens = max(
                0,
                prompt_tokens - cached_tokens - cache_write_tokens - audio_input_tokens,
            )
            text_output_tokens = max(0, completion_tokens - audio_output_tokens)
            if audio_input_tokens or audio_output_tokens:
                return (
                    text_input_tokens,
                    text_output_tokens,
                    cached_tokens,
                    cache_write_tokens,
                    0,
                    audio_input_tokens,
                    audio_output_tokens,
                )
            return (
                text_input_tokens,
                text_output_tokens,
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
