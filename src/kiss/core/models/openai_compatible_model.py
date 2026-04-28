# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""OpenAI-compatible model implementation for custom endpoints."""

import json
import logging
import re
from collections.abc import Callable
from typing import Any

from openai import OpenAI

from kiss.core.kiss_error import KISSError
from kiss.core.models.model import (
    Attachment,
    Model,
    ThinkingCallback,
    TokenCallback,
    _build_text_based_tools_prompt,
    _parse_text_based_tool_calls,
)

logger = logging.getLogger(__name__)

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
    if mime_type in _AUDIO_MIME_TO_FORMAT:
        return _AUDIO_MIME_TO_FORMAT[mime_type]
    return mime_type.split("/", 1)[1] if "/" in mime_type else mime_type


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
        self._api_model_name = (
            model_name[len("openrouter/") :] if model_name.startswith("openrouter/") else model_name
        )

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
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=1800.0,
        )
        self.conversation = []
        system_instruction = self.model_config.get("system_instruction")
        if system_instruction:
            self.conversation.append({"role": "system", "content": system_instruction})
        content: str | list[dict[str, Any]] = prompt
        if attachments:
            parts: list[dict[str, Any]] = []
            for att in attachments:
                if att.mime_type.startswith("image/"):
                    parts.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": att.to_data_url()},
                        }
                    )
                elif att.mime_type == "application/pdf":
                    parts.append(
                        {
                            "type": "file",
                            "file": {"file_data": att.to_data_url()},
                        }
                    )
                elif att.mime_type.startswith("audio/"):
                    fmt = _audio_mime_to_format(att.mime_type)
                    parts.append(
                        {
                            "type": "input_audio",
                            "input_audio": {"data": att.to_base64(), "format": fmt},
                        }
                    )
                elif att.mime_type.startswith("video/"):
                    logger.warning(
                        "OpenAI Chat Completions does not support video attachments; "
                        "skipping %s.",
                        att.mime_type,
                    )
            parts.append({"type": "text", "text": prompt})
            content = parts
        self.conversation.append({"role": "user", "content": content})

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

    def generate(self) -> tuple[str, Any]:
        """Generate content from prompt without tools.

        Returns:
            A tuple of (content, response) where content is the generated text
            and response is the raw API response object.
        """
        kwargs = self.model_config.copy()
        kwargs.pop("system_instruction", None)
        kwargs.update(
            {
                "model": self._api_model_name,
                "messages": self.conversation,
            }
        )
        self._apply_cache_control_for_openrouter_anthropic(kwargs)

        content, response = self._stream_text(kwargs)

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
        kwargs = self.model_config.copy()
        kwargs.pop("system_instruction", None)
        kwargs.update(
            {
                "model": self._api_model_name,
                "messages": self.conversation,
                "tools": tools or None,
            }
        )
        self._apply_cache_control_for_openrouter_anthropic(kwargs)

        if self.token_callback is not None:  # pragma: no cover – API streaming
            kwargs["stream"] = True
            kwargs["stream_options"] = {"include_usage": True}
            content = ""
            tool_calls_accum: dict[int, dict[str, str]] = {}
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
            response = self.client.chat.completions.create(**kwargs)
            message = response.choices[0].message
            content = message.content or ""
            function_calls, raw_tool_calls = self._parse_tool_calls_from_message(message)

        if function_calls:
            self.conversation.append(
                {"role": "assistant", "content": content, "tool_calls": raw_tool_calls}
            )
        else:
            self.conversation.append({"role": "assistant", "content": content})
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

        modified_conversation = list(self.conversation)
        if modified_conversation and modified_conversation[0]["role"] == "user":
            modified_conversation[0] = {
                "role": "user",
                "content": modified_conversation[0]["content"] + "\n" + tools_prompt,
            }
        else:
            modified_conversation.insert(0, {"role": "system", "content": tools_prompt})

        kwargs = self.model_config.copy()
        kwargs.pop("system_instruction", None)
        kwargs.update(
            {
                "model": self._api_model_name,
                "messages": modified_conversation,
            }
        )
        self._apply_cache_control_for_openrouter_anthropic(kwargs)

        content, response = self._stream_text(kwargs)

        _, content_clean = _extract_deepseek_reasoning(content)

        function_calls = _parse_text_based_tool_calls(content_clean)

        if function_calls:
            self.conversation.append({"role": "assistant", "content": content})
            self._replace_last_assistant_with_tool_calls(content, function_calls)
        else:
            self.conversation.append({"role": "assistant", "content": content})

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
        """
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
