# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Anthropic model implementation for Claude models."""

import logging
from collections.abc import Callable
from typing import Any

from anthropic import Anthropic

from kiss.core.kiss_error import KISSError
from kiss.core.models.model import (
    Attachment,
    Model,
    ThinkingCallback,
    TokenCallback,
    parse_binary_attachments,
    transcribe_audio,
)

logger = logging.getLogger(__name__)


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
                            {
                                "type": "text",
                                "text": f"[Audio transcription]\n{text}",
                            }
                        )
                    except Exception:
                        logger.warning(
                            "Anthropic does not support %s attachments and "
                            "automatic transcription failed; skipping.",
                            att.mime_type,
                        )
                elif att.mime_type.startswith("video/"):
                    logger.warning(
                        "Anthropic does not support %s attachments; skipping.",
                        att.mime_type,
                    )
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
                # Drop pre-existing whitespace-only text dicts too.
                if block.get("type") == "text" and not block.get("text", "").strip():
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
        messages contain only whitespace-only text blocks.

        Args:
            conversation: The conversation to normalize.

        Returns:
            list[dict[str, Any]]: The normalized conversation.
        """
        normalized: list[dict[str, Any]] = []
        for msg in conversation:
            msg_copy = msg.copy()
            content = msg_copy.get("content")

            # If content is a string, ensure it's non-whitespace
            if isinstance(content, str):
                if content.strip():
                    normalized.append(msg_copy)
                # Skip messages with whitespace-only string content
            # If content is a list of blocks, normalize them
            elif isinstance(content, list):
                normalized_blocks = self._normalize_content_blocks(content)
                if normalized_blocks:
                    msg_copy["content"] = normalized_blocks
                    normalized.append(msg_copy)
                # Skip messages where all blocks were dropped

        return normalized

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

        if "thinking" not in kwargs and (
            self.model_name.startswith(("claude-opus-4", "claude-sonnet-4", "claude-haiku-4"))
        ):
            if not user_set_max_tokens:
                max_tokens = 65536 if self.model_name.startswith("claude-opus-4") else 64000
            if self.model_name in ["claude-opus-4-6", "claude-opus-4-7"]:
                kwargs["thinking"] = {"type": "adaptive"}
            else:
                kwargs["thinking"] = {"type": "enabled", "budget_tokens": 10000}

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

        if enable_cache:
            kwargs["cache_control"] = {"type": "ephemeral"}

        return kwargs

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

    def generate(self) -> tuple[str, Any]:  # pragma: no cover – API call
        """Generates content from the current conversation.

        Returns:
            tuple[str, Any]: A tuple of (generated_text, raw_response).
        """
        kwargs = self._build_create_kwargs()
        response = self._create_message(kwargs)

        blocks = self._normalize_content_blocks(getattr(response, "content", None))
        content = self._extract_text_from_blocks(blocks)
        # Prefer blocks (which have been normalized) over content string
        msg_content = blocks if blocks else content
        # Only append non-empty messages
        if msg_content:
            self.conversation.append({"role": "assistant", "content": msg_content})
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

        # Prefer blocks (which have been normalized) over content string
        msg_content = blocks if blocks else content
        # Only append non-empty messages
        if msg_content:
            self.conversation.append({"role": "assistant", "content": msg_content})
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
                for att in attachments:
                    source = {
                        "type": "base64",
                        "media_type": att.mime_type,
                        "data": att.to_base64(),
                    }
                    if att.mime_type.startswith("image/"):
                        content_blocks.append({"type": "image", "source": source})
                    elif att.mime_type == "application/pdf":
                        content_blocks.append({"type": "document", "source": source})
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
    ) -> tuple[int, int, int, int]:
        """Extracts token counts from an Anthropic API response.

        Returns:
            (input_tokens, output_tokens, cache_read_tokens, cache_write_tokens).
        """
        if hasattr(response, "usage") and response.usage:
            return (
                getattr(response.usage, "input_tokens", 0) or 0,
                getattr(response.usage, "output_tokens", 0) or 0,
                getattr(response.usage, "cache_read_input_tokens", 0) or 0,
                getattr(response.usage, "cache_creation_input_tokens", 0) or 0,
            )
        return 0, 0, 0, 0

    def get_embedding(self, text: str, embedding_model: str | None = None) -> list[float]:
        """Generates an embedding vector for the given text.

        Args:
            text: The text to generate an embedding for.
            embedding_model: Optional model name (not used by Anthropic).

        Raises:
            KISSError: Anthropic does not provide an embeddings API.
        """
        raise KISSError("Anthropic does not provide an embeddings API.")
