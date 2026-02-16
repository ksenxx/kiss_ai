# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Anthropic model implementation for Claude models."""

from collections.abc import Callable
from typing import Any

from anthropic import Anthropic

from kiss.core.kiss_error import KISSError
from kiss.core.models.model import Model, TokenCallback


class AnthropicModel(Model):
    """A model that uses Anthropic's Messages API (Claude)."""

    def __init__(
        self,
        model_name: str,
        api_key: str,
        model_config: dict[str, Any] | None = None,
        token_callback: TokenCallback | None = None,
    ):
        """Initialize an AnthropicModel instance.

        Args:
            model_name: The name of the Claude model to use.
            api_key: The Anthropic API key for authentication.
            model_config: Optional dictionary of model configuration parameters.
            token_callback: Optional async callback invoked with each streamed text token.
        """
        super().__init__(model_name, model_config=model_config, token_callback=token_callback)
        self.api_key = api_key

    def initialize(self, prompt: str) -> None:
        """Initializes the conversation with an initial user prompt.

        Args:
            prompt: The initial user prompt to start the conversation.
        """
        self.client = Anthropic(api_key=self.api_key)
        self.conversation = [{"role": "user", "content": prompt}]

    def _normalize_content_blocks(self, content: Any) -> list[dict[str, Any]]:
        """Normalize Anthropic content blocks to JSON-serializable dicts.

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
                blocks.append(block)
                continue
            if hasattr(block, "model_dump"):
                blocks.append(block.model_dump())
                continue
            block_type = getattr(block, "type", None)
            if block_type == "text":
                blocks.append({"type": "text", "text": getattr(block, "text", "")})
            elif block_type == "tool_use":
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": getattr(block, "id", ""),
                        "name": getattr(block, "name", ""),
                        "input": getattr(block, "input", {}) or {},
                    }
                )
            else:
                blocks.append({"type": "text", "text": str(block)})
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
        self, function_map: dict[str, Callable[..., Any]]
    ) -> list[dict[str, Any]]:
        """Build Anthropic tools schema from a function map.

        Args:
            function_map: Dictionary mapping function names to callable functions.

        Returns:
            list[dict[str, Any]]: A list of tool schemas in Anthropic format.
        """
        tools = []
        for tool in self._build_openai_tools_schema(function_map):
            fn = tool.get("function", {})
            tools.append(
                {
                    "name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
                }
            )
        return tools

    def _build_create_kwargs(self, tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """Build keyword arguments for the Anthropic API create call.

        Args:
            tools: Optional list of tool schemas to include.

        Returns:
            dict[str, Any]: The keyword arguments for the API call.
        """
        kwargs = self.model_config.copy()

        # Anthropic requires max_tokens; accept OpenAI-style "max_completion_tokens" too.
        max_tokens = kwargs.pop("max_tokens", None)
        if max_tokens is None:
            max_tokens = kwargs.pop("max_completion_tokens", None)
        user_set_max_tokens = max_tokens is not None
        if max_tokens is None:
            max_tokens = 16384

        # Map OpenAI-style stop -> Anthropic stop_sequences (best-effort).
        if "stop" in kwargs and "stop_sequences" not in kwargs:
            stop_val = kwargs.pop("stop")
            if isinstance(stop_val, str):
                kwargs["stop_sequences"] = [stop_val]
            elif isinstance(stop_val, list):
                kwargs["stop_sequences"] = stop_val

        # Enable thinking by default for Claude 4.x+ models.
        if "thinking" not in kwargs and (
            self.model_name.startswith(("claude-opus-4", "claude-sonnet-4", "claude-haiku-4"))
        ):
            if self.model_name.startswith("claude-opus-4-6"):
                kwargs["thinking"] = {"type": "adaptive"}
            else:
                kwargs["thinking"] = {"type": "enabled", "budget_tokens": 10000}
                if not user_set_max_tokens and max_tokens < 10000:
                    max_tokens = 16384

        kwargs.update(
            {
                "model": self.model_name,
                "messages": self.conversation,
                "max_tokens": max_tokens,
            }
        )
        if tools:
            kwargs["tools"] = tools
        return kwargs

    def _create_message(self, kwargs: dict[str, Any]) -> Any:
        """Create a message, streaming tokens to the callback when set.

        Args:
            kwargs: Keyword arguments for the Anthropic API call.

        Returns:
            The raw Anthropic response message.
        """
        if self.token_callback is not None:
            with self.client.messages.stream(**kwargs) as stream:
                for event in stream:
                    if event.type == "content_block_delta":
                        delta = event.delta
                        delta_type = getattr(delta, "type", "")
                        if delta_type == "thinking_delta":
                            self._invoke_token_callback(getattr(delta, "thinking", ""))
                        elif delta_type == "text_delta":
                            self._invoke_token_callback(getattr(delta, "text", ""))
            return stream.get_final_message()
        return self.client.messages.create(**kwargs)

    def generate(self) -> tuple[str, Any]:
        """Generates content from the current conversation.

        Returns:
            tuple[str, Any]: A tuple of (generated_text, raw_response).
        """
        kwargs = self._build_create_kwargs()
        response = self._create_message(kwargs)

        blocks = self._normalize_content_blocks(getattr(response, "content", None))
        content = self._extract_text_from_blocks(blocks)
        self.conversation.append({"role": "assistant", "content": blocks or content})
        return content, response

    def generate_and_process_with_tools(
        self, function_map: dict[str, Callable[..., Any]]
    ) -> tuple[list[dict[str, Any]], str, Any]:
        """Generates content with tools and processes the response.

        Args:
            function_map: Dictionary mapping function names to callable functions.

        Returns:
            tuple[list[dict[str, Any]], str, Any]: A tuple of
                (function_calls, response_text, raw_response).
        """
        tools = self._build_anthropic_tools_schema(function_map)
        kwargs = self._build_create_kwargs(tools=tools or None)
        response = self._create_message(kwargs)

        stop_reason = getattr(response, "stop_reason", None)
        blocks = self._normalize_content_blocks(getattr(response, "content", None))

        # When max_tokens is hit, tool_use blocks may be truncated/incomplete.
        # Strip them to avoid calling tools with missing arguments.
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

        self.conversation.append({"role": "assistant", "content": blocks or content})
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
        # Collect all tool_use blocks from the most recent assistant message
        # Use a list to preserve order and handle multiple calls to the same function
        tool_use_ids: list[tuple[str, str]] = []  # [(name, id), ...]
        for msg in reversed(self.conversation):
            if msg.get("role") == "assistant" and isinstance(msg.get("content"), list):
                for b in msg["content"]:
                    if b.get("type") == "tool_use":
                        tool_use_ids.append((b.get("name", ""), b.get("id", "")))
                if tool_use_ids:
                    break

        tool_results_blocks: list[dict[str, Any]] = []
        for i, (func_name, result_dict) in enumerate(function_results):
            result_content = result_dict.get("result", str(result_dict))
            if self.usage_info_for_messages:
                result_content = f"{result_content}\n\n{self.usage_info_for_messages}"

            # Use explicit tool_use_id if provided, otherwise match by position
            tool_use_id = result_dict.get("tool_use_id")
            if tool_use_id is None and i < len(tool_use_ids):
                tool_use_id = tool_use_ids[i][1]
            if tool_use_id is None:
                tool_use_id = f"toolu_{func_name}_{i}"

            tool_results_blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": result_content,
                }
            )

        self.conversation.append({"role": "user", "content": tool_results_blocks})

    def add_message_to_conversation(self, role: str, content: str) -> None:
        """Adds a message to the conversation state.

        Args:
            role: The role of the message sender (e.g., 'user', 'assistant').
            content: The message content.
        """
        if role == "user" and self.usage_info_for_messages:
            content = f"{content}\n\n{self.usage_info_for_messages}"
        self.conversation.append({"role": role, "content": content})

    def extract_input_output_token_counts_from_response(self, response: Any) -> tuple[int, int]:
        """Extracts input and output token counts from an API response.

        Args:
            response: The raw Anthropic API response object.

        Returns:
            tuple[int, int]: A tuple of (input_tokens, output_tokens).
        """
        if hasattr(response, "usage") and response.usage:
            return (
                getattr(response.usage, "input_tokens", 0) or 0,
                getattr(response.usage, "output_tokens", 0) or 0,
            )
        return 0, 0

    def get_embedding(self, text: str, embedding_model: str | None = None) -> list[float]:
        """Generates an embedding vector for the given text.

        Args:
            text: The text to generate an embedding for.
            embedding_model: Optional model name (not used by Anthropic).

        Raises:
            KISSError: Anthropic does not provide an embeddings API.
        """
        raise KISSError("Anthropic does not provide an embeddings API.")
