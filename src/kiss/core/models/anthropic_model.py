# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Anthropic model implementation for Claude models."""

from collections.abc import Callable
from typing import Any

from anthropic import Anthropic

from kiss.core.kiss_error import KISSError
from kiss.core.models.model import Model


class AnthropicModel(Model):
    """A model that uses Anthropic's Messages API (Claude)."""

    def __init__(
        self,
        model_name: str,
        api_key: str,
        model_config: dict[str, Any] | None = None,
    ):
        super().__init__(model_name, model_config=model_config)
        self.api_key = api_key

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self.model_name})"

    __repr__ = __str__

    def initialize(self, prompt: str) -> None:
        self.client = Anthropic(api_key=self.api_key)
        self.conversation = [{"role": "user", "content": prompt}]

    def _normalize_content_blocks(self, content: Any) -> list[dict[str, Any]]:
        """Normalize Anthropic content blocks to JSON-serializable dicts."""
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
        return "".join(b.get("text", "") for b in blocks if b.get("type") == "text")

    def _build_anthropic_tools_schema(
        self, function_map: dict[str, Callable[..., Any]]
    ) -> list[dict[str, Any]]:
        """Build Anthropic tools schema from a function map."""
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
        kwargs = self.model_config.copy()

        # Anthropic requires max_tokens; accept OpenAI-style "max_completion_tokens" too.
        max_tokens = kwargs.pop("max_tokens", None)
        if max_tokens is None:
            max_tokens = kwargs.pop("max_completion_tokens", None)
        if max_tokens is None:
            max_tokens = 4096

        # Map OpenAI-style stop -> Anthropic stop_sequences (best-effort).
        if "stop" in kwargs and "stop_sequences" not in kwargs:
            stop_val = kwargs.pop("stop")
            if isinstance(stop_val, str):
                kwargs["stop_sequences"] = [stop_val]
            elif isinstance(stop_val, list):
                kwargs["stop_sequences"] = stop_val

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

    def generate(self) -> tuple[str, Any]:
        kwargs = self._build_create_kwargs()
        response = self.client.messages.create(**kwargs)
        blocks = self._normalize_content_blocks(getattr(response, "content", None))
        content = self._extract_text_from_blocks(blocks)
        self.conversation.append({"role": "assistant", "content": blocks or content})
        return content, response

    def generate_and_process_with_tools(
        self, function_map: dict[str, Callable[..., Any]]
    ) -> tuple[list[dict[str, Any]], str, Any]:
        tools = self._build_anthropic_tools_schema(function_map)
        kwargs = self._build_create_kwargs(tools=tools or None)
        response = self.client.messages.create(**kwargs)

        blocks = self._normalize_content_blocks(getattr(response, "content", None))
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
        # Map tool name -> tool_use id from the most recent assistant message
        tool_use_id_map: dict[str, str] = {}
        for msg in reversed(self.conversation):
            if msg.get("role") == "assistant" and isinstance(msg.get("content"), list):
                for b in msg["content"]:
                    if b.get("type") == "tool_use":
                        tool_use_id_map[b.get("name", "")] = b.get("id", "")
                if tool_use_id_map:
                    break

        tool_results_blocks: list[dict[str, Any]] = []
        for func_name, result_dict in function_results:
            result_content = result_dict.get("result", str(result_dict))
            if self.usage_info_for_messages:
                result_content = f"{result_content}\n\n{self.usage_info_for_messages}"
            tool_results_blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id_map.get(func_name, f"toolu_{func_name}"),
                    "content": result_content,
                }
            )

        self.conversation.append({"role": "user", "content": tool_results_blocks})

    def add_message_to_conversation(self, role: str, content: str) -> None:
        if role == "user" and self.usage_info_for_messages:
            content = f"{content}\n\n{self.usage_info_for_messages}"
        self.conversation.append({"role": role, "content": content})

    def extract_input_output_token_counts_from_response(self, response: Any) -> tuple[int, int]:
        if hasattr(response, "usage") and response.usage:
            return (
                getattr(response.usage, "input_tokens", 0) or 0,
                getattr(response.usage, "output_tokens", 0) or 0,
            )
        return 0, 0

    def get_embedding(self, text: str, embedding_model: str | None = None) -> list[float]:
        raise KISSError("Anthropic does not provide an embeddings API.")

