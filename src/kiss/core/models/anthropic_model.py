# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Anthropic model implementation for KISS agent."""

import json
from collections.abc import Callable
from typing import Any

from anthropic import Anthropic

from kiss.core.config import DEFAULT_CONFIG
from kiss.core.kiss_error import KISSError
from kiss.core.models.model import DictConversationModel


class AnthropicModel(DictConversationModel):
    """Anthropic Claude model implementation."""

    client: Anthropic | None

    def _get_tools(self, function_map: dict[str, Callable[..., Any]]) -> list[dict[str, Any]]:
        return [
            {"name": s["name"], "description": s["description"], "input_schema": s["schema"]}
            for s in self._build_function_schema(function_map)
        ]

    def _extract_function_calls_from_response(self, response: Any) -> list[dict[str, Any]]:
        if not response.content:
            return []
        return [
            {"name": b.name, "arguments": b.input if isinstance(b.input, dict) else {}, "id": b.id}
            for b in response.content
            if b.type == "tool_use"
        ]

    def _extract_text_from_response(self, response: Any) -> str:
        return "".join(b.text for b in (response.content or []) if b.type == "text")

    def initialize(self, prompt: str) -> None:
        self.client = Anthropic(api_key=DEFAULT_CONFIG.agent.api_keys.ANTHROPIC_API_KEY)
        self.conversation = [{"role": "user", "content": self._append_usage_info(prompt)}]

    def generate_content_with_tools(self, function_map: dict[str, Callable[..., Any]]) -> Any:
        assert self.client
        tools = self._get_tools(function_map)
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": 4096,
            "messages": self.conversation,
            "temperature": 1.0,
        }
        if tools:
            kwargs["tools"] = tools
        return self.client.messages.create(**kwargs)

    def add_model_response_to_conversation(self, response: Any) -> None:
        if not response.content:
            return
        blocks = []
        for b in response.content:
            if b.type == "text":
                blocks.append({"type": "text", "text": b.text})
            elif b.type == "tool_use":
                blocks.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})
        self.conversation.append({"role": "assistant", "content": blocks})

    def generate(self) -> tuple[str, Any]:
        assert self.client
        prompt = self._get_initial_prompt_text()
        if not prompt:
            raise KISSError("No prompt provided.")
        response = self.client.messages.create(
            model=self.model_name, max_tokens=4096, messages=[{"role": "user", "content": prompt}]
        )
        if not response.content:
            raise KISSError("No response from Anthropic model.")
        return "".join(b.text for b in response.content if b.type == "text"), response

    def extract_input_output_token_counts_from_response(self, response: Any) -> tuple[int, int]:
        return (
            (response.usage.input_tokens or 0, response.usage.output_tokens or 0)
            if response.usage
            else (0, 0)
        )

    def add_function_results_to_conversation_and_return(
        self, function_results: list[tuple[str, dict[str, Any]]]
    ) -> None:
        if not function_results or not self.conversation:
            return
        last = self.conversation[-1]
        if last.get("role") != "assistant" or not last.get("content"):
            return
        name_to_id = {
            b.get("name", ""): b.get("id", "")
            for b in last["content"]
            if b.get("type") == "tool_use"
        }
        blocks = []
        for name, result in function_results:
            content = json.dumps(result)
            blocks.append(
                {"type": "tool_result", "tool_use_id": name_to_id.get(name, ""), "content": content}
            )
        self.conversation.append({"role": "user", "content": blocks})

    def get_embedding(self, text: str, embedding_model: str | None = None) -> list[float]:
        raise NotImplementedError("Anthropic does not provide an embeddings API.")
