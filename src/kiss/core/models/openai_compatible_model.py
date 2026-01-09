# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Base class for OpenAI-compatible model implementations (OpenAI, Together AI)."""

import json
from collections.abc import Callable
from typing import Any

from openai import OpenAI

from kiss.core.kiss_error import KISSError
from kiss.core.models.model import DictConversationModel


class OpenAICompatibleModel(DictConversationModel):
    """Base class for models using OpenAI-compatible API (OpenAI, Together AI)."""

    DEFAULT_EMBEDDING_MODEL: str = "text-embedding-3-small"
    client: OpenAI | None

    def _get_tools(self, function_map: dict[str, Callable[..., Any]]) -> list[dict[str, Any]]:
        """Converts Python functions to OpenAI-compatible tool format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": s["name"],
                    "description": s["description"],
                    "parameters": s["schema"],
                },
            }
            for s in self._build_function_schema(function_map)
        ]

    def _extract_function_calls_from_response(self, response: Any) -> list[dict[str, Any]]:
        if not response.choices:
            return []
        calls = []
        for c in response.choices:
            if not c.message or not c.message.tool_calls:
                continue
            for tc in c.message.tool_calls:
                if tc.type == "function" and tc.function.name != "function":
                    calls.append(
                        {
                            "name": tc.function.name,
                            "arguments": self._parse_function_args(tc.function.arguments),
                            "id": tc.id,
                        }
                    )
        return calls

    def _extract_text_from_response(self, response: Any) -> str:
        if not response.choices:
            return ""
        return "".join(
            c.message.content for c in response.choices if c.message and c.message.content
        )

    def _create_api_kwargs(self, function_map: dict[str, Callable[..., Any]]) -> dict[str, Any]:
        """Creates base API kwargs for chat completion."""
        tools = self._get_tools(function_map)
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": self.conversation,
            "tool_choice": "auto",
            "temperature": 1.0,
        }
        if tools:
            kwargs["tools"] = tools
        return kwargs

    def generate_content_with_tools(self, function_map: dict[str, Callable[..., Any]]) -> Any:
        assert self.client
        return self.client.chat.completions.create(**self._create_api_kwargs(function_map))

    def add_model_response_to_conversation(self, response: Any) -> None:
        if not response.choices or not response.choices[0].message:
            return
        msg = response.choices[0].message
        d: dict[str, Any] = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]
        self.conversation.append(d)

    def generate(self) -> tuple[str, Any]:
        assert self.client
        prompt = self._get_initial_prompt_text()
        if not prompt:
            raise KISSError("No prompt provided.")
        response = self.client.chat.completions.create(
            model=self.model_name, messages=[{"role": "user", "content": prompt}]
        )
        if not response.choices or not response.choices[0].message:
            raise KISSError("No response from model.")
        return response.choices[0].message.content or "", response

    def extract_input_output_token_counts_from_response(self, response: Any) -> tuple[int, int]:
        return (
            (response.usage.prompt_tokens or 0, response.usage.completion_tokens or 0)
            if response.usage
            else (0, 0)
        )

    def add_function_results_to_conversation_and_return(
        self, function_results: list[tuple[str, dict[str, Any]]]
    ) -> None:
        if not function_results or not self.conversation:
            return
        last = self.conversation[-1]
        if last.get("role") != "assistant" or "tool_calls" not in last:
            return
        name_to_id = {
            tc["function"]["name"]: tc["id"]
            for tc in last["tool_calls"]
            if tc.get("type") == "function"
        }
        for name, result in function_results:
            content = json.dumps(result)
            self.conversation.append(
                {
                    "role": "tool",
                    "tool_call_id": name_to_id.get(name, ""),
                    "name": name,
                    "content": content,
                }
            )

    def get_embedding(self, text: str, embedding_model: str | None = None) -> list[float]:
        assert self.client
        return (
            self.client.embeddings.create(
                model=embedding_model or self.DEFAULT_EMBEDDING_MODEL, input=text
            )
            .data[0]
            .embedding
        )
