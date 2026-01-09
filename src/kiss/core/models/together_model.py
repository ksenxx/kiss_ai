# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Together AI model implementation for KISS agent."""

import json
from collections.abc import Callable
from typing import Any

from openai import OpenAI

from kiss.core.config import DEFAULT_CONFIG
from kiss.core.models.openai_compatible_model import OpenAICompatibleModel


class TogetherModel(OpenAICompatibleModel):
    """Together AI model using OpenAI-compatible API."""

    DEFAULT_EMBEDDING_MODEL = "togethercomputer/m2-bert-80M-32k-retrieval"

    def initialize(self, prompt: str) -> None:
        self.client = OpenAI(
            api_key=DEFAULT_CONFIG.agent.api_keys.TOGETHER_API_KEY,
            base_url="https://api.together.xyz/v1",
        )
        self.conversation = [{"role": "user", "content": self._append_usage_info(prompt)}]

    def _is_deepseek_model(self) -> bool:
        """Check if this is a DeepSeek model that needs special handling."""
        return self.model_name.startswith("deepseek-ai/DeepSeek-R1") or self.model_name in (
            "deepseek-ai/DeepSeek-V3",
            "deepseek-ai/DeepSeek-V3.1",
        )

    def _create_api_kwargs(self, function_map: dict[str, Callable[..., Any]]) -> dict[str, Any]:
        kwargs = super()._create_api_kwargs(function_map)
        if self.model_name.startswith("deepseek-ai/"):
            kwargs["extra_body"] = {"reasoning": {"enabled": False}}
        if self._is_deepseek_model():
            kwargs["tool_choice"] = "required"
        return kwargs

    def add_model_response_to_conversation(self, response: Any) -> None:
        if not response.choices or not response.choices[0].message:
            return
        msg = response.choices[0].message

        # DeepSeek models: convert tool calls to plain text for better multi-turn handling
        if self._is_deepseek_model() and msg.tool_calls:
            self.conversation.append(
                {
                    "role": "assistant",
                    "content": "\n".join(
                        f"Calling {tc.function.name} with arguments: {tc.function.arguments}"
                        for tc in msg.tool_calls
                    ),
                }
            )
            return
        super().add_model_response_to_conversation(response)

    def add_function_results_to_conversation_and_return(
        self, function_results: list[tuple[str, dict[str, Any]]]
    ) -> None:
        if not function_results or not self.conversation:
            return

        if self._is_deepseek_model():
            content = "\n".join(
                f"The result of {n} is: {json.dumps(r)}" for n, r in function_results
            )
            content += "\n\nIf you have the final answer, call 'finish'. Otherwise, continue."
            self.conversation.append({"role": "user", "content": content})
            return
        super().add_function_results_to_conversation_and_return(function_results)
