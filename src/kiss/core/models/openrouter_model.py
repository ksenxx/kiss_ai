# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""OpenRouter model implementation for KISS agent."""

from collections.abc import Callable
from typing import Any

from openai import OpenAI

from kiss.core.config import DEFAULT_CONFIG
from kiss.core.kiss_error import KISSError
from kiss.core.models.openai_compatible_model import OpenAICompatibleModel


class OpenRouterModel(OpenAICompatibleModel):
    """OpenRouter model using OpenAI-compatible API.

    Note: model_name is the full name with 'openrouter/' prefix (for MODEL_INFO lookups),
    while api_model_name is the name used for API calls (without 'openrouter/' prefix).
    """

    DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-small"

    def __init__(self, api_model_name: str, model_description: str = ""):
        # Store the API model name (e.g., "openai/gpt-4o")
        self.api_model_name = api_model_name
        # Use the full model name for MODEL_INFO lookups (e.g., "openrouter/openai/gpt-4o")
        full_model_name = f"openrouter/{api_model_name}"
        super().__init__(full_model_name, model_description)

    def initialize(self, prompt: str) -> None:
        self.client = OpenAI(
            api_key=DEFAULT_CONFIG.agent.api_keys.OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
        )
        self.conversation = [{"role": "user", "content": self._append_usage_info(prompt)}]

    def generate(self) -> tuple[str, Any]:
        """Generate using the API model name."""
        assert self.client
        prompt = self._get_initial_prompt_text()
        if not prompt:
            raise KISSError("No prompt provided.")
        response = self.client.chat.completions.create(
            model=self.api_model_name, messages=[{"role": "user", "content": prompt}]
        )
        if not response.choices or not response.choices[0].message:
            raise KISSError("No response from model.")
        return response.choices[0].message.content or "", response

    def generate_content_with_tools(self, function_map: dict[str, Callable[..., Any]]) -> Any:
        """Generate with tools using the API model name."""
        assert self.client
        kwargs = self._create_api_kwargs(function_map)
        # Override model name with API model name
        kwargs["model"] = self.api_model_name
        return self.client.chat.completions.create(**kwargs)

