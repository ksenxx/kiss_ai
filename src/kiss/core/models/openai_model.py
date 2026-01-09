# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""OpenAI model implementation for KISS agent."""

from openai import OpenAI

from kiss.core.config import DEFAULT_CONFIG
from kiss.core.models.openai_compatible_model import OpenAICompatibleModel


class OpenAIModel(OpenAICompatibleModel):
    """OpenAI model using the OpenAI API."""

    def initialize(self, prompt: str) -> None:
        self.client = OpenAI(api_key=DEFAULT_CONFIG.agent.api_keys.OPENAI_API_KEY)
        self.conversation = [{"role": "user", "content": self._append_usage_info(prompt)}]
