# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# add your name here

"""Novita model implementation using OpenAI-compatible API."""

import logging

from kiss.core.models.model import TokenCallback
from kiss.core.models.openai_compatible_model import OpenAICompatibleModel

logger = logging.getLogger(__name__)


class NovitaModel(OpenAICompatibleModel):
    """A model that uses Novita's OpenAI-compatible API.

    Novita provides OpenAI-compatible API endpoints for various models.
    See: https://novita.ai/docs/api-reference/introduction
    """

    def __init__(
        self,
        model_name: str,
        api_key: str,
        model_config: dict | None = None,
        token_callback: TokenCallback | None = None,
    ):
        """Initialize a NovitaModel instance.

        Args:
            model_name: The name of the Novita model to use.
            api_key: The Novita API key for authentication.
            model_config: Optional dictionary of model configuration parameters.
            token_callback: Optional async callback invoked with each streamed text token.
        """
        super().__init__(
            model_name=model_name,
            base_url="https://api.novita.ai/openai",
            api_key=api_key,
            model_config=model_config,
            token_callback=token_callback,
        )
