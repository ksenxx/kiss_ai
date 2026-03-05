# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Model implementations for different LLM providers."""

import logging

from kiss.core.models.model import Attachment, Model

logger = logging.getLogger(__name__)

try:
    from kiss.core.models.openai_compatible_model import OpenAICompatibleModel
except ImportError:
    logger.debug("Exception caught", exc_info=True)
    OpenAICompatibleModel = None  # type: ignore[assignment,misc]

try:
    from kiss.core.models.anthropic_model import AnthropicModel
except ImportError:
    logger.debug("Exception caught", exc_info=True)
    AnthropicModel = None  # type: ignore[assignment,misc]

try:
    from kiss.core.models.gemini_model import GeminiModel
except ImportError:
    logger.debug("Exception caught", exc_info=True)
    GeminiModel = None  # type: ignore[assignment,misc]

__all__ = ["Attachment", "Model", "AnthropicModel", "OpenAICompatibleModel", "GeminiModel"]
