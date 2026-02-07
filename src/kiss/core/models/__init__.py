# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Model implementations for different LLM providers."""

from kiss.core.models.model import Model

try:
    from kiss.core.models.openai_compatible_model import OpenAICompatibleModel
except ImportError:
    OpenAICompatibleModel = None  # type: ignore[assignment,misc]

try:
    from kiss.core.models.anthropic_model import AnthropicModel
except ImportError:
    AnthropicModel = None  # type: ignore[assignment,misc]

try:
    from kiss.core.models.gemini_model import GeminiModel
except ImportError:
    GeminiModel = None  # type: ignore[assignment,misc]

__all__ = ["Model", "AnthropicModel", "OpenAICompatibleModel", "GeminiModel"]
