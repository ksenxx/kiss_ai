# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Model implementations for different LLM providers."""

from kiss.core.models.model import Attachment, Model

__all__ = [
    "Attachment",
    "Model",
    "AnthropicModel",
    "ClaudeCodeModel",
    "OpenAICompatibleModel",
    "GeminiModel",
]

_LAZY_IMPORTS = {
    "AnthropicModel": "kiss.core.models.anthropic_model",
    "ClaudeCodeModel": "kiss.core.models.claude_code_model",
    "OpenAICompatibleModel": "kiss.core.models.openai_compatible_model",
    "GeminiModel": "kiss.core.models.gemini_model",
}


def __getattr__(name: str) -> type:
    """Lazily import model classes on first access to avoid loading LLM SDKs at import time."""
    if name in _LAZY_IMPORTS:
        import importlib
        import logging

        module_path = _LAZY_IMPORTS[name]
        try:
            module = importlib.import_module(module_path)
            cls: type = getattr(module, name)
            globals()[name] = cls  # Cache for subsequent accesses
            return cls
        except ImportError:
            logging.getLogger(__name__).debug("Exception caught", exc_info=True)
            globals()[name] = None  # type: ignore[assignment]
            return None  # type: ignore[return-value]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
