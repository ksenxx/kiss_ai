# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Model implementations for different LLM providers."""

import importlib

from kiss.core.models.model import Attachment, Model

__all__ = [
    "Attachment",
    "Model",
    "AnthropicModel",
    "ClaudeCodeModel",
    "CodexModel",
    "OpenAICompatibleModel",
    "OpenAICompatibleModel2",
    "GeminiModel",
]

_LAZY_IMPORTS = {
    "AnthropicModel": "kiss.core.models.anthropic_model",
    "ClaudeCodeModel": "kiss.core.models.claude_code_model",
    "CodexModel": "kiss.core.models.codex_model",
    "OpenAICompatibleModel": "kiss.core.models.openai_compatible_model",
    "OpenAICompatibleModel2": "kiss.core.models.openai_compatible_model2",
    "GeminiModel": "kiss.core.models.gemini_model",
}


def __getattr__(name: str) -> type:
    """Lazily import a model class on first access.

    Keeps the provider SDKs out of the import graph until a model of
    that provider is actually constructed.

    A failed import is deliberately **not** cached: writing ``None`` into
    ``globals()`` would stop Python from ever calling this hook for that
    name again, so a transient or environmental failure (a broken
    ``urllib3``, a half-installed venv) would keep the long-running
    ``kiss-web`` daemon from loading the class until it is restarted.
    The real :class:`ImportError` propagates so the caller can report
    what actually went wrong instead of guessing "SDK not installed".

    Args:
        name: The attribute being looked up.

    Returns:
        The imported model class.

    Raises:
        AttributeError: When *name* is not a lazily-imported model class.
        ImportError: When the class or its SDK could not be imported.
    """
    if name in _LAZY_IMPORTS:
        module = importlib.import_module(_LAZY_IMPORTS[name])
        cls: type = getattr(module, name)
        globals()[name] = cls
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
