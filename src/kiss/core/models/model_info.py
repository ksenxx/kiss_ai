# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Model information: pricing and context lengths for supported LLM providers.

FLAKY MODEL MARKERS:
- fc=False: Model has unreliable function calling (use for non-agentic tasks only)
- Models with comments like "FLAKY" have inconsistent behavior
- Models with comments like "SLOW" may timeout on some requests
"""

import json
import shutil
from pathlib import Path
from typing import Any

from kiss.core import config as config_module
from kiss.core.kiss_error import KISSError
from kiss.core.models.model import Model, ThinkingCallback, TokenCallback


class ModelInfo:
    """Container for model metadata including pricing and capabilities."""

    def __init__(
        self,
        context_length: int,
        input_price_per_million: float,
        output_price_per_million: float,
        is_function_calling_supported: bool,
        is_embedding_supported: bool,
        is_generation_supported: bool,
        cache_read_price_per_million: float | None = None,
        cache_write_price_per_million: float | None = None,
        cache_write_1h_price_per_million: float | None = None,
        thinking: str | None = None,
    ):
        self.context_length = context_length
        self.input_price_per_1M = input_price_per_million
        self.output_price_per_1M = output_price_per_million
        self.is_function_calling_supported = is_function_calling_supported
        self.is_embedding_supported = is_embedding_supported
        self.is_generation_supported = is_generation_supported
        self.cache_read_price_per_1M = cache_read_price_per_million
        self.cache_write_price_per_1M = cache_write_price_per_million
        self.cache_write_1h_price_per_1M = cache_write_1h_price_per_million
        self.thinking = thinking


PACKAGE_MODEL_INFO_PATH = Path(__file__).parent / "MODEL_INFO.json"
USER_MODEL_INFO_PATH = Path.home() / ".kiss" / "MODEL_INFO.json"


def _ensure_user_model_info_path() -> Path:
    """Return the path the loader will read MODEL_INFO.json from.

    The user-local copy at ``~/.kiss/MODEL_INFO.json`` is the source of
    truth at runtime; it is created (or refreshed) from the package copy
    bundled with this module when:

    * The user copy does not exist yet (fresh install / dev checkout), or
    * The package copy is newer than the user copy (extension/package
      upgrade brought in a more recent table).

    When the user copy can't be created (read-only filesystem, missing
    HOME), the package copy is returned directly so the module still
    loads. This makes the loader resilient enough to import in sandboxed
    test environments without losing the "user file is source of truth"
    semantics in normal use.
    """
    try:
        if USER_MODEL_INFO_PATH.exists():
            user_mtime = USER_MODEL_INFO_PATH.stat().st_mtime
            pkg_mtime = PACKAGE_MODEL_INFO_PATH.stat().st_mtime
            if pkg_mtime > user_mtime:
                USER_MODEL_INFO_PATH.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(PACKAGE_MODEL_INFO_PATH, USER_MODEL_INFO_PATH)
            return USER_MODEL_INFO_PATH
        USER_MODEL_INFO_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PACKAGE_MODEL_INFO_PATH, USER_MODEL_INFO_PATH)
        return USER_MODEL_INFO_PATH
    except OSError:
        return PACKAGE_MODEL_INFO_PATH


def _build_model_info_entry(entry: dict[str, Any]) -> ModelInfo:
    """Build a :class:`ModelInfo` from one JSON object.

    Recognised keys (all optional except ``context_length``,
    ``input_price_per_1M``, ``output_price_per_1M``):

    * ``fc`` — function-calling support (default ``True``).
    * ``emb`` — embedding model (default ``False``).
    * ``gen`` — generation support (default ``True``).
    * ``thinking`` — highest accepted ``reasoning_effort`` (default ``None``).
    * ``cache_read_price_per_1M`` / ``cache_write_price_per_1M`` /
      ``cache_write_1h_price_per_1M`` — explicit cache pricing
      overrides (default ``None``; otherwise reconstructed from
      ``_apply_cache_pricing``).
    * ``comment`` — free-form annotation, ignored by the loader (kept so
      ``update_models.py`` can persist ``"NEW"`` markers).
    """
    return ModelInfo(
        context_length=entry["context_length"],
        input_price_per_million=entry["input_price_per_1M"],
        output_price_per_million=entry["output_price_per_1M"],
        is_function_calling_supported=entry.get("fc", True),
        is_embedding_supported=entry.get("emb", False),
        is_generation_supported=entry.get("gen", True),
        cache_read_price_per_million=entry.get("cache_read_price_per_1M"),
        cache_write_price_per_million=entry.get("cache_write_price_per_1M"),
        cache_write_1h_price_per_million=entry.get("cache_write_1h_price_per_1M"),
        thinking=entry.get("thinking"),
    )


def _load_model_info() -> dict[str, ModelInfo]:
    """Load ``MODEL_INFO`` from JSON, applying cache-pricing defaults."""
    path = _ensure_user_model_info_path()
    raw = json.loads(path.read_text())
    return {name: _build_model_info_entry(entry) for name, entry in raw.items()}


_OPENAI_PREFIXES = ("gpt", "text-embedding", "o1", "o3", "o4", "codex", "computer-use")
_TOGETHER_PREFIXES = (
    "meta-llama/",
    "Qwen/",
    "mistralai/",
    "deepseek-ai/",
    "deepcogito/",
    "google/gemma",
    "moonshotai/",
    "nvidia/",
    "zai-org/",
    "openai/gpt-oss",
    "arcee-ai/",
    "essentialai/",
    "BAAI/",
    "intfloat/",
)


def _openai_compatible(
    model_name: str,
    base_url: str,
    api_key: str,
    model_config: dict[str, Any] | None,
    token_callback: TokenCallback | None,
    thinking_callback: ThinkingCallback | None = None,
) -> Model:
    from kiss.core.models import OpenAICompatibleModel

    if OpenAICompatibleModel is None:  # pragma: no cover – openai always installed
        raise KISSError("OpenAI SDK not installed. Install 'openai' to use this model.")
    return OpenAICompatibleModel(  # type: ignore[no-any-return]
        model_name=model_name,
        base_url=base_url,
        api_key=api_key,
        model_config=model_config,
        token_callback=token_callback,
        thinking_callback=thinking_callback,
    )


MODEL_INFO: dict[str, ModelInfo] = _load_model_info()

_ANTHROPIC_CACHE_PREFIXES = (
    "claude-",
    "openrouter/anthropic/",
    "openrouter/~anthropic/",
)
_OPENAI_OPENROUTER_PREFIXES = ("openrouter/openai/", "openrouter/~openai/")
_GOOGLE_OPENROUTER_PREFIXES = ("openrouter/google/", "openrouter/~google/")
_QUARTER_CACHE_OPENROUTER_PREFIXES = (
    "openrouter/moonshotai/",
    "openrouter/~moonshotai/",
    "openrouter/x-ai/",
)


def _openai_cache_read_multiplier(bare: str) -> float:
    """Return the cached-input price multiplier for an OpenAI model.

    OpenAI bills prompt-cache reads at a per-model fraction of the base input
    price (and never charges for cache writes). The multipliers below match
    OpenAI's published pricing: GPT-5.x is 0.10x when a cached-input price is
    published, GPT-4.1 and o3/o4-mini are 0.25x, while GPT-4o, GPT-4,
    GPT-3.5, o1 and o3-mini are 0.50x. GPT-5 ``pro`` variants currently show
    no cached-input discount, so cached tokens are charged at the full input
    price rather than silently undercounted.

    Args:
        bare: An OpenAI model name without any provider prefix (e.g.
            ``gpt-5.4``, ``o3-mini``, ``gpt-4o``).

    Returns:
        The fraction of the base input price charged for cached read tokens.
    """
    if "-pro" in bare:
        return 1.0
    if bare in ("gpt-latest", "gpt-mini-latest"):
        return 0.10  # OpenRouter aliases for the current GPT-5.x models
    if bare.startswith("gpt-5") or "chat-latest" in bare:
        return 0.10
    if bare.startswith("gpt-image-1-mini"):
        return 0.10
    if bare.startswith("gpt-image"):
        return 0.25
    if bare.startswith("gpt-4.1"):
        return 0.25
    if bare.startswith(("o1", "o3-mini")):
        return 0.50
    if bare.startswith(("o3", "o4")):
        return 0.25
    return 0.50  # gpt-4o, gpt-4, gpt-3.5-turbo, computer-use, ...


def _openai_bare_name(name: str) -> str | None:
    """Return the bare OpenAI model name for cache pricing, or ``None``.

    Recognizes both directly-routed OpenAI models (``gpt-*``, ``o1``/``o3``/
    ``o4``, ``computer-use``) and OpenRouter passthrough OpenAI models
    (``openrouter/openai/*`` and ``openrouter/~openai/*``). Embeddings, the
    open-weight ``gpt-oss`` models, and subscription ``codex/`` models are
    excluded (they have no per-token cache discount in this table).

    Args:
        name: The MODEL_INFO key.

    Returns:
        The OpenAI model name without provider prefix, or ``None`` if ``name``
        is not an OpenAI cache-eligible model.
    """
    if name.startswith(_OPENAI_OPENROUTER_PREFIXES):
        bare = name.split("/", 2)[2]
        return None if bare.startswith("gpt-oss") else bare
    if name.startswith(_OPENAI_PREFIXES) and not name.startswith(
        ("text-embedding", "openai/", "codex/")
    ):
        return name
    return None


def _apply_cache_pricing(name: str, info: ModelInfo) -> None:
    """Populate ``info``'s cache read/write prices from provider pricing rules.

    Cache-read tokens are billed at a fraction of the base input price and
    cache-write tokens at a (possibly different) multiple, matching each
    provider's published prompt-caching pricing. Providers without a
    documented cache discount are left as ``None`` so ``calculate_cost`` falls
    back to the full input price (a conservative over-estimate).

    Args:
        name: The MODEL_INFO key.
        info: The ModelInfo to mutate in place.
    """
    if info.cache_read_price_per_1M is not None:
        return
    if not info.is_generation_supported:
        return
    inp = info.input_price_per_1M
    if name.startswith(_ANTHROPIC_CACHE_PREFIXES):
        info.cache_read_price_per_1M = inp * 0.1
        info.cache_write_price_per_1M = inp * 1.25
        info.cache_write_1h_price_per_1M = inp * 2.0
        return
    bare = _openai_bare_name(name)
    if bare is not None:
        info.cache_read_price_per_1M = inp * _openai_cache_read_multiplier(bare)
        info.cache_write_price_per_1M = 0.0  # OpenAI does not charge for cache writes
        return
    if name.startswith("gemini-"):
        info.cache_read_price_per_1M = inp * 0.1  # Gemini context cache read
        info.cache_write_price_per_1M = 0.0
        return
    if name.startswith(_GOOGLE_OPENROUTER_PREFIXES):
        info.cache_read_price_per_1M = inp * 0.25  # OpenRouter Gemini implicit cache
        info.cache_write_price_per_1M = 0.0
        return
    if name.startswith("openrouter/deepseek/"):
        multiplier = 0.02
        if name.startswith("openrouter/deepseek/deepseek-v4-pro"):
            multiplier = 0.003625 / 0.435
        info.cache_read_price_per_1M = inp * multiplier
        info.cache_write_price_per_1M = inp  # DeepSeek cache write = input price
        return
    if name.startswith("openrouter/qwen/"):
        info.cache_read_price_per_1M = inp * 0.2
        info.cache_write_price_per_1M = inp * 1.25
        return
    if name.startswith(_QUARTER_CACHE_OPENROUTER_PREFIXES):
        info.cache_read_price_per_1M = inp * 0.25  # Moonshot / Grok cache read
        info.cache_write_price_per_1M = 0.0
        return
    # No documented cache discount: leave None (full input price fallback).


for _name, _info in MODEL_INFO.items():
    _apply_cache_pricing(_name, _info)

FLAKY_MODELS: dict[str, str] = {
    "openrouter/baidu/ernie-4.5-21b-a3b": "Ignores function calling tools",
}


def is_model_flaky(model_name: str) -> bool:
    """Check if a model is known to be flaky.

    Args:
        model_name: The name of the model to check.

    Returns:
        bool: True if the model is known to have reliability issues.
    """
    return model_name in FLAKY_MODELS


def get_flaky_reason(model_name: str) -> str:
    """Get the reason why a model is flaky.

    Args:
        model_name: The name of the model to check.

    Returns:
        str: The reason for flakiness, or empty string if not flaky.
    """
    return FLAKY_MODELS.get(model_name, "")


def _strip_provider_prefix(model_name: str) -> str:
    """Strip harbor-style provider prefixes that duplicate KISS's own routing.

    Harbor (and other frameworks) pass model names as ``provider/model``
    (e.g. ``openai/gpt-5.4``, ``anthropic/claude-opus-4-6``,
    ``google/gemini-2.5-pro``).  KISS already routes by the model name
    itself (``gpt-*`` → OpenAI, ``claude-*`` → Anthropic, etc.), so the
    provider prefix is redundant and must be stripped.

    Prefixes that KISS uses for its own routing (``openrouter/``,
    ``openai/gpt-oss``, ``meta-llama/``, etc.) are NOT stripped — they
    are already handled by the ``model()`` dispatch chain.

    Args:
        model_name: Model name, possibly with a ``provider/`` prefix.

    Returns:
        The model name with redundant provider prefix stripped.
    """
    strip_prefixes = ("openai/", "anthropic/", "google/")
    for prefix in strip_prefixes:
        if model_name.startswith(prefix):
            bare = model_name[len(prefix):]
            if bare.startswith(_OPENAI_PREFIXES) and not bare.startswith("gpt-oss"):
                return bare
            if bare.startswith("claude-"):
                return bare
            if bare.startswith("gemini-"):
                return bare
    return model_name


def model(
    model_name: str,
    model_config: dict[str, Any] | None = None,
    token_callback: TokenCallback | None = None,
    thinking_callback: ThinkingCallback | None = None,
) -> Model:
    """Get a model instance based on model name prefix.

    Args:
        model_name: The name of the model (with provider prefix if applicable).
            Accepts harbor-style ``provider/model`` names (e.g.
            ``openai/gpt-5.4``, ``anthropic/claude-opus-4-6``) — the
            redundant provider prefix is stripped automatically.
        model_config: Optional dictionary of model configuration parameters.
            If it contains "base_url", routing is bypassed and an OpenAICompatibleModel
            is built with that base_url and optional "api_key".
        token_callback: Optional callback invoked with each streamed text token.
        thinking_callback: Optional callback invoked with ``True`` when a
            thinking block starts and ``False`` when it ends.

    Returns:
        Model: An appropriate Model instance for the specified model.

    Raises:
        KISSError: If the model name is not recognized.
    """
    model_name = _strip_provider_prefix(model_name)
    if model_config and "base_url" in model_config:
        base_url = model_config["base_url"]
        api_key = model_config.get("api_key", "")
        filtered = {k: v for k, v in model_config.items() if k not in ("base_url", "api_key")}
        return _openai_compatible(
            model_name,
            base_url,
            api_key,
            filtered or None,
            token_callback,
            thinking_callback,
        )
    keys = config_module.DEFAULT_CONFIG
    if model_name.startswith("openrouter/"):
        return _openai_compatible(
            model_name,
            "https://openrouter.ai/api/v1",
            keys.OPENROUTER_API_KEY,
            model_config,
            token_callback,
            thinking_callback,
        )
    if model_name == "text-embedding-004":
        from kiss.core.models import GeminiModel

        if GeminiModel is None:  # pragma: no cover – google-genai always installed
            raise KISSError(
                "Google GenAI SDK not installed. Install 'google-genai' to use Gemini models."
            )
        return GeminiModel(  # type: ignore[no-any-return]
            model_name=model_name,
            api_key=keys.GEMINI_API_KEY,
            model_config=model_config,
            token_callback=token_callback,
            thinking_callback=thinking_callback,
        )
    if model_name.startswith("codex/"):
        from kiss.core.models import CodexModel

        if CodexModel is None:  # pragma: no cover – always available
            raise KISSError("CodexModel could not be loaded.")
        return CodexModel(  # type: ignore[no-any-return]
            model_name=model_name,
            model_config=model_config,
            token_callback=token_callback,
            thinking_callback=thinking_callback,
        )
    if model_name.startswith(_OPENAI_PREFIXES) and not model_name.startswith("openai/gpt-oss"):
        return _openai_compatible(
            model_name,
            "https://api.openai.com/v1",
            keys.OPENAI_API_KEY,
            model_config,
            token_callback,
            thinking_callback,
        )
    if model_name.startswith(_TOGETHER_PREFIXES):
        return _openai_compatible(
            model_name,
            "https://api.together.xyz/v1",
            keys.TOGETHER_API_KEY,
            model_config,
            token_callback,
            thinking_callback,
        )
    if model_name.startswith("claude-"):
        from kiss.core.models import AnthropicModel

        if AnthropicModel is None:  # pragma: no cover – anthropic always installed
            raise KISSError(
                "Anthropic SDK not installed. Install 'anthropic' to use Claude models."
            )
        return AnthropicModel(  # type: ignore[no-any-return]
            model_name=model_name,
            api_key=keys.ANTHROPIC_API_KEY,
            model_config=model_config,
            token_callback=token_callback,
            thinking_callback=thinking_callback,
        )
    if model_name.startswith("gemini-"):
        from kiss.core.models import GeminiModel

        if GeminiModel is None:  # pragma: no cover – google-genai always installed
            raise KISSError(
                "Google GenAI SDK not installed. Install 'google-genai' to use Gemini models."
            )
        return GeminiModel(  # type: ignore[no-any-return]
            model_name=model_name,
            api_key=keys.GEMINI_API_KEY,
            model_config=model_config,
            token_callback=token_callback,
            thinking_callback=thinking_callback,
        )
    if model_name.startswith("glm-"):
        return _openai_compatible(
            model_name,
            "https://api.z.ai/api/paas/v4",
            keys.ZAI_API_KEY,
            model_config,
            token_callback,
            thinking_callback,
        )
    if model_name.startswith("kimi-") or model_name.startswith("moonshot-"):
        return _openai_compatible(
            model_name,
            "https://api.moonshot.ai/v1",
            keys.MOONSHOT_API_KEY,
            model_config,
            token_callback,
            thinking_callback,
        )
    if model_name.startswith("cc/"):
        from kiss.core.models import ClaudeCodeModel

        if ClaudeCodeModel is None:  # pragma: no cover – always available
            raise KISSError("ClaudeCodeModel could not be loaded.")
        return ClaudeCodeModel(  # type: ignore[no-any-return]
            model_name=model_name,
            model_config=model_config,
            token_callback=token_callback,
            thinking_callback=thinking_callback,
        )
    raise KISSError(f"Unknown model name: {model_name}")


def get_available_models() -> list[str]:
    """Return model names for which an API key is configured and generation is supported.

    Returns:
        list[str]: Sorted list of model name strings that have a configured API key
            and support text generation.
    """
    keys = config_module.DEFAULT_CONFIG
    prefix_to_key = {
        "openrouter/": keys.OPENROUTER_API_KEY,
        "claude-": keys.ANTHROPIC_API_KEY,
        "gemini-": keys.GEMINI_API_KEY,
        "glm-": keys.ZAI_API_KEY,
        "kimi-": keys.MOONSHOT_API_KEY,
        "moonshot-": keys.MOONSHOT_API_KEY,
    }
    import shutil

    from kiss.core.models.codex_model import find_codex_executable

    has_claude_cli = shutil.which("claude") is not None
    has_codex_cli = find_codex_executable() is not None
    result = []
    for name, info in MODEL_INFO.items():
        if not info.is_generation_supported:
            continue
        if name.startswith("cc/"):
            if has_claude_cli:
                result.append(name)
            continue
        if name.startswith("codex/"):
            if has_codex_cli:
                result.append(name)
            continue
        api_key = ""
        for prefix, key in prefix_to_key.items():
            if name.startswith(prefix):
                api_key = key
                break
        if not api_key:
            if name == "text-embedding-004":  # pragma: no cover – embedding model filtered above
                api_key = keys.GEMINI_API_KEY
            elif name.startswith(_OPENAI_PREFIXES) and not name.startswith("openai/gpt-oss"):
                api_key = keys.OPENAI_API_KEY
            elif name.startswith(_TOGETHER_PREFIXES):
                api_key = keys.TOGETHER_API_KEY
        if api_key:
            result.append(name)
    return sorted(result)


def get_model_provider(model_name: str) -> str:
    """Return the human-readable provider label that routes *model_name*.

    The mapping mirrors the dispatch order in :func:`model`: subscription
    CLIs first (``cc/`` → Claude Code, ``codex/`` → Codex), then the
    prefix-routed HTTP providers (``openrouter/``, ``claude-``,
    ``gemini-``, ``glm-`` for Z.AI, ``kimi-``/``moonshot-`` for Moonshot),
    then the OpenAI and Together prefix sets.

    Args:
        model_name: A ``MODEL_INFO`` key.

    Returns:
        str: The provider label (e.g. ``"OpenAI"``, ``"Anthropic"``,
            ``"OpenRouter"``), or ``"Unknown"`` if no route matches.
    """
    if model_name.startswith("cc/"):
        return "Claude Code CLI"
    if model_name.startswith("codex/"):
        return "Codex CLI"
    if model_name.startswith("openrouter/"):
        return "OpenRouter"
    if model_name.startswith("claude-"):
        return "Anthropic"
    if model_name.startswith("gemini-") or model_name == "text-embedding-004":
        return "Gemini"
    if model_name.startswith("glm-"):
        return "Z.AI"
    if model_name.startswith("kimi-") or model_name.startswith("moonshot-"):
        return "Moonshot"
    if model_name.startswith(_OPENAI_PREFIXES) and not model_name.startswith("openai/gpt-oss"):
        return "OpenAI"
    if model_name.startswith(_TOGETHER_PREFIXES):
        return "Together"
    return "Unknown"


def get_generation_model_listing() -> list[tuple[str, str, bool]]:
    """List every generation-capable model with provider and key status.

    Walks ``MODEL_INFO`` (sorted by model name), skipping non-generation
    (embedding/image-only) entries, and reports for each model the
    provider that routes it and whether the credential needed to run it
    is currently present — an API key for HTTP providers, or the local
    executable for the ``Claude Code`` / ``Codex`` subscription CLIs.

    Returns:
        list[tuple[str, str, bool]]: ``(model_name, provider, configured)``
            triples sorted alphabetically by model name.
    """
    import shutil

    from kiss.core.models.codex_model import find_codex_executable

    keys = config_module.DEFAULT_CONFIG
    provider_configured = {
        "Anthropic": bool(keys.ANTHROPIC_API_KEY),
        "OpenAI": bool(keys.OPENAI_API_KEY),
        "Gemini": bool(keys.GEMINI_API_KEY),
        "OpenRouter": bool(keys.OPENROUTER_API_KEY),
        "Together": bool(keys.TOGETHER_API_KEY),
        "Z.AI": bool(keys.ZAI_API_KEY),
        "Moonshot": bool(keys.MOONSHOT_API_KEY),
        "Claude Code CLI": shutil.which("claude") is not None,
        "Codex CLI": find_codex_executable() is not None,
        "Unknown": False,
    }
    listing: list[tuple[str, str, bool]] = []
    for name, info in sorted(MODEL_INFO.items()):
        if not info.is_generation_supported:
            continue
        provider = get_model_provider(name)
        listing.append((name, provider, provider_configured.get(provider, False)))
    return listing


def get_completion_model_names() -> list[str]:
    """Return model names to offer for input fast-completion, best first.

    Prefers models whose provider API key is configured (so every
    suggestion is actually runnable), and falls back to every
    generation-capable model in ``MODEL_INFO`` when no provider key is set
    (e.g. completing in a fresh checkout) so the picker is never empty.

    Returns:
        list[str]: A sorted list of generation-capable model names.
    """
    available = get_available_models()
    if available:
        return available
    return sorted(
        name for name, info in MODEL_INFO.items() if info.is_generation_supported
    )


def rank_model_suggestions(query: str, names: list[str] | None = None) -> list[str]:
    """Rank model names for fast-completion of *query*.

    Case-insensitive prefix matches come first, then case-insensitive
    substring matches; each group is sorted alphabetically. An empty query
    returns all candidate names unchanged (already sorted).

    Args:
        query: The partial model name typed by the user.
        names: Candidate model names. Defaults to
            :func:`get_completion_model_names` when ``None``.

    Returns:
        list[str]: The matching model names, best first.
    """
    if names is None:
        names = get_completion_model_names()
    q = query.strip().lower()
    if not q:
        return list(names)
    prefix = sorted(n for n in names if n.lower().startswith(q))
    substring = sorted(
        n for n in names if q in n.lower() and not n.lower().startswith(q)
    )
    return prefix + substring


def get_fast_model() -> str:
    """Return a cheap/fast model based on which API keys are available.

    Priority: Anthropic → OpenAI → Gemini → OpenRouter → Together → Claude Code CLI.

    Returns:
        A fast model name for the first available provider.
    """
    import shutil

    from kiss.core.models.codex_model import find_codex_executable

    keys = config_module.DEFAULT_CONFIG
    if keys.ANTHROPIC_API_KEY:
        return "claude-haiku-4-5"
    if keys.OPENAI_API_KEY:
        return "gpt-4o"
    if keys.GEMINI_API_KEY:
        return "gemini-2.0-flash"
    if keys.OPENROUTER_API_KEY:
        return "openrouter/anthropic/claude-haiku-4.5"
    if keys.TOGETHER_API_KEY:
        return "deepseek-ai/DeepSeek-R1-0528"
    if shutil.which("claude") is not None:
        return "cc/haiku"
    if find_codex_executable() is not None:
        return "codex/default"
    return "No model"


def get_default_model() -> str:
    """Return the best default model based on which API keys are configured.

    Priority order: Anthropic > OpenAI > Gemini > OpenRouter > Together AI > Claude Code CLI.
    Falls back to ``"No model"`` if no keys are set.
    """
    import shutil

    from kiss.core.models.codex_model import find_codex_executable

    keys = config_module.DEFAULT_CONFIG
    if keys.ANTHROPIC_API_KEY:
        return "claude-opus-4-7"
    if keys.OPENAI_API_KEY:
        return "gpt-5.5"
    if keys.GEMINI_API_KEY:
        return "gemini-3.1-pro-preview"
    if keys.OPENROUTER_API_KEY:
        return "openrouter/anthropic/claude-opus-4.7"
    if keys.TOGETHER_API_KEY:
        return "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"
    if shutil.which("claude") is not None:
        return "cc/opus"
    if find_codex_executable() is not None:
        return "codex/default"
    return "No model"


def get_most_expensive_model(fc_only: bool = True) -> str:
    best_name, best_price = "", -1.0
    for name in get_available_models():
        info = MODEL_INFO[name]
        if fc_only and not info.is_function_calling_supported:
            continue
        price = info.input_price_per_1M + info.output_price_per_1M
        if price > best_price:
            best_price = price
            best_name = name
    return best_name


def _openai_long_context_prices(model_name: str) -> tuple[int, float, float, float] | None:
    """Return ``(threshold, input, output, cached)`` long-context prices if known."""
    bare = _strip_provider_prefix(model_name)
    if bare.startswith(_OPENAI_OPENROUTER_PREFIXES):
        bare = bare.split("/", 2)[2]
    if bare.startswith("gpt-5.5") and "-pro" not in bare:
        return 200_000, 10.00, 45.00, 1.00
    if (
        bare.startswith("gpt-5.4")
        and "-pro" not in bare
        and "-mini" not in bare
        and "-nano" not in bare
    ):
        return 200_000, 5.00, 22.50, 0.50
    return None


def _gemini_long_context_prices(model_name: str) -> tuple[int, float, float, float] | None:
    """Return ``(threshold, input, output, cached)`` long-context prices if known."""
    bare = _strip_provider_prefix(model_name)
    if bare.startswith(_GOOGLE_OPENROUTER_PREFIXES):
        bare = bare.split("/", 2)[2]
    if bare.startswith("gemini-2.5-pro"):
        return 200_000, 2.50, 15.00, 0.25
    return None


def calculate_cost(
    model_name: str,
    num_input_tokens: int,
    num_output_tokens: int,
    num_cache_read_tokens: int = 0,
    num_cache_write_tokens: int = 0,
    num_cache_write_1h_tokens: int = 0,
) -> float:
    """Calculates the cost in USD for the given token counts.

    Args:
        model_name: Name of the model (with or without provider prefix).
        num_input_tokens: Number of non-cached input tokens.
        num_output_tokens: Number of output tokens.
        num_cache_read_tokens: Number of tokens read from cache.
        num_cache_write_tokens: Number of standard/5-minute cache-write tokens.
        num_cache_write_1h_tokens: Number of one-hour Anthropic cache-write tokens.

    Returns:
        float: Cost in USD.

    Raises:
        KISSError: If positive usage is reported for a model without pricing.
    """
    info = MODEL_INFO.get(model_name) or MODEL_INFO.get(_strip_provider_prefix(model_name))
    total_tokens = (
        num_input_tokens
        + num_output_tokens
        + num_cache_read_tokens
        + num_cache_write_tokens
        + num_cache_write_1h_tokens
    )
    if info is None:
        if total_tokens > 0:
            raise KISSError(
                f"Cannot calculate budget for unknown model '{model_name}'. "
                "Add the model to MODEL_INFO or configure explicit pricing."
            )
        return 0.0
    cr_price = (
        info.cache_read_price_per_1M
        if info.cache_read_price_per_1M is not None
        else info.input_price_per_1M
    )
    cw_price = (
        info.cache_write_price_per_1M
        if info.cache_write_price_per_1M is not None
        else info.input_price_per_1M
    )
    cw1h_price = (
        info.cache_write_1h_price_per_1M
        if info.cache_write_1h_price_per_1M is not None
        else cw_price
    )
    input_price = info.input_price_per_1M
    output_price = info.output_price_per_1M
    long_prices = _openai_long_context_prices(model_name) or _gemini_long_context_prices(
        model_name
    )
    if long_prices is not None and total_tokens > long_prices[0]:
        _, input_price, output_price, cr_price = long_prices
    input_cost = num_input_tokens * input_price
    output_cost = num_output_tokens * output_price
    cache_read_cost = num_cache_read_tokens * cr_price
    return (
        input_cost
        + output_cost
        + cache_read_cost
        + num_cache_write_tokens * cw_price
        + num_cache_write_1h_tokens * cw1h_price
    ) / 1_000_000


def get_max_context_length(model_name: str) -> int:
    """Returns the maximum context length supported by the model.

    Args:
        model_name: Name of the model (with or without provider prefix).
    Returns:
        int: Maximum context length in tokens.
    """
    stripped = _strip_provider_prefix(model_name)
    if model_name in MODEL_INFO:
        return MODEL_INFO[model_name].context_length
    if stripped in MODEL_INFO:
        return MODEL_INFO[stripped].context_length
    raise KeyError(f"Model '{model_name}' not found in MODEL_INFO")
