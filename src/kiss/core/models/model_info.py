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
from dataclasses import dataclass
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
        fallback: str | None = None,
        extended_thinking: bool | None = None,
        adaptive_thinking: bool | None = None,
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
        #: Optional name of a fallback model to switch to when the primary
        #: model returns a non-retryable error (e.g. Anthropic responding
        #: with ``"Claude Fable 5 is not available. Please use Opus 4.8"``
        #: or ``"credit balance is too low"``).  ``None`` disables the
        #: fallback behavior for this model.
        self.fallback = fallback
        #: Tri-state override for whether this Anthropic model expects the
        #: ``thinking`` request parameter and the
        #: ``anthropic-beta: interleaved-thinking-2025-05-14`` header.
        #:
        #: * ``True``  — force extended thinking on (used for models like
        #:   ``claude-fable-5`` / ``claude-sonnet-5`` whose family names
        #:   are not covered by the legacy prefix heuristic in
        #:   :func:`kiss.core.models.anthropic_model._build_create_kwargs`).
        #: * ``False`` — force extended thinking off.
        #: * ``None``  — no explicit opinion; the adapter falls back to
        #:   its prefix-based default.
        self.extended_thinking = extended_thinking
        #: Tri-state override for ``thinking.type``.  ``True`` requests
        #: ``{"type": "adaptive", "display": "summarized"}`` (required by
        #: Claude 4.6+ / fable / sonnet-5 which reject ``"enabled"``;
        #: ``display`` must be ``"summarized"`` because it defaults to
        #: ``"omitted"``, which returns empty signature-only thinking);
        #: ``False`` forces ``{"type": "enabled", "budget_tokens": ...}``;
        #: ``None`` falls back to
        #: :func:`kiss.core.models.anthropic_model._uses_adaptive_thinking`.
        self.adaptive_thinking = adaptive_thinking


PACKAGE_MODEL_INFO_PATH = Path(__file__).parent / "MODEL_INFO.json"

#: Optional user-curated model overrides / extensions file.
#:
#: The bundled ``MODEL_INFO.json`` next to this module is the read-only
#: source of truth for shipped models — it is never copied into
#: ``~/.kiss/``.  Users add personal models or override bundled pricing
#: by editing ``~/.kiss/MY_MODELS.json``, which the loader merges on top
#: of the bundled table at import time.
USER_MY_MODELS_PATH = Path.home() / ".kiss" / "MY_MODELS.json"

#: Default JSON content seeded into ``~/.kiss/MY_MODELS.json`` on first
#: read.  Contains a short ``_documentation`` block and a single
#: commented-out example entry (key starts with ``_`` so it is ignored
#: by :func:`_read_my_models` until the user removes the prefix).
#:
#: Keeping the seed inert means a fresh install sees exactly the bundled
#: model table — no spurious "example" model appears in the picker
#: until the user opts in by renaming ``_example/my-org/my-custom-model``.
MY_MODELS_DEFAULT_CONTENT = json.dumps(
    {
        "_documentation": [
            "MY_MODELS.json — your personal model registry.",
            "",
            "Entries here OVERRIDE matching keys in the bundled MODEL_INFO.json,",
            "and entries whose key does not appear in the bundled file are ADDED.",
            "Any top-level key starting with '_' is treated as a comment and is",
            "skipped by the loader (use it for documentation or to keep an",
            "example entry inert).",
            "",
            "Per-model schema:",
            "  context_length         (int)   max input+output tokens",
            "  input_price_per_1M     (float) USD per 1M input tokens",
            "  output_price_per_1M    (float) USD per 1M output tokens",
            "  fc       (bool, default true)  function-calling supported",
            "  emb      (bool, default false) embedding model",
            "  gen      (bool, default true)  text generation supported",
            "  thinking (str,  optional)      reasoning_effort cap, e.g. 'xhigh'",
            "",
            "To activate the example below, remove the leading '_example/'",
            "from its key and adjust the values.",
        ],
        "_example/my-org/my-custom-model": {
            "context_length": 128000,
            "input_price_per_1M": 0.0,
            "output_price_per_1M": 0.0,
            "fc": True,
            "emb": False,
            "gen": True,
        },
    },
    indent=2,
) + "\n"


def _seed_my_models_file() -> None:
    """Create ``~/.kiss/MY_MODELS.json`` from the inline default if absent.

    Never overwrites an existing file — user edits survive every
    restart.  Silently swallows :class:`OSError` so a read-only HOME or
    missing parent does not break ``MODEL_INFO`` import.
    """
    try:
        if USER_MY_MODELS_PATH.exists():
            return
        USER_MY_MODELS_PATH.parent.mkdir(parents=True, exist_ok=True)
        USER_MY_MODELS_PATH.write_text(MY_MODELS_DEFAULT_CONTENT, encoding="utf-8")
    except OSError:
        pass


def _read_my_models() -> dict[str, dict[str, Any]]:
    """Return parsed model entries from ``~/.kiss/MY_MODELS.json``.

    Auto-seeds the file with :data:`MY_MODELS_DEFAULT_CONTENT` on first
    read.  Returns an empty dict when:

    * The file is missing AND cannot be seeded (read-only FS).
    * The file is unreadable or contains malformed JSON.
    * The top-level value is not a JSON object.

    Filters out any key starting with ``_`` (documentation / inert
    example entries) and any value that is not a JSON object, so
    documentation lists and stray scalars never reach the model table.
    """
    _seed_my_models_file()
    try:
        text = USER_MY_MODELS_PATH.read_text(encoding="utf-8")
    except OSError:
        return {}
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        return {}
    if not isinstance(raw, dict):
        return {}
    return {
        name: entry
        for name, entry in raw.items()
        if not name.startswith("_") and isinstance(entry, dict)
    }


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
    * ``extended_thinking`` — tri-state override for whether the
      Anthropic ``thinking`` request param (and the
      ``interleaved-thinking-2025-05-14`` beta header) is attached.
      ``True`` forces on, ``False`` forces off, ``None`` defers to the
      adapter's prefix heuristic.
    * ``adaptive_thinking`` — tri-state override for whether Anthropic
      thinking is requested with
      ``{"type": "adaptive", "display": "summarized"}`` (required by
      Claude 4.6+ / fable / sonnet-5 which reject ``"enabled"``; the
      explicit ``display: summarized`` is mandatory because the API
      default ``"omitted"`` returns empty signature-only thinking)
      instead of ``{"type": "enabled", "budget_tokens": ...}``.
      ``None`` defers to
      :func:`kiss.core.models.anthropic_model._uses_adaptive_thinking`.
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
        fallback=entry.get("fallback"),
        extended_thinking=entry.get("extended_thinking"),
        adaptive_thinking=entry.get("adaptive_thinking"),
    )


def _load_model_info() -> dict[str, ModelInfo]:
    """Load ``MODEL_INFO`` from JSON, applying cache-pricing defaults.

    The bundled :data:`PACKAGE_MODEL_INFO_PATH` is the source of truth
    for shipped models.  ``~/.kiss/MY_MODELS.json`` (auto-seeded on
    first read) is then merged on top: matching keys override the
    bundled entry, and brand-new keys are added.
    """
    raw = json.loads(PACKAGE_MODEL_INFO_PATH.read_text(encoding="utf-8"))
    for name, entry in _read_my_models().items():
        raw[name] = entry
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


@dataclass(frozen=True)
class OpenAICompatibleProvider:
    """A factory-routed OpenAI-compatible Chat Completions vendor endpoint.

    This registry entry is the single source of truth for everything the
    framework needs to know about a vendor: how model names route to it,
    which credentials it uses, and — critically — its verified protocol
    capabilities, so transport decisions (e.g. whether ``tools`` +
    ``reasoning_effort`` survive on the same Chat Completions request) are
    config-driven instead of hardcoded host allowlists scattered around the
    code base.

    Attributes:
        name: Short unique vendor id (e.g. ``"openrouter"``).
        host: Unique host substring used to look up capabilities from a
            model's ``base_url`` (substring match, so tests can point at a
            local capture server whose URL embeds the host as a path
            segment).
        base_url: The vendor's OpenAI-compatible API root used by the
            ``model()`` factory.
        prefixes: Model-name prefixes that route to this vendor.
        excludes: Model-name prefixes that must NOT route here even though
            they match ``prefixes`` (they are handled by later, non-OpenAI
            branches of the factory).
        api_key_name: Attribute name on ``config.DEFAULT_CONFIG`` holding
            the vendor's API key.
        tools_accept_reasoning_effort: Whether the vendor's Chat Completions
            endpoint accepts ``tools`` + ``reasoning_effort`` on the same
            request. ``True`` = verified live to accept (effort is kept),
            ``False`` = verified live to reject (effort is stripped from
            tool-bearing requests), ``None`` = unverified — the transport
            keeps the effort optimistically and learns the verdict from the
            vendor's actual response at runtime (adaptive probe).
        delegate_tools_to_responses: Whether tool-bearing requests carrying
            ``reasoning_effort`` should be transported via the vendor's
            ``/v1/responses`` endpoint instead of Chat Completions.
    """

    name: str
    host: str
    base_url: str
    prefixes: tuple[str, ...]
    excludes: tuple[str, ...]
    api_key_name: str
    tools_accept_reasoning_effort: bool | None
    delegate_tools_to_responses: bool


# Single source of truth for OpenAI-compatible vendor endpoints. Adding a
# new vendor here (with its declared ``tools_accept_reasoning_effort``
# capability) is ALL that is required — routing, capability handling and
# SorcarAgent's endpoint carry-over logic all derive from this table. A
# regression test (test_reasoning_effort_capability_registry.py) trips when
# an entry is added so the capability declaration is a conscious decision.
OPENAI_COMPATIBLE_PROVIDERS: tuple[OpenAICompatibleProvider, ...] = (
    OpenAICompatibleProvider(
        name="openrouter",
        host="openrouter.ai",
        base_url="https://openrouter.ai/api/v1",
        prefixes=("openrouter/",),
        excludes=(),
        api_key_name="OPENROUTER_API_KEY",
        # Verified live: 200 with a real tool call for every effort level
        # including "xhigh"; OpenRouter translates the effort per provider.
        tools_accept_reasoning_effort=True,
        delegate_tools_to_responses=False,
    ),
    OpenAICompatibleProvider(
        name="openai",
        host="api.openai.com",
        base_url="https://api.openai.com/v1",
        prefixes=_OPENAI_PREFIXES,
        # gpt-oss models are served by Together; text-embedding-004 is a
        # Gemini embedding model; codex/ is the Codex CLI transport. All
        # three match _OPENAI_PREFIXES textually and are handled by later
        # factory branches.
        excludes=("openai/gpt-oss", "text-embedding-004", "codex/"),
        api_key_name="OPENAI_API_KEY",
        # Verified live: /v1/chat/completions rejects tools +
        # reasoning_effort for GPT-5.x / o-series reasoning models
        # ("please use /v1/responses instead") — hence the delegation.
        tools_accept_reasoning_effort=False,
        delegate_tools_to_responses=True,
    ),
    OpenAICompatibleProvider(
        name="together",
        host="api.together.xyz",
        base_url="https://api.together.xyz/v1",
        prefixes=_TOGETHER_PREFIXES,
        excludes=(),
        api_key_name="TOGETHER_API_KEY",
        # Verified live: 200 with a real tool call for low/medium/high
        # (non-reasoning models ignore the effort harmlessly; "xhigh" is
        # rejected but no Together catalog entry defaults to xhigh).
        # Together serverless does NOT implement /v1/responses.
        tools_accept_reasoning_effort=True,
        delegate_tools_to_responses=False,
    ),
    OpenAICompatibleProvider(
        name="zai",
        host="api.z.ai",
        base_url="https://api.z.ai/api/paas/v4",
        prefixes=("glm-",),
        excludes=(),
        api_key_name="ZAI_API_KEY",
        # Unverified (no live probe possible without credentials): the
        # transport keeps the effort optimistically and adapts at runtime.
        tools_accept_reasoning_effort=None,
        delegate_tools_to_responses=False,
    ),
    OpenAICompatibleProvider(
        name="moonshot",
        host="api.moonshot.ai",
        base_url="https://api.moonshot.ai/v1",
        prefixes=("kimi-", "moonshot-"),
        excludes=(),
        api_key_name="MOONSHOT_API_KEY",
        # Unverified (no live probe possible without credentials): the
        # transport keeps the effort optimistically and adapts at runtime.
        tools_accept_reasoning_effort=None,
        delegate_tools_to_responses=False,
    ),
)


def openai_compatible_provider_for_base_url(
    base_url: str,
) -> OpenAICompatibleProvider | None:
    """Return the registered vendor whose host appears in *base_url*.

    Substring matching (rather than exact URL equality) lets wire tests
    point a model at a local capture server whose URL embeds the vendor
    host as a path segment, and tolerates trailing slashes or versioned
    path variants.

    Args:
        base_url: The model's API root URL.

    Returns:
        The matching :class:`OpenAICompatibleProvider`, or None when the
        endpoint is unknown (custom gateway).
    """
    for provider in OPENAI_COMPATIBLE_PROVIDERS:
        if provider.host in base_url:
            return provider
    return None


def _match_openai_compatible_provider(
    model_name: str,
) -> OpenAICompatibleProvider | None:
    """Return the registered vendor that *model_name* routes to, if any.

    Args:
        model_name: Model name after provider-prefix stripping.

    Returns:
        The matching :class:`OpenAICompatibleProvider`, or None when the
        name is handled by a non-OpenAI-compatible factory branch.
    """
    for provider in OPENAI_COMPATIBLE_PROVIDERS:
        if model_name.startswith(provider.prefixes) and not (
            provider.excludes and model_name.startswith(provider.excludes)
        ):
            return provider
    return None


def _load_model_class(class_name: str, error_message: str) -> Any:
    """Return a lazily-imported model class from :mod:`kiss.core.models`.

    Args:
        class_name: Class attribute name (e.g. ``"GeminiModel"``).
        error_message: KISSError message raised when the class failed to import.

    Returns:
        The model class.

    Raises:
        KISSError: If the class (or its SDK) could not be imported.
    """
    import kiss.core.models as models

    cls = getattr(models, class_name)
    if cls is None:  # pragma: no cover – all model SDKs are always installed
        raise KISSError(error_message)
    return cls


def _openai_compatible(
    model_name: str,
    base_url: str,
    api_key: str,
    model_config: dict[str, Any] | None,
    token_callback: TokenCallback | None,
    thinking_callback: ThinkingCallback | None = None,
) -> Model:
    cls = _load_model_class(
        "OpenAICompatibleModel",
        "OpenAI SDK not installed. Install 'openai' to use this model.",
    )
    return cls(  # type: ignore[no-any-return]
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

_XHIGH_SUFFIX = "-xhigh"


def _strip_xhigh_alias(bare: str) -> str:
    """Strip the synthetic ``-xhigh`` alias suffix from a model name.

    ``-xhigh`` is a KISS-internal alias suffix (see ``update_models.py``)
    that maps onto the same provider model id as its base entry. Provider
    pricing tables only mention the base names, so every pricing lookup
    must consult the base name. Returning the input unchanged when the
    suffix is absent keeps callers simple.

    Args:
        bare: A model name, possibly ending in ``-xhigh``.

    Returns:
        ``bare`` with a trailing ``-xhigh`` removed if present.
    """
    if bare.endswith(_XHIGH_SUFFIX):
        return bare[: -len(_XHIGH_SUFFIX)]
    return bare


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
        if bare.startswith("gpt-oss"):
            return None
        return _strip_xhigh_alias(bare)
    if name.startswith(_OPENAI_PREFIXES) and not name.startswith(
        ("text-embedding", "openai/", "codex/")
    ):
        return _strip_xhigh_alias(name)
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
            if bare.startswith(("claude-", "gemini-")) or (
                bare.startswith(_OPENAI_PREFIXES) and not bare.startswith("gpt-oss")
            ):
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
    provider = _match_openai_compatible_provider(model_name)
    if provider is not None:
        return _openai_compatible(
            model_name,
            provider.base_url,
            getattr(keys, provider.api_key_name),
            model_config,
            token_callback,
            thinking_callback,
        )
    if model_name.startswith("gemini-") or model_name == "text-embedding-004":
        cls = _load_model_class(
            "GeminiModel",
            "Google GenAI SDK not installed. Install 'google-genai' to use Gemini models.",
        )
        return cls(  # type: ignore[no-any-return]
            model_name=model_name,
            api_key=keys.GEMINI_API_KEY,
            model_config=model_config,
            token_callback=token_callback,
            thinking_callback=thinking_callback,
        )
    if model_name.startswith("claude-"):
        cls = _load_model_class(
            "AnthropicModel",
            "Anthropic SDK not installed. Install 'anthropic' to use Claude models.",
        )
        return cls(  # type: ignore[no-any-return]
            model_name=model_name,
            api_key=keys.ANTHROPIC_API_KEY,
            model_config=model_config,
            token_callback=token_callback,
            thinking_callback=thinking_callback,
        )
    if model_name.startswith("codex/") or model_name.startswith("cc/"):
        class_name = "CodexModel" if model_name.startswith("codex/") else "ClaudeCodeModel"
        cls = _load_model_class(class_name, f"{class_name} could not be loaded.")
        return cls(  # type: ignore[no-any-return]
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


def _model_for_first_configured_provider(choices: dict[str, str]) -> str:
    """Return the choice for the first provider with a configured credential.

    Providers are checked in priority order: Anthropic → OpenAI → Gemini →
    OpenRouter → Together → Claude Code CLI → Codex CLI.

    Args:
        choices: Mapping from ``config.DEFAULT_CONFIG`` API-key attribute
            names (plus ``"cc"`` / ``"codex"`` for the subscription CLIs) to
            the model name to return for that provider.

    Returns:
        The chosen model name, or ``"No model"`` when nothing is configured.
    """
    import shutil

    from kiss.core.models.codex_model import find_codex_executable

    keys = config_module.DEFAULT_CONFIG
    for key_name in (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "OPENROUTER_API_KEY",
        "TOGETHER_API_KEY",
    ):
        if getattr(keys, key_name):
            return choices[key_name]
    if shutil.which("claude") is not None:
        return choices["cc"]
    if find_codex_executable() is not None:
        return choices["codex"]
    return "No model"


def get_fast_model() -> str:
    """Return a cheap/fast model based on which API keys are available.

    Priority: Anthropic → OpenAI → Gemini → OpenRouter → Together → Claude Code CLI.

    Returns:
        A fast model name for the first available provider.
    """
    return _model_for_first_configured_provider(
        {
            "ANTHROPIC_API_KEY": "claude-sonnet-5",
            "OPENAI_API_KEY": "gpt-4o",
            "GEMINI_API_KEY": "gemini-2.0-flash",
            "OPENROUTER_API_KEY": "openrouter/anthropic/claude-haiku-4.5",
            "TOGETHER_API_KEY": "deepseek-ai/DeepSeek-R1-0528",
            "cc": "cc/haiku",
            "codex": "codex/default",
        }
    )


def get_default_model() -> str:
    """Return the best default model based on which API keys are configured.

    Priority order: Anthropic > OpenAI > Gemini > OpenRouter > Together AI > Claude Code CLI.
    Falls back to ``"No model"`` if no keys are set.
    """
    return _model_for_first_configured_provider(
        {
            "ANTHROPIC_API_KEY": "claude-fable-5",
            "OPENAI_API_KEY": "gpt-5.5-xhigh",
            "GEMINI_API_KEY": "gemini-3.1-pro-preview",
            "OPENROUTER_API_KEY": "openrouter/anthropic/claude-opus-4.7",
            "TOGETHER_API_KEY": "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
            "cc": "cc/opus",
            "codex": "codex/default",
        }
    )


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
    bare = _strip_xhigh_alias(bare)
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
    bare = _strip_xhigh_alias(bare)
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


def get_fallback_model(model_name: str) -> str | None:
    """Return the registered fallback model for *model_name*, or ``None``.

    Consulted by :meth:`kiss.core.kiss_agent.KISSAgent._try_switch_to_fallback`
    when a non-retryable provider error (model not available, credit
    balance too low, etc.) is raised.  Looks up the model both under its
    raw name and under its harbor-stripped form so callers may pass either.

    Args:
        model_name: The model name reported by the agent.

    Returns:
        The fallback model name declared in ``MODEL_INFO.json`` (or
        ``MY_MODELS.json``) via the ``"fallback"`` key, or ``None`` when
        no fallback is registered or the model is unknown.
    """
    info = MODEL_INFO.get(model_name) or MODEL_INFO.get(_strip_provider_prefix(model_name))
    return info.fallback if info is not None else None


def get_max_context_length(model_name: str) -> int:
    """Returns the maximum context length supported by the model.

    Args:
        model_name: Name of the model (with or without provider prefix).
    Returns:
        int: Maximum context length in tokens.
    """
    info = MODEL_INFO.get(model_name) or MODEL_INFO.get(_strip_provider_prefix(model_name))
    if info is None:
        raise KISSError(f"Model '{model_name}' not found in MODEL_INFO")
    return info.context_length
