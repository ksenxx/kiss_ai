# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Model information: pricing and context lengths for supported LLM providers.

``fc=False`` marks a model with unreliable function calling (use for
non-agentic tasks only).
"""

import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kiss.core import config as config_module
from kiss.core.kiss_error import KISSError
from kiss.core.models.model import Model, ThinkingCallback, TokenCallback

logger = logging.getLogger(__name__)


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
        audio_input_price_per_million: float | None = None,
        audio_output_price_per_million: float | None = None,
        alias_of: str | None = None,
        use_responses_api: bool | None = None,
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
        self.audio_input_price_per_1M = audio_input_price_per_million
        self.audio_output_price_per_1M = audio_output_price_per_million
        self.thinking = thinking
        self.fallback = fallback
        self.extended_thinking = extended_thinking
        self.adaptive_thinking = adaptive_thinking
        self.alias_of = alias_of
        self.use_responses_api = use_responses_api


PACKAGE_MODEL_INFO_PATH = Path(__file__).parent / "MODEL_INFO.json"

USER_MY_MODELS_PATH = Path.home() / ".kiss" / "MY_MODELS.json"


def user_model_info_path() -> Path:
    """Return the user-local catalog path ``$KISS_HOME/MODEL_INFO.json``.

    ``KISS_HOME`` defaults to ``~/.kiss``.  The installer (both
    ``install.sh`` and the VS Code extension's ``DependencyInstaller``)
    seeds this file from the bundled catalog on every install/update, and
    the settings panel's "Update Models" button refreshes it in place via
    ``kiss.scripts.update_models --model-info``.  An *installed* KISS
    Sorcar reads its model catalog from here (see
    :func:`_select_catalog_path`); a development checkout keeps reading
    the bundled :data:`PACKAGE_MODEL_INFO_PATH`.
    """
    return config_module.kiss_home() / "MODEL_INFO.json"


def _is_installed_package(package_file: Path | None = None) -> bool:
    """Return True when this package runs from an installed (non-git) copy.

    "Installed" means the packaged VS Code extension bundle
    (``.../kiss_project/src/kiss/...``), which ships without a ``.git``
    marker at its project root.  A development checkout — including the
    ``~/.kiss/kiss_ai`` clone the installer builds from — has ``.git``
    (a directory, or a file in a git worktree) at the root and keeps
    using the bundled catalog, so a stale ``~/.kiss/MODEL_INFO.json``
    can never shadow the checkout's own source of truth.

    Args:
        package_file: The ``model_info.py`` location to classify;
            defaults to this very file.  Parameterized so tests can
            exercise both verdicts against real temp trees.

    Returns:
        True when the project root four levels above the package file
        exists and carries no ``.git`` entry.
    """
    file = Path(__file__) if package_file is None else package_file
    parents = file.resolve().parents
    # .../<root>/src/kiss/core/models/model_info.py -> parents[4] == <root>
    if len(parents) < 5:
        return False
    root = parents[4]
    return not (root / ".git").exists()


def _select_catalog_path(
    env_value: str,
    installed: bool,
    user_path: Path,
    package_path: Path,
) -> Path:
    """Choose which ``MODEL_INFO.json`` file backs ``MODEL_INFO``.

    Precedence:

    1. ``env_value`` (the ``KISS_MODEL_INFO_PATH`` environment variable,
       set by ``update_models.py --model-info`` so the updater reads the
       exact catalog it rewrites) — when it names an existing file.  A
       non-existing override falls back to the bundled catalog, which
       lets the updater bootstrap a brand-new target file.
    2. The user-local copy (``~/.kiss/MODEL_INFO.json``) — only when the
       package is *installed* (see :func:`_is_installed_package`) and
       the copy exists.
    3. The bundled :data:`PACKAGE_MODEL_INFO_PATH`.

    Args:
        env_value: Value of ``KISS_MODEL_INFO_PATH`` (may be empty).
        installed: Whether the package runs from an installed bundle.
        user_path: The user-local catalog location.
        package_path: The bundled catalog location.

    Returns:
        The catalog path to read.
    """
    if env_value:
        override = Path(env_value).expanduser()
        if override.exists():
            return override
        return package_path
    if installed and user_path.exists():
        return user_path
    return package_path

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
            "  use_responses_api (bool, optional) route via the OpenAI v2",
            "      Responses API (/v1/responses) instead of Chat Completions",
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


def _seed_file_atomically(path: Path, content: str) -> None:
    """Create *path* holding *content*, if it does not exist yet.

    The seed is **atomic and non-clobbering**: *content* is staged in a
    sibling temp file and hard-linked into place, so a concurrent reader
    never observes the empty file that a plain ``write_text`` exposes
    between creating the target and writing to it.  If the target
    appears between the existence check and the link (a concurrent
    seeder, or a user edit), the existing file wins.

    Same guarantee and same technique as
    :func:`kiss.server.user_assets.ensure_user_asset_from_default`, which
    documents the torn read a plain ``write_text`` seed caused there.

    Args:
        path: The file to create.
        content: UTF-8 text written on first creation only.

    Raises:
        OSError: When the directory cannot be created or written.
    """
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, staged = tempfile.mkstemp(prefix=f".{path.name}-", dir=str(path.parent))
    try:
        # A buffered file object (rather than a bare os.write, whose
        # partial-write count would have to be handled) guarantees the
        # whole payload is on disk before the link publishes it.
        with os.fdopen(fd, "wb") as f:
            f.write(content.encode("utf-8"))
        os.link(staged, path)
    except FileExistsError:
        logger.debug("Exception caught", exc_info=True)
    finally:
        Path(staged).unlink(missing_ok=True)


def _seed_my_models_file() -> None:
    """Create ``~/.kiss/MY_MODELS.json`` from the inline default if absent.

    Never overwrites an existing file — user edits survive every
    restart.  Silently swallows :class:`OSError` so a read-only HOME or
    missing parent does not break ``MODEL_INFO`` import.

    A reader that caught the file mid-seed used to get ``""``, which
    :func:`_read_my_models` turns into ``{}`` — every user-defined model
    silently missing for the life of that process.
    """
    try:
        _seed_file_atomically(USER_MY_MODELS_PATH, MY_MODELS_DEFAULT_CONTENT)
    except OSError:
        logger.debug("Exception caught", exc_info=True)


def _read_my_models() -> dict[str, dict[str, Any]]:
    """Return parsed model entries from ``~/.kiss/MY_MODELS.json``.

    Auto-seeds the file with :data:`MY_MODELS_DEFAULT_CONTENT` on first
    read.  Returns an empty dict when:

    * The file is missing AND cannot be seeded (read-only FS).
    * The file is unreadable, is not valid UTF-8, or contains malformed
      JSON (``UnicodeDecodeError`` and ``json.JSONDecodeError`` are both
      ``ValueError`` subclasses).
    * The top-level value is not a JSON object.

    Filters out any key starting with ``_`` (documentation / inert
    example entries) and any value that is not a JSON object, so
    documentation lists and stray scalars never reach the model table.
    """
    _seed_my_models_file()
    try:
        raw = json.loads(USER_MY_MODELS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.debug("Ignoring unreadable or corrupt %s", USER_MY_MODELS_PATH, exc_info=True)
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
    * ``use_responses_api`` — tri-state transport choice for
      OpenAI-compatible models.  ``True`` makes the :func:`model` factory
      build an ``OpenAICompatibleModel2`` (the ``/v1/responses`` v2
      transport, live-verified by ``scripts/update_responses_api_support``),
      ``False`` or ``None`` keeps the Chat Completions v1 transport.  A
      caller's ``model_config["use_responses_api"]`` always overrides it.
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
        audio_input_price_per_million=entry.get("audio_input_price_per_1M"),
        audio_output_price_per_million=entry.get("audio_output_price_per_1M"),
        alias_of=entry.get("alias_of"),
        use_responses_api=entry.get("use_responses_api"),
    )


_CATALOG_READ_ATTEMPTS = 5
_CATALOG_RETRY_SECONDS = 0.05


def _read_model_info_json(path: Path) -> dict[str, Any]:
    """Read a ``MODEL_INFO.json`` catalog, tolerating a concurrent rewrite.

    This runs at **import time** (``MODEL_INFO`` is a module-level
    constant), so an unguarded ``json.loads`` turns any transient
    unreadability into a raw ``JSONDecodeError`` traceback out of
    ``import kiss.core.models.model_info`` — the subagent or CLI process
    dies instead of running.  ``update_models.py`` now publishes the
    catalog atomically, but an external tool (a checkout, an editor, an
    older kiss) can still truncate it — or leave behind syntactically
    valid JSON that is not a table at all — so the read retries briefly
    and then fails with a message that names the file.

    Args:
        path: The catalog file to read.

    Returns:
        The decoded catalog object.

    Raises:
        KISSError: When the catalog is still unreadable, unparseable or
            not a JSON object after :data:`_CATALOG_READ_ATTEMPTS`
            attempts.
    """
    last: Exception | None = None
    for attempt in range(_CATALOG_READ_ATTEMPTS):
        try:
            decoded = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(decoded, dict):
                # `dict(None)` / `dict(42)` raise TypeError, which would
                # escape this handler and reach the importer unclassified.
                raise ValueError(
                    f"expected a JSON object, got {type(decoded).__name__}"
                )
            return decoded
        except (OSError, ValueError) as e:
            logger.debug("Exception caught", exc_info=True)
            last = e
            if attempt + 1 < _CATALOG_READ_ATTEMPTS:
                time.sleep(_CATALOG_RETRY_SECONDS)
    raise KISSError(f"Could not read the model catalog at {path}: {last}") from last


def _sync_alias_transport_flags(
    raw: dict[str, Any], user_keys: set[str]
) -> None:
    """Mirror a user-overridden base's ``use_responses_api`` onto its aliases.

    Generated ``-{level}`` thinking aliases ship with a copy of their
    base's transport flag.  When ``MY_MODELS.json`` overrides only the
    base entry, the bundled alias copies would otherwise keep the old
    verdict and silently bypass the user's transport choice whenever an
    alias name is selected.  Aliases the user overrode explicitly are
    left untouched (explicit wins).

    Args:
        raw: The merged catalog mapping, mutated in place.
        user_keys: Catalog keys that came from ``MY_MODELS.json``.
    """
    for name, entry in raw.items():
        if name in user_keys or not isinstance(entry, dict):
            continue
        base_name = entry.get("alias_of")
        if not base_name or base_name not in user_keys:
            continue
        base = raw.get(base_name)
        if not isinstance(base, dict):
            continue
        flag = base.get("use_responses_api")
        if flag is None:
            entry.pop("use_responses_api", None)
        else:
            entry["use_responses_api"] = flag


def _load_model_info() -> dict[str, ModelInfo]:
    """Load ``MODEL_INFO`` from JSON, applying cache-pricing defaults.

    The catalog file is chosen by :func:`_select_catalog_path`:
    ``KISS_MODEL_INFO_PATH`` (when set and existing), else the
    user-local ``~/.kiss/MODEL_INFO.json`` on installed copies, else the
    bundled :data:`PACKAGE_MODEL_INFO_PATH`.  A non-bundled catalog that
    turns out unreadable, corrupt, or schema-invalid (valid JSON whose
    entries lack required fields) falls back to the bundled one so a
    damaged user copy can never brick every KISS process at import time.
    ``~/.kiss/MY_MODELS.json`` (auto-seeded on first read) is then
    merged on top: matching keys override the catalog entry, and
    brand-new keys are added.  Generated thinking aliases of a
    user-overridden base then re-mirror the base's ``use_responses_api``
    flag (see :func:`_sync_alias_transport_flags`).
    """
    catalog_path = _select_catalog_path(
        os.environ.get("KISS_MODEL_INFO_PATH", "").strip(),
        _is_installed_package(),
        user_model_info_path(),
        PACKAGE_MODEL_INFO_PATH,
    )
    try:
        return _load_catalog_file(catalog_path)
    except (KISSError, KeyError, TypeError, ValueError, AttributeError):
        if catalog_path == PACKAGE_MODEL_INFO_PATH:
            raise
        logger.warning(
            "Model catalog %s is unreadable or invalid; falling back to "
            "the bundled %s",
            catalog_path,
            PACKAGE_MODEL_INFO_PATH,
            exc_info=True,
        )
        return _load_catalog_file(PACKAGE_MODEL_INFO_PATH)


def _load_catalog_file(catalog_path: Path) -> dict[str, ModelInfo]:
    """Read *catalog_path*, merge ``MY_MODELS.json``, and build the table.

    Args:
        catalog_path: The ``MODEL_INFO.json`` file to load.

    Returns:
        The complete name → :class:`ModelInfo` mapping.

    Raises:
        KISSError: When the catalog is unreadable or not a JSON object.
        KeyError, TypeError, ValueError, AttributeError: When an entry
            does not conform to the catalog schema (e.g. a required
            field is missing or has the wrong type).
    """
    raw = _read_model_info_json(catalog_path)
    user_entries = _read_my_models()
    for name, entry in user_entries.items():
        raw[name] = entry
    _sync_alias_transport_flags(raw, set(user_entries))
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
        label: Human-readable provider name shown in the model picker and
            the history badges (e.g. ``"OpenRouter"``).  Kept here so the
            UI label is derived from the registry rather than restated in
            a hand-written table next to it.
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
    label: str
    host: str
    base_url: str
    prefixes: tuple[str, ...]
    excludes: tuple[str, ...]
    api_key_name: str
    tools_accept_reasoning_effort: bool | None
    delegate_tools_to_responses: bool


OPENAI_COMPATIBLE_PROVIDERS: tuple[OpenAICompatibleProvider, ...] = (
    OpenAICompatibleProvider(
        name="openrouter",
        label="OpenRouter",
        host="openrouter.ai",
        base_url="https://openrouter.ai/api/v1",
        prefixes=("openrouter/",),
        excludes=(),
        api_key_name="OPENROUTER_API_KEY",
        tools_accept_reasoning_effort=True,
        delegate_tools_to_responses=False,
    ),
    OpenAICompatibleProvider(
        name="openai",
        label="OpenAI",
        host="api.openai.com",
        base_url="https://api.openai.com/v1",
        prefixes=_OPENAI_PREFIXES,
        excludes=("openai/gpt-oss", "codex/"),
        api_key_name="OPENAI_API_KEY",
        tools_accept_reasoning_effort=False,
        delegate_tools_to_responses=True,
    ),
    OpenAICompatibleProvider(
        name="together",
        label="Together",
        host="api.together.xyz",
        base_url="https://api.together.xyz/v1",
        prefixes=_TOGETHER_PREFIXES,
        excludes=(),
        api_key_name="TOGETHER_API_KEY",
        tools_accept_reasoning_effort=True,
        delegate_tools_to_responses=False,
    ),
    OpenAICompatibleProvider(
        name="zai",
        label="Z.AI",
        host="api.z.ai",
        base_url="https://api.z.ai/api/paas/v4",
        prefixes=("glm-",),
        excludes=(),
        api_key_name="ZAI_API_KEY",
        tools_accept_reasoning_effort=None,
        delegate_tools_to_responses=False,
    ),
    OpenAICompatibleProvider(
        name="moonshot",
        label="Moonshot",
        host="api.moonshot.ai",
        base_url="https://api.moonshot.ai/v1",
        prefixes=("kimi-", "moonshot-"),
        excludes=(),
        api_key_name="MOONSHOT_API_KEY",
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


_NATIVE_PROVIDERS: tuple[tuple[str, str, str | None], ...] = (
    ("cc/", "Claude Code CLI", None),
    ("codex/", "Codex CLI", None),
    ("claude-", "Anthropic", "ANTHROPIC_API_KEY"),
    ("gemini-", "Gemini", "GEMINI_API_KEY"),
)
"""Providers that are not OpenAI-compatible, as ``(prefix, label, key)``.

The counterpart of :data:`OPENAI_COMPATIBLE_PROVIDERS` for the four
vendors the :func:`model` factory routes with their own SDK or CLI.  A
``None`` key means the credential is a local executable rather than an
API key.  Together the two tables are the complete routing table, and
:func:`get_model_provider` is the only place either is consulted.
"""


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

    The underlying :class:`ImportError` is chained onto the
    :class:`KISSError` and quoted in its message, because it is not
    always "the SDK is absent": an ``ImportError`` raised from *inside*
    an installed SDK (the missing-``brotli`` regression made ``urllib3``
    do exactly that) would otherwise be reported as a package the user
    has already installed, with the real traceback discarded.

    Args:
        class_name: Class attribute name (e.g. ``"GeminiModel"``).
        error_message: KISSError message raised when the class failed to import.

    Returns:
        The model class.

    Raises:
        KISSError: If the class (or its SDK) could not be imported.
    """
    import kiss.core.models as models

    try:
        return getattr(models, class_name)
    except ImportError as e:
        raise KISSError(f"{error_message} (import failed: {e})") from e


def _openai_compatible(
    model_name: str,
    base_url: str,
    api_key: str,
    model_config: dict[str, Any] | None,
    token_callback: TokenCallback | None,
    thinking_callback: ThinkingCallback | None = None,
    use_responses_api: bool = False,
) -> Model:
    """Build the OpenAI-compatible adapter for one factory-routed model.

    Args:
        model_name: The model id (possibly carrying an ``openrouter/`` prefix).
        base_url: The vendor's OpenAI-compatible API root.
        api_key: Bearer token for the endpoint.
        model_config: Optional model parameters, forwarded verbatim.
        token_callback: Called with each streamed text token.
        thinking_callback: Called when a thinking block starts/ends.
        use_responses_api: When True, build ``OpenAICompatibleModel2`` — the
            v2 transport that sends every request to ``/v1/responses`` —
            instead of the Chat Completions v1 adapter.

    Returns:
        The constructed model adapter.

    Raises:
        KISSError: If the OpenAI SDK is not installed.
    """
    cls = _load_model_class(
        "OpenAICompatibleModel2" if use_responses_api else "OpenAICompatibleModel",
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

_LEVEL_ALIAS_SUFFIXES = ("-max", "-high", "-medium", "-low")


def _strip_thinking_alias(bare: str) -> str:
    """Strip a synthetic ``-{thinking_level}`` alias suffix from a model name.

    ``-xhigh`` / ``-max`` / ``-high`` / ``-medium`` / ``-low`` are
    KISS-internal alias suffixes (see ``update_models.py``) that map onto
    the same provider model id as their base entry (``-max`` is the top of
    the Moonshot/Kimi scale, ``-xhigh`` the top of the OpenAI scale).
    Provider pricing tables and endpoints only know the base names, so
    every pricing lookup and outbound request must consult the base name.

    ``-xhigh`` is stripped unconditionally (no real upstream model ends in
    it). The other level suffixes collide with real upstream model names
    (e.g. ``openrouter/openai/o3-mini-high``), so they are stripped only
    when ``bare`` is EXACTLY a catalog key carrying the ``alias_of``
    marker that ``update_models.py`` writes on every generated alias; in
    that case the marker's recorded base key is returned verbatim. Callers
    must therefore pass the full catalog key (including any
    ``openrouter/`` prefix) BEFORE removing routing prefixes — fuzzy
    suffix-tail matching is deliberately avoided so an unrelated catalog
    alias can never rewrite a similarly-named custom model. Returning the
    input unchanged when no alias matches keeps callers simple.

    Args:
        bare: A model name, possibly ending in a thinking-level suffix.

    Returns:
        The alias's recorded base catalog key when ``bare`` is a generated
        alias, otherwise ``bare`` unchanged (modulo unconditional
        ``-xhigh`` stripping).
    """
    if bare.endswith(_XHIGH_SUFFIX):
        return bare[: -len(_XHIGH_SUFFIX)]
    for suffix in _LEVEL_ALIAS_SUFFIXES:
        if not bare.endswith(suffix):
            continue
        info = MODEL_INFO.get(bare)
        if info is not None and info.alias_of:
            return info.alias_of
        return bare
    return bare


def _openai_charges_cache_writes(bare: str) -> bool:
    """Return True when the OpenAI model bills prompt-cache writes.

    Cache writes are free on OpenAI models before the GPT-5.6 family.  For
    GPT-5.6 models and later model families, cache writes are billed at
    1.25x the uncached input token rate and reported in
    ``prompt_tokens_details.cache_write_tokens`` (Chat Completions) /
    ``input_tokens_details.cache_write_tokens`` (Responses).  Verified at
    https://developers.openai.com/api/docs/guides/prompt-caching and the
    OpenAI pricing page (gpt-5.6-sol $6.25, -terra $3.125, -luna $1.25
    per MTok cache write = exactly 1.25x their input prices).  ``-pro``
    variants publish no cache pricing, so they stay on free writes.

    Args:
        bare: An OpenAI model name without any provider prefix.

    Returns:
        True when cache-write tokens are billed at 1.25x the input price.
    """
    return bare.startswith("gpt-5.6") and "-pro" not in bare


def _openai_cache_read_multiplier(bare: str) -> float:
    """Return the cached-input price multiplier for an OpenAI model.

    OpenAI bills prompt-cache reads at a per-model fraction of the base input
    price (cache writes are free before the GPT-5.6 family; see
    :func:`_openai_charges_cache_writes`). The multipliers below match
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
        return 0.10
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
    return 0.50


def _openai_bare_name(name: str) -> str | None:
    """Return the bare OpenAI model name for cache pricing, or ``None``.

    Recognizes both directly-routed OpenAI models (``gpt-*``, ``o1``/``o3``/
    ``o4``, ``computer-use``) and OpenRouter passthrough OpenAI models
    (``openrouter/openai/*`` and ``openrouter/~openai/*``). Embeddings, the
    open-weight ``gpt-oss`` models, and subscription ``codex/`` models are
    excluded (they have no per-token cache discount in this table).

    ``openai/`` needs no exclusion here: no element of
    :data:`_OPENAI_PREFIXES` is a prefix of it, so ``openai/gpt-oss-*``
    never reaches the check in the first place (it is a Together name).

    Args:
        name: The MODEL_INFO key.

    Returns:
        The OpenAI model name without provider prefix, or ``None`` if ``name``
        is not an OpenAI cache-eligible model.
    """
    name = _strip_thinking_alias(name)
    if name.startswith(_OPENAI_OPENROUTER_PREFIXES):
        bare = name.split("/", 2)[2]
        if bare.startswith("gpt-oss"):
            return None
        return bare
    if name.startswith(_OPENAI_PREFIXES) and not name.startswith(
        ("text-embedding", "codex/")
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
        if _openai_charges_cache_writes(bare):
            info.cache_write_price_per_1M = inp * 1.25
        else:
            info.cache_write_price_per_1M = 0.0
        return
    if name.startswith("gemini-"):
        info.cache_read_price_per_1M = inp * 0.1
        info.cache_write_price_per_1M = 0.0
        return
    if name.startswith(_GOOGLE_OPENROUTER_PREFIXES):
        info.cache_read_price_per_1M = inp * 0.25
        info.cache_write_price_per_1M = 0.0
        return
    if name.startswith("openrouter/deepseek/"):
        multiplier = 0.02
        if name.startswith("openrouter/deepseek/deepseek-v4-pro"):
            multiplier = 0.003625 / 0.435
        info.cache_read_price_per_1M = inp * multiplier
        info.cache_write_price_per_1M = inp
        return
    if name.startswith("openrouter/qwen/"):
        info.cache_read_price_per_1M = inp * 0.2
        info.cache_write_price_per_1M = inp * 1.25
        return
    if name.startswith(_QUARTER_CACHE_OPENROUTER_PREFIXES):
        info.cache_read_price_per_1M = inp * 0.25
        info.cache_write_price_per_1M = 0.0
        return
    if name.startswith(("kimi-", "moonshot-")):
        info.cache_read_price_per_1M = inp * 0.25
        info.cache_write_price_per_1M = 0.0
        return


for _name, _info in MODEL_INFO.items():
    _apply_cache_pricing(_name, _info)

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


def _lookup_model_info(model_name: str) -> ModelInfo | None:
    """Return the catalog entry for *model_name*, or ``None`` when unknown.

    The one lookup rule shared by :func:`calculate_cost`,
    :func:`get_fallback_model` and :func:`get_max_context_length`: the
    name is tried as given, then in its harbor-stripped form (see
    :func:`_strip_provider_prefix`), so callers may pass either.

    Args:
        model_name: Name of the model (with or without provider prefix).

    Returns:
        The matching :class:`ModelInfo`, or ``None``.
    """
    return MODEL_INFO.get(model_name) or MODEL_INFO.get(_strip_provider_prefix(model_name))


def model_runs_task_to_completion(model_name: str) -> bool:
    """Whether *model_name* names a CLI agent that runs a whole task itself.

    ``cc/*`` (Claude Code) and ``codex/*`` (Codex) models are full coding
    agents: an agentic :class:`~kiss.core.kiss_agent.KISSAgent` hands them
    the entire task in one invocation instead of driving a turn-by-turn
    KISS tool loop, and their native tools run directly on the host (so
    they cannot honor ``docker_image`` isolation).

    Args:
        model_name: Full model name including any provider prefix.

    Returns:
        True for ``cc/*`` and ``codex/*`` model names.
    """
    return model_name.startswith(("cc/", "codex/"))


def _wants_responses_api(
    model_name: str, model_config: dict[str, Any] | None
) -> bool:
    """Decide whether the factory should build the v2 (Responses API) adapter.

    The caller's ``model_config["use_responses_api"]`` wins in either
    direction when present.  Otherwise the decision is the model's catalog
    ``use_responses_api`` flag, written by
    ``scripts/update_responses_api_support.py`` only after the model passed a
    live probe through ``/v1/responses`` (plain generation, plus a tool
    round-trip for function-calling models).  Generated ``-{level}`` thinking
    aliases carry the flag themselves (the alias entry mirrors its base), so
    a plain catalog lookup suffices.

    Args:
        model_name: Model name after harbor-prefix stripping.
        model_config: The caller-supplied model configuration, if any.

    Returns:
        True when the model should be built as ``OpenAICompatibleModel2``.
    """
    if model_config is not None:
        flag = model_config.get("use_responses_api")
        if flag is not None:
            return bool(flag)
    info = _lookup_model_info(model_name)
    return info is not None and info.use_responses_api is True


def _registered_provider_for_exact_base_url(
    base_url: str,
) -> OpenAICompatibleProvider | None:
    """Return the vendor whose default endpoint IS *base_url* exactly.

    Exact match (modulo a trailing slash), not the substring match of
    :func:`openai_compatible_provider_for_base_url`: a wire-test capture
    server whose URL merely embeds a vendor host as a path segment is a
    custom gateway with unknown ``/v1/responses`` support, and must not
    inherit the catalog's live-verified transport flag.

    Args:
        base_url: The ``model_config["base_url"]`` override.

    Returns:
        The matching provider, or None for custom gateways.
    """
    normalized = base_url.rstrip("/")
    for provider in OPENAI_COMPATIBLE_PROVIDERS:
        if normalized == provider.base_url.rstrip("/"):
            return provider
    return None


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
            If it contains "base_url", routing is bypassed and an
            OpenAI-compatible adapter is built with that base_url and
            optional "api_key".
            "use_responses_api" selects the transport for OpenAI-compatible
            models: True builds the v2 Responses-API adapter
            (OpenAICompatibleModel2), False forces the Chat Completions v1
            adapter, and when absent the model's catalog ``use_responses_api``
            flag (live-verified by ``scripts/update_responses_api_support``)
            decides — for a "base_url" override the catalog flag is honored
            only when the URL is exactly a registered vendor's default
            endpoint (custom gateways stay on v1).
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
        # An explicit config flag always decides the transport.  The
        # catalog's live-verified flag applies only when the override IS
        # the default endpoint of the SAME vendor the model name routes to
        # (e.g. a model switch that carries the provider base_url along to
        # preserve a per-task API key) — the probe verified that exact
        # model/endpoint pair.  Any other endpoint (a custom gateway, or a
        # different vendor's default URL) has unverified /v1/responses
        # support for this model and stays on the v1 adapter.
        use_v2 = model_config.get("use_responses_api")
        if use_v2 is None:
            endpoint_provider = _registered_provider_for_exact_base_url(base_url)
            use_v2 = (
                endpoint_provider is not None
                and endpoint_provider
                is _match_openai_compatible_provider(model_name)
                and _wants_responses_api(model_name, None)
            )
        return _openai_compatible(
            model_name,
            base_url,
            api_key,
            filtered or None,
            token_callback,
            thinking_callback,
            use_responses_api=bool(use_v2),
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
            use_responses_api=_wants_responses_api(model_name, model_config),
        )
    if model_name.startswith("gemini-"):
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
    if model_runs_task_to_completion(model_name):
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
    configured = _configured_providers()
    return sorted(
        name
        for name, info in MODEL_INFO.items()
        if info.is_generation_supported and configured.get(get_model_provider(name), False)
    )


def _configured_providers() -> dict[str, bool]:
    """Return, per provider label, whether its credential is present.

    Built from the same two routing tables :func:`get_model_provider` uses,
    so a vendor added to :data:`OPENAI_COMPATIBLE_PROVIDERS` shows up in
    the model picker without any further registration.

    Returns:
        Mapping of provider label to whether it is usable right now — an
        API key for HTTP providers, the executable on ``PATH`` for the
        subscription CLIs.
    """
    import shutil

    from kiss.core.models.codex_model import find_codex_executable

    keys = config_module.DEFAULT_CONFIG
    configured = {
        "Claude Code CLI": shutil.which("claude") is not None,
        "Codex CLI": find_codex_executable() is not None,
        "Unknown": False,
    }
    for provider in OPENAI_COMPATIBLE_PROVIDERS:
        configured[provider.label] = bool(getattr(keys, provider.api_key_name, ""))
    for _prefix, label, api_key_name in _NATIVE_PROVIDERS:
        if api_key_name is not None:
            configured[label] = bool(getattr(keys, api_key_name, ""))
    return configured


def get_model_provider(model_name: str) -> str:
    """Return the human-readable provider label that routes *model_name*.

    This is the single routing lookup: the label comes from the
    :data:`OPENAI_COMPATIBLE_PROVIDERS` registry and
    :data:`_NATIVE_PROVIDERS` — the same tables :func:`model` dispatches
    on — so a newly registered vendor is labelled correctly here, and in
    every caller, without any further edit.  The order mirrors
    :func:`model`: registered OpenAI-compatible vendors first (their
    ``excludes`` are what keep ``codex/`` and ``openai/gpt-oss`` out),
    then the native SDK/CLI providers.

    Args:
        model_name: A ``MODEL_INFO`` key.

    Returns:
        str: The provider label (e.g. ``"OpenAI"``, ``"Anthropic"``,
            ``"OpenRouter"``), or ``"Unknown"`` if no route matches.
    """
    provider = _match_openai_compatible_provider(model_name)
    if provider is not None:
        return provider.label
    for prefix, label, _api_key_name in _NATIVE_PROVIDERS:
        if model_name.startswith(prefix):
            return label
    return "Unknown"


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
            "ANTHROPIC_API_KEY": "claude-opus-4-7",
            "OPENAI_API_KEY": "gpt-5.6-sol-medium",
            "GEMINI_API_KEY": "gemini-3.6-flash",
            "OPENROUTER_API_KEY": "openrouter/anthropic/claude-opus-4.7",
            "TOGETHER_API_KEY": "moonshotai/Kimi-K3",
            "cc": "cc/opus",
            "codex": "codex/default",
        }
    )


def _openai_long_context_prices(
    model_name: str,
) -> tuple[int, float, float, float, float | None] | None:
    """Return ``(threshold, input, output, cached, cache_write)`` long-context prices.

    The last element is the long-context cache-write price per 1M tokens,
    or ``None`` for models whose cache writes are free (pre-GPT-5.6
    families).  Prices verified against the OpenAI pricing page
    (https://developers.openai.com/api/docs/pricing).
    """
    bare = _strip_thinking_alias(_strip_provider_prefix(model_name))
    if bare.startswith(_OPENAI_OPENROUTER_PREFIXES):
        bare = bare.split("/", 2)[2]
    if bare.startswith("gpt-5.6-sol"):
        return 272_000, 10.00, 45.00, 1.00, 12.50
    if bare.startswith("gpt-5.6-terra"):
        return 272_000, 5.00, 22.50, 0.50, 6.25
    if bare.startswith("gpt-5.6-luna"):
        return 272_000, 2.00, 9.00, 0.20, 2.50
    if bare.startswith("gpt-5.5") and "-pro" not in bare:
        return 272_000, 10.00, 45.00, 1.00, None
    if (
        bare.startswith("gpt-5.4")
        and "-pro" not in bare
        and "-mini" not in bare
        and "-nano" not in bare
    ):
        return 272_000, 5.00, 22.50, 0.50, None
    return None


def _gemini_long_context_prices(
    model_name: str,
) -> tuple[int, float, float, float, float | None] | None:
    """Return ``(threshold, input, output, cached, cache_write)`` long-context prices.

    The last element is always ``None`` for Gemini (context-cache writes
    are free; only storage-per-hour is billed, which KISS's implicit
    caching never incurs).  Prices verified against
    https://ai.google.dev/gemini-api/docs/pricing.
    """
    bare = _strip_thinking_alias(_strip_provider_prefix(model_name))
    if bare.startswith(_GOOGLE_OPENROUTER_PREFIXES):
        bare = bare.split("/", 2)[2]
    if bare.startswith(("gemini-3-pro", "gemini-3.1-pro")):
        return 200_000, 4.00, 18.00, 0.40, None
    if bare.startswith("gemini-2.5-pro"):
        return 200_000, 2.50, 15.00, 0.25, None
    return None


def calculate_cost(
    model_name: str,
    num_input_tokens: int,
    num_output_tokens: int,
    num_cache_read_tokens: int = 0,
    num_cache_write_tokens: int = 0,
    num_cache_write_1h_tokens: int = 0,
    num_audio_input_tokens: int = 0,
    num_audio_output_tokens: int = 0,
) -> float:
    """Calculates the cost in USD for the given token counts.

    Args:
        model_name: Name of the model (with or without provider prefix).
        num_input_tokens: Number of non-cached TEXT input tokens (audio
            input tokens already split out).
        num_output_tokens: Number of TEXT output tokens (audio output
            tokens already split out).
        num_cache_read_tokens: Number of tokens read from cache.
        num_cache_write_tokens: Number of standard/5-minute cache-write tokens.
        num_cache_write_1h_tokens: Number of one-hour Anthropic cache-write tokens.
        num_audio_input_tokens: Number of AUDIO input tokens (the
            ``prompt_tokens_details.audio_tokens`` subset of an OpenAI
            audio-chat response), billed at the model's audio input
            rate when registered, otherwise at the text input rate.
        num_audio_output_tokens: Number of AUDIO output tokens
            (``completion_tokens_details.audio_tokens``), billed at the
            model's audio output rate when registered, otherwise at the
            text output rate.

    Returns:
        float: Cost in USD.

    Raises:
        KISSError: If positive usage is reported for a model without pricing.
    """
    info = _lookup_model_info(model_name)
    total_tokens = (
        num_input_tokens
        + num_output_tokens
        + num_cache_read_tokens
        + num_cache_write_tokens
        + num_cache_write_1h_tokens
        + num_audio_input_tokens
        + num_audio_output_tokens
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
    prompt_tokens = total_tokens - num_output_tokens - num_audio_output_tokens
    if long_prices is not None and prompt_tokens > long_prices[0]:
        _, input_price, output_price, cr_price, long_cw_price = long_prices
        if long_cw_price is not None:
            cw_price = long_cw_price
    audio_in_price = (
        info.audio_input_price_per_1M
        if info.audio_input_price_per_1M is not None
        else input_price
    )
    audio_out_price = (
        info.audio_output_price_per_1M
        if info.audio_output_price_per_1M is not None
        else output_price
    )
    input_cost = num_input_tokens * input_price
    output_cost = num_output_tokens * output_price
    cache_read_cost = num_cache_read_tokens * cr_price
    return (
        input_cost
        + output_cost
        + cache_read_cost
        + num_cache_write_tokens * cw_price
        + num_cache_write_1h_tokens * cw1h_price
        + num_audio_input_tokens * audio_in_price
        + num_audio_output_tokens * audio_out_price
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
    info = _lookup_model_info(model_name)
    return info.fallback if info is not None else None


def get_max_context_length(model_name: str) -> int:
    """Returns the maximum context length supported by the model.

    Args:
        model_name: Name of the model (with or without provider prefix).
    Returns:
        int: Maximum context length in tokens.
    """
    info = _lookup_model_info(model_name)
    if info is None:
        raise KISSError(f"Model '{model_name}' not found in MODEL_INFO")
    return info.context_length
