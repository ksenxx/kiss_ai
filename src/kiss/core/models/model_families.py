# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Model families, and the choice of a model of another family.

A *family* is the vendor that trained the model (``anthropic``,
``openai``, ``google``, ``deepseek``, ...), whatever route serves it: a
Claude model called directly, through Bedrock, OpenRouter or the Claude
Code CLI is one family.  A check that shares the executor's family shares
its blind spots: a same-model reviewer reads a sentence the way the
executor misread it and passes the misreading, while a content filter or
an output cap that stops one family's model stops its OpenRouter twin
too.  So a second opinion (:func:`other_family_model`) and a fail-over
(:func:`fallback_of_another_family`) come from another family, chosen
among the models the installation has credentials for.

A sub-agent is a different question: it is "the same model with less
context" only when it runs the caller's model at the caller's reasoning
effort (:func:`same_model_and_effort`), whatever the family.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from kiss.core.models.model_info import (
    BEDROCK_PREFIX,
    MODEL_INFO,
    _configured_providers,
    _is_registered_endpoint_model,
    _lookup_model_info,
    _match_openai_compatible_provider,
    _native_provider,
    _strip_provider_prefix,
    _strip_thinking_alias,
    _twin_key,
    bedrock_claude_name,
    get_fallback_model,
    get_model_provider,
    registered_bedrock_model,
)

_FAMILY_PREFIXES: tuple[tuple[str, str], ...] = (
    ("claude-", "anthropic"),
    ("cc/", "anthropic"),
    ("bedrock/", "anthropic"),
    ("gpt-", "openai"),
    ("o1", "openai"),
    ("o3", "openai"),
    ("o4", "openai"),
    ("codex", "openai"),
    ("gemini-", "google"),
    ("deepseek", "deepseek"),
    ("moonshotai/", "moonshot"),
    ("kimi-", "moonshot"),
    ("glm-", "z-ai"),
    ("zai-org/", "z-ai"),
    ("qwen/", "qwen"),
    ("meta-llama/", "meta"),
    ("mistralai/", "mistral"),
    ("x-ai/", "xai"),
)
"""Name prefixes (lower-case) that name a family, checked in order.

A vendor's own API spells its models without a vendor segment
(``kimi-k3``, ``glm-4.7``) while Together and OpenRouter prefix one
(``moonshotai/Kimi-K3``, ``zai-org/GLM-4.7``, ``z-ai/glm-4.7``): the
direct spellings are listed so the two routes of one model are one
family.
"""

SECOND_OPINION_MODELS: tuple[str, ...] = (
    "gpt-6-astra",
    "claude-opus-5-5",
    "gemini-3.1-pro-preview",
    "openrouter/openai/gpt-6-astra",
    "openrouter/anthropic/claude-opus-5.5",
    "openrouter/google/gemini-3.1-pro-preview",
    "openrouter/deepseek/deepseek-v4-pro",
    "moonshotai/Kimi-K3",
    "deepseek-ai/DeepSeek-V4-Pro",
    "cc/opus",
    "codex/default",
    "glm-4.7",
    "kimi-k3",
    BEDROCK_PREFIX,
)
"""Strong models of each family, in the order a second opinion prefers them.

The bare :data:`BEDROCK_PREFIX` stands for the strongest ``bedrock/``
model the catalog registers, whose id depends on the AWS region.  It
and the two native routes before it come last: they answer only an
installation the earlier candidates leave without a second opinion.
"""


def family_of(model_name: str) -> str:
    """Return the family of *model_name*: the vendor that trained it.

    Args:
        model_name: A model name as the catalog spells it, with or
            without a route prefix (``openrouter/<vendor>/<model>``,
            ``bedrock/<id>``, ``cc/<alias>``).

    Returns:
        The family name (``"anthropic"``, ``"openai"``, ``"google"``,
        ``"deepseek"``, ...); for an unknown vendor the route's vendor
        segment, else the name up to its first ``-``.
    """
    name = model_name.strip().lower()
    if name.startswith("openrouter/"):
        name = name[len("openrouter/"):].lstrip("~")
    for prefix, family in _FAMILY_PREFIXES:
        if name.startswith(prefix):
            return family
    if "/" in name:
        return name.split("/", 1)[0]
    return name.split("-", 1)[0]


def same_family(model_a: str, model_b: str) -> bool:
    """Return whether the two models are of the same family (see :func:`family_of`)."""
    return family_of(model_a) == family_of(model_b)


_OPENAI_COMPATIBLE = "OpenAI-compatible"
"""The route of a model the factory builds an OpenAI-compatible adapter for."""

_NATIVE_EFFORT: dict[str, tuple[str, tuple[str, ...]]] = {
    _OPENAI_COMPATIBLE: ("reasoning", ("effort",)),
    "Anthropic": ("output_config", ("effort",)),
    "Bedrock": ("output_config", ("effort",)),
    "Claude Code CLI": ("output_config", ("effort",)),
    "Gemini": (
        "thinking_config",
        ("thinking_level", "thinkingLevel", "thinking_budget", "thinkingBudget"),
    ),
}
"""Where an adapter reads the effort in its provider's own spelling.

Per route (:data:`_OPENAI_COMPATIBLE`, or a
:func:`~kiss.core.models.model_info._native_provider` label), the
``model_config`` key and the fields of its value, of which the first
one set is the effort.  Gemini's SDK takes a field under either
spelling of its name, and a thinking budget stands for a level.  The
Codex CLI, absent here, reads none.  Another provider's key means
nothing to an adapter.
"""


def _asked_effort(route: str, config: dict[str, Any]) -> str | None:
    """Return the effort *config* asks the adapter of *route* for, or ``None`` when it sets none.

    The portable ``reasoning_effort`` outranks the setting in the
    provider's own spelling (:data:`_NATIVE_EFFORT`), and a null value is
    no setting (to Gemini's adapter, neither is an empty one).  An
    OpenAI-compatible adapter, though, takes a key that is there for the
    caller's choice: with nothing but nulls it asks its provider for no
    effort, the provider's default (``""`` here), where another adapter
    falls back on the catalog's level.
    """
    key, fields = _NATIVE_EFFORT.get(route, ("", ()))
    native: Any = config.get(key)
    effort = config.get("reasoning_effort")
    if route == "Gemini":
        effort = effort or None
        if hasattr(native, "model_dump"):
            native = native.model_dump()  # the SDK's ``ThinkingConfig``, taken as well as a dict
    if not isinstance(native, dict):
        native = {}
    for field in fields:
        if effort is None:
            effort = native.get(field)
    if effort is not None:
        return str(getattr(effort, "value", effort))  # an SDK enum is its value
    null_key = "reasoning_effort" in config or "effort" in native
    return "" if route == _OPENAI_COMPATIBLE and null_key else None


def _model_and_effort(
    model_name: str, model_config: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """Return *model_name* as its model id without the route, and its reasoning effort.

    The name is read in the order the model factory reads it
    (:func:`~kiss.core.models.model_info.model`): a redundant provider
    prefix comes off before the effort alias is looked up
    (``anthropic/claude-opus-5-5-max`` is ``claude-opus-5-5-max``), and a
    Bedrock model or inference-profile id is the Claude model behind it
    (``bedrock/us.anthropic.claude-opus-4-6-v1:0`` is
    ``claude-opus-4-6``; never the catalog key it is merely priced as,
    which for a model the catalog lists under Bedrock alone is another
    Bedrock id).  A ``MY_MODELS.json`` model with an endpoint of
    its own keeps the whole name it is registered under: that name is
    what its gateway serves, whatever bundled model it resembles.

    The effort is the one the pair asks its provider for, read as the
    adapter the model factory builds for the name reads it: the one
    *model_config* sets (:func:`_asked_effort`), else the catalog
    entry's (``claude-fable-5-2`` runs at ``high``, as
    ``claude-fable-5-2-high`` does); a name the catalog does not list
    carries it only in its ``-{level}`` suffix.  Gemini has no level
    above ``high`` and runs ``xhigh`` and ``max`` at it.
    """
    name = model_name.strip()
    config = model_config or {}
    registered = _is_registered_endpoint_model(name)
    if not registered:
        name = _strip_provider_prefix(name)
        name = bedrock_claude_name(name) or name
    base = _strip_thinking_alias(name)
    info = _lookup_model_info(name)
    # The factory's order: an endpoint of its own, an OpenAI-compatible vendor, a native route.
    if registered or "base_url" in config or _match_openai_compatible_provider(name):
        route = _OPENAI_COMPATIBLE
    else:
        route = _native_provider(name) or ""
    effort = _asked_effort(route, config)
    if effort is None:
        effort = (info.thinking if info else None) or name[len(base):].lstrip("-")
    effort = effort.lower()
    if route == "Gemini" and effort in ("xhigh", "max"):
        effort = "high"
    return _twin_key(base if registered else base.rsplit("/", 1)[-1]), effort


def same_model_and_effort(
    model_a: str,
    model_b: str,
    config_a: dict[str, Any] | None = None,
    config_b: dict[str, Any] | None = None,
) -> bool:
    """Return whether the two names are one model run at one reasoning effort.

    The route does not make a model another one
    (``openrouter/anthropic/claude-fable-5.2-max``,
    ``anthropic/claude-fable-5-2-max`` and
    ``bedrock/us.anthropic.claude-fable-5-2-v1:0-max`` are
    ``claude-fable-5-2-max``); the effort does (``claude-fable-5-2-max``
    and ``claude-fable-5-2-medium`` differ, as do two models of one
    family), and the effort a model configuration sets, as
    ``reasoning_effort`` or in the provider's own spelling, outranks the
    one the name stands for.  A CLI alias (``cc/opus``) is a
    name of its own and equals only itself, as does a ``MY_MODELS.json``
    model with an endpoint of its own.

    Args:
        model_a: A model name as the catalog spells it.
        model_b: Another.
        config_a: The model configuration *model_a* runs with, if any.
        config_b: The one *model_b* runs with, if any.
    """
    return _model_and_effort(model_a, config_a) == _model_and_effort(model_b, config_b)


EFFORT_LEVELS: tuple[str, ...] = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
"""The reasoning efforts the catalog names (``thinking``), in ascending order."""


def other_effort_model(model_name: str, model_config: dict[str, Any] | None = None) -> str | None:
    """Return the catalog's name for *model_name*'s model at another reasoning effort.

    The pin left to a caller with no other family configured
    (:func:`other_family_model`).  Of the catalog's effort aliases of the
    model (``claude-fable-5-2-medium`` for ``claude-fable-5-2``), it is
    the nearest effort below the caller's whose provider has a
    credential, else the nearest above; an effort off
    :data:`EFFORT_LEVELS` counts as above them all.  The caller's own
    name is never the pin: a sub-agent on it inherits the caller's
    configuration, and with it the caller's effort.

    Args:
        model_name: The caller's model.
        model_config: The model configuration it runs with, whose
            effort outranks the catalog's.

    Returns:
        The alias, or ``None`` when the catalog lists no other effort of
        the model or its provider has no credential.
    """
    name = _strip_provider_prefix(model_name.strip())
    base = _strip_thinking_alias(name)
    effort = _model_and_effort(model_name, model_config)[1]
    aliases = {
        info.thinking: alias
        for alias, info in MODEL_INFO.items()
        if info.alias_of == base and alias != name
    }
    rank = (*EFFORT_LEVELS, effort).index(effort)
    for level in (*reversed(EFFORT_LEVELS[:rank]), *EFFORT_LEVELS[rank + 1:]):
        if level in aliases and _available(aliases[level]):
            return aliases[level]
    return None


def _available(model_name: str) -> bool:
    """Whether *model_name* is in the catalog and its provider has a credential."""
    if _lookup_model_info(model_name) is None:
        return False
    try:
        provider = get_model_provider(model_name)
    except Exception:
        return False
    return bool(_configured_providers().get(provider))


def other_family_model(model_name: str) -> str | None:
    """Return the strongest available model of a family other than *model_name*'s.

    Args:
        model_name: The model whose family to avoid.

    Returns:
        The first of :data:`SECOND_OPINION_MODELS` of another family whose
        provider is configured, or ``None`` when every configured model is
        of the same family.
    """
    family = family_of(model_name)
    return next((model for model in _strong_models() if family_of(model) != family), None)


def _strong_models() -> Iterator[str]:
    """Yield the :data:`SECOND_OPINION_MODELS` whose provider is configured, in their order."""
    for candidate in SECOND_OPINION_MODELS:
        if candidate == BEDROCK_PREFIX:
            candidate = registered_bedrock_model(cheapest=False) or candidate
        if _available(candidate):
            yield candidate


def other_worker_model(
    model_name: str,
    model_config: dict[str, Any] | None = None,
    pin_config: dict[str, Any] | None = None,
    inheriting: str = "",
) -> str | None:
    """Return a configured model that is another worker than *model_name* at its effort.

    The pin a refused same-model sub-task is told to pass.  It is the
    first of these that :func:`same_model_and_effort` tells apart from
    the caller under the configuration the sub-task would run it with:
    the strongest model of another family (:func:`other_family_model`),
    the caller's model at another effort (:func:`other_effort_model`),
    the strongest models of the caller's own family.  So an effort alias
    is not offered when *pin_config* sets the effort itself, which
    outranks the one a name stands for; a caller whose model has no
    effort alias is still given a model its own credential serves; and
    the name under which the sub-task would inherit the caller's
    configuration, and with it the caller's effort, is no way out.

    Args:
        model_name: The caller's model.
        model_config: The model configuration the caller runs with.
        pin_config: The model configuration of the sub-task's own (the
            call's option, else its SEA's setting), which it runs any
            pin with.
        inheriting: The model name under which a sub-task with no
            configuration of its own inherits the caller's: the model
            the caller was launched with.

    Returns:
        The model name, or ``None`` when nothing configured differs.
    """
    family = family_of(model_name)
    candidates = (
        other_family_model(model_name),
        other_effort_model(model_name, model_config),
        *(model for model in _strong_models() if family_of(model) == family),
    )
    for candidate in candidates:
        config = model_config if pin_config is None and candidate == inheriting else pin_config
        if candidate and not same_model_and_effort(model_name, candidate, model_config, config):
            return candidate
    return None


def fallback_of_another_family(model_name: str) -> str | None:
    """Return the fail-over model for *model_name*: another family when one is configured.

    The catalog's declared fallback (:func:`get_fallback_model`) is kept
    when it is already of another family; the same family's twin (an
    OpenRouter route of the same model) answers a provider outage but not
    a refusal, a content filter or an output cap, which the family's
    models share, so another family is preferred and the twin is the last
    resort.

    Args:
        model_name: The model that failed.

    Returns:
        A model name, or ``None`` when nothing else is configured.
    """
    declared = get_fallback_model(model_name)
    if declared and not same_family(declared, model_name):
        return declared
    return other_family_model(model_name) or declared


VISION_FAMILIES: frozenset[str] = frozenset({"anthropic", "openai", "google", "xai"})
"""Families whose current chat models accept image input (but for :data:`TEXT_ONLY_NAME_MARKS`)."""

TEXT_ONLY_NAME_MARKS: tuple[str, ...] = ("gpt-oss", "gemma-2-", "gpt-audio")
"""Name fragments of the models of a vision family that take no image.

The open-weight GPT-OSS models (``openai/gpt-oss-120b`` and its
thinking aliases, ``gpt-oss-safeguard``) and Gemma 2 accept text only,
and ``gpt-audio`` text and audio: their providers declare no image
input (OpenRouter's model metadata, checked 2026-11-01).
"""

VISION_NAME_WORDS: frozenset[str] = frozenset({"vl", "vision", "omni", "image"})
"""Name segments (between ``/``, ``-``, ``_``, ``.`` or ``:``) that mark a vision model."""


def model_supports_images(model_name: str) -> bool:
    """Return whether *model_name* can see the images and PDFs a tool returns.

    Decided from the name alone: a model of a family whose chat models
    see images (:data:`VISION_FAMILIES`) unless its name marks it as
    text-only (:data:`TEXT_ONLY_NAME_MARKS`), or a name with a segment
    that marks a vision model (:data:`VISION_NAME_WORDS`, as in
    ``Qwen/Qwen3-VL-8B-Instruct`` or ``clef-omni``).  Everything else
    (deepseek, moonshot, text qwen, meta, mistral, an unknown local
    model) is taken to be text-only: such a model is handed an image
    read as raw base64 it cannot use, so image-centred work fails on it
    where a model that sees succeeds.

    Args:
        model_name: A model name as the catalog spells it, with or
            without a route prefix.

    Returns:
        True when the model accepts image input.
    """
    name = model_name.lower()
    if any(mark in name for mark in TEXT_ONLY_NAME_MARKS):
        return False
    if family_of(model_name) in VISION_FAMILIES:
        return True
    for separator in "/-_.:":
        name = name.replace(separator, " ")
    return not VISION_NAME_WORDS.isdisjoint(name.split())
