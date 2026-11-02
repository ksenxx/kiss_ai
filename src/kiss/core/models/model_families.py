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
"""

from __future__ import annotations

from kiss.core.models.model_info import (
    _configured_providers,
    _lookup_model_info,
    get_fallback_model,
    get_model_provider,
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
)
"""Strong models of each family, in the order a second opinion prefers them."""


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


def _available(model_name: str) -> bool:
    """Whether *model_name* is in the catalog and its provider has a credential."""
    if _lookup_model_info(model_name) is None:
        return False
    try:
        provider = get_model_provider(model_name)
    except Exception:
        return False
    return bool(_configured_providers().get(provider))


def other_family_model(model_name: str, exclude: tuple[str, ...] = ()) -> str | None:
    """Return the strongest available model of a family other than *model_name*'s.

    Args:
        model_name: The model whose family to avoid.
        exclude: Families to avoid as well (an earlier choice that failed).

    Returns:
        The first of :data:`SECOND_OPINION_MODELS` of another family whose
        provider is configured, or ``None`` when every configured model is
        of the same family.
    """
    avoided = {family_of(model_name), *exclude}
    for candidate in SECOND_OPINION_MODELS:
        if family_of(candidate) not in avoided and _available(candidate):
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
