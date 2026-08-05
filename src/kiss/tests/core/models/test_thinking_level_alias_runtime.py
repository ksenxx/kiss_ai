# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for runtime resolution of ``-{thinking_level}`` aliases.

``update_models.py`` materializes one catalog entry per supported
``reasoning_effort`` level (``gpt-5.6-sol-low`` … ``gpt-5.6-sol-xhigh``).
At runtime those synthetic names must:

* resolve to the base provider model id on the wire
  (:func:`kiss.core.models.openai_compatible_model._provider_model_name`),
* carry their own ``thinking`` level so the request gets the matching
  ``reasoning_effort``
  (:func:`kiss.core.models.openai_compatible_model._model_thinking_level`),
* cost exactly the same as the base model in every pricing dimension.

Real upstream models whose names merely end in ``-high`` (e.g.
``openrouter/openai/o3-mini-high``) must NOT be rewritten — they carry no
``alias_of`` marker in the catalog.

These tests run against the real bundled ``MODEL_INFO`` catalog; nothing
is mocked.
"""

from __future__ import annotations

import pytest

from kiss.core.models.model_info import (
    MODEL_INFO,
    _strip_thinking_alias,
    calculate_cost,
)
from kiss.core.models.openai_compatible_model import (
    _model_thinking_level,
    _provider_model_name,
)

ALL_LEVELS = ("low", "medium", "high", "xhigh")

KNOWN_LEVELS = ("low", "medium", "high", "xhigh", "max")
"""Every level across all vendor scales: the OpenAI ladder plus the
Moonshot/Kimi top level ``max`` (see ``_thinking_scale_for`` in
``update_models.py``)."""


def _alias_pairs() -> list[tuple[str, str]]:
    """Return every ``(base_name, alias_name)`` pair marked in MODEL_INFO."""
    pairs = []
    for name, info in MODEL_INFO.items():
        if info.alias_of:
            pairs.append((info.alias_of, name))
    return pairs


def test_catalog_ships_marked_aliases_for_every_level() -> None:
    """gpt-5.6-sol must expose one alias per thinking level (user example)."""
    for level in ALL_LEVELS:
        name = f"gpt-5.6-sol-{level}"
        info = MODEL_INFO.get(name)
        assert info is not None, f"{name} missing from MODEL_INFO"
        assert info.thinking == level
        assert info.alias_of == "gpt-5.6-sol"


def test_every_marked_alias_has_consistent_base() -> None:
    """Every alias must point at an existing base and share its pricing."""
    pairs = _alias_pairs()
    assert pairs, "MODEL_INFO has no generated thinking-level aliases"
    for base_name, alias_name in pairs:
        assert base_name in MODEL_INFO, f"{alias_name} orphaned: no {base_name}"
        base, alias = MODEL_INFO[base_name], MODEL_INFO[alias_name]
        assert alias.context_length == base.context_length
        assert alias.input_price_per_1M == base.input_price_per_1M
        assert alias.output_price_per_1M == base.output_price_per_1M
        assert alias.is_function_calling_supported == base.is_function_calling_supported
        assert alias.thinking in KNOWN_LEVELS


def test_strip_thinking_alias_strips_marked_level_suffixes() -> None:
    """Marked aliases map back to the base name for provider requests."""
    for level in ALL_LEVELS:
        assert _strip_thinking_alias(f"gpt-5.6-sol-{level}") == "gpt-5.6-sol"


def test_strip_thinking_alias_resolves_full_openrouter_keys() -> None:
    """Full openrouter catalog keys resolve to the recorded base key."""
    assert (
        _strip_thinking_alias("openrouter/openai/gpt-5.6-sol-medium")
        == "openrouter/openai/gpt-5.6-sol"
    )
    assert (
        _strip_thinking_alias("openrouter/~openai/gpt-latest-high")
        == "openrouter/~openai/gpt-latest"
    )


def test_strip_thinking_alias_requires_exact_catalog_key() -> None:
    """Only exact ``alias_of``-marked keys are rewritten — never lookalikes.

    A custom (e.g. local base_url) model that merely shares the ``-high``
    tail with some catalog alias must keep its name on the wire; the old
    fuzzy tail-scan lookup rewrote it and silently changed which model was
    invoked.
    """
    assert MODEL_INFO["openrouter/~openai/gpt-latest-high"].alias_of
    assert "gpt-latest-high" not in MODEL_INFO
    assert _strip_thinking_alias("gpt-latest-high") == "gpt-latest-high"
    assert _provider_model_name("gpt-latest-high") == "gpt-latest-high"


def test_strip_thinking_alias_keeps_real_upstream_high_models() -> None:
    """o3-mini-high / o4-mini-high are real models and must not be rewritten."""
    assert _strip_thinking_alias("openai/o3-mini-high") == "openai/o3-mini-high"
    assert _strip_thinking_alias("openai/o4-mini-high") == "openai/o4-mini-high"


def test_strip_thinking_alias_keeps_unknown_suffixed_names() -> None:
    """Names absent from the catalog are passed through untouched."""
    assert _strip_thinking_alias("acme/custom-high") == "acme/custom-high"
    assert _strip_thinking_alias("acme/custom-medium") == "acme/custom-medium"
    assert _strip_thinking_alias("acme/custom-low") == "acme/custom-low"
    assert _strip_thinking_alias("plain-model") == "plain-model"


def test_strip_thinking_alias_always_strips_xhigh() -> None:
    """-xhigh has no upstream collisions and strips unconditionally."""
    assert _strip_thinking_alias("anything-at-all-xhigh") == "anything-at-all"


def test_provider_model_name_maps_alias_to_base_wire_id() -> None:
    """The wire model id for every alias is the base provider id."""
    assert _provider_model_name("gpt-5.6-sol-high") == "gpt-5.6-sol"
    assert _provider_model_name("gpt-5.6-sol-low") == "gpt-5.6-sol"
    assert (
        _provider_model_name("openrouter/openai/gpt-5.6-sol-medium") == "openai/gpt-5.6-sol"
    )
    assert (
        _provider_model_name("openrouter/openai/o3-mini-high") == "openai/o3-mini-high"
    )


def test_model_thinking_level_reflects_alias_level() -> None:
    """Each alias must request its own reasoning_effort level."""
    for level in ALL_LEVELS:
        assert _model_thinking_level(f"gpt-5.6-sol-{level}") == level
    assert _model_thinking_level("gpt-5.6-sol") == "high"


_TOKEN_PROFILES: tuple[tuple[int, int, int, int, int], ...] = (
    (1_000, 500, 0, 0, 0),
    (1_000, 1_000, 1_000, 0, 0),
    (1_000_000, 1_000_000, 1_000_000, 0, 0),
    (0, 0, 0, 1_000_000, 0),
    (0, 0, 0, 0, 1_000_000),
    (250_000, 250_000, 250_000, 0, 0),
)


@pytest.mark.parametrize("profile", _TOKEN_PROFILES)
def test_every_alias_costs_exactly_like_its_base(
    profile: tuple[int, int, int, int, int],
) -> None:
    """Aliases route to the base provider model, so costs must be identical."""
    inp, out, cread, cwrite, cwrite1h = profile
    for base_name, alias_name in _alias_pairs():
        base_cost = calculate_cost(base_name, inp, out, cread, cwrite, cwrite1h)
        alias_cost = calculate_cost(alias_name, inp, out, cread, cwrite, cwrite1h)
        assert alias_cost == pytest.approx(base_cost), (
            f"{alias_name} cost {alias_cost} != base {base_name} cost {base_cost} "
            f"for profile {profile}"
        )
