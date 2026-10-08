# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``AnthropicModel`` effort levels: ``reasoning_effort`` and ``-{level}`` aliases.

Anthropic's Messages API takes the effort level as ``output_config.effort``
(``low`` / ``medium`` / ``high`` / ``xhigh`` / ``max``) and only knows the
base model ids — ``claude-opus-5-5-medium`` answers 404
``not_found_error``.  The adapter therefore has to:

* send the alias's base id (``alias_of``) as ``model``;
* turn the alias's catalog ``thinking`` level, or an explicit
  ``model_config["reasoning_effort"]``, into ``output_config.effort``;
* send NO effort for a bare base name so it costs what the raw API call
  costs (the vendor default is ``medium`` on Opus 5.5, ``high`` elsewhere);
* keep a caller-supplied native ``output_config.effort`` untouched.

Request kwargs are inspected through ``_build_create_kwargs``; no network
call is made.  The catalog checks run against the bundled ``MODEL_INFO``.
"""

from __future__ import annotations

from typing import Any

import pytest

from kiss.core.models.anthropic_model import (
    ANTHROPIC_EFFORT_LEVELS,
    AnthropicModel,
    _effort_level_for,
)
from kiss.core.models.model_info import MODEL_INFO, calculate_cost

_PROMPT = "Say hello in one word."


def _kwargs(name: str, model_config: dict[str, Any] | None = None) -> dict[str, Any]:
    m = AnthropicModel(name, model_config=model_config or {}, api_key="sk-test")
    m.initialize(_PROMPT)
    return m._build_create_kwargs()


# ---------------------------------------------------------------------------
# Catalog shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("base", ["claude-opus-5-5", "claude-sonnet-5-5"])
def test_catalog_ships_five_effort_aliases(base: str) -> None:
    """The user's ask: low / medium / high / xhigh / max for Opus 5.5 and Sonnet 5.5."""
    for level in ANTHROPIC_EFFORT_LEVELS:
        info = MODEL_INFO[f"{base}-{level}"]
        assert info.alias_of == base
        assert info.thinking == level
        assert info.input_price_per_1M == MODEL_INFO[base].input_price_per_1M
        assert info.output_price_per_1M == MODEL_INFO[base].output_price_per_1M


def test_catalog_respects_per_generation_ladders() -> None:
    assert "claude-opus-4-6-max" in MODEL_INFO
    assert "claude-opus-4-6-xhigh" not in MODEL_INFO
    assert "claude-sonnet-4-6-xhigh" not in MODEL_INFO
    assert "claude-opus-4-5-high" in MODEL_INFO
    assert "claude-opus-4-5-max" not in MODEL_INFO
    for name in MODEL_INFO:
        if name.startswith(("claude-haiku-4-5", "claude-sonnet-4-5")):
            assert MODEL_INFO[name].alias_of is None, name


def test_default_model_name_resolves_in_catalog() -> None:
    """``get_default_model()`` returns ``claude-opus-5-5-medium`` when only
    ``ANTHROPIC_API_KEY`` is set; that name used to be absent (→ 404)."""
    assert MODEL_INFO["claude-opus-5-5-medium"].alias_of == "claude-opus-5-5"


# ---------------------------------------------------------------------------
# Request shape
# ---------------------------------------------------------------------------


def test_alias_sends_base_id_and_effort() -> None:
    kw = _kwargs("claude-opus-5-5-medium")
    assert kw["model"] == "claude-opus-5-5"
    assert kw["output_config"] == {"effort": "medium"}
    assert "reasoning_effort" not in kw
    # Thinking heuristics must see the base id, not the alias.
    assert kw["thinking"] == {"type": "adaptive", "display": "summarized"}


@pytest.mark.parametrize("level", ANTHROPIC_EFFORT_LEVELS)
def test_every_level_alias_maps_to_output_config(level: str) -> None:
    kw = _kwargs(f"claude-sonnet-5-5-{level}")
    assert kw["model"] == "claude-sonnet-5-5"
    assert kw["output_config"]["effort"] == level


def test_bare_base_sends_no_effort() -> None:
    kw = _kwargs("claude-opus-5-5")
    assert kw["model"] == "claude-opus-5-5"
    assert "output_config" not in kw
    # The base entry's own ``thinking`` ("high", the split-writer's cap
    # marker) must not leak into the request as a default.
    assert MODEL_INFO["claude-opus-5-5"].thinking == "high"
    assert _effort_level_for("claude-opus-5-5", {}) is None


def test_explicit_reasoning_effort_on_base() -> None:
    kw = _kwargs("claude-opus-5-5", {"reasoning_effort": "low"})
    assert kw["output_config"] == {"effort": "low"}
    assert "reasoning_effort" not in kw


def test_explicit_reasoning_effort_overrides_alias_level() -> None:
    kw = _kwargs("claude-opus-5-5-medium", {"reasoning_effort": "max"})
    assert kw["output_config"] == {"effort": "max"}


def test_native_output_config_effort_wins() -> None:
    kw = _kwargs("claude-opus-5-5-medium", {"output_config": {"effort": "high"}})
    assert kw["output_config"] == {"effort": "high"}


def test_native_output_config_other_keys_are_merged() -> None:
    kw = _kwargs("claude-opus-5-5-high", {"output_config": {"format": {"type": "json"}}})
    assert kw["output_config"] == {"format": {"type": "json"}, "effort": "high"}


def test_extended_thinking_alias_keeps_budget_thinking() -> None:
    """Opus 4.5 uses ``thinking.type=enabled`` + budget; effort rides along."""
    kw = _kwargs("claude-opus-4-5-high")
    assert kw["model"] == "claude-opus-4-5"
    assert kw["output_config"] == {"effort": "high"}
    assert kw["thinking"]["type"] == "enabled"


# ---------------------------------------------------------------------------
# Default max_tokens ceiling (regression: Opus 4.5 is capped at 64000)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("claude-opus-4-5", 64000),
        ("claude-opus-4-5-20251101", 64000),
        ("claude-opus-4-6", 65536),
        ("claude-opus-4-7", 65536),
        ("claude-opus-5-5", 65536),
        ("claude-opus-5-5-xhigh", 65536),
        ("claude-sonnet-5-5", 64000),
        ("claude-sonnet-4-6", 64000),
    ],
)
def test_default_max_tokens_ceiling(name: str, expected: int) -> None:
    assert _kwargs(name)["max_tokens"] == expected


def test_user_max_tokens_is_kept() -> None:
    assert _kwargs("claude-opus-5-5-max", {"max_tokens": 4096})["max_tokens"] == 4096


# ---------------------------------------------------------------------------
# Cost parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("level", ANTHROPIC_EFFORT_LEVELS)
def test_alias_costs_the_same_as_base(level: str) -> None:
    base = calculate_cost("claude-opus-5-5", 1000, 500, 200, 100)
    alias = calculate_cost(f"claude-opus-5-5-{level}", 1000, 500, 200, 100)
    assert base > 0
    assert alias == base


# ---------------------------------------------------------------------------
# OpenRouter fallback twin
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("level", ANTHROPIC_EFFORT_LEVELS)
def test_alias_keeps_openrouter_twin(level: str) -> None:
    from kiss.core.models.model_info import openrouter_twin

    assert openrouter_twin("claude-opus-5-5") == "openrouter/anthropic/claude-opus-5.5"
    assert openrouter_twin(f"claude-opus-5-5-{level}") == "openrouter/anthropic/claude-opus-5.5"


def test_openai_alias_twin_keeps_level_when_openrouter_has_it() -> None:
    from kiss.core.models.model_info import openrouter_twin

    assert openrouter_twin("gpt-6.1-sol-medium") == "openrouter/openai/gpt-6.1-sol-medium"
    assert openrouter_twin("openrouter/openai/gpt-6.1-sol-medium") is None


def test_harbor_prefixed_alias_keeps_openrouter_twin() -> None:
    from kiss.core.models.model_info import openrouter_twin

    assert openrouter_twin("anthropic/claude-opus-5-5-medium") == (
        "openrouter/anthropic/claude-opus-5.5"
    )
    assert openrouter_twin("openai/gpt-6.1-sol-xhigh") == "openrouter/openai/gpt-6.1-sol-xhigh"
    assert openrouter_twin("openrouter/anthropic/claude-opus-5.5") is None
