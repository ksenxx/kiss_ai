# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``calculate_cost`` prices OpenRouter long-prompt tiers (``prompt_tiers``).

OpenRouter raises the rates of requests whose prompt is longer than a
threshold (``pricing.overrides`` with ``min_prompt_tokens``): Grok 4.x
doubles past 200K tokens, Qwen 3.7 Flash steps at 32K and 256K.  The
catalog used to drop those tiers, so the fallback price of such requests
was half (or a quarter) of what OpenRouter bills.  These tests pin the
semantics OpenRouter documents: a tier applies when the prompt is
strictly longer than its threshold, later tiers win per price, and a
price no applying tier sets keeps the entry's own value.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest

from kiss.core.models import model_info as mi


def _install(monkeypatch: pytest.MonkeyPatch, name: str, entry: dict[str, Any]) -> mi.ModelInfo:
    """Build *entry* like the loader does and register it as *name*."""
    info = mi._build_model_info_entry(entry)
    mi._normalize_entry(name, info)
    monkeypatch.setitem(mi.MODEL_INFO, name, info)
    return info


GROK = "openrouter/x-ai/grok-test"
GROK_ENTRY = {
    "context_length": 500000,
    "input_price_per_1M": 2.0,
    "output_price_per_1M": 6.0,
    "cache_read_price_per_1M": 0.5,
    "prompt_tiers": [
        {
            "min_prompt_tokens": 200000,
            "input_price_per_1M": 4.0,
            "output_price_per_1M": 12.0,
            "cache_read_price_per_1M": 1.0,
        }
    ],
}


def test_tier_applies_strictly_past_threshold_of_whole_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Input, cache reads and cache writes count toward the prompt; output does not."""
    _install(monkeypatch, GROK, GROK_ENTRY)
    at = mi.calculate_cost(GROK, 100_000, 50_000, 60_000, 40_000)
    assert at == pytest.approx((100_000 * 2.0 + 50_000 * 6.0 + 60_000 * 0.5 + 40_000 * 2.0) / 1e6)
    past = mi.calculate_cost(GROK, 100_000, 50_000, 60_000, 39_000, 1_001)
    assert past == pytest.approx(
        (100_000 * 4.0 + 50_000 * 12.0 + 60_000 * 1.0 + 40_001 * 4.0) / 1e6
    )
    # A sum over several requests cannot be tiered.
    summed = mi.calculate_cost(GROK, 300_000, 1_000, single_request=False)
    assert summed == pytest.approx((300_000 * 2.0 + 1_000 * 6.0) / 1e6)


def test_later_tiers_win_per_price_and_absent_prices_inherit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two steps (Qwen Flash style); a step that omits a price keeps the earlier value."""
    name = "openrouter/qwen/qwen-test-flash"
    _install(
        monkeypatch,
        name,
        {
            "context_length": 500000,
            "input_price_per_1M": 0.025,
            "output_price_per_1M": 0.2,
            "cache_read_price_per_1M": 0.005,
            "prompt_tiers": [
                {"min_prompt_tokens": 32000, "input_price_per_1M": 0.1, "output_price_per_1M": 0.4},
                {"min_prompt_tokens": 256000, "input_price_per_1M": 0.2},
            ],
        },
    )

    def cost(inp: int, cr: int = 0) -> float:
        return mi.calculate_cost(name, inp, 1_000, cr)

    assert cost(32_000) == pytest.approx((32_000 * 0.025 + 1_000 * 0.2) / 1e6)
    assert cost(32_001) == pytest.approx((32_001 * 0.1 + 1_000 * 0.4) / 1e6)
    # The 256K step sets only the input price: output stays at the 32K step's.
    assert cost(256_001) == pytest.approx((256_001 * 0.2 + 1_000 * 0.4) / 1e6)
    # The base's published cache-read price is inherited by both steps.
    assert cost(100_000, 200_000) == pytest.approx(
        (100_000 * 0.2 + 1_000 * 0.4 + 200_000 * 0.005) / 1e6
    )


def test_unpublished_cache_price_follows_the_tier_input_price(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without any cache-read price, cache hits bill at the tier's input price."""
    name = "openrouter/bytedance-seed/seed-test"
    _install(
        monkeypatch,
        name,
        {
            "context_length": 256000,
            "input_price_per_1M": 0.5,
            "output_price_per_1M": 4.0,
            "prompt_tiers": [
                {"min_prompt_tokens": 128000, "input_price_per_1M": 1.0, "output_price_per_1M": 6.0}
            ],
        },
    )
    cost = mi.calculate_cost(name, 100_000, 1_000, 50_000, 10_000, 5_000)
    assert cost == pytest.approx((165_000 * 1.0 + 1_000 * 6.0) / 1e6)


def test_rule_derived_cache_price_scales_with_the_tier(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenAI's 12-hour write (2.5x input, never published) follows a tier's input price."""
    name = "openrouter/openai/gpt-6.1-terra"
    tier = {
        "min_prompt_tokens": 272000,
        "input_price_per_1M": 2.0,
        "output_price_per_1M": 7.5,
        "cache_read_price_per_1M": 0.1,
        "cache_write_price_per_1M": 2.5,
    }
    entry = {
        "context_length": 500000,
        "input_price_per_1M": 1.0,
        "output_price_per_1M": 5.0,
        "cache_read_price_per_1M": 0.05,
        "cache_write_price_per_1M": 1.25,
        "prompt_tiers": [tier, {"min_prompt_tokens": 400000, "output_price_per_1M": 9.0}],
    }
    info = _install(monkeypatch, name, entry)
    assert info.cache_write_1h_price_per_1M == pytest.approx(2.5)
    assert info.prompt_tiers[0][1]["cache_write_1h_price_per_1M"] == pytest.approx(5.0)
    # A tier without its own input price gets no derived price.
    assert info.prompt_tiers[1][1] == {"output_price_per_1M": 9.0}
    assert mi.calculate_cost(name, 0, 0, 0, 0, 272_000) == pytest.approx(272_000 * 2.5 / 1e6)
    assert mi.calculate_cost(name, 0, 0, 0, 0, 272_001) == pytest.approx(272_001 * 5.0 / 1e6)
    assert mi.calculate_cost(name, 0, 1_000, 0, 0, 400_001) == pytest.approx(
        (400_001 * 5.0 + 1_000 * 9.0) / 1e6
    )


def test_tiers_replace_the_built_in_long_context_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    """With tiers, the name-based Gemini rule (2x past 200K) no longer applies."""
    name = "openrouter/google/gemini-3.1-pro-preview"
    assert mi._long_context_uplift(name) == (200000, 2.0, 1.5)
    _install(
        monkeypatch,
        name,
        {
            "context_length": 500000,
            "input_price_per_1M": 2.0,
            "output_price_per_1M": 12.0,
            "cache_read_price_per_1M": 0.2,
            "cache_write_price_per_1M": 0.375,
            "prompt_tiers": [
                {
                    "min_prompt_tokens": 300000,
                    "input_price_per_1M": 4.0,
                    "output_price_per_1M": 18.0,
                }
            ],
        },
    )
    # Between the rule's 200K and the tier's 300K: short-prompt rates.
    assert mi.calculate_cost(name, 250_000, 1_000) == pytest.approx(
        (250_000 * 2.0 + 1_000 * 12.0) / 1e6
    )
    # Past the tier: its rates; the published cache-write rate stays flat.
    assert mi.calculate_cost(name, 200_000, 1_000, 0, 100_001) == pytest.approx(
        (200_000 * 4.0 + 1_000 * 18.0 + 100_001 * 0.375) / 1e6
    )


def test_malformed_tiers_are_skipped_not_fatal(caplog: pytest.LogCaptureFixture) -> None:
    """The catalog is read at import time, so a bad tier must never raise."""
    good = {"min_prompt_tokens": 1000, "input_price_per_1M": 2}
    raw = [
        good,
        {"min_prompt_tokens": 1000, "input_price_per_1m": 2.0},
        {"min_prompt_tokens": 1000, "input_price_per_1M": -1.0},
        {"min_prompt_tokens": 1000, "input_price_per_1M": float("nan")},
        # A JSON integer too large for a float must not raise OverflowError.
        json.loads('{"min_prompt_tokens": 1000, "input_price_per_1M": 1' + "0" * 400 + "}"),
        {"min_prompt_tokens": 1000, "input_price_per_1M": True},
        {"min_prompt_tokens": 1000, "input_price_per_1M": "2"},
        {"min_prompt_tokens": 1000.0, "input_price_per_1M": 2.0},
        {"min_prompt_tokens": 0, "input_price_per_1M": 2.0},
        {"min_prompt_tokens": False, "input_price_per_1M": 2.0},
        {"min_prompt_tokens": 1000},
        {"input_price_per_1M": 2.0},
        "tier",
    ]
    with caplog.at_level(logging.WARNING, logger=mi.logger.name):
        assert mi._parse_prompt_tiers(raw) == ((1000, {"input_price_per_1M": 2.0}),)
    assert sum("malformed prompt tier" in r.getMessage() for r in caplog.records) == 12
    assert mi._parse_prompt_tiers(None) == ()
    assert mi._parse_prompt_tiers({"min_prompt_tokens": 1}) == ()
    entry = {"context_length": 1, "input_price_per_1M": 1.0, "output_price_per_1M": 1.0}
    assert mi._build_model_info_entry({**entry, "prompt_tiers": "junk"}).prompt_tiers == ()
    assert mi._build_model_info_entry(entry).prompt_tiers == ()


def test_bedrock_scaling_custom_seed_and_alias_mirroring_keep_tiers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every path that copies or rescales an entry's prices carries its tiers too."""
    info = _install(monkeypatch, GROK, GROK_ENTRY)
    scaled = mi._scaled_prices(info, 1.1)
    assert scaled.prompt_tiers == (
        (
            200000,
            {
                "input_price_per_1M": 4.4,
                "output_price_per_1M": 13.2,
                "cache_read_price_per_1M": 1.1,
            },
        ),
    )
    assert info.prompt_tiers[0][1]["input_price_per_1M"] == 4.0
    seeded = mi._seed_new_custom_entry(GROK)
    assert seeded["prompt_tiers"] == GROK_ENTRY["prompt_tiers"]
    assert mi._parse_prompt_tiers(json.loads(json.dumps(seeded))["prompt_tiers"]) == (
        info.prompt_tiers
    )
    alias = f"{GROK}-high"
    untiered = {k: v for k, v in GROK_ENTRY.items() if k != "prompt_tiers"}
    raw = {GROK: dict(GROK_ENTRY), alias: {**untiered, "alias_of": GROK}}
    mi._sync_alias_transport_flags(raw, {GROK})
    assert raw[alias]["prompt_tiers"] == GROK_ENTRY["prompt_tiers"]
    del raw[GROK]["prompt_tiers"]
    mi._sync_alias_transport_flags(raw, {GROK})
    assert "prompt_tiers" not in raw[alias]


def test_bundled_catalog_tiers_are_well_formed() -> None:
    """No bundled tier is dropped by the loader, and every alias carries its base's tiers."""
    raw = json.loads(mi.PACKAGE_MODEL_INFO_PATH.read_text(encoding="utf-8"))
    tiered = {k: e for k, e in raw.items() if isinstance(e, dict) and "prompt_tiers" in e}
    assert tiered
    for name, entry in tiered.items():
        assert len(mi._parse_prompt_tiers(entry["prompt_tiers"])) == len(entry["prompt_tiers"])
        assert name.startswith("openrouter/")
    for name, entry in raw.items():
        base = raw.get(entry.get("alias_of", "")) if isinstance(entry, dict) else None
        if base is not None:
            assert entry.get("prompt_tiers") == base.get("prompt_tiers"), name
