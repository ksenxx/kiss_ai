# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""OpenRouter cache-read prices reach the catalog at or below the input price.

The 2026-11-20 refresh stored ``openrouter/~deepseek/deepseek-pro-latest``
at $0.2066/M input and $0.21/M cached input, as OpenRouter's headline had
it, and ``calculate_cost`` priced a token read from cache above the fresh
input token it replaces.  These tests pin the cap on the price-sheet
conversion and drive the real ``fetch_openrouter`` -> ``compute_changes``
-> ``apply_updates_to_file`` path against an in-process listing, then
check the written catalog through ``calculate_cost``: at no prompt size
does a cached token cost more than a fresh one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import kiss.scripts.update_models as mod
from kiss.core.models import model_info
from kiss.tests.scripts.test_update_models_decisions import (
    _listing_server,
    _or_model,
    _point_fetch_at,
)

# OpenRouter's headline of ``~deepseek/deepseek-pro-latest`` on 2026-11-20 (USD per token).
ALIAS_PRICING = {
    "prompt": "0.0000002066",
    "completion": "0.000008",
    "input_cache_read": "0.00000021",
}
# A cache read above the input price of the long-prompt tier only.
TIERED_PRICING = {
    "prompt": "0.000001",
    "completion": "0.000002",
    "input_cache_read": "0.0000001",
    "overrides": [
        {
            "min_prompt_tokens": 200000,
            "prompt": "0.000002",
            "completion": "0.000004",
            "input_cache_read": "0.0000025",
        }
    ],
}
SOUND_PRICING = {"prompt": "0.000002", "completion": "0.000008", "input_cache_read": "0.0000002"}

ALIAS = "openrouter/~deepseek/deepseek-pro-latest"
TIERED = "openrouter/acme/tiered-9"
SOUND = "openrouter/acme/sound-9"
SIZES = (1, 1000, 199_999, 200_000, 200_001, 272_001, 1_000_000)


def _uncapped(pricing: dict[str, Any]) -> dict[str, Any]:
    """The sheet as OpenRouter lists it: every published price, verbatim."""
    return {
        "input_price_per_1M": round(float(pricing["prompt"]) * 1_000_000, 6),
        "output_price_per_1M": round(float(pricing["completion"]) * 1_000_000, 6),
        **mod.openrouter_cache_prices(pricing),
        **mod.openrouter_prompt_tiers(pricing),
    }


def test_a_headline_cache_read_above_the_input_price_becomes_the_input_price() -> None:
    """The 2026-11-20 sheet: $0.21/M cached input against $0.2066/M input."""
    assert _uncapped(ALIAS_PRICING)["cache_read_price_per_1M"] == 0.21
    assert mod._openrouter_price_fields(ALIAS_PRICING) == {
        "input_price_per_1M": 0.2066,
        "output_price_per_1M": 8.0,
        "cache_read_price_per_1M": 0.2066,
    }
    # A free input: nothing read from cache is billed either.
    free = {"prompt": "0", "completion": "0", "input_cache_read": "0.0000001"}
    assert mod._openrouter_price_fields(free)["cache_read_price_per_1M"] == 0.0


@pytest.mark.parametrize(
    "pricing",
    [
        SOUND_PRICING,
        # a free model without a cache-read price gains none
        {"prompt": "0", "completion": "0"},
        # equal to the input price (the alias's sheet on 2026-11-21)
        {"prompt": "0.0000000999", "completion": "0.000008", "input_cache_read": "0.0000000999"},
        # no cache-read price: calculate_cost bills the input price, no key is invented
        {"prompt": "0.0000000375", "completion": "0.00000015", "input_cache_read": ""},
        # a cache WRITE above the input price is a real price (Anthropic's 1.25x)
        {
            "prompt": "0.000004",
            "completion": "0.00002",
            "input_cache_read": "0.0000002",
            "input_cache_write": "0.000005",
        },
        # sound tiers: each long-prompt cache read below its tier's input
        {
            "prompt": "0.000000195",
            "completion": "0.000000975",
            "input_cache_read": "0.000000039",
            "overrides": [
                {
                    "min_prompt_tokens": 32000,
                    "prompt": "0.000000325",
                    "completion": "0.000001625",
                    "input_cache_read": "0.000000065",
                },
                {
                    "min_prompt_tokens": 128000,
                    "prompt": "0.00000052",
                    "completion": "0.0000026",
                    "input_cache_read": "0.000000104",
                },
            ],
        },
    ],
)
def test_a_sound_sheet_is_stored_as_listed(pricing: dict[str, Any]) -> None:
    assert mod._openrouter_price_fields(pricing) == _uncapped(pricing)


def test_a_router_listed_without_a_price_is_left_alone() -> None:
    """OpenRouter lists ``openrouter/auto`` at "-1" per token: no price, so nothing to cap at."""
    assert mod._openrouter_price_fields({"prompt": "-1", "completion": "-1"}) == {
        "input_price_per_1M": -1_000_000.0,
        "output_price_per_1M": -1_000_000.0,
    }
    listed = {"prompt": "-1", "completion": "-1", "input_cache_read": "0.0000001"}
    assert mod._openrouter_price_fields(listed) == _uncapped(listed)


def test_cache_reads_are_capped_in_the_price_regime_they_apply_in() -> None:
    """Base and long-prompt tiers: each cache read is held to the input in force with it."""
    # The tier's own cache read above the tier's own input; the base sheet is sound.
    sheet = mod._openrouter_price_fields(TIERED_PRICING)
    assert sheet["cache_read_price_per_1M"] == 0.1
    assert sheet["prompt_tiers"] == [
        {
            "min_prompt_tokens": 200000,
            "input_price_per_1M": 2.0,
            "output_price_per_1M": 4.0,
            "cache_read_price_per_1M": 2.0,
        }
    ]

    # A tier that re-prices the cache read alone keeps the base input price: capped at it.
    read_only_tier = {
        "prompt": "0.000001",
        "completion": "0.000002",
        "overrides": [
            {"min_prompt_tokens": 200000, "completion": "0.000004", "input_cache_read": "0.0000015"}
        ],
    }
    sheet = mod._openrouter_price_fields(read_only_tier)
    assert "cache_read_price_per_1M" not in sheet
    assert sheet["prompt_tiers"][0]["cache_read_price_per_1M"] == 1.0

    # A tier that lowers the input below the base's cache read, and publishes none of its
    # own: the cache read in force there is the base's, lowered where it is set.
    cheaper_long_prompts = {
        "prompt": "0.000001",
        "completion": "0.000002",
        "input_cache_read": "0.0000005",
        "overrides": [
            {"min_prompt_tokens": 200000, "prompt": "0.0000003", "completion": "0.000001"}
        ],
    }
    sheet = mod._openrouter_price_fields(cheaper_long_prompts)
    assert sheet["cache_read_price_per_1M"] == 0.3
    assert sheet["prompt_tiers"] == [
        {"min_prompt_tokens": 200000, "input_price_per_1M": 0.3, "output_price_per_1M": 1.0}
    ]

    # Tiers listed out of threshold order apply in listing order, later ones winning per
    # key: above 400K the 200K tier's prices are in force, so only its cache read is high.
    out_of_order = {
        "prompt": "0.000001",
        "completion": "0.000002",
        "input_cache_read": "0.0000001",
        "overrides": [
            {
                "min_prompt_tokens": 400000,
                "prompt": "0.000004",
                "completion": "0.000008",
                "input_cache_read": "0.0000004",
            },
            {
                "min_prompt_tokens": 200000,
                "prompt": "0.000002",
                "completion": "0.000004",
                "input_cache_read": "0.000003",
            },
        ],
    }
    sheet = mod._openrouter_price_fields(out_of_order)
    assert [tier["cache_read_price_per_1M"] for tier in sheet["prompt_tiers"]] == [0.4, 2.0]
    assert sheet["cache_read_price_per_1M"] == 0.1


def _cache_hits_above_input(names: list[str], sizes: tuple[int, ...]) -> list[tuple[str, int]]:
    """(name, prompt size) pairs ``calculate_cost`` bills more for from cache than fresh."""
    return [
        (name, size)
        for name in names
        for size in sizes
        if model_info.calculate_cost(name, 0, 0, num_cache_read_tokens=size)
        > model_info.calculate_cost(name, size, 0) * (1 + 1e-12)
    ]


def test_refresh_heals_the_stored_entry_and_no_cache_hit_bills_above_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The entry of 2026-11-20 is repaired by the next refresh, whatever the listing says."""
    listing = {
        "data": [
            _or_model("~deepseek/deepseek-pro-latest", "0", "0", pricing=ALIAS_PRICING),
            _or_model("acme/tiered-9", "0", "0", pricing=TIERED_PRICING),
            _or_model("acme/sound-9", "0", "0", pricing=SOUND_PRICING),
        ]
    }
    flags = {"fc": True, "emb": False, "gen": True}
    current = {
        ALIAS: {
            "context_length": 32000,
            "input_price_per_1M": 0.2066,
            "output_price_per_1M": 8.0,
            "cache_read_price_per_1M": 0.21,
            **flags,
        },
        SOUND: {
            "context_length": 32000,
            "input_price_per_1M": 2.0,
            "output_price_per_1M": 8.0,
            "cache_read_price_per_1M": 0.2,
            **flags,
        },
    }
    catalog = tmp_path / "MODEL_INFO.json"
    catalog.write_text(json.dumps(current))
    monkeypatch.setattr(mod, "MODEL_INFO_PATH", catalog)
    monkeypatch.setenv("KISS_MODEL_INFO_PATH", str(catalog))
    monkeypatch.setattr(model_info, "USER_MY_MODELS_PATH", tmp_path / "MY_MODELS.json")

    # The stored entry bills a cache hit above a fresh input token at every prompt size.
    monkeypatch.setattr(model_info, "MODEL_INFO", model_info._load_model_info())
    assert _cache_hits_above_input([ALIAS, SOUND], SIZES) == [(ALIAS, size) for size in SIZES]

    with _listing_server(listing) as url:
        _point_fetch_at(monkeypatch, url)
        fetched = mod.fetch_openrouter()
    updates, new_models = mod.compute_changes(current, fetched, {}, {}, {})
    assert {u["name"]: u["changes"] for u in updates} == {
        ALIAS: {"cache_read_price_per_1M": 0.2066}
    }
    assert TIERED in {candidate["name"] for candidate in new_models}

    mod.apply_updates_to_file(updates, new_models, [], current, dry_run=False)
    written = json.loads(catalog.read_text())
    assert (
        written[ALIAS]["cache_read_price_per_1M"] == written[ALIAS]["input_price_per_1M"] == 0.2066
    )
    assert written[SOUND] == current[SOUND]
    assert written[TIERED]["prompt_tiers"][0]["cache_read_price_per_1M"] == 2.0

    monkeypatch.setattr(model_info, "MODEL_INFO", model_info._load_model_info())
    assert _cache_hits_above_input([ALIAS, TIERED, SOUND], SIZES) == []
    # Capped, not dropped or discounted: a cache hit costs what the input costs.
    assert model_info.calculate_cost(ALIAS, 0, 0, num_cache_read_tokens=1_000_000) == (
        pytest.approx(0.2066)
    )
    assert model_info.calculate_cost(TIERED, 0, 0, num_cache_read_tokens=200_001) == (
        pytest.approx(2.0 * 200_001 / 1_000_000)
    )
    assert model_info.calculate_cost(TIERED, 0, 0, num_cache_read_tokens=200_000) == (
        pytest.approx(0.1 * 200_000 / 1_000_000)
    )

    # Idempotent: a second pass over the written catalog finds nothing to change.
    again, _new = mod.compute_changes(mod.get_current_model_info(), fetched, {}, {}, {})
    assert [u for u in again if u["name"] in (ALIAS, TIERED, SOUND)] == []


def test_bundled_catalog_bills_no_cache_hit_above_input_at_any_prompt_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every bundled entry, through ``calculate_cost``, around every tier threshold."""
    monkeypatch.delenv("KISS_MODEL_INFO_PATH", raising=False)
    monkeypatch.setattr(model_info, "USER_MY_MODELS_PATH", tmp_path / "MY_MODELS.json")
    loaded = model_info._load_model_info()
    monkeypatch.setattr(model_info, "MODEL_INFO", loaded)
    bad = []
    for name, info in sorted(loaded.items()):
        sizes = {1, 1000, 128_000, 200_000, 200_001, 272_000, 272_001, 1_000_000}
        for threshold, _prices in info.prompt_tiers:
            sizes |= {threshold, threshold + 1}
        bad += _cache_hits_above_input([name], tuple(sorted(sizes)))
    assert bad == []
