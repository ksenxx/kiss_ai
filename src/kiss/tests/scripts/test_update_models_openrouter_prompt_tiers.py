# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: OpenRouter long-prompt price tiers flow into MODEL_INFO.json.

OpenRouter publishes conditional prices in ``pricing.overrides``; an
entry conditioned on ``min_prompt_tokens`` raises every rate of a
request whose prompt is longer than the threshold (Grok 4.x doubles past
200K tokens, Qwen Flash has two steps).  ``update_models.py`` used to
drop them, so ``calculate_cost`` priced such requests at the short-prompt
rates.  These tests drive the real ``fetch_openrouter`` ->
``compute_changes`` -> ``apply_updates_to_file`` path against an
in-process listing and a temp catalog, then price requests through the
loaded catalog.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import kiss.scripts.update_models as mod
from kiss.core.models import model_info
from kiss.tests.scripts.test_update_models_decisions import (
    _listing_server,
    _or_model,
    _point_fetch_at,
)

GROK = "openrouter/x-ai/grok-9"
GROK_PRICING = {
    "prompt": "0.000002",
    "completion": "0.000006",
    "input_cache_read": "0.0000005",
    "web_search": "0.005",
    "overrides": [
        {
            "min_prompt_tokens": 200000,
            "prompt": "0.000004",
            "completion": "0.000012",
            "input_cache_read": "0.000001",
        },
        # Time-of-day window: a condition the importer does not model.
        {"utc_start": 0, "utc_end": 3600, "prompt": "0.000001", "completion": "0.000003"},
    ],
}
GROK_TIERS = [
    {
        "min_prompt_tokens": 200000,
        "input_price_per_1M": 4.0,
        "output_price_per_1M": 12.0,
        "cache_read_price_per_1M": 1.0,
    }
]


def test_openrouter_prompt_tiers_conversion() -> None:
    """Only ``min_prompt_tokens`` entries with valid prices become tiers, in order."""
    assert mod.openrouter_prompt_tiers(GROK_PRICING) == {"prompt_tiers": GROK_TIERS}
    two_steps = {
        "overrides": [
            {"min_prompt_tokens": 32000, "prompt": "0.0000001", "audio": "0.0000003"},
            {"min_prompt_tokens": 256000.0, "prompt": "0.0000002", "input_cache_write_1h": "1e-6"},
        ]
    }
    assert mod.openrouter_prompt_tiers(two_steps) == {
        "prompt_tiers": [
            {"min_prompt_tokens": 32000, "input_price_per_1M": 0.1},
            {
                "min_prompt_tokens": 256000,
                "input_price_per_1M": 0.2,
                "cache_write_1h_price_per_1M": 1.0,
            },
        ]
    }
    skipped = [
        {"min_prompt_tokens": 1000, "utc_days": [0], "prompt": "0.000001"},
        {"min_prompt_tokens": 1000, "video_seconds": "1", "prompt": "0.000001"},
        {"min_prompt_tokens": 1000, "prompt": "-1"},
        {"min_prompt_tokens": 1000, "prompt": "n/a"},
        {"min_prompt_tokens": 1000, "prompt": "inf"},
        {"min_prompt_tokens": 1000, "prompt": "1e305"},
        {"min_prompt_tokens": 1000, "prompt": 10**400},
        {"min_prompt_tokens": 0, "prompt": "0.000001"},
        {"min_prompt_tokens": True, "prompt": "0.000001"},
        {"min_prompt_tokens": "1000", "prompt": "0.000001"},
        {"prompt": "0.000001"},
        {"min_prompt_tokens": 1000, "audio": "0.000001"},
        {"min_prompt_tokens": 1000, "prompt": ""},
        "not-a-dict",
    ]
    assert mod.openrouter_prompt_tiers({"overrides": skipped}) == {}
    assert mod.openrouter_prompt_tiers({"overrides": None}) == {}
    assert mod.openrouter_prompt_tiers({}) == {}


def test_prompt_tier_changes() -> None:
    """New, moved and withdrawn tiers register; float noise does not."""
    fetched = {"prompt_tiers": GROK_TIERS}
    assert mod._prompt_tier_changes({}, fetched) == fetched
    assert mod._prompt_tier_changes({"prompt_tiers": GROK_TIERS}, fetched) == {}
    noisy = [dict(GROK_TIERS[0], input_price_per_1M=4.0000001)]
    assert mod._prompt_tier_changes({"prompt_tiers": noisy}, fetched) == {}
    moved = [dict(GROK_TIERS[0], input_price_per_1M=3.0)]
    assert mod._prompt_tier_changes({"prompt_tiers": moved}, fetched) == fetched
    rethresholded = [dict(GROK_TIERS[0], min_prompt_tokens=128000)]
    assert mod._prompt_tier_changes({"prompt_tiers": rethresholded}, fetched) == fetched
    fewer_fields = [{k: v for k, v in GROK_TIERS[0].items() if k != "cache_read_price_per_1M"}]
    assert mod._prompt_tier_changes({"prompt_tiers": fewer_fields}, fetched) == fetched
    assert mod._prompt_tier_changes({"prompt_tiers": "junk"}, fetched) == fetched
    assert mod._prompt_tier_changes({"prompt_tiers": [True]}, fetched) == fetched
    bool_price = [dict(GROK_TIERS[0], output_price_per_1M=True)]
    assert mod._prompt_tier_changes({"prompt_tiers": bool_price}, fetched) == fetched
    huge_price = [dict(GROK_TIERS[0], output_price_per_1M=10**400)]
    assert mod._prompt_tier_changes({"prompt_tiers": huge_price}, fetched) == fetched
    # Stored tiers model_info rejects are repaired, never kept as "unchanged".
    infinite_price = json.loads(json.dumps(GROK_TIERS).replace("4.0", "1e309", 1))
    assert infinite_price[0]["input_price_per_1M"] == float("inf")
    assert mod._prompt_tier_changes({"prompt_tiers": infinite_price}, fetched) == fetched
    float_threshold = [dict(GROK_TIERS[0], min_prompt_tokens=200000.0)]
    assert mod._prompt_tier_changes({"prompt_tiers": float_threshold}, fetched) == fetched
    assert mod._prompt_tier_changes({"prompt_tiers": GROK_TIERS}, {}) == {"prompt_tiers": None}
    assert mod._prompt_tier_changes({}, {}) == {}


def test_fetch_compute_apply_writes_prompt_tiers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tiers reach base, generated aliases and new entries, price requests, and can be withdrawn."""
    base = "openrouter/x-ai/grok-4.5"  # has a generated effort-alias ladder
    catalog = tmp_path / "MODEL_INFO.json"
    current = {
        base: {
            "context_length": 500000,
            "input_price_per_1M": 2.0,
            "output_price_per_1M": 6.0,
            "cache_read_price_per_1M": 0.5,
            "fc": True,
            "emb": False,
            "gen": True,
            "thinking": "high",
        },
    }
    catalog.write_text(json.dumps(current))
    monkeypatch.setattr(mod, "MODEL_INFO_PATH", catalog)
    listing = {
        "data": [
            _or_model(mid, "0.000002", "0.000006", context_length=500000, pricing=GROK_PRICING)
            for mid in ("x-ai/grok-4.5", "x-ai/grok-9")
        ]
    }
    with _listing_server(listing) as url:
        _point_fetch_at(monkeypatch, url)
        fetched = mod.fetch_openrouter()
    assert fetched[base]["prompt_tiers"] == GROK_TIERS

    updates, new_models = mod.compute_changes(current, fetched, {}, {}, {}, {})
    assert {u["name"]: u["changes"] for u in updates} == {base: {"prompt_tiers": GROK_TIERS}}
    assert {nm["name"]: nm for nm in new_models}[GROK]["prompt_tiers"] == GROK_TIERS
    mod.apply_updates_to_file(updates, new_models, [], current, dry_run=False)
    written = json.loads(catalog.read_text())
    aliases = [name for name, entry in written.items() if entry.get("alias_of") == base]
    assert aliases
    for name in (base, *aliases, GROK):
        assert written[name]["prompt_tiers"] == GROK_TIERS

    monkeypatch.setenv("KISS_MODEL_INFO_PATH", str(catalog))
    monkeypatch.setattr(model_info, "USER_MY_MODELS_PATH", tmp_path / "MY_MODELS.json")
    loaded = model_info._load_model_info()
    monkeypatch.setattr(model_info, "MODEL_INFO", loaded)
    for name in (base, *aliases, GROK):
        # 200,000 prompt tokens is not past the threshold; 200,001 is.
        short = model_info.calculate_cost(name, 150_000, 1_000, 50_000)
        assert short == pytest.approx((150_000 * 2.0 + 1_000 * 6.0 + 50_000 * 0.5) / 1e6)
        long = model_info.calculate_cost(name, 150_000, 1_000, 50_001)
        assert long == pytest.approx((150_000 * 4.0 + 1_000 * 12.0 + 50_001 * 1.0) / 1e6)
        summed = model_info.calculate_cost(name, 150_000, 1_000, 50_001, single_request=False)
        assert summed == pytest.approx((150_000 * 2.0 + 1_000 * 6.0 + 50_001 * 0.5) / 1e6)

    # Idempotent: a second pass over the written catalog changes nothing.
    current_again = mod.get_current_model_info()
    assert current_again[base]["prompt_tiers"] == GROK_TIERS
    updates_again, _ = mod.compute_changes(current_again, fetched, {}, {}, {}, {})
    assert [u for u in updates_again if u["name"] in (base, GROK)] == []

    # OpenRouter withdraws the tier: base and aliases lose it.
    withdrawn = {name: dict(entry) for name, entry in fetched.items()}
    del withdrawn[base]["prompt_tiers"]
    updates_w, _ = mod.compute_changes(current_again, withdrawn, {}, {}, {}, {})
    assert [u["changes"] for u in updates_w if u["name"] == base] == [{"prompt_tiers": None}]
    mod.apply_updates_to_file(updates_w, [], [], current_again, dry_run=False)
    written = json.loads(catalog.read_text())
    for name in (base, *aliases):
        assert "prompt_tiers" not in written[name]
    assert written[GROK]["prompt_tiers"] == GROK_TIERS
