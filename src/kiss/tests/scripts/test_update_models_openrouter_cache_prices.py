# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: OpenRouter cache prices flow into MODEL_INFO.json.

OpenRouter's ``/api/v1/models`` publishes ``pricing.input_cache_read`` and
``pricing.input_cache_write`` per model.  ``update_models.py`` used to drop
them, leaving ``model_info._apply_cache_pricing`` to guess cache rates from
vendor prefixes — guesses that were 10x off for DeepSeek, 2.5x off for
Google and wrong for ~130 other gateway models.  These tests drive the real
``fetch_openrouter`` → ``compute_changes`` → ``apply_updates_to_file`` path
against an in-process HTTP listing and a temp catalog, then load the result
through ``model_info._load_model_info`` to check ``calculate_cost``.
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

LISTING = {
    "data": [
        _or_model(
            "deepseek/deepseek-v4-flash",
            "0.00000004144",
            "0.00000008288",
            pricing={
                "prompt": "0.00000004144",
                "completion": "0.00000008288",
                "input_cache_read": "0.000000008288",
                "input_cache_write": None,
            },
        ),
        _or_model(
            "anthropic/claude-fable-5.1",
            "0.00001",
            "0.00005",
            pricing={
                "prompt": "0.00001",
                "completion": "0.00005",
                "input_cache_read": "0.00000025",
                "input_cache_write": "0.0000125",
            },
        ),
        _or_model(
            "cohere/command-r7b-12-2024",
            "0.0000000375",
            "0.00000015",
            pricing={"prompt": "0.0000000375", "completion": "0.00000015", "input_cache_read": ""},
        ),
    ]
}


def test_openrouter_cache_prices_conversion() -> None:
    """Per-token strings become per-1M floats; missing / empty fields are omitted."""
    assert mod.openrouter_cache_prices(
        {"input_cache_read": "0.000000008288", "input_cache_write": "0.0000125"}
    ) == {"cache_read_price_per_1M": 0.008288, "cache_write_price_per_1M": 12.5}
    assert mod.openrouter_cache_prices({"input_cache_read": None, "input_cache_write": ""}) == {}
    assert mod.openrouter_cache_prices({}) == {}


def test_cache_price_changes_tolerance() -> None:
    """Only genuine differences (> $0.0001 / 1M) register; withdrawn prices become ``None``."""
    fetched = {"cache_read_price_per_1M": 0.008288, "cache_write_price_per_1M": 12.5}
    assert mod._cache_price_changes({}, fetched) == fetched
    assert mod._cache_price_changes({"cache_read_price_per_1M": 0.008288}, fetched) == {
        "cache_write_price_per_1M": 12.5
    }
    assert mod._cache_price_changes(fetched, fetched) == {}
    assert (
        mod._cache_price_changes(
            fetched, {"cache_read_price_per_1M": 0.00829, "cache_write_price_per_1M": 12.5}
        )
        == {}
    )
    assert mod._cache_price_changes(fetched, {"cache_read_price_per_1M": 0.02}) == {
        "cache_read_price_per_1M": 0.02,
        "cache_write_price_per_1M": None,
    }
    # A price OpenRouter withdrew is flagged for removal; a never-stored one is not.
    assert mod._cache_price_changes(fetched, {}) == {
        "cache_read_price_per_1M": None,
        "cache_write_price_per_1M": None,
    }
    assert mod._cache_price_changes({}, {}) == {}


def test_fetch_compute_apply_writes_cache_prices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Existing entries gain/refresh cache prices; new entries are born with them."""
    catalog = tmp_path / "MODEL_INFO.json"
    current = {
        # Stale: wrong input price, heuristic-era cache read that must be replaced,
        # and a hand-set write price OpenRouter does not publish (must be dropped).
        "openrouter/deepseek/deepseek-v4-flash": {
            "context_length": 32000,
            "input_price_per_1M": 0.048,
            "output_price_per_1M": 0.083,
            "cache_read_price_per_1M": 0.00096,
            "cache_write_price_per_1M": 0.048,
            "fc": False,
            "emb": False,
            "gen": True,
        },
        # Already correct: must produce no update at all.
        "openrouter/cohere/command-r7b-12-2024": {
            "context_length": 32000,
            "input_price_per_1M": 0.0375,
            "output_price_per_1M": 0.15,
            "fc": False,
            "emb": False,
            "gen": True,
        },
    }
    catalog.write_text(json.dumps(current))
    monkeypatch.setattr(mod, "MODEL_INFO_PATH", catalog)

    with _listing_server(LISTING) as url:
        _point_fetch_at(monkeypatch, url)
        fetched = mod.fetch_openrouter()

    assert fetched["openrouter/deepseek/deepseek-v4-flash"]["cache_read_price_per_1M"] == 0.008288
    assert "cache_write_price_per_1M" not in fetched["openrouter/deepseek/deepseek-v4-flash"]
    assert fetched["openrouter/anthropic/claude-fable-5.1"]["cache_write_price_per_1M"] == 12.5
    assert "cache_read_price_per_1M" not in fetched["openrouter/cohere/command-r7b-12-2024"]

    updates, new_models = mod.compute_changes(current, fetched, {}, {}, {}, {})
    by_name = {u["name"]: u["changes"] for u in updates}
    assert set(by_name) == {"openrouter/deepseek/deepseek-v4-flash"}
    assert by_name["openrouter/deepseek/deepseek-v4-flash"] == {
        "input_price_per_1M": 0.041,
        "cache_read_price_per_1M": 0.008288,
        "cache_write_price_per_1M": None,
    }
    # The shared listing server also serves the decisions-model listing (typesafe/jev-*).
    new_by_name = {nm["name"]: nm for nm in new_models}
    fable = new_by_name["openrouter/anthropic/claude-fable-5.1"]
    assert fable["cache_read_price_per_1M"] == 0.25
    assert fable["cache_write_price_per_1M"] == 12.5
    assert "cache_read_price_per_1M" not in new_by_name["openrouter/typesafe/jev-1.13"]

    mod.apply_updates_to_file(updates, new_models, [], current, dry_run=False)
    written = json.loads(catalog.read_text())
    ds = written["openrouter/deepseek/deepseek-v4-flash"]
    assert ds["input_price_per_1M"] == 0.041
    assert ds["cache_read_price_per_1M"] == 0.008288
    assert "cache_write_price_per_1M" not in ds
    fb = written["openrouter/anthropic/claude-fable-5.1"]
    assert fb["cache_read_price_per_1M"] == 0.25
    assert fb["cache_write_price_per_1M"] == 12.5
    assert (
        written["openrouter/cohere/command-r7b-12-2024"]
        == current["openrouter/cohere/command-r7b-12-2024"]
    )

    # The written catalog prices cache tokens exactly as OpenRouter bills them.
    monkeypatch.setenv("KISS_MODEL_INFO_PATH", str(catalog))
    monkeypatch.setenv("HOME", str(tmp_path))  # keep the developer's MY_MODELS.json out
    loaded = model_info._load_model_info()
    ds_info = loaded["openrouter/deepseek/deepseek-v4-flash"]
    assert ds_info.cache_read_price_per_1M == pytest.approx(0.008288)
    assert ds_info.cache_write_price_per_1M is None
    fb_info = loaded["openrouter/anthropic/claude-fable-5.1"]
    assert fb_info.cache_read_price_per_1M == pytest.approx(0.25)
    assert fb_info.cache_write_price_per_1M == pytest.approx(12.5)
    assert loaded["openrouter/cohere/command-r7b-12-2024"].cache_read_price_per_1M is None

    # Idempotent: a second pass over the written catalog finds nothing to change.
    monkeypatch.setattr(model_info, "MODEL_INFO", loaded)
    current_again = mod.get_current_model_info()
    assert current_again["openrouter/deepseek/deepseek-v4-flash"]["cache_read_price_per_1M"] == (
        0.008288
    )
    assert "cache_write_price_per_1M" not in current_again["openrouter/deepseek/deepseek-v4-flash"]
    updates_again, _ = mod.compute_changes(current_again, fetched, {}, {}, {}, {})
    assert [u for u in updates_again if u["name"].startswith("openrouter/")] == []


def test_get_current_model_info_reports_only_stored_cache_prices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Derived provider defaults (e.g. Google 0.1x) must not masquerade as catalog fields."""
    catalog = tmp_path / "MODEL_INFO.json"
    catalog.write_text(
        json.dumps(
            {
                "openrouter/google/gemma-9-it": {
                    "context_length": 32000,
                    "input_price_per_1M": 0.08,
                    "output_price_per_1M": 0.16,
                    "fc": False,
                    "emb": False,
                    "gen": True,
                },
                "openrouter/google/gemini-9-pro": {
                    "context_length": 32000,
                    "input_price_per_1M": 2.0,
                    "output_price_per_1M": 12.0,
                    "cache_read_price_per_1M": 0.2,
                    "fc": True,
                    "emb": False,
                    "gen": True,
                },
            }
        )
    )
    monkeypatch.setattr(mod, "MODEL_INFO_PATH", catalog)
    monkeypatch.setenv("KISS_MODEL_INFO_PATH", str(catalog))
    monkeypatch.setenv("HOME", str(tmp_path))
    loaded = model_info._load_model_info()
    for name, info in loaded.items():
        model_info._apply_cache_pricing(name, info)
    # Load-time defaults: gemma got the Google 0.1x rule, gemini kept its stored read price
    # and did NOT receive a default write price.
    assert loaded["openrouter/google/gemma-9-it"].cache_read_price_per_1M == pytest.approx(0.008)
    assert loaded["openrouter/google/gemma-9-it"].cache_write_price_per_1M == 0.0
    assert loaded["openrouter/google/gemini-9-pro"].cache_read_price_per_1M == pytest.approx(0.2)
    assert loaded["openrouter/google/gemini-9-pro"].cache_write_price_per_1M is None
    monkeypatch.setattr(model_info, "MODEL_INFO", loaded)
    current = mod.get_current_model_info()
    assert "cache_read_price_per_1M" not in current["openrouter/google/gemma-9-it"]
    assert "cache_write_price_per_1M" not in current["openrouter/google/gemma-9-it"]
    assert current["openrouter/google/gemini-9-pro"]["cache_read_price_per_1M"] == 0.2
    assert "cache_write_price_per_1M" not in current["openrouter/google/gemini-9-pro"]
