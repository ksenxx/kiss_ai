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
        "input_price_per_1M": 0.04144,
        "output_price_per_1M": 0.08288,
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
    assert ds["input_price_per_1M"] == 0.04144
    assert ds["output_price_per_1M"] == 0.08288
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
    # Keep the developer's MY_MODELS.json out: its path is fixed at import,
    # so HOME alone does not isolate the load.
    monkeypatch.setattr(model_info, "USER_MY_MODELS_PATH", tmp_path / "MY_MODELS.json")
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


IMAGE_LISTING = {
    "data": [
        _or_model(
            "google/gemini-nano-banana-9",
            "0.0000015",
            "0.0000075",
            context_length=65536,
            pricing={
                "prompt": "0.0000015",
                "completion": "0.0000075",
                "image_output": "0.00003",
                "input_cache_read": "0.00000015",
                "web_search": "0.014",
            },
        ),
    ]
}


def test_image_output_price_flows_to_openrouter_and_direct_gemini_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A new image model is born with OpenRouter's ``image_output`` rate, twice.

    The refresh of 2026-10-24 added ``gemini-nano-banana-2.1`` from its
    OpenRouter twin with only the text prices, so a 1120-token image was
    billed $0.0084 instead of Google's $0.0336.  Both the ``openrouter/``
    entry and the direct Gemini entry priced from it must carry
    ``image_output_price_per_1M``; the direct entry must not copy the
    twin's cache rate (``model_info`` derives Google's own).
    """
    catalog = tmp_path / "MODEL_INFO.json"
    catalog.write_text("{}")
    monkeypatch.setattr(mod, "MODEL_INFO_PATH", catalog)
    with _listing_server(IMAGE_LISTING) as url:
        _point_fetch_at(monkeypatch, url)
        fetched = mod.fetch_openrouter()
    assert fetched["openrouter/google/gemini-nano-banana-9"]["image_output_price_per_1M"] == 30.0

    gemini = {"gemini-nano-banana-9": {"context_length": 65536, "source": "gemini"}}
    updates, new_models = mod.compute_changes({}, fetched, {}, gemini, {}, {})
    assert updates == []
    new_by_name = {nm["name"]: nm for nm in new_models}
    routed = new_by_name["openrouter/google/gemini-nano-banana-9"]
    assert routed["image_output_price_per_1M"] == 30.0
    direct = new_by_name["gemini-nano-banana-9"]
    assert direct["input_price_per_1M"] == 1.5
    assert direct["output_price_per_1M"] == 7.5
    assert direct["image_output_price_per_1M"] == 30.0
    assert direct["needs_pricing"] is False
    assert "cache_read_price_per_1M" not in direct

    mod.apply_updates_to_file(updates, new_models, [], {}, dry_run=False)
    written = json.loads(catalog.read_text())
    assert written["gemini-nano-banana-9"]["image_output_price_per_1M"] == 30.0
    assert "cache_read_price_per_1M" not in written["gemini-nano-banana-9"]
    assert written["openrouter/google/gemini-nano-banana-9"]["image_output_price_per_1M"] == 30.0
    assert written["openrouter/google/gemini-nano-banana-9"]["cache_read_price_per_1M"] == 0.15

    monkeypatch.setenv("KISS_MODEL_INFO_PATH", str(catalog))
    monkeypatch.setattr(model_info, "USER_MY_MODELS_PATH", tmp_path / "MY_MODELS.json")
    loaded = model_info._load_model_info()
    monkeypatch.setattr(model_info, "MODEL_INFO", loaded)
    # A 1K image (1120 tokens) costs Google's advertised $0.0336, not $0.0084.
    assert model_info.calculate_cost(
        "gemini-nano-banana-9", 0, 0, num_image_output_tokens=1120
    ) == pytest.approx(0.0336)
    assert model_info.calculate_cost(
        "openrouter/google/gemini-nano-banana-9", 0, 0, num_image_output_tokens=1120
    ) == pytest.approx(0.0336)

    # Idempotent: the written catalog reports the field back and nothing changes.
    current_again = mod.get_current_model_info()
    assert current_again["gemini-nano-banana-9"]["image_output_price_per_1M"] == 30.0
    updates_again, new_again = mod.compute_changes(current_again, fetched, {}, gemini, {}, {})
    assert [u for u in updates_again if "nano-banana" in u["name"]] == []
    assert [nm for nm in new_again if "nano-banana" in nm["name"]] == []


def test_backfill_of_unpriced_gemini_image_model_copies_image_rate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A direct image entry born ``NEW: needs pricing`` gains the image rate on backfill.

    When the model is discovered before its OpenRouter twin lists it, the
    catalog holds $0/$0; the next refresh prices it from the twin through
    the ``openrouter-xref`` path, which must carry the image rate too or the
    entry stays image-blind for good (once priced it is skipped).
    """
    catalog = tmp_path / "MODEL_INFO.json"
    current = {
        "gemini-nano-banana-9": {
            "context_length": 65536,
            "input_price_per_1M": 0.0,
            "output_price_per_1M": 0.0,
            "fc": True,
            "emb": False,
            "gen": True,
            "comment": "NEW: needs pricing",
        },
    }
    catalog.write_text(json.dumps(current))
    monkeypatch.setattr(mod, "MODEL_INFO_PATH", catalog)
    with _listing_server(IMAGE_LISTING) as url:
        _point_fetch_at(monkeypatch, url)
        fetched = mod.fetch_openrouter()

    gemini = {"gemini-nano-banana-9": {"context_length": 65536, "source": "gemini"}}
    updates, new_models = mod.compute_changes(current, fetched, {}, gemini, {}, {})
    by_name = {u["name"]: u for u in updates}
    assert by_name["gemini-nano-banana-9"]["source"] == "openrouter-xref"
    assert by_name["gemini-nano-banana-9"]["changes"] == {
        "input_price_per_1M": 1.5,
        "output_price_per_1M": 7.5,
        "image_output_price_per_1M": 30.0,
    }

    mod.apply_updates_to_file(updates, new_models, [], current, dry_run=False)
    written = json.loads(catalog.read_text())["gemini-nano-banana-9"]
    assert written["image_output_price_per_1M"] == 30.0
    assert written["comment"] == "NEW"
    assert "cache_read_price_per_1M" not in written

    monkeypatch.setenv("KISS_MODEL_INFO_PATH", str(catalog))
    monkeypatch.setattr(model_info, "USER_MY_MODELS_PATH", tmp_path / "MY_MODELS.json")
    monkeypatch.setattr(model_info, "MODEL_INFO", model_info._load_model_info())
    assert model_info.calculate_cost(
        "gemini-nano-banana-9", 0, 0, num_image_output_tokens=1120
    ) == pytest.approx(0.0336)


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
    monkeypatch.setattr(model_info, "USER_MY_MODELS_PATH", tmp_path / "MY_MODELS.json")
    loaded = model_info._load_model_info()
    for name, info in loaded.items():
        model_info._apply_cache_pricing(name, info)
    # Load-time defaults: gemma (open-weight, no published cache discount) gets
    # none and bills cached tokens at the input price; gemini kept its stored
    # read price and did NOT receive a default write price.
    assert loaded["openrouter/google/gemma-9-it"].cache_read_price_per_1M is None
    assert loaded["openrouter/google/gemma-9-it"].cache_write_price_per_1M is None
    assert loaded["openrouter/google/gemini-9-pro"].cache_read_price_per_1M == pytest.approx(0.2)
    assert loaded["openrouter/google/gemini-9-pro"].cache_write_price_per_1M is None
    monkeypatch.setattr(model_info, "MODEL_INFO", loaded)
    current = mod.get_current_model_info()
    assert "cache_read_price_per_1M" not in current["openrouter/google/gemma-9-it"]
    assert "cache_write_price_per_1M" not in current["openrouter/google/gemma-9-it"]
    assert current["openrouter/google/gemini-9-pro"]["cache_read_price_per_1M"] == 0.2
    assert "cache_write_price_per_1M" not in current["openrouter/google/gemini-9-pro"]
