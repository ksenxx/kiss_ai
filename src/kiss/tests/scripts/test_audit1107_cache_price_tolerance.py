# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Cache prices refresh with the same tolerance as input/output prices.

The 2026-11-07 catalog refresh moved ``openrouter/~z-ai/glm-flash-latest``'s
input price from $0.0285 to $0.0284 per 1M (0.35%, past
``_price_changed``'s 0.1% tolerance) while its equal cache-read price
($0.0285 -> $0.0284) was held by a fixed $0.0001/1M tolerance, leaving a
cache hit billed above a plain input token.  These tests pin the shared
relative tolerance and the catalog invariant it protects.
"""

from __future__ import annotations

import copy
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


def test_sub_cent_cache_price_move_registers() -> None:
    """A 0.35% move of a sub-cent cache rate is a change, like the input price's."""
    cur = {"cache_read_price_per_1M": 0.0285}
    assert mod._cache_price_changes(cur, {"cache_read_price_per_1M": 0.0284}) == {
        "cache_read_price_per_1M": 0.0284
    }
    # Below the 0.1% tolerance (float noise of a 6-decimal rounding): no change.
    assert (
        mod._cache_price_changes(
            {"cache_read_price_per_1M": 0.008288}, {"cache_read_price_per_1M": 0.00829}
        )
        == {}
    )
    # Same rule for write and image prices.
    assert mod._cache_price_changes(
        {"cache_write_price_per_1M": 0.05, "image_output_price_per_1M": 0.03},
        {"cache_write_price_per_1M": 0.0499, "image_output_price_per_1M": 0.03},
    ) == {"cache_write_price_per_1M": 0.0499}


def test_input_and_cache_prices_move_together() -> None:
    """compute_changes keeps equal input and cache-read prices equal."""
    name = "openrouter/~z-ai/glm-flash-latest"
    current = {
        name: {
            "context_length": 500000,
            "input_price_per_1M": 0.0285,
            "output_price_per_1M": 0.0923,
            "cache_read_price_per_1M": 0.0285,
            "fc": True,
            "emb": False,
            "gen": True,
        }
    }
    fetched = {
        name: {
            "context_length": 500000,
            "input_price_per_1M": 0.0284,
            "output_price_per_1M": 0.0923,
            "cache_read_price_per_1M": 0.0284,
            "source": "openrouter",
            "decisions": False,
        }
    }
    updates, new_models = mod.compute_changes(current, fetched, {}, {}, {})
    assert new_models == []
    assert updates == [
        {
            "name": name,
            "changes": {"input_price_per_1M": 0.0284, "cache_read_price_per_1M": 0.0284},
            "source": "openrouter",
        }
    ]


def _cache_reads_above_input(raw: dict[str, Any]) -> list[tuple[str, int, float, float]]:
    """Return ``(name, tier threshold, cache read, input)`` of each cache hit priced above input.

    Past a threshold a prompt is billed the entry's prices as replaced, key by key and in
    listing order, by every tier up to it (``calculate_cost``).
    """
    bad = []
    for name, entry in raw.items():
        if name.startswith("_") or not isinstance(entry, dict):
            continue
        tiers = entry.get("prompt_tiers") or []
        for threshold in [0, *sorted({tier["min_prompt_tokens"] for tier in tiers})]:
            prices = dict(entry)
            for tier in tiers:
                if tier["min_prompt_tokens"] <= threshold:
                    prices.update(tier)
            read, inp = prices.get("cache_read_price_per_1M"), prices.get("input_price_per_1M")
            if read is not None and inp is not None and read > inp + 1e-12:
                bad.append((name, threshold, read, inp))
    return bad


def test_bundled_catalog_never_bills_cache_reads_above_input() -> None:
    """No catalog entry (or prompt tier) prices a cache hit above an input token."""
    raw = json.loads(
        (Path(model_info.__file__).parent / "MODEL_INFO.json").read_text(encoding="utf-8")
    )
    assert _cache_reads_above_input(raw) == []


def _entry(input_price: float, **extra: Any) -> dict[str, Any]:
    """Return a catalog entry at *input_price* per 1M with the *extra* fields."""
    entry = {"context_length": 32000, "input_price_per_1M": input_price, "output_price_per_1M": 2.0}
    return {**entry, "fc": True, "emb": False, "gen": True, **extra}


def _listed(model_id: str, prompt: str, cache_read: str, **pricing: Any) -> dict[str, Any]:
    """Return a listing entry at *prompt* and *cache_read* per token, $2 per 1M of output."""
    sheet = {"prompt": prompt, "completion": "0.000002", "input_cache_read": cache_read}
    return _or_model(model_id, prompt, "0.000002", pricing={**sheet, **pricing})


def test_a_listed_cache_read_above_input_is_imported_at_the_input_price(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The importer never writes a cache hit priced above the tokens it discounts.

    OpenRouter's ``~<vendor>/<model>-latest`` router aliases publish blended prices: on
    2026-11-20 ``~deepseek/deepseek-pro-latest`` listed $0.21 per 1M cache-read tokens against
    $0.2066 per 1M input tokens, and the catalog took both as listed.  Every such price comes
    down to the input price in force with it: of an entry, of a prompt tier, of a new model,
    one the catalog already stores, and one the 0.1% tolerance kept while the input price fell
    below it, or beside a long-prompt input price just below it.  A price the listing withdrew
    is dropped as before, and a second run over the result changes nothing.
    """
    catalog = tmp_path / "MODEL_INFO.json"
    current = {
        "openrouter/~deepseek/deepseek-pro-latest": _entry(0.0999, cache_read_price_per_1M=0.0999),
        "openrouter/acme/tiered": _entry(1.25, cache_read_price_per_1M=0.125),
        "openrouter/acme/stored": _entry(0.5, cache_read_price_per_1M=0.6),
        "openrouter/acme/drifting": _entry(
            1.01,
            cache_read_price_per_1M=1.0,
            prompt_tiers=[{"min_prompt_tokens": 100000, "cache_read_price_per_1M": 1.0}],
        ),
        "openrouter/acme/long": _entry(
            1.0,
            cache_read_price_per_1M=0.3002,
            prompt_tiers=[{"min_prompt_tokens": 200000, "input_price_per_1M": 0.3}],
        ),
        "openrouter/acme/withdrawn": _entry(1.0, cache_read_price_per_1M=0.1),
    }
    catalog.write_text(json.dumps(current))
    monkeypatch.setattr(mod, "MODEL_INFO_PATH", catalog)
    tiers = [
        {"min_prompt_tokens": 100000, "input_cache_read": "0.0000013"},
        {"min_prompt_tokens": 200000, "prompt": "0.0000025", "input_cache_read": "0.0000026"},
    ]
    drifting = [{"min_prompt_tokens": 100000, "input_cache_read": "0.0000009995"}]
    cheaper = [{"min_prompt_tokens": 200000, "prompt": "0.0000003"}]
    listing = {
        "data": [
            _listed("~deepseek/deepseek-pro-latest", "0.0000002066", "0.00000021"),
            _listed("acme/tiered", "0.00000125", "0.000000125", overrides=tiers),
            _listed("acme/stored", "0.0000005", "0.0000006"),
            _listed("acme/drifting", "0.0000009995", "0.0000009995", overrides=drifting),
            _listed("acme/long", "0.000001", "0.0000003002", overrides=cheaper),
            _or_model("acme/withdrawn", "0.000001", "0.000002"),
            _listed("acme/new", "0.000001", "0.0000011"),
        ]
    }
    with _listing_server(listing, {"data": []}) as url:
        _point_fetch_at(monkeypatch, url)
        fetched = mod.fetch_openrouter()
    stored = mod.get_current_model_info()
    given = copy.deepcopy((stored, fetched))
    updates, new_models = mod.compute_changes(stored, fetched, {}, {}, {})
    assert {update["name"]: update["changes"] for update in updates} == {
        "openrouter/~deepseek/deepseek-pro-latest": {
            "input_price_per_1M": 0.2066,
            "cache_read_price_per_1M": 0.2066,
        },
        "openrouter/acme/tiered": {
            "prompt_tiers": [
                {"min_prompt_tokens": 100000, "cache_read_price_per_1M": 1.25},
                {
                    "min_prompt_tokens": 200000,
                    "input_price_per_1M": 2.5,
                    "cache_read_price_per_1M": 2.5,
                },
            ]
        },
        "openrouter/acme/stored": {"cache_read_price_per_1M": 0.5},
        "openrouter/acme/drifting": {
            "input_price_per_1M": 0.9995,
            "cache_read_price_per_1M": 0.9995,
            "prompt_tiers": [{"min_prompt_tokens": 100000, "cache_read_price_per_1M": 0.9995}],
        },
        "openrouter/acme/long": {"cache_read_price_per_1M": 0.3},
        "openrouter/acme/withdrawn": {"cache_read_price_per_1M": None},
    }
    assert [(new["name"], new["cache_read_price_per_1M"]) for new in new_models] == [
        ("openrouter/acme/new", 1.0)
    ]
    # The catalog and the listing are left as given: other passes of a run read them too.
    assert (stored, fetched) == given
    mod.apply_updates_to_file(updates, new_models, [], current)
    written = json.loads(catalog.read_text())
    assert _cache_reads_above_input(written) == []
    assert written["openrouter/acme/new"]["cache_read_price_per_1M"] == 1.0
    assert mod.compute_changes(mod.get_current_model_info(), fetched, {}, {}, {}) == ([], [])


def test_openrouter_audio_prices_are_imported() -> None:
    """OpenRouter's ``audio`` / ``audio_output`` rates reach the catalog.

    OpenRouter bills gemini-3.5-flash audio input at $3/1M against $1.50/1M
    for text; without the imported rate ``calculate_cost`` fell back to the
    text input price and under-billed audio by half (voxtral-small: $100
    vs $0.10).
    """
    pricing = {
        "prompt": "0.0000015",
        "completion": "0.000009",
        "audio": "0.000003",
        "audio_output": "0.00001",
        "input_audio_cache": "0.0000003",
    }
    assert mod.openrouter_cache_prices(pricing) == {
        "audio_input_price_per_1M": 3.0,
        "audio_output_price_per_1M": 10.0,
    }
    # A rate the listing drops is removed so the text-price fallback applies.
    assert mod._cache_price_changes({"audio_input_price_per_1M": 3.0}, {}) == {
        "audio_input_price_per_1M": None
    }
    assert "audio_input_price_per_1M" in mod._REMOVABLE_PRICE_FIELDS


def test_bundled_openrouter_gemini_bills_audio_at_the_audio_rate() -> None:
    """The refreshed catalog prices OpenRouter Gemini audio at OpenRouter's rate."""
    cost = model_info.calculate_cost(
        "openrouter/google/gemini-3.5-flash", 0, 0, num_audio_input_tokens=1_000_000
    )
    assert cost == pytest.approx(3.0)
    # Audio cache reads keep the text cache discount (OpenRouter's
    # ``input_audio_cache`` $0.30/1M = $3.00 * 0.15 / 1.50).
    assert model_info.calculate_cost(
        "openrouter/google/gemini-3.5-flash", 0, 0, num_audio_cache_read_tokens=1_000_000
    ) == pytest.approx(0.3)


def test_openrouter_audio_overrides_become_prompt_tiers() -> None:
    """An ``audio`` long-prompt override is a tier price, not an ignored key."""
    pricing = {
        "prompt": "0.00000125",
        "audio": "0.00000125",
        "overrides": [
            {
                "min_prompt_tokens": 200000,
                "prompt": "0.0000025",
                "audio": "0.0000025",
                "input_audio_cache": "0.00000025",
            }
        ],
    }
    assert mod.openrouter_prompt_tiers(pricing) == {
        "prompt_tiers": [
            {
                "min_prompt_tokens": 200000,
                "input_price_per_1M": 2.5,
                "audio_input_price_per_1M": 2.5,
            }
        ]
    }


@pytest.mark.parametrize(
    ("name", "short_rate", "long_rate"),
    [
        ("openrouter/google/gemini-2.5-pro", 1.25, 2.5),
        ("openrouter/google/gemini-3.1-pro-preview", 2.0, 4.0),
    ],
)
def test_long_prompt_audio_bills_at_the_tier_rate(
    name: str, short_rate: float, long_rate: float
) -> None:
    """Above 200K prompt tokens audio (and cached audio) doubles like text."""
    cost = model_info.calculate_cost
    assert cost(name, 0, 0, num_audio_input_tokens=100_000) == pytest.approx(0.1 * short_rate)
    assert cost(name, 0, 0, num_audio_input_tokens=300_000) == pytest.approx(0.3 * long_rate)
    assert cost(name, 0, 0, num_audio_cache_read_tokens=300_000) == pytest.approx(0.03 * long_rate)
    # A multi-request sum takes no tier.
    assert cost(name, 0, 0, num_audio_input_tokens=300_000, single_request=False) == pytest.approx(
        0.3 * short_rate
    )


def test_uplift_scales_an_explicit_audio_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    """A long-context uplift scales a registered audio rate like the text rate."""
    info = model_info.ModelInfo(
        200_000,
        2.0,
        12.0,
        True,
        False,
        True,
        audio_input_price_per_million=4.0,
        audio_output_price_per_million=20.0,
    )
    monkeypatch.setitem(model_info.MODEL_INFO, "audit1107-uplift-audio", info)
    monkeypatch.setattr(model_info, "_long_context_uplift", lambda _name: (200_000, 2.0, 1.5))
    assert model_info.calculate_cost(
        "audit1107-uplift-audio", 0, 0, num_audio_input_tokens=300_000
    ) == pytest.approx(0.3 * 8.0)
    assert model_info.calculate_cost(
        "audit1107-uplift-audio",
        100_000,
        0,
        num_audio_input_tokens=150_000,
        num_audio_output_tokens=1_000,
    ) == pytest.approx(0.1 * 4.0 + 0.15 * 8.0 + 0.001 * 30.0)
