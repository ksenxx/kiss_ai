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

import json
from pathlib import Path

import pytest

import kiss.scripts.update_models as mod
from kiss.core.models import model_info


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
    updates, new_models = mod.compute_changes(current, fetched, {}, {}, {}, {})
    assert new_models == []
    assert updates == [
        {
            "name": name,
            "changes": {"input_price_per_1M": 0.0284, "cache_read_price_per_1M": 0.0284},
            "source": "openrouter",
        }
    ]


def test_bundled_catalog_never_bills_cache_reads_above_input() -> None:
    """No catalog entry (or prompt tier) prices a cache hit above an input token."""
    raw = json.loads(
        (Path(model_info.__file__).parent / "MODEL_INFO.json").read_text(encoding="utf-8")
    )
    bad = []
    for name, entry in raw.items():
        if name.startswith("_") or not isinstance(entry, dict):
            continue
        tiers = [entry, *(entry.get("prompt_tiers") or [])]
        for tier in tiers:
            read = tier.get("cache_read_price_per_1M")
            inp = tier.get("input_price_per_1M", entry.get("input_price_per_1M"))
            if read is not None and inp is not None and read > inp + 1e-12:
                bad.append((name, tier.get("min_prompt_tokens", 0), read, inp))
    assert bad == []


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
