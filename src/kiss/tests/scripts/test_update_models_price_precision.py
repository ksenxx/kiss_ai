# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""End-to-end tests: cheap models' prices survive the catalog updater.

October 2026 cost audit: ``fetch_openrouter`` / ``fetch_together``
rounded per-1M prices to three decimals ($0.00632/M became $0.006/M)
and ``compute_changes`` ignored any input/output change of $0.005/M or
less, which is a 14% change on a $0.035/M model
(``openrouter/amazon/nova-micro-v1`` stayed at $0.040/M).  Prices are
now kept to six decimals and compared relatively, so a cheap model's
price is right to the same precision as an expensive one's.
"""

from __future__ import annotations

import pytest

import kiss.scripts.update_models as mod
from kiss.tests.scripts.test_update_models_decisions import (
    _listing_server,
    _or_model,
    _point_fetch_at,
)

LISTING = {
    "data": [
        _or_model("deepseek/deepseek-v4-flash", "0.00000000632", "0.0000000316"),
        _or_model("amazon/nova-micro-v1", "0.000000035", "0.00000014"),
        _or_model("openai/gpt-4o", "0.0000025", "0.00001"),
    ]
}


def _entry(inp: float, out: float, emb: bool = False) -> dict:
    return {
        "context_length": 32000, "input_price_per_1M": inp, "output_price_per_1M": out,
        "fc": False, "emb": emb, "gen": not emb,
    }


def test_price_changed_is_relative() -> None:
    """A tenth of a percent registers; float noise and identical prices do not."""
    assert mod._price_changed(0.04, 0.035)
    assert mod._price_changed(0.006, 0.00632)
    assert mod._price_changed(0.0, 0.0001)
    assert mod._price_changed(2.5, 2.503)
    assert not mod._price_changed(2.5, 2.5)
    assert not mod._price_changed(2.5, 2.5000001)
    assert not mod._price_changed(0.0, 0.0)


def test_fetched_openrouter_prices_keep_six_decimals(monkeypatch: pytest.MonkeyPatch) -> None:
    """$0.00632/M is stored as such, not as $0.006/M."""
    with _listing_server(LISTING) as url:
        _point_fetch_at(monkeypatch, url)
        fetched = mod.fetch_openrouter()
    flash = fetched["openrouter/deepseek/deepseek-v4-flash"]
    assert flash["input_price_per_1M"] == 0.00632
    assert flash["output_price_per_1M"] == 0.0316
    assert fetched["openrouter/amazon/nova-micro-v1"]["input_price_per_1M"] == 0.035
    assert fetched["openrouter/openai/gpt-4o"]["input_price_per_1M"] == 2.5


def test_small_absolute_but_large_relative_changes_are_applied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The stale $0.040/M entry is corrected to $0.035/M; an unchanged one is left alone."""
    current = {
        "openrouter/deepseek/deepseek-v4-flash": _entry(0.006, 0.032),
        "openrouter/amazon/nova-micro-v1": _entry(0.04, 0.14),
        "openrouter/openai/gpt-4o": _entry(2.5, 10.0),
    }
    with _listing_server(LISTING) as url:
        _point_fetch_at(monkeypatch, url)
        fetched = mod.fetch_openrouter()
    updates, new_models = mod.compute_changes(current, fetched, {}, {}, {}, {})
    # The local listing server also serves the decisions catalog; those are new.
    assert all(m.get("is_decisions") for m in new_models)
    by_name = {u["name"]: u["changes"] for u in updates}
    assert by_name == {
        "openrouter/deepseek/deepseek-v4-flash": {
            "input_price_per_1M": 0.00632, "output_price_per_1M": 0.0316,
        },
        "openrouter/amazon/nova-micro-v1": {"input_price_per_1M": 0.035},
    }


def test_together_prices_use_the_same_precision_and_threshold() -> None:
    """Together's per-1M listing gets the relative comparison too; embeddings stay frozen."""
    current = {
        "BAAI/bge-base-en-v1.5": _entry(0.01, 0.0, emb=True),
        "moonshotai/Kimi-K3": _entry(2.7, 13.5),
        "Qwen/Qwen3-Next": _entry(0.1, 0.1),
    }
    together = {
        "BAAI/bge-base-en-v1.5": {
            "context_length": 32000, "input_price_per_1M": 0.008, "output_price_per_1M": 0.0,
            "cache_read_price_per_1M": None, "source": "together",
        },
        "moonshotai/Kimi-K3": {
            "context_length": 32000, "input_price_per_1M": 2.7, "output_price_per_1M": 13.5,
            "cache_read_price_per_1M": None, "source": "together",
        },
        "Qwen/Qwen3-Next": {
            "context_length": 32000, "input_price_per_1M": 0.1, "output_price_per_1M": 0.104,
            "cache_read_price_per_1M": None, "source": "together",
        },
    }
    updates, new_models = mod.compute_changes(current, {}, together, {}, {}, {})
    assert new_models == []
    assert [(u["name"], u["changes"]) for u in updates] == [
        ("Qwen/Qwen3-Next", {"output_price_per_1M": 0.104}),
    ]
