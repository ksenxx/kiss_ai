"""Catalog price validation: a malformed custom price never bills or aborts.

``_require_number`` guards every price and the context length of a
catalog entry.  Before this check also required a finite, non-negative
number, a hand-edited ``MY_MODELS.json`` could carry a negative price
(billing a negative cost), ``1e309`` or ``NaN`` (billing ``NaN``), or an
integer beyond the float range such as ``10**400``, which passed
``isinstance`` and then raised ``OverflowError`` in the cache-price
derivation — after the loader's try/except, so the whole catalog import
aborted.  These tests pin the loader (skips the entry with a warning),
the hot-add lookup (unknown model) and the builder (``ValueError``).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from kiss.core.models import model_info as mi_module
from kiss.core.models.model_info import (
    _build_model_info_entry,
    _load_model_info,
    _lookup_model_info,
)

_NAME = "price-validation-repro"

_BAD_PRICES: list[tuple[str, Any]] = [
    ("negative", -1),
    ("infinite", json.loads("1e309")),
    ("nan", json.loads("NaN")),
    ("over-float-int", 10**400),
    # Finite itself, but its 2.5x long-lifetime cache-write price is not:
    # the entry loaded with an ``inf`` rate and every cost was ``NaN``
    # (``0 tokens x inf``).
    ("overflowing", 1e308),
    # The other end: as an input price it makes the cached-audio rate
    # ``audio x cache_read / input`` infinite.
    ("vanishing", 1e-300),
]


def _entry(input_price: Any) -> dict[str, Any]:
    return {
        "context_length": 100_000,
        "input_price_per_1M": input_price,
        "output_price_per_1M": 1,
    }


@pytest.mark.parametrize(("label", "price"), _BAD_PRICES, ids=[b[0] for b in _BAD_PRICES])
def test_builder_rejects_non_finite_or_negative_price(label: str, price: Any) -> None:
    """A negative, infinite, NaN or over-float price is a ValueError; a string a TypeError."""
    with pytest.raises(ValueError, match="input_price_per_1M must be a finite, non-negative"):
        _build_model_info_entry(_entry(price))
    with pytest.raises(ValueError, match="context_length must be a finite"):
        _build_model_info_entry({**_entry(1), "context_length": price})
    with pytest.raises(ValueError, match="cache_read_price_per_1M must be a finite"):
        _build_model_info_entry({**_entry(1), "cache_read_price_per_1M": price})
    with pytest.raises(TypeError, match="input_price_per_1M must be a number"):
        _build_model_info_entry(_entry("1.5"))


def test_builder_accepts_valid_prices() -> None:
    """Zero, integer and float prices (and an absent optional price) still build."""
    info = _build_model_info_entry({**_entry(0), "cache_read_price_per_1M": None})
    assert info.input_price_per_1M == 0
    assert _build_model_info_entry(_entry(2.5)).input_price_per_1M == 2.5


@pytest.mark.parametrize(("label", "price"), _BAD_PRICES, ids=[b[0] for b in _BAD_PRICES])
def test_loader_skips_malformed_custom_price(
    label: str, price: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The catalog loads without the entry (and without raising) and bills nothing for it."""
    fake_path = tmp_path / "MY_MODELS.json"
    # ``10**400`` is a valid JSON integer; json.dumps writes inf/nan as
    # the non-standard literals json.loads reads back.
    fake_path.write_text(json.dumps({_NAME: _entry(price), "valid-custom": _entry(3)}))
    monkeypatch.setattr(mi_module, "USER_MY_MODELS_PATH", fake_path)
    table = _load_model_info()
    assert _NAME not in table
    assert table["valid-custom"].input_price_per_1M == 3
    for info in table.values():
        for field in mi_module._PER_1M_PRICE_FIELDS:
            value = getattr(info, field)
            assert value is None or (math.isfinite(value) and value >= 0), field


@pytest.mark.parametrize(("label", "price"), _BAD_PRICES, ids=[b[0] for b in _BAD_PRICES])
def test_hot_added_malformed_custom_price_is_unknown_model(
    label: str, price: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A malformed entry registered after start is unknown, not a crash or a bad bill."""
    monkeypatch.setattr(mi_module, "_read_my_models", lambda: {_NAME: _entry(price)})
    assert _NAME not in mi_module.MODEL_INFO
    assert _lookup_model_info(_NAME) is None
    with pytest.raises(mi_module.KISSError, match="unknown model"):
        mi_module.calculate_cost(_NAME, 1000, 10)
