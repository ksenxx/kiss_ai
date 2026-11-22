# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The model picker lists a custom-endpoint model at the prices it is billed at.

``MY_MODELS.json`` may register a model served from the user's own endpoint
together with its prices (an installer's router models, say, at $3 / $15 per
1M tokens).  ``calculate_cost`` bills a run on such a model at those prices,
but ``AgentServer._get_models`` listed every one of them with ``inp: 0`` and
``out: 0``, so the picker showed ``$0.00 / $0.00`` for a model that is not
free (found by the 2026-11-21 cost audit: 24 router models of the live
registry).  The entry now carries the prices of the one lookup
``calculate_cost`` uses, whatever the way the name resolves: its own
registry entry, read as the file is now, or the catalog model it names.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from kiss.core import config as config_module
from kiss.core import user_config
from kiss.core.kiss_error import KISSError
from kiss.core.models import model_info
from kiss.core.models.model_info import (
    MODEL_INFO,
    calculate_cost,
    get_model_prices,
    save_custom_model,
)

_KEY_NAMES = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "OPENROUTER_API_KEY",
)
_ENDPOINT = "https://router.example.test/v1"


@pytest.fixture
def registry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    """An empty ``MY_MODELS.json`` of its own; no vendor key, no CLI on PATH."""
    for name in _KEY_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("KISS_MODEL", raising=False)
    monkeypatch.setenv("PATH", "")
    monkeypatch.setenv("KISS_WORKDIR", str(tmp_path))
    monkeypatch.setattr(config_module, "DEFAULT_CONFIG", config_module.Config())
    path = tmp_path / "MY_MODELS.json"
    monkeypatch.setattr(model_info, "USER_MY_MODELS_PATH", path)
    yield path
    config_module.DEFAULT_CONFIG = config_module.Config()


def _picker_rows(monkeypatch: pytest.MonkeyPatch) -> dict[str, dict[str, Any]]:
    """The ``models`` event of the real ``AgentServer._get_models``, by model name."""
    import kiss.server.server as server_module

    monkeypatch.setattr(server_module, "_load_last_model", lambda: "")
    monkeypatch.setattr(server_module, "_save_last_model", lambda *_a, **_k: None, raising=False)
    server = server_module.AgentServer()
    events: list[dict[str, Any]] = []
    lock = threading.Lock()

    def capture(event: dict[str, Any]) -> None:
        with lock:
            events.append(dict(event))

    monkeypatch.setattr(server.printer, "broadcast", capture)
    server._get_models()
    with lock:
        models_event = [e for e in events if e.get("type") == "models"][-1]
    rows: dict[str, dict[str, Any]] = {}
    for row in models_event["models"]:
        assert row["name"] not in rows, f"{row['name']} is listed twice"
        rows[row["name"]] = row
    return rows


def _billed_per_million(name: str) -> tuple[float, float]:
    """What ``calculate_cost`` charges for 1M input and for 1M output tokens of *name*."""
    return calculate_cost(name, 1_000_000, 0), calculate_cost(name, 0, 1_000_000)


def test_registered_endpoint_model_is_listed_at_its_registered_prices(
    registry: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A router model registered with prices does not read as free."""
    entries = {
        "acme-router-large": {
            "context_length": 500000,
            "input_price_per_1M": 3.0,
            "output_price_per_1M": 15.0,
            "cache_read_price_per_1M": 0.3,
            "fc": True,
            "emb": False,
            "gen": True,
            "endpoint": _ENDPOINT,
        },
        "acme-router-small": {
            "context_length": 262144,
            "input_price_per_1M": 0.07,
            "output_price_per_1M": 0.15,
            "fc": True,
            "emb": False,
            "gen": True,
            "endpoint": _ENDPOINT,
        },
    }
    assert not set(entries) & set(MODEL_INFO)
    registry.write_text(json.dumps(entries), encoding="utf-8")

    rows = _picker_rows(monkeypatch)

    for name, entry in entries.items():
        row = rows[name]
        assert row["vendor"] == "Custom"
        assert row["endpoint"] == _ENDPOINT
        listed = (row["inp"], row["out"])
        assert listed == (entry["input_price_per_1M"], entry["output_price_per_1M"])
        assert listed == _billed_per_million(name)
        assert listed == get_model_prices(name)


def test_catalog_model_served_from_a_custom_endpoint_is_listed_at_its_billed_prices(
    registry: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bundled name with the user's endpoint (no vendor key) keeps the catalog's prices."""
    name = next(
        n
        for n, info in sorted(MODEL_INFO.items())
        if info.is_generation_supported
        and not info.prompt_tiers
        and info.input_price_per_1M > 0
        and not n.startswith(("cc/", "codex/", "bedrock/", "openrouter/"))
    )
    info = MODEL_INFO[name]
    registry.write_text(
        json.dumps(
            {
                name: {
                    "context_length": info.context_length,
                    "input_price_per_1M": info.input_price_per_1M,
                    "output_price_per_1M": info.output_price_per_1M,
                    "endpoint": _ENDPOINT,
                }
            }
        ),
        encoding="utf-8",
    )

    row = _picker_rows(monkeypatch)[name]

    assert row["endpoint"] == _ENDPOINT
    assert (row["inp"], row["out"]) == (info.input_price_per_1M, info.output_price_per_1M)
    assert (row["inp"], row["out"]) == get_model_prices(name)


def test_settings_panel_models_are_listed_at_the_prices_they_were_saved_with(
    registry: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real save path: a catalog model behind a gateway, and a model of no known price."""
    base = next(
        n
        for n, info in sorted(MODEL_INFO.items())
        if n.startswith("gpt-") and info.is_generation_supported and info.input_price_per_1M > 0
    )
    gateway_name = f"openai/{base}"
    assert gateway_name not in MODEL_INFO
    assert save_custom_model(gateway_name, endpoint=_ENDPOINT) is None
    assert save_custom_model("my-local-model", endpoint="http://localhost:8080/v1") is None

    rows = _picker_rows(monkeypatch)

    base_info = MODEL_INFO[base]
    gateway = rows[gateway_name]
    assert (gateway["inp"], gateway["out"]) == (
        base_info.input_price_per_1M,
        base_info.output_price_per_1M,
    )
    assert (gateway["inp"], gateway["out"]) == get_model_prices(gateway_name)
    local = rows["my-local-model"]
    assert (local["inp"], local["out"]) == (0.0, 0.0)
    assert _billed_per_million("my-local-model") == (0.0, 0.0)


def test_an_endpoint_entry_that_cannot_be_priced_is_listed_as_unpriced(
    registry: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A registry row without the loader's keys is still selectable, at no known price."""
    registry.write_text(
        json.dumps({"half-written-model": {"endpoint": _ENDPOINT}}), encoding="utf-8"
    )

    row = _picker_rows(monkeypatch)["half-written-model"]

    assert row["endpoint"] == _ENDPOINT
    assert (row["inp"], row["out"]) == (0.0, 0.0)
    assert get_model_prices("half-written-model") == (0.0, 0.0)
    # No price, which is not a price of zero: a run on it cannot be costed.
    with pytest.raises(KISSError, match="unknown model"):
        calculate_cost("half-written-model", 1000, 0)


def test_configured_endpoint_row_carries_the_prices_of_a_registry_entry_of_its_name(
    registry: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``custom/<tail>`` of the configured endpoint, registered with prices, is listed at them."""
    monkeypatch.setattr(user_config, "load_config", lambda: {"custom_endpoint": _ENDPOINT})

    # Unregistered: the row has no price, and a run on it cannot be costed.
    row = _picker_rows(monkeypatch)["custom/v1"]
    assert (row["inp"], row["out"]) == (0.0, 0.0)
    with pytest.raises(KISSError, match="unknown model"):
        calculate_cost("custom/v1", 1000, 0)

    registry.write_text(
        json.dumps(
            {
                "custom/v1": {
                    "context_length": 500000,
                    "input_price_per_1M": 3.0,
                    "output_price_per_1M": 15.0,
                    "endpoint": _ENDPOINT,
                }
            }
        ),
        encoding="utf-8",
    )

    row = _picker_rows(monkeypatch)["custom/v1"]

    assert row["endpoint"] == _ENDPOINT
    assert (row["inp"], row["out"]) == (3.0, 15.0)
    assert (row["inp"], row["out"]) == _billed_per_million("custom/v1")


def test_get_model_prices_is_the_catalog_price_of_a_bundled_model() -> None:
    """For a bundled name the prices are the catalog entry's; an unknown name has none."""
    name, info = next(
        (n, i)
        for n, i in sorted(MODEL_INFO.items())
        if i.is_generation_supported and i.input_price_per_1M > 0 and i.output_price_per_1M > 0
    )
    assert get_model_prices(name) == (info.input_price_per_1M, info.output_price_per_1M)
    assert get_model_prices("no-such-vendor/no-such-model-7") == (0.0, 0.0)
