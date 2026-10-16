# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for Claude models on Amazon Bedrock (``bedrock/<id>``).

A ``bedrock/<Bedrock model or inference-profile id>`` name runs the Claude
adapter over the Anthropic SDK's ``AnthropicBedrock`` client.  These tests
drive the real SDK against a real local HTTP endpoint that answers in
Bedrock's binary AWS event-stream format (no mocks, patches or test
doubles) and verify:

* the ``bedrock/`` name maps to its ``claude-*`` catalog key for pricing,
  context length, fallback and the effort alias suffixes;
* the routing tables label it ``Bedrock`` and report whether credentials
  are configured;
* the wire request: model id in the URL path, bearer or SigV4 auth, the
  Anthropic-only ``service_tier`` and top-level ``cache_control`` fields
  dropped, the interleaved-thinking beta carried in the body;
* the client is reused across runs and rebuilt only when credentials change;
* the direct Anthropic route is unchanged.

Run with::

    uv run pytest src/kiss/tests/core/models/test_bedrock_claude.py -v
"""

from __future__ import annotations

import base64
import json
import os
import struct
import threading
import zlib
from typing import Any

import pytest

from kiss.agents.sorcar.sorcar_agent import _model_family
from kiss.core import config as config_module
from kiss.core.models import model_info
from kiss.core.models.anthropic_model import AnthropicModel
from kiss.core.user_config import API_KEY_ENV_VARS, get_current_api_keys
from kiss.core.utils import config_to_dict
from kiss.server.autocomplete import ranked_function_calling_models
from kiss.server.helpers import model_vendor
from kiss.tests.core.models.anthropic_sse_harness import (
    ScriptedAnthropicServer,
    text_message_stream,
)

_OPUS_ID = "us.anthropic.claude-opus-4-6-v1:0"
_OPUS = "bedrock/" + _OPUS_ID
_BEDROCK_CONTENT_TYPE = "application/vnd.amazon.eventstream"
_AWS_FIELDS = (
    "AWS_BEARER_TOKEN_BEDROCK",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_REGION",
)


def _event_stream_header(name: str, value: str) -> bytes:
    """Encode one string header of an AWS event-stream message."""
    name_bytes, value_bytes = name.encode(), value.encode()
    return (
        bytes([len(name_bytes)])
        + name_bytes
        + b"\x07"
        + struct.pack(">H", len(value_bytes))
        + value_bytes
    )


def _event_stream_frame(payload: bytes) -> bytes:
    """Encode one ``chunk`` event in the AWS event-stream binary framing.

    Prelude (total length, headers length, prelude CRC32), the three
    string headers Bedrock sends, the JSON payload and the message CRC32:
    exactly what botocore's ``EventStreamBuffer`` decodes.
    """
    headers = (
        _event_stream_header(":event-type", "chunk")
        + _event_stream_header(":content-type", "application/json")
        + _event_stream_header(":message-type", "event")
    )
    prelude = struct.pack(">II", 12 + len(headers) + len(payload) + 4, len(headers))
    prelude += struct.pack(">I", zlib.crc32(prelude))
    body = prelude + headers + payload
    return body + struct.pack(">I", zlib.crc32(body))


def bedrock_stream(sse_chunks: list[bytes]) -> list[bytes]:
    """Re-encode Anthropic SSE chunks as Bedrock's event stream.

    Bedrock wraps every Messages event as ``{"bytes": <base64 JSON>}``
    inside a binary event-stream frame.
    """
    frames = []
    for chunk in sse_chunks:
        for line in chunk.decode().splitlines():
            if line.startswith("data: "):
                data = line[len("data: "):].encode()
                payload = json.dumps({"bytes": base64.b64encode(data).decode()})
                frames.append(_event_stream_frame(payload.encode()))
    return frames


def _bedrock_server(text: str = "ok") -> ScriptedAnthropicServer:
    """Start a local endpoint answering one text turn in Bedrock framing."""
    return ScriptedAnthropicServer(
        chunks=bedrock_stream(text_message_stream(text)),
        content_type=_BEDROCK_CONTENT_TYPE,
    )


def _set_aws(monkeypatch, **values: str) -> None:
    """Set the AWS config fields to *values* (others cleared) for one test."""
    for field in _AWS_FIELDS:
        monkeypatch.setattr(config_module.DEFAULT_CONFIG, field, values.get(field, ""))


# --------------------------------------------------------------------------
# Catalog mapping
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "catalog_key", "wire_id"),
    [
        (_OPUS, "claude-opus-4-6", _OPUS_ID),
        ("bedrock/anthropic.claude-opus-4-6-v1", "claude-opus-4-6", "anthropic.claude-opus-4-6-v1"),
        ("bedrock/anthropic.claude-sonnet-4-6", "claude-sonnet-4-6", "anthropic.claude-sonnet-4-6"),
        (
            "bedrock/global.anthropic.claude-sonnet-4-6",
            "claude-sonnet-4-6",
            "global.anthropic.claude-sonnet-4-6",
        ),
        (
            "bedrock/eu.anthropic.claude-haiku-4-5-20251001-v1:0",
            "claude-haiku-4-5-20251001",
            "eu.anthropic.claude-haiku-4-5-20251001-v1:0",
        ),
        (
            "bedrock/anthropic.claude-opus-4-5-20251101-v1:0",
            "claude-opus-4-5-20251101",
            "anthropic.claude-opus-4-5-20251101-v1:0",
        ),
        (_OPUS + "-high", "claude-opus-4-6-high", _OPUS_ID),
        (_OPUS + "-low", "claude-opus-4-6-low", _OPUS_ID),
        (_OPUS + "-max", "claude-opus-4-6-max", _OPUS_ID),
    ],
)
def test_bedrock_name_maps_to_catalog_key(name: str, catalog_key: str, wire_id: str) -> None:
    """Region prefix, vendor segment, version suffix and date are normalised."""
    assert model_info.bedrock_catalog_name(name) == catalog_key
    assert model_info.bedrock_model_id(name) == wire_id
    assert model_info.is_known_model(name)


def test_undated_key_used_when_dated_snapshot_is_not_in_catalog() -> None:
    """A dated Bedrock id whose snapshot the catalog lacks falls back to the undated key."""
    assert "claude-opus-4-6-20260101" not in model_info.MODEL_INFO
    name = "bedrock/anthropic.claude-opus-4-6-20260101-v1:0"
    assert model_info.bedrock_catalog_name(name) == "claude-opus-4-6"


@pytest.mark.parametrize(
    "name",
    [
        "bedrock/anthropic.claude-nope-9-v1:0",
        "bedrock/amazon.nova-pro-v1:0",
        # Haiku 4.5 has no effort aliases, Opus 4.6 no -xhigh one.
        "bedrock/anthropic.claude-haiku-4-5-20251001-v1:0-high",
        _OPUS + "-xhigh",
        "claude-opus-4-6",
        "openrouter/anthropic/claude-opus-4.6",
    ],
)
def test_unknown_or_non_bedrock_names_have_no_catalog_key(name: str) -> None:
    """Unknown Claude ids, non-Claude Bedrock ids and non-Bedrock names map to nothing."""
    assert model_info.bedrock_catalog_name(name) is None


def test_unknown_bedrock_name_is_not_known_and_unpriced() -> None:
    """An unknown ``bedrock/`` id is unknown to the catalog, like any unknown model."""
    name = "bedrock/anthropic.claude-nope-9-v1:0"
    assert not model_info.is_known_model(name)
    assert model_info.bedrock_model_id(name) == "anthropic.claude-nope-9-v1:0"
    with pytest.raises(Exception):
        model_info.calculate_cost(name, 10, 10)


def test_bedrock_name_is_priced_as_its_catalog_entry() -> None:
    """Cost, context length and the OpenRouter twin follow the ``claude-*`` entry.

    A ``global.`` profile costs exactly the catalog entry; the ``us.``
    profile of the same model costs Bedrock's 10% geographic premium on
    top of it (every price, cache included).
    """
    direct = "claude-opus-4-6-high"
    bedrock = _OPUS + "-high"
    global_profile = "bedrock/global.anthropic.claude-opus-4-6-v1:0-high"
    usage = (1234, 567, 89, 10, 5)
    assert model_info.calculate_cost(global_profile, *usage) == model_info.calculate_cost(
        direct, *usage
    )
    assert model_info.calculate_cost(bedrock, *usage) == pytest.approx(
        1.1 * model_info.calculate_cost(direct, *usage)
    )
    assert model_info.get_max_context_length(bedrock) == model_info.get_max_context_length(direct)
    assert model_info.openrouter_twin(bedrock) == model_info.openrouter_twin(direct)
    assert model_info.openrouter_twin(bedrock) is not None


@pytest.mark.parametrize(
    ("name", "factor"),
    [
        # Geographic and in-region profiles of Claude 4.5+ : +10%.
        (_OPUS, 1.1),
        ("bedrock/eu.anthropic.claude-haiku-4-5-20251001-v1:0", 1.1),
        ("bedrock/apac.anthropic.claude-sonnet-4-5-20250929-v1:0", 1.1),
        ("bedrock/jp.anthropic.claude-fable-5-1-v1:0", 1.1),
        ("bedrock/au.anthropic.claude-opus-5-v1:0", 1.1),
        ("bedrock/anthropic.claude-opus-4-5-20251101-v1:0", 1.1),
        (_OPUS + "-max", 1.1),
        # Global profiles: the catalog price.
        ("bedrock/global.anthropic.claude-opus-4-6-v1:0", 1.0),
        ("bedrock/global.anthropic.claude-sonnet-5-5-v1:0-high", 1.0),
        # Opus 4.1 and older cost the same on every profile.
        ("bedrock/anthropic.claude-opus-4-1-20250805-v1:0", 1.0),
        ("bedrock/us.anthropic.claude-opus-4-1-20250805-v1:0", 1.0),
        ("bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0", 1.0),
        ("bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0", 1.0),
        ("bedrock/us.anthropic.claude-3-5-haiku-20241022-v1:0", 1.0),
        # Not a Bedrock Claude id.
        ("claude-opus-4-6", 1.0),
        ("anthropic/claude-opus-4-6", 1.0),
        ("bedrock/amazon.nova-pro-v1:0", 1.0),
    ],
)
def test_bedrock_price_factor(name: str, factor: float) -> None:
    """The geographic premium applies to non-``global.`` profiles of Claude 4.5 and later."""
    assert model_info.bedrock_price_factor(name) == factor


def test_geographic_profile_prices_match_the_bedrock_price_table() -> None:
    """``us.`` prices equal AWS's "Geo and In-region Cross-region Inference" table.

    Checked against https://aws.amazon.com/bedrock/pricing/ (US East,
    2026-10-16): Opus 4.6 $5.50 / $27.50, 5m write $6.875, 1h write
    $11.00, cache read $0.55; Haiku 5.5 $0.11 / $0.55 and, past 100K
    prompt tokens, $0.55 / $2.75; Fable 5.1 $11 / $55, cache read $0.275.
    """
    million = 1_000_000
    opus = _OPUS
    assert model_info.calculate_cost(opus, million, 0) == pytest.approx(5.5)
    assert model_info.calculate_cost(opus, 0, million) == pytest.approx(27.5)
    assert model_info.calculate_cost(opus, 0, 0, million) == pytest.approx(0.55)
    assert model_info.calculate_cost(opus, 0, 0, 0, million) == pytest.approx(6.875)
    assert model_info.calculate_cost(opus, 0, 0, 0, 0, million) == pytest.approx(11.0)
    haiku = "bedrock/us.anthropic.claude-haiku-5-5-v1:0"
    assert model_info.calculate_cost(haiku, 10_000, 0) == pytest.approx(0.11 * 0.01)
    assert model_info.calculate_cost(haiku, 0, 10_000) == pytest.approx(0.55 * 0.01)
    assert model_info.calculate_cost(haiku, million, 0) == pytest.approx(0.55)
    assert model_info.calculate_cost(haiku, million, million) == pytest.approx(0.55 + 2.75)
    fable = "bedrock/eu.anthropic.claude-fable-5-1-v1:0"
    assert model_info.calculate_cost(fable, million, million, million) == pytest.approx(
        11.0 + 55.0 + 0.275
    )
    # The catalog entry itself is never scaled in place.
    assert model_info.MODEL_INFO["claude-opus-4-6"].input_price_per_1M == 5.0
    assert model_info.calculate_cost("claude-opus-4-6", million, 0) == 5.0


def test_get_fallback_model_follows_catalog_entry(monkeypatch) -> None:
    """With an OpenRouter key the Bedrock model falls back to the catalog twin."""
    monkeypatch.setattr(config_module.DEFAULT_CONFIG, "OPENROUTER_API_KEY", "or-key")
    assert model_info.get_fallback_model(_OPUS) == model_info.get_fallback_model("claude-opus-4-6")
    assert model_info.get_fallback_model(_OPUS) is not None
    monkeypatch.setattr(config_module.DEFAULT_CONFIG, "OPENROUTER_API_KEY", "")
    assert model_info.get_fallback_model(_OPUS) is None


# --------------------------------------------------------------------------
# Routing tables and credentials
# --------------------------------------------------------------------------


def test_provider_labels() -> None:
    """Every routing/display table labels ``bedrock/`` names as Bedrock, Anthropic family."""
    assert model_info.get_model_provider(_OPUS) == "Bedrock"
    assert model_info.get_model_provider("claude-opus-4-6") == "Anthropic"
    assert model_vendor(_OPUS) == ("Bedrock", 1)
    assert _model_family(_OPUS) == "anthropic"


def test_picker_groups_bedrock_models_contiguously(monkeypatch, tmp_path) -> None:
    """Ranked models keep every vendor in one run, Bedrock right after Anthropic.

    The picker renders a group header whenever the vendor changes between
    neighbours.  A Bedrock entry is priced like its Claude twin, so with
    Anthropic's sort order it would land between the Claude models and
    split the Anthropic group in two.
    """
    monkeypatch.setattr(model_info, "USER_MY_MODELS_PATH", tmp_path / "MY_MODELS.json")
    assert model_info.save_custom_model(_OPUS) is None
    monkeypatch.setitem(model_info.MODEL_INFO, _OPUS, _reloaded_catalog()[_OPUS])
    monkeypatch.setattr(config_module.DEFAULT_CONFIG, "ANTHROPIC_API_KEY", "anthropic-key")
    _set_aws(monkeypatch, AWS_BEARER_TOKEN_BEDROCK="bearer")
    ranked = ranked_function_calling_models()
    assert _OPUS in ranked
    vendors = [model_vendor(name)[0] for name in ranked]
    headers = [vendor for i, vendor in enumerate(vendors) if i == 0 or vendors[i - 1] != vendor]
    assert len(headers) == len(set(headers)), headers
    assert headers.index("Bedrock") == headers.index("Anthropic") + 1


def test_configured_providers_tracks_bedrock_credentials(monkeypatch) -> None:
    """Bedrock is configured with a bearer token or a key pair, not a lone key id."""
    _set_aws(monkeypatch)
    assert model_info.bedrock_credentials_configured() is False
    assert model_info._configured_providers()["Bedrock"] is False
    _set_aws(monkeypatch, AWS_BEARER_TOKEN_BEDROCK="bearer")
    assert model_info._configured_providers()["Bedrock"] is True
    _set_aws(monkeypatch, AWS_ACCESS_KEY_ID="AKIA")
    assert model_info.bedrock_credentials_configured() is False
    _set_aws(monkeypatch, AWS_ACCESS_KEY_ID="AKIA", AWS_SECRET_ACCESS_KEY="secret")
    assert model_info.bedrock_credentials_configured() is True


def test_factory_builds_bedrock_adapter(monkeypatch) -> None:
    """``model()`` routes a ``bedrock/`` name to AnthropicModel in Bedrock mode."""
    _set_aws(monkeypatch, AWS_BEARER_TOKEN_BEDROCK="bearer-from-config")
    adapter = model_info.model(_OPUS + "-high")
    assert isinstance(adapter, AnthropicModel)
    assert adapter.bedrock is True
    assert adapter.api_key == "bearer-from-config"
    assert adapter._api_model_name == _OPUS_ID
    assert adapter._catalog_model_name == "claude-opus-4-6-high"
    direct = model_info.model("claude-opus-4-6-high")
    assert isinstance(direct, AnthropicModel)
    assert direct.bedrock is False
    assert direct._api_model_name == "claude-opus-4-6"
    assert direct._catalog_model_name == "claude-opus-4-6-high"


def test_unknown_bedrock_id_still_routes_to_bedrock(monkeypatch) -> None:
    """A Claude id the catalog lacks is still sent to Bedrock, under its own id."""
    _set_aws(monkeypatch)
    adapter = model_info.model("bedrock/anthropic.claude-nope-9-v1:0")
    assert isinstance(adapter, AnthropicModel)
    assert adapter.bedrock is True
    assert adapter._api_model_name == "anthropic.claude-nope-9-v1:0"
    # The capability heuristics see the bare Claude name, not the Bedrock wrapper.
    assert adapter._catalog_model_name == "claude-nope-9"
    assert model_info.bedrock_claude_name("bedrock/amazon.nova-pro-v1:0") is None


def test_custom_model_entry_puts_bedrock_id_in_the_picker(monkeypatch, tmp_path) -> None:
    """A ``bedrock/<id>`` saved under Custom Models (name only) is a runnable, priced catalog model.

    That is how a Bedrock id reaches the picker without the catalog
    inventing Bedrock ids: the entry copies its Claude entry's context
    length and prices, is labelled Bedrock, and is listed by
    ``get_available_models()`` once Bedrock credentials are configured.
    """
    monkeypatch.setattr(model_info, "USER_MY_MODELS_PATH", tmp_path / "MY_MODELS.json")
    assert model_info.save_custom_model(_OPUS) is None
    entry = json.loads((tmp_path / "MY_MODELS.json").read_text())[_OPUS]
    claude = model_info.MODEL_INFO["claude-opus-4-6"]
    assert entry["context_length"] == claude.context_length
    # A ``us.`` profile: the Claude prices plus Bedrock's 10% geographic premium.
    assert entry["input_price_per_1M"] == 5.5 == 1.1 * claude.input_price_per_1M
    assert entry["output_price_per_1M"] == 27.5
    # The next daemon start merges the file into the catalog.
    monkeypatch.setattr(model_info, "MODEL_INFO", model_info._load_model_info())
    assert _OPUS in model_info.MODEL_INFO
    _set_aws(monkeypatch)
    assert _OPUS not in model_info.get_available_models()
    _set_aws(monkeypatch, AWS_BEARER_TOKEN_BEDROCK="bearer")
    assert _OPUS in model_info.get_available_models()
    # The loaded entry already carries the premium; it is not applied twice.
    assert model_info.calculate_cost(_OPUS, 1000, 1000) == pytest.approx(
        1.1 * model_info.calculate_cost("claude-opus-4-6", 1000, 1000)
    )
    assert model_info.calculate_cost(_OPUS, 1_000_000, 0) == pytest.approx(5.5)


def test_global_profile_entry_copies_the_claude_prices_unchanged(monkeypatch, tmp_path) -> None:
    """A ``global.`` profile saved under Custom Models bills exactly like its Claude entry."""
    name = "bedrock/global.anthropic.claude-opus-4-6-v1:0"
    monkeypatch.setattr(model_info, "USER_MY_MODELS_PATH", tmp_path / "MY_MODELS.json")
    assert model_info.save_custom_model(name) is None
    entry = json.loads((tmp_path / "MY_MODELS.json").read_text())[name]
    claude = model_info.MODEL_INFO["claude-opus-4-6"]
    assert entry["input_price_per_1M"] == claude.input_price_per_1M == 5.0
    assert entry["cache_read_price_per_1M"] == claude.cache_read_price_per_1M
    monkeypatch.setattr(model_info, "MODEL_INFO", _reloaded_catalog())
    usage = (1234, 567, 89, 10, 5)
    assert model_info.calculate_cost(name, *usage) == model_info.calculate_cost(
        "claude-opus-4-6", *usage
    )


def test_hot_added_geographic_entry_is_not_scaled_twice(monkeypatch, tmp_path) -> None:
    """A ``bedrock/`` entry saved after start-up bills its stored (already scaled) prices."""
    monkeypatch.setattr(model_info, "USER_MY_MODELS_PATH", tmp_path / "MY_MODELS.json")
    name = "bedrock/us.anthropic.claude-sonnet-5-5-v1:0"
    assert name not in model_info.MODEL_INFO
    assert model_info.save_custom_model(name) is None
    entry = json.loads((tmp_path / "MY_MODELS.json").read_text())[name]
    assert entry["input_price_per_1M"] == pytest.approx(2.2)
    assert entry["cache_read_price_per_1M"] == pytest.approx(0.11)
    million = 1_000_000
    # Catalog resolution (through the claude-* key) wins and is scaled once.
    assert model_info.calculate_cost(name, million, 0) == pytest.approx(2.2)
    assert model_info.calculate_cost(name, 0, 0, million) == pytest.approx(0.11)
    # A hand-written entry under the name itself holds the prices that name pays.
    hand = "bedrock/us.anthropic.claude-opus-9-9-v1:0"
    assert hand not in model_info.MODEL_INFO and model_info.bedrock_catalog_name(hand) is None
    raw = json.loads((tmp_path / "MY_MODELS.json").read_text())
    raw[hand] = {"context_length": 1000, "input_price_per_1M": 7.0, "output_price_per_1M": 70.0}
    (tmp_path / "MY_MODELS.json").write_text(json.dumps(raw))
    assert model_info.calculate_cost(hand, million, million) == pytest.approx(77.0)


def test_config_fields_and_key_store_inventory() -> None:
    """The AWS fields exist on Config, are in the key store, and never reach trajectories."""
    for field in _AWS_FIELDS:
        assert field in config_module.Config.model_fields
        assert field in API_KEY_ENV_VARS
        assert field in get_current_api_keys()
        assert field not in config_to_dict()


def _pin_env(monkeypatch, name: str) -> None:
    """Register *name*'s true pre-test value as the earliest monkeypatch record.

    ``save_api_key``/``load_api_keys`` write ``os.environ`` directly; a
    later mid-test ``delenv`` would otherwise record the leaked value as
    its undo and resurrect it at teardown (see the workspace-id tests).
    """
    current = os.environ.get(name)
    if current is not None:
        monkeypatch.setenv(name, current)
    else:
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)


def test_key_store_round_trip(tmp_path, monkeypatch) -> None:
    """The settings panel's store persists every AWS field and the daemon start re-imports it."""
    from kiss.core.user_config import api_keys_env_path, load_api_keys, save_api_key

    for field in _AWS_FIELDS:
        _pin_env(monkeypatch, field)
        monkeypatch.setattr(
            config_module.DEFAULT_CONFIG, field, getattr(config_module.DEFAULT_CONFIG, field)
        )
        monkeypatch.delenv(field, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("KISS_HOME", str(tmp_path / ".kiss"))

    for field in _AWS_FIELDS:
        save_api_key(field, "value-" + field)
    stored = api_keys_env_path().read_text(encoding="utf-8")
    for field in _AWS_FIELDS:
        assert f"export {field}=value-{field}" in stored
        monkeypatch.delenv(field, raising=False)
    load_api_keys()
    for field in _AWS_FIELDS:
        assert os.environ[field] == "value-" + field
        assert getattr(config_module.DEFAULT_CONFIG, field) == "value-" + field
    assert model_info.bedrock_credentials_configured() is True

    save_api_key("AWS_BEARER_TOKEN_BEDROCK", "")
    save_api_key("AWS_SECRET_ACCESS_KEY", "")
    assert "AWS_BEARER_TOKEN_BEDROCK" not in api_keys_env_path().read_text(encoding="utf-8")
    assert "AWS_BEARER_TOKEN_BEDROCK" not in os.environ
    assert config_module.DEFAULT_CONFIG.AWS_BEARER_TOKEN_BEDROCK == ""
    assert model_info.bedrock_credentials_configured() is False


# --------------------------------------------------------------------------
# Live Bedrock (only with real credentials in the environment)
# --------------------------------------------------------------------------

_LIVE_CREDENTIALS = bool(os.environ.get("AWS_BEARER_TOKEN_BEDROCK")) or bool(
    os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY")
)


@pytest.mark.skipif(not _LIVE_CREDENTIALS, reason="no Bedrock credentials in the environment")
def test_live_bedrock_generation(monkeypatch) -> None:
    """One real turn on Bedrock's Claude Haiku 4.5 cross-region profile."""
    monkeypatch.delenv("ANTHROPIC_BEDROCK_BASE_URL", raising=False)
    adapter = model_info.model("bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0")
    assert isinstance(adapter, AnthropicModel) and adapter.bedrock
    adapter.initialize("Reply with the single word: pong")
    text, _ = adapter.generate()
    assert "pong" in text.lower()


# --------------------------------------------------------------------------
# Wire requests through the real SDK against a local Bedrock endpoint
# --------------------------------------------------------------------------


def _run_turn(
    server: ScriptedAnthropicServer,
    monkeypatch,
    name: str = _OPUS + "-high",
    bearer: str = "bearer-token",
    model_config: dict | None = None,
) -> AnthropicModel:
    """Run one real turn through ``AnthropicBedrock`` against *server*."""
    monkeypatch.setenv("ANTHROPIC_BEDROCK_BASE_URL", server.base_url)
    adapter = AnthropicModel(name, api_key=bearer, bedrock=True, model_config=model_config)
    adapter.initialize("Say ok.")
    adapter.generate()
    return adapter


def test_bearer_token_turn(monkeypatch) -> None:
    """Model id in the URL, token as a Bearer header, Anthropic-only fields gone."""
    _set_aws(monkeypatch, AWS_REGION="eu-west-1")
    config = {"service_tier": "auto", "system_instruction": "Be terse."}
    with _bedrock_server("ok from bedrock") as server:
        adapter = _run_turn(server, monkeypatch, model_config=config)
        path = server.request_paths[-1]
        headers = server.request_headers[-1]
        body = server.requests[-1]
    assert path == f"/model/{_OPUS_ID}/invoke-with-response-stream"
    assert headers["authorization"] == "Bearer bearer-token"
    assert "model" not in body
    assert "service_tier" not in body
    assert "cache_control" not in body
    assert body["anthropic_version"] == "bedrock-2023-05-31"
    assert body["anthropic_beta"] == ["interleaved-thinking-2025-05-14"]
    assert body["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert body["output_config"] == {"effort": "high"}
    assert body["max_tokens"] == 65536
    # The explicit cache breakpoint on the system prefix is Bedrock-supported and kept.
    assert body["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert adapter.conversation[-1]["role"] == "assistant"
    assert adapter.conversation[-1]["content"][0]["text"] == "ok from bedrock"


def test_sigv4_turn_with_key_pair(monkeypatch) -> None:
    """Without a bearer token the request is SigV4-signed with the key pair and region."""
    _set_aws(
        monkeypatch,
        AWS_ACCESS_KEY_ID="AKIATESTKEY",
        AWS_SECRET_ACCESS_KEY="test-secret",
        AWS_SESSION_TOKEN="session-token",
        AWS_REGION="ap-northeast-1",
    )
    with _bedrock_server() as server:
        _run_turn(server, monkeypatch, bearer="")
        headers = server.request_headers[-1]
    auth = headers["authorization"]
    assert auth.startswith("AWS4-HMAC-SHA256 Credential=AKIATESTKEY/")
    assert "/ap-northeast-1/bedrock/aws4_request" in auth
    assert headers["x-amz-security-token"] == "session-token"


def test_region_selects_endpoint_and_defaults_to_us_east_1(monkeypatch) -> None:
    """The client's endpoint follows AWS_REGION; unset means us-east-1."""
    monkeypatch.delenv("ANTHROPIC_BEDROCK_BASE_URL", raising=False)
    _set_aws(monkeypatch, AWS_BEARER_TOKEN_BEDROCK="b")
    adapter = AnthropicModel(_OPUS, api_key="b", bedrock=True)
    adapter.initialize("hi")
    assert str(adapter.client.base_url).startswith("https://bedrock-runtime.us-east-1.amazonaws.com")
    _set_aws(monkeypatch, AWS_BEARER_TOKEN_BEDROCK="b", AWS_REGION="eu-central-1")
    adapter.initialize("hi")
    assert str(adapter.client.base_url).startswith("https://bedrock-runtime.eu-central-1.amazonaws.com")


def test_client_reused_until_credentials_change(monkeypatch) -> None:
    """initialize() keeps the client across runs and rebuilds it only on a credential change."""
    monkeypatch.delenv("ANTHROPIC_BEDROCK_BASE_URL", raising=False)
    _set_aws(monkeypatch, AWS_ACCESS_KEY_ID="AKIA1", AWS_SECRET_ACCESS_KEY="s1")
    adapter = AnthropicModel(_OPUS, api_key="", bedrock=True)
    adapter.initialize("one")
    first = adapter.client
    adapter.initialize("two")
    assert adapter.client is first
    _set_aws(monkeypatch, AWS_ACCESS_KEY_ID="AKIA2", AWS_SECRET_ACCESS_KEY="s2")
    adapter.initialize("three")
    assert adapter.client is not first


def test_keep_prompt_cache_warm_pings_bedrock(monkeypatch) -> None:
    """The keep-alive ping is sent on Bedrock only when a cache breakpoint exists.

    Bedrock has no top-level ``cache_control``; its only breakpoint is the
    one on the system prefix.  So the ping goes out with a system prompt,
    and is skipped (no request at all) with caching off or without a
    system prompt, where every ping would be a full-price request that
    warms nothing.
    """
    _set_aws(monkeypatch)
    with _bedrock_server() as server:
        adapter = _run_turn(server, monkeypatch, model_config={"system_instruction": "Be terse."})
        before = len(server.requests)
        assert adapter.keep_prompt_cache_warm({}) is not None
        ping = server.requests[-1]
        assert len(server.requests) == before + 1
        assert "cache_control" not in ping
        assert ping["system"][0]["cache_control"] == {"type": "ephemeral"}
        assert ping["messages"][-1]["role"] == "user"
        cold = AnthropicModel(
            _OPUS, api_key="t", bedrock=True, model_config={"enable_cache": False}
        )
        cold.initialize("hi")
        assert cold.keep_prompt_cache_warm({}) is None
        no_system = AnthropicModel(
            _OPUS, api_key="t", bedrock=True, model_config={"enable_cache": True}
        )
        no_system.initialize("hi")
        assert no_system.keep_prompt_cache_warm({}) is None
        assert len(server.requests) == before + 1


def test_keep_prompt_cache_warm_sees_a_breakpoint_supplied_through_extra_body(
    monkeypatch,
) -> None:
    """A cached system prompt configured via ``extra_body`` is kept warm too.

    The SDK merges ``model_config["extra_body"]`` over the outer request
    fields, so ``extra_body["system"]`` with a ``cache_control`` block is a
    genuine breakpoint on the wire even though the outer ``system`` kwarg
    is absent.  Two adapters sending byte-identical bodies (one through
    ``system_instruction``, one through ``extra_body``) must both ping;
    an ``extra_body`` system without a breakpoint still must not.
    """
    _set_aws(monkeypatch)
    with _bedrock_server() as server:
        normal = _run_turn(server, monkeypatch, model_config={"system_instruction": "Be terse."})
        first = server.requests[-1]
        explicit = _run_turn(
            server, monkeypatch, model_config={"extra_body": {"system": first["system"]}}
        )
        assert server.requests[-1] == first
        before = len(server.requests)
        assert normal.keep_prompt_cache_warm({}) is not None
        assert explicit.keep_prompt_cache_warm({}) is not None
        assert len(server.requests) == before + 2
        assert server.requests[-1]["system"] == server.requests[-2]["system"] == first["system"]
        plain = AnthropicModel(
            _OPUS, api_key="t", bedrock=True, model_config={"extra_body": {"system": "Be terse."}}
        )
        plain.initialize("hi")
        assert plain.keep_prompt_cache_warm({}) is None
        assert len(server.requests) == before + 2


def test_concurrent_initialize_tolerates_empty_exported_bearer_token(monkeypatch) -> None:
    """Two tasks building Bedrock clients at once both survive an empty exported token.

    ``_bedrock_client`` unsets an empty ``AWS_BEARER_TOKEN_BEDROCK`` so the
    SDK does not take ``""`` for a token.  Two task threads (two chats, or
    ``run_parallel`` sub-agents) can both read the empty value before
    either removes it; the loser of that race used to die with
    ``KeyError``.  The interleaving is pinned twice, by wrappers around
    the real stdlib functions: ``os.environ.get`` holds each thread on a
    barrier once, right after its first read of that variable, so both
    have read it before either removes it; and ``os.unsetenv`` (the first
    thing ``os.environ.__delitem__`` does, before it drops the key from
    its dict) holds each thread once more, so both are inside the
    deletion before either finishes it.  The second barrier is the one
    ``os.environ.pop(key, None)`` lost: it reads again and then deletes
    outside its ``KeyError`` handler.
    """
    monkeypatch.delenv("ANTHROPIC_BEDROCK_BASE_URL", raising=False)
    _set_aws(monkeypatch, AWS_ACCESS_KEY_ID="AKIA", AWS_SECRET_ACCESS_KEY="secret")
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "")
    real_get = os.environ.get
    real_unsetenv = os.unsetenv
    both_have_read = threading.Barrier(2, timeout=10)
    both_are_deleting = threading.Barrier(2, timeout=10)
    synced = threading.local()

    def get_then_wait_for_the_other_reader(key: str, default: Any = None) -> Any:
        value = real_get(key, default)
        if key == "AWS_BEARER_TOKEN_BEDROCK" and not getattr(synced, "read", False):
            synced.read = True
            both_have_read.wait()
        return value

    def wait_for_the_other_deleter_then_unsetenv(key: bytes) -> None:
        if key == b"AWS_BEARER_TOKEN_BEDROCK" and not getattr(synced, "deleting", False):
            synced.deleting = True
            both_are_deleting.wait()
        real_unsetenv(key)

    monkeypatch.setattr(os, "unsetenv", wait_for_the_other_deleter_then_unsetenv)
    errors: list[BaseException] = []

    def build_client() -> None:
        try:
            AnthropicModel(_OPUS, api_key="", bedrock=True).initialize("hi")
        except BaseException as exc:
            errors.append(exc)

    os.environ.get = get_then_wait_for_the_other_reader  # type: ignore[method-assign]
    try:
        threads = [threading.Thread(target=build_client) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
    finally:
        del os.environ.get  # type: ignore[method-assign]
    assert not any(thread.is_alive() for thread in threads)
    assert errors == []
    assert "AWS_BEARER_TOKEN_BEDROCK" not in os.environ


def test_direct_anthropic_route_unchanged(monkeypatch) -> None:
    """The direct route still sends top-level cache_control and service_tier."""
    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "")
    with ScriptedAnthropicServer() as server:
        monkeypatch.setenv("ANTHROPIC_BASE_URL", server.base_url)
        adapter = AnthropicModel(
            "claude-opus-4-6-high", api_key="k", model_config={"service_tier": "auto"}
        )
        adapter.initialize("Say ok.")
        adapter.generate()
        body = server.requests[-1]
        path = server.request_paths[-1]
    assert path == "/v1/messages"
    assert body["model"] == "claude-opus-4-6"
    assert body["cache_control"] == {"type": "ephemeral"}
    assert body["service_tier"] == "auto"
    assert body["output_config"] == {"effort": "high"}
    assert body["max_tokens"] == 65536


# --------------------------------------------------------------------------
# Fixes after review
# --------------------------------------------------------------------------


def _reloaded_catalog() -> dict[str, model_info.ModelInfo]:
    """Return the catalog as the next daemon start builds it (MY_MODELS merged, cache rules)."""
    catalog = model_info._load_model_info()
    for name, info in catalog.items():
        model_info._normalize_entry(name, info)
    return catalog


def _live_claude(agent: Any) -> AnthropicModel:
    """Return *agent*'s live model, which must be the Claude adapter."""
    live = agent.model
    assert isinstance(live, AnthropicModel)
    return live


def test_set_model_keeps_each_provider_on_its_own_credential(monkeypatch) -> None:
    """Switching Claude direct <-> Bedrock never carries the other provider's key over.

    Both are the ``anthropic`` family (same conversation and config), but
    the Anthropic API key authenticates nothing on Bedrock and the Bedrock
    token nothing on the Anthropic API.  A same-provider switch still
    keeps a per-task key.
    """
    from kiss.agents.sorcar.sorcar_agent import SorcarAgent

    monkeypatch.delenv("ANTHROPIC_BEDROCK_BASE_URL", raising=False)
    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "")
    monkeypatch.setattr(config_module.DEFAULT_CONFIG, "ANTHROPIC_API_KEY", "anthropic-token")
    _set_aws(monkeypatch, AWS_BEARER_TOKEN_BEDROCK="bedrock-token")
    agent: Any = SorcarAgent("test-bedrock-switch")
    agent._use_web_tools = False
    set_model = next(t for t in agent._get_tools() if callable(t) and t.__name__ == "set_model")
    agent.model = model_info.model("claude-opus-4-6")
    agent.model_name = "claude-opus-4-6"
    agent.model.initialize("hi")

    set_model(_OPUS)
    live = _live_claude(agent)
    assert live.model_name == _OPUS and live.bedrock is True
    assert live.api_key == "bedrock-token"
    assert live.client.api_key == "bedrock-token"

    set_model("claude-opus-4-6")
    live = _live_claude(agent)
    assert live.bedrock is False
    assert live.api_key == "anthropic-token"
    assert live.client.api_key == "anthropic-token"

    live.api_key = "task-specific-anthropic-key"
    set_model("claude-sonnet-4-6")
    live = _live_claude(agent)
    assert live.api_key == "task-specific-anthropic-key"
    assert live.client.api_key == "task-specific-anthropic-key"


def test_bedrock_id_the_catalog_lists_only_on_openrouter_is_priced_by_that_entry(
    monkeypatch,
) -> None:
    """Opus 4.1 has no ``claude-*`` entry; its Bedrock id resolves to the OpenRouter twin."""
    from kiss.core.models.anthropic_model import (
        _supports_extended_thinking,
        _uses_adaptive_thinking,
    )

    name = "bedrock/anthropic.claude-opus-4-1-20250805-v1:0"
    twin = "openrouter/anthropic/claude-opus-4.1"
    assert "claude-opus-4-1" not in model_info.MODEL_INFO
    assert "claude-opus-4-1-20250805" not in model_info.MODEL_INFO
    assert model_info.bedrock_catalog_name(name) == twin
    assert model_info.bedrock_claude_name(name) == "claude-opus-4-1-20250805"
    assert model_info.is_known_model(name)
    assert model_info.get_max_context_length(name) == model_info.get_max_context_length(twin)
    assert model_info.calculate_cost(name, 1000, 1000, 100, 10) == model_info.calculate_cost(
        twin, 1000, 1000, 100, 10
    )
    assert model_info.calculate_cost(name, 1_000_000, 0) == 15.0
    assert model_info.openrouter_twin(name) == twin
    monkeypatch.setattr(config_module.DEFAULT_CONFIG, "OPENROUTER_API_KEY", "or-key")
    assert model_info.get_fallback_model(name) == twin
    seed = model_info._seed_new_custom_entry(name)
    assert seed["input_price_per_1M"] == 15.0 and seed["output_price_per_1M"] == 75.0
    assert seed["cache_read_price_per_1M"] == 1.5
    # The adapter sends Bedrock's own id and judges capabilities by the Claude name:
    # Opus 4.1 thinks with ``enabled`` (pre-4.6), not ``adaptive``.
    _set_aws(monkeypatch, AWS_BEARER_TOKEN_BEDROCK="b")
    adapter = model_info.model(name)
    assert isinstance(adapter, AnthropicModel)
    assert adapter._api_model_name == "anthropic.claude-opus-4-1-20250805-v1:0"
    assert adapter._catalog_model_name == "claude-opus-4-1-20250805"
    assert _supports_extended_thinking(adapter._catalog_model_name)
    assert not _uses_adaptive_thinking(adapter._catalog_model_name)


def test_persisted_bedrock_entry_bills_like_its_claude_entry_after_restart(
    monkeypatch, tmp_path
) -> None:
    """A Custom-Models ``bedrock/`` entry keeps its geo-scaled cache prices across a restart."""
    monkeypatch.setattr(model_info, "USER_MY_MODELS_PATH", tmp_path / "MY_MODELS.json")
    assert model_info.save_custom_model(_OPUS) is None
    entry = json.loads((tmp_path / "MY_MODELS.json").read_text())[_OPUS]
    claude = model_info.MODEL_INFO["claude-opus-4-6"]
    assert (claude.cache_read_price_per_1M, claude.cache_write_1h_price_per_1M) == (0.5, 10.0)
    assert entry["cache_read_price_per_1M"] == pytest.approx(0.55)
    # Stored rounded past float noise: 6.875, not 6.875000000000001.
    assert entry["cache_write_price_per_1M"] == 6.875
    assert claude.cache_write_price_per_1M == 6.25
    assert entry["cache_write_1h_price_per_1M"] == pytest.approx(11.0)
    monkeypatch.setattr(model_info, "MODEL_INFO", _reloaded_catalog())
    million = 1_000_000
    for usage in (
        (0, 0, million, 0, 0),  # cache reads
        (0, 0, 0, million, 0),  # 5-minute cache writes
        (0, 0, 0, 0, million),  # 1-hour cache writes
        (million, million, 0, 0, 0),
    ):
        assert model_info.calculate_cost(_OPUS, *usage) == pytest.approx(
            1.1 * model_info.calculate_cost("claude-opus-4-6", *usage)
        )
    assert model_info.calculate_cost(_OPUS, 0, 0, million, 0, 0) == pytest.approx(0.55)


def test_hand_written_bedrock_entry_gets_the_anthropic_cache_rules(monkeypatch, tmp_path) -> None:
    """A MY_MODELS.json ``bedrock/`` entry with base prices only gets Anthropic's cache rates."""
    name = "bedrock/global.anthropic.claude-opus-5-5-v1:0"
    haiku = "bedrock/global.anthropic.claude-haiku-5-5-v1:0"
    path = tmp_path / "MY_MODELS.json"
    path.write_text(
        json.dumps(
            {
                name: {"context_length": 500000, "input_price_per_1M": 4.0,
                       "output_price_per_1M": 20.0},
                haiku: {"context_length": 1000000, "input_price_per_1M": 0.1,
                        "output_price_per_1M": 0.5},
            }
        )
    )
    monkeypatch.setattr(model_info, "USER_MY_MODELS_PATH", path)
    monkeypatch.setattr(model_info, "MODEL_INFO", _reloaded_catalog())
    info = model_info.MODEL_INFO[name]
    # Opus 5.5 reads at 0.05x, not the generic 0.1x: the family is read off the Claude name.
    assert info.cache_read_price_per_1M == pytest.approx(0.2)
    assert info.cache_write_price_per_1M == pytest.approx(5.0)
    assert info.cache_write_1h_price_per_1M == pytest.approx(8.0)
    # Haiku 5.5's long-prompt uplift applies to the Bedrock name too.
    long_prompt = model_info.calculate_cost(haiku, 200_000, 0)
    assert long_prompt == pytest.approx(model_info.calculate_cost("claude-haiku-5-5", 200_000, 0))
    assert long_prompt == pytest.approx(0.1)


def test_geographic_entry_seeded_at_the_global_rate_is_repriced(monkeypatch, tmp_path) -> None:
    """A ``us.`` registry entry holding its Claude entry's prices (an old seed) is repriced.

    Before the geographic premium was accounted for, Custom Models saved
    the ``claude-*`` prices verbatim under the Bedrock name.  Such an
    entry is recognised by its input and output prices equalling the
    catalog entry's and billed at the premium, at the next start and
    when looked up hot; a hand-written entry with its own prices is
    left alone, and the already-scaled entry a new save writes is not
    scaled again.
    """
    path = tmp_path / "MY_MODELS.json"
    claude = model_info.MODEL_INFO["claude-opus-4-6"]
    old_seed = {
        "context_length": claude.context_length,
        "input_price_per_1M": claude.input_price_per_1M,
        "output_price_per_1M": claude.output_price_per_1M,
        "cache_read_price_per_1M": claude.cache_read_price_per_1M,
        "cache_write_price_per_1M": claude.cache_write_price_per_1M,
        "cache_write_1h_price_per_1M": claude.cache_write_1h_price_per_1M,
    }
    own = "bedrock/eu.anthropic.claude-sonnet-4-6-v1:0"
    path.write_text(
        json.dumps(
            {
                _OPUS: old_seed,
                own: {
                    "context_length": 1000,
                    "input_price_per_1M": 3.3,
                    "output_price_per_1M": 9.0,
                },
            }
        )
    )
    monkeypatch.setattr(model_info, "USER_MY_MODELS_PATH", path)
    million = 1_000_000
    # Hot lookup (entry saved after start-up; the catalog key wins and is scaled once).
    assert model_info.calculate_cost(_OPUS, million, 0) == pytest.approx(5.5)
    monkeypatch.setattr(model_info, "MODEL_INFO", _reloaded_catalog())
    info = model_info.MODEL_INFO[_OPUS]
    assert info.input_price_per_1M == pytest.approx(5.5)
    assert info.output_price_per_1M == pytest.approx(27.5)
    assert info.cache_read_price_per_1M == pytest.approx(0.55)
    assert info.cache_write_price_per_1M == pytest.approx(6.875)
    assert info.cache_write_1h_price_per_1M == pytest.approx(11.0)
    assert model_info.calculate_cost(_OPUS, million, million, million) == pytest.approx(
        5.5 + 27.5 + 0.55
    )
    assert model_info.calculate_cost(_OPUS + "-high", million, 0) == pytest.approx(5.5)
    # The catalog entry is untouched and a hand-written entry keeps its own prices.
    assert model_info.MODEL_INFO["claude-opus-4-6"].input_price_per_1M == 5.0
    assert model_info.calculate_cost(own, million, million) == pytest.approx(12.3)
    # A fresh save writes the scaled prices, which a reload does not scale again.
    fresh = "bedrock/us.anthropic.claude-fable-5-1-v1:0"
    assert model_info.save_custom_model(fresh) is None
    assert json.loads(path.read_text())[fresh]["input_price_per_1M"] == pytest.approx(11.0)
    monkeypatch.setattr(model_info, "MODEL_INFO", _reloaded_catalog())
    assert model_info.calculate_cost(fresh, million, 0) == pytest.approx(11.0)
    assert model_info.calculate_cost(fresh, 0, 0, million) == pytest.approx(0.275)


def test_cold_import_reprices_old_seeds_and_keeps_unknown_bedrock_entries(tmp_path) -> None:
    """A fresh interpreter loads a registry of ``bedrock/`` entries and prices them right.

    The import-time normalization runs before the whole module is
    defined, so it must not reach helpers declared further down: a
    ``bedrock/`` id of a Claude model no catalog knows (kept at its own
    prices) and an old global-rate seed (repriced) must both load.
    """
    import subprocess
    import sys

    home = tmp_path / "home"
    (home / ".kiss").mkdir(parents=True)
    claude = model_info.MODEL_INFO["claude-opus-4-6"]
    unknown = "bedrock/us.anthropic.claude-opus-9-9-v1:0"
    (home / ".kiss" / "MY_MODELS.json").write_text(
        json.dumps(
            {
                _OPUS: {
                    "context_length": claude.context_length,
                    "input_price_per_1M": claude.input_price_per_1M,
                    "output_price_per_1M": claude.output_price_per_1M,
                },
                unknown: {"context_length": 1000, "input_price_per_1M": 7.0,
                          "output_price_per_1M": 70.0},
            }
        )
    )
    env = dict(os.environ)
    env["HOME"] = str(home)
    env["KISS_HOME"] = str(home / ".kiss")
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json\n"
            "from kiss.core.models.model_info import calculate_cost\n"
            f"print(json.dumps([calculate_cost({_OPUS!r}, 1000000, 1000000, 1000000),"
            f" calculate_cost({unknown!r}, 1000000, 1000000)]))\n",
        ],
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    repriced, kept = json.loads(proc.stdout)
    assert repriced == pytest.approx(5.5 + 27.5 + 0.55)
    assert kept == pytest.approx(77.0)


def test_explicit_anthropic_only_fields_never_reach_bedrock(monkeypatch) -> None:
    """A configured ``cache_control``/``service_tier`` is dropped as a kwarg and in extra_body."""
    _set_aws(monkeypatch)
    config = {
        "cache_control": {"type": "ephemeral"},
        "extra_body": {
            "cache_control": {"type": "ephemeral"},
            "service_tier": "auto",
            "metadata": {"user_id": "u1"},
        },
    }
    with _bedrock_server() as server:
        _run_turn(server, monkeypatch, model_config=config)
        body = server.requests[-1]
    assert "cache_control" not in body
    assert "service_tier" not in body
    assert body["metadata"] == {"user_id": "u1"}


def test_direct_route_keeps_an_explicit_cache_control(monkeypatch) -> None:
    """On the Anthropic API a caller's own top-level ``cache_control`` is sent as given."""
    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "")
    with ScriptedAnthropicServer() as server:
        monkeypatch.setenv("ANTHROPIC_BASE_URL", server.base_url)
        adapter = AnthropicModel(
            "claude-opus-4-6",
            api_key="k",
            model_config={"cache_control": {"type": "ephemeral", "ttl": "1h"}},
        )
        adapter.initialize("Say ok.")
        adapter.generate()
        body = server.requests[-1]
    assert body["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


def test_empty_bearer_token_in_the_environment_counts_as_unset(monkeypatch) -> None:
    """``AWS_BEARER_TOKEN_BEDROCK=''`` exported in the shell must not disable the key pair."""
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "")
    _set_aws(
        monkeypatch,
        AWS_ACCESS_KEY_ID="AKIATESTKEY",
        AWS_SECRET_ACCESS_KEY="test-secret",
        AWS_REGION="us-west-2",
    )
    assert model_info.bedrock_credentials_configured() is True
    with _bedrock_server() as server:
        monkeypatch.setenv("ANTHROPIC_BEDROCK_BASE_URL", server.base_url)
        adapter = model_info.model(_OPUS)
        assert isinstance(adapter, AnthropicModel) and adapter.api_key == ""
        adapter.initialize("Say ok.")
        adapter.generate()
        auth = server.request_headers[-1]["authorization"]
        # keep_prompt_cache_warm() sends through ``with_options(...)``, a
        # copy built by the SDK constructor: it must not trip over the
        # empty variable either (the empty export is unset for the SDK).
        copy = adapter.client.with_options(max_retries=0)
    assert adapter.client.api_key is None
    assert copy.api_key is None and copy.aws_access_key == "AKIATESTKEY"
    assert "AWS_BEARER_TOKEN_BEDROCK" not in os.environ
    assert auth.startswith("AWS4-HMAC-SHA256 Credential=AKIATESTKEY/")
    assert "/us-west-2/bedrock/aws4_request" in auth


def test_bedrock_only_configuration_has_default_and_fast_models(monkeypatch, tmp_path) -> None:
    """With Bedrock as the only configured provider the registered Bedrock models are chosen."""
    for key in (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "OPENROUTER_API_KEY",
        "TOGETHER_API_KEY",
    ):
        monkeypatch.setattr(config_module.DEFAULT_CONFIG, key, "")
    (tmp_path / "bin").mkdir()
    monkeypatch.setenv("PATH", str(tmp_path / "bin"))  # no claude / codex CLI
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(model_info, "USER_MY_MODELS_PATH", tmp_path / "MY_MODELS.json")
    haiku = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"
    assert model_info.save_custom_model(_OPUS) is None
    assert model_info.save_custom_model(haiku) is None
    monkeypatch.setattr(model_info, "MODEL_INFO", _reloaded_catalog())

    _set_aws(monkeypatch)
    assert model_info.get_default_model() == "No model"
    assert model_info.get_fast_model() == "No model"
    _set_aws(monkeypatch, AWS_BEARER_TOKEN_BEDROCK="bearer")
    available = model_info.get_available_models()
    assert _OPUS in available and haiku in available
    assert all(name.startswith("bedrock/") for name in available)
    assert model_info.get_default_model() == _OPUS
    assert model_info.get_fast_model() == haiku
    # A configured first-party key still takes precedence over Bedrock.
    monkeypatch.setattr(config_module.DEFAULT_CONFIG, "GEMINI_API_KEY", "g")
    assert model_info.get_default_model() == "gemini-3.8-flash"


def test_no_registered_bedrock_model_means_no_bedrock_default(monkeypatch, tmp_path) -> None:
    """Bedrock credentials without a registered ``bedrock/`` model fall through to the CLIs."""
    for key in (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "OPENROUTER_API_KEY",
        "TOGETHER_API_KEY",
    ):
        monkeypatch.setattr(config_module.DEFAULT_CONFIG, key, "")
    (tmp_path / "bin").mkdir()
    monkeypatch.setenv("PATH", str(tmp_path / "bin"))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(
        model_info,
        "MODEL_INFO",
        {k: v for k, v in model_info.MODEL_INFO.items() if not k.startswith("bedrock/")},
    )
    _set_aws(monkeypatch, AWS_BEARER_TOKEN_BEDROCK="bearer")
    assert model_info.registered_bedrock_model(cheapest=True) is None
    assert model_info.get_default_model() == "No model"
