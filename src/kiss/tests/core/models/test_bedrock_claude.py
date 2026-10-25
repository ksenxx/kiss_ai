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
* prompt caching without the top-level field: the breakpoint on the
  system prefix plus a moving one on the last block of the conversation,
  so each turn reads the earlier turns from cache;
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
                data = line[len("data: ") :].encode()
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


def test_ambient_irsa_token_configures_bedrock(monkeypatch, tmp_path) -> None:
    """An EKS/IRSA web-identity token counts only together with its role ARN."""
    _set_aws(monkeypatch)
    token = tmp_path / "token"
    monkeypatch.setenv("AWS_WEB_IDENTITY_TOKEN_FILE", str(token))
    monkeypatch.setenv("AWS_ROLE_ARN", "arn:aws:iam::123456789012:role/s10s")
    assert model_info.bedrock_credentials_configured() is False  # token file missing
    token.write_text("jwt", encoding="utf-8")
    assert model_info.bedrock_credentials_configured() is True
    monkeypatch.delenv("AWS_ROLE_ARN")  # botocore raises InvalidConfigError here
    assert model_info.bedrock_credentials_configured() is False
    # The profile form of the same configuration (role_arn + token file).
    config = tmp_path / "config"
    monkeypatch.setenv("AWS_CONFIG_FILE", str(config))
    monkeypatch.delenv("AWS_WEB_IDENTITY_TOKEN_FILE")
    config.write_text(
        f"[default]\nrole_arn = arn:aws:iam::1:role/r\nweb_identity_token_file = {token}\n",
        encoding="utf-8",
    )
    assert model_info.bedrock_credentials_configured() is True
    # The environment wins over the profile, even when it is exported empty.
    monkeypatch.setenv("AWS_WEB_IDENTITY_TOKEN_FILE", "")
    assert model_info.bedrock_credentials_configured() is False
    config.write_text(
        "[default]\nrole_arn = arn:aws:iam::1:role/r\nweb_identity_token_file = /nonexistent\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("AWS_WEB_IDENTITY_TOKEN_FILE", str(token))
    assert model_info.bedrock_credentials_configured() is True
    # IRSA comes before the shared files in the chain: a stale half pair there is irrelevant.
    credentials = tmp_path / "credentials"
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(credentials))
    credentials.write_text("[default]\naws_access_key_id = stale\n", encoding="utf-8")
    assert model_info.bedrock_credentials_configured() is True
    monkeypatch.delenv("AWS_WEB_IDENTITY_TOKEN_FILE")
    config.unlink()
    assert model_info.bedrock_credentials_configured() is False  # now the half pair aborts


def test_ambient_container_and_env_pair_configure_bedrock(monkeypatch) -> None:
    """ECS container credentials and an exported key pair count."""
    _set_aws(monkeypatch)
    assert model_info.bedrock_credentials_configured() is False
    monkeypatch.setenv("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", "/v2/credentials/id")
    assert model_info.bedrock_credentials_configured() is True
    monkeypatch.delenv("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI")
    monkeypatch.setenv("AWS_CONTAINER_CREDENTIALS_FULL_URI", "http://169.254.170.23/v1/credentials")
    assert model_info.bedrock_credentials_configured() is True
    monkeypatch.setenv("AWS_CONTAINER_CREDENTIALS_FULL_URI", "")  # exported empty: no endpoint
    assert model_info.bedrock_credentials_configured() is False
    monkeypatch.delenv("AWS_CONTAINER_CREDENTIALS_FULL_URI")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA")
    assert model_info.bedrock_credentials_configured() is False  # half a pair: botocore refuses
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    assert model_info.bedrock_credentials_configured() is True


def test_ambient_profile_configures_bedrock(monkeypatch, tmp_path) -> None:
    """``AWS_PROFILE`` counts when the selected profile can source credentials."""
    _set_aws(monkeypatch)
    config = tmp_path / "config"
    monkeypatch.setenv("AWS_CONFIG_FILE", str(config))
    monkeypatch.setenv("AWS_PROFILE", "dev")
    assert model_info.bedrock_credentials_configured() is False  # ProfileNotFound
    config.write_text("[profile dev]\nregion = us-west-2\n", encoding="utf-8")
    assert model_info.bedrock_credentials_configured() is False  # region alone
    config.write_text(
        "[profile dev]\nsso_session = corp\n[sso-session corp]\nsso_region = us-east-1\n",
        encoding="utf-8",
    )
    assert model_info.bedrock_credentials_configured() is False  # botocore skips SSO here
    config.write_text(
        "[profile dev]\nsso_session = corp\nsso_account_id = 1\nsso_role_name = r\n",
        encoding="utf-8",
    )
    assert model_info.bedrock_credentials_configured() is True
    config.write_text(
        "[profile dev]\ncredential_process = /usr/bin/get-creds\naws_access_key_id = half\n",
        encoding="utf-8",
    )
    # The process is NOT run, and it precedes the config-file provider that
    # would choke on the half pair.
    assert model_info.bedrock_credentials_configured() is True
    config.write_text(
        "[profile dev]\nrole_arn = arn:aws:iam::1:role/r\nsource_profile = base\n", encoding="utf-8"
    )
    assert model_info.bedrock_credentials_configured() is True


def test_ambient_shared_files_follow_botocore(monkeypatch, tmp_path) -> None:
    """The shared files are read the way botocore reads them, not by keyword."""
    _set_aws(monkeypatch)
    config = tmp_path / "config"
    credentials = tmp_path / "credentials"
    monkeypatch.setenv("AWS_CONFIG_FILE", str(config))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(credentials))
    config.write_text(
        "[default]\nregion = us-west-2\n# credential_process = unused\n", encoding="utf-8"
    )
    assert model_info.bedrock_credentials_configured() is False  # a comment is not a directive
    credentials.write_text(
        "[prod]\naws_access_key_id = AKIA\naws_secret_access_key = s\n", encoding="utf-8"
    )
    assert model_info.bedrock_credentials_configured() is False  # an unselected profile
    credentials.write_text(
        "[default]\nAWS_ACCESS_KEY_ID = AKIA\nAWS_SECRET_ACCESS_KEY = s\n", encoding="utf-8"
    )
    assert model_info.bedrock_credentials_configured() is True  # botocore lower-cases keys
    credentials.write_text("[default\nbroken", encoding="utf-8")
    assert model_info.bedrock_credentials_configured() is False  # unparsable file
    # ``$HOME`` in an override is expanded; an EMPTY override is a missing file.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", "$HOME/credentials")
    credentials.write_text(
        "[default]\naws_access_key_id = AKIA\naws_secret_access_key = s\n", encoding="utf-8"
    )
    assert model_info.bedrock_credentials_configured() is True
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", "")
    assert model_info.bedrock_credentials_configured() is False


def test_ambient_legacy_credential_files_configure_bedrock(monkeypatch, tmp_path) -> None:
    """The legacy EC2 credential file and boto config are part of the chain too."""
    _set_aws(monkeypatch)
    legacy = tmp_path / "ec2-credentials"
    monkeypatch.setenv("AWS_CREDENTIAL_FILE", str(legacy))
    assert model_info.bedrock_credentials_configured() is False
    legacy.write_text("AWSAccessKeyId=AKIA\n", encoding="utf-8")
    assert model_info.bedrock_credentials_configured() is False  # botocore: KeyError
    legacy.write_text("AWSAccessKeyId=AKIA\nAWSSecretKey=s\n", encoding="utf-8")
    assert model_info.bedrock_credentials_configured() is True
    monkeypatch.setenv("AWS_CREDENTIAL_EXPIRATION", "not-a-date")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "s")
    assert model_info.bedrock_credentials_configured() is False  # botocore: ParserError
    monkeypatch.delenv("AWS_CREDENTIAL_EXPIRATION")
    assert model_info.bedrock_credentials_configured() is True
    for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        monkeypatch.delenv(name)
    monkeypatch.delenv("AWS_CREDENTIAL_FILE")
    boto = tmp_path / "boto.cfg"
    monkeypatch.setenv("BOTO_CONFIG", str(boto))
    boto.write_text(
        "[Credentials]\naws_access_key_id = AKIA\naws_secret_access_key = s\n", encoding="utf-8"
    )
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

    That is how a Bedrock id the bundled catalog does not list (one of
    another region, or newer than the catalog) reaches the picker: the
    entry copies its Claude entry's context length and prices, is
    labelled Bedrock, and is listed by ``get_available_models()`` once
    Bedrock credentials are configured.
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


@pytest.mark.parametrize(
    ("name", "tokens"),
    [
        ("gpt-audio", {"num_audio_input_tokens": 1000, "num_audio_output_tokens": 1000}),
        ("gemini-2.5-flash-image", {"num_image_output_tokens": 1000}),
    ],
)
def test_custom_entry_of_a_catalog_model_keeps_audio_and_image_prices(
    monkeypatch, tmp_path, name, tokens
) -> None:
    """Saving a catalog model under Custom Models keeps its audio/image rates after reload.

    The loaded entry replaces the catalog's, so a seed without these
    fields would bill audio and image tokens at the text rates.
    """
    before = model_info.calculate_cost(name, 0, 0, **tokens)
    monkeypatch.setattr(model_info, "USER_MY_MODELS_PATH", tmp_path / "MY_MODELS.json")
    assert model_info.save_custom_model(name) is None
    monkeypatch.setattr(model_info, "MODEL_INFO", _reloaded_catalog())
    assert before > 0
    assert model_info.calculate_cost(name, 0, 0, **tokens) == pytest.approx(before)


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

_LIVE_AWS = {field: os.environ.get(field, "") for field in _AWS_FIELDS}
"""The developer's real AWS credentials, captured before any test touches the process.

The autouse ``no_ambient_aws_credentials`` fixture scrubs them from
``os.environ`` for every test, and any earlier test that runs
``save_api_key``/``load_api_keys`` re-syncs ``DEFAULT_CONFIG`` from that
scrubbed environment — so the live test must pin its own credentials
onto the config singleton instead of trusting what is left in it.
"""

_LIVE_CREDENTIALS = bool(_LIVE_AWS["AWS_BEARER_TOKEN_BEDROCK"]) or bool(
    _LIVE_AWS["AWS_ACCESS_KEY_ID"] and _LIVE_AWS["AWS_SECRET_ACCESS_KEY"]
)


@pytest.mark.skipif(not _LIVE_CREDENTIALS, reason="no Bedrock credentials in the environment")
def test_live_bedrock_generation(monkeypatch) -> None:
    """One real turn on Bedrock's Claude Haiku 4.5 cross-region profile."""
    _set_aws(monkeypatch, **_LIVE_AWS)
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
    assert str(adapter.client.base_url).startswith(
        "https://bedrock-runtime.us-east-1.amazonaws.com"
    )
    _set_aws(monkeypatch, AWS_BEARER_TOKEN_BEDROCK="b", AWS_REGION="eu-central-1")
    adapter.initialize("hi")
    assert str(adapter.client.base_url).startswith(
        "https://bedrock-runtime.eu-central-1.amazonaws.com"
    )


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

    Bedrock has no top-level ``cache_control``; its breakpoints are the
    one on the system prefix and the moving one on the last block of the
    conversation.  The ping re-reads the entry the last real request
    wrote, and its conversation breakpoint sits on the last real message
    (the assistant turn whose tool call is running), not on the
    placeholder the ping appends: the placeholder is never written to
    the cache, while the assistant turn is, so the next real request
    reads it instead of writing it.  With caching off there is no
    breakpoint and the ping is skipped (no request at all), as every
    ping would be a full-price request that warms nothing.  Without a
    system prompt the conversation is still cached, so the ping still
    goes out.
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
        assert [m["role"] for m in ping["messages"]] == ["user", "assistant", "user"]
        assert ping["messages"][0]["content"] == "Say ok."
        assert _breakpoint_blocks(ping["messages"]) == [(1, 0)]
        assert ping["messages"][1]["content"][0]["text"] == "ok"
        cold = AnthropicModel(
            _OPUS, api_key="t", bedrock=True, model_config={"enable_cache": False}
        )
        cold.initialize("hi")
        assert cold.keep_prompt_cache_warm({}) is None
        assert len(server.requests) == before + 1
        no_system = AnthropicModel(
            _OPUS, api_key="t", bedrock=True, model_config={"enable_cache": True}
        )
        no_system.initialize("hi")
        assert no_system.keep_prompt_cache_warm({}) is not None
        assert len(server.requests) == before + 2
        assert "system" not in server.requests[-1]
        assert _breakpoint_blocks(server.requests[-1]["messages"]) == [(0, 0)]


def test_keep_prompt_cache_warm_sees_a_breakpoint_supplied_through_extra_body(
    monkeypatch,
) -> None:
    """A cached system prompt configured via ``extra_body`` is kept warm too.

    The SDK merges ``model_config["extra_body"]`` over the outer request
    fields, so ``extra_body["system"]`` with a ``cache_control`` block is a
    genuine breakpoint on the wire even though the outer ``system`` kwarg
    is absent.  Two adapters sending byte-identical bodies (one through
    ``system_instruction``, one through ``extra_body``) must both ping;
    an ``extra_body`` system without a breakpoint pings only for the
    conversation breakpoint, and not at all once caching is off.
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
        assert plain.keep_prompt_cache_warm({}) is not None
        assert len(server.requests) == before + 3
        assert server.requests[-1]["system"] == "Be terse."
        assert _breakpoint_blocks(server.requests[-1]["messages"]) == [(0, 0)]
        off = AnthropicModel(
            _OPUS,
            api_key="t",
            bedrock=True,
            model_config={"extra_body": {"system": "Be terse."}, "enable_cache": False},
        )
        off.initialize("hi")
        assert off.keep_prompt_cache_warm({}) is None
        assert len(server.requests) == before + 3


# --------------------------------------------------------------------------
# Conversation caching: the moving breakpoint on the last message block
# --------------------------------------------------------------------------


def _breakpoint_blocks(messages: list[dict[str, Any]]) -> list[tuple[int, int]]:
    """Return the ``(message index, block index)`` of every ``cache_control`` block."""
    found = []
    for m_index, message in enumerate(messages):
        content = message["content"]
        blocks = content if isinstance(content, list) else []
        for b_index, block in enumerate(blocks):
            if "cache_control" in block:
                found.append((m_index, b_index))
    return found


def test_bedrock_caches_the_conversation_through_a_moving_breakpoint(monkeypatch) -> None:
    """Every Bedrock request marks the last block of its last message.

    Bedrock's InvokeModel rejects the top-level ``cache_control`` that
    gives the direct API automatic caching, so without an explicit
    breakpoint in ``messages`` the whole conversation is billed at the
    full input rate on every turn.  The adapter emulates automatic
    caching: one ``cache_control`` on the last cacheable block of the
    final message, moved forward each turn, so Bedrock reads the earlier
    turns from cache (it looks back twenty blocks from the breakpoint for
    the longest cached prefix) and writes only the new blocks.  Earlier
    messages are re-sent byte for byte, with no stale breakpoint left on
    them, and the adapter's own conversation never carries the marker.
    """
    _set_aws(monkeypatch)
    with _bedrock_server("ok") as server:
        adapter = _run_turn(server, monkeypatch, model_config={"system_instruction": "Be terse."})
        first = server.requests[-1]
        assert first["system"][0]["cache_control"] == {"type": "ephemeral"}
        assert _breakpoint_blocks(first["messages"]) == [(0, 0)]
        assert first["messages"][0]["content"][0] == {
            "type": "text",
            "text": "Say ok.",
            "cache_control": {"type": "ephemeral"},
        }
        # Second turn: a plain-string user message.
        adapter.add_message_to_conversation("user", "Again.")
        adapter.generate()
        second = server.requests[-1]
        assert _breakpoint_blocks(second["messages"]) == [(2, 0)]
        assert second["messages"][2]["content"] == [
            {"type": "text", "text": "Again.", "cache_control": {"type": "ephemeral"}}
        ]
        # The cached prefix is unchanged: the first turn is re-sent without its marker.
        assert second["messages"][0]["content"] == "Say ok."
        assert second["messages"][1] == {
            "role": "assistant",
            "content": [{"type": "text", "text": "ok"}],
        }
        # Third turn: a tool-use reply and its result end in a tool_result block.
        tool_turn = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Running."},
                {"type": "tool_use", "id": "toolu_1", "name": "f", "input": {}},
            ],
        }
        adapter.conversation[-1] = tool_turn
        adapter.add_function_results_to_conversation_and_return([("f", {"result": "done"})])
        adapter.generate()
        third = server.requests[-1]
        assert _breakpoint_blocks(third["messages"]) == [(4, 0)]
        assert third["messages"][4]["content"] == [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_1",
                "content": "done",
                "cache_control": {"type": "ephemeral"},
            }
        ]
        assert third["messages"][:4] == second["messages"][:2] + [
            {"role": "user", "content": "Again."},
            tool_turn,
        ]
    assert not any("cache_control" in json.dumps(message) for message in adapter.conversation)


def test_bedrock_breakpoint_skips_thinking_blocks_and_respects_an_existing_one(
    monkeypatch,
) -> None:
    """Blocks that cannot carry ``cache_control`` are skipped; a given one is kept.

    Thinking blocks cannot carry a breakpoint, so a final message that
    ends in one is marked on its last cacheable block instead; a final
    message that is nothing but thinking gets no marker at all.  A
    message that already carries a breakpoint (a caller-built
    conversation) is sent as is: no second marker is added.
    """
    _set_aws(monkeypatch)
    with _bedrock_server() as server:
        adapter = _run_turn(server, monkeypatch)
        adapter.conversation[-1] = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Prefill."},
                {"type": "thinking", "thinking": "hmm", "signature": "sig"},
            ],
        }
        adapter.generate()
        assert _breakpoint_blocks(server.requests[-1]["messages"]) == [(1, 0)]
        assert server.requests[-1]["messages"][1]["content"][1] == {
            "type": "thinking",
            "thinking": "hmm",
            "signature": "sig",
        }
        adapter.conversation[-1] = {
            "role": "assistant",
            "content": [{"type": "redacted_thinking", "data": "x"}],
        }
        adapter.generate()
        assert _breakpoint_blocks(server.requests[-1]["messages"]) == []
        adapter.conversation[-1] = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Mine.", "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "Tail."},
            ],
        }
        adapter.generate()
        assert _breakpoint_blocks(server.requests[-1]["messages"]) == [(3, 0)]
        assert server.requests[-1]["messages"][3]["content"][1] == {"type": "text", "text": "Tail."}


def test_bedrock_conversation_breakpoint_stays_within_the_limit(monkeypatch) -> None:
    """A request already at four explicit breakpoints gets no fifth.

    Bedrock allows four ``cache_control`` markers per request and rejects
    a fifth.  A caller-built conversation can carry its own markers on
    earlier messages; together with the system breakpoint they count
    toward the limit, and the moving marker is added only while there is
    room for it.
    """
    _set_aws(monkeypatch)
    marked = {"type": "text", "text": "Pinned.", "cache_control": {"type": "ephemeral"}}
    with _bedrock_server() as server:
        adapter = _run_turn(server, monkeypatch, model_config={"system_instruction": "Be terse."})
        adapter.conversation[0] = {"role": "user", "content": [marked, marked]}
        adapter.add_message_to_conversation("user", "Room for one more.")
        adapter.generate()
        assert _breakpoint_blocks(server.requests[-1]["messages"]) == [(0, 0), (0, 1), (2, 0)]
        adapter.conversation[0] = {"role": "user", "content": [marked, marked, marked]}
        adapter.add_message_to_conversation("user", "Full.")
        adapter.generate()
        assert _breakpoint_blocks(server.requests[-1]["messages"]) == [(0, 0), (0, 1), (0, 2)]
        assert server.requests[-1]["messages"][-1]["content"] == "Full."


def test_bedrock_conversation_breakpoint_is_off_with_caching_off(monkeypatch) -> None:
    """``enable_cache: False`` sends the conversation with no breakpoint anywhere."""
    _set_aws(monkeypatch)
    config = {"system_instruction": "Be terse.", "enable_cache": False}
    with _bedrock_server() as server:
        _run_turn(server, monkeypatch, model_config=config)
        body = server.requests[-1]
    assert body["system"] == "Be terse."
    assert "cache_control" not in body
    assert _breakpoint_blocks(body["messages"]) == []


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
        _claude_profile,
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
    assert _supports_extended_thinking(_claude_profile(adapter._catalog_model_name))
    assert not _uses_adaptive_thinking(_claude_profile(adapter._catalog_model_name))


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
                name: {
                    "context_length": 500000,
                    "input_price_per_1M": 4.0,
                    "output_price_per_1M": 20.0,
                },
                haiku: {
                    "context_length": 1000000,
                    "input_price_per_1M": 0.1,
                    "output_price_per_1M": 0.5,
                },
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
                unknown: {
                    "context_length": 1000,
                    "input_price_per_1M": 7.0,
                    "output_price_per_1M": 70.0,
                },
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
    """Bedrock as the only configured provider picks the Bedrock twins of the Anthropic defaults.

    The bundled catalog lists the Bedrock ids of the updater's region; the
    default and fast models are the ``global.`` profiles of the Anthropic
    default and fast models (``claude-opus-5-5``, ``claude-sonnet-5-5``).
    A catalog without those twins (the user registered a few ids by hand)
    falls back to ranking the entries by output price.
    """
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
    assert model_info.get_default_model() == "bedrock/global.anthropic.claude-opus-5-5"
    assert model_info.get_fast_model() == "bedrock/global.anthropic.claude-sonnet-5-5"
    # A configured first-party key still takes precedence over Bedrock.
    monkeypatch.setattr(config_module.DEFAULT_CONFIG, "GEMINI_API_KEY", "g")
    assert model_info.get_default_model() == "gemini-3.8-flash"
    monkeypatch.setattr(config_module.DEFAULT_CONFIG, "GEMINI_API_KEY", "")

    # Only hand-registered ids, none of them a twin of the Anthropic defaults.
    monkeypatch.setattr(
        model_info,
        "MODEL_INFO",
        {k: v for k, v in model_info.MODEL_INFO.items() if k in (_OPUS, haiku)},
    )
    assert model_info.get_default_model() == _OPUS
    assert model_info.get_fast_model() == haiku


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
