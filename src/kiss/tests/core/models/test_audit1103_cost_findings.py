# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Cost audit 2026-11-03: the four defects the gpt-6-astra review found, each fixed.

1. A generated thinking alias (``gpt-6-astra-high``) of a ``MY_MODELS.json``
   model with its own endpoint is billed at that entry's prices (mirrored
   onto the alias), so it must run against that endpoint too:
   :func:`custom_model_config` resolves the alias to its base's entry.
2. A harbor-prefixed name registered under itself with an endpoint
   (``openai/gpt-6-astra`` for a gateway using that id) keeps its identity
   in the :func:`model` factory and its own prices in
   :func:`_lookup_model_info`, instead of the bundled ``gpt-6-astra``'s.
3. Together reports the cached prompt tokens of some models flat on
   ``usage.cached_tokens``: the OpenAI-compatible extractor reads it when
   the nested count is absent.
4. A Claude Code turn whose messages were served at different speeds
   (fast mode fell back to standard) prices each message at its own
   modifiers, not the whole turn at the priced one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from openai.types.completion_usage import CompletionUsage

from kiss.core.models import model_info
from kiss.core.models.claude_code_model import ClaudeCodeModel
from kiss.core.models.model_info import (
    MODEL_INFO,
    ModelInfo,
    _lookup_model_info,
    calculate_cost,
    custom_model_config,
    model,
)
from kiss.core.models.openai_compatible_model import OpenAICompatibleModel

ENDPOINT = "https://gateway.invalid/v1"
CUSTOM = {
    "context_length": 500000,
    "input_price_per_1M": 0.07,
    "output_price_per_1M": 0.15,
    "fc": True,
    "emb": False,
    "gen": True,
    "endpoint": ENDPOINT,
    "api_key": "gw-key",
}
ENDPOINTLESS = {k: v for k, v in CUSTOM.items() if k != "endpoint"}


def _registry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, entries: dict[str, Any]) -> None:
    """Point the live ``MY_MODELS.json`` registry at a private file holding *entries*."""
    path = tmp_path / "MY_MODELS.json"
    path.write_text(json.dumps(entries), encoding="utf-8")
    monkeypatch.setattr(model_info, "USER_MY_MODELS_PATH", path)


# --- 1. generated alias of a custom-endpoint base --------------------------------


def test_generated_alias_runs_against_its_bases_custom_endpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _registry(monkeypatch, tmp_path, {"gpt-6-astra": CUSTOM})
    assert MODEL_INFO["gpt-6-astra-high"].alias_of == "gpt-6-astra"
    assert MODEL_INFO["gpt-6-astra-xhigh"].alias_of == "gpt-6-astra"
    expected = {"base_url": ENDPOINT, "api_key": "gw-key"}
    assert custom_model_config("gpt-6-astra") == expected
    assert custom_model_config("gpt-6-astra-high") == expected
    assert custom_model_config("gpt-6-astra-xhigh") == expected
    # Not an alias of the registered model: nothing to inherit.
    assert custom_model_config("gpt-5.6-sol-high") is None
    assert custom_model_config("no-such-model-high") is None


def test_an_explicit_alias_entry_wins_over_its_base(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    own = {**CUSTOM, "endpoint": "https://other.invalid/v1", "api_key": "other-key"}
    _registry(monkeypatch, tmp_path, {"gpt-6-astra": CUSTOM, "gpt-6-astra-high": own})
    assert custom_model_config("gpt-6-astra-high") == {
        "base_url": "https://other.invalid/v1",
        "api_key": "other-key",
    }
    # An explicit alias entry without an endpoint stays endpoint-less.
    _registry(
        monkeypatch,
        tmp_path,
        {"gpt-6-astra": CUSTOM, "gpt-6-astra-high": ENDPOINTLESS},
    )
    assert custom_model_config("gpt-6-astra-high") is None


# --- 2. harbor-prefixed registered endpoint model --------------------------------


def test_registered_prefixed_endpoint_model_keeps_its_id_and_prices(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    name = "openai/gpt-6-astra"
    assert name not in MODEL_INFO  # hot-added: registered after the catalog was loaded
    _registry(monkeypatch, tmp_path, {name: CUSTOM})
    config = custom_model_config(name)
    assert config == {"base_url": ENDPOINT, "api_key": "gw-key"}
    built = model(name, model_config=config)
    assert isinstance(built, OpenAICompatibleModel)
    assert built.model_name == name
    assert built._api_model_name == name
    assert built.base_url == ENDPOINT
    # The adapter bills under its name: the registered prices, not $10/$50.
    assert calculate_cost(built.model_name, 100_000, 1_000) == pytest.approx(0.00715)
    assert calculate_cost("gpt-6-astra", 100_000, 1_000) == pytest.approx(1.05)
    info = _lookup_model_info(name)
    assert info is not None and info.input_price_per_1M == 0.07


def test_registered_prefixed_model_loaded_at_import_keeps_its_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    name = "openai/gpt-6-astra"
    _registry(monkeypatch, tmp_path, {name: CUSTOM})
    loaded = model_info._build_model_info_entry(CUSTOM)
    assert isinstance(loaded, ModelInfo)
    monkeypatch.setitem(MODEL_INFO, name, loaded)
    built = model(name, model_config=custom_model_config(name))
    assert built.model_name == name
    assert calculate_cost(name, 100_000, 1_000) == pytest.approx(0.00715)


def test_unregistered_or_endpointless_prefixed_names_still_strip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # No registry entry: the harbor prefix is redundant, as before.
    _registry(monkeypatch, tmp_path, {})
    built = model("openai/gpt-6-astra", model_config={"base_url": ENDPOINT, "api_key": "k"})
    assert built.model_name == "gpt-6-astra"
    assert _lookup_model_info("openai/gpt-6-astra") is MODEL_INFO["gpt-6-astra"]
    # Registered WITHOUT an endpoint, or the call not routed to an endpoint:
    # the name routes by its bundled base, as before.
    _registry(monkeypatch, tmp_path, {"openai/gpt-6-astra": ENDPOINTLESS})
    assert custom_model_config("openai/gpt-6-astra") is None
    monkeypatch.setattr(model_info.config_module.DEFAULT_CONFIG, "OPENAI_API_KEY", "sk-test")
    assert model("openai/gpt-6-astra").model_name == "gpt-6-astra"


# --- 3. Together's flat usage.cached_tokens --------------------------------------


class _Response:
    def __init__(self, usage: Any) -> None:
        self.usage = usage


def _oai() -> OpenAICompatibleModel:
    built = model("gpt-6-astra", model_config={"base_url": ENDPOINT, "api_key": "k"})
    assert isinstance(built, OpenAICompatibleModel)
    return built


def _usage(**fields: Any) -> CompletionUsage:
    """A usage object as the SDK parses it from the wire (extra fields kept)."""
    return CompletionUsage.model_validate(fields)


def test_flat_cached_tokens_are_read_when_the_nested_count_is_absent() -> None:
    # https://docs.together.ai/docs/inference/openai-compatibility: some
    # non-reasoning models return ``cached_tokens`` flat on ``usage``; the
    # SDK keeps the extra field.
    usage = _usage(prompt_tokens=1000, completion_tokens=100, total_tokens=1100, cached_tokens=900)
    assert usage.model_extra == {"cached_tokens": 900}
    assert _oai().extract_input_output_token_counts_from_response(_Response(usage)) == (
        100,
        100,
        900,
        0,
    )
    # The cost follows: 100 uncached + 900 cached input tokens, not 1000 uncached.
    assert calculate_cost("gpt-6-astra", 100, 100, 900) < calculate_cost("gpt-6-astra", 1000, 100)


def test_nested_cached_tokens_win_and_the_flat_count_is_never_added() -> None:
    usage = _usage(
        prompt_tokens=1000,
        completion_tokens=100,
        total_tokens=1100,
        prompt_tokens_details={"cached_tokens": 600},
        cached_tokens=900,
    )
    assert _oai().extract_input_output_token_counts_from_response(_Response(usage)) == (
        400,
        100,
        600,
        0,
    )
    # Neither shape: no cache read; a junk flat value is ignored.
    plain = _usage(prompt_tokens=1000, completion_tokens=100, total_tokens=1100)
    assert _oai().extract_input_output_token_counts_from_response(_Response(plain)) == (
        1000,
        100,
        0,
        0,
    )
    junk = _usage(prompt_tokens=1000, completion_tokens=100, total_tokens=1100, cached_tokens="900")
    assert _oai().extract_input_output_token_counts_from_response(_Response(junk)) == (
        1000,
        100,
        0,
        0,
    )


# --- 4. Claude Code: each message at the speed it was served at -------------------

_OPUS = "claude-opus-5-5"


def _cc(tmp_path: Path) -> ClaudeCodeModel:
    return ClaudeCodeModel("cc/opus", {"work_dir": str(tmp_path), "timeout": 30})


def _turn(messages: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    """A multi-request API-billed turn whose ``usage`` sums *messages* (priced value wins)."""
    usage: dict[str, Any] = {}
    for message in messages:
        for key, value in message.items():
            if isinstance(value, int):
                usage[key] = usage.get(key, 0) + value
            elif key == "speed" and (value == "fast" or key not in usage):
                usage[key] = value
            elif key == "inference_geo" and (value == "us" or key not in usage):
                usage[key] = value
    return {
        "api_billed": True,
        "model": _OPUS,
        "requests": len(messages),
        "usage": usage,
        "request_usages": [dict(m) for m in messages],
        **extra,
    }


def test_mixed_speed_turn_prices_each_message_at_its_own_speed(tmp_path: Path) -> None:
    fast = {"input_tokens": 1000, "output_tokens": 100, "speed": "fast"}
    standard = {"input_tokens": 1000, "output_tokens": 100, "speed": "standard"}
    cost = _cc(tmp_path).extract_cost_from_response(_turn([fast, standard]))
    one = calculate_cost(_OPUS, 1000, 100)
    assert cost == pytest.approx(one * 2.0 + one * 1.0)
    # Both fast: the whole turn at 2x, as before.
    assert _cc(tmp_path).extract_cost_from_response(_turn([fast, fast])) == pytest.approx(one * 4.0)
    # US-only inference on one message only.
    us = {"input_tokens": 1000, "output_tokens": 100, "inference_geo": "us"}
    anywhere = {"input_tokens": 1000, "output_tokens": 100, "inference_geo": "global"}
    assert _cc(tmp_path).extract_cost_from_response(_turn([us, anywhere])) == pytest.approx(
        one * 1.1 + one
    )


def test_message_without_a_modifier_field_inherits_the_turns(tmp_path: Path) -> None:
    fast = {"input_tokens": 1000, "output_tokens": 100, "speed": "fast"}
    unknown = {"input_tokens": 1000, "output_tokens": 100}
    one = calculate_cost(_OPUS, 1000, 100)
    # The turn's merged usage says fast (priced value wins): a message that
    # reported no speed is not assumed cheaper.
    assert _cc(tmp_path).extract_cost_from_response(_turn([fast, unknown])) == pytest.approx(
        one * 4.0
    )


def test_the_turns_modifiers_only_fill_in_a_messages_missing_field(tmp_path: Path) -> None:
    # The CLI's terminal result sums the counters but copies the LAST
    # message's ``speed`` / ``inference_geo`` (its usage reducer), so a
    # result saying "standard" after a fast message prices only a message
    # that reported no speed of its own.
    fast = {"input_tokens": 1000, "output_tokens": 100, "speed": "fast"}
    unknown = {"input_tokens": 1000, "output_tokens": 100}
    one = calculate_cost(_OPUS, 1000, 100)
    turn = _turn([fast, unknown])
    turn["usage"]["speed"] = "standard"
    assert _cc(tmp_path).extract_cost_from_response(turn) == pytest.approx(one * 2.0 + one)
    turn = _turn([unknown, fast])
    turn["usage"]["speed"] = "fast"
    assert _cc(tmp_path).extract_cost_from_response(turn) == pytest.approx(one * 4.0)


def test_counters_beyond_the_listed_messages_bill_at_the_turns_modifiers(
    tmp_path: Path,
) -> None:
    fast = {"input_tokens": 1000, "output_tokens": 100, "speed": "fast"}
    standard = {"input_tokens": 1000, "output_tokens": 100, "speed": "standard"}
    turn = _turn([fast, standard])
    turn["usage"]["cache_read_input_tokens"] = 5000  # reported by the result only
    one = calculate_cost(_OPUS, 1000, 100)
    rest = calculate_cost(_OPUS, 0, 0, 5000, single_request=False)
    assert _cc(tmp_path).extract_cost_from_response(turn) == pytest.approx(
        one * 2.0 + one + rest * 2.0
    )


def test_long_context_premium_is_still_per_request_in_a_mixed_turn(tmp_path: Path) -> None:
    long_fast = {"input_tokens": 150_000, "output_tokens": 1000, "speed": "fast"}
    short = {"input_tokens": 1000, "output_tokens": 100, "speed": "standard"}
    turn = {**_turn([long_fast, short]), "model": "claude-haiku-5-5"}
    expected = calculate_cost("claude-haiku-5-5", 150_000, 1000) * 2.0 + calculate_cost(
        "claude-haiku-5-5", 1000, 100
    )
    assert calculate_cost("claude-haiku-5-5", 150_000, 1000) > calculate_cost(
        "claude-haiku-5-5", 150_000, 1000, single_request=False
    )
    assert _cc(tmp_path).extract_cost_from_response(turn) == pytest.approx(expected)


def test_tool_call_cutoff_turn_prices_each_message_at_its_own_speed(tmp_path: Path) -> None:
    """A turn KISS ends at a tool_calls block (no terminal ``result``) bills
    its two messages from the stream's per-message usage, each at the speed
    it was served at."""
    cc = _cc(tmp_path)
    cc.initialize("test")
    cc._api_billed = True

    def start(speed: str) -> dict[str, Any]:
        usage = {"input_tokens": 1000, "output_tokens": 1, "speed": speed}
        return {"type": "message_start", "message": {"model": _OPUS, "usage": usage}}

    delta = {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn"},
        "usage": {"output_tokens": 100},
    }
    block = '<tool_calls>[{"name": "finish", "arguments": {"summary": "x"}}]</tool_calls>'
    events = [
        start("fast"),
        delta,
        {"type": "message_stop"},
        start("standard"),
        delta,
        {"type": "assistant", "message": {"content": [{"type": "text", "text": block}]}},
        {
            "type": "result",
            "result": "SHOULD NOT BE USED",
            "usage": {"input_tokens": 9, "output_tokens": 9},
        },
    ]
    _content, response = cc._parse_stream_events(
        iter(json.dumps(e) for e in events), stop_on_tool_calls=True
    )
    assert response["requests"] == 2
    assert [u.get("speed") for u in response["request_usages"]] == ["fast", "standard"]
    one = calculate_cost(_OPUS, 1000, 100)
    assert cc.extract_cost_from_response(response) == pytest.approx(one * 2.0 + one)


def test_single_request_turn_is_priced_as_before(tmp_path: Path) -> None:
    fast = {"input_tokens": 1000, "output_tokens": 100, "speed": "fast"}
    turn = _turn([fast])
    assert turn["requests"] == 1
    assert _cc(tmp_path).extract_cost_from_response(turn) == pytest.approx(
        calculate_cost(_OPUS, 1000, 100) * 2.0
    )
