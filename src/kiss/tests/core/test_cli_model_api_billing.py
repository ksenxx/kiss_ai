# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""End-to-end tests: what a ``cc/`` or ``codex/`` turn costs.

With the subscription switch on, the CLI bills the user's plan and a turn
costs KISS nothing.  With the switch off the CLI inherits the API key and
every token is API spend: the turn must be priced at the underlying API
model's catalog rate, exactly as a direct ``claude-*`` / ``gpt-*`` call
would be, so the budget guard and the cost in the UI stay honest.

Each test runs the real adapter (and, where stated, a real
:class:`KISSAgent`) against a stand-in ``claude`` / ``codex`` executable
on PATH that prints a scripted ``stream-json`` / JSONL turn.

Run with::

    uv run pytest src/kiss/tests/core/test_cli_model_api_billing.py -v
"""

import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import FrameType
from typing import Any

import pytest

from kiss.core import processes
from kiss.core.config import DEFAULT_CONFIG
from kiss.core.kiss_agent import KISSAgent
from kiss.core.kiss_error import KISSError
from kiss.core.models import codex_model
from kiss.core.models.claude_code_model import ClaudeCodeModel
from kiss.core.models.codex_model import CodexModel
from kiss.core.models.model_info import calculate_cost
from kiss.tests.conftest import install_cli_script, posix_only

pytestmark = posix_only("the fake CLIs are POSIX scripts")

TASK = "Say the word."

# A stand-in CLI: reads the prompt on stdin, prints the events listed in
# its JSON file (one per line) and exits 0, or with the code of an
# ``{"type": "exit", "code": N}`` entry (a CLI that dies mid-turn).
_FAKE_CLI = """#!/usr/bin/env python3
import json, pathlib, sys
sys.stdin.read()
for event in json.loads(pathlib.Path({events_path!r}).read_text()):
    if event.get("type") == "exit":
        sys.exit(event["code"])
    print(json.dumps(event), flush=True)
"""

_CLAUDE_USAGE = {
    "input_tokens": 1000,
    "output_tokens": 500,
    "cache_read_input_tokens": 2000,
    "cache_creation_input_tokens": 400,
    "cache_creation": {"ephemeral_5m_input_tokens": 300, "ephemeral_1h_input_tokens": 100},
}
_CLAUDE_COUNTS = (1000, 500, 2000, 300, 100)

_CODEX_USAGE = {"input_tokens": 1000, "cached_input_tokens": 600, "output_tokens": 200}
_CODEX_COUNTS = (400, 200, 600, 0)


def _install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str) -> Path:
    """Put a fake *name* CLI on PATH; return the file its events are read from."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    events_path = tmp_path / f"{name}-events.json"
    install_cli_script(bin_dir / name, _FAKE_CLI.format(events_path=str(events_path)))
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return events_path


def _claude_turn(
    text: str = "word",
    init_model: str | None = "claude-opus-5-5",
    model_usage: dict[str, Any] | None = None,
    total_cost_usd: float | None = None,
) -> list[dict[str, Any]]:
    """A complete ``stream-json`` turn: optional init event, text, result."""
    events: list[dict[str, Any]] = []
    if init_model:
        events.append({"type": "system", "subtype": "init", "model": init_model})
    events.append({"type": "assistant", "message": {"content": [{"type": "text", "text": text}]}})
    result: dict[str, Any] = {"type": "result", "result": text, "usage": _CLAUDE_USAGE}
    if model_usage is not None:
        result["modelUsage"] = model_usage
    if total_cost_usd is not None:
        result["total_cost_usd"] = total_cost_usd
    events.append(result)
    return events


def _codex_turn(text: str = "word") -> list[dict[str, Any]]:
    """A complete ``codex exec --json`` turn."""
    return [
        {"type": "thread.started", "thread_id": "t-1"},
        {"type": "item.completed", "item": {"type": "agent_message", "text": text}},
        {"type": "turn.completed", "usage": _CODEX_USAGE},
    ]


def _haiku_request(input_tokens: int, with_start: bool) -> list[dict[str, Any]]:
    """One Claude Haiku 5.5 API request of *input_tokens* prompt and 200
    output tokens, optionally without its ``message_start`` event."""
    start = {
        "type": "message_start",
        "message": {"model": "claude-haiku-5-5", "usage": {"input_tokens": input_tokens}},
    }
    delta = {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn"},
        "usage": {"input_tokens": input_tokens, "output_tokens": 200},
    }
    return [start, delta] if with_start else [delta]


@pytest.fixture
def fake_claude(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fake ``claude`` on PATH with an API key in the environment."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api-test")
    monkeypatch.setattr(DEFAULT_CONFIG, "CLAUDE_CODE_OAUTH_TOKEN", "")
    return _install(tmp_path, monkeypatch, "claude")


@pytest.fixture
def fake_codex(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fake ``codex`` on PATH with an API key in the environment."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))
    monkeypatch.setattr(DEFAULT_CONFIG, "CODEX_ACCESS_TOKEN", "")
    return _install(tmp_path, monkeypatch, "codex")


def _run_agent(model_name: str, tmp_path: Path) -> KISSAgent:
    """Run one non-agentic turn of a real agent on *model_name*."""
    agent = KISSAgent("cli billing e2e")
    agent.run(
        model_name=model_name,
        prompt_template=TASK,
        is_agentic=False,
        verbose=False,
        model_config={"work_dir": str(tmp_path), "timeout": 30},
    )
    return agent


# --- Claude Code --------------------------------------------------------


def test_cc_subscription_turn_is_free(
    fake_claude: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Switch on: the plan pays, the agent's budget sees $0 (tokens counted)."""
    monkeypatch.setattr(DEFAULT_CONFIG, "claude_code_use_subscription", True)
    fake_claude.write_text(json.dumps(_claude_turn()))
    agent = _run_agent("cc/opus", tmp_path)
    assert agent.budget_used == 0.0
    assert agent.total_tokens_used == sum(_CLAUDE_COUNTS)
    assert agent.last_call_usage is not None
    assert agent.last_call_usage["cost"] == 0.0


def test_cc_api_key_turn_is_billed_at_the_resolved_model(
    fake_claude: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Switch off: ``cc/opus`` is priced as the ``claude-opus-5-5`` the CLI
    resolved it to, cache tiers (5m and 1h writes, reads) included."""
    monkeypatch.setattr(DEFAULT_CONFIG, "claude_code_use_subscription", False)
    fake_claude.write_text(json.dumps(_claude_turn()))
    agent = _run_agent("cc/opus", tmp_path)
    expected = calculate_cost("claude-opus-5-5", *_CLAUDE_COUNTS)
    assert expected > 0
    assert agent.budget_used == pytest.approx(expected)
    assert agent.last_call_usage is not None
    assert agent.last_call_usage["cost"] == pytest.approx(expected)
    assert agent.last_call_usage["cache_write"] == 400


def test_cc_api_key_turn_uses_the_name_after_the_prefix_when_the_stream_is_silent(
    fake_claude: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No event names the model: ``cc/claude-sonnet-5-5`` bills as ``claude-sonnet-5-5``."""
    monkeypatch.setattr(DEFAULT_CONFIG, "claude_code_use_subscription", False)
    fake_claude.write_text(json.dumps(_claude_turn(init_model=None)))
    agent = _run_agent("cc/claude-sonnet-5-5", tmp_path)
    assert agent.budget_used == pytest.approx(calculate_cost("claude-sonnet-5-5", *_CLAUDE_COUNTS))


def test_cc_api_key_turn_prices_a_dated_snapshot_from_model_usage(
    fake_claude: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The result's ``modelUsage`` id outranks the init event, and a dated
    snapshot id bills as its undated catalog entry."""
    monkeypatch.setattr(DEFAULT_CONFIG, "claude_code_use_subscription", False)
    fake_claude.write_text(
        json.dumps(
            _claude_turn(
                init_model="claude-opus-5-5",
                model_usage={"claude-sonnet-4-5-20250929": {"inputTokens": 1000}},
            )
        )
    )
    agent = _run_agent("cc/sonnet", tmp_path)
    assert agent.budget_used == pytest.approx(calculate_cost("claude-sonnet-4-5", *_CLAUDE_COUNTS))


def test_cc_api_key_turn_on_an_uncatalogued_model_uses_the_cli_estimate(
    fake_claude: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A model id the catalog does not know bills at the CLI's ``total_cost_usd``."""
    monkeypatch.setattr(DEFAULT_CONFIG, "claude_code_use_subscription", False)
    turn = _claude_turn(init_model="claude-zeta-9", total_cost_usd=0.42)
    fake_claude.write_text(json.dumps(turn))
    agent = _run_agent("cc/opus", tmp_path)
    assert agent.budget_used == pytest.approx(0.42)


def test_cc_api_key_turn_on_an_uncatalogued_model_without_an_estimate_raises(
    fake_claude: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No catalog price and no CLI estimate: the budget cannot be kept, so the run stops."""
    monkeypatch.setattr(DEFAULT_CONFIG, "claude_code_use_subscription", False)
    fake_claude.write_text(json.dumps(_claude_turn(init_model="claude-zeta-9")))
    with pytest.raises(KISSError, match="unknown model 'claude-zeta-9'"):
        _run_agent("cc/opus", tmp_path)


def test_cc_api_key_turn_cut_at_tool_calls_is_billed_from_the_partial_usage(
    fake_claude: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tool-bearing turn stops at the ``tool_calls`` block, before the
    ``result`` event (the CLI is killed there, so the later
    ``message_delta`` is never read); the usage and the model id seen in
    ``message_start`` still price the turn."""
    monkeypatch.setattr(DEFAULT_CONFIG, "claude_code_use_subscription", False)
    tool_calls = json.dumps({"tool_calls": [{"name": "echo", "arguments": {"x": "hi"}}]})
    fake_claude.write_text(
        json.dumps(
            [
                {
                    "type": "message_start",
                    "message": {
                        "model": "claude-opus-5-5",
                        "usage": {"input_tokens": 1000, "output_tokens": 1},
                    },
                },
                {
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": tool_calls + "\nand then more text"},
                },
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {"input_tokens": 1000, "output_tokens": 500},
                },
            ]
        )
    )
    model = ClaudeCodeModel("cc/opus", {"work_dir": str(tmp_path), "timeout": 30})
    model.conversation = [{"role": "user", "content": TASK}]
    calls, _content, response = model.generate_and_process_with_tools(
        {"echo": lambda x: x}  # noqa: E731 - the tool is never run here
    )
    assert [c["name"] for c in calls] == ["echo"]
    assert response["model"] == "claude-opus-5-5"
    assert response["usage"]["input_tokens"] == 1000
    assert model.extract_cost_from_response(response) == pytest.approx(
        calculate_cost("claude-opus-5-5", 1000, response["usage"]["output_tokens"])
    )


def test_cc_usage_recovered_from_a_failed_turn_is_priced_like_a_whole_turn(
    fake_claude: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI dies after one complete message: the generation raises, the
    usage it had reported is recovered once, and it is API spend under the
    model the stream named."""
    monkeypatch.setattr(DEFAULT_CONFIG, "claude_code_use_subscription", False)
    fake_claude.write_text(
        json.dumps(
            [
                {
                    "type": "message_start",
                    "message": {"model": "claude-opus-5-5", "usage": {"input_tokens": 10}},
                },
                {"type": "message_delta", "usage": {"input_tokens": 10, "output_tokens": 20}},
                {"type": "exit", "code": 1},
            ]
        )
    )
    model = ClaudeCodeModel("cc/opus", {"work_dir": str(tmp_path), "timeout": 30})
    model.conversation = [{"role": "user", "content": TASK}]
    with pytest.raises(KISSError):
        model.generate()
    partial = model.take_partial_usage_response()
    assert partial == {
        "usage": {"input_tokens": 10, "output_tokens": 20},
        "model": "claude-opus-5-5",
        "api_billed": True,
        "requests": 1,
        "request_usages": [{"input_tokens": 10, "output_tokens": 20}],
    }
    assert model.extract_cost_from_response(partial) == pytest.approx(
        calculate_cost("claude-opus-5-5", 10, 20)
    )
    assert model.take_partial_usage_response() is None, "recovered once"
    assert model.extract_cost_from_response("not a dict") is None


def _modifier_turn(
    first: dict[str, str], second: dict[str, str], result_usage: dict[str, Any]
) -> list[dict[str, Any]]:
    """Two API messages whose ``message_start`` usage carries *first* / *second*
    modifiers, then the CLI's terminal ``result`` summing only the counters."""
    return [
        {"type": "system", "subtype": "init", "model": "claude-opus-5-5"},
        {
            "type": "message_start",
            "message": {
                "model": "claude-opus-5-5",
                "usage": {"input_tokens": 600, "output_tokens": 1, **first},
            },
        },
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 300},
        },
        {"type": "message_stop"},
        {
            "type": "message_start",
            "message": {
                "model": "claude-opus-5-5",
                "usage": {"input_tokens": 400, "output_tokens": 1, **second},
            },
        },
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 200},
        },
        {"type": "message_stop"},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "word"}]}},
        {"type": "result", "result": "word", "usage": result_usage},
    ]


@pytest.mark.parametrize(
    ("first", "second", "factors"),
    [
        ({"inference_geo": "us"}, {"inference_geo": "us"}, (1.1, 1.1)),
        ({"speed": "fast"}, {"speed": "fast"}, (2.0, 2.0)),
        (
            {"speed": "fast", "inference_geo": "us"},
            {"speed": "fast", "inference_geo": "us"},
            (2.2, 2.2),
        ),
        ({"speed": "standard", "inference_geo": "global"}, {}, (1.0, 1.0)),
        # Messages disagree (fast mode fell back to standard speed AND
        # standard pricing at its rate limit): each message is priced at
        # the speed it was served at.
        ({"speed": "standard"}, {"speed": "fast"}, (1.0, 2.0)),
        ({"inference_geo": "us"}, {"inference_geo": "global"}, (1.1, 1.0)),
    ],
)
def test_cc_api_key_turn_is_scaled_by_the_speed_and_geo_the_messages_report(
    fake_claude: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    first: dict[str, str],
    second: dict[str, str],
    factors: tuple[float, float],
) -> None:
    """Fast mode (2x) and US-only inference (1.1x) are reported per API
    message in ``usage``; the CLI's ``result`` sums only the counters, so
    the modifiers seen in the stream must still price the turn, each
    message at the modifiers its own usage reports
    (https://platform.claude.com/docs/en/build-with-claude/fast-mode,
    https://code.claude.com/docs/en/fast-mode#handle-rate-limits,
    https://platform.claude.com/docs/en/manage-claude/data-residency).
    The counters the ``result`` holds beyond the messages (the cache
    tokens of this fixture) bill at the turn's merged modifiers, where
    the priced value wins."""
    monkeypatch.setattr(DEFAULT_CONFIG, "claude_code_use_subscription", False)
    fake_claude.write_text(json.dumps(_modifier_turn(first, second, _CLAUDE_USAGE)))
    agent = _run_agent("cc/opus", tmp_path)
    cost = calculate_cost
    expected = (
        cost("claude-opus-5-5", 600, 300) * factors[0]
        + cost("claude-opus-5-5", 400, 200) * factors[1]
        + cost("claude-opus-5-5", 0, 0, 2000, 300, 100, single_request=False) * max(factors)
    )
    assert agent.budget_used == pytest.approx(expected)


def test_cc_result_usage_modifiers_fill_in_what_the_stream_did_not_report(
    fake_claude: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``result`` whose usage names the modifiers names the LAST message's
    (the CLI's usage reducer sums the counters and copies ``speed`` /
    ``inference_geo`` from the newest message): they price a message that
    reported none of its own, never outrank one that did, and price the
    counters the result holds beyond the messages."""
    monkeypatch.setattr(DEFAULT_CONFIG, "claude_code_use_subscription", False)
    usage = {**_CLAUDE_USAGE, "speed": "standard", "inference_geo": "global"}
    fake_claude.write_text(json.dumps(_modifier_turn({"speed": "fast"}, {}, usage)))
    agent = _run_agent("cc/opus", tmp_path)
    cost = calculate_cost
    expected = (
        cost("claude-opus-5-5", 600, 300) * 2.0
        + cost("claude-opus-5-5", 400, 200)
        + cost("claude-opus-5-5", 0, 0, 2000, 300, 100, single_request=False)
    )
    assert agent.budget_used == pytest.approx(expected)


def test_cc_usage_recovered_from_a_failed_turn_keeps_its_modifiers(
    fake_claude: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The modifiers of a message cut short survive into the recovered usage."""
    monkeypatch.setattr(DEFAULT_CONFIG, "claude_code_use_subscription", False)
    fake_claude.write_text(
        json.dumps(
            [
                {
                    "type": "message_start",
                    "message": {
                        "model": "claude-opus-5-5",
                        "usage": {"input_tokens": 10, "speed": "fast", "inference_geo": "us"},
                    },
                },
                {"type": "message_delta", "usage": {"input_tokens": 10, "output_tokens": 20}},
                {"type": "exit", "code": 1},
            ]
        )
    )
    model = ClaudeCodeModel("cc/opus", {"work_dir": str(tmp_path), "timeout": 30})
    model.conversation = [{"role": "user", "content": TASK}]
    with pytest.raises(KISSError):
        model.generate()
    partial = model.take_partial_usage_response()
    assert partial["usage"] == {
        "input_tokens": 10,
        "output_tokens": 20,
        "speed": "fast",
        "inference_geo": "us",
    }
    assert model.extract_cost_from_response(partial) == pytest.approx(
        calculate_cost("claude-opus-5-5", 10, 20) * 2.2
    )


def test_cc_us_geo_on_a_pre_4_6_model_bills_standard(
    fake_claude: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Claude Haiku 4.5 predates ``inference_geo``; the field cannot raise its price."""
    monkeypatch.setattr(DEFAULT_CONFIG, "claude_code_use_subscription", False)
    turn = _modifier_turn({"inference_geo": "us"}, {"inference_geo": "us"}, _CLAUDE_USAGE)
    for event in turn:
        if event["type"] == "system":
            event["model"] = "claude-haiku-4-5"
        if event["type"] == "message_start":
            event["message"]["model"] = "claude-haiku-4-5"
    fake_claude.write_text(json.dumps(turn))
    agent = _run_agent("cc/haiku", tmp_path)
    assert agent.budget_used == pytest.approx(
        calculate_cost("claude-haiku-4-5", *_CLAUDE_COUNTS, single_request=False)
    )


def test_cc_turn_billing_is_fixed_when_the_cli_is_launched(
    fake_claude: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flipping the switch from the settings panel while a turn runs does
    not reprice that turn: it was launched with the API key, so it is API
    spend; the next turn, launched under the subscription, is free."""
    monkeypatch.setattr(DEFAULT_CONFIG, "claude_code_use_subscription", False)
    fake_claude.write_text(json.dumps(_claude_turn()))
    model = ClaudeCodeModel("cc/opus", {"work_dir": str(tmp_path), "timeout": 30})
    model.conversation = [{"role": "user", "content": TASK}]
    model.token_callback = lambda _token: setattr(
        DEFAULT_CONFIG, "claude_code_use_subscription", True
    )
    _text, response = model.generate()
    assert DEFAULT_CONFIG.claude_code_use_subscription is True
    assert model.extract_cost_from_response(response) == pytest.approx(
        calculate_cost("claude-opus-5-5", *_CLAUDE_COUNTS)
    )
    model.token_callback = None
    _text, response = model.generate()
    assert model.extract_cost_from_response(response) is None


def test_cc_switch_off_without_an_api_key_still_bills_the_subscription(
    fake_claude: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No API key variable in the inherited environment: the CLI falls back
    to its login, so nothing is API spend."""
    monkeypatch.setattr(DEFAULT_CONFIG, "claude_code_use_subscription", False)
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    fake_claude.write_text(json.dumps(_claude_turn()))
    assert _run_agent("cc/opus", tmp_path).budget_used == 0.0


def test_cc_multi_request_turn_is_not_tiered_as_one_long_prompt(
    fake_claude: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Haiku 5.5 bills 5x above 100K prompt tokens per request.  A turn of
    two 60K-token requests sums to 120K but neither crossed the line, so
    the turn bills at the short-context rate; one 120K request does not."""
    monkeypatch.setattr(DEFAULT_CONFIG, "claude_code_use_subscription", False)
    usage = {"input_tokens": 120_000, "output_tokens": 400}
    message = {"model": "claude-haiku-5-5", "usage": {"input_tokens": 60_000}}
    result = {"type": "result", "result": "word", "usage": usage}
    text = {"type": "assistant", "message": {"content": [{"type": "text", "text": "word"}]}}
    two_requests = [
        {"type": "message_start", "message": message},
        {"type": "message_delta", "usage": {"input_tokens": 60_000, "output_tokens": 200}},
        {"type": "message_start", "message": message},
        {"type": "message_delta", "usage": {"input_tokens": 60_000, "output_tokens": 200}},
        text,
        result,
    ]
    fake_claude.write_text(json.dumps(two_requests))
    short_rate = calculate_cost("claude-haiku-5-5", 120_000, 400, single_request=False)
    long_rate = calculate_cost("claude-haiku-5-5", 120_000, 400)
    assert long_rate == pytest.approx(5 * short_rate)
    assert _run_agent("cc/haiku", tmp_path).budget_used == pytest.approx(short_rate)

    fake_claude.write_text(json.dumps(two_requests[:2] + [text, result]))
    assert _run_agent("cc/haiku", tmp_path).budget_used == pytest.approx(long_rate)


def test_cc_multi_request_turn_bills_the_long_context_tier_of_each_long_request(
    fake_claude: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each request crossing Haiku 5.5's 100K line bills 5x on its own; the
    others stay at the short rate, whatever the turn's summed counters say."""
    monkeypatch.setattr(DEFAULT_CONFIG, "claude_code_use_subscription", False)
    text = {"type": "assistant", "message": {"content": [{"type": "text", "text": "word"}]}}
    result = {
        "type": "result",
        "result": "word",
        "usage": {"input_tokens": 180_000, "output_tokens": 400},
        "total_cost_usd": 0.0,
    }
    events = _haiku_request(120_000, True) + _haiku_request(60_000, True) + [text, result]
    fake_claude.write_text(json.dumps(events))
    expected = calculate_cost("claude-haiku-5-5", 120_000, 200) + calculate_cost(
        "claude-haiku-5-5", 60_000, 200
    )
    assert expected > calculate_cost("claude-haiku-5-5", 180_000, 400, single_request=False)
    agent = _run_agent("cc/haiku", tmp_path)
    assert agent.budget_used == pytest.approx(expected)
    assert agent.total_tokens_used == 180_400


def test_cc_finished_messages_without_message_start_count_as_requests(
    fake_claude: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two short requests whose stream lacks ``message_start`` are two
    requests, not one 120K prompt billed at Haiku 5.5's long-context rate."""
    monkeypatch.setattr(DEFAULT_CONFIG, "claude_code_use_subscription", False)
    text = {"type": "assistant", "message": {"content": [{"type": "text", "text": "word"}]}}
    result = {
        "type": "result",
        "result": "word",
        "usage": {"input_tokens": 120_000, "output_tokens": 400},
        "total_cost_usd": 0.0,
    }
    events = _haiku_request(60_000, False) + _haiku_request(60_000, False) + [text, result]
    fake_claude.write_text(json.dumps(events))
    agent = _run_agent("cc/claude-haiku-5-5", tmp_path)
    assert agent.budget_used == pytest.approx(2 * calculate_cost("claude-haiku-5-5", 60_000, 200))


# --- Codex -------------------------------------------------------------


def test_codex_subscription_turn_is_free(
    fake_codex: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Switch on: the ChatGPT plan pays, the agent's budget sees $0."""
    monkeypatch.setattr(DEFAULT_CONFIG, "codex_use_subscription", True)
    fake_codex.write_text(json.dumps(_codex_turn()))
    agent = _run_agent("codex/gpt-5.5", tmp_path)
    assert agent.budget_used == 0.0
    assert agent.total_tokens_used == sum(_CODEX_COUNTS)


def test_codex_api_key_turn_is_billed_at_the_named_model(
    fake_codex: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Switch off: ``codex/gpt-5.5`` costs what a direct ``gpt-5.5`` call costs."""
    monkeypatch.setattr(DEFAULT_CONFIG, "codex_use_subscription", False)
    fake_codex.write_text(json.dumps(_codex_turn()))
    agent = _run_agent("codex/gpt-5.5", tmp_path)
    expected = calculate_cost("gpt-5.5", *_CODEX_COUNTS)
    assert expected > 0
    assert agent.budget_used == pytest.approx(expected)
    assert agent.last_call_usage is not None
    assert agent.last_call_usage["cache_read"] == 600


def test_codex_turn_billing_is_fixed_when_the_cli_is_launched(
    fake_codex: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flipping the switch while a turn runs does not reprice that turn."""
    monkeypatch.setattr(DEFAULT_CONFIG, "codex_use_subscription", False)
    fake_codex.write_text(json.dumps(_codex_turn()))
    model = CodexModel("codex/gpt-5.5", {"work_dir": str(tmp_path), "timeout": 30})
    model.conversation = [{"role": "user", "content": TASK}]
    model.token_callback = lambda _token: setattr(DEFAULT_CONFIG, "codex_use_subscription", True)
    _text, response = model.generate()
    assert DEFAULT_CONFIG.codex_use_subscription is True
    assert model.extract_cost_from_response(response) == pytest.approx(
        calculate_cost("gpt-5.5", *_CODEX_COUNTS)
    )
    model.token_callback = None
    _text, response = model.generate()
    assert model.extract_cost_from_response(response) is None


def test_codex_switch_off_without_an_api_key_still_bills_the_subscription(
    fake_codex: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No key anywhere (environment or settings): the CLI uses its login."""
    monkeypatch.setattr(DEFAULT_CONFIG, "codex_use_subscription", False)
    monkeypatch.delenv("OPENAI_API_KEY")
    monkeypatch.delenv("CODEX_API_KEY", raising=False)
    monkeypatch.setattr(DEFAULT_CONFIG, "OPENAI_API_KEY", "")
    fake_codex.write_text(json.dumps(_codex_turn()))
    assert _run_agent("codex/gpt-5.5", tmp_path).budget_used == 0.0


def test_codex_multi_request_turn_is_not_tiered_as_one_long_prompt(
    fake_codex: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``turn.completed`` sums the usage of every request in the turn.  Two
    200K-token requests (one per tool call) sum past gpt-5.5's 272K
    long-context line without any request crossing it: short-context
    rates.  The same sum from a turn without tool items is one request
    and bills the long-context tier."""
    monkeypatch.setattr(DEFAULT_CONFIG, "codex_use_subscription", False)
    usage = {"input_tokens": 400_000, "cached_input_tokens": 0, "output_tokens": 2000}
    tool_item = {
        "type": "item.completed",
        "item": {"type": "mcp_tool_call", "server": "kiss", "tool": "echo", "status": "completed"},
    }
    text_item = {"type": "item.completed", "item": {"type": "agent_message", "text": "word"}}
    done = {"type": "turn.completed", "usage": usage}
    fake_codex.write_text(json.dumps([tool_item, text_item, done]))
    short_rate = calculate_cost("gpt-5.5", 400_000, 2000, single_request=False)
    long_rate = calculate_cost("gpt-5.5", 400_000, 2000)
    assert short_rate == pytest.approx(2 * calculate_cost("gpt-5.5", 200_000, 1000))
    assert long_rate > short_rate
    assert _run_agent("codex/gpt-5.5", tmp_path).budget_used == pytest.approx(short_rate)

    reasoning_item = {"type": "item.completed", "item": {"type": "reasoning", "text": "hmm"}}
    fake_codex.write_text(json.dumps([reasoning_item, text_item, done]))
    assert _run_agent("codex/gpt-5.5", tmp_path).budget_used == pytest.approx(long_rate)


def test_codex_usage_of_a_turn_that_fails_after_completing_is_billed_once(
    fake_codex: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI reports ``turn.completed`` and then exits non-zero: the
    generation raises, but the usage it reported is real spend, recovered
    exactly once.  A turn that succeeds leaves nothing to recover."""
    monkeypatch.setattr(DEFAULT_CONFIG, "codex_use_subscription", False)
    fake_codex.write_text(json.dumps([*_codex_turn(), {"type": "exit", "code": 1}]))
    model = CodexModel("codex/gpt-5.5", {"work_dir": str(tmp_path), "timeout": 30})
    model.conversation = [{"role": "user", "content": TASK}]
    with pytest.raises(KISSError):
        model.generate()
    partial = model.take_partial_usage_response()
    assert model.extract_input_output_token_counts_from_response(partial) == _CODEX_COUNTS
    assert model.extract_cost_from_response(partial) == pytest.approx(
        calculate_cost("gpt-5.5", *_CODEX_COUNTS)
    )
    assert model.take_partial_usage_response() is None, "recovered once"

    fake_codex.write_text(json.dumps(_codex_turn()))
    model.generate()
    assert model.take_partial_usage_response() is None

    fake_codex.write_text(json.dumps([*_codex_turn(), {"type": "exit", "code": 1}]))
    agent = KISSAgent("codex failed turn")
    with pytest.raises(KISSError):
        agent.run(
            model_name="codex/gpt-5.5",
            prompt_template=TASK,
            is_agentic=False,
            verbose=False,
            model_config={"work_dir": str(tmp_path), "timeout": 30},
        )
    assert agent.total_tokens_used == sum(_CODEX_COUNTS)
    assert agent.budget_used == pytest.approx(calculate_cost("gpt-5.5", *_CODEX_COUNTS))


def test_codex_api_key_turn_on_an_unresolved_cli_default_model_warns_once_and_stays_free(
    fake_codex: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``codex/default`` whose default model the CLI did not report cannot
    be priced: it stays at the catalog's $0 with one warning per model
    instance."""
    monkeypatch.setattr(DEFAULT_CONFIG, "codex_use_subscription", False)
    monkeypatch.setattr(codex_model, "codex_default_model", lambda env: None)
    fake_codex.write_text(json.dumps(_codex_turn()))
    model = CodexModel("codex/default", {"work_dir": str(tmp_path), "timeout": 30})
    model.conversation = [{"role": "user", "content": TASK}]
    with caplog.at_level(logging.WARNING, logger="kiss.core.models.codex_model"):
        for _ in range(2):
            _text, response = model.generate()
            assert "priced_model" not in response
            assert model.extract_cost_from_response(response) is None
    warnings = [r for r in caplog.records if "no price for 'default'" in r.getMessage()]
    assert len(warnings) == 1
    assert model.extract_cost_from_response("not a dict") is None


def _no_resolver(env: dict[str, str]) -> str | None:
    """A ``codex_default_model`` that must not run."""
    raise AssertionError("a subscription turn resolved the default model")


def test_codex_api_key_turn_on_the_cli_default_model_is_billed_at_that_model(
    fake_codex: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``codex/default`` runs the CLI's default model, which the app server
    reports: an API-billed turn costs what that model costs, a turn cut
    after ``turn.completed`` too, and a subscription turn never asks."""
    monkeypatch.setattr(DEFAULT_CONFIG, "codex_use_subscription", False)
    asked: list[dict[str, str]] = []

    def resolve_default(env: dict[str, str], timeout: float = 15.0) -> str:
        asked.append(env)
        return "gpt-5.5"

    monkeypatch.setattr(codex_model, "codex_default_model", resolve_default)
    fake_codex.write_text(json.dumps(_codex_turn()))
    expected = calculate_cost("gpt-5.5", *_CODEX_COUNTS)
    assert expected > 0
    assert _run_agent("codex/default", tmp_path).budget_used == pytest.approx(expected)
    assert asked and asked[0]["CODEX_API_KEY"] == "sk-openai-test"

    fake_codex.write_text(json.dumps([*_codex_turn(), {"type": "exit", "code": 1}]))
    model = CodexModel("codex/default", {"work_dir": str(tmp_path), "timeout": 30})
    model.conversation = [{"role": "user", "content": TASK}]
    with pytest.raises(KISSError):
        model.generate()
    partial = model.take_partial_usage_response()
    assert partial["priced_model"] == "gpt-5.5"
    assert model.extract_cost_from_response(partial) == pytest.approx(expected)

    monkeypatch.setattr(DEFAULT_CONFIG, "codex_use_subscription", True)
    monkeypatch.setattr(codex_model, "codex_default_model", _no_resolver)
    fake_codex.write_text(json.dumps(_codex_turn()))
    assert _run_agent("codex/default", tmp_path).budget_used == 0.0


# A stand-in ``codex app-server``: records its pid, then answers the
# JSON-RPC lines on stdin as *mode* says until stdin closes.
_FAKE_APP_SERVER = """#!/usr/bin/env python3
import json, os, pathlib, sys, time
pathlib.Path({pid_path!r}).write_text(str(os.getpid()))
mode = {mode!r}
if mode == "exit":
    sys.exit(1)
if mode == "close-stdin":
    sys.stdin.readline()
    os.close(0)  # before answering: the client's next write breaks the pipe
    print(json.dumps({{"id": 1, "result": {{}}}}), flush=True)
    time.sleep(60)
for line in sys.stdin:
    rpc_id = json.loads(line).get("id")
    if mode == "hang" or rpc_id is None:
        continue
    if mode == "garbage":
        print("not json", flush=True)
        print(json.dumps({{"id": rpc_id, "result": "nope"}}), flush=True)
    elif rpc_id == 1:
        print("WARN noise on stdout", flush=True)
        print(json.dumps({{"method": "configWarning", "params": {{}}}}), flush=True)
        print(json.dumps({{"id": 1, "result": {{"userAgent": "fake"}}}}), flush=True)
    else:
        data = [{{"id": "gpt-6-astra", "isDefault": False}}]
        if mode == "ok":
            data.append({{"id": "gpt-6.1-sol", "isDefault": True}})
        print(json.dumps({{"id": rpc_id, "result": {{"data": data}}}}), flush=True)
"""


def _install_app_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str) -> Path:
    """Put a fake ``codex app-server`` in *mode* first on PATH; return its pid file.

    Every mode is written to the same path, so the resolver's cache
    (keyed by the CLI path) sees one CLI whose behaviour changed."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    pid_path = tmp_path / f"app-server-{mode}.pid"
    install_cli_script(
        bin_dir / "codex", _FAKE_APP_SERVER.format(pid_path=str(pid_path), mode=mode)
    )
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return pid_path


def _assert_reaped(pid_path: Path) -> None:
    """The fake server whose pid *pid_path* holds is gone (killed and reaped)."""
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_path.read_text()), 0)


def test_codex_default_model_asks_the_app_server_and_caches_the_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The resolver skips notifications and noise, returns the ``isDefault``
    model, kills the server, and reuses the answer per ``CODEX_HOME``."""
    monkeypatch.setattr(codex_model, "_default_model_cache", {})
    pid_path = _install_app_server(tmp_path, monkeypatch, "ok")
    env = {**os.environ, "CODEX_HOME": str(tmp_path / "home-a")}
    assert codex_model.codex_default_model(env) == "gpt-6.1-sol"
    _assert_reaped(pid_path)

    hang_pid_path = _install_app_server(tmp_path, monkeypatch, "hang")
    assert codex_model.codex_default_model(env, timeout=0.5) == "gpt-6.1-sol", "cached"
    assert not hang_pid_path.exists()
    # Another CODEX_HOME is another cache entry: the hanging server is
    # asked, and the resolver gives up at the deadline.
    started = time.monotonic()
    assert codex_model.codex_default_model({**env, "CODEX_HOME": "b"}, timeout=0.5) is None
    assert time.monotonic() - started < 10
    _assert_reaped(hang_pid_path)


@pytest.mark.parametrize("mode", ["hang", "garbage", "no-default", "exit", "close-stdin"])
def test_codex_default_model_is_none_when_the_app_server_does_not_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """A server that hangs, answers garbage, flags no default, dies, or
    stops reading before the second request (a broken pipe) yields
    ``None`` within the timeout, and the server is killed.  The failure
    is cached (briefly): a fixed server is not asked again yet."""
    monkeypatch.setattr(codex_model, "_default_model_cache", {})
    pid_path = _install_app_server(tmp_path, monkeypatch, mode)
    env = {**os.environ}
    started = time.monotonic()
    assert codex_model.codex_default_model(env, timeout=1.0) is None
    assert time.monotonic() - started < 10
    _assert_reaped(pid_path)
    ok_pid_path = _install_app_server(tmp_path, monkeypatch, "ok")
    assert codex_model.codex_default_model(env) is None, "failure cached"
    assert not ok_pid_path.exists()


def test_codex_default_model_deadline_holds_against_a_notification_backlog() -> None:
    """Queued notifications are not read past the deadline, and a reply
    that is only reached after it is not accepted."""
    import queue

    lines: queue.Queue[str | None] = queue.Queue()
    for _ in range(100_000):
        lines.put('{"method": "notification"}')
    lines.put('{"id": 2, "result": {"data": []}}')
    started = time.monotonic()
    assert codex_model._await_rpc_result(lines, 2, started + 0.001) is None
    assert lines.qsize() > 90_000, "stopped reading at the deadline"
    assert codex_model._await_rpc_result(lines, 2, started - 1) is None


def test_codex_default_model_late_failure_keeps_a_concurrent_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two turns miss the cache together; the one that resolves the
    default caches it, and the other's later failure does not replace it."""
    monkeypatch.setattr(codex_model, "_default_model_cache", {})
    monkeypatch.setattr(codex_model, "find_codex_executable", lambda: "/fake/codex")
    release = threading.Event()
    calls: list[str] = []

    def query(path: str, env: dict[str, str], timeout: float) -> str | None:
        if not calls:
            calls.append("slow")
            release.wait(10)
            return None
        calls.append("fast")
        return "gpt-6.1-sol"

    monkeypatch.setattr(codex_model, "_query_default_model", query)
    results: list[str | None] = []
    slow = threading.Thread(target=lambda: results.append(codex_model.codex_default_model({})))
    slow.start()
    while not calls:
        time.sleep(0.01)
    assert codex_model.codex_default_model({}) == "gpt-6.1-sol"
    release.set()
    slow.join(10)
    assert results == ["gpt-6.1-sol"], "the late failure returns the cached answer"
    assert codex_model.codex_default_model({}) == "gpt-6.1-sol"
    assert calls == ["slow", "fast"], "served from the cache"


@pytest.mark.parametrize(
    ("callee", "error"),
    [
        (threading.Thread.start, RuntimeError("can't start new thread")),
        (queue.Queue.__init__, KeyboardInterrupt("stop")),
    ],
    ids=["reader-cannot-start", "stop-at-the-line-queue"],
)
def test_codex_default_model_ends_the_app_server_when_its_reader_cannot_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, callee: Any, error: BaseException
) -> None:
    """A ``Thread.start`` that fails once the app-server is spawned (no
    thread left) still kills and reaps the server: the spawn is owned by
    the resolver's cleanup before any fallible step runs.  A Stop raised
    where the resolver makes its line queue (it was made between the
    spawn and that cleanup) leaves no server running either."""
    monkeypatch.setattr(codex_model, "_default_model_cache", {})
    _install_app_server(tmp_path, monkeypatch, "hang")
    spawned: list[subprocess.Popen[str]] = []

    def fail_the_reader_start(frame: FrameType, event: str, arg: Any) -> None:
        if event == "return" and frame.f_code is processes.popen_process_group.__code__:
            spawned.append(arg)
        elif (
            event == "call"
            and frame.f_code is callee.__code__
            and frame.f_back is not None
            and frame.f_back.f_code is codex_model._query_default_model.__code__
        ):
            sys.setprofile(None)
            raise error

    sys.setprofile(fail_the_reader_start)
    try:
        with pytest.raises(type(error), match=str(error)):
            codex_model.codex_default_model(dict(os.environ))
    finally:
        sys.setprofile(None)
    for server in spawned:  # none when the Stop was raised before the spawn
        try:
            assert server.poll() is not None, "the app-server nobody reads must be ended"
        finally:
            if server.poll() is None:
                processes.kill_process_tree(server)
                server.wait()


def test_codex_default_model_is_none_without_a_runnable_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No ``codex`` anywhere, or one that cannot be executed: ``None``."""
    monkeypatch.setattr(codex_model, "_default_model_cache", {})
    monkeypatch.setattr(codex_model, "_UI_CANDIDATE_PATHS", ())
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    assert codex_model.codex_default_model(dict(os.environ)) is None
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    install_cli_script(bin_dir / "codex", "#!/nonexistent/interpreter\n")
    monkeypatch.setenv("PATH", str(bin_dir))
    assert codex_model.codex_default_model(dict(os.environ)) is None


# --- Cost audit 2026-10-20 ----------------------------------------------


def test_cc_api_key_turn_with_a_helper_model_bills_the_cli_estimate(
    fake_claude: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``modelUsage`` naming two models (Haiku helper first) bills at the
    CLI's per-model ``total_cost_usd``, not the whole turn at Haiku's rate."""
    monkeypatch.setattr(DEFAULT_CONFIG, "claude_code_use_subscription", False)
    model_usage = {
        "claude-haiku-4-5": {"inputTokens": 50, "costUSD": 0.0001},
        "claude-opus-4-5": {"inputTokens": 1000, "costUSD": 0.0899},
    }
    turn = _claude_turn(init_model="claude-opus-4-5", model_usage=model_usage, total_cost_usd=0.09)
    fake_claude.write_text(json.dumps(turn))
    agent = _run_agent("cc/opus", tmp_path)
    assert agent.budget_used == pytest.approx(0.09)


def test_cc_helper_model_does_not_replace_the_stream_model(
    fake_claude: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a CLI estimate a two-model turn prices at the stream's main
    model, never at whichever model ``modelUsage`` lists first."""
    monkeypatch.setattr(DEFAULT_CONFIG, "claude_code_use_subscription", False)
    model_usage = {
        "claude-haiku-4-5": {"inputTokens": 50},
        "claude-opus-4-5": {"inputTokens": 1000},
    }
    turn = _claude_turn(init_model="claude-opus-4-5", model_usage=model_usage)
    fake_claude.write_text(json.dumps(turn))
    agent = _run_agent("cc/opus", tmp_path)
    assert agent.budget_used == pytest.approx(calculate_cost("claude-opus-4-5", *_CLAUDE_COUNTS))


def _save_codex_login(tmp_path: Path, auth: dict[str, Any]) -> None:
    """Write the ``auth.json`` ``codex login`` leaves under ``CODEX_HOME``."""
    home = tmp_path / "codex-home"
    home.mkdir(exist_ok=True)
    (home / "auth.json").write_text(json.dumps(auth))


def _without_key_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    """Switch off with no API key in the environment or the settings."""
    monkeypatch.setattr(DEFAULT_CONFIG, "codex_use_subscription", False)
    monkeypatch.delenv("OPENAI_API_KEY")
    monkeypatch.delenv("CODEX_API_KEY", raising=False)
    monkeypatch.delenv("CODEX_ACCESS_TOKEN", raising=False)
    monkeypatch.setattr(DEFAULT_CONFIG, "OPENAI_API_KEY", "")


def test_codex_switch_off_with_a_saved_api_key_login_is_billed(
    fake_codex: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No key variable, but ``codex login --with-api-key`` saved one: the CLI
    bills that key per token, so the turn is API spend."""
    _without_key_variables(monkeypatch)
    _save_codex_login(tmp_path, {"auth_mode": "apikey", "OPENAI_API_KEY": "sk-saved"})
    fake_codex.write_text(json.dumps(_codex_turn()))
    agent = _run_agent("codex/gpt-5.5", tmp_path)
    assert agent.budget_used == pytest.approx(calculate_cost("gpt-5.5", *_CODEX_COUNTS))


def test_codex_saved_api_key_login_behind_an_access_token_is_free(
    fake_codex: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``CODEX_ACCESS_TOKEN`` in the child environment is read before the
    saved key, so the subscription pays."""
    _without_key_variables(monkeypatch)
    monkeypatch.setenv("CODEX_ACCESS_TOKEN", "tok")
    _save_codex_login(tmp_path, {"auth_mode": "apikey", "OPENAI_API_KEY": "sk-saved"})
    fake_codex.write_text(json.dumps(_codex_turn()))
    assert _run_agent("codex/gpt-5.5", tmp_path).budget_used == 0.0


def test_codex_switch_off_with_a_chatgpt_login_is_free(
    fake_codex: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ChatGPT sign-in saved by ``codex login`` bills the plan."""
    _without_key_variables(monkeypatch)
    _save_codex_login(tmp_path, {"auth_mode": "chatgpt", "tokens": {"id_token": "x"}})
    fake_codex.write_text(json.dumps(_codex_turn()))
    assert _run_agent("codex/gpt-5.5", tmp_path).budget_used == 0.0


def _stop_on_token(token: str) -> None:
    """A token callback that behaves like the user pressing Stop."""
    raise KeyboardInterrupt


def _finish(result: str) -> str:
    """Finish the task with *result*."""
    return result


def test_turn_billed_when_the_tool_filter_flush_raises_after_generate(
    fake_claude: Path, fake_codex: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A lone ``{`` is held as a possible ``tool_calls`` block until the
    filter flushes it after generate() returned; Stop raises there, and
    the completed turn's usage is still recovered, exactly once."""
    monkeypatch.setattr(DEFAULT_CONFIG, "claude_code_use_subscription", False)
    monkeypatch.setattr(DEFAULT_CONFIG, "codex_use_subscription", False)
    fake_codex.write_text(json.dumps(_codex_turn("{")))
    held = {"type": "assistant", "message": {"content": [{"type": "text", "text": "{"}]}}
    result = {"type": "result", "result": "{", "usage": _CLAUDE_USAGE, "total_cost_usd": 0.0}
    fake_claude.write_text(json.dumps([held, result]))
    config = {"work_dir": str(tmp_path), "timeout": 30}
    codex = CodexModel("codex/gpt-5.5", config, token_callback=_stop_on_token)
    claude = ClaudeCodeModel("cc/claude-opus-5-5", config, token_callback=_stop_on_token)
    cases = [
        (codex, _CODEX_COUNTS, calculate_cost("gpt-5.5", *_CODEX_COUNTS)),
        (claude, _CLAUDE_COUNTS, calculate_cost("claude-opus-5-5", *_CLAUDE_COUNTS)),
    ]
    for model, counts, cost in cases:
        model.conversation = [{"role": "user", "content": TASK}]
        with pytest.raises(KeyboardInterrupt):
            model.generate_and_process_with_tools({"finish": _finish})
        partial = model.take_partial_usage_response()
        assert model.extract_input_output_token_counts_from_response(partial) == counts
        assert model.extract_cost_from_response(partial) == pytest.approx(cost)
        assert model.take_partial_usage_response() is None, "recovered once"
