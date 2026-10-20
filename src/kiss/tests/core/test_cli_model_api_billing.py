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
from pathlib import Path
from typing import Any

import pytest

from kiss.core.config import DEFAULT_CONFIG
from kiss.core.kiss_agent import KISSAgent
from kiss.core.kiss_error import KISSError
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
    }
    assert model.extract_cost_from_response(partial) == pytest.approx(
        calculate_cost("claude-opus-5-5", 10, 20)
    )
    assert model.take_partial_usage_response() is None, "recovered once"
    assert model.extract_cost_from_response("not a dict") is None


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


def test_codex_api_key_turn_on_the_cli_default_model_warns_once_and_stays_free(
    fake_codex: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``codex/default`` names no catalog model: the spend cannot be priced,
    so it stays at the catalog's $0 with one warning per model instance."""
    monkeypatch.setattr(DEFAULT_CONFIG, "codex_use_subscription", False)
    fake_codex.write_text(json.dumps(_codex_turn()))
    model = CodexModel("codex/default", {"work_dir": str(tmp_path), "timeout": 30})
    model.conversation = [{"role": "user", "content": TASK}]
    with caplog.at_level(logging.WARNING, logger="kiss.core.models.codex_model"):
        for _ in range(2):
            _text, response = model.generate()
            assert model.extract_cost_from_response(response) is None
    warnings = [r for r in caplog.records if "no price for 'default'" in r.getMessage()]
    assert len(warnings) == 1
    assert model.extract_cost_from_response("not a dict") is None
