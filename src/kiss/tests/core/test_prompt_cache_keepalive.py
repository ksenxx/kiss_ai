# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the prompt-cache keep-alive during long tool calls.

A real :class:`KISSAgent` runs the native Anthropic adapter against the
scripted local Messages server (:mod:`local_anthropic_server`); the tests
check which tool calls get a keep-alive, what each ping sends, that pings
never enter the conversation, and that their cost lands in the budget.
The OpenAI-compatible adapter needs no keep-alive and is checked to send
none.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

import kiss.core.prompt_cache_keepalive as keepalive
from kiss.core.config import DEFAULT_CONFIG
from kiss.core.kiss_agent import KISSAgent
from kiss.core.kiss_error import BudgetExceededError
from kiss.core.models.anthropic_model import KEEP_ALIVE_MAX_TOKENS, KEEP_ALIVE_TEXT
from kiss.core.models.model_info import calculate_cost
from kiss.core.prompt_cache_keepalive import is_long_running_call
from kiss.tests.agents.sorcar import local_model_server as openai_server
from kiss.tests.core import local_anthropic_server as anthropic_server


def test_is_long_running_call() -> None:
    assert is_long_running_call("Bash", {"command": "ls", "timeout_seconds": 300})
    assert is_long_running_call("Bash", {"command": "ls", "timeout_seconds": "900"})
    assert is_long_running_call("run_agent", {"agent": "cron", "task": "x", "timeout": "600"})
    assert is_long_running_call("run_parallel", {"tasks": "[]"})
    assert is_long_running_call("run_commands_parallel", {"commands": "[]"})
    assert is_long_running_call("bash_job", {"job_id": "j1", "action": "wait"})
    assert not is_long_running_call("bash_job", {"job_id": "j1", "action": "tail"})
    assert not is_long_running_call("bash_job", {"job_id": "j1"})
    assert not is_long_running_call("Bash", {"command": "ls", "timeout_seconds": 120})
    assert not is_long_running_call("Bash", {"command": "ls", "timeout_seconds": "soon"})
    assert not is_long_running_call("Bash", {"command": "ls"})
    assert not is_long_running_call("Read", {"file_path": "/x", "timeout": None})


def slow_tool(timeout_seconds: int) -> str:
    """Sleep for one second (standing in for a long build) and report it.

    Args:
        timeout_seconds: Declared timeout; only its size matters to the keep-alive.
    """
    time.sleep(1.0)
    return f"slept with timeout {timeout_seconds}"


def failing_tool(timeout_seconds: int) -> str:
    """Sleep for one second, then fail the way a fan-out whose children overspent does.

    Args:
        timeout_seconds: Declared timeout; only its size matters to the keep-alive.
    """
    time.sleep(1.0)
    raise BudgetExceededError("children spent the budget")


def _run_anthropic_agent(
    monkeypatch: pytest.MonkeyPatch,
    url: str,
    script_len: int,
    tools: list[Any] | None = None,
    model_config: dict[str, Any] | None = None,
) -> KISSAgent:
    monkeypatch.setenv("ANTHROPIC_BASE_URL", url)
    monkeypatch.setattr(DEFAULT_CONFIG, "ANTHROPIC_API_KEY", "local")
    agent = KISSAgent("keepalive-test")
    agent.run(
        model_name=anthropic_server.MODEL,
        prompt_template="Call slow_tool once, then finish.",
        tools=tools or [slow_tool],
        max_steps=script_len + 2,
        max_budget=50.0,
        verbose=False,
        model_config=model_config,
    )
    return agent


def _script(timeout_seconds: int) -> list[dict[str, Any]]:
    return [
        anthropic_server.tool_use_message(
            "slow_tool", {"timeout_seconds": timeout_seconds}, 20_000, "toolu_1"
        ),
        anthropic_server.tool_use_message("finish", {"result": "<p>done</p>"}, 21_000, "toolu_2"),
    ]


def test_long_tool_call_pings_the_cache_and_pays_for_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(keepalive, "KEEP_ALIVE_INTERVAL_SECONDS", 0.3)
    script = _script(600)
    with anthropic_server.serve(script) as (url, requests):
        agent = _run_anthropic_agent(monkeypatch, url, len(script))
    steps = [r for r in requests if r.get("stream")]
    pings = [r for r in requests if not r.get("stream")]
    assert len(steps) == 2
    # A 1 s tool with a 0.3 s interval gets about 3 pings; the first one is
    # timed from the step's request start, so the count moves with how long
    # the scripted step itself took (never 0, never the cap of 10).
    assert 1 <= len(pings) <= 4
    step_one = steps[0]
    for ping in pings:
        # The cached prefix is byte-identical: tools, system, thinking and the
        # messages the step sent, followed by the assistant tool-use turn ...
        assert ping["tools"] == step_one["tools"]
        assert ping.get("system") == step_one.get("system")
        assert ping.get("thinking") == step_one.get("thinking")
        assert ping.get("tool_choice") == step_one.get("tool_choice")
        assert ping["messages"][: len(step_one["messages"])] == step_one["messages"]
        assistant_turn, placeholder = ping["messages"][len(step_one["messages"]) :]
        assert assistant_turn["role"] == "assistant"
        assert assistant_turn["content"][0]["type"] == "tool_use"
        assert assistant_turn["content"][0]["id"] == "toolu_1"
        # ... and one placeholder tool_result answering the in-flight call.
        assert placeholder == {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": KEEP_ALIVE_TEXT}
            ],
        }
        assert ping["max_tokens"] == KEEP_ALIVE_MAX_TOKENS
    # The finish step saw the real tool result, not the placeholder.
    final_messages = steps[1]["messages"]
    assert final_messages[: len(step_one["messages"])] == step_one["messages"]
    real_result = final_messages[-1]["content"][0]
    assert real_result["type"] == "tool_result"
    assert "slept with timeout 600" in str(real_result["content"])
    assert KEEP_ALIVE_TEXT not in str(final_messages)
    assert KEEP_ALIVE_TEXT not in str(agent.model.conversation)
    # Every ping is billed: the budget holds the two steps plus the pings.
    usage = anthropic_server.PING_USAGE
    ping_cost = calculate_cost(
        anthropic_server.MODEL,
        usage["input_tokens"],
        usage["output_tokens"],
        usage["cache_read_input_tokens"],
        usage["cache_creation_input_tokens"],
    )
    step_cost = sum(
        calculate_cost(anthropic_server.MODEL, turn["prompt_tokens"], 30) for turn in script
    )
    assert agent.budget_used == pytest.approx(step_cost + len(pings) * ping_cost)
    assert agent.total_tokens_used == sum(t["prompt_tokens"] + 30 for t in script) + len(
        pings
    ) * sum(usage.values())


def test_short_tool_call_gets_no_ping(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(keepalive, "KEEP_ALIVE_INTERVAL_SECONDS", 0.3)
    script = _script(120)
    with anthropic_server.serve(script) as (url, requests):
        agent = _run_anthropic_agent(monkeypatch, url, len(script))
    assert [bool(r.get("stream")) for r in requests] == [True, True]
    step_cost = sum(
        calculate_cost(anthropic_server.MODEL, turn["prompt_tokens"], 30) for turn in script
    )
    assert agent.budget_used == pytest.approx(step_cost)


def test_failed_ping_stops_the_keepalive_and_the_tool_completes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(keepalive, "KEEP_ALIVE_INTERVAL_SECONDS", 0.3)
    script = _script(600)
    with anthropic_server.serve(script, ping_status=400) as (url, requests):
        agent = _run_anthropic_agent(monkeypatch, url, len(script))
    pings = [r for r in requests if not r.get("stream")]
    assert len(pings) == 1  # the first ping fails and no second one is sent
    step_cost = sum(
        calculate_cost(anthropic_server.MODEL, turn["prompt_tokens"], 30) for turn in script
    )
    assert agent.budget_used == pytest.approx(step_cost)
    assert "slept with timeout 600" in str(agent.model.conversation)


def test_no_ping_when_prompt_caching_is_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(keepalive, "KEEP_ALIVE_INTERVAL_SECONDS", 0.2)
    script = _script(600)
    with anthropic_server.serve(script) as (url, requests):
        _run_anthropic_agent(monkeypatch, url, len(script), model_config={"enable_cache": False})
    # Without cache_control there is nothing to keep warm: the model returns
    # None on the first tick and the thread ends.
    assert [bool(r.get("stream")) for r in requests] == [True, True]
    assert all("cache_control" not in r for r in requests)


def test_pings_are_billed_when_the_tool_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(keepalive, "KEEP_ALIVE_INTERVAL_SECONDS", 0.3)
    script = [
        anthropic_server.tool_use_message(
            "failing_tool", {"timeout_seconds": 600}, 20_000, "toolu_1"
        ),
    ]
    with anthropic_server.serve(script) as (url, requests):
        monkeypatch.setenv("ANTHROPIC_BASE_URL", url)
        monkeypatch.setattr(DEFAULT_CONFIG, "ANTHROPIC_API_KEY", "local")
        agent = KISSAgent("keepalive-raise")
        with pytest.raises(BudgetExceededError, match="children spent"):
            agent.run(
                model_name=anthropic_server.MODEL,
                prompt_template="Call failing_tool.",
                tools=[failing_tool],
                max_steps=3,
                max_budget=50.0,
                verbose=False,
            )
    pings = [r for r in requests if not r.get("stream")]
    assert len(pings) >= 1
    usage = anthropic_server.PING_USAGE
    ping_cost = calculate_cost(
        anthropic_server.MODEL,
        usage["input_tokens"],
        usage["output_tokens"],
        usage["cache_read_input_tokens"],
        usage["cache_creation_input_tokens"],
    )
    step_cost = calculate_cost(anthropic_server.MODEL, 20_000, 30)
    # The pings completed before the tool raised are still in the budget.
    assert agent.budget_used == pytest.approx(step_cost + len(pings) * ping_cost)


def test_openai_compatible_model_sends_no_ping(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(keepalive, "KEEP_ALIVE_INTERVAL_SECONDS", 0.2)
    script = [
        openai_server.tool_call_body("slow_tool", {"timeout_seconds": 600}, 5000),
        openai_server.finish_body("<p>done</p>"),
    ]
    with openai_server.serve(script) as (url, requests):
        agent = KISSAgent("keepalive-openai")
        agent.run(
            model_name=openai_server.MODEL,
            prompt_template="Call slow_tool once, then finish.",
            tools=[slow_tool],
            max_steps=4,
            max_budget=5.0,
            verbose=False,
            model_config={"base_url": url, "api_key": "local"},
        )
    # keep_prompt_cache_warm() returns None for this adapter: the first tick
    # ends the thread and no extra request reaches the server.
    assert len(requests) == 2
    assert "slept with timeout 600" in str(agent.model.conversation)


def test_keepalive_stops_at_max_pings(monkeypatch: pytest.MonkeyPatch) -> None:
    # A 0.05 s interval would fit ~20 pings into the 1 s tool call; the cap holds.
    monkeypatch.setattr(keepalive, "KEEP_ALIVE_INTERVAL_SECONDS", 0.05)
    script = _script(600)
    with anthropic_server.serve(script) as (url, requests):
        _run_anthropic_agent(monkeypatch, url, len(script))
    pings = [r for r in requests if not r.get("stream")]
    assert len(pings) == keepalive.KEEP_ALIVE_MAX_PINGS
