"""Tests for KISSAgent retry behavior using real model-like objects without mocks."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from kiss.core.base import Base
from kiss.core.kiss_agent import KISSAgent
from kiss.core.kiss_error import KISSError


class _RetryableErrorModel:
    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.calls = 0
        self.model_name = "gpt-4o-mini"
        self.conversation: list[dict[str, Any]] = []

    def initialize(self, prompt: str, attachments: list[Any] | None = None) -> None:
        self.conversation.append({"role": "user", "content": prompt})

    def generate_and_process_with_tools(
        self, function_map: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], str, Any]:
        self.calls += 1
        if self.calls <= self.failures:
            raise Exception("Internal server error")
        return ([{"name": "finish", "arguments": {"result": "ok"}}], "done", object())

    def add_message_to_conversation(self, role: str, content: str) -> None:
        self.conversation.append({"role": role, "content": content})

    def set_usage_info_for_messages(self, usage_info: str) -> None:
        self.last_usage = usage_info

    def add_function_results_to_conversation_and_return(
        self, function_results: list[tuple[str, dict[str, Any]]]
    ) -> None:
        self.function_results = function_results

    def extract_input_output_token_counts_from_response(
        self, response: Any
    ) -> tuple[int, int, int, int]:
        return (0, 0, 0, 0)


class _NonRetryableErrorModel(_RetryableErrorModel):
    def generate_and_process_with_tools(
        self, function_map: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], str, Any]:
        raise Exception("Unauthorized: invalid API key")


@pytest.fixture(autouse=True)
def _restore_base_state() -> Iterator[None]:
    original_counter = Base.agent_counter
    original_budget = Base.global_budget_used
    yield
    Base.agent_counter = original_counter
    Base.global_budget_used = original_budget


def _make_agent(model_obj: Any, max_steps: int = 5) -> KISSAgent:
    agent = KISSAgent("RetryTest")
    agent.model = model_obj
    agent.model_name = model_obj.model_name
    agent.verbose = False
    agent.printer = None
    agent.is_agentic = True
    agent.max_steps = max_steps
    agent.max_budget = 1.0
    agent.function_map = {"finish": agent.finish}
    agent.messages = []
    agent.step_count = 0
    agent.total_tokens_used = 0
    agent.budget_used = 0.0
    agent.session_info = ""
    agent.run_start_timestamp = 0
    return agent


# ---------------------------------------------------------------------------
# kiss/core/kiss_agent.py — KISSAgent
# ---------------------------------------------------------------------------

def test_run_agentic_loop_retries_then_succeeds() -> None:
    model = _RetryableErrorModel(failures=2)
    agent = _make_agent(model)

    result = agent._run_agentic_loop()

    assert result == "ok"
    assert model.calls == 3
    retry_messages = [
        m for m in agent.messages if "Failed to get response from Model:" in str(m["content"])
    ]
    assert len(retry_messages) == 2
    assert (
        "Failed to get response from Model: Internal server error."
        in retry_messages[0]["content"]
    )


# ---------------------------------------------------------------------------
# kiss/core/kiss_agent.py — KISSAgent
# ---------------------------------------------------------------------------

def test_run_agentic_loop_raises_after_three_consecutive_retryable_errors() -> None:
    model = _RetryableErrorModel(failures=3)
    agent = _make_agent(model)

    with pytest.raises(KISSError, match="failed with 3 consecutive errors"):
        agent._run_agentic_loop()

    assert model.calls == 3
    retry_messages = [
        m for m in agent.messages if "Failed to get response from Model:" in str(m["content"])
    ]
    assert len(retry_messages) == 2


def test_run_agentic_loop_raises_immediately_for_non_retryable_error() -> None:
    agent = _make_agent(_NonRetryableErrorModel(failures=0))

    with pytest.raises(
        KISSError,
        match="Non-retryable error from model: Unauthorized: invalid API key",
    ):
        agent._run_agentic_loop()
