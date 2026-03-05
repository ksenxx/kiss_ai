"""Unit tests for KISSAgent guardrails around empty/no-tool model responses."""

from __future__ import annotations

from typing import Any

import pytest
import yaml

from kiss.core.base import Base
from kiss.core.kiss_agent import KISSAgent
from kiss.core.kiss_error import KISSError


class _FakeModel:
    def __init__(self, responses: list[tuple[list[dict[str, Any]], str, dict[str, Any]]]) -> None:
        self.responses = list(responses)
        self.model_name = "gpt-5.3-codex-spark"
        self.added_messages: list[tuple[str, str]] = []
        self.function_results: list[tuple[str, dict[str, Any]]] = []

    def generate_and_process_with_tools(
        self, _function_map: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
        if not self.responses:
            raise AssertionError("No fake responses left")
        return self.responses.pop(0)

    def set_usage_info_for_messages(self, _usage: str) -> None:
        return None

    def add_message_to_conversation(self, role: str, content: str) -> None:
        self.added_messages.append((role, content))

    def add_function_results_to_conversation_and_return(
        self,
        function_results: list[tuple[str, dict[str, Any]]],
    ) -> None:
        self.function_results = function_results

    @staticmethod
    def extract_input_output_token_counts_from_response(
        response: dict[str, Any],
    ) -> tuple[int, int, int, int]:
        usage = response.get("usage", {})
        return (
            int(usage.get("input_tokens", 0) or 0),
            int(usage.get("output_tokens", 0) or 0),
            int(usage.get("cached_input_tokens", 0) or 0),
            0,
        )


def _finish(success: bool, summary: str) -> str:
    return yaml.dump({"success": success, "summary": summary}, sort_keys=False)


def _make_agent(
    model: _FakeModel,
    function_map: dict[str, Any],
) -> KISSAgent:
    agent = KISSAgent("guardrail-test")
    agent.model = model
    agent.model_name = model.model_name
    agent.function_map = function_map
    agent.messages = []
    agent.step_count = 1
    agent.max_steps = 100
    agent.max_budget = 10.0
    agent.total_tokens_used = 0
    agent.budget_used = 0.0
    agent.printer = None
    agent.consecutive_no_tool_calls = 0
    agent.non_finish_tool_calls_executed = 0
    return agent


@pytest.fixture(autouse=True)
def _restore_global_budget() -> Any:
    original = Base.global_budget_used
    try:
        yield
    finally:
        Base.global_budget_used = original


def test_empty_zero_usage_response_raises_fast() -> None:
    model = _FakeModel(
        [
            (
                [],
                "",
                {"usage": {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0}},
            )
        ]
    )
    agent = _make_agent(model, {"finish": _finish})
    with pytest.raises(KISSError, match="empty response with zero token usage"):
        agent._execute_step()


def test_consecutive_no_tool_calls_hit_limit() -> None:
    model = _FakeModel(
        [
            (
                [],
                "Planning only.",
                {"usage": {"input_tokens": 10, "cached_input_tokens": 0, "output_tokens": 5}},
            ),
            (
                [],
                "Still planning.",
                {"usage": {"input_tokens": 10, "cached_input_tokens": 0, "output_tokens": 5}},
            ),
            (
                [],
                "No tools again.",
                {"usage": {"input_tokens": 10, "cached_input_tokens": 0, "output_tokens": 5}},
            ),
        ]
    )
    agent = _make_agent(model, {"finish": _finish})
    assert agent._execute_step() is None
    assert agent._execute_step() is None
    with pytest.raises(KISSError, match="no tool calls for 3 consecutive steps"):
        agent._execute_step()


def test_success_finish_without_real_tool_work_is_rejected() -> None:
    model = _FakeModel(
        [
            (
                [{"name": "finish", "arguments": {"success": True, "summary": "done"}}],
                '{"tool_calls":[{"name":"finish","arguments":{"success":true,"summary":"done"}}]}',
                {"usage": {"input_tokens": 50, "cached_input_tokens": 0, "output_tokens": 20}},
            )
        ]
    )
    agent = _make_agent(model, {"finish": _finish})
    assert agent._execute_step() is None
    assert model.added_messages
    assert "before any non-finish tools were executed" in model.added_messages[-1][1]


def test_success_finish_after_non_finish_tool_is_allowed() -> None:
    def Write(path: str, content: str) -> str:  # noqa: N802
        return f"wrote {path} ({len(content)} chars)"

    model = _FakeModel(
        [
            (
                [
                    {
                        "name": "Write",
                        "arguments": {"path": "C:/tmp/main.rs", "content": "fn main() {}"},
                    },
                    {"name": "finish", "arguments": {"success": True, "summary": "done"}},
                ],
                "tool calls",
                {"usage": {"input_tokens": 100, "cached_input_tokens": 0, "output_tokens": 30}},
            )
        ]
    )
    agent = _make_agent(model, {"finish": _finish, "Write": Write})
    result = agent._execute_step()
    parsed = yaml.safe_load(result)
    assert parsed["success"] is True
