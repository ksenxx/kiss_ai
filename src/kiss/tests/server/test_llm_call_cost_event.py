"""End-to-end tests for the per-call ``llm_call`` cost event.

``KISSAgent._execute_step`` emits one ``llm_call`` event per model call
with the call's own tokens, USD cost and duration; ``JsonPrinter``
broadcasts and records it; and the autorouter SEA's
``observed_call_costs`` tool aggregates the persisted events per model.
The model here is a scripted stand-in (the repository's pattern for
driving ``_execute_step`` without a network call); everything else —
agent, printer, persistence, tool — is real.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from typing import Any

import pytest

import kiss.agents.sorcar.persistence as th
from kiss.agents.seas.autorouter.autorouter_sea import observed_call_costs
from kiss.core.kiss_agent import KISSAgent
from kiss.core.models.model_info import calculate_cost
from kiss.server.json_printer import JsonPrinter

from .test_history_task_meta_server import _redirect, _restore


class _ScriptedModel:
    """Returns one finish call with a fixed token usage, or raises."""

    def __init__(
        self, usage: tuple[int, ...], raise_partial: bool = False, raise_unbilled: bool = False
    ) -> None:
        self.model_name = "gpt-4o-mini"
        self.conversation: list[dict[str, Any]] = []
        self.usage = usage
        self.raise_partial = raise_partial
        self.raise_unbilled = raise_unbilled
        self._partial: Any = None

    def initialize(self, prompt: str, attachments: list[Any] | None = None) -> None:
        self.conversation.append({"role": "user", "content": prompt})

    def generate_and_process_with_tools(
        self,
        function_map: dict[str, Any],
        tools_schema: Any = None,
    ) -> tuple[list[dict[str, Any]], str, Any]:
        if self.raise_partial:
            self._partial = object()
            raise RuntimeError("truncated after billing")
        if self.raise_unbilled:
            raise RuntimeError("connection reset before any usage")
        return [{"name": "finish", "arguments": {"result": "ok"}}], "done", object()

    def take_partial_usage_response(self) -> Any:
        partial, self._partial = self._partial, None
        return partial

    def add_message_to_conversation(self, role: str, content: str) -> None:
        self.conversation.append({"role": role, "content": content})

    def set_usage_info_for_messages(self, usage_info: str) -> None:
        pass

    def add_function_results_to_conversation_and_return(self, results: Any) -> None:
        pass

    def extract_input_output_token_counts_from_response(self, response: Any) -> tuple[int, ...]:
        return self.usage


def _agent(model: _ScriptedModel, printer: JsonPrinter | None) -> KISSAgent:
    agent = KISSAgent("LlmCallTest")
    agent.model = model  # type: ignore[assignment]
    agent.model_name = model.model_name
    agent.verbose = False
    agent.printer = printer  # type: ignore[assignment]
    agent.is_agentic = True
    agent.max_steps = 5
    agent.max_budget = 10.0
    agent.function_map = {"finish": agent.finish}
    agent.messages = []
    agent.step_count = 1
    agent.run_start_timestamp = 0
    agent._cached_tools_schema = None
    return agent


def _recording_printer() -> JsonPrinter:
    printer = JsonPrinter()
    printer._thread_local.task_id = "t-llm-call"
    printer.start_recording()
    return printer


class TestAgentEmitsLlmCall:
    """One ``llm_call`` per model call, priced like the budget."""

    def test_event_carries_the_calls_own_tokens_cost_and_duration(self) -> None:
        printer = _recording_printer()
        agent = _agent(_ScriptedModel((1000, 100, 300, 20)), printer)
        assert agent.last_call_usage is None
        agent._execute_step()
        events = printer.stop_recording()
        kinds = [e["type"] for e in events]
        assert kinds.index("llm_call") < kinds.index("usage_info")
        (call,) = [e for e in events if e["type"] == "llm_call"]
        expected_cost = calculate_cost("gpt-4o-mini", 1000, 100, 300, 20, 0)
        assert call["model"] == "gpt-4o-mini"
        assert call["step"] == 1
        assert call["input_tokens"] == 1000
        assert call["output_tokens"] == 100
        assert call["cache_read"] == 300
        assert call["cache_write"] == 20
        assert call["cost"] == pytest.approx(expected_cost)
        assert call["cost"] == pytest.approx(agent.budget_used)
        assert isinstance(call["duration_ms"], int) and call["duration_ms"] >= 0
        assert call["ts"] > 0
        assert agent.last_call_usage == {
            "input_tokens": 1000,
            "output_tokens": 100,
            "cache_read": 300,
            "cache_write": 20,
            "cost": expected_cost,
        }

    def test_seven_field_usage_and_no_printer(self) -> None:
        agent = _agent(_ScriptedModel((10, 5, 0, 0, 0, 7, 3)), None)
        agent._execute_step()
        assert agent.last_call_usage is not None
        # Per-call counts include the audio subsets (billed separately).
        assert agent.last_call_usage["input_tokens"] == 17
        assert agent.last_call_usage["output_tokens"] == 8
        assert agent.budget_used == pytest.approx(
            calculate_cost(
                "gpt-4o-mini", 10, 5, 0, 0, 0, num_audio_input_tokens=7, num_audio_output_tokens=3
            )
        )

    def test_billed_failure_emits_and_unbilled_failure_does_not(self) -> None:
        printer = _recording_printer()
        agent = _agent(_ScriptedModel((500, 0, 0, 0), raise_partial=True), printer)
        with pytest.raises(RuntimeError, match="truncated"):
            agent._execute_step()
        billed = [e for e in printer.stop_recording() if e["type"] == "llm_call"]
        assert len(billed) == 1 and billed[0]["input_tokens"] == 500
        assert billed[0]["cost"] == pytest.approx(calculate_cost("gpt-4o-mini", 500, 0, 0, 0, 0))

        printer = _recording_printer()
        agent = _agent(_ScriptedModel((500, 0, 0, 0), raise_unbilled=True), printer)
        with pytest.raises(RuntimeError, match="connection reset"):
            agent._execute_step()
        assert [e for e in printer.stop_recording() if e["type"] == "llm_call"] == []
        assert agent.last_call_usage is None

    def test_non_agentic_generation_emits_too(self) -> None:
        """``_generate_once`` (non-agentic and whole-task runs) is a billed call."""
        model = _ScriptedModel((200, 40, 0, 0))
        model.generate = lambda: ("plain answer", object())  # type: ignore[attr-defined]
        printer = _recording_printer()
        agent = _agent(model, printer)
        agent.step_count = 0
        assert agent._generate_once() == "plain answer"
        (call,) = [e for e in printer.stop_recording() if e["type"] == "llm_call"]
        assert call["input_tokens"] == 200 and call["output_tokens"] == 40
        assert call["step"] == 1
        assert call["cost"] == pytest.approx(calculate_cost("gpt-4o-mini", 200, 40, 0, 0, 0))

        failing = _ScriptedModel((300, 0, 0, 0))

        def raise_after_billing() -> Any:
            failing._partial = object()
            raise RuntimeError("timed out")

        failing.generate = raise_after_billing  # type: ignore[attr-defined]
        printer = _recording_printer()
        agent = _agent(failing, printer)
        with pytest.raises(RuntimeError, match="timed out"):
            agent._generate_once()
        (call,) = [e for e in printer.stop_recording() if e["type"] == "llm_call"]
        assert call["input_tokens"] == 300

    def test_printer_coerces_malformed_fields(self) -> None:
        printer = _recording_printer()
        printer.print("", type="llm_call", model=None, cost=None, step=None)
        (call,) = [e for e in printer.stop_recording() if e["type"] == "llm_call"]
        assert call["model"] == "" and call["cost"] == 0.0 and call["step"] == 0
        assert call["input_tokens"] == 0 and call["duration_ms"] == 0


class TestObservedCallCosts:
    """The autorouter reads the persisted events back per model."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)
        self.saved_home = os.environ.get("KISS_HOME")
        os.environ["KISS_HOME"] = str(th._KISS_DIR)

    def teardown_method(self) -> None:
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        if self.saved_home is None:
            os.environ.pop("KISS_HOME", None)
        else:
            os.environ["KISS_HOME"] = self.saved_home
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @staticmethod
    def _call(
        model: str, cost: float, inp: int, out: int, cache: int, ms: int, cache_write: int = 0
    ) -> dict[str, Any]:
        return {
            "type": "llm_call",
            "model": model,
            "step": 1,
            "duration_ms": ms,
            "input_tokens": inp,
            "output_tokens": out,
            "cache_read": cache,
            "cache_write": cache_write,
            "cost": cost,
        }

    def test_aggregates_recent_tasks_per_model(self) -> None:
        assert observed_call_costs() == "[]"  # no database yet
        recent, _ = th._add_task("recent task")
        other, _ = th._add_task("another recent task")
        old, _ = th._add_task("old task")
        th._get_db().execute(
            "UPDATE task_history SET timestamp = timestamp - 30 * 86400 WHERE id = ?", (old,)
        )
        th._get_db().commit()
        events: list[tuple[str, dict[str, object]]] = [
            (recent, self._call("m1", 0.02, 100, 10, 100, 2000)),
            # 250 uncached + 50 cache-write tokens: both are prompt tokens.
            (recent, self._call("m1", 0.04, 250, 30, 100, 4000, cache_write=50)),
            (other, self._call("m2", 0.001, 50, 5, 0, 500)),
            (other, {"type": "tool_call", "name": "Bash", "extras": {"x": "llm_call"}}),
            (other, {"type": "llm_call", "model": ""}),
            (old, self._call("m1", 9.0, 1, 1, 0, 1)),
        ]
        for task_id, event in events:
            th._append_chat_event(event, task_id=task_id)
        th._flush_chat_events()
        report = json.loads(observed_call_costs(days=7))
        assert [r["model"] for r in report] == ["m1", "m2"]
        m1 = report[0]
        assert m1["calls"] == 2 and m1["tasks"] == 1
        assert m1["total_usd"] == pytest.approx(0.06)
        assert m1["mean_usd_per_call"] == pytest.approx(0.03)
        assert m1["median_usd_per_call"] == pytest.approx(0.03)
        assert m1["mean_input_tokens"] == 300  # (100+100 + 300+100) / 2
        assert m1["mean_output_tokens"] == 20
        assert m1["cache_read_share"] == pytest.approx(200 / 600, abs=1e-3)
        assert m1["mean_seconds"] == 3.0
        assert report[1]["cache_read_share"] == 0.0
        only = json.loads(observed_call_costs(days=7, model="m2"))
        assert [r["model"] for r in only] == ["m2"]
        # A window reaching the old task counts its call too.
        wide = json.loads(observed_call_costs(days=60, model="m1"))
        assert wide[0]["calls"] == 3
        assert json.loads(observed_call_costs(model="nonesuch")) == []
