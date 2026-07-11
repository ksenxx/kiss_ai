# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the final ``type="result"`` event emitted by
:class:`kiss.core.relentless_agent.RelentlessAgent`.

Reproduces (and locks in the fix for) the front-end regression where the
Result panel of a multi-session task would show a stale
"Status: Continue" banner — instead of the true terminal FAILED/SUCCESS
outcome — whenever either:

* a later session terminated with ``is_continue=False`` after one or more
  ``is_continue=True`` continuations, or
* every sub-session slot was exhausted without a terminal return.

The clean-approach fix guarantees that whenever RelentlessAgent computes a
terminal outcome that could not be carried by the inner ``KISSAgent``'s
per-session Result event, RelentlessAgent itself emits one final merged
Result event with the merged summary and correct status flags.
"""

from __future__ import annotations

import http.server
import json
import tempfile
import threading
from typing import Any

import pytest
import yaml

from kiss.core.kiss_error import KISSError
from kiss.core.printer import Printer
from kiss.core.relentless_agent import (
    RelentlessAgent,
    _build_exhaustion_summary,
    _prior_sessions_section,
)


class RecordingPrinter(Printer):
    """Printer that records every event as ``(type, content, kwargs)``."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[tuple[str, Any, dict[str, Any]]] = []
        # ``KISSAgent._reset`` reads ``printer.token_callback`` and — when
        # truthy — routes the model call through the streaming code path.
        # Our fake OpenAI server returns plain JSON (not SSE), so force the
        # non-streaming path by overriding the abstract method with a
        # ``None`` instance attribute (see ``openai_compatible_model.py``:
        # ``if self.token_callback is None: fallback to non-streaming``).
        self.token_callback = None  # type: ignore[method-assign,assignment]

    def print(self, content: Any, type: str = "text", **kwargs: Any) -> str:  # noqa: A002
        self.events.append((type, content, kwargs))
        return str(content)

    def token_callback(self, token: str) -> None:  # type: ignore[no-redef]
        # Concrete implementation to satisfy the ABC; the instance attribute
        # assigned in ``__init__`` shadows it at runtime.
        return None

    def reset(self) -> None:
        return None

    def result_events(self) -> list[tuple[Any, dict[str, Any]]]:
        """Return ``(content, kwargs)`` for every ``type="result"`` event."""
        return [(c, kw) for (t, c, kw) in self.events if t == "result"]


def _make_tool_call_response(
    name: str, arguments: dict[str, Any], call_id: str = "call_1"
) -> dict[str, Any]:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(arguments),
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 10,
            "total_tokens": 20,
        },
    }


def _start_openai_server(responses: list[dict[str, Any]]) -> tuple[Any, int]:
    """Start a fake OpenAI-compatible server that replays sequential responses."""
    call_count = [0]

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            content_length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(content_length)
            idx = min(call_count[0], len(responses) - 1)
            call_count[0] += 1
            body = json.dumps(responses[idx]).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, port


def _run_agent(
    responses: list[dict[str, Any]],
    *,
    max_sub_sessions: int,
    printer: Printer,
) -> str:
    server, port = _start_openai_server(responses)
    try:
        agent = RelentlessAgent("FinalResultTest")
        with tempfile.TemporaryDirectory() as td:
            return agent.run(
                model_name="gpt-4o-mini",
                prompt_template="Do multi-step work.",
                max_steps=5,
                max_budget=1.0,
                max_sub_sessions=max_sub_sessions,
                work_dir=td,
                verbose=False,
                printer=printer,
                model_config={
                    "base_url": f"http://127.0.0.1:{port}/v1",
                    "api_key": "sk-test",
                },
            )
    finally:
        server.shutdown()


def _parse_result_payload(content: Any) -> dict[str, Any]:
    parsed = yaml.safe_load(str(content))
    assert isinstance(parsed, dict)
    return parsed


class TestFailureAfterContinueEmitsMergedFinalResult:
    """After N-1 continues + final ``is_continue=False, success=False``, the
    last emitted Result event MUST carry the MERGED summary and the terminal
    FAILED status — not the stale per-session "Continue" from the earlier
    inner emit."""

    def test_final_result_event_carries_merged_failure(self) -> None:
        resp_continue = _make_tool_call_response(
            "finish",
            {"success": False, "is_continue": True, "summary": "did A"},
        )
        resp_fail = _make_tool_call_response(
            "finish",
            {"success": False, "is_continue": False, "summary": "gave up: reason X"},
        )
        printer = RecordingPrinter()
        result = _run_agent(
            [resp_continue, resp_fail], max_sub_sessions=5, printer=printer
        )

        parsed = yaml.safe_load(result)
        assert parsed["success"] is False
        assert parsed["is_continue"] is False

        # The LAST Result event on the wire is what the front-end renders
        # into the Result panel — assert it carries the merged failure.
        # Exactly 3 events: 2 inner per-session Results (one per finish
        # tool call) + 1 outer merged Result from RelentlessAgent.
        result_events = printer.result_events()
        assert len(result_events) == 3, (
            "expected 2 inner + 1 outer merged Result; "
            f"got {len(result_events)}"
        )
        final_content, _ = result_events[-1]
        final_payload = _parse_result_payload(final_content)
        assert final_payload["success"] is False
        assert final_payload["is_continue"] is False
        summary = final_payload["summary"]
        assert "### Previous Session 1" in summary
        assert "did A" in summary
        assert "### Final Session" in summary
        assert "gave up: reason X" in summary


class TestExhaustionEmitsMergedFinalResult:
    """When every sub-session slot is used up without a terminal return, the
    last emitted Result event MUST carry a synthetic FAILED payload — the
    front-end otherwise sticks on the last per-session ``is_continue=True``
    Result and shows "Status: Continue" for a task that has actually
    failed."""

    def test_exhaustion_emits_terminal_failed_result(self) -> None:
        resp_continue = _make_tool_call_response(
            "finish",
            {"success": False, "is_continue": True, "summary": "step done"},
        )
        printer = RecordingPrinter()
        with pytest.raises(KISSError, match=r"Task failed after 2 sub-sessions"):
            _run_agent([resp_continue], max_sub_sessions=2, printer=printer)

        result_events = printer.result_events()
        # Exactly 3 events: 2 inner per-session Results (both
        # is_continue=True) + 1 outer merged exhaustion Result
        # (is_continue=False).
        assert len(result_events) == 3, (
            f"expected 2 inner + 1 outer exhaustion Result; got {len(result_events)}"
        )
        final_content, kwargs = result_events[-1]
        final_payload = _parse_result_payload(final_content)
        assert final_payload["success"] is False
        assert final_payload["is_continue"] is False
        summary = final_payload["summary"]
        assert "Task failed after 2 sub-sessions" in summary
        assert "### Previous Session 1" in summary
        assert "### Previous Session 2" in summary
        assert "step done" in summary
        # Exhaustion layout uses trailing separator + banner (no "Final
        # Session" header), matching the front-end's fallback split rule.
        assert "### Final Session" not in summary
        assert summary.rstrip().endswith("Task failed after 2 sub-sessions")
        # Merged event uses accumulated totals from the outer agent.
        assert "step_count" in kwargs
        assert "total_tokens" in kwargs
        assert "cost" in kwargs


class TestSingleSessionDoesNotEmitExtraFinalResult:
    """When the very first session terminates (single-session task), the
    inner ``KISSAgent``'s per-session Result event is already authoritative
    — RelentlessAgent must NOT emit an additional merged Result event
    (which would duplicate/overwrite the correct inner event)."""

    def test_single_session_success_no_extra_final_result(self) -> None:
        resp_done = _make_tool_call_response(
            "finish",
            {"success": True, "is_continue": False, "summary": "all done"},
        )
        printer = RecordingPrinter()
        result = _run_agent([resp_done], max_sub_sessions=5, printer=printer)

        parsed = yaml.safe_load(result)
        assert parsed["success"] is True
        # Exactly one Result event: the inner per-session emit.
        result_events = printer.result_events()
        assert len(result_events) == 1, (
            f"expected exactly one Result event for single-session success; "
            f"got {len(result_events)}"
        )
        inner_payload = _parse_result_payload(result_events[0][0])
        assert inner_payload["success"] is True
        assert inner_payload["summary"] == "all done"

    def test_single_session_failure_no_extra_final_result(self) -> None:
        resp_fail = _make_tool_call_response(
            "finish",
            {"success": False, "is_continue": False, "summary": "nope"},
        )
        printer = RecordingPrinter()
        result = _run_agent([resp_fail], max_sub_sessions=5, printer=printer)

        parsed = yaml.safe_load(result)
        assert parsed["success"] is False
        result_events = printer.result_events()
        assert len(result_events) == 1
        inner_payload = _parse_result_payload(result_events[0][0])
        assert inner_payload["success"] is False
        assert inner_payload["is_continue"] is False


class OffsetAwarePrinter(RecordingPrinter):
    """RecordingPrinter that mimics ``JsonPrinter``'s offset semantics.

    ``perform_task`` sets ``tokens_offset`` / ``budget_offset`` /
    ``steps_offset`` on the printer at the START of each sub-session to
    the aggregate BEFORE that session.  When the printer renders a
    ``type="result"`` event, it ADDS these offsets to the ``total_tokens``
    /  ``step_count`` / ``cost`` values from ``kwargs``.  If
    RelentlessAgent's merged emit passes CUMULATIVE totals (which already
    include the prior sessions), the offsets would double-count them.

    This printer captures the RESOLVED (offset-applied) values so tests
    can assert no double-count.
    """

    tokens_offset: int = 0
    budget_offset: float = 0.0
    steps_offset: int = 0

    def print(self, content: Any, type: str = "text", **kwargs: Any) -> str:  # noqa: A002
        if type == "result":
            resolved = dict(kwargs)
            resolved["total_tokens"] = (
                int(kwargs.get("total_tokens", 0)) + self.tokens_offset
            )
            resolved["step_count"] = (
                int(kwargs.get("step_count", 0)) + self.steps_offset
            )
            cost_str = str(kwargs.get("cost", "$0.0000"))
            if cost_str.startswith("$"):
                try:
                    resolved["cost"] = (
                        f"${float(cost_str[1:]) + self.budget_offset:.4f}"
                    )
                except ValueError:  # pragma: no cover
                    resolved["cost"] = cost_str
            self.events.append((type, content, resolved))
            return str(content)
        return super().print(content, type=type, **kwargs)


class TestMergedResultMetricsAreNotDoubleCounted:
    """The merged Result event's metric values MUST NOT double-count prior
    sessions.  When RelentlessAgent emits the merged event with CUMULATIVE
    totals, it must temporarily zero the printer's offset attributes (which
    hold the pre-session aggregate) so the offset-applying printer resolves
    them to exactly the cumulative aggregate — not aggregate + prior."""

    def test_offsets_zeroed_during_merged_emit(self) -> None:
        printer = OffsetAwarePrinter()
        resp_continue = _make_tool_call_response(
            "finish",
            {"success": False, "is_continue": True, "summary": "did A"},
        )
        resp_fail = _make_tool_call_response(
            "finish",
            {"success": False, "is_continue": False, "summary": "gave up"},
        )
        _run_agent(
            [resp_continue, resp_fail], max_sub_sessions=5, printer=printer
        )

        result_events = printer.result_events()
        assert len(result_events) == 3
        # Inner Result of session 2 has: raw total_tokens = session-2
        # tokens; resolved = raw + tokens_offset (= session-1 tokens),
        # which is the correct cumulative aggregate after session 2.
        _, inner_session2_kw = result_events[1]
        inner_session2_tokens = int(inner_session2_kw["total_tokens"])
        # Merged Result: resolved value MUST equal the cumulative
        # aggregate — same as the inner session-2 resolved value (both
        # reflect state at end of the terminal session).  If offsets
        # weren't zeroed during the merged emit, this would be
        # ``inner_session2_tokens + (session-1 tokens) = double-counted``.
        _, merged_kw = result_events[2]
        merged_tokens = int(merged_kw["total_tokens"])
        assert merged_tokens == inner_session2_tokens, (
            f"merged Result total_tokens ({merged_tokens}) should equal the "
            f"inner terminal-session resolved total_tokens "
            f"({inner_session2_tokens}); a larger value indicates the "
            f"printer's tokens_offset was double-counted."
        )
        # Same invariant for steps.
        inner_session2_steps = int(inner_session2_kw["step_count"])
        merged_steps = int(merged_kw["step_count"])
        assert merged_steps == inner_session2_steps, (
            f"merged Result step_count ({merged_steps}) should equal "
            f"the inner terminal-session resolved step_count "
            f"({inner_session2_steps})."
        )
        # Cost invariant.
        inner_session2_cost = str(inner_session2_kw["cost"])
        merged_cost = str(merged_kw["cost"])
        assert merged_cost == inner_session2_cost, (
            f"merged Result cost ({merged_cost}) should equal inner "
            f"terminal-session resolved cost ({inner_session2_cost})."
        )

    def test_offsets_restored_after_merged_emit(self) -> None:
        """After the merged emit completes, the printer's offsets MUST be
        restored to whatever they were before — the emit is a temporary
        adjustment, not a permanent reset."""
        printer = OffsetAwarePrinter()
        # Pre-set non-zero offsets to prove they survive the emit.
        # RelentlessAgent's per-session offset assignments happen inside
        # ``perform_task`` and will overwrite these; here we just verify
        # the SAVE/RESTORE contract on the helper directly.
        printer.tokens_offset = 999
        printer.budget_offset = 9.99
        printer.steps_offset = 7
        agent = RelentlessAgent("OffsetRestoreTest")
        # Minimal manual reset to attach a printer without running a task.
        agent.printer = printer
        agent.total_steps = 3
        agent.total_tokens_used = 42
        agent.budget_used = 0.5
        agent._emit_merged_result_event(
            {"success": True, "is_continue": False, "summary": "ok"}
        )
        assert printer.tokens_offset == 999
        assert printer.budget_offset == 9.99
        assert printer.steps_offset == 7


class TestEmptyTerminalSummaryStillSplittable:
    """When the terminal session returns an empty summary AFTER prior
    continuations, the merged payload MUST still contain the
    ``\\n\\n---\\n\\n`` separator (a "### Final Session\\n(no summary)"
    placeholder) so the front-end ``splitMultiSessionSummary`` can split
    into "Previous Sessions" + "Result" panels."""

    def test_empty_final_summary_uses_placeholder(self) -> None:
        resp_continue = _make_tool_call_response(
            "finish",
            {"success": False, "is_continue": True, "summary": "did A"},
        )
        resp_terminal = _make_tool_call_response(
            "finish",
            {"success": False, "is_continue": False, "summary": ""},
        )
        printer = RecordingPrinter()
        result = _run_agent(
            [resp_continue, resp_terminal], max_sub_sessions=5, printer=printer
        )

        parsed = yaml.safe_load(result)
        summary = parsed["summary"]
        assert "### Previous Session 1" in summary
        assert "\n\n---\n\n" in summary
        # Placeholder ensures the separator + a non-empty final segment.
        assert "### Final Session" in summary
        # Splittable by the front-end rule.
        head, sep, tail = summary.rpartition("\n\n---\n\n")
        assert sep == "\n\n---\n\n"
        assert head.strip()
        assert tail.strip()

        result_events = printer.result_events()
        assert len(result_events) == 3
        final_payload = _parse_result_payload(result_events[-1][0])
        assert "### Final Session" in final_payload["summary"]


class TestExhaustionSummaryHelper:
    """Unit tests for ``_build_exhaustion_summary`` — the small pure helper
    that shapes the merged exhaustion summary."""

    def test_no_prior_sessions_returns_banner_only(self) -> None:
        banner = "Task failed after 1 sub-sessions"
        assert _build_exhaustion_summary([], banner) == banner

    def test_prepends_prior_sessions_with_trailing_banner(self) -> None:
        banner = "Task failed after 2 sub-sessions"
        summary = _build_exhaustion_summary(["did A", "did B"], banner)
        assert summary == (
            f"{_prior_sessions_section(['did A', 'did B'])}\n\n---\n\n{banner}"
        )
        # The trailing "\n\n---\n\n<banner>" layout is what the front-end
        # `splitMultiSessionSummary` fallback rule detects (no "### Final
        # Session" header).
        assert summary.endswith(f"\n\n---\n\n{banner}")
        assert "### Previous Session 1" in summary
        assert "### Previous Session 2" in summary
