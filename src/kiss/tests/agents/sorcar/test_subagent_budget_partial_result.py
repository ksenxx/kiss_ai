# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A sub-agent that exhausts its budget returns a partial result.

Before: ``BudgetExceededError`` propagated out of the child, the fan-out
engine turned it into ``Unhandled exception: … budget exceeded`` and
the child's history row read ``Task failed`` — every step the child
took was lost to the parent (51 such sub-agents in one week).

Now: a sub-agent (``_subagent_info`` set) returns
``finish(success=False, is_continue=False, summary=<what it did>)``.
A top-level task keeps raising, because the server, the CLI and the
result panel all report the failure from the exception.

Every test talks to a real local OpenAI-compatible HTTP server and
runs the real agents; no mocks.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import tempfile
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
import yaml

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.relentless_agent import RelentlessAgent
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.core.kiss_agent import KISSAgent
from kiss.core.kiss_error import BudgetExceededError
from kiss.core.printer import Printer

_MODEL = "gpt-4o-mini"  # $0.15/M input, $0.60/M output


def note(text: str) -> str:
    """Record a note and keep going.

    Args:
        text: The note to record.

    Returns:
        A confirmation string.
    """
    return f"noted: {text}"


def _tool_call_body(
    name: str, arguments: dict[str, Any], prompt_tokens: int, completion_tokens: int = 100,
) -> bytes:
    return json.dumps({
        "id": "chatcmpl-partial",
        "object": "chat.completion",
        "created": 0,
        "model": _MODEL,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": f"Working on it: calling {name}.",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }).encode()


class _Handler(BaseHTTPRequestHandler):
    """Answer every request with the class-level ``body``."""

    body: bytes = b""

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        """Silence the access log."""

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        if length:
            self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(self.body)))
        self.end_headers()
        self.wfile.write(self.body)


@contextmanager
def _serve(body: bytes) -> Iterator[str]:
    """Serve *body* to every POST on a local port for the block's duration."""
    handler = type("Handler", (_Handler,), {"body": body})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=30)


@pytest.fixture
def note_forever() -> Iterator[str]:
    """Endpoint that calls ``note`` on every turn; ~$0.0011 per step."""
    with _serve(_tool_call_body("note", {"text": "still working"}, 7000)) as url:
        yield url


@pytest.fixture
def continue_expensively() -> Iterator[str]:
    """Endpoint that ends every session with ``is_continue=True`` for ~$0.30."""
    with _serve(_tool_call_body(
        "finish",
        {"success": False, "is_continue": True, "summary_in_html": "<p>did part A</p>"},
        2_000_000,
    )) as url:
        yield url


class RecordingPrinter(Printer):
    """Printer that records every event as ``(type, content, kwargs)``."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[tuple[str, Any, dict[str, Any]]] = []
        self.token_callback = None  # type: ignore[method-assign,assignment]

    def print(self, content: Any, type: str = "text", **kwargs: Any) -> str:  # noqa: A002
        self.events.append((type, content, kwargs))
        return str(content)

    def token_callback(self, token: str) -> None:  # type: ignore[no-redef]
        return None

    def reset(self) -> None:
        return None


def _run(agent: RelentlessAgent, base_url: str, max_budget: float, **kwargs: Any) -> str:
    with tempfile.TemporaryDirectory() as td:
        return agent.run(
            model_name=_MODEL,
            prompt_template="Keep taking notes; never finish.",
            tools=[note],
            max_steps=50,
            max_budget=max_budget,
            work_dir=td,
            verbose=False,
            model_config={"base_url": base_url, "api_key": "local"},
            **kwargs,
        )


class TestSubagentPartialResult:
    """A budget-exhausted sub-agent returns what it did instead of raising."""

    def test_subagent_returns_partial_result(self, note_forever: str) -> None:
        printer = RecordingPrinter()
        agent = RelentlessAgent("sub")
        agent._subagent_info = {"parent_task_id": "parent-1"}  # type: ignore[attr-defined]
        result = _run(agent, note_forever, max_budget=0.0025, printer=printer)

        payload = yaml.safe_load(result)
        assert payload["success"] is False
        assert payload["is_continue"] is False
        summary = payload["summary"]
        assert summary.startswith("<h3>Partial result: ")
        assert "budget exceeded" in summary
        assert f"in {agent.total_steps} steps" in summary
        assert agent.total_steps == 3, agent.total_steps
        # The trajectory tail: the model's narration and its tool calls.
        # Step 3's response pushed the spend over the cap, so its tool
        # never ran and it is not recorded; steps 1 and 2 are.
        assert summary.count("<li><pre>") == 2
        assert "Working on it: calling note." in summary
        assert "note(text=&#x27;still working&#x27;)" in summary
        # The usage block the agentic loop appends is stripped.
        assert "```text" not in summary
        # The parent-facing result event carries the same payload.
        results = [c for (t, c, _kw) in printer.events if t == "result"]
        assert len(results) == 1
        assert yaml.safe_load(str(results[0]))["summary"] == summary
        assert agent.budget_used >= 0.0025

    def test_top_level_task_still_raises(self, note_forever: str) -> None:
        agent = RelentlessAgent("top")
        with pytest.raises(BudgetExceededError, match="budget exceeded"):
            _run(agent, note_forever, max_budget=0.0025)
        assert agent.total_steps == 3

    def test_exhausted_between_sessions_keeps_prior_summaries(
        self, continue_expensively: str,
    ) -> None:
        agent = RelentlessAgent("sub-sessions")
        agent._subagent_info = {"parent_task_id": "parent-2"}  # type: ignore[attr-defined]
        result = _run(agent, continue_expensively, max_budget=0.10, max_sub_sessions=5)

        payload = yaml.safe_load(result)
        assert payload["success"] is False
        assert payload["is_continue"] is False
        summary = payload["summary"]
        assert "<h3>Previous Session 1</h3>" in summary
        assert "<p>did part A</p>" in summary
        assert "\n\n---\n\n<h3>Partial result: " in summary
        assert "budget exhausted" in summary
        assert "between sessions" in summary
        assert agent.total_steps == 1

    def test_top_level_exhausted_between_sessions_still_raises(
        self, continue_expensively: str,
    ) -> None:
        agent = RelentlessAgent("top-sessions")
        with pytest.raises(BudgetExceededError, match="budget exhausted"):
            _run(agent, continue_expensively, max_budget=0.10, max_sub_sessions=5)


def _spend_tool(agent: KISSAgent) -> Callable[[str], str]:
    def spend(note: str) -> str:
        """Do work that costs the agent budget (like a run_parallel fan-out).

        Args:
            note: What the spend is for.

        Returns:
            A confirmation string.
        """
        agent.budget_used += 1.0
        return f"spent on {note}"

    return spend


class TestLimitHitAfterToolRanIsRecorded:
    """A step whose tool pushed the spend over the cap stays in the trajectory."""

    def test_step_recorded_before_budget_error(self) -> None:
        body = _tool_call_body("spend", {"note": "children"}, 10)
        with _serve(body) as base_url:
            agent = KISSAgent("spender")
            with pytest.raises(BudgetExceededError):
                agent.run(
                    _MODEL,
                    "Spend once.",
                    tools=[_spend_tool(agent)],
                    max_steps=5,
                    max_budget=0.5,
                    model_config={"base_url": base_url, "api_key": "local"},
                    verbose=False,
                )
        assert agent.step_count == 1
        model_messages = [m for m in agent.messages if m["role"] == "model"]
        assert len(model_messages) == 1
        assert "spend(note='children')" in model_messages[0]["content"]
        assert "[spend]: spent on children" in agent.messages[-1]["content"]


class TestFanOutReceivesPartialResult:
    """Through the real fan-out engine, the parent sees the partial result."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
        kiss_dir = Path(self.tmpdir) / ".kiss"
        kiss_dir.mkdir(parents=True)
        th._KISS_DIR = kiss_dir
        th._DB_PATH = kiss_dir / "sorcar.db"
        th._db_conn = None

    def teardown_method(self) -> None:
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        (th._DB_PATH, th._db_conn, th._KISS_DIR) = self.saved
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_child_partial_result_reaches_parent_and_history(self) -> None:
        # $0.051 per step (85k output tokens, under the 70 % context
        # hand-off of the 128k window): the child's $0.50 share is gone
        # at step 10, after nine recorded steps — one more than the
        # quoted tail holds.
        body = _tool_call_body(
            "Bash", {"command": "echo probing", "description": "d"}, 100, 85_000,
        )
        with _serve(body) as base_url:
            parent = SorcarAgent("fanout-parent")
            parent.model_name = _MODEL
            parent.model_config = {"base_url": base_url, "api_key": "local"}
            parent.work_dir = self.tmpdir
            parent.max_budget = 1.0
            parent.budget_used = 0.0
            parent._use_web_tools = False
            results = parent._run_tasks_parallel(["probe the budget"], max_workers=1)

        assert len(results) == 1
        payload = yaml.safe_load(results[0])
        assert payload["success"] is False
        assert "Unhandled exception" not in payload["summary"]
        assert "<h3>Partial result: " in payload["summary"]
        assert "budget exceeded" in payload["summary"]
        assert "Bash(command=&#x27;echo probing&#x27;" in payload["summary"]
        assert "(1 earlier steps omitted.)" in payload["summary"]
        assert payload["summary"].count("<li><pre>") == 8
        # The child's spend was attributed to the parent.
        assert parent.budget_used >= 0.5

        conn = sqlite3.connect(th._DB_PATH)
        rows = conn.execute("SELECT result FROM task_history").fetchall()
        conn.close()
        assert len(rows) == 1
        assert rows[0][0] != "Task failed"
        assert "Partial result" in rows[0][0]
