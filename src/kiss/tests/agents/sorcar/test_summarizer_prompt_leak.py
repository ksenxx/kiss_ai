# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E tests: the failed-session summarizer's events as the user sees them.

When an executor sub-session crashes (e.g. the model returns consecutive
empty responses, or the step limit is exceeded), ``RelentlessAgent``
runs an internal :class:`KISSAgent` summarizer that inherits the
parent's printer. Two things went wrong with its events:

* ``KISSAgent._set_prompt`` unconditionally printed the summarizer's
  internal ``SUMMARIZER_PROMPT`` with ``type="prompt"``, so the
  front-end displayed an unexpected "# Summarizer\\n\\nThe executor's
  trajectory is saved at: ..." prompt message in the task's event
  stream (observed in production task
  ``1ae49939d2a34039b72e8234eed52b02`` in ``~/.kiss/history.db``).
* The summarizer used the built-in plain-text ``finish(result)``, so
  its terminal ``result`` event carried the HTML summary as bare
  ``text`` with no ``summary`` / ``is_continue`` keys; the chat webview
  escaped it, showing ``<h3>``/``<ol>`` literally (observed in task
  ``ab25a870`` in ``~/.s10s/history.db``), unlike the end-of-task
  result panel which comes from the structured ``finish``.

These tests drive a real ``RelentlessAgent`` against a real
``ThreadingHTTPServer`` speaking the OpenAI chat-completions protocol
and records every printer event with a real ``Printer`` subclass.
Nothing in the system under test (agents, model adapters, printers) is
mocked or patched; only the remote LLM service is replaced by a real
local HTTP server speaking the same protocol.
"""

from __future__ import annotations

import json
import tempfile
import threading
from collections.abc import Generator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from kiss.agents.sorcar.relentless_agent import RelentlessAgent
from kiss.core.kiss_error import KISSError
from kiss.core.printer import Printer

_MODEL = "gpt-4o-mini"
_TASK = "Do nothing forever."


class EventCollectorPrinter(Printer):
    """A real printer that records every event it is asked to render."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []
        self._lock = threading.Lock()

    def print(self, content: Any, type: str = "text", **kwargs: Any) -> str:
        """Record the (type, content) pair of every print call.

        Args:
            content: The content to display.
            type: Content type (e.g. "text", "prompt", "tool_call").
            **kwargs: Additional type-specific options (ignored).

        Returns:
            str: Always the empty string.
        """
        with self._lock:
            self.events.append((type, str(content)))
        return ""

    def token_callback(self, token: str) -> None:
        """Ignore streamed tokens.

        Args:
            token: The text token (ignored).
        """

    def reset(self) -> None:
        """Reset streaming state (no-op)."""


def _executor_response() -> dict:
    """Non-finish tool call — keeps the executor looping until ``max_steps``."""
    return {
        "id": "chatcmpl-exec",
        "object": "chat.completion",
        "model": _MODEL,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Calling noop.",
                    "tool_calls": [
                        {
                            "id": "call_noop",
                            "type": "function",
                            "function": {"name": "noop", "arguments": "{}"},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {
            "prompt_tokens": 1000,
            "completion_tokens": 100,
            "total_tokens": 1100,
        },
    }


_SUMMARY_HTML = "<h3>summary-from-test</h3>\n<ol>\n<li>Read <code>a.py</code></li>\n</ol>"


def _summarizer_finish_response() -> dict:
    """``finish(result=<HTML>)`` so the summarizer returns at once.

    The summarizer's ``KISSAgent`` registers ``SummarizerFinish.finish``,
    which keeps the one-argument ``finish(result)`` signature.
    """
    args = json.dumps({"result": _SUMMARY_HTML})
    return {
        "id": "chatcmpl-sum",
        "object": "chat.completion",
        "model": _MODEL,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_fin",
                            "type": "function",
                            "function": {"name": "finish", "arguments": args},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {
            "prompt_tokens": 1000,
            "completion_tokens": 100,
            "total_tokens": 1100,
        },
    }


def _summarizer_text_response() -> dict:
    """A text-only summarizer turn: the HTML summary as content, no tool call.

    Two of these in a row trip ``KISSAgent``'s text-only net, which
    finishes the summarizer implicitly through its registered ``finish``.
    """
    return {
        "id": "chatcmpl-sum-text",
        "object": "chat.completion",
        "model": _MODEL,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": _SUMMARY_HTML},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 1000,
            "completion_tokens": 100,
            "total_tokens": 1100,
        },
    }


_summarizer_called = threading.Event()
#: Set by a test to make the summarizer answer with text only (no finish call).
_summarizer_text_only = threading.Event()


def _sse_chunks(payload: dict) -> list[dict]:
    """Convert a full chat-completions payload into streaming chunks.

    Args:
        payload: A complete (non-streaming) chat-completions response.

    Returns:
        list[dict]: Chunks in OpenAI streaming format — a delta chunk
        carrying the message content and tool calls, a finish chunk, and
        a usage-only chunk (matching ``stream_options.include_usage``).
    """
    message = payload["choices"][0]["message"]
    delta: dict = {"role": "assistant", "content": message["content"]}
    if message.get("tool_calls"):
        delta["tool_calls"] = [
            {**tc, "index": i} for i, tc in enumerate(message["tool_calls"])
        ]
    base = {"id": payload["id"], "object": "chat.completion.chunk", "model": _MODEL}
    return [
        {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
        {
            **base,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": payload["choices"][0]["finish_reason"],
                }
            ],
        },
        {**base, "choices": [], "usage": payload["usage"]},
    ]


class _PromptLeakHandler(BaseHTTPRequestHandler):
    """Routes by prompt content: summarizer prompts get a finish call,
    executor prompts get a non-finish tool call. Supports both streaming
    (SSE) and non-streaming chat-completions requests, because an agent
    with a printer sets a token callback which enables streaming.
    """

    def do_POST(self) -> None:  # noqa: N802
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode(errors="ignore")
        is_summarizer = "Summarizer" in body
        if is_summarizer:
            _summarizer_called.set()
        if not is_summarizer:
            payload = _executor_response()
        elif _summarizer_text_only.is_set():
            payload = _summarizer_text_response()
        else:
            payload = _summarizer_finish_response()
        try:
            wants_stream = bool(json.loads(body).get("stream"))
        except Exception:
            wants_stream = False
        if wants_stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for chunk in _sse_chunks(payload):
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            return
        body_out = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body_out)))
        self.end_headers()
        self.wfile.write(body_out)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


@pytest.fixture
def prompt_leak_server() -> Generator[str]:
    """Start a real OpenAI-protocol HTTP server for the agent to call."""
    _summarizer_called.clear()
    _summarizer_text_only.clear()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _PromptLeakHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class TestSummarizerPromptDoesNotLeakIntoEvents:
    """The internal Summarizer prompt must never surface as a user event."""

    def test_summarizer_prompt_not_printed_as_prompt_event(
        self, prompt_leak_server: str
    ) -> None:
        """Force an executor crash → summarizer cycle and verify the
        summarizer's internal prompt is not emitted to the shared printer.

        With ``max_steps=3`` the executor's ReAct loop exhausts its
        three iterations without a finish and raises ``KISSError``
        (with ``step_count == 3``); because no cause is chained and
        ``step_count > 1``, ``perform_task`` runs the summarizer with
        the parent's printer. Before the fix, the summarizer's
        ``# Summarizer`` prompt appeared as a ``type="prompt"`` event
        in the recorded stream.
        """
        printer = EventCollectorPrinter()
        agent = RelentlessAgent("summarizer-prompt-leak")
        with tempfile.TemporaryDirectory() as td:
            with pytest.raises(KISSError):
                agent.run(
                    model_name=_MODEL,
                    prompt_template=_TASK,
                    max_steps=3,
                    max_budget=1.00,
                    max_sub_sessions=1,
                    work_dir=td,
                    printer=printer,
                    model_config={
                        "base_url": prompt_leak_server,
                        "api_key": "test-key",
                    },
                )

        assert _summarizer_called.is_set(), (
            "summarizer never called the model — the test cannot prove the "
            "prompt leak because the summarizer branch was not exercised"
        )

        prompt_events = [text for t, text in printer.events if t == "prompt"]
        assert prompt_events, "the task prompt itself must still be printed"

        leaked = [p for p in prompt_events if "# Summarizer" in p]
        assert not leaked, (
            "the internal Summarizer prompt leaked into the user-visible "
            f"event stream as a type='prompt' event: {leaked[0][:200]!r}"
        )

        non_task = [p for p in prompt_events if _TASK not in p]
        assert not non_task, (
            "unexpected non-task prompt event(s) leaked into the "
            f"user-visible event stream: {non_task[0][:200]!r}"
        )


def _summarizer_result_events(base_url: str) -> list[dict[str, Any]]:
    """Run a crash → summarizer cycle through a real recording ``JsonPrinter``.

    Args:
        base_url: The local OpenAI-protocol server's base URL.

    Returns:
        list[dict]: The ``result`` events the webview would receive, in
        order (the summarizer's, then the merged end-of-task failure).
    """
    from kiss.server.json_printer import JsonPrinter

    printer = JsonPrinter()
    printer._thread_local.task_id = "summarizer-result-event-html"
    printer.start_recording()
    agent = RelentlessAgent("summarizer-result-event-html")
    with tempfile.TemporaryDirectory() as td:
        with pytest.raises(KISSError):
            agent.run(
                model_name=_MODEL,
                prompt_template=_TASK,
                max_steps=3,
                max_budget=1.00,
                max_sub_sessions=1,
                work_dir=td,
                printer=printer,
                model_config={"base_url": base_url, "api_key": "test-key"},
            )
    events = printer.stop_recording()
    assert _summarizer_called.is_set(), "summarizer branch was not exercised"
    return [e for e in events if e["type"] == "result"]


def _assert_structured_continuation(results: list[dict[str, Any]]) -> None:
    """Check the summarizer's result event is a renderable continuation.

    Args:
        results: The ``result`` events of one crash → summarizer cycle.
    """
    continuations = [e for e in results if e.get("is_continue")]
    assert len(continuations) == 1, (
        f"expected exactly one continuation result event, got {results!r}"
    )
    cont = continuations[0]
    assert cont["success"] is False
    assert cont["summary"] == _SUMMARY_HTML, cont
    assert "<h3>summary-from-test</h3>" in cont["text"]
    # Every result event the webview gets must be parseable: a
    # bare-text result (the old summarizer finish) is rendered escaped.
    assert all("summary" in e for e in results), results
    # The merged end-of-task result repeats the session summary as
    # HTML too (its "Previous Session" section), unescaped.
    final = results[-1]
    assert final is not cont
    assert "<h3>summary-from-test</h3>" in final["summary"]
    assert "&lt;h3&gt;" not in final["summary"]


class TestSummarizerResultEventIsStructured:
    """The summarizer's ``result`` event renders as a continuation panel.

    ``createResultPanel`` in the chat webview renders ``summary`` as
    markup and prepends "Status: Continue" for ``is_continue``; an
    event with only ``text`` is escaped into a ``<pre>`` block, which
    is how the summarizer's HTML used to show its tags literally.
    """

    def test_explicit_finish_call(self, prompt_leak_server: str) -> None:
        """The summarizer calls ``finish(result=<HTML>)``: its result
        event is ``success: false, is_continue: true`` with the HTML in
        ``summary``.
        """
        _assert_structured_continuation(_summarizer_result_events(prompt_leak_server))

    def test_text_only_implicit_finish(self, prompt_leak_server: str) -> None:
        """The summarizer only talks (two text-only turns, no ``finish``
        call): ``KISSAgent``'s text-only net finishes it implicitly, and
        that result must go through the registered ``finish`` too rather
        than surface as bare text.
        """
        _summarizer_text_only.set()
        _assert_structured_continuation(_summarizer_result_events(prompt_leak_server))
