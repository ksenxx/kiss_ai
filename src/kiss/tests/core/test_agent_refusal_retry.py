# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""A safety refusal with no fallback model gets one retry with the task's context stated.

An unattended run that stops at the first ``stop_reason=refusal`` loses the
whole task to one false positive of the safety classifier.  The agentic loop adds
:data:`kiss.core.kiss_agent.REFUSAL_RETRY_MSG` once and asks again; a
second refusal is final.  As in the stall tests, a local HTTP server
speaks the Anthropic SSE wire format to the real SDK client (no mocks).
"""

from __future__ import annotations

import json
import threading
from collections.abc import Generator
from http.server import BaseHTTPRequestHandler

import pytest

from kiss.core.kiss_agent import REFUSAL_RETRY_MSG, KISSAgent
from kiss.core.kiss_error import ModelRefusalError
from kiss.tests.core.models.test_anthropic_stream_stall_timeout import (
    _FAST_FAIL_BUDGET,
    _MODEL,
    _DaemonThreadingHTTPServer,
    _finish_tool_stream,
    _message_start,
    _run_bounded,
    _sse,
)
from kiss.tests.core.test_anthropic_stream_stall_timeout import _ensure_api_key, _register_model


class _RefusalState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        self.refuse_first_n = 1
        self.request_count = 0
        self.bodies: list[dict] = []

    def next_request_refuses(self, body: dict) -> bool:
        with self.lock:
            self.request_count += 1
            self.bodies.append(body)
            return self.request_count <= self.refuse_first_n


_STATE = _RefusalState()


class _RefusalHandler(BaseHTTPRequestHandler):
    """Answers the first N requests with ``stop_reason: refusal`` and an empty message."""

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}
        model_name = body.get("model", _MODEL)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        if _STATE.next_request_refuses(body):
            chunks = [
                _message_start(model_name),
                _sse("message_delta", {"type": "message_delta",
                                       "delta": {"stop_reason": "refusal", "stop_sequence": None},
                                       "usage": {"output_tokens": 0}}),
                _sse("message_stop", {"type": "message_stop"}),
            ]
        else:
            chunks = _finish_tool_stream(model_name)
        for chunk in chunks:
            self.wfile.write(chunk)
            self.wfile.flush()

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


@pytest.fixture
def refusal_server() -> Generator[str]:
    _STATE.reset()
    server = _DaemonThreadingHTTPServer(("127.0.0.1", 0), _RefusalHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def _run(agent: KISSAgent) -> object:
    return _run_bounded(
        lambda: agent.run(
            model_name=_MODEL,
            prompt_template="Audit the firmware image in ./fw.bin for the known bootkit.",
            max_steps=5, max_budget=1.0, verbose=False,
        ),
        deadline=2 * _FAST_FAIL_BUDGET,
    )


def test_one_refusal_is_retried_with_the_context_note(
    monkeypatch: pytest.MonkeyPatch, refusal_server: str,
) -> None:
    _register_model(monkeypatch)
    _ensure_api_key(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", refusal_server)
    agent = KISSAgent("test-refusal-retry")
    assert _run(agent) == ("ok", "recovered")
    assert _STATE.request_count == 2
    # the retry carried the note as a user message after the refused turn
    retry_messages = _STATE.bodies[1]["messages"]
    texts = [m["content"] if isinstance(m["content"], str) else json.dumps(m["content"])
             for m in retry_messages if m["role"] == "user"]
    assert any(REFUSAL_RETRY_MSG in t for t in texts)
    assert any(m["role"] == "user" and REFUSAL_RETRY_MSG in str(m["content"])
               for m in agent.messages)


def test_a_second_refusal_is_final(monkeypatch: pytest.MonkeyPatch, refusal_server: str) -> None:
    _register_model(monkeypatch)
    _ensure_api_key(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", refusal_server)
    _STATE.refuse_first_n = 10
    agent = KISSAgent("test-refusal-final")
    outcome = _run(agent)
    assert isinstance(outcome, ModelRefusalError)
    assert _STATE.request_count == 2
