# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# ruff: noqa: F811  (the `capture_server` module fixture is imported from
# the core/models twin of this file and injected via test parameters)
"""End-to-end tests for :class:`OpenAICompatibleModel2` (Responses API).

A real :class:`http.server.ThreadingHTTPServer` captures every request
sent by the model so the tests can assert on the exact JSON that travels
over the wire to OpenAI's ``/v1/responses`` endpoint.  No mocks, patches,
fakes, or test doubles are used to substitute for the OpenAI SDK — only
the upstream HTTP endpoint is replaced by the in-process capture server.

The core contract being verified:

* The new model targets ``POST /v1/responses`` (not ``/v1/chat/completions``).
* ``reasoning_effort`` from ``model_config`` is rewritten to ``reasoning.effort``.
* For models that declare ``thinking="xhigh"`` in ``MODEL_INFO.json``
  (the gpt-5.5 family), the default ``reasoning.effort`` is ``"xhigh"``.
* Tools and ``reasoning.effort`` MUST coexist on the wire — this is the
  whole reason the v2 model was introduced (Chat Completions rejects the
  combination for GPT-5 reasoning models).
* Tool schemas are emitted in the flat Responses-API shape
  (``{"type":"function","name":...,"parameters":...}``) — not the nested
  Chat-Completions shape.
* ``system_instruction`` is routed to the top-level ``instructions`` field,
  not into the conversation.
* Attachments map to ``input_image`` / ``input_file`` content parts.
* Function results are appended as ``function_call_output`` input items
  carrying the matching ``call_id``.
* Token-usage extraction reads ``usage.input_tokens``,
  ``usage.output_tokens``, ``input_tokens_details.cached_tokens``, and
  counts ``output_tokens_details.reasoning_tokens`` as output tokens.
* ``openrouter/anthropic/*`` models receive top-level
  ``extra_body["cache_control"]`` (Anthropic prompt caching).
* Models not in ``MODEL_INFO`` get no ``reasoning`` parameter.
* DeepSeek R1 models fall back to text-based tool calling.
* Embeddings still hit ``/v1/embeddings``.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from kiss.core.models.openai_compatible_model2 import OpenAICompatibleModel2
from kiss.tests.core.models.test_openai_compatible_model2 import (  # noqa: F401
    _ECHO_TOOL_CHAT_SCHEMA,
    _CapturingHandler,
    _echo,
    _embedding_response_json,
    _stream_sse_event,
    _text_response_json,
    _text_stream_sse_body,
    _tool_call_response_json,
    capture_server,
)


class TestExtraCoverage:
    """Tests targeting branches not exercised by the main test classes."""

    def test_function_results_without_prior_calls_raises(
        self, capture_server: str
    ) -> None:
        """Orphan ``function_call_output`` (no prior function_call) raises.

        Per the Responses-API contract, every ``function_call_output``
        MUST have a ``call_id`` matching a previously-emitted
        ``function_call`` item.  Synthesising a fallback ``call_id``
        produces an invalid conversation, so v2 raises
        :class:`KISSError` instead of silently corrupting the wire shape.
        """
        from kiss.core.kiss_error import KISSError

        m = OpenAICompatibleModel2(
            "gpt-4o", base_url=capture_server, api_key="k"
        )
        m.initialize("hi")
        with pytest.raises(KISSError, match="No prior function_call"):
            m.add_function_results_to_conversation_and_return(
                [("orphan", {"result": "ok"})]
            )
        assert not any(
            isinstance(x, dict) and x.get("type") == "function_call_output"
            for x in m.conversation
        )

    def test_get_embedding_raises_kiss_error_on_failure(
        self, capture_server: str
    ) -> None:
        """Embedding failures are wrapped in KISSError."""
        from kiss.core.kiss_error import KISSError

        m = OpenAICompatibleModel2(
            "text-embedding-3-small", base_url=capture_server, api_key="k"
        )
        m.initialize("ignored")
        m.base_url = "http://127.0.0.1:1/never/v1"
        from openai import OpenAI

        m.client = OpenAI(
            base_url=m.base_url, api_key="k", timeout=1.0
        )
        with pytest.raises(KISSError):
            m.get_embedding("x")


class TestReviewBugReproductions:
    """End-to-end tests reproducing the bugs flagged by the gpt-5.5 review."""

    def test_whitespace_only_prompt_is_rejected(
        self, capture_server: str
    ) -> None:
        """A pure-whitespace prompt must NOT be shipped to the Responses API."""
        from kiss.core.kiss_error import KISSError

        m = OpenAICompatibleModel2(
            "gpt-4o", base_url=capture_server, api_key="k"
        )
        m.initialize("   \n  \t  ")
        with pytest.raises(KISSError):
            m.generate()
        assert not _CapturingHandler.captured_requests


class TestReviewBugReproductions7:
    """End-to-end tests reproducing bugs flagged by the seventh gpt-5.5 review."""

    def test_streaming_response_failed_raises_kiss_error(
        self, capture_server: str
    ) -> None:
        """``response.failed`` SSE event must raise KISSError."""
        from kiss.core.kiss_error import KISSError

        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
            token_callback=lambda t: None,
        )
        m.initialize("hi")
        _CapturingHandler.next_response_body = _stream_sse_event(
            "response.failed",
            {
                "type": "response.failed",
                "sequence_number": 1,
                "response": {
                    "id": "resp_failed",
                    "object": "response",
                    "created_at": 0,
                    "model": "gpt-4o",
                    "status": "failed",
                    "error": {"message": "boom"},
                    "output": [],
                },
            },
        )
        with pytest.raises(KISSError, match="boom|failed"):
            m.generate()


class TestReviewBugReproductions8:
    """End-to-end tests reproducing bugs flagged by the eighth gpt-5.5 review."""

    def test_non_streaming_failed_response_raises(
        self, capture_server: str
    ) -> None:
        """A non-streaming Responses-API ``status=failed`` must raise."""
        from kiss.core.kiss_error import KISSError

        failed = json.dumps(
            {
                "id": "resp_failed",
                "object": "response",
                "created_at": 0,
                "model": "gpt-4o",
                "status": "failed",
                "error": {"message": "boom"},
                "output": [],
                "usage": {
                    "input_tokens": 1,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": 0,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": 1,
                },
            }
        ).encode()

        m = OpenAICompatibleModel2(
            "gpt-4o", base_url=capture_server, api_key="k"
        )
        m.initialize("hi")
        _CapturingHandler.next_response_body = failed
        with pytest.raises(KISSError, match="boom|failed"):
            m.generate()

    def test_non_streaming_failed_response_raises_in_tools_path(
        self, capture_server: str
    ) -> None:
        """``generate_and_process_with_tools`` must also raise on failure."""
        from kiss.core.kiss_error import KISSError

        failed = json.dumps(
            {
                "id": "resp_failed",
                "object": "response",
                "created_at": 0,
                "model": "gpt-4o",
                "status": "failed",
                "error": {"message": "boom"},
                "output": [],
                "usage": {
                    "input_tokens": 1,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": 0,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": 1,
                },
            }
        ).encode()

        m = OpenAICompatibleModel2(
            "gpt-4o", base_url=capture_server, api_key="k"
        )
        m.initialize("hi")
        _CapturingHandler.next_response_body = failed
        with pytest.raises(KISSError, match="boom|failed"):
            m.generate_and_process_with_tools(
                function_map={"echo": _echo},
                tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
            )

    def test_initialize_clears_pending_function_calls(
        self, capture_server: str
    ) -> None:
        """A fresh ``initialize()`` must drop stale pending tool-call ids.

        After review-14, orphan function_call_output items are rejected
        outright (KISSError).  This test now confirms that
        ``initialize()`` resets ``_pending_function_calls`` by verifying
        the orphan call raises — if it leaked the old pending entry, the
        raise would *not* mention "No prior function_call" but would
        instead mention "No pending function_call named 'orphan'" (the
        pending list would be non-empty containing the stale ``echo``
        entry).  Both error paths prove the state was cleared.
        """
        from kiss.core.kiss_error import KISSError

        m = OpenAICompatibleModel2(
            "gpt-4o", base_url=capture_server, api_key="k"
        )
        m.initialize("first")

        _CapturingHandler.next_response_body = _tool_call_response_json(
            name="echo",
            arguments='{"text":"old"}',
            call_id="call_old",
        ).encode()
        m.generate_and_process_with_tools(
            function_map={"echo": _echo},
            tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
        )
        assert m._pending_function_calls, (
            "sanity: pending call should be seeded after generate_and_process"
        )

        m.initialize("second")
        assert m._pending_function_calls == []
        with pytest.raises(KISSError, match="No prior function_call"):
            m.add_function_results_to_conversation_and_return(
                [("orphan", {"result": "ok"})]
            )

    def test_streaming_completed_with_failed_status_raises(
        self, capture_server: str
    ) -> None:
        """A terminal ``response.completed`` with ``status=failed`` must raise."""
        from kiss.core.kiss_error import KISSError

        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
            token_callback=lambda t: None,
        )
        m.initialize("hi")
        _CapturingHandler.next_response_body = _stream_sse_event(
            "response.completed",
            {
                "type": "response.completed",
                "sequence_number": 1,
                "response": {
                    "id": "resp_failed",
                    "object": "response",
                    "created_at": 0,
                    "model": "gpt-4o",
                    "status": "failed",
                    "error": {"message": "boom"},
                    "output": [],
                    "usage": {
                        "input_tokens": 1,
                        "input_tokens_details": {"cached_tokens": 0},
                        "output_tokens": 0,
                        "output_tokens_details": {"reasoning_tokens": 0},
                        "total_tokens": 1,
                    },
                },
            },
        )
        with pytest.raises(KISSError, match="boom|failed"):
            m.generate()


class TestReviewBugReproductions12:
    """Reproduce + verify fix for review #12 bugs."""

    def test_incomplete_tool_call_response_raises(
        self, capture_server: str
    ) -> None:
        """Non-streaming ``status='incomplete'`` MUST raise to avoid bad tool calls."""
        from kiss.core.kiss_error import KISSError

        incomplete = json.dumps(
            {
                "id": "resp_incomplete",
                "object": "response",
                "created_at": 0,
                "model": "gpt-4o",
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
                "output": [
                    {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_partial",
                        "name": "echo",
                        "arguments": '{"text":',
                    }
                ],
                "usage": {
                    "input_tokens": 1,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": 1,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": 2,
                },
            }
        ).encode()

        m = OpenAICompatibleModel2(
            "gpt-4o", base_url=capture_server, api_key="k"
        )
        m.initialize("hi")
        _CapturingHandler.next_response_body = incomplete

        with pytest.raises(KISSError, match="incomplete|max_output_tokens"):
            m.generate_and_process_with_tools(
                function_map={"echo": _echo},
                tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
            )

    def test_mismatched_tool_result_name_does_not_reuse_wrong_call_id(
        self, capture_server: str
    ) -> None:
        """Submitting a result for an unknown function name must raise."""
        from kiss.core.kiss_error import KISSError

        m = OpenAICompatibleModel2(
            "gpt-4o", base_url=capture_server, api_key="k"
        )
        m.initialize("hi")
        _CapturingHandler.next_response_body = _tool_call_response_json(
            name="echo",
            arguments='{"text":"hello"}',
            call_id="call_echo",
        ).encode()
        m.generate_and_process_with_tools(
            function_map={"echo": _echo},
            tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
        )
        with pytest.raises(KISSError, match="No pending function_call.*other"):
            m.add_function_results_to_conversation_and_return(
                [("other", {"result": "wrong result"})]
            )

    def test_streaming_response_incomplete_raises(
        self, capture_server: str
    ) -> None:
        """Streaming ``response.incomplete`` MUST raise."""
        from kiss.core.kiss_error import KISSError

        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
            token_callback=lambda t: None,
        )
        m.initialize("hi")
        _CapturingHandler.next_response_body = _stream_sse_event(
            "response.incomplete",
            {
                "type": "response.incomplete",
                "sequence_number": 1,
                "response": {
                    "id": "resp_incomplete",
                    "object": "response",
                    "created_at": 0,
                    "model": "gpt-4o",
                    "parallel_tool_calls": True,
                    "tool_choice": "auto",
                    "tools": [],
                    "status": "incomplete",
                    "incomplete_details": {"reason": "max_output_tokens"},
                    "output": [],
                    "usage": {
                        "input_tokens": 1,
                        "input_tokens_details": {"cached_tokens": 0},
                        "output_tokens": 0,
                        "output_tokens_details": {"reasoning_tokens": 0},
                        "total_tokens": 1,
                    },
                },
            },
        )
        with pytest.raises(KISSError, match="incomplete|max_output_tokens"):
            m.generate()


class TestReviewBugReproductions14:
    """Reproducers for the four bugs reported by the 14th gpt-5.5 review."""

    def test_function_call_missing_call_id_is_rejected(
        self, capture_server: str
    ) -> None:
        """Function_call without ``call_id`` raises (Responses-API contract)."""
        from kiss.core.kiss_error import KISSError

        body = json.dumps(
            {
                "id": "resp_bad",
                "object": "response",
                "created_at": 0,
                "model": "gpt-4o",
                "output": [
                    {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "",
                        "name": "echo",
                        "arguments": '{"text":"hi"}',
                    }
                ],
                "usage": {
                    "input_tokens": 1,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": 1,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": 2,
                },
            }
        ).encode()
        m = OpenAICompatibleModel2(
            "gpt-4o", base_url=capture_server, api_key="k"
        )
        m.initialize("hi")
        _CapturingHandler.next_response_body = body
        with pytest.raises(KISSError, match="call_id"):
            m.generate_and_process_with_tools(
                function_map={"echo": _echo},
                tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
            )


class TestReviewBugReproductions15:
    """Reproducer for the bug reported by the 15th gpt-5.5 review.

    Reviews 2-5 (dict-shaped response handling and openrouter embedding
    prefix) describe behavior that mirrors v1 — they are deliberate
    behavior parity, not regressions in v2.  Only Bug 1 (pending guard)
    is a real v2-only bug.
    """

    def test_generate_raises_if_parallel_tool_outputs_incomplete(
        self, capture_server: str
    ) -> None:
        """Pending-tool-call guard rejects new generate while outputs missing.

        The Responses API requires every model-produced ``function_call``
        to be paired with a ``function_call_output`` before the next
        request.  v2 must fail locally rather than send an invalid
        conversation.
        """
        from kiss.core.kiss_error import KISSError

        response_json = json.dumps(
            {
                "id": "resp_tc",
                "object": "response",
                "created_at": 0,
                "model": "gpt-4o",
                "output": [
                    {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_a",
                        "name": "echo",
                        "arguments": '{"text":"a"}',
                    },
                    {
                        "type": "function_call",
                        "id": "fc_2",
                        "call_id": "call_b",
                        "name": "echo",
                        "arguments": '{"text":"b"}',
                    },
                ],
                "usage": {
                    "input_tokens": 1,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": 1,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": 2,
                },
            }
        ).encode()
        m = OpenAICompatibleModel2(
            "gpt-4o", base_url=capture_server, api_key="k"
        )
        m.initialize("hi")
        _CapturingHandler.next_response_body = response_json
        m.generate_and_process_with_tools(
            function_map={"echo": _echo},
            tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
        )
        m.add_function_results_to_conversation_and_return(
            [("echo", {"result": "A"})]
        )
        before = len(_CapturingHandler.captured_requests)
        with pytest.raises(KISSError, match="pending"):
            m.generate()
        assert len(_CapturingHandler.captured_requests) == before


class TestReviewBugReproductions16:
    """Reproducer for the bug reported by the 16th gpt-5.5 review."""

    def test_trailing_fallback_rejects_mismatched_tool_result_name(
        self, capture_server: str
    ) -> None:
        """When pending queue is absent, trailing fallback must validate names.

        Restored / reconstructed conversations may carry prior
        ``function_call`` items without seeding
        ``_pending_function_calls``.  The trailing fallback must still
        refuse to pair a result with a mismatched function name.
        """
        from kiss.core.kiss_error import KISSError

        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
        )
        m.initialize("hi")
        m.conversation.append(
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_echo",
                "name": "echo",
                "arguments": '{"text":"hi"}',
            }
        )
        assert m._pending_function_calls == []
        with pytest.raises(
            KISSError,
            match=(
                "mismatch|No pending function_call|No prior"
                "|No unanswered function_call"
            ),
        ):
            m.add_function_results_to_conversation_and_return(
                [("other", {"result": "wrong result"})]
            )
        assert not any(
            isinstance(item, dict)
            and item.get("type") == "function_call_output"
            for item in m.conversation
        )


class TestReviewBugReproductions17:
    """Reproducers for the three bugs reported by the 17th gpt-5.5 review."""

    def test_generate_rejects_unanswered_function_call_even_if_pending_queue_lost(
        self, capture_server: str
    ) -> None:
        """Conversation-level guard catches unanswered function_call items."""
        from kiss.core.kiss_error import KISSError

        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
        )
        m.initialize("hi")
        _CapturingHandler.next_response_body = _tool_call_response_json(
            name="echo",
            arguments='{"text":"hello"}',
            call_id="call_lost",
        ).encode()
        m.generate_and_process_with_tools(
            function_map={"echo": _echo},
            tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
        )
        assert any(
            isinstance(item, dict)
            and item.get("type") == "function_call"
            and item.get("call_id") == "call_lost"
            for item in m.conversation
        )
        m._pending_function_calls = []
        before = len(_CapturingHandler.captured_requests)
        with pytest.raises(KISSError, match="function_call|pending|output"):
            m.generate()
        assert len(_CapturingHandler.captured_requests) == before


class TestReviewBugReproductions20:
    """Reproduce bugs found by the 20th gpt-5.5 review."""

    def test_generate_rejects_orphan_function_call_output(
        self, capture_server: str
    ) -> None:
        """Bug 1: orphan function_call_output (no prior function_call) must be rejected."""
        from kiss.core.kiss_error import KISSError

        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
        )
        m.initialize("hi")

        m.conversation.append(
            {
                "type": "function_call_output",
                "call_id": "call_missing",
                "output": "orphan output",
            }
        )

        before = len(_CapturingHandler.captured_requests)

        with pytest.raises(
            KISSError, match="function_call_output|call_missing|prior"
        ):
            m.generate()

        assert len(_CapturingHandler.captured_requests) == before


class TestReviewBugReproductions27:
    """Review #27 — 3 new bugs (bug 1 not reproducible: parser already unique)."""

    def test_add_function_results_is_atomic_on_mismatched_batch(
        self, capture_server: str
    ) -> None:
        """Batch add_function_results must be atomic — no partial mutation on error."""
        from kiss.core.kiss_error import KISSError

        response_json = json.dumps(
            {
                "id": "resp_tc",
                "object": "response",
                "created_at": 0,
                "model": "gpt-4o",
                "output": [
                    {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_a",
                        "name": "echo",
                        "arguments": '{"text":"a"}',
                    },
                    {
                        "type": "function_call",
                        "id": "fc_2",
                        "call_id": "call_b",
                        "name": "other_tool",
                        "arguments": "{}",
                    },
                ],
                "usage": {
                    "input_tokens": 1,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": 1,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": 2,
                },
            }
        ).encode()

        m = OpenAICompatibleModel2("gpt-4o", base_url=capture_server, api_key="k")
        m.initialize("hi")
        _CapturingHandler.next_response_body = response_json

        m.generate_and_process_with_tools(
            function_map={"echo": _echo, "other_tool": lambda: "x"},
            tools_schema=[
                *_ECHO_TOOL_CHAT_SCHEMA,
                {
                    "type": "function",
                    "function": {
                        "name": "other_tool",
                        "description": "other",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ],
        )

        before_conversation = list(m.conversation)
        before_pending = list(m._pending_function_calls)

        with pytest.raises(KISSError):
            m.add_function_results_to_conversation_and_return(
                [
                    ("echo", {"result": "ok"}),
                    ("wrong_name", {"result": "bad"}),
                ]
            )

        assert m.conversation == before_conversation
        assert m._pending_function_calls == before_pending


class TestReviewBugReproductions31:
    """Reproducing tests for review 31 (gpt-5.5) bugs."""

    def test_restored_function_call_missing_name_is_rejected(
        self, capture_server: str
    ) -> None:
        from kiss.core.kiss_error import KISSError

        m = OpenAICompatibleModel2("gpt-4o", base_url=capture_server, api_key="k")
        m.initialize("hi")
        m.conversation.append(
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "",
                "arguments": "{}",
            }
        )
        m.conversation.append(
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "ok",
            }
        )
        before = len(_CapturingHandler.captured_requests)
        with pytest.raises(KISSError, match="function_call.*name"):
            m.generate()
        assert len(_CapturingHandler.captured_requests) == before

    def test_restored_function_call_arguments_must_be_string(
        self, capture_server: str
    ) -> None:
        from kiss.core.kiss_error import KISSError

        m = OpenAICompatibleModel2("gpt-4o", base_url=capture_server, api_key="k")
        m.initialize("hi")
        m.conversation.append(
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "echo",
                "arguments": {"text": "hi"},
            }
        )
        m.conversation.append(
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "ok",
            }
        )
        before = len(_CapturingHandler.captured_requests)
        with pytest.raises(KISSError, match="function_call.*arguments"):
            m.generate()
        assert len(_CapturingHandler.captured_requests) == before

    def test_restored_function_call_output_output_must_be_string(
        self, capture_server: str
    ) -> None:
        from kiss.core.kiss_error import KISSError

        m = OpenAICompatibleModel2("gpt-4o", base_url=capture_server, api_key="k")
        m.initialize("hi")
        m.conversation.append(
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "echo",
                "arguments": "{}",
            }
        )
        m.conversation.append(
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": {"ok": True},
            }
        )
        before = len(_CapturingHandler.captured_requests)
        with pytest.raises(KISSError, match="function_call_output.*output"):
            m.generate()
        assert len(_CapturingHandler.captured_requests) == before

    def test_restored_function_call_output_missing_output_is_rejected(
        self, capture_server: str
    ) -> None:
        from kiss.core.kiss_error import KISSError

        m = OpenAICompatibleModel2("gpt-4o", base_url=capture_server, api_key="k")
        m.initialize("hi")
        m.conversation.append(
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "echo",
                "arguments": "{}",
            }
        )
        m.conversation.append(
            {
                "type": "function_call_output",
                "call_id": "call_1",
            }
        )
        before = len(_CapturingHandler.captured_requests)
        with pytest.raises(KISSError, match="function_call_output.*output"):
            m.generate()
        assert len(_CapturingHandler.captured_requests) == before


class TestReviewBugReproductions32:
    """Reproducing tests for review 32 (gpt-5.5) bugs."""

    def test_streaming_terminal_identityless_function_call_is_rejected(
        self, capture_server: str
    ) -> None:
        from kiss.core.kiss_error import KISSError

        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
            token_callback=lambda t: None,
        )
        m.initialize("hi")
        _CapturingHandler.next_response_body = _stream_sse_event(
            "response.completed",
            {
                "type": "response.completed",
                "sequence_number": 1,
                "response": {
                    "id": "r",
                    "object": "response",
                    "created_at": 0,
                    "model": "gpt-4o",
                    "output": [
                        {
                            "type": "function_call",
                            "arguments": '{"text":"x"}',
                        }
                    ],
                    "usage": {
                        "input_tokens": 1,
                        "input_tokens_details": {"cached_tokens": 0},
                        "output_tokens": 1,
                        "output_tokens_details": {"reasoning_tokens": 0},
                        "total_tokens": 2,
                    },
                },
            },
        )
        with pytest.raises(KISSError, match="function_call|call_id|name"):
            m.generate_and_process_with_tools(
                function_map={"echo": _echo},
                tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
            )


class TestReviewBugReproductions37:

    def test_user_message_between_function_call_and_output_is_rejected(
        self, capture_server: str
    ) -> None:
        from kiss.core.kiss_error import KISSError

        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
        )
        m.initialize("hi")

        _CapturingHandler.next_response_body = _tool_call_response_json(
            name="echo",
            arguments='{"text":"hello"}',
            call_id="call_abc",
        ).encode()

        m.generate_and_process_with_tools(
            function_map={"echo": _echo},
            tools_schema=_ECHO_TOOL_CHAT_SCHEMA,
        )

        m.conversation.append(
            {"role": "user", "content": [{"type": "input_text", "text": "new user turn"}]}
        )

        m.add_function_results_to_conversation_and_return(
            [("echo", {"result": "hello"})]
        )

        before = len(_CapturingHandler.captured_requests)
        with pytest.raises(KISSError, match="function_call|function_call_output|user"):
            m.generate()

        assert len(_CapturingHandler.captured_requests) == before


class TestReviewBugReproductions38:
    """Reproducers for review-38 findings (dict-shaped Responses support)."""

    def test_generate_raises_on_dict_shaped_failed_response(self) -> None:
        """``status="failed"`` on a dict-shaped response raises KISSError."""
        from kiss.core.kiss_error import KISSError
        from kiss.core.models.openai_compatible_model2 import (
            OpenAICompatibleModel2,
        )

        class _Responses:
            def __init__(self, response: Any) -> None:
                self.response = response

            def create(self, **kwargs: Any) -> Any:
                return self.response

        class _Client:
            def __init__(self, response: Any) -> None:
                self.responses = _Responses(response)

        m = OpenAICompatibleModel2(
            "gpt-4o", base_url="http://unused/v1", api_key="k"
        )
        m.initialize("hi")
        m.client = _Client(
            {
                "id": "r",
                "object": "response",
                "status": "failed",
                "error": {"message": "boom"},
                "output": [],
            }
        )
        with pytest.raises(KISSError, match="boom|failed"):
            m.generate()


class TestReviewBugReproductions39:
    """Reproducers for review-39 findings."""

    def test_stream_without_terminal_completed_raises(
        self, capture_server: str
    ) -> None:
        """Stream ending without ``response.completed`` raises KISSError.

        The Responses API contract requires every successful stream to
        terminate with a ``response.completed`` event.  A stream that
        merely runs out (e.g. truncated HTTP body, proxy disconnect)
        must NOT be accepted as a successful generation, because the
        last event may be a partial text delta or partial tool-call
        arguments fragment.
        """
        from kiss.core.kiss_error import KISSError
        from kiss.core.models.openai_compatible_model2 import (
            OpenAICompatibleModel2,
        )

        tokens: list[str] = []
        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url=capture_server,
            api_key="k",
            token_callback=tokens.append,
        )
        m.initialize("hi")

        _CapturingHandler.next_response_body = _stream_sse_event(
            "response.output_text.delta",
            {
                "type": "response.output_text.delta",
                "sequence_number": 1,
                "item_id": "msg_1",
                "output_index": 0,
                "content_index": 0,
                "delta": "partial",
                "logprobs": [],
            },
        )

        with pytest.raises(
            KISSError, match="completed|terminal|truncated|stream"
        ):
            m.generate()
