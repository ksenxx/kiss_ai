# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# ruff: noqa: F811  (the `anthropic_server` module fixture is imported from
# the core/models twin of this file and injected via test parameters)
"""Regression test: ``AnthropicModel`` must enable extended (adaptive)
thinking for ``claude-opus-5``.

The bug this test locks in:

    "When I use claude-opus-5 as the model for a task, the thinking
     tokens are not shown."

Root cause — ``claude-opus-5`` fell into the exact gap that previously
bit ``claude-fable-5`` (see test_anthropic_fable_thinking.py):

1. The ``MODEL_INFO.json`` entry for ``claude-opus-5`` carried no
   ``extended_thinking`` / ``adaptive_thinking`` flags, so the tri-state
   override in ``_supports_extended_thinking`` / ``_uses_adaptive_thinking``
   did not apply.
2. The legacy fallback heuristics only matched Claude *4.x* names:
   ``startswith(("claude-opus-4", "claude-sonnet-4", "claude-haiku-4"))``
   for extended thinking and the ``claude-opus-4-<minor>=6`` prefix rule
   for adaptive thinking.  ``claude-opus-5`` matched neither.

Consequences in ``AnthropicModel._build_create_kwargs``:

* no ``thinking`` request param was sent at all, so the API returned no
  thinking blocks / no ``thinking_delta`` SSE events and the thoughts
  panel stayed empty;
* the ``anthropic-beta: interleaved-thinking-2025-05-14`` extra header
  was never attached;
* ``tool_choice={"type": "any"}`` was force-set whenever tools were
  present, which by itself silently disables thinking on the API side;
* the ``max_tokens`` default was the plain 16384: the 64000-vs-65536
  selection lived inside the ``if _supports_extended_thinking(...)``
  block that opus-5 never entered, and the branch's own
  ``startswith("claude-opus-4")`` family test was version-blind anyway
  (it is now version-aware).

The fix (this test verifies every half):

1. ``MODEL_INFO.json`` now sets ``extended_thinking`` and
   ``adaptive_thinking`` to ``true`` for ``claude-opus-5``.
2. The fallback heuristics are version-aware: any
   ``claude-<family>-<major>`` name with family in
   opus/sonnet/haiku/fable and major >= 5 gets extended + adaptive
   thinking (the established Claude 4.x rules are preserved verbatim),
   so future families cannot regress the same way.
3. The ``max_tokens`` default is version-aware (Opus family -> 65536).

Includes a REAL end-to-end streaming test: a local
``ThreadingHTTPServer`` returns Anthropic-format SSE with a
``thinking_delta`` stream for ``claude-opus-5`` and the test asserts the
thinking text actually reaches the UI (``JsonPrinter`` events) and the
raw thinking callback.  No mocks, patches, or fakes of production code.
"""

from __future__ import annotations

import anthropic

import kiss.tests.core.models.test_anthropic_opus5_thinking as _models_twin
from kiss.core.models.anthropic_model import (
    AnthropicModel,
)
from kiss.server.json_printer import JsonPrinter
from kiss.tests.core.models.test_anthropic_opus5_thinking import (  # noqa: F401
    _LAST_REQUEST,
    _AnthropicOpus5Handler,
    _opus5_thinking_events,
    _strip_thinking_events,
    anthropic_server,
)


def _build_opus_5_model(server_url: str, printer: JsonPrinter) -> AnthropicModel:
    """Return an AnthropicModel for claude-opus-5 wired to the local server."""
    m = AnthropicModel(
        "claude-opus-5",
        api_key="test-key",
        token_callback=printer.token_callback,
        thinking_callback=printer.thinking_callback,
    )
    m.client = anthropic.Anthropic(api_key="test-key", base_url=server_url)
    m.conversation = [{"role": "user", "content": "What is 2+2?"}]
    return m


class TestOpus5EndToEndThinkingStream:
    """Real SSE streaming through a local server: thinking must reach the UI."""

    def test_thinking_text_reaches_ui_events(self, anthropic_server: str) -> None:
        """End-to-end bug reproduction: the thoughts panel must receive
        ``thinking_start`` / ``thinking_delta`` / ``thinking_end`` events
        carrying the streamed thinking text for claude-opus-5."""
        _models_twin._RESPONSE_EVENTS = _opus5_thinking_events()

        printer = JsonPrinter()
        printer._thread_local.task_id = "test-opus5-thinking"
        printer.start_recording()
        m = _build_opus_5_model(anthropic_server, printer)
        m._create_message(m._build_create_kwargs())

        recorded = printer.stop_recording()
        types = [e["type"] for e in recorded]

        assert types.count("thinking_start") == 1, types
        assert types.count("thinking_end") == 1, types

        start_idx = types.index("thinking_start")
        end_idx = types.index("thinking_end")
        assert start_idx < end_idx, types
        thinking_deltas = [
            e for e in recorded[start_idx + 1 : end_idx] if e["type"] == "thinking_delta"
        ]
        thought = "".join(d["text"] for d in thinking_deltas)
        assert thought == "Pondering the request.", thought

        text_deltas = [e for e in recorded if e["type"] == "text_delta"]
        text = "".join(d.get("text", "") for d in text_deltas)
        assert text == "Four.", text

    def test_request_carries_thinking_param_and_beta_header(
        self, anthropic_server: str
    ) -> None:
        """The actual HTTP request for opus-5 must carry the adaptive
        thinking param and the interleaved-thinking beta header."""
        _models_twin._RESPONSE_EVENTS = _opus5_thinking_events()

        printer = JsonPrinter()
        printer._thread_local.task_id = "test-opus5-request"
        printer.start_recording()
        m = _build_opus_5_model(anthropic_server, printer)
        m._create_message(m._build_create_kwargs())
        printer.stop_recording()

        payload = _LAST_REQUEST.get("json")
        assert isinstance(payload, dict), _LAST_REQUEST
        assert payload.get("thinking") == {
            "type": "adaptive",
            "display": "summarized",
        }, payload.get("thinking")
        assert payload.get("max_tokens") == 65536, payload.get("max_tokens")
        beta = _LAST_REQUEST.get("anthropic-beta", "")
        assert isinstance(beta, str)
        assert "interleaved-thinking-2025-05-14" in beta, beta
