# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# ruff: noqa: F811  (the `fable_server` module fixture is imported from
# the core/models twin of this file and injected via test parameters)
"""Regression tests: claude-fable-5 must actually reveal thinking tokens.

The user-reported bug this file locks in:

    "why are you not revealing thinking tokens, when you run a task
     using claude-fable-5?"

Two independent bugs in ``AnthropicModel._build_create_kwargs`` combined
to hide fable-5 thinking on every agentic turn:

1. **Missing ``display: "summarized"``.**  KISS sent
   ``thinking={"type": "adaptive"}``.  On Claude Fable 5 (and Mythos 5,
   Sonnet 5, Opus 4.8, Opus 4.7) ``thinking.display`` defaults to
   ``"omitted"``: the API returns thinking blocks with an EMPTY
   ``thinking`` field (encrypted signature only) and emits NO
   ``thinking_delta`` stream events, so ``thinking_callback`` never
   fires with any text.  Per the Anthropic docs
   (platform.claude.com/docs/en/build-with-claude/adaptive-thinking) the
   client must explicitly request
   ``thinking={"type": "adaptive", "display": "summarized"}``.

2. **Forced ``tool_choice={"type": "any"}``.**  For adaptive-thinking
   models with tools (KISSAgent always passes tools), KISS forced
   ``tool_choice={"type": "any"}``.  Tool use with thinking only
   supports ``tool_choice`` ``auto``/``none``; forcing tool use makes
   the API silently disable thinking for the request ("graceful
   thinking degradation"), so no thinking blocks are produced at all —
   verified against the live Anthropic API (``any`` → only
   ``tool_use`` blocks; ``auto`` → ``thinking`` + ``text`` +
   ``tool_use``).

Test strategy (no mocks, patches, or fakes):

* kwargs-level tests assert the wire request now carries
  ``display: "summarized"`` and no forced ``tool_choice``;
* an end-to-end SSE test drives a real ``anthropic`` client against a
  local ``ThreadingHTTPServer`` that (a) captures the exact JSON request
  body and (b) replays the adaptive-thinking stream shape the live API
  produces once the fix is in place, asserting that thinking text flows
  to ``thinking_callback``/``token_callback`` and to the ``JsonPrinter``
  event stream.
"""

from __future__ import annotations

import anthropic

from kiss.core.models.anthropic_model import AnthropicModel
from kiss.server.json_printer import JsonPrinter
from kiss.tests.core.models.test_anthropic_fable_thinking_display import (  # noqa: F401
    _CAPTURED_REQUESTS,
    _OPENAI_FINISH_TOOL,
    _fable_thinking_tool_events,
    _FableHandler,
    fable_server,
)


class TestFable5ThinkingEndToEnd:
    """End-to-end: fable-5 agentic turn must surface thinking tokens."""

    def test_agentic_turn_streams_thinking_to_json_printer(
        self, fable_server: str
    ) -> None:
        """The JsonPrinter event stream must contain thinking_start,
        thinking_delta text, and thinking_end for a fable-5 agentic turn."""
        _CAPTURED_REQUESTS.clear()
        printer = JsonPrinter()
        printer._thread_local.task_id = "test-fable-thinking"
        printer.start_recording()

        m = AnthropicModel(
            "claude-fable-5",
            api_key="test-key",
            token_callback=printer.token_callback,
            thinking_callback=printer.thinking_callback,
        )
        m.client = anthropic.Anthropic(api_key="test-key", base_url=fable_server)
        m.conversation = [{"role": "user", "content": "What is 27*31? Then finish."}]

        m.generate_and_process_with_tools({}, tools_schema=[_OPENAI_FINISH_TOOL])

        recorded = printer.stop_recording()
        types = [e["type"] for e in recorded]
        assert types.count("thinking_start") == 1, types
        assert types.count("thinking_end") == 1, types

        start_idx = types.index("thinking_start")
        end_idx = types.index("thinking_end")
        thought = "".join(
            e["text"]
            for e in recorded[start_idx + 1 : end_idx]
            if e["type"] == "thinking_delta"
        )
        assert thought == "I'm calculating 27 times 31.", thought
