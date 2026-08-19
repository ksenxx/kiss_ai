# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration test: GeminiModel must invoke thinking_callback.

Gemini models with thinking enabled return parts where ``part.thought``
is ``True`` for thinking content.  The ``thinking_callback`` must be
invoked with ``True`` at the start of a thinking block and ``False`` at
the end so that the browser UI routes thinking tokens to the thinking
panel rather than the main text area.

Bug reproduction: without the fix, ``_stream_parts()`` calls
``_invoke_token_callback()`` for all parts without checking
``part.thought``, so thinking tokens are broadcast as ``text_delta``
events — thoughts appear outside the thinking panel.

Uses real ``google.genai.types.Part`` objects — no mocks, patches, or
fakes.
"""

from __future__ import annotations

from google.genai import types

from kiss.core.models.gemini_model import GeminiModel
from kiss.server.json_printer import JsonPrinter


class TestGeminiStreamPartsThinkingCallback:
    """Verify _stream_parts invokes thinking_callback for thought parts."""

    def test_browser_printer_routes_thinking_tokens_correctly(self) -> None:
        """Thinking tokens must be broadcast as thinking_delta, not text_delta.

        This is the core bug reproduction: without thinking_callback, the
        JsonPrinter never sets _current_block_type to 'thinking',
        so thinking tokens are broadcast as text_delta events.
        """
        printer = JsonPrinter()
        printer._thread_local.task_id = "gemini-thinking-test"
        printer.start_recording()

        m = GeminiModel(
            "gemini-2.5-flash",
            api_key="test-key",
            token_callback=printer.token_callback,
            thinking_callback=printer.thinking_callback,
        )

        m._stream_parts([types.Part(text="Deep reasoning here.", thought=True)])
        m._stream_parts([types.Part(text="The result is X.")])

        recorded = printer.stop_recording()
        event_types = [e["type"] for e in recorded]

        assert "thinking_start" in event_types, (
            f"No thinking_start — types: {event_types}"
        )
        assert "thinking_end" in event_types, (
            f"No thinking_end — types: {event_types}"
        )

        start_idx = event_types.index("thinking_start")
        end_idx = event_types.index("thinking_end")
        between = recorded[start_idx + 1 : end_idx]
        thinking_deltas = [e for e in between if e["type"] == "thinking_delta"]
        assert thinking_deltas, (
            "No thinking_delta events between thinking_start/end — "
            "thinking tokens leaked as text_delta"
        )

        thought_text = "".join(d["text"] for d in thinking_deltas)
        assert "Deep reasoning here." in thought_text

        text_deltas = [e for e in recorded if e["type"] == "text_delta"]
        text_content = "".join(d.get("text", "") for d in text_deltas)
        assert "Deep reasoning here." not in text_content, (
            f"Thinking text leaked into text_delta: {text_content}"
        )
        assert "The result is X." in text_content
