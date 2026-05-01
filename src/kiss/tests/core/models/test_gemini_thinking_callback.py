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

from kiss.agents.vscode.browser_ui import BaseBrowserPrinter
from kiss.core.models.gemini_model import GeminiModel


class TestGeminiStreamPartsThinkingCallback:
    """Verify _stream_parts invokes thinking_callback for thought parts."""

    def test_thinking_callback_fires_for_thought_parts(self) -> None:
        """thinking_callback must receive True then False around thinking parts."""
        tokens: list[str] = []
        thinking_events: list[bool] = []

        m = GeminiModel(
            "gemini-2.5-flash",
            api_key="test-key",
            token_callback=lambda t: tokens.append(t),
            thinking_callback=lambda s: thinking_events.append(s),
        )

        # Simulate a chunk with a thinking part followed by a text part
        thinking_part = types.Part(text="Let me think about this.", thought=True)
        text_part = types.Part(text="The answer is 42.")

        m._stream_parts([thinking_part, text_part])

        assert True in thinking_events, (
            "thinking_callback(True) was never called — "
            "thinking tokens leak as text_delta events"
        )
        assert False in thinking_events, (
            "thinking_callback(False) was never called — "
            "thinking panel will never close"
        )
        first_true = thinking_events.index(True)
        last_false = len(thinking_events) - 1 - thinking_events[::-1].index(False)
        assert first_true < last_false

        combined = "".join(tokens)
        assert "Let me think" in combined
        assert "The answer is 42." in combined

    def test_thinking_across_multiple_chunks(self) -> None:
        """Thinking state must carry across multiple _stream_parts calls."""
        tokens: list[str] = []
        thinking_events: list[bool] = []

        m = GeminiModel(
            "gemini-2.5-flash",
            api_key="test-key",
            token_callback=lambda t: tokens.append(t),
            thinking_callback=lambda s: thinking_events.append(s),
        )

        # Chunk 1: thinking part
        m._stream_parts([types.Part(text="Thinking chunk 1", thought=True)])
        # Chunk 2: more thinking
        m._stream_parts([types.Part(text=" and chunk 2", thought=True)])
        # Chunk 3: regular text — should close thinking and start text
        m._stream_parts([types.Part(text="Final answer.")])

        # Should have: True (start thinking), False (end thinking when text starts)
        assert thinking_events[0] is True
        # Thinking should have been closed before text
        assert False in thinking_events

        combined = "".join(tokens)
        assert "Thinking chunk 1" in combined
        assert "and chunk 2" in combined
        assert "Final answer." in combined

    def test_no_thinking_callback_when_no_thought_parts(self) -> None:
        """thinking_callback must NOT fire when there are no thought parts."""
        thinking_events: list[bool] = []

        m = GeminiModel(
            "gemini-2.5-flash",
            api_key="test-key",
            token_callback=lambda t: None,
            thinking_callback=lambda s: thinking_events.append(s),
        )

        m._stream_parts([types.Part(text="Just regular text.")])

        assert thinking_events == [], (
            f"thinking_callback fired unexpectedly: {thinking_events}"
        )

    def test_browser_printer_routes_thinking_tokens_correctly(self) -> None:
        """Thinking tokens must be broadcast as thinking_delta, not text_delta.

        This is the core bug reproduction: without thinking_callback, the
        BaseBrowserPrinter never sets _current_block_type to 'thinking',
        so thinking tokens are broadcast as text_delta events.
        """
        printer = BaseBrowserPrinter()
        printer.start_recording()

        m = GeminiModel(
            "gemini-2.5-flash",
            api_key="test-key",
            token_callback=printer.token_callback,
            thinking_callback=printer.thinking_callback,
        )

        # Simulate streaming: thinking part then text part
        m._stream_parts([types.Part(text="Deep reasoning here.", thought=True)])
        m._stream_parts([types.Part(text="The result is X.")])

        recorded = printer.stop_recording()
        event_types = [e["type"] for e in recorded]

        # Must have thinking_start / thinking_end events
        assert "thinking_start" in event_types, (
            f"No thinking_start — types: {event_types}"
        )
        assert "thinking_end" in event_types, (
            f"No thinking_end — types: {event_types}"
        )

        # Thinking tokens must be thinking_delta, not text_delta
        start_idx = event_types.index("thinking_start")
        end_idx = event_types.index("thinking_end")
        between = recorded[start_idx + 1 : end_idx]
        thinking_deltas = [e for e in between if e["type"] == "thinking_delta"]
        assert thinking_deltas, (
            "No thinking_delta events between thinking_start/end — "
            "thinking tokens leaked as text_delta"
        )

        # Verify the thinking text content
        thought_text = "".join(d["text"] for d in thinking_deltas)
        assert "Deep reasoning here." in thought_text

        # No thinking content should be in text_delta events
        text_deltas = [e for e in recorded if e["type"] == "text_delta"]
        text_content = "".join(d.get("text", "") for d in text_deltas)
        assert "Deep reasoning here." not in text_content, (
            f"Thinking text leaked into text_delta: {text_content}"
        )
        assert "The result is X." in text_content

    def test_end_thinking_stream_closes_open_block(self) -> None:
        """_end_thinking_stream must close an open thinking block.

        After a streaming loop ends, if the last chunk was a thinking part,
        the thinking block must still be closed.
        """
        thinking_events: list[bool] = []

        m = GeminiModel(
            "gemini-2.5-flash",
            api_key="test-key",
            token_callback=lambda t: None,
            thinking_callback=lambda s: thinking_events.append(s),
        )

        # Stream only thinking parts (no text follows)
        m._stream_parts([types.Part(text="Only thinking.", thought=True)])
        # Must close the block explicitly
        m._end_thinking_stream()

        assert thinking_events == [True, False], (
            f"Expected [True, False] but got {thinking_events}"
        )

    def test_end_thinking_stream_noop_when_not_thinking(self) -> None:
        """_end_thinking_stream must be a no-op when no thinking block is open."""
        thinking_events: list[bool] = []

        m = GeminiModel(
            "gemini-2.5-flash",
            api_key="test-key",
            token_callback=lambda t: None,
            thinking_callback=lambda s: thinking_events.append(s),
        )

        m._stream_parts([types.Part(text="Regular text.")])
        m._end_thinking_stream()

        assert thinking_events == [], (
            f"thinking_callback fired unexpectedly: {thinking_events}"
        )
