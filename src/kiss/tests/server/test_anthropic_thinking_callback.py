# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# ruff: noqa: F811  (the `anthropic_server` module fixture is imported from
# the core/models twin of this file and injected via test parameters)
"""Integration test: AnthropicModel must invoke thinking_callback.

Claude models with extended thinking send ``thinking_delta`` events during
streaming.  The ``thinking_callback`` must be invoked with ``True`` at the
start of a thinking block and ``False`` at the end so that the browser UI
routes thinking tokens to the thinking panel rather than the main text area.

Bug reproduction: without the fix, thinking tokens arrive at the
``JsonPrinter.token_callback`` while ``_current_block_type`` is
still ``""`` (not ``"thinking"``), causing them to be broadcast as
``text_delta`` events — thoughts appear outside the thinking panel.

Uses a real ThreadingHTTPServer that returns Anthropic-format SSE — no
mocks, patches, or fakes.
"""

from __future__ import annotations

from kiss.core.models.anthropic_model import AnthropicModel
from kiss.server.json_printer import JsonPrinter
from kiss.tests.core.models.test_anthropic_thinking_callback import (  # noqa: F401
    _anthropic_sse_events,
    _AnthropicHandler,
    anthropic_server,
)


class TestAnthropicThinkingCallback:
    """Verify that AnthropicModel invokes thinking_callback for thinking blocks."""

    def test_browser_printer_routes_thinking_tokens_correctly(
        self, anthropic_server: str
    ) -> None:
        """Thinking tokens must be broadcast as thinking_delta, not text_delta.

        This is the core bug reproduction: without thinking_callback, the
        JsonPrinter never sets _current_block_type to 'thinking', so
        thinking tokens are broadcast as text_delta events — thoughts appear
        outside the thinking panel.
        """
        printer = JsonPrinter()
        printer._thread_local.task_id = "test-task"
        printer.start_recording()

        m = AnthropicModel(
            "claude-sonnet-4-20250514",
            api_key="test-key",
            token_callback=printer.token_callback,
            thinking_callback=printer.thinking_callback,
        )
        m.client = __import__("anthropic").Anthropic(
            api_key="test-key", base_url=anthropic_server
        )
        m.conversation = [{"role": "user", "content": "Think about this."}]

        kwargs = m._build_create_kwargs()
        m._create_message(kwargs)

        recorded = printer.stop_recording()
        types = [e["type"] for e in recorded]

        assert "thinking_start" in types, f"No thinking_start — types: {types}"
        assert "thinking_end" in types, f"No thinking_end — types: {types}"

        start_idx = types.index("thinking_start")
        end_idx = types.index("thinking_end")
        between = recorded[start_idx + 1 : end_idx]
        thinking_deltas = [e for e in between if e["type"] == "thinking_delta"]
        assert thinking_deltas, (
            "No thinking_delta events between thinking_start/end — "
            "thinking tokens leaked as text_delta"
        )

        thought_text = "".join(d["text"] for d in thinking_deltas)
        assert "Let me think" in thought_text

        text_deltas = [e for e in recorded if e["type"] == "text_delta"]
        text_content = "".join(d.get("text", "") for d in text_deltas)
        assert "Let me think" not in text_content, (
            f"Thinking text leaked into text_delta: {text_content}"
        )
