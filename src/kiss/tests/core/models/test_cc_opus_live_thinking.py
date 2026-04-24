"""Live integration test: cc/opus model must not show empty "Thinking" bar.

This test invokes the real ``claude`` CLI with ``cc/opus`` and verifies that:
1. No empty thinking_start/thinking_end pair is emitted (i.e., thinking_start
   without any thinking_delta in between).
2. If thinking tokens ARE returned, they stream correctly.
3. Text content is still delivered.

Requires the ``claude`` CLI to be installed and authenticated.
"""

import shutil

import pytest

from kiss.agents.vscode.browser_ui import BaseBrowserPrinter
from kiss.core.models.claude_code_model import ClaudeCodeModel


@pytest.mark.skipif(
    shutil.which("claude") is None,
    reason="claude CLI not installed",
)
class TestCCOpusLiveThinking:
    """Live test against the real claude CLI to verify thinking block behaviour."""

    def test_no_empty_thinking_bar(self) -> None:
        """cc/opus must not produce thinking_start without thinking_delta content.

        This reproduces the original bug: Claude opus sends opaque thinking
        blocks with only signature_delta (no readable thinking text).  The
        old code emitted thinking_start/thinking_end for these, producing
        an empty collapsible "Thinking" bar in the UI.
        """
        printer = BaseBrowserPrinter()
        printer.start_recording()

        model = ClaudeCodeModel(
            "cc/opus",
            token_callback=printer.token_callback,
            thinking_callback=printer.thinking_callback,
        )
        model.initialize("What is 2+2? Reply with just the number.")

        content, _ = model.generate()

        recorded = printer.stop_recording()

        thinking_depth = 0
        thinking_had_content = False
        for event in recorded:
            if event["type"] == "thinking_start":
                thinking_depth += 1
                thinking_had_content = False
            elif event["type"] == "thinking_delta":
                if event.get("text", ""):
                    thinking_had_content = True
            elif event["type"] == "thinking_end":
                assert thinking_had_content, (
                    "thinking_start was emitted but no thinking_delta with content "
                    "was seen before thinking_end — this would show an empty "
                    "'Thinking' bar in the UI"
                )
                thinking_depth -= 1

        assert thinking_depth == 0, "Unbalanced thinking blocks"

        assert content.strip(), f"Expected non-empty response, got: {content!r}"

        text_deltas = [e for e in recorded if e["type"] == "text_delta"]
        assert text_deltas, "No text_delta events were recorded"
