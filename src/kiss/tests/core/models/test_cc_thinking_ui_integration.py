"""Integration test: cc/* model + BaseBrowserPrinter must emit streaming
thinking events (``thinking_start`` → ``thinking_delta`` → ``thinking_end``)
exactly once — no duplicate collapse from redundant ``assistant`` snapshots.

Reproduces the UI bug: with ``--include-partial-messages`` the Claude CLI
emits granular ``content_block_*`` events AND a redundant final
``assistant`` event with the same content.  When the parser processes both,
the browser UI receives two ``thinking_end`` events, causing the thoughts
panel to collapse into a "Thinking (click to expand)" bar and the streamed
tokens to be hidden.
"""

import json

from kiss.agents.vscode.browser_ui import BaseBrowserPrinter
from kiss.core.models.claude_code_model import ClaudeCodeModel


def test_cc_model_streams_thinking_tokens_to_browser_ui() -> None:
    """Thinking tokens must stream as ``thinking_delta`` events with exactly
    one ``thinking_start`` / ``thinking_end`` pair — never double-collapsed.
    """
    printer = BaseBrowserPrinter()
    printer.start_recording()

    model = ClaudeCodeModel(
        "cc/haiku",
        token_callback=printer.token_callback,
        thinking_callback=printer.thinking_callback,
    )
    model.initialize("test")

    events = [
        {"type": "stream_event", "event": {
            "type": "content_block_start",
            "content_block": {"type": "thinking", "thinking": ""}}},
        {"type": "stream_event", "event": {
            "type": "content_block_delta",
            "delta": {"type": "thinking_delta", "thinking": "Let me "}}},
        {"type": "stream_event", "event": {
            "type": "content_block_delta",
            "delta": {"type": "thinking_delta", "thinking": "reason..."}}},
        {"type": "stream_event", "event": {"type": "content_block_stop"}},
        {"type": "stream_event", "event": {
            "type": "content_block_start",
            "content_block": {"type": "text", "text": ""}}},
        {"type": "stream_event", "event": {
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": "Answer"}}},
        {"type": "stream_event", "event": {"type": "content_block_stop"}},
        {"type": "assistant", "message": {
            "id": "msg_abc",
            "content": [
                {"type": "thinking", "thinking": "Let me reason..."},
                {"type": "text", "text": "Answer"},
            ]}},
        {"type": "result", "result": "Answer", "usage": {}},
    ]
    model._parse_stream_events(iter(json.dumps(e) for e in events))

    recorded = printer.stop_recording()
    types = [e["type"] for e in recorded]

    assert types.count("thinking_start") == 1, types
    assert types.count("thinking_end") == 1, types

    start_idx = types.index("thinking_start")
    end_idx = types.index("thinking_end")
    thinking_deltas = [
        e for e in recorded[start_idx + 1 : end_idx]
        if e["type"] == "thinking_delta"
    ]
    assert thinking_deltas, "No thinking_delta events recorded — tokens not streamed"
    full_thought = "".join(d["text"] for d in thinking_deltas)
    assert full_thought == "Let me reason...", full_thought

    assert end_idx < types.index("text_delta")


def test_cc_model_no_thinking_end_before_thinking_deltas() -> None:
    """No ``thinking_end`` may appear before all ``thinking_delta`` events —
    a premature end would collapse the panel and hide subsequent tokens.
    """
    printer = BaseBrowserPrinter()
    printer.start_recording()

    model = ClaudeCodeModel(
        "cc/haiku",
        token_callback=printer.token_callback,
        thinking_callback=printer.thinking_callback,
    )
    model.initialize("test")

    events = [
        {"type": "stream_event", "event": {
            "type": "content_block_start",
            "content_block": {"type": "thinking"}}},
        {"type": "stream_event", "event": {
            "type": "content_block_delta",
            "delta": {"type": "thinking_delta", "thinking": "one"}}},
        {"type": "assistant", "message": {
            "id": "m1",
            "content": [{"type": "thinking", "thinking": "one"}]}},
        {"type": "stream_event", "event": {
            "type": "content_block_delta",
            "delta": {"type": "thinking_delta", "thinking": "two"}}},
        {"type": "stream_event", "event": {"type": "content_block_stop"}},
        {"type": "result", "result": "", "usage": {}},
    ]
    model._parse_stream_events(iter(json.dumps(e) for e in events))

    recorded = printer.stop_recording()
    types = [e["type"] for e in recorded]

    assert types.count("thinking_start") == 1, types
    assert types.count("thinking_end") == 1, types

    start_idx = types.index("thinking_start")
    end_idx = types.index("thinking_end")
    for i, t in enumerate(types):
        if t == "thinking_delta":
            assert start_idx < i < end_idx, (
                f"thinking_delta at {i} outside [{start_idx}, {end_idx}]: {types}"
            )
