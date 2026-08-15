# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""What the daemon broadcasts around a tool call.

Originally ``JsonPrinter`` only rendered the return value of a handful of
"core" tools (Bash, Read, Edit, Write) and swallowed the rest, so a
screenshot or a go_to_url left a tool call with nothing after it.  Commit
a8d394a7 removed that whitelist: every tool's return value is broadcast
except ``finish``, whose value the agentic loop renders as its own
``result`` panel.  These tests pin that, and the event ORDER the webview
is entitled to rely on -- a tool's result reaching the transcript before
the next thinking block starts.

The webview's own half of this story -- that a thought following a tool
call always lands inside a Thoughts panel, even when its result never
arrives -- is covered by the real DOM tests in
``agents/vscode/test/thoughtsPanelArming.test.js``.
"""

from __future__ import annotations


def test_noncore_tool_result_is_broadcast() -> None:
    """Verify that tool_result for non-core tools IS broadcast.

    Earlier the printer suppressed ``tool_result`` for tools outside
    the {Bash, Read, Edit, Write} whitelist.  Commit a8d394a7 removed
    that whitelist — every tool's return value is now rendered except
    ``finish`` (the agentic loop renders ``finish`` as a dedicated
    ``result`` panel).  This test pins the new behaviour so any future
    "suppress non-core tools" regression fails loudly.
    """
    from kiss.server.json_printer import JsonPrinter

    printer = JsonPrinter()
    printer._thread_local.task_id = "t1"
    printer.start_recording()

    printer.print("screenshot", type="tool_call", tool_input={})
    printer.print("image data...", type="tool_result", tool_name="screenshot")

    events = printer.stop_recording()
    types = [e["type"] for e in events]

    assert "tool_call" in types, "tool_call event must be broadcast"
    assert "tool_result" in types, (
        "tool_result for non-core tool 'screenshot' must be broadcast "
        "(the core_tools whitelist was removed in commit a8d394a7)."
    )


def test_core_tool_result_is_broadcast() -> None:
    """Verify that tool_result for core tools IS broadcast."""
    from kiss.server.json_printer import JsonPrinter

    printer = JsonPrinter()
    printer._thread_local.task_id = "t1"
    printer.start_recording()

    printer.print("Read", type="tool_call", tool_input={"file_path": "test.py"})
    printer.print("file contents...", type="tool_result", tool_name="Read")

    events = printer.stop_recording()
    types = [e["type"] for e in events]

    assert "tool_result" in types, (
        "tool_result for core tool 'Read' must be broadcast"
    )



def test_thinking_after_noncore_tool_gets_panel_events() -> None:
    """Simulate a non-core tool turn and verify the post-a8d394a7 event order.

    After commit a8d394a7 the printer broadcasts ``tool_result`` for
    every non-``finish`` tool, so the screenshot tool_call is followed
    by its tool_result *before* the next thinking_start.  The frontend
    still creates a Thoughts panel for the thinking_start because
    ``pendingPanel`` is set ``true`` on the tool_call itself (the
    invariant pinned by the tests above).
    """
    from kiss.server.json_printer import JsonPrinter

    printer = JsonPrinter()
    printer._thread_local.task_id = "t1"
    printer.start_recording()

    printer.thinking_callback(True)
    printer.token_callback("Let me read the file")
    printer.thinking_callback(False)
    printer.print("Read", type="tool_call", tool_input={"file_path": "test.html"})
    printer.print("file contents", type="tool_result", tool_name="Read")
    printer.print("screenshot", type="tool_call", tool_input={})
    printer.print("screenshot taken", type="tool_result", tool_name="screenshot")

    printer.thinking_callback(True)
    printer.token_callback("I see the issue, need to add SVG")
    printer.thinking_callback(False)
    printer.token_callback("I'll fix the HTML now")
    printer.print("Write", type="tool_call", tool_input={"file_path": "test.html"})
    printer.print("ok", type="tool_result", tool_name="Write")

    events = printer.stop_recording()
    types = [e["type"] for e in events]

    screenshot_idx = None
    for i, e in enumerate(events):
        if e["type"] == "tool_call" and e.get("name") == "screenshot":
            screenshot_idx = i
            break
    assert screenshot_idx is not None

    post_screenshot = types[screenshot_idx + 1 :]
    thinking_start_offset = post_screenshot.index("thinking_start")
    between = post_screenshot[:thinking_start_offset]
    assert "tool_result" in between, (
        "tool_result for the screenshot tool must be broadcast between "
        f"its tool_call and the next thinking_start, got: {between}"
    )



