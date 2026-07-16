# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests: thinking/text after a tool_call must go into Thoughts panels.

Originally a non-core ``tool_result`` (screenshot, go_to_url, scroll,
click, …) was suppressed by ``JsonPrinter`` and the frontend's
``pendingPanel`` flag stayed ``false`` after such a tool_call — the next
thinking/text block then bypassed Thoughts-panel creation.

The backend now broadcasts ``tool_result`` for every tool *except*
``finish`` (commit a8d394a7), so the suppression branch is gone.  The
frontend fix — set ``pendingPanel = true`` on ``tool_call`` itself — is
still required as a defensive invariant: it guarantees a Thoughts
panel for the next thinking/text block regardless of how the printer
decides to render the tool's return value.  These tests pin that
invariant in:
  1. ``processOutputEvent``
  2. ``processOutputEventForBgTab``
  3. ``replayEventsInto``
  4. Step-counting loops (bg tab task_events replay + replayTaskEvents)
"""

from __future__ import annotations

import re
from pathlib import Path

MAIN_JS = (
    Path(__file__).resolve().parents[3]
    / "agents"
    / "vscode"
    / "media"
    / "main.js"
)

BROWSER_UI = (
    Path(__file__).resolve().parents[3]
    / "server"
    / "json_printer.py"
)


def _read_main_js() -> str:
    assert MAIN_JS.is_file(), f"main.js not found at {MAIN_JS}"
    return MAIN_JS.read_text()


def _extract_function_body(src: str, name: str) -> str:
    """Extract the full body of function *name* from JavaScript source."""
    pattern = re.compile(rf"function {re.escape(name)}\s*\([^)]*\)\s*\{{")
    m = pattern.search(src)
    assert m, f"function {name} not found in main.js"
    start = m.end() - 1
    depth = 0
    i = start
    while i < len(src):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return src[start : i + 1]
        i += 1
    raise AssertionError(f"unmatched braces in function {name}")


# ── Frontend code-analysis tests ──────────────────────────────────────────


def test_process_output_event_tool_call_sets_pending_panel_true() -> None:
    """processOutputEvent must set pendingPanel = true on tool_call.

    Without this, non-core tools (screenshot, go_to_url, scroll) whose
    tool_result is suppressed leave pendingPanel = false, causing the
    next thinking/text to render outside a Thoughts panel.
    """
    src = _read_main_js()
    body = _extract_function_body(src, "processOutputEvent")

    # Find the tool_call handler block:
    #   if (t === 'tool_call') { ... pendingPanel = true; ... }
    m = re.search(
        r"t\s*===\s*'tool_call'[^}]*?pendingPanel\s*=\s*(\w+)",
        body,
    )
    assert m, (
        "processOutputEvent must set pendingPanel on tool_call events"
    )
    assert m.group(1) == "true", (
        f"processOutputEvent sets pendingPanel = {m.group(1)} on tool_call; "
        "must be true so that non-core tools (screenshot, go_to_url, scroll) "
        "whose tool_result is suppressed still trigger Thoughts panel creation "
        "for the subsequent thinking/text."
    )


def test_process_output_event_for_bg_tab_tool_call_sets_pending_true() -> None:
    """processOutputEventForBgTab must set bgPendingPanel = true on tool_call."""
    src = _read_main_js()
    body = _extract_function_body(src, "processOutputEventForBgTab")

    m = re.search(
        r"t\s*===\s*'tool_call'[^}]*?bgPendingPanel\s*=\s*(\w+)",
        body,
    )
    assert m, (
        "processOutputEventForBgTab must set bgPendingPanel on tool_call"
    )
    assert m.group(1) == "true", (
        f"processOutputEventForBgTab sets bgPendingPanel = {m.group(1)} "
        "on tool_call; must be true."
    )


def test_replay_events_into_tool_call_sets_pending_true() -> None:
    """replayEventsInto must set rPendingPanel = true on tool_call."""
    src = _read_main_js()
    body = _extract_function_body(src, "replayEventsInto")

    m = re.search(
        r"t\s*===\s*'tool_call'[^}]*?rPendingPanel\s*=\s*(\w+)",
        body,
    )
    assert m, (
        "replayEventsInto must set rPendingPanel on tool_call"
    )
    assert m.group(1) == "true", (
        f"replayEventsInto sets rPendingPanel = {m.group(1)} "
        "on tool_call; must be true."
    )


# ── Backend test: tool_result is broadcast for every non-finish tool ──────


def test_noncore_tool_result_is_broadcast() -> None:
    """Verify that tool_result for non-core tools IS broadcast.

    Earlier the printer suppressed ``tool_result`` for tools outside
    the {Bash, Read, Edit, Write} whitelist.  Commit a8d394a7 removed
    that whitelist — every tool's return value is now rendered except
    ``finish`` (the agentic loop renders ``finish`` as a dedicated
    ``result`` panel).  This test pins the new behaviour so any future
    "suppress non-core tools" regression fails loudly.
    """
    from kiss.agents.vscode.json_printer import JsonPrinter

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
    from kiss.agents.vscode.json_printer import JsonPrinter

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


# ── End-to-end event sequence test ────────────────────────────────────────


def test_thinking_after_noncore_tool_gets_panel_events() -> None:
    """Simulate a non-core tool turn and verify the post-a8d394a7 event order.

    After commit a8d394a7 the printer broadcasts ``tool_result`` for
    every non-``finish`` tool, so the screenshot tool_call is followed
    by its tool_result *before* the next thinking_start.  The frontend
    still creates a Thoughts panel for the thinking_start because
    ``pendingPanel`` is set ``true`` on the tool_call itself (the
    invariant pinned by the tests above).
    """
    from kiss.agents.vscode.json_printer import JsonPrinter

    printer = JsonPrinter()
    printer._thread_local.task_id = "t1"
    printer.start_recording()

    # Turn 1: thinking → tool_call(Read) → tool_result → tool_call(screenshot)
    printer.thinking_callback(True)
    printer.token_callback("Let me read the file")
    printer.thinking_callback(False)
    printer.print("Read", type="tool_call", tool_input={"file_path": "test.html"})
    printer.print("file contents", type="tool_result", tool_name="Read")
    printer.print("screenshot", type="tool_call", tool_input={})
    printer.print("screenshot taken", type="tool_result", tool_name="screenshot")

    # Turn 2: thinking → text → tool_call(Write)
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

    # tool_result(screenshot) is now broadcast and appears between the
    # screenshot tool_call and the next thinking_start.
    post_screenshot = types[screenshot_idx + 1 :]
    thinking_start_offset = post_screenshot.index("thinking_start")
    between = post_screenshot[:thinking_start_offset]
    assert "tool_result" in between, (
        "tool_result for the screenshot tool must be broadcast between "
        f"its tool_call and the next thinking_start, got: {between}"
    )


# ── Step-counting consistency tests ───────────────────────────────────────


def test_step_count_code_uses_tool_call_pending_true() -> None:
    """Step-counting code in task_events and replayTaskEvents must use
    bgPending/rPending = true on tool_call so non-core tools don't
    under-count steps.
    """
    src = _read_main_js()

    # Check the bg tab task_events step counting loop
    # Pattern: if (t === 'tool_call') { ... bgPending = true/false; }
    bg_step_match = re.search(
        r"bgSteps\s*===\s*0.*?tool_call.*?bgPending\s*=\s*(\w+)",
        src,
        re.DOTALL,
    )
    # If it exists, verify it sets bgPending = true
    if bg_step_match:
        assert bg_step_match.group(1) == "true", (
            f"Step counting sets bgPending = {bg_step_match.group(1)} on "
            "tool_call; must be true for consistency."
        )

    # Check the replayTaskEvents step counting loop
    replay_step_match = re.search(
        r"rSteps\s*===\s*0.*?tool_call.*?rPending\s*=\s*(\w+)",
        src,
        re.DOTALL,
    )
    if replay_step_match:
        assert replay_step_match.group(1) == "true", (
            f"Step counting sets rPending = {replay_step_match.group(1)} on "
            "tool_call; must be true for consistency."
        )
