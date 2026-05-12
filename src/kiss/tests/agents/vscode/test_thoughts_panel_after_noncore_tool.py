"""Integration tests: thinking/text after non-core tools must go into Thoughts panels.

Bug: When the model calls a non-core tool (screenshot, go_to_url, scroll,
click, etc.), the printer does NOT broadcast a ``tool_result`` event because
``tool_name not in core_tools`` (core_tools = {Bash, Read, Edit, Write}).
The frontend ``processOutputEvent`` only sets ``pendingPanel = true`` on
``tool_result``, and sets ``pendingPanel = false`` on ``tool_call``.  So
after a non-core tool_call whose tool_result is suppressed, ``pendingPanel``
stays ``false`` and ``stepCount > 0``, causing the next ``thinking_start``
or ``text_delta`` to bypass Thoughts-panel creation and render directly
in the main output area.

Real-world example (from task 1022 — "the dependency lines are not showing
up?"):

    [7]  tool_call(Read)       → pendingPanel = false
    [8]  tool_result(Read)     → pendingPanel = true   (Read is core)
    [9]  text_end              → no-op
    [10] tool_call(screenshot) → pendingPanel = false   ← BUG
    --- tool_result for screenshot NOT broadcast (not core) ---
    [11] thinking_start        → pendingPanel=false, stepCount>0 → NO PANEL ❌

Fix: Set ``pendingPanel = true`` (not false) on ``tool_call`` events so
that after every tool call — regardless of whether the tool_result is
broadcast — the next thinking/text block gets its own Thoughts panel.
This change must be applied in:
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
    / "agents"
    / "vscode"
    / "browser_ui.py"
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


# ── Backend test: non-core tool_result is suppressed ──────────────────────


def test_noncore_tool_result_is_suppressed() -> None:
    """Verify that tool_result for non-core tools is NOT broadcast.

    This is the root cause: the printer suppresses tool_result for
    tools not in core_tools (Bash, Read, Edit, Write), so the frontend
    never sees tool_result and pendingPanel stays false.
    """
    from kiss.agents.vscode.browser_ui import BaseBrowserPrinter

    printer = BaseBrowserPrinter()
    printer.start_recording()

    # Simulate a non-core tool call + result
    printer.print("screenshot", type="tool_call", tool_input={})
    printer.print("image data...", type="tool_result", tool_name="screenshot")

    events = printer.stop_recording()
    types = [e["type"] for e in events]

    # tool_call should produce text_end + tool_call
    assert "tool_call" in types, "tool_call event must be broadcast"

    # tool_result should NOT be broadcast for non-core tools
    assert "tool_result" not in types, (
        "tool_result for non-core tool 'screenshot' should be suppressed. "
        "This is by design (reduces UI noise), but means the frontend "
        "must handle panel creation without relying on tool_result."
    )


def test_core_tool_result_is_broadcast() -> None:
    """Verify that tool_result for core tools IS broadcast."""
    from kiss.agents.vscode.browser_ui import BaseBrowserPrinter

    printer = BaseBrowserPrinter()
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
    """Simulate the real event sequence and verify panel-creation events.

    Replays the event sequence from task 1022 through the printer to
    confirm that the backend emits the events in the order that the
    frontend must handle.  The test verifies:
    1. Non-core tool_result is suppressed
    2. The event stream matches the pattern that causes the bug
    """
    from kiss.agents.vscode.browser_ui import BaseBrowserPrinter

    printer = BaseBrowserPrinter()
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

    # The critical pattern: after tool_call(screenshot), no tool_result,
    # then thinking_start.  Without the fix, the frontend wouldn't create
    # a Thoughts panel for this thinking_start.
    screenshot_idx = None
    for i, e in enumerate(events):
        if e["type"] == "tool_call" and e.get("name") == "screenshot":
            screenshot_idx = i
            break
    assert screenshot_idx is not None

    # Verify no tool_result between screenshot tool_call and next thinking_start
    post_screenshot = types[screenshot_idx + 1 :]
    thinking_start_offset = post_screenshot.index("thinking_start")
    between = post_screenshot[:thinking_start_offset]
    assert "tool_result" not in between, (
        f"Expected no tool_result between screenshot tool_call and thinking_start, "
        f"got: {between}"
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
