"""Integration tests: first thinking events must always create a Thoughts panel.

Bug: When a new sub-session starts in RelentlessAgent (after a summarizer
finishes with a ``result`` event), or when a background tab receives new
task events after a previous task, the first ``thinking_start`` event does
not create a new ``llm-panel`` (Thoughts panel).

Root cause: The ``processOutputEvent`` state machine creates a new
``llm-panel`` only when ``(pendingPanel || stepCount === 0) && (t ===
'thinking_start' || t === 'text_delta')``.  After a ``result`` event,
``pendingPanel`` is still ``false`` (set to ``false`` by the preceding
``tool_call('finish')``), and ``stepCount`` is non-zero.  So the first
thinking of the next session falls through without a panel.

Similarly, when a ``clear`` event targets a background tab, the tab's
streaming state (``streamStepCount``, ``streamPendingPanel``, etc.) is
not reset, so the next task's first thinking also misses its panel.

Fix:
1. ``processOutputEvent``: set ``pendingPanel = true`` after a ``result``
   event so the next thinking/text creates a new panel.
2. ``processOutputEventForBgTab``: same for background tabs.
3. ``clear`` handler: reset background tab streaming state.
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


def _read_main_js() -> str:
    assert MAIN_JS.is_file(), f"main.js not found at {MAIN_JS}"
    return MAIN_JS.read_text()


def _extract_function_body(src: str, name: str) -> str:
    """Extract the full body of function *name* from JavaScript source."""
    pattern = re.compile(rf"function {re.escape(name)}\s*\([^)]*\)\s*\{{")
    m = pattern.search(src)
    assert m, f"function {name} not found in main.js"
    start = m.end() - 1  # include opening brace
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


def _extract_switch_case(src: str, label: str) -> str:
    """Extract the body of ``case '<label>':`` in main.js."""
    m = re.search(rf"case\s+'{re.escape(label)}':\s*\{{?\s*(.*?)break;", src, re.DOTALL)
    assert m, f"could not locate case '{label}': in main.js"
    return m.group(1)


def test_result_event_sets_pending_panel_in_active_tab() -> None:
    """processOutputEvent must set pendingPanel = true after a result event.

    Without this, the first thinking_start of a new RelentlessAgent
    sub-session would not create a Thoughts panel because pendingPanel
    is false and stepCount > 0.
    """
    src = _read_main_js()
    body = _extract_function_body(src, "processOutputEvent")

    # Find the section that handles 'result' events (after handleOutputEvent)
    # The fix should set pendingPanel = true somewhere after the result check
    # Match: after "t === 'result'" there must be "pendingPanel = true"
    # without crossing into another event type handler
    assert re.search(
        r"t\s*===\s*'result'.*?pendingPanel\s*=\s*true",
        body,
        re.DOTALL,
    ), (
        "processOutputEvent must set pendingPanel = true after handling a "
        "'result' event, so the next thinking_start creates a Thoughts panel. "
        "Without this, new RelentlessAgent sub-sessions show thinking text "
        "outside of a panel."
    )


def test_result_event_sets_pending_panel_in_bg_tab() -> None:
    """processOutputEventForBgTab must set bgPendingPanel = true after result.

    Same issue as the active-tab case but for background tabs.
    """
    src = _read_main_js()
    body = _extract_function_body(src, "processOutputEventForBgTab")

    # Use a pattern that specifically matches t === 'result' (not tool_result)
    assert re.search(
        r"t\s*===\s*'result'.*?bgPendingPanel\s*=\s*true",
        body,
        re.DOTALL,
    ), (
        "processOutputEventForBgTab must set bgPendingPanel = true after a "
        "'result' event so the next thinking_start on a background tab "
        "creates a Thoughts panel."
    )


def test_clear_resets_background_tab_streaming_state() -> None:
    """The ``clear`` handler must reset background tab streaming state.

    When a ``clear`` event targets a non-active tab, the tab's streaming
    state (streamStepCount, streamPendingPanel, etc.) must be reset so
    the first thinking event of the new task creates a panel.
    """
    src = _read_main_js()

    # Find the 'clear' case in the handleEvent switch
    # The case body should reset streaming state for background tabs
    clear_m = re.search(
        r"case\s+'clear'\s*:\s*\{(.*?)break;\s*\}",
        src,
        re.DOTALL,
    )
    assert clear_m, "could not locate case 'clear': in main.js"
    clear_body = clear_m.group(1)

    # The fix should reset streamStepCount or streamPendingPanel for the bg tab
    assert "streamStepCount" in clear_body or "streamPendingPanel" in clear_body, (
        "The 'clear' handler must reset the background tab's streaming state "
        "(streamStepCount, streamPendingPanel, etc.) when the event targets "
        "a non-active tab. Without this reset, the first thinking event of "
        "a new task on the background tab won't create a Thoughts panel."
    )


def test_first_thinking_panel_creation_condition() -> None:
    """The panel creation condition must account for result-to-thinking transitions.

    Verify that processOutputEvent's panel creation condition includes
    ``pendingPanel`` (which is now set to true after result events),
    ensuring new sessions always get a Thoughts panel.
    """
    src = _read_main_js()
    body = _extract_function_body(src, "processOutputEvent")

    # The panel creation line should be:
    # (pendingPanel || stepCount === 0) && (t === 'thinking_start' || t === 'text_delta')
    assert re.search(
        r"\(pendingPanel\s*\|\|\s*stepCount\s*===\s*0\)",
        body,
    ), (
        "processOutputEvent panel creation must use "
        "(pendingPanel || stepCount === 0) as part of the condition"
    )


def test_bg_tab_first_thinking_panel_creation_condition() -> None:
    """Background tab panel creation condition must also account for result events."""
    src = _read_main_js()
    body = _extract_function_body(src, "processOutputEventForBgTab")

    assert re.search(
        r"\(bgPendingPanel\s*\|\|\s*bgStepCount\s*===\s*0\)",
        body,
    ), (
        "processOutputEventForBgTab panel creation must use "
        "(bgPendingPanel || bgStepCount === 0) as part of the condition"
    )
