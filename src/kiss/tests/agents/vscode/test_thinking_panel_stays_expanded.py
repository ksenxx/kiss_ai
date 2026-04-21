"""Regression tests for the thinking panel UI behaviour in ``main.js``.

The bug (as reported by the user): when the ``cc/*`` model is used the
browser UI would collapse the thinking panel on ``thinking_end`` and show a
``Thinking (click to expand)`` bar instead of the streamed thinking tokens.

These tests enforce two invariants of the webview JavaScript:

1. The ``Thinking (click to expand)`` label MUST NOT appear anywhere in
   ``main.js`` — the user explicitly said it "MUST not be shown".

2. The ``thinking_end`` handler MUST NOT hide the streamed content by
   adding the ``hidden`` CSS class to ``.cnt``; the panel must stay
   expanded so the user keeps seeing the streamed tokens.

The tests operate on the JavaScript source (no mocks, no DOM shims).
They complement the Python-side integration tests in
``src/kiss/tests/core/models/test_cc_thinking_ui_integration.py`` which
verify that ``thinking_start`` / ``thinking_delta`` / ``thinking_end``
events are broadcast in the correct order with real streaming tokens.
"""

from __future__ import annotations

import re
from pathlib import Path

MAIN_JS = (
    Path(__file__).parent.parent.parent.parent
    / "agents"
    / "vscode"
    / "media"
    / "main.js"
)


def _read_main_js() -> str:
    assert MAIN_JS.is_file(), f"main.js not found at {MAIN_JS}"
    return MAIN_JS.read_text()


def _extract_thinking_end_case(src: str) -> str:
    """Return the source of the ``case 'thinking_end':`` branch.

    The body runs until (but not including) the next ``case '`` label or
    the closing ``}`` of the ``switch`` — whichever comes first.  The
    test only needs the body to exist and to have been narrowed down to
    the thinking_end arm, so we extract up to the first ``break;``.
    """
    m = re.search(r"case\s+'thinking_end':\s*(.*?)break;", src, re.DOTALL)
    assert m, "could not locate case 'thinking_end': in main.js"
    return m.group(1)


def test_main_js_does_not_contain_click_to_expand_label() -> None:
    """The ``Thinking (click to expand)`` label is forbidden in main.js."""
    src = _read_main_js()
    assert "click to expand" not in src, (
        "main.js still contains the forbidden 'click to expand' label. "
        "The user explicitly said this bar MUST NOT be shown — the "
        "thinking panel must stay expanded with the streamed tokens."
    )


def test_thinking_end_handler_does_not_hide_content() -> None:
    """``thinking_end`` must keep the ``.cnt`` element visible.

    Concretely: it must not add the ``hidden`` class and must not set
    a ``(click to expand)`` label.  The panel stays expanded so the
    streamed thinking tokens remain on screen.
    """
    src = _read_main_js()
    body = _extract_thinking_end_case(src)

    assert "hidden" not in body, (
        "thinking_end handler still hides the .cnt element "
        "(found 'hidden' in the case body). Panel must stay expanded."
    )
    assert "click to expand" not in body, (
        "thinking_end handler still sets a 'click to expand' label."
    )
    assert "collapsed" not in body, (
        "thinking_end handler still flips the arrow to the collapsed state."
    )


def test_thinking_start_handler_creates_expanded_panel() -> None:
    """Sanity: ``thinking_start`` still creates the panel with a visible ``.cnt``.

    Confirms the streaming deltas have somewhere to append to.
    """
    src = _read_main_js()
    m = re.search(r"case\s+'thinking_start':\s*(.*?)break;", src, re.DOTALL)
    assert m, "could not locate case 'thinking_start':"
    body = m.group(1)
    assert "thinkEl" in body
    assert "cnt" in body, "thinking_start must create the .cnt element"
    # No 'hidden' class is added at creation — the panel is expanded.
    assert "add('hidden')" not in body
