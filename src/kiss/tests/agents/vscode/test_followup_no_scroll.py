# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression test: ``Suggested next`` MUST NOT scroll the chat to the bottom.

The bug (reported by the user): rendering a ``followup_suggestion`` event in
the live webview message handler appended the ``Suggested next`` bar to the
output area AND then called ``sb()`` (the scroll-to-bottom helper).  This
forced the chat view to jump to the end every time a follow-up was streamed,
even if the user had scrolled up to read earlier content.

Symmetrically, the *replay* path (``replayEventsInto`` in ``main.js``) renders
the same bar and does NOT scroll — so the live handler was the outlier.

This test enforces, at the JavaScript-source level, that the live
``case 'followup_suggestion':`` arm of ``handleEvent`` does NOT invoke ``sb()``.
A pure static-source check (no DOM shim, no Node) is sufficient because the
fix is the removal of a single ``sb();`` call inside that case body.
"""

from __future__ import annotations

import re
from pathlib import Path

_MAIN_JS = (
    Path(__file__).resolve().parents[4]
    / "kiss"
    / "agents"
    / "vscode"
    / "media"
    / "main.js"
)


def _read_main_js() -> str:
    assert _MAIN_JS.is_file(), f"main.js not found at {_MAIN_JS}"
    return _MAIN_JS.read_text()


def _extract_live_followup_case(src: str) -> str:
    """Return the body of the live ``case 'followup_suggestion':`` arm.

    The live handler lives inside ``function handleEvent(ev) { switch(...) { ... } }``
    near line ~2380 of main.js.  ``replayEventsInto`` (around line ~3010)
    handles the same event but on a different code path; we want only the
    live one.

    We disambiguate by extracting the case body that is followed by
    ``break;`` (the live arm uses ``break;`` inside the switch; the replay
    arm uses ``return;`` because it lives inside a ``forEach`` callback).
    """
    # Match the case label, then capture everything up to the first ``break;``
    # that closes a brace block (the case body is wrapped in ``{ ... }``).
    pattern = (
        r"case\s+'followup_suggestion':\s*\{(?P<body>.*?)\n\s*break;\s*\n\s*\}"
    )
    matches = list(re.finditer(pattern, src, re.DOTALL))
    assert matches, (
        "Could not locate the live case 'followup_suggestion': { ... break; } "
        "arm in main.js — did the dispatcher structure change?"
    )
    # There should be exactly one such arm (the replay path uses ``return;``).
    assert len(matches) == 1, (
        f"Expected exactly one live followup_suggestion case body, found "
        f"{len(matches)}.  Update the test to disambiguate."
    )
    return matches[0].group("body")


def test_live_followup_suggestion_handler_does_not_call_sb() -> None:
    """The live followup_suggestion handler must not auto-scroll the chat.

    Concretely: the case body must not contain a call to the scroll-to-bottom
    helper ``sb()``.  Calling it forces the output area to jump to the end
    on every follow-up render, overriding the user's scroll position.
    """
    src = _read_main_js()
    body = _extract_live_followup_case(src)

    # ``sb`` is the only scroll-to-bottom helper in main.js; it is invoked as
    # ``sb()`` (no arguments) and never as a substring of another identifier
    # (we still anchor to be safe).
    assert not re.search(r"\bsb\s*\(\s*\)", body), (
        "Live case 'followup_suggestion' still calls sb() — this scrolls the "
        "chat to the bottom every time a 'Suggested next' bar is shown, which "
        "the user explicitly forbade.  Remove the sb() call from the case "
        "body in src/kiss/agents/vscode/media/main.js."
    )


def test_live_followup_suggestion_handler_still_appends_the_bar() -> None:
    """Sanity: removing sb() must not also remove the actual bar render.

    The case body must still:
      * create a ``followup-bar`` element,
      * set its inner HTML containing the ``Suggested next`` label, and
      * append it to the output area ``O``.
    """
    src = _read_main_js()
    body = _extract_live_followup_case(src)

    assert "followup-bar" in body, (
        "Live followup_suggestion handler no longer creates a 'followup-bar' "
        "element."
    )
    assert "Suggested next" in body, (
        "Live followup_suggestion handler no longer renders the "
        "'Suggested next' label."
    )
    assert re.search(r"O\.appendChild\s*\(", body), (
        "Live followup_suggestion handler no longer appends the bar to the "
        "output area O."
    )


def test_replay_followup_suggestion_does_not_call_sb() -> None:
    """Symmetry guard: the replay path also must not call sb().

    ``replayEventsInto`` renders the same ``followup-bar`` during task
    replay; it has historically NOT scrolled, and we lock that in to prevent
    a future refactor from re-introducing the live-side bug here.
    """
    src = _read_main_js()
    # Match the replay arm: ``if (t === 'followup_suggestion') { ... return; }``
    m = re.search(
        r"if\s*\(\s*t\s*===\s*'followup_suggestion'\s*\)\s*\{(?P<body>.*?)\n\s*return;\s*\n\s*\}",
        src,
        re.DOTALL,
    )
    assert m, "Could not locate the replay-side followup_suggestion branch."
    body = m.group("body")
    assert not re.search(r"\bsb\s*\(\s*\)", body), (
        "Replay followup_suggestion branch now calls sb(); it must not — "
        "rendering 'Suggested next' must never auto-scroll the chat."
    )
