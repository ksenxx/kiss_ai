# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression test: copying text from the fixed task panel must not
produce trailing newlines.

Bug history
-----------
Originally fixed in commit ``7ce815be`` ("style: prevent text selection
on task panel chevron") with two structural guards:

1. The task-panel HTML template in ``SorcarTab.ts`` was minified so
   there is no whitespace text node between ``</button>`` and
   ``<div id="task-panel-text">``.  Otherwise ``Selection.toString()``
   would serialize that whitespace (a newline) as part of the copied
   payload.
2. ``main.css`` was given ``user-select: none`` on
   ``#task-panel-chevron`` and ``#task-panel-chevron *`` so the user's
   triple-click / select-all gesture cannot extend the selection into
   the chevron's surrounding whitespace.

Both guards are still required, but the bug came back through a third
data path: ``taskPanelText.textContent`` is also written by code that
does **not** trim its input.  In particular:

* ``restoreTab`` (background-tab → active-tab switch) used to assign
  ``tab.taskPanelHTML`` directly to ``taskPanelText.textContent`` with
  no trim.
* ``openSubagentTab`` used to store ``ev.description`` directly into
  ``subTab.taskPanelHTML``, so the next ``restoreTab`` would write
  trailing-newline-laden text into the live panel.

This test pins down all three guards so the bug cannot regress
silently a third time.
"""

from __future__ import annotations

import re
from pathlib import Path

VSCODE_DIR = Path(__file__).resolve().parents[3] / "agents" / "vscode"
MAIN_JS = VSCODE_DIR / "media" / "main.js"
MAIN_CSS = VSCODE_DIR / "media" / "main.css"
SORCAR_TAB_TS = VSCODE_DIR / "src" / "SorcarTab.ts"


def _read(path: Path) -> str:
    assert path.is_file(), f"file not found: {path}"
    return path.read_text()


# ── Guard 1: minified HTML template ─────────────────────────────────


def test_task_panel_html_has_no_inter_element_whitespace() -> None:
    """The ``#task-panel`` element's children must sit immediately
    adjacent in the source — no whitespace, no newlines — so that
    ``Selection.toString()`` cannot serialize a whitespace text node
    between the chevron button and the task text div.
    """
    src = _read(SORCAR_TAB_TS)

    # Locate the literal "<div id=\"task-panel\"" up to the closing
    # "</div>" of the task panel.  The match must contain a chevron
    # button directly followed by the task-text div with NO whitespace.
    m = re.search(
        r'<div id="task-panel">(.*?)</div>\s*</div>',
        src,
        re.DOTALL,
    )
    assert m, "could not locate <div id=\"task-panel\"> in SorcarTab.ts"
    inner = m.group(1)

    # The chevron button's closing tag must be directly followed by
    # the task-text div opening tag.  Any whitespace between them is
    # rendered as a text node and gets included in the selection
    # range when the user copies the panel.
    assert re.search(r'</button><div id="task-panel-text">', inner), (
        "task-panel HTML must keep </button> and <div id=\"task-panel-text\"> "
        "directly adjacent (no whitespace, no newline) — otherwise the "
        "whitespace text node leaks into the clipboard as a trailing newline "
        "when the user copy-selects the panel."
    )


# ── Guard 2: chevron is unselectable ────────────────────────────────


def _css_rules_for_selector(css: str, selector: str) -> list[str]:
    pattern = re.compile(
        re.escape(selector) + r"\s*\{([^}]*)\}",
        re.DOTALL,
    )
    return [m.group(1) for m in pattern.finditer(css)]


def test_task_panel_chevron_is_unselectable() -> None:
    """``#task-panel-chevron`` and all its descendants must have
    ``user-select: none`` so the chevron's own text (the SVG glyph and
    any wrapping whitespace) can never be selected.
    """
    css = _read(MAIN_CSS)

    chev_blocks = _css_rules_for_selector(css, "#task-panel-chevron")
    chev_desc_blocks = _css_rules_for_selector(css, "#task-panel-chevron *")

    def _has_user_select_none(blocks: list[str]) -> bool:
        for b in blocks:
            if re.search(r"user-select\s*:\s*none", b):
                return True
        return False

    assert _has_user_select_none(chev_blocks), (
        "#task-panel-chevron must set user-select: none so triple-click "
        "selection cannot extend into the chevron"
    )
    assert _has_user_select_none(chev_desc_blocks), (
        "#task-panel-chevron * must set user-select: none so the inner "
        "SVG / polyline does not become selectable"
    )


# ── Guard 3: every write-path trims ─────────────────────────────────


def test_set_task_text_trims_input() -> None:
    """``setTaskText`` must trim its input before writing to
    ``taskPanelText.textContent``.
    """
    src = _read(MAIN_JS)
    m = re.search(
        r"function setTaskText\([^)]*\)\s*\{(.*?)^\s{2}\}",
        src,
        re.DOTALL | re.MULTILINE,
    )
    assert m, "could not locate function setTaskText in main.js"
    body = m.group(1)
    assert re.search(r"\(\s*text\s*\|\|\s*''\s*\)\s*\.\s*trim\s*\(\)", body), (
        "setTaskText must call (text || '').trim() before assigning to "
        "taskPanelText.textContent — otherwise trailing newlines on the "
        "incoming event payload bleed into the user's clipboard."
    )


def test_restore_tab_trims_task_panel_text() -> None:
    """``restoreTab`` must trim ``tab.taskPanelHTML`` before assigning
    to ``taskPanelText.textContent``.  ``taskPanelHTML`` can be written
    by background-tab handlers from raw event payloads, so the restore
    step is the last line of defence.
    """
    src = _read(MAIN_JS)
    m = re.search(
        r"function restoreTab\([^)]*\)\s*\{(.*?)^\s{2}\}",
        src,
        re.DOTALL | re.MULTILINE,
    )
    assert m, "could not locate function restoreTab in main.js"
    body = m.group(1)
    assert re.search(
        r"taskPanelText\.textContent\s*=\s*\(\s*tab\.taskPanelHTML\s*\|\|\s*''\s*\)\s*\.\s*trim\s*\(\)",
        body,
    ), (
        "restoreTab must assign `(tab.taskPanelHTML || '').trim()` to "
        "taskPanelText.textContent — a raw assignment lets trailing "
        "newlines stored on a background tab surface in the live panel "
        "and then leak into the user's clipboard on copy."
    )


def test_no_raw_textcontent_assignment_to_task_panel() -> None:
    """No code path may assign to ``taskPanelText.textContent`` from a
    source that hasn't been trimmed.  The only acceptable RHS values are:

    * an empty string ``''`` (clearing the panel)
    * the local ``t`` inside ``setTaskText`` (already trimmed)
    * an explicit ``.trim()`` expression

    Any other assignment is a regression risk and this test will flag it.
    """
    src = _read(MAIN_JS)

    assignments = re.findall(
        r"taskPanelText\.textContent\s*=\s*([^;]+);",
        src,
    )
    assert assignments, (
        "expected at least one assignment to taskPanelText.textContent in "
        "main.js — refactor probably renamed the element"
    )

    for rhs in assignments:
        rhs = rhs.strip()
        if rhs == "''" or rhs == '""':
            continue
        if rhs == "t":
            # setTaskText's local — verified trimmed by test_set_task_text_trims_input
            continue
        if ".trim()" in rhs:
            continue
        raise AssertionError(
            f"taskPanelText.textContent is assigned from an untrimmed "
            f"expression `{rhs}` — this is exactly how the trailing-newline "
            f"clipboard bug regressed in the past.  Either trim the RHS or "
            f"route the assignment through setTaskText()."
        )
