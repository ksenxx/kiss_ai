"""End-to-end tests for the page-side JavaScript of the bridge's
snapshot/click/type_text commands.

The sources under test are extracted from the shipped extension
(``chrome_extension/background.js``), never copied, so this suite breaks
when the extension's behavior changes. The background.js template literals
are unescaped the way V8 unescapes them before the code reaches the page
(``\\\\`` -> ``\\``; the sources use no other escapes).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from patchright.sync_api import sync_playwright

BACKGROUND_JS = (
    Path(__file__).parents[3] / "agents" / "sorcar" / "chrome_extension" / "background.js"
)

TEST_PAGE = """<!doctype html><html><head><title>Snapshot Test Page</title></head><body>
<button id="b1">Save detector</button>
<button style="display:none">Hidden by display</button>
<button style="visibility:hidden">Hidden by visibility</button>
<a href="https://example.com/docs">Docs</a>
<input type="text" placeholder="Search detectors">
<input type="hidden" value="secret">
<select><option>Reve.Art</option><option selected>hunters games ctf</option></select>
<textarea placeholder="Description"></textarea>
<div role="button" id="tenant">Reve.Art tenant</div>
<div role="presentation">pure decoration</div>
<div tabindex="-1">not interactive</div>
<div tabindex="0">focusable div</div>
<button disabled>Disabled action</button>
<input type="checkbox" checked>
<div contenteditable="true">edit me</div>
<button aria-label="icon-only-button"></button>
<div style="cursor:pointer"><span>Nested pointer label</span></div>
<div style="cursor:pointer">Pointer card</div>
<div style="cursor:pointer" aria-label="Close dialog"><svg><path d="M1 1l8 8"/></svg></div>
<div style="cursor:pointer" id="bare-icon"><i></i></div>
</body></html>"""


def _extract(name: str) -> str:
    """Pull one JS source constant out of background.js, V8-unescaped."""
    match = re.search(rf"const {name} = `([^`]+)`;", BACKGROUND_JS.read_text(encoding="utf-8"))
    assert match, f"{name} not found in background.js"
    return match.group(1).replace("\\\\", "\\")


SNAPSHOT_JS = _extract("KISS_SNAPSHOT_JS")
CLICK_JS = _extract("KISS_CLICK_JS")
TYPE_JS = _extract("KISS_TYPE_JS")


@pytest.fixture(scope="module")
def page():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        pg = browser.new_page()
        pg.set_content(TEST_PAGE)
        yield pg
        browser.close()


def _uid_of(snapshot: str, needle: str) -> int:
    for line in snapshot.splitlines():
        if needle in line:
            return int(line.split("]", 1)[0][1:])
    raise AssertionError(f"'{needle}' not in snapshot:\n{snapshot}")


def test_snapshot_lists_visible_interactive_elements(page) -> None:
    snap = page.evaluate(SNAPSHOT_JS)
    head = snap.splitlines()[0]
    assert head.startswith("Snapshot Test Page — ")
    assert "interactive elements" in head
    # Visible interactives are listed, numbered from 1, in DOM order.
    assert '[1] <button> "Save detector"' in snap
    assert '<a> "Docs" -> https://example.com/docs' in snap
    assert '<input type=text> "Search detectors"' in snap
    assert '<select> "hunters games ctf"' in snap
    assert '<textarea> "Description"' in snap
    assert '<div role=button> "Reve.Art tenant"' in snap
    assert '<div> "focusable div"' in snap
    assert '<button> "Disabled action" [disabled]' in snap
    assert '<input type=checkbox> [checked]' in snap
    assert '<div> "edit me"' in snap
    assert '<button> "icon-only-button"' in snap
    # Non-visible and non-interactive elements are excluded.
    assert "Hidden by display" not in snap
    assert "Hidden by visibility" not in snap
    assert "pure decoration" not in snap
    assert "not interactive" not in snap
    assert "secret" not in snap


def test_cursor_pointer_divs_are_listed_innermost_only(page) -> None:
    """React-style clickable divs (no role/tabindex) must appear; when a
    clickable wraps another clickable, only the inner one is listed — a
    click on it bubbles up through every clickable ancestor."""
    snap = page.evaluate(SNAPSHOT_JS)
    assert '<span> "Nested pointer label"' in snap
    assert '<div> "Pointer card"' in snap
    pointer_lines = [ln for ln in snap.splitlines() if "pointer" in ln.lower()]
    assert len(pointer_lines) == 2  # the wrapper div must not be listed too


def test_icon_leaves_collapse_to_their_container(page) -> None:
    """Icon glyphs are the same click target as their container with a worse
    label: a labeled icon button lists as its label, an unlabeled one as a
    bare div — never as cryptic <svg>/<path>/<i> lines."""
    snap = page.evaluate(SNAPSHOT_JS)
    assert '<div> "Close dialog"' in snap
    for line in snap.splitlines():
        assert "<svg" not in line and "<path" not in line and "<i>" not in line
    # The bare icon control is still present (as its container div).
    bare = [ln for ln in snap.splitlines() if ln.rstrip().endswith("<div>")]
    assert len(bare) == 1


def test_snapshot_retags_on_every_run(page) -> None:
    first = page.evaluate(SNAPSHOT_JS)
    uid = _uid_of(first, "Save detector")
    assert page.evaluate("document.querySelectorAll('[data-kiss-uid]').length") > 0
    second = page.evaluate(SNAPSHOT_JS)
    assert _uid_of(second, "Save detector") == uid  # same DOM, same numbering
    # ... and the retagging did not pile up duplicate attributes.
    tagged = page.evaluate("document.querySelectorAll('[data-kiss-uid]').length")
    assert tagged == len([ln for ln in second.splitlines() if ln.startswith("[")])


def test_click_dispatches_the_full_event_sequence(page) -> None:
    page.evaluate(SNAPSHOT_JS)
    page.evaluate(
        """window.__events = [];
        const el = document.getElementById('tenant');
        for (const t of ['pointerdown', 'mousedown', 'pointerup', 'mouseup', 'click']) {
          el.addEventListener(t, () => window.__events.push(t));
        }"""
    )
    snap = page.evaluate(SNAPSHOT_JS)
    uid = _uid_of(snap, "Reve.Art tenant")
    out = page.evaluate(f"({CLICK_JS})({uid})")
    assert out == f'clicked [{uid}] <div> "Reve.Art tenant"'
    assert page.evaluate("window.__events") == [
        "pointerdown", "mousedown", "pointerup", "mouseup", "click",
    ]


def test_click_stale_uid_errors(page) -> None:
    page.evaluate(SNAPSHOT_JS)
    with pytest.raises(Exception, match=r"no longer exists.*fresh snapshot"):
        page.evaluate(f"({CLICK_JS})(999)")


def test_type_text_sets_value_and_fires_framework_events(page) -> None:
    page.evaluate(
        """window.__typed = [];
        const el = document.querySelector('input[type=text]');
        for (const t of ['input', 'change']) {
          el.addEventListener(t, () => window.__typed.push(t));
        }"""
    )
    snap = page.evaluate(SNAPSHOT_JS)
    uid = _uid_of(snap, "Search detectors")
    out = page.evaluate(f"({TYPE_JS})({uid}, {json.dumps('new detector query')})")
    assert out == f"typed 18 chars into [{uid}] <input>"
    assert page.evaluate("document.querySelector('input[type=text]').value") == "new detector query"
    assert page.evaluate("window.__typed") == ["input", "change"]


def test_type_text_replaces_contenteditable(page) -> None:
    snap = page.evaluate(SNAPSHOT_JS)
    uid = _uid_of(snap, "edit me")
    page.evaluate(f"({TYPE_JS})({uid}, {json.dumps('replaced text')})")
    actual = page.evaluate("document.querySelector('[contenteditable]').textContent")
    assert actual == "replaced text"


def test_type_text_rejects_non_text_elements(page) -> None:
    snap = page.evaluate(SNAPSHOT_JS)
    uid = _uid_of(snap, "Save detector")
    with pytest.raises(Exception, match="not a text field"):
        page.evaluate(f"({TYPE_JS})({uid}, {json.dumps('nope')})")


def test_shadow_dom_elements_are_listed_clicked_and_typed(page) -> None:
    page.evaluate(
        """const host = document.createElement('div');
        host.id = 'shadow-host';
        document.body.appendChild(host);
        const root = host.attachShadow({ mode: 'open' });
        root.innerHTML = '<button id="sh">Shadow action</button>'
          + '<input type="text" placeholder="Shadow field">';
        window.__shadow_clicked = false;
        root.getElementById('sh').addEventListener('click', () => {
          window.__shadow_clicked = true;
        });"""
    )
    snap = page.evaluate(SNAPSHOT_JS)
    click_uid = _uid_of(snap, "Shadow action")
    type_uid = _uid_of(snap, "Shadow field")
    page.evaluate(f"({CLICK_JS})({click_uid})")
    assert page.evaluate("window.__shadow_clicked") is True
    page.evaluate(f"({TYPE_JS})({type_uid}, {json.dumps('deep text')})")
    value = page.evaluate(
        "document.getElementById('shadow-host').shadowRoot.querySelector('input').value"
    )
    assert value == "deep text"
