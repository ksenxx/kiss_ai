"""End-to-end tests for the page-side JavaScript of the bridge's
snapshot/click/type_text commands.

The sources under test are extracted from the shipped extension
(``chrome_extension/background.js``), never copied, so this suite breaks
when the extension's behavior changes. The background.js template literals
are unescaped the way V8 unescapes them before the code reaches the page
(``\\\\`` -> ``\\``; the sources use no other escapes).  The snapshot is
called the way the worker calls it, with the id of the bridge's command
(the test page counts the calls, as the bridge numbers its commands); the
name of the connection, the last argument of all three sources, is left
out: these tests have a single connection.
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
<button style="cursor:pointer">Export report<span style="display:none">hidden label</span></button>
</body></html>"""


def _extract(name: str) -> str:
    """Pull one JS source constant out of background.js, V8-unescaped, its shared source inlined."""
    match = re.search(rf"const {name} = `([^`]+)`;", BACKGROUND_JS.read_text(encoding="utf-8"))
    assert match, f"{name} not found in background.js"
    source = match.group(1).replace("\\\\", "\\")
    if "${KISS_TARGET_JS}" in source:
        source = source.replace("${KISS_TARGET_JS}", _extract("KISS_TARGET_JS"))
    return source


SNAPSHOT_JS = f"({_extract('KISS_SNAPSHOT_JS')})(window.kissId = (window.kissId || 0) + 1)"
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
    # A hidden descendant (text, cursor inherited) must not suppress its
    # visible clickable ancestor: the button stays, the hidden label goes.
    assert '<button> "Export report"' in snap
    assert "hidden label" not in snap
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


def _uids(snapshot: str) -> set[int]:
    return {int(ln.split("]", 1)[0][1:]) for ln in snapshot.splitlines() if ln.startswith("[")}


def test_a_uid_is_never_reused_for_another_element(page) -> None:
    """A handle from an earlier snapshot still names its element once the page changed.

    Every task shares the one Chrome: agent A snapshots, the page gains a
    control, agent B snapshots.  A's saved uid must click A's element, or
    fail as stale once it is gone; a renumbering turned "Save" into "Delete".
    """
    page.evaluate(
        """document.body.dataset.action = '';
        const save = document.createElement('button');
        save.id = 'save-draft';
        save.textContent = 'Save draft';
        save.onclick = () => { document.body.dataset.action = 'save'; };
        document.body.prepend(save);"""
    )
    first = page.evaluate(SNAPSHOT_JS)
    uid = _uid_of(first, "Save draft")
    page.evaluate(
        """const del = document.createElement('button');
        del.id = 'delete-account';
        del.textContent = 'Delete account';
        del.onclick = () => { document.body.dataset.action = 'delete'; };
        document.body.prepend(del);"""
    )
    second = page.evaluate(SNAPSHOT_JS)
    assert _uid_of(second, "Save draft") == uid
    assert _uid_of(second, "Delete account") not in _uids(first)  # a fresh uid for a new element
    assert page.evaluate(f"({CLICK_JS})({uid})") == f'clicked [{uid}] <button> "Save draft"'
    assert page.evaluate("document.body.dataset.action") == "save"
    # A cloned node carries the attribute along (and the copy may precede
    # the original), but the uid belongs to the element object: the old
    # handle clicks the original, before and after the next snapshot, and
    # the copy gets a fresh uid.
    page.evaluate(
        """const copy = document.getElementById('save-draft').cloneNode(true);
        copy.id = 'save-copy';
        copy.onclick = () => { document.body.dataset.action = 'copy'; };
        document.body.prepend(copy);
        document.body.dataset.action = '';"""
    )
    assert page.evaluate(f"({CLICK_JS})({uid})") == f'clicked [{uid}] <button> "Save draft"'
    assert page.evaluate("document.body.dataset.action") == "save"
    cloned = page.evaluate(SNAPSHOT_JS)
    save_uids = _uids("\n".join(ln for ln in cloned.splitlines() if '"Save draft"' in ln))
    assert uid in save_uids and len(save_uids) == 2
    assert not (save_uids - {uid}) & _uids(second)  # a fresh uid for the copy
    tagged = page.evaluate("document.querySelectorAll('[data-kiss-uid]').length")
    assert tagged == len(_uids(cloned))
    page.evaluate("document.body.dataset.action = ''")
    assert page.evaluate(f"({CLICK_JS})({uid})") == f'clicked [{uid}] <button> "Save draft"'
    assert page.evaluate("document.body.dataset.action") == "save"
    page.evaluate("document.getElementById('save-copy').remove()")
    # The page replaces the element with a clone (the attribute travels
    # with it) that carries another label and handler: the old handle fails
    # as stale before any snapshot, and at the next one the replacement
    # gets a fresh uid while the old one is gone for good.
    page.evaluate(
        """const old = document.getElementById('save-draft');
        const swapped = old.cloneNode(true);
        swapped.id = 'delete-all';
        swapped.textContent = 'Delete everything';
        swapped.onclick = () => { document.body.dataset.action = 'delete-all'; };
        old.replaceWith(swapped);
        document.body.dataset.action = '';"""
    )
    with pytest.raises(Exception, match=r"no longer exists.*fresh snapshot"):
        page.evaluate(f"({CLICK_JS})({uid})")
    with pytest.raises(Exception, match=r"no longer exists.*fresh snapshot"):
        page.evaluate(f"({TYPE_JS})({uid}, 'x')")
    assert page.evaluate("document.body.dataset.action") == ""
    third = page.evaluate(SNAPSHOT_JS)
    assert uid not in _uids(third)
    assert _uid_of(third, "Delete everything") not in _uids(cloned)
    with pytest.raises(Exception, match=r"no longer exists.*fresh snapshot"):
        page.evaluate(f"({CLICK_JS})({uid})")
    assert page.evaluate("document.body.dataset.action") == ""
    page.evaluate("document.getElementById('delete-all').remove()")
    page.evaluate("document.getElementById('delete-account').remove()")


def test_click_works_on_svg_controls(page) -> None:
    """An ``<svg role=button>`` the snapshot lists is clickable: SVG elements lack ``click()``."""
    page.evaluate(
        """const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.id = 'svg-button';
        svg.setAttribute('role', 'button');
        svg.setAttribute('aria-label', 'Remove item');
        svg.setAttribute('width', '20');
        svg.setAttribute('height', '20');
        window.__svg_clicked = false;
        svg.addEventListener('click', () => { window.__svg_clicked = true; });
        document.body.appendChild(svg);"""
    )
    snap = page.evaluate(SNAPSHOT_JS)
    uid = _uid_of(snap, "Remove item")
    assert page.evaluate(f"({CLICK_JS})({uid})") == f'clicked [{uid}] <svg> "Remove item"'
    assert page.evaluate("window.__svg_clicked") is True
    page.evaluate("document.getElementById('svg-button').remove()")


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
    # An empty text clears the populated field, with the same events (A15-8).
    assert page.evaluate(f"({TYPE_JS})({uid}, '')") == f"typed 0 chars into [{uid}] <input>"
    assert page.evaluate("document.querySelector('input[type=text]').value") == ""
    assert page.evaluate("window.__typed") == ["input", "change", "input", "change"]


def test_type_text_replaces_contenteditable(page) -> None:
    page.evaluate(
        """window.__edits = [];
        document.querySelector('[contenteditable]').addEventListener(
          'input', (e) => window.__edits.push(e.target.textContent));"""
    )
    snap = page.evaluate(SNAPSHOT_JS)
    uid = _uid_of(snap, "edit me")
    page.evaluate(f"({TYPE_JS})({uid}, {json.dumps('replaced text')})")
    actual = page.evaluate("document.querySelector('[contenteditable]').textContent")
    assert actual == "replaced text"
    # An empty text clears the editor, and its input listeners hear of that
    # too (an editor's model is kept by them): each replacement is one edit.
    assert page.evaluate(f"({TYPE_JS})({uid}, '')") == f"typed 0 chars into [{uid}] <div>"
    assert page.evaluate("document.querySelector('[contenteditable]').textContent") == ""
    assert page.evaluate("window.__edits") == ["replaced text", ""]


def test_native_controls_are_listed_whatever_their_tabindex_or_role(page) -> None:
    """A button or field stays clickable and editable with ``tabindex=-1`` or a layout role.

    A negative tabindex only takes an element out of the Tab order (every
    button but one of a roving-tabindex toolbar has it), and a date
    picker's days are ``<button role="gridcell">``: neither makes a
    native control something the agent cannot press, nor an element the
    page gave a click handler (a table's clickable row).
    """
    page.evaluate(
        """const box = document.createElement('div');
        box.id = 'native-box';
        box.innerHTML = '<button tabindex="-1">Roving toolbar button</button>'
          + '<input tabindex="-1" placeholder="Unfocusable field">'
          + '<button role="gridcell">Day fourteen</button>'
          + '<div role="row" onclick="void 0">Clickable row</div>'
          + '<div tabindex="-1" role="dialog">Dialog panel</div>';
        document.body.appendChild(box);"""
    )
    snap = page.evaluate(SNAPSHOT_JS)
    page.evaluate("document.getElementById('native-box').remove()")
    assert '<button> "Roving toolbar button"' in snap
    assert '<input type=text> "Unfocusable field"' in snap
    assert '<button role=gridcell> "Day fourteen"' in snap
    assert '<div role=row> "Clickable row"' in snap
    assert "Dialog panel" not in snap


def test_every_editing_host_is_listed_and_typed_into(page) -> None:
    """``contenteditable`` with no value or ``plaintext-only`` is an editor like ``true``."""
    page.evaluate(
        """const box = document.createElement('div');
        box.id = 'editors';
        box.innerHTML = '<div contenteditable>Blank attribute editor</div>'
          + '<div contenteditable="plaintext-only" id="plain">Plaintext editor</div>'
          + '<div contenteditable="true"><input id="nested" placeholder="Nested field"'
          + ' value="old value"></div>'
          + '<div contenteditable="false">Read-only block</div>';
        document.body.appendChild(box);"""
    )
    snap = page.evaluate(SNAPSHOT_JS)
    assert '<div> "Blank attribute editor"' in snap
    assert "Read-only block" not in snap
    uid = _uid_of(snap, "Plaintext editor")
    page.evaluate(f"({TYPE_JS})({uid}, {json.dumps('typed plainly')})")
    assert page.evaluate("document.getElementById('plain').textContent") == "typed plainly"
    # A field inside an editor is a field: its value is replaced, not typed into.
    page.evaluate(f"({TYPE_JS})({_uid_of(snap, 'Nested field')}, {json.dumps('new value')})")
    assert page.evaluate("document.getElementById('nested').value") == "new value"
    page.evaluate("document.getElementById('editors').remove()")


def test_click_reports_an_element_whose_value_is_a_number(page) -> None:
    """The click's report does not fail once the click is made (``progress.value`` is a number)."""
    page.evaluate(
        """const bar = document.createElement('progress');
        bar.id = 'bar';
        bar.value = 0.5;
        bar.setAttribute('onclick', '');  // what makes the snapshot list it
        window.__bar_clicks = 0;
        bar.addEventListener('click', () => { window.__bar_clicks += 1; });
        document.body.appendChild(bar);"""
    )
    snap = page.evaluate(SNAPSHOT_JS)
    uid = _uid_of(snap, "<progress>")
    assert page.evaluate(f"({CLICK_JS})({uid})") == f'clicked [{uid}] <progress> "0.5"'
    assert page.evaluate("window.__bar_clicks") == 1
    page.evaluate("document.getElementById('bar').remove()")


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
          + '<input type="text" placeholder="Shadow field">'
          + '<div contenteditable="true">Shadow editor</div>';
        window.__shadow_clicked = false;
        root.getElementById('sh').addEventListener('click', () => {
          window.__shadow_clicked = true;
        });
        // What a framework rooted outside the component hears: a person's
        // pointer, mouse and input events all cross the shadow boundary.
        window.__outside = [];
        for (const t of ['pointerdown', 'mousedown', 'pointerup', 'mouseup', 'click', 'input']) {
          document.addEventListener(t, (e) => {
            if (e.target === host) window.__outside.push(t);
          });
        }"""
    )
    snap = page.evaluate(SNAPSHOT_JS)
    click_uid = _uid_of(snap, "Shadow action")
    type_uid = _uid_of(snap, "Shadow field")
    editor_uid = _uid_of(snap, "Shadow editor")
    page.evaluate(f"({CLICK_JS})({click_uid})")
    assert page.evaluate("window.__shadow_clicked") is True
    page.evaluate(f"({TYPE_JS})({type_uid}, {json.dumps('deep text')})")
    value = page.evaluate(
        "document.getElementById('shadow-host').shadowRoot.querySelector('input').value"
    )
    assert value == "deep text"
    assert page.evaluate("window.__outside") == [
        "pointerdown", "mousedown", "pointerup", "mouseup", "click", "input",
    ]
    page.evaluate(f"({TYPE_JS})({editor_uid}, {json.dumps('deep edit')})")
    edited = page.evaluate(
        "document.getElementById('shadow-host').shadowRoot.querySelector('div').textContent"
    )
    assert edited == "deep edit"
