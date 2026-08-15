# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration test: the settings action buttons sit on one line.

Regression: the settings action buttons once lacked
``white-space: nowrap`` and could shrink until multi-word labels wrapped
onto a second line, making the row 46px tall with ragged, half-height
neighbours.  The current row contains "Tips", "Git Commit", "Update",
and "Reset Server".

The fix has two halves, both exercised here:

* ``remote-codex.css`` widens ``--settings-panel-w`` past the one-line
  requirement, so on desktop the row fits without wrapping at all.
* ``main.css`` marks the buttons ``white-space: nowrap; flex: 0 0 auto``
  and lets ``.config-update-row`` ``flex-wrap``, so a label can never
  break mid-text — a too-narrow panel (the 90vw mobile sheet) pushes
  whole buttons onto the next line instead.

Everything is measured in a real headless Chromium against the real
``media/main.css``, the real ``media/remote-codex.css`` and the real
``<div id="settings-panel">`` markup from ``media/chat.html``, under the
same ``--vscode-*`` variables that ``web_server._build_html()`` injects
into the remote page.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from playwright.sync_api import ViewportSize, sync_playwright

_MEDIA_DIR = (
    Path(__file__).resolve().parents[4] / "kiss" / "agents" / "vscode" / "media"
)
_MAIN_CSS = _MEDIA_DIR / "main.css"
_REMOTE_CSS = _MEDIA_DIR / "remote-codex.css"
_HTML = _MEDIA_DIR / "chat.html"

#: Ids of the four buttons of ``.config-update-row``, in DOM order.
_BUTTON_IDS = (
    "tips-btn",
    "autocommit-btn",
    "cfg-update-btn",
    "cfg-server-reset-btn",
)

#: Desktop mode only applies at >= 900px wide (see remote-codex.css).
_DESKTOP_VIEWPORT = ViewportSize(width=1440, height=900)

#: A phone, where the drawer falls back to the responsive 90vw mobile sheet.
_PHONE_VIEWPORT = ViewportSize(width=390, height=844)


def _extract_settings_panel_markup() -> str:
    """Return the literal ``<div id="settings-panel">…</div>`` block.

    Balances ``<div>`` / ``</div>`` pairs in the shipped ``chat.html``
    from the panel's opening tag to its matching close tag so the test
    lays out the very same DOM the user sees.
    """
    src = _HTML.read_text(encoding="utf-8")
    match = re.search(r'<div id="settings-panel">', src)
    assert match, "settings-panel div not found in chat.html"
    start = match.start()
    i = match.end()
    depth = 1
    div_open = re.compile(r"<div\b", re.IGNORECASE)
    div_close = re.compile(r"</div>", re.IGNORECASE)
    while depth > 0:
        nxt_open = div_open.search(src, i)
        nxt_close = div_close.search(src, i)
        assert nxt_close, "unbalanced <div> for settings-panel"
        if nxt_open and nxt_open.start() < nxt_close.start():
            depth += 1
            i = nxt_open.end()
        else:
            depth -= 1
            i = nxt_close.end()
    return src[start:i].replace("{{VERSION_SUFFIX}}", "")


def _build_remote_page(body_class: str = "remote-chat remote-desktop") -> str:
    """Build the standalone remote page that renders the settings drawer.

    Inlines both real stylesheets in load order, puts *body_class* on
    ``<body>`` exactly like the remote webapp does
    (``remote-chat remote-desktop`` for the docked desktop layout,
    plain ``remote-chat`` for the 90vw mobile sheet), and re-declares
    the ``--vscode-*`` variables (notably the 16px base font that all
    ``rem`` sizes derive from) with the same values
    ``web_server._build_html()`` injects.  No JavaScript beyond opening
    the drawer, so the measurements isolate layout from runtime
    behaviour.

    Args:
        body_class: Class attribute for ``<body>``.

    Returns:
        The complete HTML string.
    """
    panel = _extract_settings_panel_markup()
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <style>
    :root {{
      --vscode-font-size: 16px;
      --vscode-font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI',
        Roboto, 'Helvetica Neue', Arial, sans-serif;
      --vscode-editor-font-size: 16px;
      --vscode-editor-background: #1e1e1e;
      --vscode-editor-foreground: #cccccc;
      --vscode-input-background: #3c3c3c;
      --vscode-input-foreground: #cccccc;
      --vscode-input-border: #3c3c3c;
      --vscode-sideBar-background: #252526;
      --vscode-panel-border: #80808059;
      --vscode-descriptionForeground: #8b8b8b;
      --vscode-textLink-foreground: #3794ff;
    }}
    html, body {{ height: 100%; margin: 0; padding: 0; }}
  </style>
  <style>{_MAIN_CSS.read_text(encoding="utf-8")}</style>
  <style>{_REMOTE_CSS.read_text(encoding="utf-8")}</style>
  <title>settings action row layout test</title>
</head>
<body class="{body_class}">
  <div id="app"></div>
  {panel}
  <script>
    document.getElementById('settings-panel').classList.add('open');
  </script>
</body>
</html>
"""


_MEASURE_JS = """
(ids) => {
  const panel = document.getElementById('settings-panel');
  const panelStyle = getComputedStyle(panel);
  const panelRect = panel.getBoundingClientRect();
  const buttons = ids.map((id) => {
    const el = document.getElementById(id);
    const label = el.querySelector('span');
    const style = getComputedStyle(el);
    const rect = el.getBoundingClientRect();
    return {
      id: id,
      text: label.textContent.trim(),
      top: Math.round(rect.top),
      right: rect.right,
      height: rect.height,
      whiteSpace: style.whiteSpace,
      // A label that fits on one line reports the same scroll and
      // client width; a wrapped one is taller than one line box.
      labelScrollWidth: label.scrollWidth,
      labelClientWidth: label.clientWidth,
      labelHeight: label.getBoundingClientRect().height,
      labelLineHeight: parseFloat(getComputedStyle(label).lineHeight) ||
        label.getBoundingClientRect().height,
    };
  });
  return {
    panelWidth: panelRect.width,
    panelContentRight: panelRect.right - parseFloat(panelStyle.paddingRight),
    rowHeight: document
      .querySelector('.config-update-row')
      .getBoundingClientRect().height,
    buttons: buttons,
  };
}
"""


def _measure_action_row(viewport: ViewportSize, body_class: str) -> dict:
    """Lay the settings drawer out in Chromium and measure its action row.

    Args:
        viewport: Size of the browser viewport to render at.
        body_class: Class attribute for ``<body>``, which selects the
            desktop or the mobile layout.

    Returns:
        The drawer geometry plus a per-button record for each of the
        four action buttons (see ``_MEASURE_JS``).
    """
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            context = browser.new_context(viewport=viewport)
            page = context.new_page()
            page.set_content(_build_remote_page(body_class), wait_until="load")
            page.wait_for_selector("#cfg-server-reset-btn")
            result = page.evaluate(_MEASURE_JS, list(_BUTTON_IDS))
            context.close()
        finally:
            browser.close()
    assert isinstance(result, dict)
    return result


@pytest.fixture(scope="module")
def _measurements():
    """Measure the action row in the docked desktop settings drawer."""
    return _measure_action_row(_DESKTOP_VIEWPORT, "remote-chat remote-desktop")


@pytest.fixture(scope="module")
def _phone_measurements():
    """Measure the action row in the narrow 90vw mobile settings sheet."""
    return _measure_action_row(_PHONE_VIEWPORT, "remote-chat")


def test_all_four_action_buttons_share_one_line(_measurements) -> None:
    """Tips / Git Commit / Update / Reset Server must share a row.

    Every button's top edge has to land at the same y; a wrapped row
    would push the later buttons down.
    """
    tops = {b["id"]: b["top"] for b in _measurements["buttons"]}
    assert len(set(tops.values())) == 1, (
        "The settings action buttons are not on the same line — top "
        f"edges: {tops}.  Panel width is "
        f"{_measurements['panelWidth']}px; widen --settings-panel-w."
    )


def test_action_button_labels_do_not_wrap(_measurements) -> None:
    """No button label may break onto a second line.

    Checks both the declared ``white-space`` and the rendered geometry:
    a wrapped label overflows its box horizontally and grows past a
    single line box vertically.
    """
    for button in _measurements["buttons"]:
        assert button["whiteSpace"] == "nowrap", (
            f"{button['text']!r} allows wrapping "
            f"(white-space: {button['whiteSpace']}) — its label can "
            "break mid-text when the panel is narrow."
        )
        assert button["labelScrollWidth"] <= button["labelClientWidth"], (
            f"{button['text']!r} is squeezed: label needs "
            f"{button['labelScrollWidth']}px but only has "
            f"{button['labelClientWidth']}px."
        )
        assert button["labelHeight"] <= button["labelLineHeight"] * 1.5, (
            f"{button['text']!r} wrapped onto a second line: label is "
            f"{button['labelHeight']}px tall for a "
            f"{button['labelLineHeight']}px line box."
        )


def test_action_row_fits_inside_the_desktop_drawer(_measurements) -> None:
    """The row must fit the drawer's content box, not overflow it.

    Guards the other half of the fix: ``--settings-panel-w`` has to be
    wide enough that nowrap buttons stay inside the 16px padding
    instead of spilling past the panel's edge.
    """
    last = _measurements["buttons"][-1]
    assert last["right"] <= _measurements["panelContentRight"] + 0.5, (
        f"{last['text']!r} ends at x={last['right']:.1f} but the "
        f"drawer's content box ends at "
        f"x={_measurements['panelContentRight']:.1f} — the button row "
        "overflows the settings panel."
    )


def test_action_row_is_a_single_button_tall(_measurements) -> None:
    """The row must be exactly one uniform button tall.

    With wrapped labels the row grew to 46px and its buttons ended up
    ragged (30px for the one-word labels, 46px for the two-word ones),
    so this checks both that the buttons agree on a height and that the
    row is no taller than one of them.
    """
    heights = {b["text"]: b["height"] for b in _measurements["buttons"]}
    shortest = min(heights.values())
    assert max(heights.values()) == pytest.approx(shortest, abs=0.5), (
        f"The action buttons have ragged heights {heights} — the taller "
        "ones are two text lines tall."
    )
    assert _measurements["rowHeight"] == pytest.approx(shortest, abs=0.5), (
        f"The action row is {_measurements['rowHeight']}px tall but a "
        f"single-line button is {shortest}px — the row spans more than "
        "one line."
    )


def test_narrow_sheet_keeps_buttons_inside_and_labels_unwrapped(
    _phone_measurements,
) -> None:
    """On a phone, labels stay intact and every button remains contained.

    ``main.css`` is shared with the 90vw mobile sheet and the VS Code
    sidebar.  Depending on available font metrics, ``flex-wrap`` may
    keep the three buttons on one row or move whole buttons to another;
    it must never wrap inside a label or overflow the panel.
    """
    buttons = _phone_measurements["buttons"]
    for button in buttons:
        assert button["labelScrollWidth"] <= button["labelClientWidth"], (
            f"{button['text']!r} is squeezed on a narrow sheet: label "
            f"needs {button['labelScrollWidth']}px but has "
            f"{button['labelClientWidth']}px — it wrapped mid-text "
            "instead of moving to the next line."
        )
        assert button["labelHeight"] <= button["labelLineHeight"] * 1.5, (
            f"{button['text']!r} wrapped onto a second text line on a "
            f"narrow sheet: label is {button['labelHeight']}px tall for "
            f"a {button['labelLineHeight']}px line box."
        )
        assert button["right"] <= _phone_measurements["panelContentRight"] + 0.5, (
            f"{button['text']!r} ends at x={button['right']:.1f}, past "
            "the sheet's content box at "
            f"x={_phone_measurements['panelContentRight']:.1f} — nowrap "
            "buttons must wrap, not overflow."
        )
