# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end cascade tests of the remote page's stylesheets.

These tests render the REAL page returned by
:func:`kiss.server.web_server._build_html` (``media/chat.html`` with
``main.css``, ``remote-codex.css``, the inlined palettes and
``brand.css``) in headless Chromium and read what the browser computes:

* The Machine row of the Filter and group menu ships ``hidden``
  (chat.html) and stays hidden until a remote machine is registered;
  the row's own ``display`` must not undo the attribute.
* The sign-in window's box wears the same frame as the password dialog
  and the server-reset dialog.
* The icons of a history row's menu read in the theme's secondary text
  step at rest and in its foreground under the pointer, a favourite in
  the theme's favourite colour and a copied button in the theme's ok
  colour.
* The settings dialog carries the theme's widget-border hairline and
  the large elevation.
* The Schedule / Machines inline form's Save button is the form's
  primary button, not the red button of the delete confirmation it
  shares a class with.
* The elements whose classes meet in main.js (a promptlet row, the
  prompt toast's field, a strip tab, the Stop button, the folder
  picker's buttons) render as the winning rules say.
* The states the remote skin restyles still show: the composer's
  pending question and dragged file, the voice button's recording and
  transcribing colours, the Filter and group button's fill under the
  pointer and while its menu is open and its dot while a filter is on,
  the Stop button's hover and a focused settings field.

The static server never answers the websocket, so main.js keeps the
page in its shipped state (no machine registered).  No mocks, patches
or fakes: a real HTTP server serves the real assets to a real browser.
"""

from __future__ import annotations

import functools
import http.server
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from playwright.sync_api import Browser, Page, sync_playwright

from kiss.server.web_server import MEDIA_DIR, _build_html
from kiss.tests.conftest import goto_retrying_network_change

# main.js reveals #app only after the websocket handshake, which never
# happens against a static server; an !important rule beats the veil's
# inline styles while every real CSS rule still applies.
_PREPARE_JS = """
() => {
  const style = document.createElement('style');
  style.textContent = `
    #app { display: flex !important; }
    #kiss-server-loading { display: none !important; }
    #auth-modal { display: none !important; }
  `;
  document.head.appendChild(style);
}
"""

# The frame of a dialog box, as computed.
_FRAME_JS = """
(sel) => {
  const cs = getComputedStyle(document.querySelector(sel));
  return {
    background: cs.backgroundColor,
    border: [cs.borderTopWidth, cs.borderTopStyle, cs.borderTopColor].join(' '),
    radius: cs.borderTopLeftRadius,
    shadow: cs.boxShadow,
  };
}
"""

# A chat's header with its fold button and a history row as
# renderHistory builds them (the row a `.sidebar-item .running-item`
# with its "…" button and that button's menu of action buttons, each
# holding its 12px icon), the row's menu open.
_ROW_BUILD_JS = """
() => {
  const icon = '<svg width="12" height="12"></svg>';
  const header = document.createElement('div');
  header.id = 'probe-header';
  header.className = 'history-chat-header';
  header.innerHTML =
    '<button class="sidebar-item-collapse history-chat-collapse">' + icon + '</button>';
  const row = document.createElement('div');
  row.id = 'probe-row';
  row.className = 'sidebar-item running-item';
  row.innerHTML =
    '<button class="sidebar-item-more">' + icon + '</button>' +
    '<div class="sidebar-item-actions" popover="auto">' +
    '<button class="sidebar-item-favorite">' + icon + '</button>' +
    '<button class="sidebar-item-favorite favorited">' + icon + '</button>' +
    '<button class="sidebar-item-pin">' + icon + '</button>' +
    '<button class="sidebar-item-pin pinned">' + icon + '</button>' +
    '<button class="sidebar-item-copy">' + icon + '</button>' +
    '<button class="sidebar-item-copy copied">' + icon + '</button>' +
    '</div>';
  document.getElementById('history-list').append(header, row);
  row.querySelector('.sidebar-item-actions').showPopover();
}
"""

# The colours of that row's buttons and of the header's fold button,
# next to the palette names resolved in the same place.
_ROW_COLOURS_JS = """
() => {
  const row = document.getElementById('probe-row');
  const probe = (value) => {
    const el = document.createElement('span');
    el.style.color = value;
    row.appendChild(el);
    const colour = getComputedStyle(el).color;
    el.remove();
    return colour;
  };
  const colour = (sel) => getComputedStyle(row.querySelector(sel)).color;
  return {
    fg: probe('var(--fg)'),
    dim: probe('var(--dim)'),
    faint: probe('var(--faint)'),
    favorite: probe('var(--favorite)'),
    ok: probe('var(--status-ok)'),
    favoriteBtn: colour('.sidebar-item-favorite:not(.favorited)'),
    favorited: colour('.sidebar-item-favorite.favorited'),
    pinBtn: colour('.sidebar-item-pin:not(.pinned)'),
    pinned: colour('.sidebar-item-pin.pinned'),
    copyBtn: colour('.sidebar-item-copy:not(.copied)'),
    copied: colour('.sidebar-item-copy.copied'),
    collapseBtn: getComputedStyle(
      document.querySelector('#probe-header > .sidebar-item-collapse')).color,
  };
}
"""

# The settings dialog's border and shadow next to the palette values
# they must resolve to.
_SETTINGS_FRAME_JS = """
() => {
  const panel = document.getElementById('settings-panel');
  const probe = document.createElement('div');
  probe.style.color = 'var(--border-widget)';
  probe.style.boxShadow = 'var(--shadow-lg)';
  panel.parentElement.appendChild(probe);
  const cs = getComputedStyle(panel);
  const ps = getComputedStyle(probe);
  const out = {
    border: [cs.borderTopWidth, cs.borderTopStyle, cs.borderTopColor].join(' '),
    shadow: cs.boxShadow,
    widgetBorder: ps.color,
    shadowLg: ps.boxShadow,
  };
  probe.remove();
  return out;
}
"""


def _serve(directory: Path) -> Iterator[str]:
    """Serve ``directory`` over HTTP on an ephemeral port; yield its URL."""
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler,
        directory=str(directory),
    )
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}/index.html"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


@pytest.fixture(scope="module")
def remote_url(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """Serve the real remote HTML page plus the real media assets."""
    root = tmp_path_factory.mktemp("remote-stylesheets")
    (root / "index.html").write_text(_build_html(), encoding="utf-8")
    (root / "media").symlink_to(MEDIA_DIR, target_is_directory=True)
    yield from _serve(root)


@pytest.fixture(scope="module")
def browser() -> Iterator[Browser]:
    """Launch one headless Chromium for the whole module."""
    with sync_playwright() as pw:
        chromium = pw.chromium.launch()
        try:
            yield chromium
        finally:
            chromium.close()


@pytest.fixture
def page(browser: Browser, remote_url: str) -> Iterator[Page]:
    """The remote page at a desktop width, #app shown."""
    context = browser.new_context(viewport={"width": 1280, "height": 900})
    pg = context.new_page()
    try:
        goto_retrying_network_change(pg, remote_url, wait_until="load")
        pg.wait_for_selector("#history-filter-btn", state="attached")
        pg.evaluate(_PREPARE_JS)
        yield pg
    finally:
        context.close()


# The rows of the Filter and group menu, by the setting they open.
_FILTER_ROW = '#history-filter-menu > .fg-row[data-fg="{}"]'.format

_MACHINE_ROW_JS = """
(sel) => {
  const row = document.querySelector(sel);
  const r = row.getBoundingClientRect();
  return {hidden: row.hidden, display: getComputedStyle(row).display,
          width: r.width, height: r.height};
}
"""


def _machine_row(page: Page) -> dict[str, Any]:
    """The Machine row's ``hidden`` flag, display and box."""
    row: dict[str, Any] = page.evaluate(_MACHINE_ROW_JS, _FILTER_ROW("machine"))
    return row


def test_machine_filter_stays_hidden_until_a_machine_is_registered(page: Page) -> None:
    """With the Filter and group menu open and no remote machine
    registered, the Machine row takes no room; once main.js clears
    ``hidden`` (a machine was registered) it lays out like the SEA row
    of the same menu."""
    page.click("#history-filter-btn")
    page.wait_for_selector("#history-filter-menu", state="visible")
    sea = page.locator(_FILTER_ROW("sea")).bounding_box()
    assert sea and sea["width"] > 0 and sea["height"] > 0, sea

    row = _machine_row(page)
    assert row["hidden"] is True, row
    assert row["display"] == "none", row
    assert row["width"] == 0 and row["height"] == 0, row

    page.locator(_FILTER_ROW("machine")).evaluate("row => { row.hidden = false; }")
    row = _machine_row(page)
    assert row["display"] != "none", row
    assert row["width"] > 0 and row["height"] == pytest.approx(sea["height"]), row


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_sign_in_box_matches_the_other_dialogs(page: Page, theme: str) -> None:
    """The sign-in window reuses the auth modal's box, so on the remote
    page it gets the same frame (background, hairline, radius, shadow)
    as the password dialog and the server-reset confirmation."""
    if theme == "dark":
        page.evaluate("() => document.body.classList.remove('light-theme')")
    signin = page.evaluate(_FRAME_JS, "#signin-modal .auth-modal-content")
    password = page.evaluate(_FRAME_JS, "#auth-modal .auth-modal-content")
    reset = page.evaluate(_FRAME_JS, ".server-reset-confirm-content")
    assert password == reset, (password, reset)
    assert signin == password, (signin, password)
    assert password["shadow"] != "none", password


# A frame of the daemon, handed to main.js the way transport.js does.
_DAEMON_SAYS_JS = "(ev) => window.dispatchEvent(new MessageEvent('message', {data: ev}))"

# Of the controls `sels` name, those that scrolling cannot bring inside
# the window and under the pointer.
_UNREACHABLE_JS = """
(sels) => sels.filter((sel) => {
  const el = document.querySelector(sel);
  el.scrollIntoView({block: 'nearest'});
  const r = el.getBoundingClientRect();
  const hit = document.elementFromPoint((r.left + r.right) / 2, (r.top + r.bottom) / 2);
  return r.top < 0 || r.bottom > innerHeight || !el.contains(hit);
})
"""


def test_sign_in_window_taller_than_the_screen_scrolls_to_its_buttons(
    page: Page, remote_url: str
) -> None:
    """With its paste row open the sign-in window is taller than a phone
    held sideways (568x320, the text three steps up at 16px).  Its box
    stays on screen and scrolls: the provider button at its top and
    "Not now" at its bottom can both be brought under the pointer, and
    "Not now" closes the window."""
    for _ in range(3):
        page.click('[data-font-step="1"]')
    page.set_viewport_size({"width": 568, "height": 320})
    page.evaluate(_DAEMON_SAYS_JS, {
        "type": "signin", "title": "Sign in",
        "providers": [{"id": "google", "label": "Login with Google"}],
    })
    with page.expect_popup():
        page.click("#signin-modal-buttons button")
    # The page to sign in on: one this server has.
    page.evaluate(_DAEMON_SAYS_JS, {
        "type": "signinUrl", "provider": "google",
        "url": remote_url.replace("index.html", "media/brand.css"),
    })
    page.wait_for_selector("#signin-modal-paste", state="visible")
    box = page.locator("#signin-modal .auth-modal-content").bounding_box()
    assert box and box["y"] >= 0 and box["y"] + box["height"] <= 320, box
    assert page.evaluate(_UNREACHABLE_JS, [
        "#signin-modal-buttons button", "#signin-modal-paste-input",
        "#signin-modal-paste-btn", "#signin-modal-later",
    ]) == []
    page.click("#signin-modal-later")
    page.wait_for_selector("#signin-modal", state="hidden")


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_history_row_buttons_use_the_theme_colours(page: Page, theme: str) -> None:
    """The fold button of a chat's header and the icons of a history
    row's menu read in the theme's secondary text step at rest, an
    icon in the foreground under the pointer; a favourite turns the
    favourite colour, a set pin the foreground and a copied button the
    ok colour."""
    _themed(page, theme)
    page.evaluate(_ROW_BUILD_JS)
    got = page.evaluate(_ROW_COLOURS_JS)
    for key in ("favoriteBtn", "pinBtn", "copyBtn", "collapseBtn"):
        assert got[key] == got["dim"], (key, got)
    assert got["favorited"] == got["favorite"], got
    assert got["pinned"] == got["fg"], got
    assert got["copied"] == got["ok"], got
    assert len({got["fg"], got["dim"], got["faint"], got["favorite"], got["ok"]}) == 5, got
    page.hover("#probe-row .sidebar-item-copy:not(.copied)")
    hovered = page.evaluate(_ROW_COLOURS_JS)
    assert hovered["copyBtn"] == got["fg"], hovered
    assert hovered["favoriteBtn"] == got["dim"], hovered


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_settings_dialog_has_the_widget_hairline_and_elevation(page: Page, theme: str) -> None:
    """The settings dialog is outlined by the theme's widget border and
    lifted by the large shadow on the remote page."""
    if theme == "dark":
        page.evaluate("() => document.body.classList.remove('light-theme')")
    got = page.evaluate(_SETTINGS_FRAME_JS)
    assert got["border"] == f"1px solid {got['widgetBorder']}", got
    assert got["shadow"] == got["shadowLg"] != "none", got


# A colour of the palette, resolved where the sidebar rows live.
_PALETTE_JS = """
(value) => {
  const el = document.createElement('span');
  el.style.color = value;
  document.getElementById('history-list').appendChild(el);
  const colour = getComputedStyle(el).color;
  el.remove();
  return colour;
}
"""

# A button's text, border and background colours as computed.
_BUTTON_JS = """
(sel) => {
  const cs = getComputedStyle(document.querySelector(sel));
  return {color: cs.color, border: cs.borderTopColor, background: cs.backgroundColor};
}
"""

# The Schedule / Machines inline form's Save button as main.js builds it
# (the delete confirmation's class for its shape plus its own) next to
# a delete confirmation's Yes button.
_INLINE_FORM_JS = """
() => {
  const wrap = document.createElement('div');
  wrap.innerHTML =
    '<div class="sidebar-inline-form"><div class="sidebar-inline-form-buttons">' +
    '<button id="probe-save" class="sidebar-confirm-yes sidebar-inline-form-save">' +
    'Save</button><button class="sidebar-confirm-no">Cancel</button></div></div>' +
    '<button id="probe-delete" class="sidebar-confirm-yes">Yes, delete</button>';
  document.getElementById('history-list').appendChild(wrap);
}
"""


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_inline_form_save_is_the_primary_button_not_the_delete_red(
    page: Page, theme: str
) -> None:
    """The inline form's Save button shares the delete confirmation's
    class for its shape, but it is not destructive: it is the form's
    primary button (the primary fill and text, the primary hover fill
    under the pointer), while "Yes, delete" keeps the error red on the
    raised surface and takes a faint red wash under the pointer."""
    _themed(page, theme)  # hover colours settle at once
    page.evaluate(_INLINE_FORM_JS)
    error = _colour(page, "var(--error)")
    primary = {
        "color": _colour(page, "var(--primary-fg)"),
        "border": "rgba(0, 0, 0, 0)",
        "background": _colour(page, "var(--primary-bg)"),
    }
    assert primary["color"] != error
    assert page.evaluate(_BUTTON_JS, "#probe-save") == primary
    destructive = {
        "color": error,
        "border": _colour(page, "color-mix(in srgb, var(--error) 45%, transparent)"),
        "background": _colour(page, "var(--surface2)"),
    }
    assert page.evaluate(_BUTTON_JS, "#probe-delete") == destructive
    page.hover("#probe-save")
    save = page.evaluate(_BUTTON_JS, "#probe-save")
    assert save == dict(primary, background=_colour(page, "var(--primary-hover-bg)")), save
    page.hover("#probe-delete")
    delete = page.evaluate(_BUTTON_JS, "#probe-delete")
    wash = _colour(page, "color-mix(in srgb, var(--error) 12%, transparent)")
    assert delete == dict(destructive, background=wash), delete
    # Cancel is the secondary button: its hover takes the neutral fill
    # and the stronger frame; a spent (disabled) delete does not react
    # to the pointer.
    page.hover(".sidebar-confirm-no")
    cancel = page.evaluate(_BUTTON_JS, ".sidebar-confirm-no")
    assert cancel == {
        "color": _colour(page, "var(--fg)"),
        "border": _colour(page, "var(--border-hover)"),
        "background": _colour(page, "var(--hover)"),
    }, cancel
    page.evaluate("() => { document.getElementById('probe-delete').disabled = true; }")
    spent = page.evaluate(_BUTTON_JS, "#probe-delete")
    page.hover("#probe-delete", force=True)
    assert page.evaluate(_BUTTON_JS, "#probe-delete") == spent


# Elements whose classes meet in main.js, as computed: a promptlet row
# (`sidebar-item tricks-item`, its editor `editing`) with its copy
# button, the prompt toast's field (`kiss-notification-input
# explorer-input`), a tab of the strip, the Stop button and the folder
# picker's buttons.
_MET_CLASSES_JS = """
() => {
  const wrap = document.createElement('div');
  wrap.innerHTML =
    '<div id="probe-trick" class="sidebar-item tricks-item">' +
    '<span class="sidebar-item-text">a promptlet</span>' +
    '<button id="probe-copy" class="sidebar-item-copy"></button></div>' +
    '<div id="probe-trick-edit" class="sidebar-item tricks-item editing"></div>' +
    '<div class="kiss-notification-field">' +
    '<input id="probe-prompt" class="kiss-notification-input explorer-input"></div>' +
    '<div class="folder-picker-actions">' +
    '<button id="probe-cancel" class="folder-picker-cancel">Cancel</button>' +
    '<button id="probe-select" class="folder-picker-select">Select Folder</button>' +
    '</div>' +
    '<span id="probe-primary" style="background: var(--primary-bg)"></span>' +
    '<span id="probe-raised" style="background: var(--surface2)"></span>';
  document.getElementById('history-list').appendChild(wrap);
  const tab = document.createElement('button');
  tab.className = 'chat-tab';
  tab.id = 'probe-tab';
  document.getElementById('tab-list').appendChild(tab);
  const cs = (sel) => getComputedStyle(document.querySelector(sel));
  const pick = (sel, props) => Object.fromEntries(props.map(p => [p, cs(sel)[p]]));
  const out = {
    trick: pick('#probe-trick', ['padding', 'cursor', 'borderRadius', 'whiteSpace', 'gap']),
    trickEditing: cs('#probe-trick-edit').cursor,
    prompt: cs('#probe-prompt').height,
    tab: cs('#probe-tab').padding,
    stop: cs('#stop-btn').borderTopStyle,
    select: pick('#probe-select', ['borderTopColor', 'backgroundColor', 'padding']),
    cancel: pick('#probe-cancel', ['borderTopColor', 'backgroundColor', 'padding']),
    primary: cs('#probe-primary').backgroundColor,
    raised: cs('#probe-raised').backgroundColor,
    copyCursor: cs('#probe-copy').cursor,
  };
  tab.remove();
  wrap.remove();
  return out;
}
"""


def test_met_classes_render_as_the_winning_rules_say(page: Page) -> None:
    """Where two classes meet on one element, the later equally specific
    rule wins: a promptlet row is laid out as a sidebar item, the prompt
    toast's field as an explorer input, the strip's tabs as the chat
    page's, the Stop button has no border, the folder picker's Select
    button takes the primary fill and its Cancel the raised secondary
    one, and a promptlet row's copy button keeps the pointer."""
    got = page.evaluate(_MET_CLASSES_JS)
    frame = page.evaluate(_PALETTE_JS, "var(--border-input)")
    assert got["trick"] == {
        "padding": "6px 10px 3.5px", "cursor": "pointer", "borderRadius": "6px",
        "whiteSpace": "nowrap", "gap": "6px",
    }, got
    assert got["trickEditing"] == "default", got
    assert got["prompt"] == "20px", got
    assert got["tab"] == "2px 8px", got
    assert got["stop"] == "none", got
    assert got["select"] == {
        "borderTopColor": "rgba(0, 0, 0, 0)", "backgroundColor": got["primary"],
        "padding": "6px 12px",
    }, got
    assert got["cancel"] == {
        "borderTopColor": frame, "backgroundColor": got["raised"], "padding": "6px 12px",
    }, got
    assert got["primary"] != got["raised"], got
    assert got["copyCursor"] == "pointer", got


# The settings dialog, open, with its restart confirmation open or not.
_RESTART_CONFIRM_JS = """
(open) => {
  document.getElementById('settings-panel').classList.add('open');
  document.getElementById('server-reset-confirm-modal').classList.toggle('open', open);
}
"""

# Where the dialog, its confirmation (scrim and box) and its Restart
# button are, and whether the confirmation covers the dialog's centre.
_RESTART_RECTS_JS = """
() => {
  const rect = (sel) => {
    const r = document.querySelector(sel).getBoundingClientRect();
    return [r.left, r.top, r.right, r.bottom].map(Math.round);
  };
  const panel = rect('#settings-panel');
  const modal = document.getElementById('server-reset-confirm-modal');
  const hit = document.elementFromPoint((panel[0] + panel[2]) / 2, (panel[1] + panel[3]) / 2);
  return {
    panel, scrim: rect('#server-reset-confirm-modal'),
    box: rect('.server-reset-confirm-content'), reset: rect('#cfg-server-reset-btn'),
    covered: modal.contains(hit),
  };
}
"""

# Two frames: a wheel's scroll is applied by the next frame.
_TWO_FRAMES_JS = "() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))"


def _within(inner: list[int], outer: list[int]) -> bool:
    return (
        inner[0] >= outer[0] - 1 and inner[1] >= outer[1] - 1
        and inner[2] <= outer[2] + 1 and inner[3] <= outer[3] + 1
    )


def test_restart_confirmation_stays_over_the_settings_while_they_scroll(page: Page) -> None:
    """In a short window the settings overflow their dialog.  The
    restart confirmation (a scrim and a box inside the dialog) covers
    the dialog's visible area, and a wheel over it does not carry it
    away: the settings under it do not scroll while it is open.  Once
    it is dismissed they scroll again."""
    page.set_viewport_size({"width": 1280, "height": 480})
    page.evaluate(_RESTART_CONFIRM_JS, True)
    before = page.evaluate(_RESTART_RECTS_JS)
    assert before["covered"], before
    assert _within(before["box"], before["panel"]), before
    assert _within(before["scrim"], before["panel"]), before
    page.mouse.move((before["panel"][0] + before["panel"][2]) / 2,
                    (before["panel"][1] + before["panel"][3]) / 2)
    page.mouse.wheel(0, 1500)
    page.evaluate(_TWO_FRAMES_JS)
    after = page.evaluate(_RESTART_RECTS_JS)
    assert after == before, (before, after)
    # Dismissed, the settings scroll under the wheel: Restart moves up.
    page.evaluate(_RESTART_CONFIRM_JS, False)
    page.mouse.wheel(0, 300)
    page.wait_for_function(
        "top => document.getElementById('cfg-server-reset-btn')"
        ".getBoundingClientRect().top < top - 100",
        arg=before["reset"][1],
    )


# The settings dialog open, its remote-password eye toggle
# (setupSecretInput, a positioned button in the scrolling section)
# scrolled up under the Close button; the point where the two overlap
# and what the browser hits there.
_CLOSE_OVER_PASSWORD_TOGGLE_JS = """
() => {
  document.getElementById('settings-panel').classList.add('open');
  const close = document.getElementById('settings-panel-close');
  const eye = document.getElementById('cfg-remote-password-toggle');
  const section = document.querySelector('#settings-panel > .sidebar-section');
  const c = close.getBoundingClientRect();
  const before = eye.getBoundingClientRect();
  section.scrollTop += (before.top + before.bottom - c.top - c.bottom) / 2;
  const e = eye.getBoundingClientRect();
  const left = Math.max(c.left, e.left), right = Math.min(c.right, e.right);
  const top = Math.max(c.top, e.top), bottom = Math.min(c.bottom, e.bottom);
  const x = (left + right) / 2, y = (top + bottom) / 2;
  const hit = document.elementFromPoint(x, y);
  return {
    x, y, overlap: right - left > 2 && bottom - top > 2,
    hitClose: close.contains(hit), hitEye: eye.contains(hit),
    hit: hit && (hit.id || hit.tagName),
  };
}
"""

_SETTINGS_STATE_JS = """
() => ({
  open: document.getElementById('settings-panel').classList.contains('open'),
  type: document.getElementById('cfg-remote-password').type,
})
"""


@pytest.mark.parametrize("size", [(1280, 600), (375, 667)])
def test_settings_close_button_stays_over_the_scrolled_password_toggle(
    page: Page, size: tuple[int, int]
) -> None:
    """The settings scroll under the dialog's Close button.  The
    remote-password eye toggle is a positioned button, so without a
    stacking level on Close it would paint over Close once scrolled
    under it, and the click meant to close the dialog would unmask the
    password instead.  Close wins the hit, and clicking it closes the
    dialog with the password still masked."""
    page.set_viewport_size({"width": size[0], "height": size[1]})
    got = page.evaluate(_CLOSE_OVER_PASSWORD_TOGGLE_JS)
    assert got["overlap"], got
    assert got["hitClose"] and not got["hitEye"], got
    page.mouse.click(got["x"], got["y"])
    assert page.evaluate(_SETTINGS_STATE_JS) == {"open": False, "type": "password"}


# The composer box's border colour, ring and background as computed.
_COMPOSER_JS = """
() => {
  const cs = getComputedStyle(document.getElementById('input-container'));
  return {border: cs.borderTopColor, ring: cs.boxShadow, background: cs.backgroundColor};
}
"""

# Add or remove a class of the element `sel` names.
_SET_CLASS_JS = "([sel, name, on]) => document.querySelector(sel).classList.toggle(name, on)"


def _themed(page: Page, theme: str) -> None:
    """Switch the page to `theme`, state colours settling at once.

    Motion is turned off first: the stylesheet's durations are 0 under
    ``prefers-reduced-motion`` only.  A frame drawn after the switch and
    before that started the switch's own colour transitions at their
    full length, and a colour read right then was still the light
    theme's.
    """
    page.emulate_media(reduced_motion="reduce")
    if theme == "dark":
        page.evaluate("() => document.body.classList.remove('light-theme')")


def _colour(page: Page, value: str) -> str:
    """The colour `value` computes to in the page's palette."""
    colour: str = page.evaluate(_PALETTE_JS, value)
    return colour


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_composer_shows_a_pending_question_and_a_dragged_file(page: Page, theme: str) -> None:
    """While an ask_user question waits for its answer the composer is
    outlined in red, focused or not (the answer is typed into it); a
    file dragged over it gives it the accent frame and wash, focused
    or not."""
    _themed(page, theme)
    rest = page.evaluate(_COMPOSER_JS)
    page.evaluate(_SET_CLASS_JS, ["body", "ask-answering", True])
    asking = page.evaluate(_COMPOSER_JS)
    assert asking["border"] == _colour(page, "color-mix(in srgb, var(--red) 60%, transparent)")
    ring = _colour(page, "color-mix(in srgb, var(--red) 30%, transparent)")
    assert asking["ring"] == f"{ring} 0px 0px 0px 1px" != rest["ring"], (rest, asking)
    page.focus("#task-input")
    assert page.evaluate(_COMPOSER_JS) == asking
    page.evaluate(_SET_CLASS_JS, ["body", "ask-answering", False])
    page.evaluate(_SET_CLASS_JS, ["#input-container", "drag-over", True])
    dragging = page.evaluate(_COMPOSER_JS)
    assert dragging["border"] == _colour(page, "var(--accent)"), dragging
    assert dragging["background"] != rest["background"], (rest, dragging)
    page.evaluate("() => document.activeElement.blur()")
    blurred = page.evaluate(_COMPOSER_JS)
    assert blurred["border"] == dragging["border"], blurred
    assert blurred["background"] == dragging["background"], blurred


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_voice_button_shows_recording_and_transcribing(page: Page, theme: str) -> None:
    """A listening voice button is filled and reads in the strong
    foreground; while a round records it turns the theme's red and
    while it transcribes the warning colour, the fill unchanged."""
    _themed(page, theme)
    page.evaluate(_SET_CLASS_JS, ["#voice-btn", "active", True])
    listening = page.evaluate(_BUTTON_JS, "#voice-btn")
    assert listening["color"] == _colour(page, "var(--fg-strong)"), listening
    rounds = (("voice-triggered", "var(--red)"), ("voice-transcribing", "var(--warning)"))
    for state, colour in rounds:
        page.evaluate(_SET_CLASS_JS, ["#voice-btn", state, True])
        got = page.evaluate(_BUTTON_JS, "#voice-btn")
        assert got["color"] == _colour(page, colour) != listening["color"], (state, got)
        assert got["background"] == listening["background"], (state, got)
        # A round outlives the listening it started in (trigger turned off).
        page.evaluate(_SET_CLASS_JS, ["#voice-btn", "active", False])
        assert page.evaluate(_BUTTON_JS, "#voice-btn")["color"] == got["color"], state
        page.evaluate(_SET_CLASS_JS, ["#voice-btn", "active", True])
        page.evaluate(_SET_CLASS_JS, ["#voice-btn", state, False])


# The dot the Filter and group button wears while a filter is on.
_DOT_JS = """
(sel) => {
  const cs = getComputedStyle(document.querySelector(sel), '::after');
  return {content: cs.content, background: cs.backgroundColor, width: cs.width};
}
"""


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_filter_button_fills_on_hover_and_while_open_and_wears_a_dot(
    page: Page, theme: str
) -> None:
    """The Filter and group button is a ghost icon button: no fill at
    rest, the icon-button wash under the pointer and, pointer or not,
    while its menu is open.  Once an option of the menu narrows the list
    it wears a dot in the accent colour, which stays when the menu is
    closed and the button is back at rest."""
    _themed(page, theme)
    button = "#history-filter-btn"
    rest = page.evaluate(_BUTTON_JS, button)
    assert rest["background"] == "rgba(0, 0, 0, 0)", rest
    assert rest["color"] == _colour(page, "var(--dim)"), rest
    assert page.evaluate(_DOT_JS, button)["content"] == "none"
    page.hover(button)
    hovered = page.evaluate(_BUTTON_JS, button)
    assert hovered["background"] == _colour(page, "var(--hover-strong)"), hovered
    assert hovered["color"] == _colour(page, "var(--fg)"), hovered

    page.click(button)
    page.wait_for_selector("#history-filter-menu", state="visible")
    page.mouse.move(0, 0)
    assert page.evaluate(_BUTTON_JS, button) == hovered

    page.click(_FILTER_ROW("status"))
    page.click("#fg-status > .fg-option >> nth=0")
    dot = page.evaluate(_DOT_JS, button)
    assert dot == {"content": '""', "background": _colour(page, "var(--accent)"), "width": "6px"}
    page.click(button)
    page.wait_for_selector("#history-filter-menu", state="hidden")
    page.mouse.move(0, 0)
    assert page.evaluate(_BUTTON_JS, button) == rest
    assert page.evaluate(_DOT_JS, button) == dot


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_stop_button_takes_the_send_hover_fill(page: Page, theme: str) -> None:
    """The Stop button (shown while a task runs) is the Send button's
    filled circle, and under the pointer it takes Send's hover fill."""
    _themed(page, theme)
    page.evaluate("() => { document.getElementById('stop-btn').style.display = 'flex'; }")
    rest = page.evaluate(_BUTTON_JS, "#stop-btn")
    page.hover("#stop-btn")
    hover = page.evaluate(_BUTTON_JS, "#stop-btn")
    assert hover["background"] == _colour(page, "var(--send-hover-bg)") != rest["background"]
    assert hover["color"] == rest["color"], (rest, hover)


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_focused_settings_field_takes_the_accent_border(page: Page, theme: str) -> None:
    """A settings field is framed by the theme's input border; focused,
    the frame turns the accent like its outline."""
    _themed(page, theme)
    page.evaluate(_SET_CLASS_JS, ["#settings-panel", "open", True])
    page.click("#custom-models-toggle")  # the textarea's section starts folded
    for field in ("#cfg-max-budget", "#cfg-voice-auto-submit", "#cfg-custom-headers"):
        frame = _colour(page, "var(--border-input)")
        assert page.evaluate(_BUTTON_JS, field)["border"] == frame, field
        page.focus(field)
        assert page.evaluate(_BUTTON_JS, field)["border"] == _colour(page, "var(--accent)"), field


# The promptlets panel, open, holding a row and an open in-place editor
# as main.js builds them.
_PROMPTLET_ROWS_JS = """
() => {
  document.getElementById('tricks-panel').classList.add('open');
  document.getElementById('tricks-list').innerHTML =
    '<div id="probe-row" class="sidebar-item tricks-item">' +
    '<span class="sidebar-item-text">a promptlet</span></div>' +
    '<div id="probe-editor" class="sidebar-item tricks-item editing">' +
    '<textarea class="tricks-edit-input"></textarea></div>';
}
"""


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_open_promptlet_editor_does_not_wash_under_the_pointer(page: Page, theme: str) -> None:
    """A promptlet row is washed under the pointer (a click injects it);
    the row holding its open editor is not."""
    _themed(page, theme)
    page.evaluate(_PROMPTLET_ROWS_JS)
    rest = page.evaluate(_BUTTON_JS, "#probe-editor")["background"]
    page.hover("#probe-row")
    assert page.evaluate(_BUTTON_JS, "#probe-row")["background"] == _colour(page, "var(--hover)")
    page.hover("#probe-editor")
    assert page.evaluate(_BUTTON_JS, "#probe-editor")["background"] == rest
