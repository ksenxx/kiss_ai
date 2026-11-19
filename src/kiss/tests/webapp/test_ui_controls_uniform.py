# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Every surface draws its controls with one typeface, one type scale
and the page's own colour scheme.

Rendered in a real Chromium against the production remote server
(where browser defaults would leak through) and against the trajectory
visualizer:

* native inputs and buttons inherit the body typeface instead of the
  browser's Arial;
* the settings dialog's fields are body-sized and its section toggles
  (API Keys, Custom Models) share the history panel's Filters toggle;
* the history panel's collapsible chat headers are not bold;
* checkboxes take the accent, and ``color-scheme`` follows the theme so
  select popups and scrollbars are drawn dark on dark;
* every control wears one keyboard-focus ring, a full-width list row
  inside its box, and no button of the two side panels has its ring
  cut by the box that clips it;
* the send button gains no glow on hover;
* the visualizer paints in the same neutral palette, with plain
  headings and a neutral model role, and highlights code in the chat
  UI's syntax hues.
"""

from __future__ import annotations

import re
import threading
from pathlib import Path

from playwright.sync_api import Route, sync_playwright

import kiss.viz_trajectory as viz_pkg
from kiss.server.web_server import _LIGHT_THEME_CSS
from kiss.tests.conftest import goto_retrying_network_change
from kiss.tests.webapp.test_codex_task_panel_style import (
    _start_live_server,
)

VIZ_TEMPLATE = Path(viz_pkg.__file__).parent / "templates" / "index.html"
MEDIA = Path(__file__).resolve().parents[2] / "webapp" / "media"

# The files the visualizer links, as the jsDelivr CDN serves them: the
# highlight.js npm package ships its style sheets but no browser bundle
# (that one is in @highlightjs/cdn-assets).  The scripts are answered
# from the copies the chat UI ships (the same highlight.js, marked one
# patch on), the theme with rules of the real github-dark sheet.
_CDN = "https://cdn.jsdelivr.net/npm/"
_GITHUB_DARK_CSS = (
    "pre code.hljs{display:block;overflow-x:auto;padding:1em}"
    ".hljs{color:#c9d1d9;background:#0d1117}"
    ".hljs-doctag,.hljs-keyword,.hljs-meta .hljs-keyword,.hljs-template-tag,"
    ".hljs-template-variable,.hljs-type,.hljs-variable.language_{color:#ff7b72}"
    ".hljs-title,.hljs-title.class_,.hljs-title.class_.inherited__,"
    ".hljs-title.function_{color:#d2a8ff}"
    ".hljs-meta .hljs-string,.hljs-regexp,.hljs-string{color:#a5d6ff}"
    ".hljs-code,.hljs-comment,.hljs-formula{color:#8b949e}"
)
_CDN_FILES: dict[str, Path | str] = {
    _CDN + "marked@12.0.0/marked.min.js": MEDIA / "marked.min.js",
    _CDN + "@highlightjs/cdn-assets@11.9.0/highlight.min.js": MEDIA / "highlight.min.js",
    _CDN + "highlight.js@11.9.0/styles/github-dark.min.css": _GITHUB_DARK_CSS,
}


def _serve_cdn(route: Route) -> None:
    """Answer a request of the visualizer page offline: a CDN file the
    CDN has, or a failed request for anything else."""
    found = _CDN_FILES.get(route.request.url)
    if found is None:
        route.abort()
    elif isinstance(found, Path):
        route.fulfill(content_type="application/javascript", body=found.read_text("utf-8"))
    else:
        route.fulfill(content_type="text/css", body=found)


_PROBE_JS = """
() => {
  const cs = el => getComputedStyle(el);
  const body = document.body;
  const header = document.createElement('button');
  header.className = 'history-chat-header';
  header.textContent = 'chat';
  document.getElementById('sidebar').appendChild(header);
  const filters = document.getElementById('history-filters-toggle');
  const subpanel = document.querySelector('.config-subpanel-toggle');
  return {
    bodyFont: cs(body).fontFamily,
    searchFont: cs(document.getElementById('history-search')).fontFamily,
    closeFont: cs(document.getElementById('sidebar-close')).fontFamily,
    budgetFont: cs(document.getElementById('cfg-max-budget')).fontFamily,
    bodySize: cs(body).fontSize,
    budgetSize: cs(document.getElementById('cfg-max-budget')).fontSize,
    selectSize: cs(document.querySelector('.config-label select')).fontSize,
    headerWeight: cs(header).fontWeight,
    filters: [cs(filters).textTransform, cs(filters).fontSize,
              cs(filters).letterSpacing, cs(filters).color],
    subpanel: [cs(subpanel).textTransform, cs(subpanel).fontSize,
               cs(subpanel).letterSpacing, cs(subpanel).color],
    checkboxAccent: cs(document.getElementById('cfg-use-web-tools')).accentColor,
    linkColor: cs(document.querySelector('a') || body).color,
    scheme: cs(body).colorScheme,
  };
}
"""


# The keyboard-focus ring of a native button (no focus rule of its own),
# of a control that has one (the model button) and of a full-width list
# row (the chat header _PROBE_JS adds to the history panel); and the
# fill of the three resize handles under keyboard focus (the handle
# between the chat and the content pane shows only while the pane is
# open; its fill is read without its transition).
_RING_JS = """
() => {
  const ring = el => {
    el.focus();
    const s = getComputedStyle(el);
    return [el.matches(':focus-visible'), s.outlineStyle, s.outlineWidth,
            s.outlineColor, s.outlineOffset];
  };
  const fill = id => {
    const el = document.getElementById(id);
    el.style.transition = 'none';
    el.focus();
    const s = getComputedStyle(el);
    return [el.matches(':focus-visible'), s.outlineStyle, s.backgroundColor];
  };
  document.body.classList.add('content-pane-open');
  const handles = ['sidebar-resizer', 'meta-resizer', 'pane-resizer'].map(fill);
  document.body.classList.remove('content-pane-open');
  return {
    native: ring(document.getElementById('new-chat-btn')),
    styled: ring(document.getElementById('model-btn')),
    row: ring(document.querySelector('#sidebar > .history-chat-header')),
    handles,
  };
}
"""


# The buttons and the resize handles of the two side panels that are on
# screen in full while their keyboard-focus ring is not: each control is
# focused, and its box grown by the ring (the outline's offset and width)
# is held against the padding box of every ancestor that clips.  The
# right panel's sections are opened one after the other, so each shows
# its buttons.
_CUT_RINGS_JS = """
() => {
  const cut = new Set();
  const inside = (box, clip, grow) =>
    box.top - grow > clip.top - 0.05 && box.bottom + grow < clip.bottom + 0.05 &&
    box.left - grow > clip.left - 0.05 && box.right + grow < clip.right + 0.05;
  const measure = () => {
    for (const control of document.querySelectorAll(
        '#sidebar button, #meta-panel button, #sidebar-resizer, #meta-resizer')) {
      const box = control.getBoundingClientRect();
      if (!box.width) continue;
      control.focus({preventScroll: true});
      const style = getComputedStyle(control);
      const ring = style.outlineStyle === 'none' ? 0
        : parseFloat(style.outlineOffset) + parseFloat(style.outlineWidth);
      for (let el = control.parentElement; el; el = el.parentElement) {
        if (getComputedStyle(el).overflowY === 'visible') continue;
        const r = el.getBoundingClientRect();
        const clip = {top: r.top + el.clientTop, left: r.left + el.clientLeft};
        clip.bottom = clip.top + el.clientHeight;
        clip.right = clip.left + el.clientWidth;
        if (inside(box, clip, 0) && !inside(box, clip, ring)) {
          cut.add(control.id || control.className);
        }
      }
    }
  };
  measure();
  for (const toggle of document.querySelectorAll('#meta-panel .meta-section-toggle')) {
    toggle.click();
    measure();
  }
  return [...cut];
}
"""


def test_remote_page_controls_share_typeface_scale_and_scheme(tmp_path: Path) -> None:
    """Served remote page: inherited typeface, body-sized settings
    fields, shared section toggles, non-bold chat headers, accent
    checkboxes, dark/light colour scheme, one keyboard-focus ring, and
    no send-button glow."""
    ready = threading.Event()
    done = threading.Event()
    state: dict[str, object] = {}
    thread = threading.Thread(
        target=_start_live_server, args=(tmp_path, ready, done, state), daemon=True
    )
    thread.start()
    try:
        assert ready.wait(30), "RemoteAccessServer failed to start"
        startup_error = state.get("error")
        if isinstance(startup_error, BaseException):
            raise AssertionError("RemoteAccessServer startup failed") from startup_error
        port = state["port"]
        with sync_playwright() as p:
            browser = p.chromium.launch(args=["--ignore-certificate-errors"])
            try:
                page = browser.new_page(
                    ignore_https_errors=True, viewport={"width": 1400, "height": 900}
                )
                goto_retrying_network_change(
                    page, f"https://127.0.0.1:{port}/", wait_until="load"
                )
                page.wait_for_selector("#output", state="attached")
                # The light palette is the default; the toggle switches to
                # the dark palette.
                page.wait_for_function(
                    "() => document.body.classList.contains('light-theme')", timeout=10000
                )
                light = page.evaluate(_PROBE_JS)
                # A real key press leaves the page in keyboard modality, so
                # the controls focused next match :focus-visible.
                page.keyboard.press("Tab")
                rings = page.evaluate(_RING_JS)
                cut_rings = page.evaluate(_CUT_RINGS_JS)

                page.evaluate("() => document.getElementById('theme-btn').click()")
                page.wait_for_function(
                    "() => !document.body.classList.contains('light-theme')", timeout=10000
                )
                dark_scheme = page.evaluate("() => getComputedStyle(document.body).colorScheme")
                page.evaluate("() => document.getElementById('theme-btn').click()")
                page.wait_for_function(
                    "() => document.body.classList.contains('light-theme')", timeout=10000
                )

                # main.css's own send button (the remote sheet restyles it):
                # drop the remote class, hover, and read the shadow.
                page.evaluate("() => document.body.classList.remove('remote-chat')")
                page.hover("#send-btn")
                page.wait_for_timeout(400)
                send_shadow = page.evaluate(
                    "() => getComputedStyle(document.getElementById('send-btn')).boxShadow"
                )
            finally:
                browser.close()
    finally:
        done.set()
        thread.join(timeout=30)
    assert not thread.is_alive(), "RemoteAccessServer failed to stop"

    assert light["searchFont"] == light["bodyFont"], light
    assert light["closeFont"] == light["bodyFont"], light
    assert light["budgetFont"] == light["bodyFont"], light
    assert light["budgetSize"] == light["bodySize"], light
    assert light["selectSize"] == light["bodySize"], light
    assert light["headerWeight"] == "400", light
    assert light["filters"][0] == "none", light
    assert light["filters"] == light["subpanel"], light
    # The light palette's textLink.foreground drives the accent.
    accent = re.search(r"--theme-textLink-foreground: #([0-9a-f]{6});", _LIGHT_THEME_CSS)
    assert accent, "the light palette declares its accent"
    red, green, blue = (int(accent.group(1)[i : i + 2], 16) for i in (0, 2, 4))
    assert light["checkboxAccent"] == f"rgb({red}, {green}, {blue})", light
    assert light["scheme"] == "light", light
    # A native button wears the ring of a control with a focus rule of its
    # own, not the browser's `auto` ring; a row wears it inside its box.
    assert rings["native"] == rings["styled"], rings
    assert rings["native"][:3] == [True, "solid", "1px"], rings
    assert rings["native"][4] == "1px", rings
    assert rings["row"] == [*rings["native"][:4], "-1px"], rings
    # A resize handle is as tall as its panel, where a ring would be cut:
    # it shows keyboard focus by a fill in the ring's colour instead.
    assert rings["handles"] == [[True, "none", rings["native"][3]]] * 3, rings
    assert cut_rings == [], f"side-panel controls with a cut focus ring: {cut_rings}"
    assert dark_scheme == "dark"
    assert send_shadow == "none"


def test_visualizer_uses_the_shared_neutral_palette() -> None:
    """The trajectory visualizer: the chat UI's dark palette, headings
    and the model role in the body colour, a role's colour on its text
    only, sentence-case section labels, a neutral fill on the selected
    row, the chat UI's syntax hues without the CDN sheet, dark scheme."""
    html = VIZ_TEMPLATE.read_text(encoding="utf-8")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page(viewport={"width": 1200, "height": 800})
            page.route("**/*", lambda route: route.abort())
            page.set_content(html, wait_until="domcontentloaded")
            probe = page.evaluate(
                """
                () => {
                  const box = document.getElementById('messages-container');
                  box.innerHTML = '<div class="message">' +
                    '<span class="message-role model">model</span>' +
                    '<span class="message-role user">user</span>' +
                    '<div class="message-content"><h2>Title</h2><p>Body</p>' +
                    '<pre><code class="hljs"><span class="hljs-keyword">def</span>' +
                    '</code></pre></div></div>';
                  document.getElementById('jobs-list').innerHTML =
                    '<div class="list-item active">job</div>';
                  const cs = el => getComputedStyle(el);
                  const one = sel => cs(document.querySelector(sel));
                  const label = one('.sidebar-header h2');
                  const user = one('.message-role.user');
                  const row = one('.list-item.active');
                  return {
                    bodyBg: cs(document.body).backgroundColor,
                    bodyColor: cs(document.body).color,
                    modelRole: one('.message-role.model').color,
                    userRole: user.color,
                    roleFills: [user.backgroundColor, one('.message-role.model').backgroundColor],
                    roleBorder: user.borderTopWidth,
                    heading: one('.message-content h2').color,
                    label: [label.textTransform, label.letterSpacing, label.fontWeight],
                    row: [row.backgroundColor, row.borderTopWidth, row.boxShadow],
                    keyword: one('.hljs-keyword').color,
                    scheme: cs(document.documentElement).colorScheme,
                    sidebarBg: one('.jobs-sidebar').backgroundColor,
                  };
                }
                """
            )
        finally:
            browser.close()
    assert probe["bodyBg"] == "rgb(11, 11, 11)", probe
    assert probe["sidebarBg"] == "rgb(21, 21, 21)", probe
    assert probe["bodyColor"] == "rgb(240, 239, 236)", probe
    assert probe["modelRole"] == probe["bodyColor"], probe
    assert probe["heading"] == probe["bodyColor"], probe
    assert probe["userRole"] == "rgb(77, 159, 255)", probe
    assert probe["roleFills"][0] == probe["roleFills"][1], probe
    assert probe["roleBorder"] == "0px", probe
    assert probe["label"] == ["none", "normal", "500"], probe
    assert probe["row"] == ["rgba(255, 255, 255, 0.15)", "0px", "none"], probe
    assert probe["keyword"] == "rgb(255, 141, 119)", probe
    assert probe["scheme"] == "dark", probe


def test_visualizer_highlights_code_in_the_chat_ui_hues() -> None:
    """A message's fenced code block is highlighted by the highlight.js
    bundle the page links, in the chat UI's hues and on the visualizer's
    code surface, not in the colours of the linked github-dark sheet."""
    html = VIZ_TEMPLATE.read_text(encoding="utf-8")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page(viewport={"width": 1200, "height": 800})
            page.route("**/*", _serve_cdn)
            page.set_content(html, wait_until="load")
            probe = page.evaluate(
                """
                () => {
                  renderMessages([{role: 'model', content:
                    '```python\\ndef f():\\n    return "x"  # note\\n```'}]);
                  const colour = sel => {
                    const el = document.querySelector('.message-content pre ' + sel);
                    return el ? getComputedStyle(el).color : null;
                  };
                  const code = getComputedStyle(document.querySelector('pre code'));
                  return {
                    keyword: colour('.hljs-keyword'),
                    name: colour('.hljs-title.function_'),
                    string: colour('.hljs-string'),
                    comment: colour('.hljs-comment'),
                    code: [code.color, code.backgroundColor, code.paddingTop],
                    block: getComputedStyle(document.querySelector('pre')).backgroundColor,
                  };
                }
                """
            )
        finally:
            browser.close()
    assert probe == {
        "keyword": "rgb(255, 141, 119)",
        "name": "rgb(200, 166, 255)",
        "string": "rgb(229, 176, 126)",
        "comment": "rgb(139, 139, 129)",
        "code": ["rgb(240, 239, 236)", "rgba(0, 0, 0, 0)", "0px"],
        "block": "rgb(21, 21, 21)",
    }
