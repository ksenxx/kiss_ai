# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E tests: remote-webapp task-panel-matched typography + history rows.

Features on the remote webapp (served by ``RemoteAccessServer``):

1. The inline task panel (``.ev.task-panel``, the first event of a
   task's thread) inherits main.css's look verbatim (the page
   foreground over the user bubble's neutral wash, with no outline;
   the remote page merely swaps the palette variables), its text the
   transcript's 14px reading size (``--fs-lg`` of the page's injected
   13px root size).  The event panels likewise keep main.css's
   typography — a contract pinned end to end by
   ``test_remote_panels.py``.
2. History rows (``.running-item``) are neutral on every surface and
   in both views: no pastel background, no coloured border, no
   per-chat ``--task-color`` custom property, no inline colours.  The
   legacy flat list (``#history-list.legacy-view``, Group by: None in
   the Filter and group menu, whose button sits right of the search
   box) is the same flat rows, newest first.  The chat-panel headers
   of the grouped view are plain rows on the list background (the
   neutral ``--hover`` fill only under the pointer), show one line of
   text and carry no tooltip.
3. A chat header and a task panel show ONE line of text, cut with an
   ellipsis, that ends in the row's buttons (a header's fold chevron
   and the "…" button, a task panel's "…" button alone); the row's
   actions are the lines of the "…" button's menu.  A task panel has
   no fold button and nothing hangs under its text: a task's metadata
   is in the Task Info panel.

Static (regex) tests pin the CSS/JS wiring; the live tests boot the
production ``RemoteAccessServer`` + headless Chromium and assert the
COMPUTED styles and the real layout boxes.
"""

from __future__ import annotations

import asyncio
import re
import threading
from pathlib import Path
from typing import Any

import pytest
from playwright.sync_api import Locator, expect, sync_playwright
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from kiss.agents.sorcar.machines import local_machine_name
from kiss.tests.conftest import goto_retrying_network_change
from kiss.tests.webapp.test_history_filter_menu import pick_filter_option

MEDIA_DIR = Path(__file__).resolve().parents[2] / "webapp" / "media"
CODEX_CSS = MEDIA_DIR / "remote-codex.css"
MAIN_CSS = MEDIA_DIR / "main.css"
MAIN_JS = MEDIA_DIR / "main.js"


def test_main_js_history_rows_carry_no_colour() -> None:
    """renderHistory writes no inline colours on history rows in either
    view: no background, no text colour, no per-chat ``--task-color``."""
    js = MAIN_JS.read_text(encoding="utf-8")
    start = js.index("function renderHistory(")
    # The function ends where the next one of its nesting level begins.
    following = re.search(r"\n  function \w+\(", js[start:])
    assert following is not None, "a function follows renderHistory in main.js"
    body = js[start : start + following.start()]
    assert "running-item" in body, "the slice is the function that builds the rows"
    assert "style.backgroundColor" not in body, (
        "renderHistory must not set an inline background color on "
        "history rows (inline styles beat every stylesheet)"
    )
    assert "style.color" not in body, (
        "renderHistory must not set an inline text color on history rows"
    )
    assert "--task-color" not in js, "no per-chat colour is stamped anywhere"
    assert "chatIdBgColor" not in js, "the per-chat hue hash is gone"


def test_main_css_page_rows_are_neutral() -> None:
    """main.css's task panels are flat: transparent until hovered,
    theme foreground text, no per-chat colour bar in either view."""
    css = MAIN_CSS.read_text(encoding="utf-8")
    # A task panel is a `sidebar-item running-item` row (main.js): the
    # base rule paints it transparent in the theme foreground, and the
    # .running-item rule must not restate or override either.
    base = re.search(r"\n\.sidebar-item\s*\{([^}]*)\}", css)
    assert base, ".sidebar-item rule missing from main.css"
    assert "background: transparent" in base.group(1), base.group(1)
    assert "color: var(--fg)" in base.group(1), base.group(1)
    m = re.search(r"\n\.running-item\s*\{([^}]*)\}", css)
    assert m, ".running-item rule missing from main.css"
    rule = m.group(1)
    assert "background" not in rule, (
        f".running-item must paint no background of its own; got: {rule!r}"
    )
    assert not re.search(r"(?<![-\w])color:", rule), (
        f".running-item must keep the theme foreground; got: {rule!r}"
    )
    assert "--task-color" not in css, "main.css reads no per-chat colour"
    assert "legacy-view" not in css, "the legacy flat list has no styling of its own"


def test_main_css_chat_header_neutral_single_line() -> None:
    """The grouped view's chat-panel header is a plain row on the list
    background (the faint neutral --hover fill only under the pointer,
    never an accent band) and shows its title on one ellipsized line."""
    css = MAIN_CSS.read_text(encoding="utf-8")
    header = re.search(r"\n\.history-chat-header\s*\{([^}]*)\}", css)
    assert header, ".history-chat-header rule missing"
    assert "background: transparent" in header.group(1), header.group(1)
    assert "var(--accent)" not in header.group(1), header.group(1)
    hover = re.search(r"\n\.history-chat-header:hover\s*\{([^}]*)\}", css)
    assert hover, ".history-chat-header:hover rule missing"
    assert hover.group(1).strip() == "background-color: var(--hover);", (
        f"the hover background must be the neutral --hover fill; got: {hover.group(1)!r}"
    )
    # --hover is the theme's list.hoverBackground; its dark fallback is
    # white at a faint alpha (two hex digits after #ffffff).
    fill = re.search(
        r"--hover:\s*var\(--theme-list-hoverBackground,\s*#ffffff([0-9a-f]{2})\);", css
    )
    assert fill, "--hover must be list.hoverBackground with a translucent white fallback"
    assert 3 <= int(fill.group(1), 16) / 255 * 100 <= 10, (
        f"the hover fill is not faint: {fill.group(0)}"
    )
    title = re.search(r"\n\.history-chat-title\s*\{([^}]*)\}", css)
    assert title, ".history-chat-title rule missing"
    assert "white-space: nowrap" in title.group(1)
    assert "text-overflow: ellipsis" in title.group(1)
    assert "line-clamp" not in title.group(1)


def _hue_of(rgba: str) -> float:
    """The HSL hue (degrees) of a computed ``rgb(...)`` / ``rgba(...)``
    color string."""
    nums = re.findall(r"[\d.]+", rgba)
    r, g, b = (float(n) / 255 for n in nums[:3])
    hi, lo = max(r, g, b), min(r, g, b)
    if hi == lo:
        return 0.0
    d = hi - lo
    if hi == r:
        h = ((g - b) / d) % 6
    elif hi == g:
        h = (b - r) / d + 2
    else:
        h = (r - g) / d + 4
    return h * 60


def _rgb_of(color: str) -> list[float]:
    """The RGB channels (0..255) of a computed ``rgb(...)`` /
    ``rgba(...)`` / ``color(srgb ...)`` color string."""
    nums = [float(n) for n in re.findall(r"[\d.]+", color.replace("srgb", ""))[:3]]
    return [n * 255 for n in nums] if "srgb" in color else nums


def _chroma_of(color: str) -> float:
    """The spread between the largest and smallest RGB channel (0..255)
    of a computed color string; 0 for a pure grey."""
    return max(_rgb_of(color)) - min(_rgb_of(color))


def _alpha_of(rgba: str) -> float:
    """The alpha (0..1) of a computed ``rgb(...)`` / ``rgba(...)`` /
    ``color(srgb ...)`` color string; 1 when the color is opaque."""
    nums = re.findall(r"[\d.]+", rgba.replace("srgb", ""))
    return float(nums[3]) if len(nums) > 3 else 1.0


# The remote palette's selection fill (remote-codex.css maps every
# semantic name onto a --theme-* variable with the dark palette's value
# as the fallback).
SELECTED = "var(--theme-list-inactiveSelectionBackground, #ffffff26)"


def test_remote_history_row_is_neutral() -> None:
    """The remote page adds no row fill of its own: the rows are the
    flat main.css rows (transparent, editor foreground), lit only by the
    remote hover colour under the pointer."""
    codex_css = CODEX_CSS.read_text(encoding="utf-8")
    assert re.search(r"body\.remote-chat \.running-item\s*\{", codex_css) is None, (
        "remote-codex.css must not restyle .running-item (no selection fill)"
    )
    assert f"background-color: {SELECTED}" not in codex_css, (
        "no history row is painted with list.inactiveSelectionBackground"
    )
    assert "--task-color" not in codex_css, (
        "remote-codex.css must not reference --task-color anywhere"
    )


_INJECT_PAGE_JS = r"""
(() => {
  const out = document.getElementById('output');
  // The transcript goes into a tab the daemon's tab registry knows
  // (createNewTab registers it with openTab; the page's initial tab
  // is registered only once a task runs in it).  The chat-panel
  // expand below opens the injected chat in another registered tab,
  // and the daemon's next tabs_state snapshot prunes every tab it
  // does not list -- an unregistered transcript tab vanished with
  // its injected panels whenever that snapshot landed before the
  // probes (it does once the daemon handshake is quick, e.g. late in
  // a warmed-up pytest process).
  window._testApi.createNewTab();
  const welcome = document.getElementById('welcome');
  if (welcome) welcome.style.display = 'none';
  const app = document.getElementById('app');
  if (app) app.style.display = '';
  const loading = document.getElementById('kiss-server-loading');
  if (loading) loading.style.display = 'none';

  // Open the transcript with its task panel (the typography
  // reference), exactly as the daemon's setTaskText + clear pair
  // starts a task.
  // The chat is marked running like a real task's chat: a Chats-panel
  // click below opens another chat, and the chat left behind is
  // retired unless it is busy -- an idle transcript would vanish
  // with its injected panels.
  const tabId = window._testApi.getActiveTabId();
  for (const ev of [
    {type: 'setTaskText', tabId, text: 'Fix the flux capacitor'},
    {type: 'clear', tabId},
    {type: 'status', tabId, running: true},
  ]) {
    window.dispatchEvent(new MessageEvent('message', {data: ev}));
  }
  if (!out.querySelector(':scope > .task-panel > .task-panel-text')) {
    throw new Error('the transcript did not open with its task panel');
  }

  out.insertAdjacentHTML('beforeend', `
    <div class="think">Reasoning about the task panel type.</div>
    <div class="ev txt md-body">Plain assistant text with
      <code>inline code</code> and a table.
      <pre><code class="hljs language-python">print("x")</code></pre>
      <table><tr><th>h</th></tr><tr><td>cell</td></tr></table>
    </div>
    <div class="ev tc tc-bash">
      <div class="tc-h tc-h-bash collapse-header">
        <span class="collapse-chv">\u25BE</span>Bash</div>
      <div class="tc-b">
        <div class="tc-arg"><span class="tc-arg-name">path:</span>
          <span class="tp">media/main.css</span></div>
        <pre><code class="language-bash">ls</code></pre>
      </div>
      <div class="bash-panel"><div class="bash-panel-content">main.css
</div></div>
      <div class="tr"><div class="rl">Result</div>
        <div class="tr-content">ok</div></div>
    </div>
    <div class="ev sys">system output line</div>
    <div class="llm-panel">
      <div class="llm-panel-hdr">Thoughts</div>
      <div class="ev txt">nested agent text</div>
    </div>
    <div class="ev system-prompt">
      <div class="system-prompt-h">System prompt</div>
      <div class="system-prompt-body md-body">system prompt body</div>
    </div>
    <div class="ev prompt">
      <div class="prompt-h">Prompt</div>
      <div class="prompt-body md-body">prompt body</div>
    </div>
    <div class="ev merge-info">
      <div class="merge-info-hdr">\u2731 Reviewing 1 change(s)</div>
      <div class="merge-info-body">Red = old, Green = new.</div>
      <div class="merge-file-diff">
        <div class="merge-file-name">a.py</div>
        <pre class="merge-ctx">context line</pre>
        <pre class="merge-hunk"><span class="merge-hunk-label">Hunk
          1</span>-old\n+new</pre>
      </div>
    </div>
    <div class="ev wt-result-ok">Committed 2 files.</div>`);

  // Result card through the PRODUCTION renderer.
  if (!window._testApi || typeof window._testApi.processEvent !== 'function') {
    throw new Error('production output renderer is unavailable');
  }
  window._testApi.processEvent({
    type: 'result',
    success: false,
    is_continue: true,
    summary: 'result card body text',
    total_tokens: 1200,
    cost: '$0.01',
  });

  return out.children.length;
})()
"""

_INJECT_HISTORY_JS = r"""
(overrides => {
  // Lay the sidebar out (the desktop remote page auto-docks it, but
  // be explicit).
  document.getElementById('sidebar').classList.add('open');
  // Neutralize the date-range filter too: the page's boot getHistory
  // response autofills #hf-from/#hf-to from the REAL task database's
  // dateRange, and when the oldest listed task is newer than this
  // session's fixed Nov-2023 timestamp (e.g. after parallel test
  // sessions mutate the shared DB) the From bound would
  // display:none the injected row.  Clearing the inputs and
  // dispatching ``change`` pins the empty range
  // (``historyDateRangeUserSet``) so no later autofill re-hides the
  // row — the same path the Date submenu's Clear dates button uses.
  for (const id of ['hf-from', 'hf-to']) {
    const el = document.getElementById(id);
    if (!el) continue;
    el.value = '';
    el.dispatchEvent(new Event('change', {bubbles: true}));
  }
  const session = {
    id: 'chat-abc123',
    task_id: 42,
    title: 'Test task',
    // A daemon before 2026-10-31 sent the text a second time as
    // `preview`; the page reads `title` first.
    preview: 'Test task preview',
    steps: 3,
    tokens: 1234,
    cost: 0.5,
    timestamp: 1700000000,
    startTs: 1700000000000,
    endTs: 1700000061000,
    work_dir: '/tmp/w',
    model: 'gpt-x',
    is_parallel: true,
    is_running: false,
    failed: false,
    has_events: true,
    ...overrides,
  };
  for (let generation = 0; generation <= 30; generation++) {
    window.postMessage(
      {type: 'history', offset: 0, generation, sessions: [session]},
      '*',
    );
  }
})
"""

_SHIELD_HISTORY_JS = r"""
(() => {
  // Block SERVER-pushed history re-renders for the rest of the test.
  //
  // The page runs against the production RemoteAccessServer, whose
  // WebSocket shim forwards every server frame to main.js via
  // ``window.dispatchEvent(new MessageEvent('message', ...))``.  When
  // sibling test sessions mutate the shared task database the server
  // broadcasts, the page re-requests history, and the real response
  // re-renders ``#history-list`` — deleting the row injected by
  // ``_INJECT_HISTORY_JS`` between two probes (observed as
  // ``null.querySelector`` under parallel suite runs).  Overriding
  // ``window.dispatchEvent`` drops those ``{type:'history'}`` frames;
  // the test's own injections use ``window.postMessage``, whose
  // browser-internal dispatch does not go through this override, so
  // they still render.  Installed AFTER the boot history render so
  // page initialization is unaffected.
  if (window.__kissHistoryShield) return;
  window.__kissHistoryShield = true;
  const orig = window.dispatchEvent.bind(window);
  window.dispatchEvent = ev => {
    if (ev && ev.type === 'message' && ev.data
        && ev.data.type === 'history') {
      return true;
    }
    return orig(ev);
  };
})()
"""

_EXPAND_GROUP_JS = r"""
(() => {
  // Chat panels are collapsed by default (the injected task is not
  // running): open the injected chat's panel so its row lays out.
  const g = document.querySelector('#history-list .history-chat-group');
  if (g && g.classList.contains('collapsed')) {
    // The "Show details" chevron only unfolds the task rows; it opens
    // no tab (the header itself would).
    g.querySelector('.history-chat-collapse').click();
  }
  return g ? !g.classList.contains('collapsed') : false;
})()
"""

_PROBE_STYLES_JS = r"""(() => {
  const tp = getComputedStyle(document.querySelector('#output .task-panel'));
  const tpText = getComputedStyle(
    document.querySelector('#output .task-panel .task-panel-text'),
  );
  const tpHeader = getComputedStyle(
    document.querySelector('#output .task-panel .task-panel-h'),
  );

  // The thinking text injected by _INJECT_TRANSCRIPT_JS: the task
  // panel's header must paint the SAME foreground (var(--dim)).
  const think = document.querySelector('#output .think');
  const thinkCs = think ? getComputedStyle(think) : null;

  // Resolve var(--fg) (the colour of the task text and of the task
  // panel's wash) to rgb().
  const fgProbe = document.createElement('div');
  fgProbe.style.color = 'var(--fg)';
  document.body.appendChild(fgProbe);
  const fgColor = getComputedStyle(fgProbe).color;
  fgProbe.remove();

  // The old per-chat accent (djb2 hash of the chat id), resolved to an
  // rgb() string: nothing on the row may carry it any more.
  const id = 'chat-abc123';
  let hash = 5381;
  for (let i = 0; i < id.length; i++) {
    hash = (hash << 5) + hash + id.charCodeAt(i);
    hash |= 0;
  }
  const hsl = 'hsl(' + (Math.abs(hash) % 360) + ', 55%, 75%)';
  const probe = document.createElement('div');
  probe.style.color = hsl;
  document.body.appendChild(probe);
  const oldAccent = getComputedStyle(probe).color;
  probe.remove();

  const row = document.querySelector('#history-list .running-item');
  const rowCs = row ? getComputedStyle(row) : null;
  return {
    taskPanelFontSize: tpText.fontSize,
    taskPanelColor: tpText.color,
    taskPanelHeaderColor: tpHeader.color,
    taskPanelBg: tp.backgroundColor,
    taskPanelBorderWidth: tp.borderTopWidth,
    taskPanelBorderStyle: tp.borderTopStyle,
    taskPanelBorderColor: tp.borderTopColor,
    thinkBg: thinkCs ? thinkCs.backgroundColor : 'MISSING',
    thinkColor: thinkCs ? thinkCs.color : 'MISSING',
    fgColor,
    oldAccent,
    row: rowCs ? {
      borderLeftWidth: rowCs.borderLeftWidth,
      borderLeftColor: rowCs.borderLeftColor,
      backgroundColor: rowCs.backgroundColor,
      color: rowCs.color,
    } : 'MISSING',
  };
})()"""


# Read with Group by: None chosen in the Filter and group menu.
_LEGACY_VIEW_PROBE_JS = r"""(() => {
  const button = document.getElementById('history-filter-btn');
  const search = document.getElementById('history-search');
  const bBox = button.getBoundingClientRect();
  const sBox = search.getBoundingClientRect();
  const list = document.getElementById('history-list');
  const row = list.querySelector(':scope > .running-item');
  const cs = getComputedStyle(row);
  return {
    buttonVisible: bBox.width > 0 && bBox.height > 0,
    buttonLeft: bBox.left,
    searchRight: sBox.right,
    legacyClass: list.classList.contains('legacy-view'),
    groups: list.querySelectorAll('.history-chat-group').length,
    borderRightWidth: cs.borderRightWidth,
    borderLeftWidth: cs.borderLeftWidth,
    borderRightColor: cs.borderRightColor,
    borderLeftColor: cs.borderLeftColor,
    inlineStyle: row.getAttribute('style'),
    activeRow: row.classList.contains('history-active-task'),
  };
})()"""


def _start_live_server(
    tmp_path: Path,
    ready: threading.Event,
    done: threading.Event,
    state: dict[str, object],
) -> None:
    """Run the production RemoteAccessServer until *done* is set.

    Stores the bound ephemeral port in ``state['port']`` (or the
    startup exception in ``state['error']``) and sets *ready*.
    """
    from kiss.server.web_server import RemoteAccessServer
    from kiss.tests.local_ws import _generate_self_signed_cert

    certfile = tmp_path / "cert.pem"
    keyfile = tmp_path / "key.pem"
    _generate_self_signed_cert(certfile, keyfile)

    async def scenario() -> None:
        server = RemoteAccessServer(
            host="127.0.0.1",
            port=0,
            work_dir=str(tmp_path),
            certfile=str(certfile),
            keyfile=str(keyfile),
            url_file=tmp_path / "remote-url.json",
            local_endpoint_file=tmp_path / "sorcar-local.json",
        )
        started = False
        try:
            await server.start_async()
            started = True
            assert server._ws_server is not None
            state["port"] = next(iter(server._ws_server.sockets)).getsockname()[1]
            ready.set()
            while not done.is_set():
                await asyncio.sleep(0.02)
        except BaseException as exc:  # pragma: no cover - defensive
            state["error"] = exc
            ready.set()
        finally:
            if started:
                await server.stop_async()

    asyncio.run(scenario())


@pytest.mark.timeout(180)
def test_live_task_panel_typography_and_history_rows(
    tmp_path: Path,
) -> None:
    """Served page + real Chromium: the task panel keeps main.css's
    look (the page foreground over the user bubble's neutral wash, no
    outline, its label in the thinking text's dim foreground) under the
    remote palette; chat headers are plain rows with no fill of their
    own; history rows are flat, carry no per-chat colour and are one
    line of text with no fold button."""
    ready = threading.Event()
    done = threading.Event()
    state: dict[str, object] = {}
    thread = threading.Thread(
        target=_start_live_server,
        args=(tmp_path, ready, done, state),
        daemon=True,
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
                    ignore_https_errors=True,
                    viewport={"width": 1400, "height": 900},
                )
                goto_retrying_network_change(
                    page,
                    f"https://127.0.0.1:{port}/",
                    wait_until="domcontentloaded",
                )
                page.wait_for_selector("#output", state="attached")
                count = page.evaluate(_INJECT_PAGE_JS)
                assert count >= 9, "transcript injection failed"
                page.wait_for_selector(
                    "#history-list .sidebar-empty, #history-list .sidebar-item",
                    state="attached",
                    timeout=60000,
                )
                page.evaluate(_SHIELD_HISTORY_JS)
                for attempt in range(3):
                    page.evaluate(_INJECT_HISTORY_JS, {})
                    try:
                        page.wait_for_selector(
                            "#history-list .running-item",
                            state="attached",
                            timeout=10000,
                        )
                        break
                    except PlaywrightTimeoutError:
                        if attempt == 2:
                            raise
                # The idle chat's panel starts collapsed: the row is
                # hidden under its chat header until the header opens
                # the panel.
                group_probe = page.evaluate(
                    """() => {
                        const g = document.querySelector(
                            '#history-list .history-chat-group'
                        );
                        const row = document.querySelector(
                            '#history-list .running-item'
                        );
                        return {
                            collapsed: g.classList.contains(
                                'collapsed'
                            ),
                            headerText: g.querySelector(
                                '.history-chat-title'
                            ).textContent,
                            whiteSpace: getComputedStyle(
                                g.querySelector('.history-chat-title')
                            ).whiteSpace,
                            headerBg: getComputedStyle(
                                g.querySelector('.history-chat-header')
                            ).backgroundColor,
                            accent: (() => {
                                const probe = document.createElement('i');
                                probe.style.color = 'var(--accent)';
                                document.body.appendChild(probe);
                                const c = getComputedStyle(probe).color;
                                probe.remove();
                                return c;
                            })(),
                            headerTooltip: g.querySelector(
                                '.history-chat-header'
                            ).hasAttribute('data-tooltip'),
                            headerTitleAttr: g.querySelector(
                                '.history-chat-header'
                            ).hasAttribute('title'),
                            rowHidden: row.offsetParent === null,
                        };
                    }"""
                )
                assert group_probe["collapsed"] is True, (
                    "an idle chat's panel must start collapsed: " + repr(group_probe)
                )
                assert group_probe["rowHidden"] is True, (
                    "a collapsed chat panel must hide its task rows: " + repr(group_probe)
                )
                assert group_probe["headerText"] == "Test task", (
                    "the chat header shows the chat's first task text: " + repr(group_probe)
                )
                assert group_probe["whiteSpace"] == "nowrap", (
                    "the chat header shows one line of text: " + repr(group_probe)
                )
                assert _alpha_of(group_probe["headerBg"]) == 0, (
                    "the chat header is a plain row with no fill of its own: "
                    + repr(group_probe)
                )
                assert group_probe["headerTooltip"] is False, (
                    "the chat header carries no tooltip: " + repr(group_probe)
                )
                assert group_probe["headerTitleAttr"] is False, (
                    "the chat header carries no title attribute: " + repr(group_probe)
                )
                page.evaluate(_EXPAND_GROUP_JS)
                page.wait_for_selector(
                    "#history-list .running-item",
                    state="visible",
                    timeout=10000,
                )
                # The shield must hold: replay the exact dispatch the
                # server shim uses for a pushed history refresh (empty
                # real DB ⇒ empty sessions) across every plausible
                # generation and assert the injected row survives.
                # Without the shield this deterministically clears
                # ``#history-list`` — the race that flaked this test
                # under parallel suite runs.
                survived = page.evaluate(
                    """() => {
                        for (let g = 0; g <= 60; g++) {
                            window.dispatchEvent(new MessageEvent(
                                'message',
                                {data: {type: 'history', offset: 0,
                                        generation: g, sessions: []}},
                            ));
                        }
                        return !!document.querySelector(
                            '#history-list .running-item'
                        );
                    }"""
                )
                assert survived, (
                    "server-pushed history refresh clobbered the "
                    "injected history row despite the dispatch shield"
                )
                # A history panel is the one line of task text: it has
                # no fold button, there being nothing to unfold.  A
                # finished task's row carries one delete button (its
                # inline Delete / Cancel confirmation stays hidden), in
                # its "…" menu.
                buttons_probe = page.evaluate(
                    """() => {
                        const row = document.querySelector(
                            '#history-list .running-item'
                        );
                        return {
                            deleteButtons: row.querySelectorAll(
                                '.sidebar-item-delete'
                            ).length,
                            foldButtons: row.querySelectorAll(
                                '.sidebar-item-collapse'
                            ).length,
                        };
                    }"""
                )
                assert buttons_probe["deleteButtons"] == 1, (
                    "a finished task's panel renders one delete button: " + repr(buttons_probe)
                )
                assert buttons_probe["foldButtons"] == 0, (
                    "a history panel has no fold button: " + repr(buttons_probe)
                )
                # The task panel must hug its content: no
                # forced min-height and no oversized padding leaving
                # blank bands above/below the task text.
                spacing_probe = page.evaluate(
                    """() => {
                        const row = document.querySelector(
                            '#history-list .running-item'
                        );
                        const text = row.querySelector(
                            '.sidebar-item-text'
                        );
                        const actions = row.querySelector(
                            '.sidebar-item-actions'
                        );
                        const cs = getComputedStyle(row);
                        const rowBox = row.getBoundingClientRect();
                        const textBox = text.getBoundingClientRect();
                        return {
                            minHeight: cs.minHeight,
                            paddingTop: parseFloat(cs.paddingTop),
                            paddingBottom: parseFloat(
                                cs.paddingBottom
                            ),
                            spaceAbove: textBox.top - rowBox.top,
                            spaceBelow:
                                rowBox.bottom - textBox.bottom,
                            actionsOpen:
                                actions.matches(':popover-open'),
                            actionsHeight:
                                actions.getBoundingClientRect().height,
                        };
                    }"""
                )
                assert spacing_probe["minHeight"] in ("0px", "auto"), (
                    "a history panel must not reserve a min-height: " + repr(spacing_probe)
                )
                assert spacing_probe["paddingTop"] <= 8, (
                    "history panel must not pad extra space above "
                    "the task text: " + repr(spacing_probe)
                )
                assert spacing_probe["paddingBottom"] <= 8, (
                    "history panel must not pad extra space below "
                    "the task text: " + repr(spacing_probe)
                )
                # The blank band between the panel edge and the task
                # text itself must stay small: padding plus border
                # plus at most a few px of flex centering slack.
                assert spacing_probe["spaceAbove"] <= 12, (
                    "extra space above the task text in a history panel: " + repr(spacing_probe)
                )
                # Nothing lies under the task text: the action strip
                # is the row's "…" menu, closed and out of the row's
                # flow, so the same small band is all that remains
                # below the text.
                assert spacing_probe["spaceBelow"] <= 12, (
                    "extra space below the task text in a history panel: " + repr(spacing_probe)
                )
                assert spacing_probe["actionsOpen"] is False, (
                    "the row's menu is closed until its button opens it: " + repr(spacing_probe)
                )
                assert spacing_probe["actionsHeight"] == 0, (
                    "a closed menu takes no room in the panel: " + repr(spacing_probe)
                )
                # The pointer has not been on the row: the style probe
                # reads the row's own fill, not its :hover background.
                probes = page.evaluate(_PROBE_STYLES_JS)
                # The legacy flat list: Group by: None in the menu of
                # the button right of the search box switches to it,
                # and its rows carry no per-chat colour bar (computed
                # style, not just the declaration).  Group by: Chat
                # brings the chat panels back.
                pick_filter_option(page, "group", "None")
                legacy = page.evaluate(_LEGACY_VIEW_PROBE_JS)
                pick_filter_option(page, "group", "Chat")
                legacy["restoredGroups"] = page.locator(
                    "#history-list .history-chat-group"
                ).count()
            finally:
                browser.close()
    finally:
        done.set()
        thread.join(timeout=30)
    assert not thread.is_alive(), "RemoteAccessServer failed to stop"

    assert legacy["buttonVisible"] is True, repr(legacy)
    assert legacy["buttonLeft"] >= legacy["searchRight"], (
        "the Filter and group button sits to the right of the search box: " + repr(legacy)
    )
    assert legacy["legacyClass"] is True and legacy["groups"] == 0, (
        "the legacy view is a flat list: " + repr(legacy)
    )
    assert legacy["borderRightWidth"] == legacy["borderLeftWidth"] == "1px", (
        "the legacy row carries no per-chat colour bar: " + repr(legacy)
    )
    # No row is outlined: the row of the task the chat on screen is
    # bound to is marked by the selection fill of
    # ``.running-item.history-active-task`` alone (pinned by
    # test_history_panel_tints).
    assert _alpha_of(legacy["borderRightColor"]) == 0, (
        "the legacy row's outline is transparent: " + repr(legacy)
    )
    assert _alpha_of(legacy["borderLeftColor"]) == 0, (
        "the legacy row's outline is transparent: " + repr(legacy)
    )
    assert legacy["inlineStyle"] is None, (
        "the legacy row carries no inline colour: " + repr(legacy)
    )
    assert legacy["restoredGroups"] >= 1, (
        "Group by: Chat restores the chat panels: " + repr(legacy)
    )
    thread_error = state.get("error")
    if isinstance(thread_error, BaseException):
        raise AssertionError("RemoteAccessServer thread failed") from thread_error

    assert float(probes["taskPanelFontSize"].removesuffix("px")) == pytest.approx(
        14, abs=0.01
    ), (
        "the task text is the transcript's reading size, --fs-lg "
        "(1.077rem of the injected 13px root size): " + repr(probes)
    )
    assert probes["thinkColor"] != "MISSING", (
        "the injected transcript must render a .think text block: " + repr(probes)
    )
    assert probes["taskPanelHeaderColor"] == probes["thinkColor"], (
        "the task panel header must use the SAME foreground as the "
        "thinking text (main.css .task-panel-h: var(--dim)): " + repr(probes)
    )
    assert probes["taskPanelColor"] == probes["fgColor"], (
        "the task text reads like a user message, in the page foreground "
        "(main.css .task-panel-text: var(--fg)): " + repr(probes)
    )
    # The thinking text is plain text inside the Thoughts panel, with no
    # tint of its own; the task panel alone sits on the user bubble's
    # neutral wash (--bubble-user-tint: the page foreground at 7%), with
    # no outline.
    assert _alpha_of(probes["thinkBg"]) == 0, (
        "the thinking text paints no background of its own: " + repr(probes)
    )
    assert _rgb_of(probes["taskPanelBg"]) == pytest.approx(
        _rgb_of(probes["fgColor"]), abs=1
    ), "the task panel background must be a wash of the page foreground: " + repr(probes)
    assert 0.06 <= _alpha_of(probes["taskPanelBg"]) <= 0.08, (
        "the task panel wash is 7% of the page foreground: " + repr(probes)
    )
    assert probes["taskPanelBorderStyle"] == "none", probes
    assert probes["taskPanelBorderWidth"] == "0px", (
        "the task panel carries no outline: " + repr(probes)
    )

    row = probes["row"]
    assert row != "MISSING", "history row was not rendered"
    accent = probes["oldAccent"]
    assert row["borderLeftWidth"] != "4px", (
        f"the per-chat left border is gone; row: {row}"
    )
    assert row["borderLeftColor"] != accent, (
        f"nothing on the row may carry the old per-chat color {accent}; row: {row}"
    )
    assert row["backgroundColor"] != accent, (
        f"row background must not be the per-chat pastel; row: {row}"
    )
    # A flat row: no fill of its own; the page's editor.foreground (the
    # light palette's #0b0b0b, the default theme) rather than a colour
    # of its own.
    assert row["backgroundColor"] == "rgba(0, 0, 0, 0)", row
    assert row["color"] == probes["fgColor"] == "rgb(11, 11, 11)", (
        f"row text must be the editor foreground; row: {row}"
    )


# A task text far longer than the history panel is wide.
_LONG_TASK = (
    "Measure how long the nightly import takes on every shard, compare it with "
    "last week's run and write the ten slowest tables into the report"
)

# A row of the Chats panel (a chat header or a task panel): its text's
# line, its "…" button and its menu.
_PROBE_ROW_LINE_JS = r"""row => {
  const text = row.querySelector(
    ':scope > .sidebar-item-text, :scope > .history-chat-title');
  const more = row.querySelector(':scope > .sidebar-item-more');
  const menu = row.querySelector(':scope > .sidebar-item-actions');
  const rowBox = row.getBoundingClientRect();
  const textBox = text.getBoundingClientRect();
  const moreBox = more.getBoundingClientRect();
  return {
    textLines: textBox.height / parseFloat(getComputedStyle(text).lineHeight),
    ellipsized: text.scrollWidth > text.clientWidth
      && getComputedStyle(text).textOverflow === 'ellipsis',
    aroundText: rowBox.height - textBox.height,
    moreOnTextLine: moreBox.top < textBox.bottom && moreBox.bottom > textBox.top,
    moreAfterText: moreBox.left >= textBox.right - 0.5,
    moreInsideRow: moreBox.right <= rowBox.right,
    menuOpen: menu.matches(':popover-open'),
    menuHeight: menu.getBoundingClientRect().height,
  };
}"""

# The open menu of a row: where it hangs and what each of its lines
# shows.
_PROBE_ROW_MENU_JS = r"""row => {
  const box = el => el.getBoundingClientRect();
  const more = row.querySelector(':scope > .sidebar-item-more');
  const menu = row.querySelector(':scope > .sidebar-item-actions');
  // The text a pseudo-element draws ('' when it draws none).
  const drawn = (el, pseudo) => {
    const content = getComputedStyle(el, pseudo).content;
    return /^".*"$/.test(content) ? content.slice(1, -1) : '';
  };
  const probe = document.createElement('i');
  probe.style.color = 'var(--error)';
  document.body.appendChild(probe);
  const errorColor = getComputedStyle(probe).color;
  probe.remove();
  const shown = [...menu.children].filter(el => el.getClientRects().length > 0);
  return {
    open: menu.matches(':popover-open'),
    label: menu.getAttribute('aria-label'),
    opener: more.getAttribute('aria-label'),
    belowButton: box(menu).top - box(more).bottom,
    rightEdges: box(menu).right - box(more).right,
    insideWindow: box(menu).left >= 0 && box(menu).top >= 0
      && box(menu).right <= innerWidth && box(menu).bottom <= innerHeight,
    tops: shown.map(el => box(el).top),
    buttons: shown.filter(el => el.tagName === 'BUTTON').map(el => ({
      name: drawn(el, '::after'),
      label: el.getAttribute('aria-label'),
      height: box(el).height,
      nameWidth: box(el).width - box(el.querySelector('svg')).width,
      icon: [box(el.querySelector('svg')).width, box(el.querySelector('svg')).height],
      destructive: getComputedStyle(el, '::after').color === errorColor,
    })),
    notes: shown.filter(el => el.tagName !== 'BUTTON').map(el => {
      const lead = el.querySelector('.sidebar-item-launched') || el;
      return drawn(lead, '::before') + el.textContent;
    }),
  };
}"""

_MENU_IS_OPEN_JS = (
    "row => row.querySelector(':scope > .sidebar-item-actions').matches(':popover-open')"
)


def _open_row_menu(row: Locator) -> dict[str, Any]:
    """Open the "…" menu of *row* as a user with a mouse does; returns its probe."""
    row.hover()
    row.locator(":scope > .sidebar-item-more").click()
    menu: dict[str, Any] = row.evaluate(_PROBE_ROW_MENU_JS)
    return menu


def _assert_one_line_row(line: dict[str, Any], what: str) -> None:
    """*line* (``_PROBE_ROW_LINE_JS``) is one ellipsized line of text
    ending in the "…" button, with the row's menu closed."""
    assert line["textLines"] == pytest.approx(1, abs=0.1), f"{what}: {line}"
    assert line["ellipsized"] is True, f"{what}: a long text is cut with an ellipsis: {line}"
    assert line["aroundText"] <= 16, f"{what}: the row is as tall as its line of text: {line}"
    assert line["moreOnTextLine"] and line["moreAfterText"] and line["moreInsideRow"], (
        f'{what}: the "…" button ends the text\'s line: {line}'
    )
    assert line["menuOpen"] is False and line["menuHeight"] == 0, (
        f"{what}: the menu takes no room until it is opened: {line}"
    )


def _assert_row_menu(
    menu: dict[str, Any], label: str, names: list[str], notes: list[str],
) -> None:
    """*menu* (``_PROBE_ROW_MENU_JS``) hangs under its button and lists
    the buttons *names*, the destructive one last, then the lines
    matching the patterns *notes*."""
    assert menu["open"] is True and menu["label"] == menu["opener"] == label, menu
    assert menu["belowButton"] == pytest.approx(4, abs=0.5), menu
    assert menu["rightEdges"] == pytest.approx(0, abs=0.5), menu
    assert menu["insideWindow"] is True, menu
    # One line per button and per note, from top to bottom.
    assert len(menu["tops"]) == len(names) + len(notes), menu
    assert menu["tops"] == sorted(set(menu["tops"])), menu
    assert [button["name"] for button in menu["buttons"]] == names, menu
    for button in menu["buttons"]:
        # The name drawn beside the icon is the button's accessible name.
        assert button["name"] == button["label"], menu
        assert button["nameWidth"] > 40, menu
        assert button["height"] >= 30, menu
        assert button["icon"] == [14, 14], menu
    assert [button["destructive"] for button in menu["buttons"]] == [
        *[False] * (len(names) - 1), True,
    ], menu
    for note, pattern in zip(menu["notes"], notes, strict=True):
        assert re.fullmatch(pattern, note), menu


@pytest.mark.timeout(180)
def test_live_history_rows_are_one_line_with_their_actions_in_a_menu(
    tmp_path: Path,
) -> None:
    """Served page + real Chromium: a chat header and a task panel of
    the task history show ONE line - the text, cut with an ellipsis,
    and at its right end the "…" button, after the fold chevron on a
    header.  The "…" button opens the row's menu under it: one named
    line per action, the destructive one last, then when the task was
    launched and the machine it ran on (the daemon's own for a task
    that ran here).  Choosing a line closes the menu and acts on the
    row; Escape closes the menu and nothing else.

    The layout is measured from real layout boxes of the remote webapp
    exactly as ``RemoteAccessServer`` serves it (``body.remote-chat``
    + remote-codex.css).
    """
    ready = threading.Event()
    done = threading.Event()
    state: dict[str, object] = {}
    thread = threading.Thread(
        target=_start_live_server,
        args=(tmp_path, ready, done, state),
        daemon=True,
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
                    ignore_https_errors=True,
                    viewport={"width": 1400, "height": 900},
                )
                goto_retrying_network_change(
                    page,
                    f"https://127.0.0.1:{port}/",
                    wait_until="domcontentloaded",
                )
                assert page.evaluate("document.body.classList.contains('remote-chat')"), (
                    "the served page must be the remote webapp"
                )
                page.wait_for_selector(
                    "#history-list .sidebar-empty, #history-list .sidebar-item",
                    state="attached",
                    timeout=60000,
                )
                page.evaluate(_SHIELD_HISTORY_JS)
                long_remote_task = {
                    "title": _LONG_TASK,
                    "preview": _LONG_TASK,
                    "machine": "gpu-box",
                }
                for attempt in range(3):
                    page.evaluate(_INJECT_HISTORY_JS, long_remote_task)
                    try:
                        page.wait_for_selector(
                            "#history-list .running-item", state="attached", timeout=10000,
                        )
                        break
                    except PlaywrightTimeoutError:
                        if attempt == 2:
                            raise
                header = page.locator("#history-list .history-chat-header")
                task = page.locator("#history-list .running-item")
                header_line = header.evaluate(_PROBE_ROW_LINE_JS)
                header_menu = _open_row_menu(header)
                # Choosing a line of the menu closes it and acts on the
                # row: the pencil turns the title into a text box, which
                # Escape turns back.
                header.locator(":scope > .sidebar-item-actions > .history-chat-rename").click()
                rename_box = header.locator(".history-chat-title-input")
                rename_box.wait_for(state="visible", timeout=10000)
                header_menu_left_open = header.evaluate(_MENU_IS_OPEN_JS)
                page.keyboard.press("Escape")
                rename_box.wait_for(state="detached", timeout=10000)
                # The chevron left of the "…" button unfolds the chat's task rows.
                header.locator(":scope > .history-chat-collapse").click()
                task.wait_for(state="visible", timeout=10000)
                task_line = task.evaluate(_PROBE_ROW_LINE_JS)
                task_menu = _open_row_menu(task)
                page.keyboard.press("Escape")
                task_menu_left_open = task.evaluate(_MENU_IS_OPEN_JS)
                task_listed_after_escape = task.is_visible()
                # A chat that ran here names the machine of the daemon serving the page.
                page.evaluate(_INJECT_HISTORY_JS, {"title": "a task that ran here"})
                expect(header.locator(".sidebar-item-machine")).to_have_text(
                    local_machine_name(), timeout=10000,
                )
                local_menu = _open_row_menu(header)
            finally:
                browser.close()
    finally:
        done.set()
        thread.join(timeout=30)
    assert not thread.is_alive(), "RemoteAccessServer failed to stop"
    thread_error = state.get("error")
    if isinstance(thread_error, BaseException):
        raise AssertionError("RemoteAccessServer thread failed") from thread_error

    _assert_one_line_row(header_line, "chat header")
    _assert_row_menu(
        header_menu,
        "Chat actions",
        ["Favourite chat", "Pin chat", "Edit the chat summary", "Delete chat"],
        [r"Last task .+ ago", r"on gpu-box"],
    )
    assert header_menu_left_open is False, "choosing a line closes the menu"
    _assert_one_line_row(task_line, "task panel")
    _assert_row_menu(
        task_menu,
        "Task actions",
        ["Favourite task", "Pin task", "Copy task to clipboard", "Delete task"],
        [r"Launched .+ ago", r"on gpu-box"],
    )
    assert task_menu_left_open is False, "Escape closes the menu"
    assert task_listed_after_escape is True, "Escape closes the menu only"
    assert local_menu["notes"][-1] == f"on {local_machine_name()}", local_menu
