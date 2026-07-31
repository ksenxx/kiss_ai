# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E tests: remote-webapp task-panel-matched typography + history rows.

Features on the remote webapp (served by ``RemoteAccessServer``):

1. The pinned task panel (``#task-panel``) inherits main.css's
   inverted look verbatim (the remote page merely swaps the palette
   variables), sized by the page's injected 16px
   ``--vscode-editor-font-size``.  The event panels likewise inherit
   the extension's main.css typography — that extension-parity
   contract is pinned end to end by
   ``test_remote_panels_match_extension.py``.
2. History rows (``.running-item``) drop their per-chat pastel
   BACKGROUND color; the per-chat color moves to a thick LEFT border.
   The VS Code webview keeps its pastel look via an equivalent
   ``main.css`` rule driven by the same ``--task-color`` custom
   property (main.js must stop writing inline colors, which no
   stylesheet can override).
3. ALL task metadata (steps, tok, cost, duration, time, work dir,
   model, wt, parallel, auto-commit, chat id, task id) renders as ONE
   wrapping line instead of three separately-clipped lines.  Field
   AVAILABILITY is unchanged: legacy rows whose persisted ``extra``
   JSON predates the run-mode metadata still omit the model/wt/
   parallel/auto-commit group (the span-omission contract pinned by
   ``historyTaskMeta.test.js`` / ``historyTaskIds.test.js``); every
   row persisted by the current backend carries all fields, and the
   live test below renders a fully-populated row.

Static (regex) tests pin the CSS/JS wiring; the live test boots the
production ``RemoteAccessServer`` + headless Chromium and asserts the
COMPUTED styles.
"""

from __future__ import annotations

import asyncio
import re
import threading
from pathlib import Path

import pytest
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

MEDIA_DIR = (
    Path(__file__).resolve().parents[3] / "agents" / "vscode" / "media"
)
CODEX_CSS = MEDIA_DIR / "remote-codex.css"
MAIN_CSS = MEDIA_DIR / "main.css"
MAIN_JS = MEDIA_DIR / "main.js"
WEB_SERVER_PY = (
    Path(__file__).resolve().parents[3] / "server" / "web_server.py"
)


def _find_rule(css: str, selector: str) -> str:
    """Return the union of declaration bodies of every
    ``body.remote-chat``-scoped rule for *selector*, or fail."""
    pattern = (
        r"body\.remote-chat[^{,]*"
        + re.escape(selector)
        + r"\s*(?:,[^{]*)?\{([^}]*)\}"
    )
    bodies = re.findall(pattern, CODEX_CSS.read_text(encoding="utf-8"))
    assert bodies, f"body.remote-chat scoped rule for {selector!r} missing"
    return "\n".join(bodies)


def test_remote_page_font_size_vars_match_task_panel() -> None:
    """The task panel sizes itself with --vscode-editor-font-size and
    the chat panels with rem units derived from --vscode-font-size;
    the remote page must inject the SAME 16px for both so panel
    contents and the task panel share one size."""
    src = WEB_SERVER_PY.read_text(encoding="utf-8")
    assert "--vscode-font-size: 16px" in src
    assert "--vscode-editor-font-size: 16px" in src



def test_main_js_history_rows_use_task_color_var_not_inline() -> None:
    """renderHistory must expose the per-chat color as a --task-color
    custom property instead of unoverridable inline styles.  (The
    Frequent tab's renderer is out of scope and keeps its own inline
    colors.)"""
    js = MAIN_JS.read_text(encoding="utf-8")
    start = js.index("function renderHistory(")
    end = js.index("function openCustomDatePicker(")
    body = js[start:end]
    assert "style.backgroundColor" not in body, (
        "renderHistory must not set an inline background color on "
        "history rows (inline styles beat every stylesheet)"
    )
    assert "style.color = '#1a1a1a'" not in body, (
        "renderHistory must not set an inline text color on history rows"
    )
    assert (
        "setProperty('--task-color', chatIdBgColor(String(s.id)))" in body
    ), "renderHistory must set the --task-color custom property per row"


def test_main_css_keeps_webview_pastel_look_via_task_color() -> None:
    """The VS Code webview must look exactly as before: main.css drives
    the old inline pastel background/dark text from --task-color."""
    css = MAIN_CSS.read_text(encoding="utf-8")
    m = re.search(r"\n\.running-item\s*\{([^}]*)\}", css)
    assert m, ".running-item rule missing from main.css"
    rule = m.group(1)
    assert "background-color: var(--task-color" in rule, (
        ".running-item must paint the per-chat pastel background from "
        f"var(--task-color) in the webview; got: {rule!r}"
    )
    assert "color: #1a1a1a" in rule, (
        ".running-item must keep the webview's dark text on the pastel "
        f"background; got: {rule!r}"
    )


def test_remote_history_row_color_moves_to_left_border() -> None:
    """On the remote page the row background is neutral and the
    per-chat color paints a thick left border instead."""
    rule = _find_rule(
        CODEX_CSS.read_text(encoding="utf-8"), ".running-item"
    )
    assert "border-left: 4px solid var(--task-color" in rule, (
        "the per-chat color must move to the row's left border; "
        f"got: {rule!r}"
    )
    assert "background-color: rgb(255 255 255 / 4%)" in rule, (
        f"the row background must be a neutral dark tint; got: {rule!r}"
    )
    assert "color: #ececec" in rule, (
        f"the row text must be light on the dark background; got: {rule!r}"
    )


REMOTE_METADATA_COLOR_RULES = [
    (".running-item-metrics", "color: #afafaf"),
    (".running-item-workspace", "color: #8e8e8e"),
    (".running-item-ids", "color: #8e8e8e"),
    (".running-item .ids-copy-btn", "color: #ececec"),
    (".running-item .sidebar-item-delete", "color: #ececec"),
    (".running-item .sidebar-item-copy", "color: #ececec"),
    (".running-item .sidebar-item-favorite", "color: #ececec"),
]


@pytest.mark.parametrize(("selector", "decl"), REMOTE_METADATA_COLOR_RULES)
def test_remote_history_metadata_readable_on_dark(
    selector: str, decl: str
) -> None:
    """main.css metadata/buttons colors are near-black (designed for
    the pastel background); the remote's dark neutral rows need light
    replacements."""
    rule = _find_rule(CODEX_CSS.read_text(encoding="utf-8"), selector)
    assert decl in rule, f"{selector} must set {decl}; got: {rule!r}"



def test_remote_metadata_container_flows_as_one_line() -> None:
    """.running-item-info must stop stacking the three spans as flex
    columns so they flow inline as one wrapping line."""
    rule = _find_rule(
        CODEX_CSS.read_text(encoding="utf-8"), ".running-item-info"
    )
    assert "display: block" in rule, (
        f".running-item-info must be a block flow container; got: {rule!r}"
    )
    assert "overflow-wrap: anywhere" in rule, (
        "long unbroken tokens (work dirs, ids) must wrap; "
        f"got: {rule!r}"
    )


@pytest.mark.parametrize(
    "selector",
    [".running-item-metrics", ".running-item-workspace", ".running-item-ids"],
)
def test_remote_metadata_spans_wrap_instead_of_clip(selector: str) -> None:
    """The spans lose nowrap/ellipsis so nothing is clipped."""
    rule = _find_rule(CODEX_CSS.read_text(encoding="utf-8"), selector)
    assert "white-space: normal" in rule, (
        f"{selector} must wrap instead of nowrap-clipping; got: {rule!r}"
    )
    assert "overflow: visible" in rule, (
        f"{selector} must not hide overflowing metadata; got: {rule!r}"
    )


@pytest.mark.parametrize(
    "selector",
    [".running-item-workspace::before", ".running-item-ids::before"],
)
def test_remote_metadata_separator_between_groups(selector: str) -> None:
    """The workspace and ids groups join the single line with the same
    dot separator used inside each group."""
    rule = _find_rule(CODEX_CSS.read_text(encoding="utf-8"), selector)
    assert "\u2022" in rule, (
        f"{selector} must insert a ' \u2022 ' separator; got: {rule!r}"
    )


_INJECT_PAGE_JS = r"""
(() => {
  const out = document.getElementById('output');
  const welcome = document.getElementById('welcome');
  if (welcome) welcome.style.display = 'none';
  const app = document.getElementById('app');
  if (app) app.style.display = '';
  const loading = document.getElementById('kiss-server-loading');
  if (loading) loading.style.display = 'none';

  // Pin the task panel (the typography reference) with real text.
  document.getElementById('task-panel-text').textContent =
    'Fix the flux capacitor';
  document.getElementById('task-panel').classList.add('visible');

  out.insertAdjacentHTML('beforeend', `
    <div class="ev think">
      <div class="lbl"><span class="arrow">\u25BE</span> Thinking</div>
      <div class="cnt">Reasoning about the task panel type.</div>
    </div>
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
  if (!window._demoApi || typeof window._demoApi.processEvent !== 'function') {
    throw new Error('production output renderer is unavailable');
  }
  window._demoApi.processEvent({
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
(() => {
  // Lay the sidebar out (the desktop remote page auto-docks it, but
  // be explicit) and disable the Workspace filter: it is on by
  // default and would display:none a row whose work_dir differs
  // from the client's workspace — the wrap-geometry probes need the
  // row rendered.
  document.getElementById('sidebar').classList.add('open');
  const wsChk = document.getElementById('hf-workspace');
  if (wsChk) wsChk.checked = false;
  // Neutralize the date-range filter too: the page's boot getHistory
  // response autofills #hf-from/#hf-to from the REAL task database's
  // dateRange, and when the oldest listed task is newer than this
  // session's fixed Nov-2023 timestamp (e.g. after parallel test
  // sessions mutate the shared DB) the From bound would
  // display:none the injected row.  Clearing the inputs and
  // dispatching ``change`` pins the empty range
  // (``historyDateRangeUserSet``) so no later autofill re-hides the
  // row — the same path the filter bar's own clear button uses.
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
    preview: 'Test task preview',
    steps: 3,
    tokens: 1234,
    cost: 0.5,
    timestamp: 1700000000,
    startTs: 1700000000000,
    endTs: 1700000061000,
    work_dir: '/tmp/w',
    model: 'gpt-x',
    is_worktree: true,
    is_parallel: true,
    auto_commit_mode: true,
    is_running: false,
    failed: false,
    has_events: true,
  };
  for (let generation = 0; generation <= 30; generation++) {
    window.postMessage(
      {type: 'history', offset: 0, generation, sessions: [session]},
      '*',
    );
  }
})()
"""

_PROBE_STYLES_JS = (
    r"""(() => {
  const tp = getComputedStyle(document.getElementById('task-panel'));

  // Expected per-chat accent: same djb2 hash as chatIdBgColor,
  // resolved to an rgb() string via a probe element.
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
  const expectedAccent = getComputedStyle(probe).color;
  probe.remove();

  const row = document.querySelector('#history-list .running-item');
  const rowCs = row ? getComputedStyle(row) : null;
  const info = document.querySelector('.running-item-info');
  const metrics = document.querySelector('.running-item-metrics');
  const workspace = document.querySelector('.running-item-workspace');
  const ids = document.querySelector('.running-item-ids');
  const span = el => {
    if (!el) return 'MISSING';
    const cs = getComputedStyle(el);
    return cs.display + ' | ' + cs.whiteSpace + ' | ' + cs.overflow;
  };
  // Wrapping geometry: the inline metadata flow over N line boxes.
  let infoLineRects = 0;
  let infoClipped = false;
  if (info) {
    const range = document.createRange();
    range.selectNodeContents(info);
    infoLineRects = range.getClientRects().length;
    infoClipped = info.scrollWidth > info.clientWidth + 1;
  }
  return {
    taskPanelFontSize: tp.fontSize,
    taskPanelColor: tp.color,
    taskPanelBg: tp.backgroundColor,
    infoLineRects,
    infoClipped,
    expectedAccent,
    row: rowCs ? {
      borderLeftWidth: rowCs.borderLeftWidth,
      borderLeftColor: rowCs.borderLeftColor,
      backgroundColor: rowCs.backgroundColor,
      color: rowCs.color,
    } : 'MISSING',
    infoDisplay: info ? getComputedStyle(info).display : 'MISSING',
    metrics: span(metrics),
    workspace: span(workspace),
    ids: span(ids),
    metricsText: metrics ? metrics.textContent : 'MISSING',
    workspaceText: workspace ? workspace.textContent : 'MISSING',
    idsText: ids ? ids.textContent : 'MISSING',
    workspaceSep: workspace
      ? getComputedStyle(workspace, '::before').content : 'MISSING',
    idsSep: ids ? getComputedStyle(ids, '::before').content : 'MISSING',
  };
})()"""
)


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
    from kiss.server.web_server import (
        RemoteAccessServer,
        _generate_self_signed_cert,
    )

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
            uds_path=tmp_path / "sorcar.sock",
        )
        started = False
        try:
            await server.start_async()
            started = True
            assert server._ws_server is not None
            state["port"] = next(
                iter(server._ws_server.sockets)
            ).getsockname()[1]
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
    """Served page + real Chromium: the pinned task panel keeps the
    extension's inverted look under the remote palette; history rows
    paint the per-chat color on the left border over a neutral
    background; all metadata flows as one wrapping line."""
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
            raise AssertionError(
                "RemoteAccessServer startup failed"
            ) from startup_error
        port = state["port"]

        with sync_playwright() as p:
            browser = p.chromium.launch(
                args=["--ignore-certificate-errors"]
            )
            try:
                page = browser.new_page(
                    ignore_https_errors=True,
                    viewport={"width": 1400, "height": 900},
                )
                page.goto(
                    f"https://127.0.0.1:{port}/",
                    wait_until="domcontentloaded",
                )
                page.wait_for_selector("#output", state="attached")
                count = page.evaluate(_INJECT_PAGE_JS)
                assert count >= 9, "transcript injection failed"
                page.wait_for_selector(
                    "#history-list .sidebar-empty, "
                    "#history-list .sidebar-item",
                    state="attached",
                    timeout=60000,
                )
                for attempt in range(3):
                    page.evaluate(_INJECT_HISTORY_JS)
                    try:
                        page.wait_for_selector(
                            "#history-list .running-item",
                            state="visible",
                            timeout=10000,
                        )
                        break
                    except PlaywrightTimeoutError:
                        if attempt == 2:
                            raise
                page.wait_for_function(
                    """() => {
                        const info = document.querySelector(
                            '.running-item-info'
                        );
                        if (!info) return false;
                        const range = document.createRange();
                        range.selectNodeContents(info);
                        return range.getClientRects().length >= 2;
                    }""",
                    timeout=10000,
                )
                probes = page.evaluate(_PROBE_STYLES_JS)
            finally:
                browser.close()
    finally:
        done.set()
        thread.join(timeout=30)
    assert not thread.is_alive(), "RemoteAccessServer failed to stop"
    thread_error = state.get("error")
    if isinstance(thread_error, BaseException):
        raise AssertionError(
            "RemoteAccessServer thread failed"
        ) from thread_error

    assert probes["taskPanelFontSize"] == "16px", (
        "the task panel must size itself from the injected 16px "
        "--vscode-editor-font-size: " + repr(probes)
    )
    assert probes["taskPanelColor"] == "rgb(13, 13, 13)", (
        "the task panel text must use the inverted #0d0d0d foreground "
        "(main.css --panel-fg: var(--bg) under the remote palette): "
        + repr(probes)
    )
    assert probes["taskPanelBg"] == "rgb(236, 236, 236)", (
        "the task panel background must be the inverted light #ececec "
        "surface (main.css --panel-bg: var(--fg) under the remote "
        "palette): " + repr(probes)
    )

    row = probes["row"]
    assert row != "MISSING", "history row was not rendered"
    accent = probes["expectedAccent"]
    assert row["borderLeftWidth"] == "4px", row
    assert row["borderLeftColor"] == accent, (
        f"left border must carry the per-chat color {accent}; row: {row}"
    )
    assert row["backgroundColor"] != accent, (
        f"row background must not be the per-chat pastel; row: {row}"
    )
    assert row["backgroundColor"] == "rgba(255, 255, 255, 0.04)", row
    assert row["color"] == "rgb(236, 236, 236)", (
        f"row text must be light (not the old #1a1a1a); row: {row}"
    )

    assert probes["infoDisplay"] == "block", probes
    for key in ("metrics", "workspace", "ids"):
        assert probes[key] == "inline | normal | visible", (
            f"{key} span must flow inline and wrap: " + repr(probes)
        )
    assert "\u2022" in probes["workspaceSep"], probes
    assert "\u2022" in probes["idsSep"], probes
    assert probes["infoLineRects"] >= 2, (
        "the single metadata line must wrap over multiple line boxes: "
        + repr(probes)
    )
    assert probes["infoClipped"] is False, (
        "the metadata flow must not be clipped horizontally: "
        + repr(probes)
    )
    assert "3 steps" in probes["metricsText"], probes
    assert "1,234 tok" in probes["metricsText"], probes
    assert "$0.5000" in probes["metricsText"], probes
    assert "00:01:01" in probes["metricsText"], probes
    assert re.search(
        r"Nov 1[45], 2023, \d{1,2}:\d{2}\s?[AP]M", probes["metricsText"]
    ), probes
    assert (
        probes["workspaceText"]
        == "/tmp/w \u2022 gpt-x \u2022 wt \u2022 parallel \u2022 auto-commit"
    ), probes
    assert "chat chat-abc123" in probes["idsText"], probes
    assert "task 42" in probes["idsText"], probes
