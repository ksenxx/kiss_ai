# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E tests: the remote webapp chat panels use ONE body-text size.

The remote webapp (served by ``RemoteAccessServer``) renders each chat
panel's body text at a different size: plain assistant text (``.txt``),
system output (``.sys``) and result-card stat values (``.rs b``) sit at
``--fs-base`` (16px on the remote page) while thinking content
(``.think .cnt``), tool-call bodies (``.tc-b``), tool results (``.tr``),
bash output (``.bash-panel-content``), result-card bodies (``.rc-body``),
prompt bodies and nested ``.llm-panel .txt`` sit at ``--fs-md``
(≈14.5px), with hard-coded outliers: 12px merge hunks/context,
a 15px remote result-card title, and an inline
``font-size: var(--fs-xl)`` status line emitted by
``createResultPanel`` in ``main.js``.

These tests reproduce the non-uniformity and pin the fix:

* Static (regex) tests: ``remote-codex.css`` must contain a
  ``body.remote-chat``-scoped uniform-typography rule pinning
  ``font-size: var(--fs-base)`` on every panel body-text selector,
  the hard-coded 15px/12px outliers must be gone, and ``main.js``
  must not emit any inline ``font-size`` (class-based instead).
* Live test: the production ``RemoteAccessServer`` + headless Chromium
  render a transcript covering every panel type and the COMPUTED
  ``font-size`` of every panel's body text must be exactly equal.

The VS Code extension webview (which shares ``chat.html`` +
``main.css``) must be unaffected: all CSS overrides live in
``remote-codex.css`` under ``body.remote-chat``, and the ``main.js``
status line keeps its ``--fs-xl`` look in the webview via an
equivalent ``main.css`` class rule.
"""

from __future__ import annotations

import asyncio
import re
import threading
from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright

MEDIA_DIR = (
    Path(__file__).resolve().parents[3] / "agents" / "vscode" / "media"
)
CODEX_CSS = MEDIA_DIR / "remote-codex.css"
MAIN_CSS = MEDIA_DIR / "main.css"
MAIN_JS = MEDIA_DIR / "main.js"
DEMO_JS = MEDIA_DIR / "demo.js"


def _read_codex_css() -> str:
    return CODEX_CSS.read_text(encoding="utf-8")


def _find_rule(css: str, selector: str) -> str:
    """Return the union of declaration bodies of every
    ``body.remote-chat``-scoped rule for *selector*, or fail."""
    pattern = (
        r"body\.remote-chat[^{,]*"
        + re.escape(selector)
        + r"\s*(?:,[^{]*)?\{([^}]*)\}"
    )
    bodies = re.findall(pattern, css)
    assert bodies, f"body.remote-chat scoped rule for {selector!r} missing"
    return "\n".join(bodies)


# Every chat-panel BODY-TEXT selector.  main.css sizes them
# inconsistently (--fs-base vs --fs-md vs hard-coded px); the remote
# stylesheet must pin every one of them to the SAME var(--fs-base).
PANEL_BODY_TEXT_SELECTORS = [
    ".txt",
    ".llm-panel .txt",
    ".think .cnt",
    ".llm-panel .think .cnt",
    ".tc-b",
    ".tp",
    ".tr",
    ".bash-panel-content",
    ".rc-h h3",
    ".rs",
    ".rs b",
    ".rc-body",
    ".system-prompt-body",
    ".prompt-body",
    ".sys",
    ".merge-info-hdr",
    ".merge-info-body",
    ".merge-ctx",
    ".merge-hunk",
    ".txt code",
    ".md-body code",
    ".md-body pre code",
    ".md-body th",
    ".md-body td",
    ".rc-status",
]


@pytest.mark.parametrize("selector", PANEL_BODY_TEXT_SELECTORS)
def test_panel_body_text_pinned_to_uniform_size(selector: str) -> None:
    """remote-codex.css pins *selector* to font-size: var(--fs-base)."""
    rule = _find_rule(_read_codex_css(), selector)
    assert "font-size: var(--fs-base)" in rule, (
        f"{selector} must be pinned to the uniform var(--fs-base) "
        f"body size in remote-codex.css; got declarations: {rule!r}"
    )


def test_no_hardcoded_result_card_title_size() -> None:
    """The old off-scale 15px .rc-h h3 override must be gone."""
    css = re.sub(r"/\*.*?\*/", "", _read_codex_css(), flags=re.S)
    for match in re.finditer(r"[^{}]*\.rc-h h3[^{]*\{([^}]*)\}", css):
        assert "15px" not in match.group(1), (
            ".rc-h h3 must not keep the hard-coded 15px size"
        )


def test_main_js_result_status_uses_class_not_inline_style() -> None:
    """createResultPanel must not emit an inline font-size (inline
    styles are unbeatable by stylesheets); it uses .rc-status classes."""
    js = MAIN_JS.read_text(encoding="utf-8")
    assert "font-size:var(--fs-xl)" not in js, (
        "main.js must not emit the inline font-size:var(--fs-xl) "
        "status line; use the .rc-status class instead"
    )
    assert 'class="rc-status"' in js, (
        "the Continue status line must carry class rc-status"
    )
    assert 'class="rc-status rc-status-fail"' in js, (
        "the FAILED status line must carry class rc-status rc-status-fail"
    )


def test_demo_js_result_status_uses_class_not_inline_style() -> None:
    """demo.js (demo-replay result renderer, loaded by the remote page
    via {{DEMO_SRC}}) must use the same .rc-status classes — its old
    inline style.cssText font-size also beat every stylesheet rule."""
    js = DEMO_JS.read_text(encoding="utf-8")
    assert "font-size:var(--fs-xl)" not in js, (
        "demo.js must not emit an inline font-size:var(--fs-xl) "
        "status line; use the .rc-status class instead"
    )
    assert "'rc-status'" in js, (
        "demo.js Continue status line must carry class rc-status"
    )
    assert "'rc-status rc-status-fail'" in js, (
        "demo.js FAILED status line must carry class "
        "rc-status rc-status-fail"
    )


def test_main_css_styles_rc_status_like_old_inline_style() -> None:
    """main.css replicates the old inline declarations so the VS Code
    webview keeps its exact former look (yellow/red, bold, fs-xl)."""
    css = MAIN_CSS.read_text(encoding="utf-8")
    m = re.search(r"\.rc-status\s*\{([^}]*)\}", css)
    assert m, ".rc-status rule missing from main.css"
    rule = m.group(1)
    assert "color: var(--yellow)" in rule
    assert "font-weight: 700" in rule
    assert "font-size: var(--fs-xl)" in rule
    assert "margin-bottom: 10px" in rule
    fail = re.search(r"\.rc-status\.rc-status-fail\s*\{([^}]*)\}", css)
    assert fail, ".rc-status.rc-status-fail rule missing from main.css"
    assert "color: var(--red)" in fail.group(1)


# ── Live end-to-end: computed font sizes are EQUAL ──────────────────

# Injects a transcript covering every chat-panel body-text surface
# rendered by media/main.js.
_INJECT_THREAD_JS = r"""
(() => {
  const out = document.getElementById('output');
  const welcome = document.getElementById('welcome');
  if (welcome) welcome.style.display = 'none';
  const app = document.getElementById('app');
  if (app) app.style.display = '';
  const loading = document.getElementById('kiss-server-loading');
  if (loading) loading.style.display = 'none';

  out.insertAdjacentHTML('beforeend', `
    <div class="ev think">
      <div class="lbl"><span class="arrow">\u25BE</span> Thinking</div>
      <div class="cnt">Reasoning about the uniform type scale.</div>
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
      <div class="ev think"><div class="lbl">Thinking</div>
        <div class="cnt">nested thinking</div></div>
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

  // Render the result card through the PRODUCTION renderer
  // (processOutputEvent -> handleOutputEvent -> createResultPanel)
  // so the test exercises the real .rc/.rc-status DOM, not a
  // hand-written approximation of it.
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

# Selector → element whose computed font-size is the panel's body text.
_FONT_SIZE_PROBES = {
    "txt": ".ev.txt",
    "txtCode": ".ev.txt code",
    "txtPreCode": ".ev.txt pre code",
    "txtTh": ".ev.txt th",
    "txtTd": ".ev.txt td",
    "thinkCnt": ".ev.think .cnt",
    "tcB": ".tc-b",
    "tcArg": ".tc-arg",
    "tp": ".tp",
    "tcPreCode": ".tc-b pre code",
    "bashContent": ".bash-panel-content",
    "tr": ".tr .tr-content",
    "sys": ".ev.sys",
    "llmTxt": ".llm-panel .txt",
    "llmThinkCnt": ".llm-panel .think .cnt",
    "systemPromptBody": ".system-prompt-body",
    "promptBody": ".prompt-body",
    "mergeInfoHdr": ".merge-info-hdr",
    "mergeInfoBody": ".merge-info-body",
    "mergeCtx": ".merge-ctx",
    "mergeHunk": ".merge-hunk",
    "mergeFileName": ".merge-file-name",
    "wtResultOk": ".wt-result-ok",
    "rcH3": ".rc-h h3",
    "rs": ".rs",
    "rsB": ".rs b",
    "rcBody": ".rc-body",
    "rcStatus": ".rc-status",
}

_COMPUTED_FONT_SIZES_JS = (
    "(() => { const probes = "
    + repr(_FONT_SIZE_PROBES).replace("'", '"')
    + """;
  const sizes = {};
  for (const key of Object.keys(probes)) {
    const el = document.querySelector(probes[key]);
    sizes[key] = el ? getComputedStyle(el).fontSize : 'MISSING';
  }
  return sizes;
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
    from kiss.agents.vscode.web_server import (
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
def test_live_chat_panel_font_sizes_are_uniform(tmp_path: Path) -> None:
    """Served page + real Chromium: every chat panel's body text
    computes to the SAME font-size (one uniform size thread-wide)."""
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
                count = page.evaluate(_INJECT_THREAD_JS)
                assert count >= 9, "transcript injection failed"
                page.wait_for_timeout(200)
                sizes = page.evaluate(_COMPUTED_FONT_SIZES_JS)
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

    missing = [k for k, v in sizes.items() if v == "MISSING"]
    assert not missing, f"probe elements missing from the page: {missing}"

    # THE uniformity assertion: every panel's body text is the same
    # size — the remote page's base body size (16px).
    assert sizes["txt"] == "16px", sizes
    distinct = {v for v in sizes.values()}
    assert distinct == {"16px"}, (
        "chat-panel body text is NOT uniform; computed sizes: "
        + repr(sizes)
    )
