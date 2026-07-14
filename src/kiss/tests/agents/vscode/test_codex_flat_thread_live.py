# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Live e2e: the served remote webapp renders the Codex flat thread.

This is the screenshot-backed end-to-end test for the Codex restyle.
It starts the production :class:`RemoteAccessServer` with a real TLS
listener on an ephemeral port, loads the served page in headless
Chromium (Playwright), injects a representative chat transcript
(thinking block, streamed markdown with a highlighted code fence, a
collapsed tool call, a Bash tool call with mono output + result, an
Edit diff, and the final result card), then:

* asserts the COMPUTED styles in the real browser match the Codex
  flat-thread design (page ``#0d0d0d``; transparent 12px-radius tool
  panels; quiet sentence-case ``#8e8e8e`` headers; ``#171717`` inset
  mono blocks with bottom-only rounding when nested; flat borderless
  result card; transparent ``.hljs`` inside the inset ``pre``), and
* takes full-page screenshots of both the welcome view and the
  injected thread, asserting the PNG files are actually produced.

No mocks: the production server, the production stylesheets, and a
real Chromium render path are exercised end to end.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright

# JS injected into the live page: hide the welcome pane and append a
# transcript exercising every restyled panel class produced by
# media/main.js's renderer.
_INJECT_THREAD_JS = r"""
(() => {
  const out = document.getElementById('output');
  const welcome = document.getElementById('welcome');
  if (welcome) welcome.style.display = 'none';
  const app = document.getElementById('app');
  if (app) app.style.display = '';
  const loading = document.getElementById('kiss-server-loading');
  if (loading) loading.style.display = 'none';
  const tp = document.getElementById('task-panel');
  tp.classList.add('visible');
  tp.style.display = 'block';
  document.getElementById('task-panel-text').textContent =
    'Improve the aesthetics of the chat panels so they are ' +
    'indistinguishable from Codex';

  out.insertAdjacentHTML('beforeend', `
    <div class="ev think">
      <div class="lbl"><span class="arrow">\u25BE</span> Thinking</div>
      <div class="cnt">I need to inspect the stylesheet and align the
panel design tokens with the Codex thread language.</div>
    </div>
    <div class="ev txt md-body">I'll start by reading the stylesheet.
      <pre><code class="hljs language-python">print("Codex")</code></pre>
    </div>
    <div class="ev tc copyable collapsed">
      <button class="panel-copy-btn" aria-label="Copy panel">Copy</button>
      <div class="tc-h collapse-header"><span class="collapse-chv">\u25BE</span>Read
        <span class="collapse-preview">remote-codex.css</span></div>
      <div class="tc-b">hidden body</div>
    </div>
    <div class="ev tc tc-bash">
      <div class="tc-h tc-h-bash collapse-header">
        <span class="collapse-chv">\u25BE</span>Bash</div>
      <div class="tc-b">
        <div class="tc-arg"><span class="tc-arg-name">description:</span>
          List stylesheet files</div>
        <pre><code class="language-bash">ls media/*.css</code></pre>
      </div>
      <div class="bash-panel"><div class="bash-panel-content">main.css
remote-codex.css</div></div>
      <div class="tr"><div class="rl">Result</div>
        <div class="tr-content">main.css\nremote-codex.css</div></div>
      <div class="panel-time">2.1s</div>
    </div>
    <div class="ev tc">
      <div class="tc-h collapse-header">
        <span class="collapse-chv">\u25BE</span>Edit</div>
      <div class="tc-b">
        <div class="tc-arg"><span class="tc-arg-name">path:</span>
          <span class="tp">media/remote-codex.css</span></div>
        <div class="diff-old">- background: #171717;</div>
        <div class="diff-new">+ background: transparent;</div>
      </div>
      <div class="panel-time">0.4s</div>
    </div>
    <div class="ev rc">
      <div class="rc-h"><h3>Task completed</h3>
        <div class="rs"><span>Tokens<b>12,408</b></span>
        <span>Cost<b>$0.31</b></span><span>Time<b>48s</b></span></div>
      </div>
      <div class="rc-body">The chat panels now use the flat Codex
thread language.</div>
    </div>`);
  return out.children.length;
})()
"""

_COMPUTED_STYLES_JS = r"""
(() => {
  const gs = (sel) => getComputedStyle(document.querySelector(sel));
  const body = getComputedStyle(document.body);
  const tc = gs('.tc');
  const tch = gs('.tc-h');
  const tr = gs('.tr');
  const rc = gs('.rc');
  const think = gs('.think .lbl');
  const hljs = gs('.hljs');
  const pre = gs('.md-body pre');
  const collapsed = document.querySelector('.tc.collapsed');
  const preview = getComputedStyle(
    collapsed.querySelector('.collapse-preview'));
  const hiddenBody = getComputedStyle(collapsed.querySelector('.tc-b'));
  return {
    bodyBg: body.backgroundColor,
    tcBg: tc.backgroundColor,
    tcRadius: tc.borderRadius,
    tchTransform: tch.textTransform,
    tchColor: tch.color,
    tchBg: tch.backgroundColor,
    trBg: tr.backgroundColor,
    trRadius: tr.borderRadius,
    rcBg: rc.backgroundColor,
    rcBorder: rc.borderStyle,
    thinkTransform: think.textTransform,
    thinkColor: think.color,
    hljsBg: hljs.backgroundColor,
    preBg: pre.backgroundColor,
    preRadius: pre.borderRadius,
    previewDisplay: preview.display,
    previewColor: preview.color,
    collapsedBodyDisplay: hiddenBody.display,
  };
})()
"""


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
def test_live_thread_renders_codex_flat_style_with_screenshots(
    tmp_path: Path,
) -> None:
    """Served page + real Chromium: computed styles match Codex and
    screenshots of the welcome view and injected thread are captured."""
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

        welcome_png = tmp_path / "codex_welcome.png"
        thread_png = tmp_path / "codex_thread.png"
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
                page.screenshot(path=str(welcome_png))

                count = page.evaluate(_INJECT_THREAD_JS)
                assert count >= 6, "transcript injection failed"
                page.wait_for_timeout(200)
                page.screenshot(path=str(thread_png), full_page=True)

                styles = page.evaluate(_COMPUTED_STYLES_JS)
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

    # Screenshot proof: both PNGs exist and are non-trivial.
    assert welcome_png.stat().st_size > 1000
    assert thread_png.stat().st_size > 1000

    # Page sits on the Codex near-black canvas.
    assert styles["bodyBg"] == "rgb(13, 13, 13)"
    # Tool-call panels are flat (transparent) with the 12px radius.
    assert styles["tcBg"] == "rgba(0, 0, 0, 0)"
    assert styles["tcRadius"] == "12px"
    # Headers are quiet: sentence case, muted #8e8e8e, no strip.
    assert styles["tchTransform"] == "none"
    assert styles["tchColor"] == "rgb(142, 142, 142)"
    assert styles["tchBg"] == "rgba(0, 0, 0, 0)"
    # Mono output keeps the #171717 inset surface; nested inside a
    # .tc only the bottom corners stay rounded.
    assert styles["trBg"] == "rgb(23, 23, 23)"
    assert styles["trRadius"] in ("12px", "0px 0px 12px 12px")
    # Result card renders like a plain assistant reply.
    assert styles["rcBg"] == "rgba(0, 0, 0, 0)"
    assert styles["rcBorder"] == "none"
    # Thinking label is quiet and muted.
    assert styles["thinkTransform"] == "none"
    assert styles["thinkColor"] == "rgb(142, 142, 142)"
    # Markdown code fences: pre carries the #171717 12px surface and
    # the inner .hljs is transparent (main.css's !important bg beaten).
    assert styles["preBg"] == "rgb(23, 23, 23)"
    assert styles["preRadius"] == "12px"
    assert styles["hljsBg"] == "rgba(0, 0, 0, 0)"
    # Collapsed tool call: preview visible and muted, body hidden.
    assert styles["previewDisplay"] == "block"
    assert styles["previewColor"] == "rgb(142, 142, 142)"
    assert styles["collapsedBodyDisplay"] == "none"
