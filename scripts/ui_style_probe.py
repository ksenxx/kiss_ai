#!/usr/bin/env python3
# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Dump the computed style of every visible control on every KISS UI surface.

Usage::

    uv run python scripts/ui_style_probe.py tmp/probe.txt

Renders the same scenes as ``ui_theme_screenshots.py`` (welcome,
transcript, model menu, history, "..." menu, settings) on the VS Code
webview, the remote webapp at desktop and phone width, and the share
page, and writes one line per distinct control::

    selector | h x w | fs | pad | radius | border | bg | colour | weight ...

so the same button, field, chip or menu item can be compared across
surfaces line by line (a control falling back to the browser's default
typeface shows up as ``Arial``; a stray radius or font size stands out
against its neighbours).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, TextIO

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ui_theme_screenshots as shots  # noqa: E402
from playwright.sync_api import Page, sync_playwright  # noqa: E402

PROBE_JS = r"""
() => {
  const sel = 'button, input, textarea, select, a, [role=menuitem], [role=tab], ' +
    '[role=option], .chip, .pill, .badge, .tc-h, .think-h, .prompt-h, .rc-h, .tr, ' +
    '.history-chat-header, .history-item, .menu-item, .dropdown-item, ' +
    '.kiss-notification, #task-panel, .card, h1, h2, h3, label, summary, .hdr, ' +
    '.sidebar-hdr-row, .status, #tab-status-bar, #tab-bar';
  const out = [];
  const seen = new Set();
  for (const el of document.querySelectorAll(sel)) {
    const r = el.getBoundingClientRect();
    if (r.width < 2 || r.height < 2) continue;
    const cs = getComputedStyle(el);
    if (cs.visibility === 'hidden' || cs.display === 'none' || cs.opacity === '0') continue;
    let key = el.tagName.toLowerCase();
    if (el.id) key += '#' + el.id;
    const state = /^(visible|open|active|selected|hover)$/;
    const cls = [...el.classList].filter(c => !state.test(c)).slice(0, 3);
    if (cls.length) key += '.' + cls.join('.');
    const sig = key + '|' + Math.round(r.height) + '|' + cs.fontSize + '|' + cs.borderRadius;
    if (seen.has(sig)) continue;
    seen.add(sig);
    out.push({
      key, h: Math.round(r.height), w: Math.round(r.width), fs: cs.fontSize,
      pad: cs.padding, radius: cs.borderRadius,
      border: cs.borderTopWidth + ' ' + cs.borderTopStyle + ' ' + cs.borderTopColor,
      bg: cs.backgroundColor, color: cs.color, weight: cs.fontWeight,
      tt: cs.textTransform, ls: cs.letterSpacing,
      ff: cs.fontFamily.split(',')[0],
      text: (el.textContent || '').trim().replace(/\s+/g, ' ').slice(0, 40),
      shadow: cs.boxShadow === 'none' ? '' : 'shadow',
    });
  }
  return out;
}
"""


def dump(page: Page, name: str, out: TextIO) -> None:
    """Write one line per distinct visible control of *page* under a *name* header."""
    rows: list[dict[str, Any]] = page.evaluate(PROBE_JS)
    out.write(f"\n===== {name} ({len(rows)} rows)\n")
    for r in rows:
        tt = r["tt"] if r["tt"] != "none" else ""
        ls = f"ls {r['ls']}" if r["ls"] != "normal" else ""
        out.write(
            f"{r['key'][:60]:60} | {r['h']:>3}x{r['w']:<4} | fs {r['fs']:>8} "
            f"| pad {r['pad']:<16} | r {r['radius']:<10} | b {r['border']:<36} "
            f"| bg {r['bg']:<28} | c {r['color']:<24} | w {r['weight']} {tt} {ls} "
            f"{r['shadow']} | {r['ff']} | {r['text']!r}\n"
        )


def scenes(page: Page, name: str, out: TextIO) -> str:
    """Walk the six scenes of one surface, dumping each; return the transcript HTML."""
    dump(page, f"{name} welcome", out)
    page.evaluate(shots.TRANSCRIPT_JS)
    page.wait_for_timeout(300)
    dump(page, f"{name} transcript", out)
    page.click("#model-btn")
    page.wait_for_timeout(200)
    dump(page, f"{name} model-menu", out)
    page.keyboard.press("Escape")
    page.mouse.click(5, 5)
    page.evaluate("() => document.getElementById('sidebar').classList.add('open')")
    page.evaluate(
        "s => { const req = window.__posted.filter(m => m.type === 'getHistory').pop();"
        " window.__post({type: 'history', sessions: s, offset: 0,"
        " generation: req ? req.generation : 0}); }",
        [{**s, "preview": s["title"], "has_events": True} for s in shots.SESSIONS],
    )
    page.wait_for_timeout(300)
    dump(page, f"{name} history", out)
    page.evaluate("() => document.getElementById('sidebar').classList.remove('open')")
    page.click("#more-btn")
    page.wait_for_timeout(200)
    dump(page, f"{name} more-menu", out)
    page.click("#settings-btn")
    page.wait_for_timeout(300)
    dump(page, f"{name} settings", out)
    return str(page.evaluate("() => document.getElementById('output').innerHTML"))


def main() -> None:
    """Probe every surface and write the report to the path given on the command line."""
    out_path = Path(sys.argv[1] if len(sys.argv) > 1 else "tmp/ui-style-probe.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    server, base = shots.start_media_server()
    with out_path.open("w") as out, sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 460, "height": 900})
        page = ctx.new_page()
        shots.open_page(page, base, shots.vscode_page_html("dark-plus"), "vscode-dark-plus")
        transcript = scenes(page, "vscode-dark", out)
        ctx.close()
        for width, height, label in ((1280, 800, "desktop"), (390, 844, "phone")):
            ctx = browser.new_context(viewport={"width": width, "height": height})
            page = ctx.new_page()
            shots.open_page(page, base, shots.remote_page_html(), f"remote-{label}")
            scenes(page, f"remote-{label}-dark", out)
            ctx.close()
        share_html = shots.web_server._build_share_page(
            "Design tokens", '<div class="share-task">' + transcript + "</div>"
        )
        ctx = browser.new_context(viewport={"width": 900, "height": 900})
        page = ctx.new_page()
        page.set_content(share_html, wait_until="load")
        dump(page, "share-dark", out)
        ctx.close()
        browser.close()
    server.shutdown()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
