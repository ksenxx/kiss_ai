# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E geometry tests: the status bar's machine name never overlaps.

The top status bar (``#tab-status-bar`` in ``media/chat.html``) shows
the running task's tokens / cost / steps metrics on the right and the
"Ready" status text on the left; the server machine's hostname (new
``#status-machine``, filled from the ``configData`` event) sits in the
middle between them.

An earlier draft absolutely centered the hostname over the flex row,
which painted it straight across the Tokens / Cost metrics at common
sidebar widths (measured 30-56px of overlap between 300 and 600px).
The shipped layout keeps the hostname a flex item — its
``margin-left: auto`` pairs with ``#status-tokens``'s so the free
space splits evenly around it — and lets it shrink first
(``flex-shrink: 1000``, ``min-width: 0``, ellipsis) when the bar gets
cramped.

These tests render the real ``chat.html`` status bar with the real
``main.css`` cascade in a real headless Chromium (the extension
surface; the remote webapp adds only color overrides for this bar) and
assert on actual bounding boxes: at every representative width — the
narrow 300px sidebar through a wide 1000px editor — with a long real
hostname and full metrics, no two items of the bar overlap, the
hostname stays inside the bar, and on wide bars it straddles the bar's
midpoint.  With the absolutely-centered CSS these assertions fail.
"""

from __future__ import annotations

import re
from pathlib import Path

from playwright.sync_api import sync_playwright

_MEDIA_DIR = (
    Path(__file__).resolve().parents[3] / "agents" / "vscode" / "media"
)

_LONG_HOSTNAME = "ksen-vm-32.c.r2eg-441800.internal"
_WIDTHS = [300, 350, 400, 500, 600, 800, 1000]


def _build_status_bar_page() -> str:
    """Render the real status bar markup with the real stylesheet."""
    src = (_MEDIA_DIR / "chat.html").read_text(encoding="utf-8")
    m = re.search(
        r'<div id="tab-status-bar">.*?</div>\s*</div>', src, re.DOTALL,
    )
    assert m, "tab-status-bar block not found in chat.html"
    bar = m.group(0)
    assert 'id="status-machine"' in bar
    main_css = (_MEDIA_DIR / "main.css").read_text(encoding="utf-8")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <style>
    :root {{
      --vscode-font-size: 13px;
      --vscode-font-family: -apple-system, BlinkMacSystemFont,
        'Segoe UI', Roboto, sans-serif;
      --vscode-editor-background: #1e1e1e;
      --vscode-editor-foreground: #cccccc;
      --vscode-sideBar-background: #252526;
      --vscode-panel-border: #80808059;
      --vscode-descriptionForeground: #8b8b8b;
      --vscode-textLink-foreground: #3794ff;
      --vscode-terminal-ansiRed: #f44747;
      --vscode-terminal-ansiGreen: #6a9955;
      --vscode-terminal-ansiYellow: #d7ba7d;
      --vscode-terminal-ansiMagenta: #c586c0;
      --vscode-terminal-ansiCyan: #4ec9b0;
    }}
    html, body {{ margin: 0; padding: 0; }}
  </style>
  <style>{main_css}</style>
  <title>status machine layout test</title>
</head>
<body>
  <div id="app">{bar}</div>
</body>
</html>"""


_FILL_BAR_JS = """
(machine) => {
  document.getElementById('status-machine').textContent = machine;
  document.getElementById('status-tokens').textContent = 'Tokens: 123,456';
  document.getElementById('status-budget').textContent = 'Cost: $12.34';
  document.getElementById('status-steps').textContent = 'Steps: 87/100';
}
"""

_MEASURE_JS = """
() => {
  const ids = ['status-text', 'status-machine', 'status-tokens',
               'status-budget', 'status-steps'];
  const boxes = {};
  for (const id of ids) {
    const r = document.getElementById(id).getBoundingClientRect();
    boxes[id] = {left: r.left, right: r.right, width: r.width};
  }
  const bar = document.getElementById('tab-status-bar')
    .getBoundingClientRect();
  boxes.bar = {left: bar.left, right: bar.right, width: bar.width};
  return boxes;
}
"""


def test_machine_name_never_overlaps_the_metrics() -> None:
    """At every width the bar's items keep disjoint bounding boxes."""
    page_html = _build_status_bar_page()
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(
            viewport={"width": _WIDTHS[0], "height": 200},
        )
        page.set_content(page_html, wait_until="load")
        page.evaluate(_FILL_BAR_JS, _LONG_HOSTNAME)
        for width in _WIDTHS:
            page.set_viewport_size({"width": width, "height": 200})
            boxes = page.evaluate(_MEASURE_JS)
            order = [
                "status-text",
                "status-machine",
                "status-tokens",
                "status-budget",
                "status-steps",
            ]
            for a, b in zip(order[:-1], order[1:], strict=True):
                gap = boxes[b]["left"] - boxes[a]["right"]
                assert gap >= -0.5, (
                    f"{a} overlaps {b} by {-gap:.1f}px at {width}px"
                )
            machine = boxes["status-machine"]
            bar = boxes["bar"]
            assert machine["left"] >= bar["left"] - 0.5
            assert machine["right"] <= bar["right"] + 0.5
        browser.close()


def test_machine_name_sits_around_the_middle_on_wide_bars() -> None:
    """On a wide bar the (untruncated) hostname straddles the middle."""
    page_html = _build_status_bar_page()
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1000, "height": 200})
        page.set_content(page_html, wait_until="load")
        page.evaluate(_FILL_BAR_JS, _LONG_HOSTNAME)
        boxes = page.evaluate(_MEASURE_JS)
        machine = boxes["status-machine"]
        bar = boxes["bar"]
        mid = (bar["left"] + bar["right"]) / 2
        assert machine["left"] < mid < machine["right"], (
            f"hostname [{machine['left']:.0f}, {machine['right']:.0f}] "
            f"must straddle the bar midpoint {mid:.0f}"
        )
        # And it is rendered whole (no truncation) at this width.
        text_width = page.evaluate(
            "() => { const el = document.getElementById('status-machine');"
            " return el.scrollWidth <= el.clientWidth; }",
        )
        assert text_width, "the hostname must not be ellipsized at 1000px"
        browser.close()
