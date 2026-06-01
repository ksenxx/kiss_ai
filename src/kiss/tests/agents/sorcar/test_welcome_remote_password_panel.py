"""Tests for welcome-page remote-password panel visibility.

Verifies that ``renderRemoteUrl`` in ``main.js`` hides the
``#welcome-config`` panel when the Cloudflare tunnel is not active
and shows it when the tunnel is active.

The panel hosts the remote-access password field; showing it without an
active tunnel would be misleading (the password is only used for the
public tunnel URL).
"""

import json
import re
import subprocess
import unittest
from pathlib import Path
from typing import Any

_MAIN_JS = (
    Path(__file__).resolve().parents[4]
    / "kiss"
    / "agents"
    / "vscode"
    / "media"
    / "main.js"
)


def _extract_function(source: str, name: str) -> str:
    """Extract a JS function body from ``source`` by brace-matching."""
    match = re.search(rf"function {name}\(", source)
    assert match, f"Function {name} not found"
    brace_start = source.index("{", match.end())
    depth = 1
    i = brace_start + 1
    while depth > 0:
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
        i += 1
    return source[match.start():i]


_DOM_SHIM = r"""
class MiniEl {
  constructor(tag) {
    this.tagName = (tag || 'div').toUpperCase();
    this.style = {};
    this.children = [];
    this.innerHTML = '';
    this.textContent = '';
    this._attrs = {};
    this._listeners = {};
  }
  appendChild(c) {
    c.parentElement = this;
    this.children.push(c);
    return c;
  }
  setAttribute(k, v) { this._attrs[k] = v; }
  getAttribute(k) { return this._attrs[k]; }
  addEventListener(type, fn) {
    (this._listeners[type] = this._listeners[type] || []).push(fn);
  }
  querySelector() { return null; }
}

const elements = {};
function regEl(id, tag) {
  const e = new MiniEl(tag || 'div');
  e.id = id;
  elements[id] = e;
  return e;
}

regEl('remote-url');
regEl('welcome-remote-url');
regEl('welcome-config');

const bodyEl = new MiniEl('body');
bodyEl.classList = {
  _set: new Set(),
  add(c) { this._set.add(c); },
  remove(c) { this._set.delete(c); },
  contains(c) { return this._set.has(c); },
};

global.document = {
  body: bodyEl,
  getElementById: id => elements[id] || null,
  createElement: tag => new MiniEl(tag),
};

// Minimal helpers the function relies on (mkEl/checkSvg/copySvg etc are
// only needed if displayUrl is truthy and _buildRemoteUrlBar is called;
// we stub the bar to keep the shim small).
global._buildRemoteUrlBar = function(url, isNtfy) {
  const e = new MiniEl('div');
  e.textContent = url + (isNtfy ? ':ntfy' : '');
  return e;
};
"""


def _run_render(
    url: str, ntfy: str | None, tunnel_active: object,
) -> dict[str, Any]:
    """Invoke renderRemoteUrl(url, ntfyUrl, tunnelActive) in Node.

    Returns a dict with the post-call state of #welcome-config and the
    text content of the URL bars.
    """
    src = _MAIN_JS.read_text()
    fn = _extract_function(src, "renderRemoteUrl")
    # Replace the local _buildRemoteUrlBar reference with the global stub.
    fn = fn.replace("_buildRemoteUrlBar(", "global._buildRemoteUrlBar(")
    ntfy_arg = "undefined" if ntfy is None else repr(ntfy)
    if tunnel_active is None:
        ta_arg = "undefined"
    else:
        ta_arg = "true" if tunnel_active else "false"
    script = (
        _DOM_SHIM
        + "\n" + fn
        + f"\nrenderRemoteUrl({url!r}, {ntfy_arg}, {ta_arg});\n"
        + "const wc = document.getElementById('welcome-config');\n"
        + "const wu = document.getElementById('welcome-remote-url');\n"
        + "console.log(JSON.stringify({"
        + "welcomeConfigDisplay: wc.style.display,"
        + "welcomeUrlText: (wu.children[0]||{}).textContent || ''"
        + "}));\n"
    )
    proc = subprocess.run(
        ["node", "-e", script],
        capture_output=True, text=True, timeout=15,
    )
    assert proc.returncode == 0, f"node failed: {proc.stderr}\n{script}"
    parsed: dict[str, Any] = json.loads(proc.stdout.strip())
    return parsed


class WelcomeRemotePasswordPanelTest(unittest.TestCase):
    """Verify the welcome-config visibility behavior."""

    def test_tunnel_active_shows_panel(self) -> None:
        """tunnelActive=True → welcome-config is visible."""
        result = _run_render(
            "https://abc.trycloudflare.com", None, True,
        )
        self.assertEqual(result["welcomeConfigDisplay"], "")
        self.assertEqual(
            result["welcomeUrlText"], "https://abc.trycloudflare.com",
        )

    def test_tunnel_inactive_hides_panel(self) -> None:
        """tunnelActive=False → welcome-config is hidden."""
        result = _run_render(
            "https://localhost:8787", None, False,
        )
        self.assertEqual(result["welcomeConfigDisplay"], "none")

    def test_tunnel_inactive_empty_url_hides_panel(self) -> None:
        """Empty URL + tunnelActive=False → hidden."""
        result = _run_render("", None, False)
        self.assertEqual(result["welcomeConfigDisplay"], "none")

    def test_undefined_tunnel_active_falls_back_to_url(self) -> None:
        """Older backends omit tunnelActive; fall back to URL presence."""
        result = _run_render(
            "https://abc.trycloudflare.com", None, None,
        )
        self.assertEqual(result["welcomeConfigDisplay"], "")

    def test_undefined_tunnel_active_no_url_hides(self) -> None:
        """Older backends + no URL → hidden."""
        result = _run_render("", None, None)
        self.assertEqual(result["welcomeConfigDisplay"], "none")


if __name__ == "__main__":
    unittest.main()
