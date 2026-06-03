# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration test: a finished sub-agent tab's purple ◉ indicator
must render SOLID (no pulse animation), not be removed.

Bug
---
After a sub-agent finishes, the ``subagentDone`` event sets
``tab.isDone=true`` and ``renderTabBar`` rerenders the tab bar.  The
previous implementation suppressed the ``◉`` indicator entirely on
done sub-agent tabs, while the user-visible requirement is a SOLID
purple ◉ in the tab title (the running state pulses; the done state
must be a solid circle).

This integration test drives the real ``renderTabBar`` from
``media/main.js`` end-to-end in Node.js with a minimal DOM shim,
exercising the exact same code path the webview executes.  It also
verifies the corresponding CSS rule in ``media/main.css`` produces a
non-pulsing purple indicator when ``.subagent-indicator`` carries the
``.done`` modifier class.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import unittest
from pathlib import Path
from typing import Any, cast

_MEDIA = (
    Path(__file__).resolve().parents[4]
    / "kiss" / "agents" / "vscode" / "media"
)
_MAIN_JS = _MEDIA / "main.js"
_MAIN_CSS = _MEDIA / "main.css"


def _extract_render_tab_bar(src: str) -> str:
    """Return the source of ``function renderTabBar()`` from main.js.

    The function is defined at module scope inside the top-level
    IIFE.  We locate the ``function renderTabBar(`` token and walk
    balanced braces to find its closing ``}``.
    """
    start = src.index("function renderTabBar(")
    # Find the opening ``{`` of the function body.
    open_brace = src.index("{", start)
    depth = 0
    i = open_brace
    while i < len(src):
        c = src[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return src[start : i + 1]
        i += 1
    raise AssertionError("Could not find end of renderTabBar")


_NODE_HARNESS = r"""
// Minimal DOM shim: just enough for renderTabBar's subagent branch
// (createElement + className/textContent/title + appendChild +
// addEventListener + dataset).

function _matches(el, selector) {
  // Tiny selector matcher: only ``.classA.classB`` (compound class).
  const wanted = selector.split('.').filter(Boolean);
  const have = (el.className || '').split(/\s+/).filter(Boolean);
  return wanted.every(c => have.includes(c));
}

function _walk(root, fn) {
  for (const c of root.children || []) {
    fn(c);
    _walk(c, fn);
  }
}

function makeElement(tag) {
  const el = {
    tagName: String(tag).toUpperCase(),
    className: '',
    textContent: '',
    title: '',
    dataset: {},
    style: {},
    children: [],
    _parent: null,
    appendChild(child) {
      child._parent = this;
      this.children.push(child);
      return child;
    },
    insertBefore(child, ref) {
      const i = this.children.indexOf(ref);
      child._parent = this;
      if (i < 0) this.children.push(child);
      else this.children.splice(i, 0, child);
      return child;
    },
    removeChild(child) {
      const i = this.children.indexOf(child);
      if (i >= 0) this.children.splice(i, 1);
      return child;
    },
    querySelector(selector) {
      let found = null;
      _walk(this, c => {
        if (!found && _matches(c, selector)) found = c;
      });
      return found;
    },
    querySelectorAll(selector) {
      const out = [];
      _walk(this, c => {
        if (_matches(c, selector)) out.push(c);
      });
      return out;
    },
    scrollIntoView() {},
    addEventListener() {},
    set innerHTML(v) {
      this._innerHTML = String(v == null ? '' : v);
      if (this._innerHTML === '') this.children = [];
    },
    get innerHTML() {
      return this._innerHTML || '';
    },
  };
  return el;
}

const tabListEl = makeElement('div');
const tabBarEl = makeElement('div');

global.document = {
  getElementById(id) {
    if (id === 'tab-list') return tabListEl;
    if (id === 'tab-bar') return tabBarEl;
    return null;
  },
  createElement: makeElement,
};

// Symbols renderTabBar references at module scope inside main.js.
// We declare them here so the extracted body type-checks under Node.
var tabs = __TABS__;
var activeTabId = __ACTIVE__;
function closeTab() {}
function switchToTab() {}

__RENDER_TAB_BAR__

renderTabBar();

// Flatten children into a serialisable summary for the Python side.
function summarise(node) {
  return {
    tag: node.tagName,
    className: node.className,
    textContent: node.textContent,
    title: node.title,
    children: node.children.map(summarise),
  };
}

console.log(JSON.stringify({
  tabs: tabListEl.children.map(summarise),
}));
"""


def _node_render(tabs: list[dict[str, Any]], active_id: str) -> dict[str, Any]:
    """Run ``renderTabBar`` under Node with *tabs*; return the DOM
    summary of the produced tab bar.
    """
    body = _extract_render_tab_bar(_MAIN_JS.read_text(encoding="utf-8"))
    script = (
        _NODE_HARNESS
        .replace("__RENDER_TAB_BAR__", body)
        .replace("__TABS__", json.dumps(tabs))
        .replace("__ACTIVE__", json.dumps(active_id))
    )
    r = subprocess.run(
        ["node", "-e", script],
        capture_output=True,
        text=True,
        timeout=15,
    )
    if r.returncode != 0:
        raise AssertionError(
            f"node failed (rc={r.returncode}): {r.stderr}\nstdout: {r.stdout}"
        )
    return cast(dict[str, Any], json.loads(r.stdout.strip()))


def _find_indicator(tab_summary: dict[str, Any]) -> dict[str, Any] | None:
    """Return the ``.subagent-indicator`` child element of a tab, or None."""
    for child in tab_summary.get("children", []):
        cls = child.get("className", "")
        if "subagent-indicator" in cls.split():
            return cast(dict[str, Any], child)
    return None


@unittest.skipUnless(shutil.which("node"), "node not on PATH")
class TestSubagentTabDoneSolidIndicator(unittest.TestCase):
    """End-to-end checks on renderTabBar + main.css for the
    finished-sub-agent indicator."""

    def test_done_subagent_tab_renders_solid_indicator(self) -> None:
        """``tab.isDone=true`` → indicator IS rendered with ``.done``
        class (which kills the pulse animation in CSS).
        """
        out = _node_render(
            tabs=[{
                "id": "sub-1",
                "title": "1. work",
                "isSubagentTab": True,
                "isDone": True,
                "isRunning": False,
                "hasRunTask": True,
                "lastTaskFailed": False,
            }],
            active_id="sub-1",
        )
        tab = out["tabs"][0]
        self.assertIn("subagent-tab", tab["className"].split())
        indicator = _find_indicator(tab)
        assert indicator is not None, (
            f"Done sub-agent tab MUST render the ◉ indicator (solid). "
            f"tab={tab!r}"
        )
        classes = indicator["className"].split()
        self.assertIn(
            "done",
            classes,
            f"Done sub-agent indicator MUST have the ``.done`` class "
            f"so the CSS rule disables the pulse animation. "
            f"className={indicator['className']!r}",
        )
        self.assertEqual(
            indicator["textContent"],
            "◉",
            f"Indicator glyph must remain ◉, got {indicator['textContent']!r}",
        )

    def test_running_subagent_tab_renders_pulsing_indicator(self) -> None:
        """``tab.isDone=false`` → indicator rendered WITHOUT ``.done``
        class, so the default ``subagent-pulse`` animation applies.
        """
        out = _node_render(
            tabs=[{
                "id": "sub-2",
                "title": "1. work",
                "isSubagentTab": True,
                "isDone": False,
                "isRunning": True,
                "hasRunTask": False,
                "lastTaskFailed": False,
            }],
            active_id="sub-2",
        )
        tab = out["tabs"][0]
        indicator = _find_indicator(tab)
        assert indicator is not None, tab
        classes = indicator["className"].split()
        self.assertNotIn(
            "done",
            classes,
            f"Running sub-agent indicator MUST NOT carry ``.done`` "
            f"(that would freeze the pulse).  className="
            f"{indicator['className']!r}",
        )
        self.assertEqual(indicator["textContent"], "◉")

    def test_css_done_rule_stops_animation_keeps_purple(self) -> None:
        """The CSS rule for ``.subagent-indicator.done`` must:

        * disable the pulse animation (``animation: none``), and
        * NOT recolour the indicator away from the default purple
          (no ``color:`` override that points elsewhere).
        """
        css = _MAIN_CSS.read_text(encoding="utf-8")
        m = re.search(
            r"\.chat-tab\.subagent-tab\s+\.subagent-indicator\.done\s*\{([^}]*)\}",
            css,
        )
        assert m is not None, (
            "Expected ``.chat-tab.subagent-tab .subagent-indicator.done`` "
            "rule in main.css"
        )
        block = m.group(1)
        self.assertIn(
            "animation: none", block,
            f"Done rule must include ``animation: none`` to stop the "
            f"pulse.  Got: {block!r}",
        )
        # No ``color:`` declaration that overrides the default purple.
        # (The base ``.subagent-indicator`` rule sets
        # ``color: var(--purple);`` — overriding it here would change
        # the done state's colour.)
        color_decl = re.search(r"\bcolor\s*:\s*([^;]+);", block)
        self.assertIsNone(
            color_decl,
            f"Done rule must NOT override ``color`` — the indicator "
            f"must stay purple (solid).  Got override: "
            f"{color_decl.group(1) if color_decl else None!r}",
        )

    def test_css_base_indicator_is_purple_and_pulses(self) -> None:
        """The base ``.subagent-indicator`` rule must keep the purple
        colour + pulse animation contract the running state depends
        on.
        """
        css = _MAIN_CSS.read_text(encoding="utf-8")
        m = re.search(
            r"\.subagent-indicator\s*\{([^}]*)\}",
            css,
        )
        assert m is not None, "Expected ``.subagent-indicator`` rule in main.css"
        block = m.group(1)
        self.assertIn("var(--purple)", block, block)
        self.assertIn("subagent-pulse", block, block)


if __name__ == "__main__":
    unittest.main()
