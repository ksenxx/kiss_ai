"""Tests for the "remove oldest tabs when count exceeds 5" feature.

main.js limits the number of open chat tabs to MAX_TABS (5).  When the
user creates a new tab and the count would exceed MAX_TABS, the oldest
tab(s) are removed and a ``closeTab`` message is sent to the backend
for each one so server-side state is also released.

These tests verify both:

1. Source-level: the constant exists, ``trimOldestTabs`` exists, and
   ``createNewTab`` calls it.
2. Behavioral: ``trimOldestTabs`` extracted from main.js and run under
   Node with stub ``tabs`` and ``vscode`` actually trims the oldest
   entries and posts ``closeTab`` for each one.
"""

import json
import re
import shutil
import subprocess
import unittest
from pathlib import Path

MAIN_JS = (
    Path(__file__).resolve().parents[3]
    / "agents"
    / "vscode"
    / "media"
    / "main.js"
)


def _read_main_js() -> str:
    return MAIN_JS.read_text()


def _extract_balanced_function(src: str, signature: str) -> str:
    """Return the full source of ``function NAME(...) { ... }`` starting
    at ``signature`` (e.g. ``"function trimOldestTabs("``).  Assumes
    balanced ``{}`` and no braces inside strings/regex/comments after
    the opening brace, which holds for this small helper.
    """
    start = src.index(signature)
    brace = src.index("{", start)
    depth = 0
    i = brace
    while i < len(src):
        c = src[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return src[start : i + 1]
        i += 1
    raise AssertionError(f"unbalanced braces after {signature!r}")


class TestMaxTabsSource(unittest.TestCase):
    """The source of main.js declares MAX_TABS=5 and a trim helper."""

    js: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        cls.js = _read_main_js()

    def test_max_tabs_constant_is_five(self) -> None:
        """MAX_TABS is defined as the integer literal 5."""
        match = re.search(r"\bconst\s+MAX_TABS\s*=\s*(\d+)\s*;", self.js)
        assert match is not None, "MAX_TABS constant not found in main.js"
        assert match.group(1) == "5", (
            f"MAX_TABS must be 5; got {match.group(1)}"
        )

    def test_trim_function_exists(self) -> None:
        """trimOldestTabs is declared as a top-level function."""
        assert "function trimOldestTabs(" in self.js, (
            "expected function trimOldestTabs() in main.js"
        )

    def test_create_new_tab_calls_trim(self) -> None:
        """createNewTab invokes trimOldestTabs after appending the new tab."""
        body = _extract_balanced_function(self.js, "function createNewTab(")
        assert "trimOldestTabs(" in body, (
            "createNewTab must call trimOldestTabs() so that the count "
            "is enforced whenever a new tab is added"
        )
        push_idx = body.index("tabs.push(")
        trim_idx = body.index("trimOldestTabs(")
        assert trim_idx > push_idx, (
            "trimOldestTabs() must be called after the new tab is "
            "pushed so the count actually exceeds MAX_TABS first"
        )


@unittest.skipIf(shutil.which("node") is None, "node is not installed")
class TestTrimOldestTabsBehavior(unittest.TestCase):
    """Behavioral test: extract trimOldestTabs from main.js and run it
    under Node with stub ``tabs`` / ``vscode`` to verify it actually
    drops the oldest entries and posts a ``closeTab`` for each."""

    js: str = ""
    trim_fn: str = ""
    max_tabs_decl: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        cls.js = _read_main_js()
        cls.trim_fn = _extract_balanced_function(cls.js, "function trimOldestTabs(")
        match = re.search(r"\bconst\s+MAX_TABS\s*=\s*\d+\s*;", cls.js)
        assert match is not None
        cls.max_tabs_decl = match.group(0)

    def _run(self, initial_ids: list[str], active_id: str) -> dict:
        """Run trimOldestTabs on a stub ``tabs`` array and return the
        resulting state as JSON."""
        script = f"""
        {self.max_tabs_decl}
        let tabs = {json.dumps([{"id": tid} for tid in initial_ids])};
        let activeTabId = {json.dumps(active_id)};
        const __sent = [];
        const vscode = {{ postMessage: (m) => __sent.push(m) }};
        {self.trim_fn}
        trimOldestTabs();
        console.log(JSON.stringify({{
            ids: tabs.map(t => t.id),
            sent: __sent,
            activeTabId,
        }}));
        """
        out = subprocess.check_output(["node", "-e", script], text=True)
        result: dict = json.loads(out.strip().splitlines()[-1])
        return result

    def test_no_trim_when_at_or_below_limit(self) -> None:
        """With <= 5 tabs, trim is a no-op."""
        result = self._run(["a", "b", "c", "d", "e"], "e")
        assert result["ids"] == ["a", "b", "c", "d", "e"]
        assert result["sent"] == []

    def test_trims_one_oldest_when_over_by_one(self) -> None:
        """With 6 tabs, trim removes the oldest (index 0)."""
        result = self._run(["a", "b", "c", "d", "e", "f"], "f")
        assert result["ids"] == ["b", "c", "d", "e", "f"]
        assert result["sent"] == [{"type": "closeTab", "tabId": "a"}]

    def test_trims_multiple_when_over_by_more(self) -> None:
        """With 8 tabs, the three oldest are removed in order."""
        result = self._run(
            ["a", "b", "c", "d", "e", "f", "g", "h"], "h"
        )
        assert result["ids"] == ["d", "e", "f", "g", "h"]
        assert result["sent"] == [
            {"type": "closeTab", "tabId": "a"},
            {"type": "closeTab", "tabId": "b"},
            {"type": "closeTab", "tabId": "c"},
        ]

    def test_active_tab_unchanged_when_not_oldest(self) -> None:
        """The active tab is preserved when it is not among the oldest."""
        result = self._run(["a", "b", "c", "d", "e", "f"], "f")
        assert result["activeTabId"] == "f"


if __name__ == "__main__":
    unittest.main()
