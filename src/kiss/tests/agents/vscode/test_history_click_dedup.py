"""Regression tests: clicking a task in the history sidebar must NOT

1. clobber an already-open chat tab's state with a redundant
   ``setTaskText`` + ``resumeSession`` when the tab keyed by the
   row's chat_id is already open, and

2. hijack the parent's chat tab when a sub-agent row whose own tab
   is NOT open is clicked — the new tab must be keyed by the
   persisted ``subagent_tab_id`` (so it never collides with the
   parent tab whose id IS the shared chat_id).

These tests pin the frontend behavior in ``media/main.js`` via
static pattern checks (the project's standard approach for webview
JS — see ``test_subagent_tabs_distinct.py``).

Behavior contract
-----------------
- ``createNewTab(presetId)`` returns ``false`` when a tab keyed by
  ``presetId`` is already open (just ``switchToTab`` is invoked),
  and ``true`` when a new tab is actually allocated.

- The history-row click handler stores the return value of
  ``createNewTab(...)`` and only fires ``setTaskText`` /
  ``resumeSession`` when a new tab was actually created.

- For sub-agent rows whose tab is NOT already open and have
  persisted events, the handler calls ``createNewTab`` with
  ``s.subagent_tab_id`` (the persisted sub-agent tab id), NOT
  ``s.id`` (the parent's chat_id).
"""

from __future__ import annotations

import re
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


def _extract_block(js: str, header_re: str) -> str:
    """Return source from the first match of ``header_re`` to end of file.

    Used to scope assertions to a single function/handler body.
    """
    m = re.search(header_re, js)
    assert m is not None, f"could not locate {header_re} in main.js"
    return js[m.start():]


def _slice_braces(js: str, start: int) -> str:
    """Return the balanced-braces block starting at the first ``{`` at/after start."""
    open_idx = js.index("{", start)
    depth = 0
    for i in range(open_idx, len(js)):
        ch = js[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return js[open_idx:i + 1]
    raise AssertionError("unbalanced braces from offset")


class TestCreateNewTabReturnValue:
    """``createNewTab`` must signal whether a new tab was actually created."""

    def test_short_circuits_with_false_when_preset_id_matches(self) -> None:
        js = _read_main_js()
        # Scope to the createNewTab function body.
        body = _extract_block(js, r"function createNewTab\(presetId\)")
        # Find the existingTab branch and confirm it returns false.
        m = re.search(
            r"const existingTab = tabs\.find\(t => t\.id === presetId\);"
            r"\s*if \(existingTab\) \{[^}]*return false;[^}]*\}",
            body,
        )
        assert m is not None, (
            "createNewTab must return false when a tab keyed by presetId "
            "already exists; the short-circuit `if (existingTab) { ... }` "
            "must end with `return false;`"
        )

    def test_returns_true_on_normal_path(self) -> None:
        js = _read_main_js()
        body = _extract_block(js, r"function createNewTab\(presetId\)")
        # The normal path posts ``newChat`` and ends with the final
        # ``return true;``.  Find the first ``return true;`` after the
        # ``newChat`` postMessage.
        nc = body.index("vscode.postMessage({type: 'newChat'")
        rest = body[nc:]
        m = re.search(r"return true;\s*\}", rest)
        assert m is not None, (
            "createNewTab must end its fresh-tab path with `return true;` "
            "so callers can distinguish a real create from a short-circuit"
        )


class TestHistoryClickHandlerSkipsRedundantReplay:
    """When ``createNewTab(s.id)`` short-circuits, setTaskText and resumeSession
    must NOT fire — otherwise an already-open live tab gets its panel text
    overwritten and a wasteful replay is broadcast back to it.
    """

    def test_regular_row_guards_resume_on_create_return_value(self) -> None:
        js = _read_main_js()
        # Locate the regular (non-subagent) branch.  We pin it by the
        # comment that introduces it and the exact ``createNewTab(s.id)``
        # call site (sub-agent branch uses ``s.subagent_tab_id``).
        tail = _extract_block(
            js,
            r"// When the clicked history row has a known chat_id \(s\.id\) and",
        )
        # The branch must capture createNewTab's return value and gate
        # both setTaskText and the resumeSession postMessage on it.
        m = re.search(
            r"const created = createNewTab\(s\.id\);\s*"
            r"if \(created\) \{\s*"
            r"setTaskText\(s\.preview \|\| s\.title \|\| ''\);\s*"
            r"vscode\.postMessage\(\{\s*"
            r"type: 'resumeSession',",
            tail,
        )
        assert m is not None, (
            "the regular has_events branch must wrap setTaskText + "
            "resumeSession in `if (created) { ... }` where "
            "`created = createNewTab(s.id)` — otherwise re-opening a "
            "history row for an already-open live tab clobbers its "
            "panel text and triggers a redundant replay"
        )


class TestSubagentRowDoesNotHijackParentTab:
    """When a sub-agent history row is clicked and its own tab is NOT
    already open, the new tab must be keyed by the persisted
    ``subagent_tab_id`` so it never collides with an open parent tab
    (the parent's tab id IS the shared chat_id ``s.id``).
    """

    def test_subagent_branch_uses_subagent_tab_id_for_create(self) -> None:
        js = _read_main_js()
        # Locate the sub-agent branch — pinned by its comment header and
        # the existing-tab early return on ``s.subagent_tab_id``.
        m_start = re.search(
            r"if \(s\.is_subagent && s\.subagent_tab_id\) \{",
            js,
        )
        assert m_start is not None
        block = _slice_braces(js, m_start.end() - 1)
        # The fallback branch (sub-tab not open + has events) must call
        # createNewTab with s.subagent_tab_id, NOT s.id.
        m_call = re.search(
            r"createNewTab\(s\.subagent_tab_id\)",
            block,
        )
        assert m_call is not None, (
            "sub-agent rows without an already-open sub-tab must "
            "createNewTab(s.subagent_tab_id) so the new tab id matches "
            "the persisted orig_sub_tab_id (avoids hijacking the "
            "parent's tab whose id == s.id)"
        )
        # And must NOT fall through to createNewTab(s.id) inside the
        # sub-agent branch.
        assert re.search(
            r"createNewTab\(s\.id\)",
            block,
        ) is None, (
            "sub-agent branch must not call createNewTab(s.id) — that "
            "is the parent chat_id and would hijack an open parent tab"
        )

    def test_subagent_branch_guards_resume_on_create_return_value(self) -> None:
        js = _read_main_js()
        m_start = re.search(
            r"if \(s\.is_subagent && s\.subagent_tab_id\) \{",
            js,
        )
        assert m_start is not None
        block = _slice_braces(js, m_start.end() - 1)
        m = re.search(
            r"const created = createNewTab\(s\.subagent_tab_id\);\s*"
            r"if \(created\) \{\s*"
            r"setTaskText\([^)]+\);\s*"
            r"vscode\.postMessage\(\{\s*"
            r"type: 'resumeSession',",
            block,
        )
        assert m is not None, (
            "sub-agent branch must gate setTaskText + resumeSession on "
            "createNewTab's return value (same symmetry as the regular "
            "branch); short-circuiting on an already-open sub-tab is "
            "already handled by the earlier `switchToTab` early return"
        )

    def test_subagent_branch_has_open_tab_early_return_unchanged(self) -> None:
        """The "sub-tab already open → switchToTab and return" early
        return must remain — this is the canonical case (sub-agent is
        currently running, frontend created the tab via
        ``openSubagentTab``).  We keep this pinned so the dedup logic
        for the running-sub-agent case is not accidentally removed.
        """
        js = _read_main_js()
        m_start = re.search(
            r"if \(s\.is_subagent && s\.subagent_tab_id\) \{",
            js,
        )
        assert m_start is not None
        block = _slice_braces(js, m_start.end() - 1)
        m = re.search(
            r"const existing = tabs\.find\(t => t\.id === s\.subagent_tab_id\);"
            r"\s*if \(existing\) \{\s*"
            r"switchToTab\(s\.subagent_tab_id\);\s*"
            r"closeSidebar\(\);\s*"
            r"return;\s*\}",
            block,
        )
        assert m is not None, (
            "the already-open sub-tab early return must remain intact"
        )
