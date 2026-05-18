"""Regression tests: clicking a task in the history sidebar must NOT
clobber an already-open chat tab's state with a redundant
``setTaskText`` + ``resumeSession`` when the tab keyed by the row's
chat_id is already open.

Sub-agent rows persist a minimal payload — only the parent
``task_history.id`` — and reopen as a regular chat tab that the
backend (``_replay_session``) then flips into a sub-agent tab via
``openSubagentTab``.  There is no separate sub-agent branch in the
history-click handler anymore.

Behavior contract
-----------------
- ``createNewTab(presetId)`` returns ``false`` when a tab keyed by
  ``presetId`` is already open (just ``switchToTab`` is invoked),
  and ``true`` when a new tab is actually allocated.

- The history-row click handler stores the return value of
  ``createNewTab(...)`` and only fires ``setTaskText`` /
  ``resumeSession`` when a new tab was actually created.
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
    """Return source from the first match of ``header_re`` to end of file."""
    m = re.search(header_re, js)
    assert m is not None, f"could not locate {header_re} in main.js"
    return js[m.start():]


class TestCreateNewTabReturnValue:
    """``createNewTab`` must signal whether a new tab was actually created."""

    def test_short_circuits_with_false_when_preset_id_matches(self) -> None:
        js = _read_main_js()
        body = _extract_block(js, r"function createNewTab\(presetId\)")
        m = re.search(
            r"const existingTab = tabs\.find\(t => t\.id === presetId\);"
            r"\s*if \(existingTab\) \{[^}]*return false;[^}]*\}",
            body,
        )
        assert m is not None, (
            "createNewTab must return false when a tab keyed by presetId "
            "already exists"
        )

    def test_returns_true_on_normal_path(self) -> None:
        js = _read_main_js()
        body = _extract_block(js, r"function createNewTab\(presetId\)")
        nc = body.index("vscode.postMessage({type: 'newChat'")
        rest = body[nc:]
        m = re.search(r"return true;\s*\}", rest)
        assert m is not None, (
            "createNewTab must end its fresh-tab path with `return true;`"
        )


class TestHistoryClickHandlerSkipsRedundantReplay:
    """The history-click handler guards ``setTaskText`` and
    ``resumeSession`` on ``createNewTab``'s return value so an
    already-open live tab does not get its panel overwritten."""

    def test_regular_row_guards_resume_on_create_return_value(self) -> None:
        js = _read_main_js()
        # Locate the regular branch by its comment header.
        tail = _extract_block(
            js,
            r"// When the clicked history row has a known chat_id \(s\.id\)",
        )
        m = re.search(
            r"const created = createNewTab\(\);\s*"
            r"if \(created\) \{\s*"
            r"setTaskText\(s\.preview \|\| s\.title \|\| ''\);\s*"
            r"vscode\.postMessage\(\{\s*"
            r"type: 'resumeSession',",
            tail,
        )
        assert m is not None, (
            "the has_events branch must wrap setTaskText + resumeSession "
            "in `if (created) { ... }` where "
            "`created = createNewTab()` — protects already-open live tabs"
        )


class TestSubagentRowFollowsRegularPath:
    """Sub-agent rows reopen via the same path as a regular task.  The
    backend's ``_replay_session`` flips the tab to ``isSubagentTab``
    via the ``openSubagentTab`` broadcast.  There must not be a
    separate ``s.is_subagent && s.subagent_tab_id`` branch in the
    history-click handler (it was removed when the persisted payload
    was reduced to ``{parent_task_id}``)."""

    def test_no_legacy_subagent_tab_id_branch(self) -> None:
        js = _read_main_js()
        assert "s.subagent_tab_id" not in js, (
            "the persisted payload no longer carries subagent_tab_id; "
            "the legacy `s.is_subagent && s.subagent_tab_id` branch in "
            "the history-click handler must be removed"
        )

    def test_no_legacy_parent_tab_id_field_on_sessions(self) -> None:
        js = _read_main_js()
        # ``s.parent_tab_id`` was forwarded on the session dict for
        # the badge placement; with ``treat as regular task`` it's
        # gone — only the backend computes the parent relationship.
        assert "s.parent_tab_id" not in js, (
            "frontend must not read s.parent_tab_id — the persisted "
            "payload only carries parent_task_id, used backend-side"
        )
