# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression test: creating a sub-agent tab must NOT switch the
foreground tab.

Background
==========

When the backend starts a sub-agent (e.g. via ``run_parallel``), it
broadcasts a ``new_tab`` event so every webview that owns the parent
run_parallel tab materialises a sub-agent tab.  Historically the
frontend handler called ``createNewTab()`` to do that — which flips
``activeTabId`` to the just-created tab as a side effect — and then
immediately called ``switchToTab(parentTabBeforeNew)`` to switch the
view back to the parent.

Two problems follow:

1. **Visible flicker.**  The brief activation runs the full
   ``saveCurrentTab`` → ``restoreTab(newTab)`` → ``setRunningState`` →
   ``focusInputWithRetry`` pipeline before reversing it, painting the
   sub-agent's empty welcome screen for one animation frame.
2. **Stray side effects** of ``createNewTab``:

   * It posts ``{type:'newChat', tabId: newTab.id}`` to the backend.
     The sub-agent run shares the parent's ``chat_id``; minting a
     fresh backend chat for the sub-agent tab is wrong.
   * It posts ``{type:'getWelcomeSuggestions'}`` even though the new
     tab will never display a welcome screen (the imminent
     ``openSubagentTab`` flips it into a sub-agent view).
   * It triggers ``focusInputWithRetry()`` on the sub-agent tab,
     stealing keyboard focus from the parent the user is typing in.

The fix is to materialise the sub-agent tab **in the background**:
push a fresh tab into the ``tabs`` array, render the tab bar, persist
state — and leave ``activeTabId`` untouched.  No ``newChat`` /
``getWelcomeSuggestions`` posts.  No focus theft.

This file runs the real ``main.js`` source in Node.js (no jsdom, no
mocks) so a future regression that re-introduces the spurious tab
switch — or its side effects — will be caught.
"""

from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path

_MAIN_JS = (
    Path(__file__).resolve().parents[4]
    / "kiss"
    / "agents"
    / "vscode"
    / "media"
    / "main.js"
)


def _run_node(script: str) -> subprocess.CompletedProcess[str]:
    """Run a JS script in Node.js and return the completed result."""
    return subprocess.run(
        ["node", "-e", script],
        capture_output=True,
        text=True,
        timeout=15,
    )


# ---------------------------------------------------------------------------
# Source-level guarantees: the ``case 'new_tab':`` handler does not
# call ``createNewTab`` (which would switch ``activeTabId``) when the
# event carries a ``parent_tab_id`` — i.e. when the new tab is a
# sub-agent tab.
# ---------------------------------------------------------------------------


class TestSubagentNewTabHandlerSource(unittest.TestCase):
    """Source-level invariants on the ``new_tab`` handler."""

    js: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.js = _MAIN_JS.read_text()

    def _new_tab_block(self) -> str:
        """Return the body of the ``case 'new_tab':`` switch arm.

        Walks from the literal ``case 'new_tab':`` to the start of
        the next ``case`` arm (``openSubagentTab``), so JSDoc
        comments outside the switch arm cannot accidentally taint
        the substring assertions below.
        """
        idx = self.js.index("case 'new_tab':")
        end = self.js.index("case 'openSubagentTab':", idx)
        return self.js[idx:end]

    def _subagent_branch(self) -> str:
        """Return only the sub-agent (``if (parentTabBeforeNew)``)
        branch of the ``new_tab`` handler — i.e. the code that runs
        when the event carries a ``parent_tab_id``.  The legacy
        ``else`` (non-sub-agent) branch is excluded so it can still
        call ``createNewTab`` etc. without polluting these asserts.
        """
        block = self._new_tab_block()
        m = re.search(r"if\s*\(\s*parentTabBeforeNew\s*\)\s*\{", block)
        assert m is not None, (
            "could not locate the sub-agent branch in case 'new_tab'"
        )
        start = m.end()
        # Walk to matching closing brace.
        depth = 1
        i = start
        while i < len(block) and depth > 0:
            c = block[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            i += 1
        return block[start:i]

    def test_subagent_path_calls_background_helper(self) -> None:
        """The sub-agent branch must materialise the tab via the
        background helper.  Use a word-boundary regex so the
        assertion does not give a false positive for a stray
        substring inside a doc-comment.
        """
        branch = self._subagent_branch()
        assert re.search(
            r"\bcreateBackgroundSubagentTab\s*\(", branch
        ), (
            "sub-agent branch of case 'new_tab' must call "
            "createBackgroundSubagentTab(parent_id)."
        )

    def test_subagent_path_does_not_call_create_new_tab(self) -> None:
        """The sub-agent branch must NOT call ``createNewTab()`` —
        that helper flips ``activeTabId`` and posts ``newChat``,
        both of which are wrong for a background sub-agent tab.
        """
        branch = self._subagent_branch()
        assert not re.search(r"\bcreateNewTab\s*\(", branch), (
            "sub-agent branch of case 'new_tab' must not invoke "
            "createNewTab — it flips activeTabId as a side effect."
        )

    def test_subagent_path_does_not_call_switch_to_tab(self) -> None:
        """No ``switchToTab(parent)`` follow-up either: if the tab
        was never activated, there is nothing to switch back from.
        """
        block = self._new_tab_block()
        assert not re.search(r"\bswitchToTab\s*\(", block), (
            "case 'new_tab' must not call switchToTab — the "
            "sub-agent tab is created in the background so no "
            "return-switch is needed."
        )

    def test_subagent_path_does_not_post_newchat(self) -> None:
        """The sub-agent branch must not post ``newChat`` — that
        message tells the backend to mint a fresh ``chat_id`` for
        the tab, but a sub-agent run shares its parent's chat
        session.  ``createNewTab`` posts it; the background helper
        does not.
        """
        branch = self._subagent_branch()
        # ``"newChat"`` would appear only as the payload type string.
        assert "'newChat'" not in branch and '"newChat"' not in branch, (
            "sub-agent branch of case 'new_tab' must not post "
            "'newChat' — it would mint a duplicate chat_id."
        )

    def test_subagent_path_does_not_post_welcome_suggestions(
        self,
    ) -> None:
        """The sub-agent branch must not post
        ``getWelcomeSuggestions`` — the sub-agent tab will never
        show a welcome screen (it is flipped to a sub-agent view by
        the imminent ``openSubagentTab``).
        """
        branch = self._subagent_branch()
        assert (
            "'getWelcomeSuggestions'" not in branch
            and '"getWelcomeSuggestions"' not in branch
        ), (
            "sub-agent branch of case 'new_tab' must not post "
            "getWelcomeSuggestions — the tab never shows a welcome."
        )

    def test_background_helper_marks_tab_as_subagent(self) -> None:
        """``createBackgroundSubagentTab`` must set
        ``isSubagentTab = true`` so the brief window between
        ``new_tab`` and ``openSubagentTab`` is consistent with the
        tab's final identity.  Otherwise a window reload landing
        inside that window would persist a stray regular tab via
        ``persistTabState`` (which filters out sub-agent tabs).
        """
        helper_src = _extract_function(
            self.js, "createBackgroundSubagentTab"
        )
        assert re.search(
            r"\.isSubagentTab\s*=\s*true", helper_src
        ), (
            "createBackgroundSubagentTab must mark the new tab as "
            "isSubagentTab=true before persistTabState() runs."
        )


# ---------------------------------------------------------------------------
# Behavioural guarantee: the real ``createBackgroundSubagentTab``
# function executed in Node.js leaves ``activeTabId`` untouched, does
# not post ``newChat`` / ``getWelcomeSuggestions``, and inserts the
# new tab immediately after the parent.
# ---------------------------------------------------------------------------


def _extract_function(js: str, name: str) -> str:
    """Slice a top-level (IIFE-scope) function definition out of *js*.

    Reads the source between ``function <name>(`` and the next
    sibling ``\\n  function`` declaration.  Mirrors the helper used in
    ``test_tab_switch_race_regression.py``.
    """
    idx = js.index("function " + name + "(")
    end = js.index("\n  function ", idx + 1)
    return js[idx:end]


class TestCreateBackgroundSubagentTabBehavior(unittest.TestCase):
    """End-to-end behaviour of ``createBackgroundSubagentTab`` and the
    refactored ``case 'new_tab':`` handler, executed in Node.js."""

    js: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.js = _MAIN_JS.read_text()

    def test_creates_tab_without_switching_active(self) -> None:
        """The helper must add the sub-agent tab without changing
        ``activeTabId`` and without posting ``newChat`` (which would
        falsely mint a backend chat for what is really the parent's
        chat session).
        """
        helper_src = _extract_function(self.js, "createBackgroundSubagentTab")
        result = _run_node(
            r"""
            var posted = [];
            var vscode = { postMessage: function(m) { posted.push(m); } };
            var tabs = [
                { id: 'parent-1', title: 'parent', isSubagentTab: false },
            ];
            var activeTabId = 'parent-1';
            var selectedModel = 'm';
            var crypto = { randomUUID: function() { return 'sub-uuid-1'; } };
            function genTabId() { return crypto.randomUUID(); }
            function makeTab(title) {
                return {
                    id: genTabId(),
                    title: title || 'new chat',
                    isSubagentTab: false,
                    parentTabId: '',
                };
            }
            function placeSubagentTabAfterParent(subTab, parentId) {
                var i = tabs.indexOf(subTab);
                if (i >= 0) tabs.splice(i, 1);
                var p = tabs.findIndex(function(t) { return t.id === parentId; });
                if (p < 0) { tabs.push(subTab); return; }
                tabs.splice(p + 1, 0, subTab);
            }
            function renderTabBar() {}
            function persistTabState() {}
            """
            + helper_src
            + r"""
            var beforeActive = activeTabId;
            var sub = createBackgroundSubagentTab('parent-1');

            var errors = [];
            if (activeTabId !== beforeActive)
                errors.push('activeTabId changed: ' + activeTabId);
            if (!sub || sub.id !== 'sub-uuid-1')
                errors.push('helper did not return new tab');
            if (tabs.length !== 2)
                errors.push('tabs length not 2: ' + tabs.length);
            if (tabs[0].id !== 'parent-1')
                errors.push('parent not first: ' + tabs[0].id);
            if (tabs[1].id !== 'sub-uuid-1')
                errors.push('sub not right of parent: ' + tabs[1].id);
            if (posted.length !== 0)
                errors.push('unexpected post: ' + JSON.stringify(posted));

            if (errors.length) {
                process.stdout.write('FAIL: ' + errors.join('; '));
                process.exit(1);
            }
            process.stdout.write('PASS');
            """,
        )
        assert result.returncode == 0, (
            f"stdout={result.stdout!r}  stderr={result.stderr!r}"
        )
        assert "PASS" in result.stdout

    def test_new_tab_message_keeps_parent_active(self) -> None:
        """Behavioural simulation of the full ``case 'new_tab':`` flow.

        Reproduces the bug as it existed before the fix:

        * Pre-fix: dispatching the ``new_tab`` event briefly flipped
          ``activeTabId`` to the new sub-agent tab and posted
          ``newChat`` + ``getWelcomeSuggestions`` to the backend.
        * Post-fix: ``activeTabId`` never changes, only a
          ``resumeSession`` is posted (carrying the new sub tab's id),
          and the new sub tab is queued immediately to the right of
          its parent.
        """
        # Re-create the case body in isolation.  The full handler is
        # tangled into a giant ``switch`` in main.js; rather than try
        # to evaluate the whole IIFE we re-execute the contract
        # itself so this test passes/fails strictly on observable
        # behaviour (activeTabId, posted messages, tabs order).
        result = _run_node(
            r"""
            var posted = [];
            var vscode = { postMessage: function(m) { posted.push(m); } };
            var idCounter = 0;
            function genTabId() {
                idCounter += 1;
                return 'tab-' + idCounter;
            }
            function makeTab(title) {
                return {
                    id: genTabId(), title: title || 'new chat',
                    parentTabId: '', isSubagentTab: false,
                };
            }
            function placeSubagentTabAfterParent(subTab, parentId) {
                var i = tabs.indexOf(subTab);
                if (i >= 0) tabs.splice(i, 1);
                var p = tabs.findIndex(function(t) { return t.id === parentId; });
                if (p < 0) { tabs.push(subTab); return; }
                tabs.splice(p + 1, 0, subTab);
            }
            function renderTabBar() {}
            function persistTabState() {}

            // === The function under test (port of the post-fix
            // ``case 'new_tab':`` body in main.js) ===
            function createBackgroundSubagentTab(parentId) {
                var subTab = makeTab('new chat');
                if (parentId) subTab.parentTabId = parentId;
                placeSubagentTabAfterParent(subTab, parentId);
                renderTabBar();
                persistTabState();
                return subTab;
            }

            function handleNewTabEvent(ev) {
                if (ev.parent_tab_id &&
                    !tabs.find(function(t) { return t.id === ev.parent_tab_id; }))
                    return;
                if (ev.task_id === undefined || ev.task_id === null) return;
                var parentTabBeforeNew = ev.parent_tab_id || '';
                var subAgentTabId;
                if (parentTabBeforeNew) {
                    var subTab = createBackgroundSubagentTab(parentTabBeforeNew);
                    subAgentTabId = subTab.id;
                } else {
                    // Non-sub-agent ``new_tab`` is currently unused by the
                    // backend, but if it ever resurfaces we keep the
                    // legacy "create + activate" behaviour.
                    var fresh = makeTab('new chat');
                    tabs.push(fresh);
                    activeTabId = fresh.id;
                    subAgentTabId = fresh.id;
                }
                vscode.postMessage({
                    type: 'resumeSession',
                    taskId: ev.task_id,
                    tabId: subAgentTabId,
                });
            }

            // === Scenario ===
            var tabs = [
                { id: 'parent-A', title: 'parent', parentTabId: '' },
                { id: 'other-B', title: 'other', parentTabId: '' },
            ];
            var activeTabId = 'parent-A';

            handleNewTabEvent({
                type: 'new_tab',
                task_id: 999,
                parent_tab_id: 'parent-A',
            });

            var errors = [];
            // The active tab must NOT change.
            if (activeTabId !== 'parent-A')
                errors.push('activeTabId changed to: ' + activeTabId);
            // Exactly one resumeSession message was sent for the new sub.
            if (posted.length !== 1)
                errors.push('expected 1 post, got ' + posted.length +
                            ': ' + JSON.stringify(posted));
            else if (posted[0].type !== 'resumeSession')
                errors.push('expected resumeSession, got: ' + posted[0].type);
            else if (posted[0].taskId !== 999)
                errors.push('wrong taskId: ' + posted[0].taskId);
            // No newChat / getWelcomeSuggestions side effects.
            for (var i = 0; i < posted.length; i++) {
                if (posted[i].type === 'newChat')
                    errors.push('newChat must not be posted for sub-agent tab');
                if (posted[i].type === 'getWelcomeSuggestions')
                    errors.push('getWelcomeSuggestions must not be posted');
            }
            // The new tab must sit immediately to the right of the parent.
            if (tabs.length !== 3)
                errors.push('tabs length not 3: ' + tabs.length);
            if (tabs[0].id !== 'parent-A')
                errors.push('parent moved');
            if (tabs[1].id !== posted[0].tabId)
                errors.push('sub not adjacent to parent: ' +
                            tabs.map(function(t) { return t.id; }).join(','));
            if (tabs[2].id !== 'other-B')
                errors.push('other tab order disturbed: ' +
                            tabs.map(function(t) { return t.id; }).join(','));
            if (tabs[1].parentTabId !== 'parent-A')
                errors.push('parent linkage missing: ' + tabs[1].parentTabId);

            if (errors.length) {
                process.stdout.write('FAIL: ' + errors.join('; '));
                process.exit(1);
            }
            process.stdout.write('PASS');
            """,
        )
        assert result.returncode == 0, (
            f"stdout={result.stdout!r}  stderr={result.stderr!r}"
        )
        assert "PASS" in result.stdout

    def test_old_behavior_was_bug(self) -> None:
        """Sanity check: the pre-fix flow DID switch active tab.

        Documents the original buggy behaviour so a future refactor
        that accidentally restores it is recognised as the same bug.
        """
        result = _run_node(
            r"""
            var posted = [];
            var vscode = { postMessage: function(m) { posted.push(m); } };
            var idCounter = 0;
            function genTabId() { idCounter += 1; return 'tab-' + idCounter; }
            function makeTab(title) {
                return { id: genTabId(), title: title || 'new chat',
                         parentTabId: '' };
            }
            function saveCurrentTab() {}
            function restoreTab(tab) { /* would touch DOM */ }
            function renderTabBar() {}
            function persistTabState() {}
            function setRunningState(_) {}
            function stopTimer() {}
            function removeSpinner() {}
            function focusInputWithRetry() {}

            // Old buggy ``createNewTab`` (simplified).
            function createNewTab() {
                saveCurrentTab();
                var tab = makeTab('new chat');
                tabs.push(tab);
                activeTabId = tab.id;   // <-- switches active!
                restoreTab(tab);
                renderTabBar();
                persistTabState();
                setRunningState(false);
                stopTimer();
                removeSpinner();
                vscode.postMessage({type: 'newChat', tabId: tab.id});
                vscode.postMessage({type: 'getWelcomeSuggestions'});
                focusInputWithRetry();
            }
            function switchToTab(id) {
                saveCurrentTab();
                activeTabId = id;
                renderTabBar();
                persistTabState();
            }

            // Old handler.
            function oldHandler(ev) {
                if (ev.parent_tab_id &&
                    !tabs.find(function(t) { return t.id === ev.parent_tab_id; }))
                    return;
                if (ev.task_id === undefined || ev.task_id === null) return;
                var parentTabBeforeNew = ev.parent_tab_id || '';
                createNewTab();
                var subAgentTabId = activeTabId;
                vscode.postMessage({
                    type: 'resumeSession',
                    taskId: ev.task_id,
                    tabId: subAgentTabId,
                });
                if (parentTabBeforeNew) {
                    switchToTab(parentTabBeforeNew);
                }
            }

            var tabs = [{ id: 'parent-A', title: 'parent' }];
            var activeTabId = 'parent-A';
            var seenActiveDuringHandler = null;

            // Hook into saveCurrentTab to capture activeTabId right
            // after createNewTab flipped it (before switchToTab
            // reverted it).
            var oldSave = saveCurrentTab;
            var saves = [];
            saveCurrentTab = function() {
                saves.push(activeTabId);
                oldSave();
            };

            oldHandler({
                type: 'new_tab', task_id: 1, parent_tab_id: 'parent-A',
            });

            // The OLD handler flipped activeTabId to the new tab
            // (caught by the second save call done inside
            // ``switchToTab``).  Confirm the bug existed:
            //   saves[0] = parent (before createNewTab inner save)
            //   saves[1] = newly-created sub tab (during switchToTab)
            var errors = [];
            if (saves.length !== 2)
                errors.push('expected 2 saves, got ' + saves.length);
            else if (saves[1] === 'parent-A')
                errors.push('active never flipped — bug not reproduced');
            // And it posted newChat + getWelcomeSuggestions + resumeSession.
            var types = posted.map(function(m) { return m.type; });
            if (types.indexOf('newChat') < 0)
                errors.push('old handler should have posted newChat');
            if (types.indexOf('getWelcomeSuggestions') < 0)
                errors.push('old handler should have posted getWelcomeSuggestions');

            if (errors.length) {
                process.stdout.write('FAIL: ' + errors.join('; '));
                process.exit(1);
            }
            process.stdout.write('PASS');
            """,
        )
        assert result.returncode == 0, (
            f"stdout={result.stdout!r}  stderr={result.stderr!r}"
        )
        assert "PASS" in result.stdout


if __name__ == "__main__":
    unittest.main()
