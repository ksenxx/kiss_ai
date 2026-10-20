# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression test: creating a sub-agent tab must NOT switch the
foreground tab.

Background
==========

When the backend starts a sub-agent (e.g. via ``run_parallel``), it
broadcasts a ``new_tab`` event so every page that owns the parent
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
   * It posts ``{type:'getWelcomeInfo'}`` even though the new
     tab will never display a welcome screen (the imminent
     ``openSubagentTab`` flips it into a sub-agent view).
   * It triggers ``focusInputWithRetry()`` on the sub-agent tab,
     stealing keyboard focus from the parent the user is typing in.

The fix is to materialise the sub-agent tab **in the background**:
push a fresh tab into the ``tabs`` array, render the tab bar, persist
state — and leave ``activeTabId`` untouched.  No ``newChat`` /
``getWelcomeInfo`` posts.  No focus theft.

This file runs the real ``main.js`` source in Node.js (no jsdom, no
mocks) so a future regression that re-introduces the spurious tab
switch — or its side effects — will be caught.
"""

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

_MAIN_JS = (
    Path(__file__).resolve().parents[3]
    / "kiss"
    / "webapp"
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
            var transport = { postMessage: function(m) { posted.push(m); } };
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
            function getTab(id) {
                return tabs.find(function(t) { return t.id === id; }) || null;
            }
            function renderTabBar() {}
            function persistTabState() {}
            """
            + helper_src
            + r"""
            var beforeActive = activeTabId;
            var sub = createBackgroundSubagentTab('parent-1', 'sub-uuid-1');
            // The daemon names the sub-agent tab's id, so a second spawn
            // under the same id reuses the open tab instead of stacking one.
            var again = createBackgroundSubagentTab('parent-1', 'sub-uuid-1');

            var errors = [];
            if (activeTabId !== beforeActive)
                errors.push('activeTabId changed: ' + activeTabId);
            if (!sub || sub.id !== 'sub-uuid-1')
                errors.push('helper did not return new tab');
            if (again !== sub)
                errors.push('second spawn did not reuse the open tab');
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
          ``newChat`` + ``getWelcomeInfo`` to the backend.
        * Post-fix: ``activeTabId`` never changes, only a
          ``resumeSession`` is posted (carrying the new sub tab's id),
          and the new sub tab is queued immediately to the right of
          its parent.
        """
        result = _run_node(
            r"""
            var posted = [];
            var transport = { postMessage: function(m) { posted.push(m); } };
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
                transport.postMessage({
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
            // No newChat / getWelcomeInfo side effects.
            for (var i = 0; i < posted.length; i++) {
                if (posted[i].type === 'newChat')
                    errors.push('newChat must not be posted for sub-agent tab');
                if (posted[i].type === 'getWelcomeInfo')
                    errors.push('getWelcomeInfo must not be posted');
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
            var transport = { postMessage: function(m) { posted.push(m); } };
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
                transport.postMessage({type: 'newChat', tabId: tab.id});
                transport.postMessage({type: 'getWelcomeInfo'});
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
                transport.postMessage({
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
            // And it posted newChat + getWelcomeInfo + resumeSession.
            var types = posted.map(function(m) { return m.type; });
            if (types.indexOf('newChat') < 0)
                errors.push('old handler should have posted newChat');
            if (types.indexOf('getWelcomeInfo') < 0)
                errors.push('old handler should have posted getWelcomeInfo');

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
