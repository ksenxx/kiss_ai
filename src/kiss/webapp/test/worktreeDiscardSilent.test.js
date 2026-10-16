// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// A plain successful worktree discard is silent in the transcript: the
// user asked for it and the branch is gone, there is nothing to read.
// Merge results, discards that carry a warning, and failed discards are
// still printed -- and a background tab's printed results are buffered
// for it, never rendered into the tab on screen.

'use strict';

const assert = require('assert');
const {makePage, send} = require('./simplify2_harness.js');

function main() {
  const {win, posted} = makePage();
  const ready = posted.find(m => m.type === 'ready');
  assert.ok(ready && ready.tabId, 'the page must post ready with a tabId');
  const activeTab = ready.tabId;

  send(win, {
    type: 'worktree_result',
    success: true,
    message: "Discarded branch 'kiss/wt-20260101-120000'.",
    tabId: activeTab,
  });
  assert.strictEqual(
    win.document.querySelectorAll('.wt-result-ok, .wt-result-err').length,
    0,
    'a plain successful discard must not print a line in the chat',
  );

  send(win, {
    type: 'worktree_result',
    success: true,
    message: "Successfully merged branch 'kiss/wt-1' into 'main'.",
    tabId: activeTab,
  });
  assert.strictEqual(
    win.document.querySelectorAll('.wt-result-ok').length,
    1,
    'a merge result must still be printed in the chat',
  );

  send(win, {
    type: 'worktree_result',
    success: true,
    message:
      "Discarded branch 'kiss/wt-2'.\n\u26a0\ufe0f  Could not checkout 'main': dirty",
    tabId: activeTab,
  });
  assert.strictEqual(
    win.document.querySelectorAll('.wt-result-ok').length,
    2,
    'a discard result with a warning must still be printed',
  );

  const api = win._testApi;
  assert.ok(api, '_testApi must be exposed by main.js');
  const tab1 = api.getActiveTabId();
  api.createNewTab();
  const tab2 = api.getActiveTabId();
  assert.ok(tab2 && tab2 !== tab1, 'a fresh second tab must be active');
  send(win, {
    type: 'worktree_result',
    success: true,
    message: "Discarded branch 'kiss/wt-bg'.",
    tabId: tab1,
  });
  send(win, {
    type: 'worktree_result',
    success: false,
    message: 'discard failed: worktree locked BGERR1',
    tabId: tab1,
  });
  const activeText = win.document.getElementById('output').textContent;
  assert.ok(
    !activeText.includes("Discarded branch 'kiss/wt-bg'.") &&
      !activeText.includes('BGERR1'),
    'background-tab worktree results must not render in the active tab',
  );
  // Chat tabs have no row of their own: the Chats panel's pick is
  // switchToTab.
  assert.ok(api.openTabs().some(t => t.id === tab1), 'tab1 must still be open');
  api.switchToTab(tab1);
  assert.strictEqual(api.getActiveTabId(), tab1, 'tab1 must be active now');
  const restored = win.document.getElementById('output').textContent;
  assert.ok(
    !restored.includes("Discarded branch 'kiss/wt-bg'."),
    'a silent discard must not be buffered into a background tab transcript',
  );
  assert.ok(
    restored.includes('BGERR1'),
    'a FAILED discard for a background tab must still be buffered and shown',
  );
  win.close();
  console.log('worktreeDiscardSilent: all passed.');
}

main();
