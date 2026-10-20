// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// A live warning for a tab open in the background is routed into that
// tab's transcript (the active-tab, foreign-tab and escaping cases are
// bughunt3_warning_event.test.js).

'use strict';

const assert = require('assert');
const {makePage, send, stubRaf} = require('./simplify2_harness');

function testBackgroundTabWarningSurvivesTabSwitch() {
  const {win} = makePage({beforeScripts: stubRaf});
  const api = win._testApi;
  assert.ok(api, '_testApi must be exposed by main.js');

  const tab1 = api.getActiveTabId();
  assert.ok(tab1, 'initial tab id must exist');

  api.createNewTab();
  const tab2 = api.getActiveTabId();
  assert.ok(tab2 && tab2 !== tab1, 'a fresh second tab must be active');

  send(win, {type: 'system_output', text: 'control-sysout-QQQ7', tabId: tab1});
  send(win, {
    type: 'warning',
    message: 'bg stash-pop warning ZZZ9',
    tabId: tab1,
  });

  const activeText = win.document.getElementById('output').textContent;
  assert.ok(
    !activeText.includes('control-sysout-QQQ7') &&
      !activeText.includes('bg stash-pop warning ZZZ9'),
    'background-tab events must not render in the active tab',
  );

  assert.ok(
    api.openTabs().some(t => t.id === tab1),
    'tab1 must still be open in the background',
  );
  // The Chats-panel pick brings tab1 back on screen.
  api.switchToTab(tab1);
  assert.strictEqual(api.getActiveTabId(), tab1, 'tab1 must be active now');

  const text = win.document.getElementById('output').textContent;
  assert.ok(
    text.includes('control-sysout-QQQ7'),
    'control display event must be in the restored transcript ' +
      '(default bg-tab route)',
  );
  assert.ok(
    text.includes('bg stash-pop warning ZZZ9'),
    'BUG: a live warning for a background tab was dropped instead of ' +
      'being routed into the tab outputFragment — the user never sees ' +
      'that their uncommitted changes are stuck in the git stash',
  );
  win.close();
  console.log('  ok - bg-tab warning survives switching to the tab');
}

function runTests() {
  testBackgroundTabWarningSurvivesTabSwitch();
}

try {
  runTests();
  console.log('\n1 passed, 0 failed');
  process.exit(0);
} catch (err) {
  console.error('FAIL:', err && err.message ? err.message : err);
  process.exit(1);
}
