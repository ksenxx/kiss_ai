// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// The launch tab switch (main.js `switchToLatestRunningTab`) must stay
// spent after the user's gesture ended it, even when the gesture came
// BEFORE the running-task news.  Companion of launchTabSwitch.test.js's
// testBackendHiccupDoesNotRelaunch, which covers the news-then-tap order.
//
// Before the fix, `launchNewsSeen` was only recorded when the switch was
// still allowed, so a tap that pre-empted the news left it false; an
// overlay-type daemon restart (`daemonStatus connected:false` without
// `reconnecting`) then began a fresh launch and the replayed `status
// running` moved the user off the tab they had picked.

'use strict';

const assert = require('assert');
const {makePage: makeChatPage, send, stubRaf} = require('./simplify2_harness');

function makePage() {
  const {win} = makeChatPage({
    bodyClass: 'remote-chat',
    initialState: {chatId: 'tab-a'},
    beforeScripts: stubRaf,
    sourceUrl: 'launch-main.js',
  });
  send(win, {type: 'daemonStatus', connected: true});
  return win;
}

function tabsState(win, entries) {
  send(win, {
    type: 'tabs_state',
    tabs: entries.map(e => ({
      tabId: e.tabId,
      chatId: e.chatId,
      title: e.title,
      workDir: '',
    })),
  });
}

function userTaps(win) {
  win.document.dispatchEvent(
    new win.MouseEvent('pointerdown', {bubbles: true}),
  );
}

const TWO = [
  {tabId: 'tab-a', chatId: 'chat-a', title: 'alpha'},
  {tabId: 'tab-b', chatId: 'chat-b', title: 'beta'},
];

function testTapBeforeNewsThenRestartKeepsTheTab() {
  const win = makePage();
  tabsState(win, TWO);
  // The user taps before any running-task news arrives.
  userTaps(win);
  send(win, {type: 'status', running: true, tabId: 'tab-b', startTs: 500});
  assert.strictEqual(
    win._testApi.getActiveTabId(),
    'tab-a',
    'the tap ended the first launch',
  );
  // Daemon restart behind the loading overlay (no `reconnecting` banner),
  // then the registry resync and the replayed running status.
  send(win, {type: 'daemonStatus', connected: false});
  send(win, {type: 'daemonStatus', connected: true});
  tabsState(win, TWO);
  send(win, {type: 'status', running: true, tabId: 'tab-b', startTs: 500});
  assert.strictEqual(
    win._testApi.getActiveTabId(),
    'tab-a',
    'the restart must not re-arm the launch and move the user',
  );
  win.close();
  console.log('  ok - a tap before the news keeps its tab across a restart');
}

// News that arrives before the backend is live (no launch yet) must not
// spend the launch: the first launch after `connected` still switches.
function testNewsBeforeConnectDoesNotSpendTheLaunch() {
  const {win} = makeChatPage({
    bodyClass: 'remote-chat',
    initialState: {chatId: 'tab-a'},
    beforeScripts: stubRaf,
    sourceUrl: 'launch-main.js',
  });
  tabsState(win, TWO);
  send(win, {type: 'status', running: true, tabId: 'tab-b', startTs: 500});
  assert.strictEqual(
    win._testApi.getActiveTabId(),
    'tab-a',
    'no launch before the backend is live',
  );
  send(win, {type: 'daemonStatus', connected: true});
  tabsState(win, TWO);
  send(win, {type: 'status', running: true, tabId: 'tab-b', startTs: 500});
  assert.strictEqual(
    win._testApi.getActiveTabId(),
    'tab-b',
    'the launch after connect still switches',
  );
  win.close();
  console.log('  ok - news before connect does not spend the launch');
}

try {
  testTapBeforeNewsThenRestartKeepsTheTab();
  testNewsBeforeConnectDoesNotSpendTheLaunch();
  console.log('audit1020_launch_rearm.test.js passed');
  process.exit(0);
} catch (err) {
  console.error('FAIL:', err && err.message ? err.message : err);
  process.exit(1);
}
