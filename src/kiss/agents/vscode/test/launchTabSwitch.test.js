// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end tests for the launch tab switch, in the extension webview and in
// the remote web app.
//
// Opening a chat window while agents are still working should put the newest
// of those tasks on screen -- that is the one the user just walked away from.
// Both clients learn about them the same way now: the daemon answers `ready`
// with the shared-registry `tabs_state` snapshot and then replays every
// chat-bound tab, broadcasting a `status {running:true, startTs}` for each
// task that is still going.
//
// The permission to move the user lasts only until the first real gesture, so
// a daemon that replays mid-session (a reconnect says `ready` again, and the
// replay statuses come back) cannot yank anybody off the transcript they are
// reading. See tabsStateNoTabSwitch.test.js for that half of the contract.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

// A launched window whose backend is live. `daemonStatus connected` is how
// both clients announce that: the remote socket shim synthesises it the moment
// the password is accepted (right before it flushes the queued `ready`), and
// the extension host sends it from its own `ready` handler. Until it arrives
// the chat sits behind the "server is starting" overlay, so a launch has not
// begun and no gesture can end one.
function makeWebview(opts) {
  const {remote = false, state, connect = true} = opts || {};
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
  html = html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');
  if (remote) html = html.replace('<body', '<body class="remote-chat"');

  const dom = new JSDOM(html, {
    runScripts: 'dangerously',
    pretendToBeVisual: true,
    url: 'https://localhost/',
  });
  const win = dom.window;
  win.Element.prototype.scrollIntoView = function () {};
  win.Element.prototype.scrollTo = function () {};
  win.HTMLElement.prototype.scrollTo = function () {};
  win.requestAnimationFrame = function (cb) {
    cb();
    return 0;
  };

  const posted = [];
  let persisted = state;
  win.acquireVsCodeApi = function () {
    return {
      postMessage: msg => posted.push(msg),
      getState: () => persisted,
      setState: s => {
        persisted = s;
      },
    };
  };

  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(
    fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8') +
      '\n//# sourceURL=launch-main.js',
  );
  if (connect) send(win, {type: 'daemonStatus', connected: true});
  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function activeTabId(win) {
  return win._testApi.getActiveTabId();
}

function tabIds(win) {
  return Array.from(win.document.querySelectorAll('.chat-tab'))
    .map(el => el.getAttribute('data-tab-id'))
    .filter(id => !!id);
}

// A real click in the tab bar: it carries the pointerdown a user's finger
// would, which is what ends a launch.
function clickTab(win, tabId) {
  const el = win.document.querySelector(
    `.chat-tab[data-tab-id=${JSON.stringify(tabId)}]`,
  );
  assert.ok(el, `tab ${tabId} must exist in the tab bar`);
  el.dispatchEvent(new win.MouseEvent('pointerdown', {bubbles: true}));
  el.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

// The registry snapshot the daemon broadcasts after `ready` (and after every
// registry mutation): the canonical tab set every client reconciles against.
function tabsState(win, entries) {
  send(win, {
    type: 'tabs_state',
    tabs: entries.map(e => ({
      tabId: e.tabId,
      chatId: e.chatId || '',
      title: e.title || 'new chat',
      workDir: e.workDir || '',
    })),
  });
}

// The replay's running-task news: the daemon broadcasts one of these for
// every registry tab whose task is still going.
function statusRunning(win, tabId, startTs) {
  send(win, {type: 'status', running: true, tabId: tabId, startTs: startTs});
}

// The remote web app: the replay names one running task, so that is where
// the window opens.
function testRemoteSingleRunningTaskTakesTheScreen() {
  const {win} = makeWebview({remote: true});
  const boot = activeTabId(win);

  tabsState(win, [
    {tabId: 'tab-1', chatId: 'chat-1', title: 'ship the parser'},
  ]);
  statusRunning(win, 'tab-1', 7);

  assert.ok(
    !tabIds(win).includes(boot),
    'the boot placeholder is replaced by the canonical snapshot',
  );
  assert.strictEqual(
    activeTabId(win),
    'tab-1',
    'launching with a running task must put that task on screen',
  );
  win.close();
  console.log('  ok - a single running task is on screen at launch');
}

// Several running tasks: the newest wins, whatever order the replays land in.
function testRemoteNewestRunningTaskWins() {
  for (const order of ['ascending', 'shuffled']) {
    const rows = [
      {tabId: 'tab-old', startTs: 100},
      {tabId: 'tab-new', startTs: 300},
      {tabId: 'tab-mid', startTs: 200},
    ];
    rows.sort((a, b) =>
      order === 'ascending' ? a.startTs - b.startTs : b.startTs - a.startTs,
    );
    const {win} = makeWebview({remote: true});
    tabsState(win, [
      {tabId: 'tab-old', chatId: 'chat-old', title: 'old'},
      {tabId: 'tab-new', chatId: 'chat-new', title: 'new'},
      {tabId: 'tab-mid', chatId: 'chat-mid', title: 'mid'},
    ]);
    for (const row of rows) statusRunning(win, row.tabId, row.startTs);
    assert.strictEqual(
      activeTabId(win),
      'tab-new',
      `the newest task must win (replays ${order})`,
    );
    assert.deepStrictEqual(
      tabIds(win),
      ['tab-old', 'tab-new', 'tab-mid'],
      'every running task must still get its own tab, in registry order',
    );
    win.close();
  }
  console.log('  ok - the newest running task wins, in any replay order');
}

// Timestamps go missing when a task predates the history row. The backend
// replays its registry oldest first, so the last tab is still the newest.
function testRemoteMissingTimestampsFallBackToOrder() {
  const {win} = makeWebview({remote: true});
  tabsState(win, [
    {tabId: 'tab-first', chatId: 'chat-first', title: 'first'},
    {tabId: 'tab-last', chatId: 'chat-last', title: 'last'},
  ]);
  statusRunning(win, 'tab-first');
  statusRunning(win, 'tab-last');
  assert.strictEqual(
    activeTabId(win),
    'tab-last',
    'with no timestamps the last registry tab is the newest task',
  );
  win.close();
  console.log('  ok - replays without timestamps fall back to registry order');
}

// A chat the window already had a tab for is switched to, not duplicated:
// a second snapshot listing the same tabs must not grow the tab bar, and the
// replayed status lands on the tab that was already there.
function testRemoteAlreadyOpenRunningChatIsSwitchedTo() {
  const {win, posted} = makeWebview({
    remote: true,
    state: {chatId: 'tab-read'},
  });
  const entries = [
    {tabId: 'tab-read', chatId: 'chat-read', title: 'reading'},
    {tabId: 'tab-work', chatId: 'chat-work', title: 'working'},
  ];
  tabsState(win, entries);
  assert.strictEqual(activeTabId(win), 'tab-read', 'restored the selection');
  const before = tabIds(win);

  tabsState(win, entries);
  statusRunning(win, 'tab-work', 42);

  assert.deepStrictEqual(
    tabIds(win),
    before,
    'a chat that already has a tab must not get a second one',
  );
  assert.strictEqual(
    posted.filter(m => m.type === 'resumeSession').length,
    0,
    'the daemon replays registry tabs itself; the client resumes nothing',
  );
  assert.strictEqual(
    activeTabId(win),
    'tab-work',
    'launching must land on the tab that already holds the running task',
  );
  win.close();
  console.log('  ok - an already-open running chat is switched to in place');
}

// Nothing running: the window opens where the user left it.
function testNoRunningTaskKeepsTheRestoredTab() {
  const {win} = makeWebview({
    remote: true,
    state: {chatId: 'tab-2'},
  });
  tabsState(win, [
    {tabId: 'tab-1', chatId: 'chat-1', title: 'one'},
    {tabId: 'tab-2', chatId: 'chat-2', title: 'two'},
  ]);
  send(win, {type: 'status', running: false, tabId: 'tab-1'});
  assert.strictEqual(
    activeTabId(win),
    'tab-2',
    'with nothing running the restored selection stays on screen',
  );
  win.close();
  console.log('  ok - an idle launch keeps the restored tab');
}

// Junk in the snapshot must not move anybody or tear tabs down.
function testMalformedSnapshotDoesNotSwitch() {
  const {win, posted} = makeWebview({remote: true});
  const boot = activeTabId(win);
  send(win, {type: 'tabs_state', tabs: [null, {}, {taskId: 'no-tab-id'}]});
  send(win, {type: 'tabs_state', tabs: 'not-an-array'});
  send(win, {type: 'tabs_state'});
  statusRunning(win, 'tab-nowhere', 5);
  assert.strictEqual(activeTabId(win), boot, 'the active tab is kept');
  assert.deepStrictEqual(tabIds(win), [boot], 'no tab may appear or vanish');
  assert.strictEqual(
    posted.filter(m => m.type === 'resumeSession').length,
    0,
    'nothing may be resumed',
  );
  win.close();
  console.log('  ok - a malformed snapshot switches nothing');
}

// The extension webview takes the same path: the daemon replays a `status`
// per registry tab; the newest of them takes the screen.
function testExtensionNewestRestoredStatusWins() {
  const {win} = makeWebview({state: {chatId: 'tab-a'}});
  tabsState(win, [
    {tabId: 'tab-a', chatId: 'chat-a', title: 'alpha'},
    {tabId: 'tab-b', chatId: 'chat-b', title: 'beta'},
    {tabId: 'tab-c', chatId: 'chat-c', title: 'gamma'},
  ]);
  assert.strictEqual(activeTabId(win), 'tab-a', 'restored on the first tab');

  statusRunning(win, 'tab-b', 500);
  assert.strictEqual(
    activeTabId(win),
    'tab-b',
    'the first running task found at launch takes the screen',
  );

  statusRunning(win, 'tab-c', 900);
  assert.strictEqual(
    activeTabId(win),
    'tab-c',
    'a newer running task replaces it',
  );

  statusRunning(win, 'tab-a', 100);
  assert.strictEqual(
    activeTabId(win),
    'tab-c',
    'an older running task must not pull the user back',
  );
  win.close();
  console.log('  ok - the extension lands on the newest restored task');
}

// Typing the remote password must not spend the launch. The socket shim holds
// `ready` back until the password is accepted, so every one of those
// keystrokes lands before the backend is live -- and the chat is behind the
// "server is starting" overlay the whole time.
function testGestureBeforeTheBackendIsLiveKeepsTheLaunch() {
  const {win} = makeWebview({remote: true, connect: false});

  win.document.dispatchEvent(
    new win.MouseEvent('pointerdown', {bubbles: true}),
  );
  win.document.dispatchEvent(
    new win.KeyboardEvent('keydown', {key: 'p', bubbles: true}),
  );

  send(win, {type: 'daemonStatus', connected: true});
  tabsState(win, [
    {tabId: 'tab-1', chatId: 'chat-1', title: 'still going'},
  ]);
  statusRunning(win, 'tab-1', 5);

  assert.strictEqual(
    activeTabId(win),
    'tab-1',
    'a password keystroke must not spend the launch it precedes',
  );
  win.close();
  console.log('  ok - a gesture before the backend is live keeps the launch');
}

// The real password-protected sequence. The socket shim reports a live backend
// TWICE: once on `auth_required`, only to reveal the app so its password modal
// (a child of #app) can render, and again on `auth_ok` -- and only the second
// one lets `ready` through. Every keystroke of the password lands between the
// two, so the launch has to survive them.
function testPasswordPromptDoesNotSpendTheLaunch() {
  const {win} = makeWebview({remote: true, connect: false});

  // auth_required: the app is revealed so the modal can be seen.
  send(win, {type: 'daemonStatus', connected: true});
  win.document.dispatchEvent(
    new win.MouseEvent('pointerdown', {bubbles: true}),
  );
  for (const key of ['s', 'e', 'c', 'r', 'e', 't', 'Enter']) {
    win.document.dispatchEvent(
      new win.KeyboardEvent('keydown', {key: key, bubbles: true}),
    );
  }

  // auth_ok: the password was accepted, so the queued `ready` goes out and
  // the snapshot and replays come back.
  send(win, {type: 'daemonStatus', connected: true});
  tabsState(win, [
    {tabId: 'tab-1', chatId: 'chat-1', title: 'still going'},
  ]);
  statusRunning(win, 'tab-1', 5);

  assert.strictEqual(
    activeTabId(win),
    'tab-1',
    'typing the remote password must not spend the launch it precedes',
  );
  win.close();
  console.log('  ok - the password prompt does not spend the launch');
}

// Landing on a task means staying there when it finishes: the result is what
// the user was brought to see.
function testTaskFinishingDoesNotMoveTheUser() {
  const {win} = makeWebview({remote: true});
  tabsState(win, [
    {tabId: 'tab-slow', chatId: 'chat-slow', title: 'slow'},
    {tabId: 'tab-quick', chatId: 'chat-quick', title: 'quick'},
  ]);

  // Both replays report themselves running, so the launch is fully resolved.
  statusRunning(win, 'tab-slow', 100);
  statusRunning(win, 'tab-quick', 900);
  assert.strictEqual(activeTabId(win), 'tab-quick', 'the newest task won');

  send(win, {type: 'status', running: false, tabId: 'tab-quick'});
  assert.strictEqual(
    activeTabId(win),
    'tab-quick',
    'a task finishing must leave the user on its result, not move them to ' +
      'an older task that happens to still be running',
  );
  win.close();
  console.log('  ok - a task finishing never moves the user');
}

// A task that had already finished by the time its replay landed never won
// the launch in the first place, so the still-running one takes the screen.
function testFinishedTaskIsNoLongerACandidate() {
  const {win} = makeWebview({remote: true});
  tabsState(win, [
    {tabId: 'tab-slow', chatId: 'chat-slow', title: 'slow'},
    {tabId: 'tab-quick', chatId: 'chat-quick', title: 'quick'},
  ]);
  statusRunning(win, 'tab-quick', 900);
  assert.strictEqual(activeTabId(win), 'tab-quick', 'the newest task won');

  // The quick task finishes: its replay lands the closing status, and the
  // slow one's replay still says it is running.
  send(win, {type: 'status', running: false, tabId: 'tab-quick'});
  statusRunning(win, 'tab-slow', 100);
  assert.strictEqual(
    activeTabId(win),
    'tab-slow',
    'once the newest task is done the launch falls through to the one ' +
      'still running',
  );
  win.close();
  console.log('  ok - a finished task drops out of the launch candidates');
}

// The first tap ends the launch: replays after it must not move the user.
function testPointerDownEndsTheLaunch() {
  const {win} = makeWebview({remote: true, state: {chatId: 'tab-here'}});
  tabsState(win, [
    {tabId: 'tab-here', chatId: 'chat-here', title: 'reading'},
  ]);

  win.document.dispatchEvent(
    new win.MouseEvent('pointerdown', {bubbles: true}),
  );
  tabsState(win, [
    {tabId: 'tab-here', chatId: 'chat-here', title: 'reading'},
    {tabId: 'tab-bg', chatId: 'chat-bg', title: 'background'},
  ]);
  statusRunning(win, 'tab-bg', 9);

  assert.strictEqual(
    activeTabId(win),
    'tab-here',
    'a tap ends the launch, so a reconnect replay must not switch tabs',
  );
  assert.ok(
    tabIds(win).includes('tab-bg'),
    'the task must still get a reachable background tab',
  );
  win.close();
  console.log('  ok - a tap ends the launch');
}

// So does the first keystroke.
function testKeyDownEndsTheLaunch() {
  const {win} = makeWebview({state: {chatId: 'tab-a'}});
  tabsState(win, [
    {tabId: 'tab-a', chatId: 'chat-a', title: 'alpha'},
    {tabId: 'tab-b', chatId: 'chat-b', title: 'beta'},
  ]);
  win.document.dispatchEvent(
    new win.KeyboardEvent('keydown', {key: 'a', bubbles: true}),
  );
  statusRunning(win, 'tab-b', 500);
  assert.strictEqual(
    activeTabId(win),
    'tab-a',
    'a keystroke ends the launch, so a status must not switch tabs',
  );
  win.close();
  console.log('  ok - a keystroke ends the launch');
}

// A daemon that dies and comes back mid-session hides the chat behind the
// loading overlay and replays everything when it returns. That must not hand
// the launch a second chance at a user who has already picked their tab.
function testBackendHiccupDoesNotRelaunch() {
  const {win} = makeWebview({state: {chatId: 'tab-a'}});
  tabsState(win, [
    {tabId: 'tab-a', chatId: 'chat-a', title: 'alpha'},
    {tabId: 'tab-b', chatId: 'chat-b', title: 'beta'},
  ]);
  // The launch runs its course and lands on the running task.
  statusRunning(win, 'tab-b', 500);
  assert.strictEqual(activeTabId(win), 'tab-b', 'the launch had its chance');

  // The user goes back to the other chat by hand.
  clickTab(win, 'tab-a');
  assert.strictEqual(activeTabId(win), 'tab-a', 'the user chose this tab');

  // The daemon drops and returns, resyncing the registry and replaying
  // every running task.
  send(win, {type: 'daemonStatus', connected: false});
  send(win, {type: 'daemonStatus', connected: true});
  tabsState(win, [
    {tabId: 'tab-a', chatId: 'chat-a', title: 'alpha'},
    {tabId: 'tab-b', chatId: 'chat-b', title: 'beta'},
  ]);
  statusRunning(win, 'tab-b', 500);
  assert.strictEqual(
    activeTabId(win),
    'tab-a',
    'a backend hiccup must not relaunch and move the user',
  );
  win.close();
  console.log('  ok - a backend hiccup does not relaunch');
}

// A window left untouched is not launching forever: a task that starts much
// later must not steal a tab from a screen somebody may be watching.
function testLaunchWindowExpires() {
  const {win} = makeWebview({state: {chatId: 'tab-a'}});
  tabsState(win, [
    {tabId: 'tab-a', chatId: 'chat-a', title: 'alpha'},
    {tabId: 'tab-b', chatId: 'chat-b', title: 'beta'},
  ]);
  const realNow = win.Date.now();
  win.Date.now = function () {
    return realNow + 60000;
  };
  statusRunning(win, 'tab-b', realNow + 60000);
  assert.strictEqual(
    activeTabId(win),
    'tab-a',
    'a task starting a minute after launch must not switch tabs',
  );
  win.close();
  console.log('  ok - the launch window expires');
}

// A sub-agent tab is an implementation detail of the task that spawned it, so
// it is never a launch target: the parent chat is.
function testSubagentTabIsNotALaunchTarget() {
  const {win} = makeWebview({state: {chatId: 'tab-a'}});
  tabsState(win, [{tabId: 'tab-a', chatId: 'chat-a', title: 'alpha'}]);
  send(win, {
    type: 'openSubagentTab',
    tab_id: 'tab-sub',
    parent_tab_id: 'tab-a',
    description: 'help out',
    task_id: 'sub-1',
  });
  const subTabId = tabIds(win).find(id => id !== 'tab-a');
  assert.strictEqual(subTabId, 'tab-sub', 'the sub-agent gets its own tab');
  assert.strictEqual(
    activeTabId(win),
    'tab-a',
    'a sub-agent starting must not switch tabs',
  );

  statusRunning(win, subTabId, 900);
  assert.strictEqual(
    activeTabId(win),
    'tab-a',
    'a running sub-agent must not become the launch tab',
  );
  win.close();
  console.log('  ok - a sub-agent tab is never the launch tab');
}

function runTests() {
  const tests = [
    testRemoteSingleRunningTaskTakesTheScreen,
    testRemoteNewestRunningTaskWins,
    testRemoteMissingTimestampsFallBackToOrder,
    testRemoteAlreadyOpenRunningChatIsSwitchedTo,
    testNoRunningTaskKeepsTheRestoredTab,
    testMalformedSnapshotDoesNotSwitch,
    testExtensionNewestRestoredStatusWins,
    testFinishedTaskIsNoLongerACandidate,
    testTaskFinishingDoesNotMoveTheUser,
    testGestureBeforeTheBackendIsLiveKeepsTheLaunch,
    testPasswordPromptDoesNotSpendTheLaunch,
    testPointerDownEndsTheLaunch,
    testKeyDownEndsTheLaunch,
    testBackendHiccupDoesNotRelaunch,
    testLaunchWindowExpires,
    testSubagentTabIsNotALaunchTarget,
  ];
  for (const t of tests) t();
  console.log(`\n${tests.length} passed, 0 failed`);
}

try {
  runTests();
  process.exit(0);
} catch (err) {
  console.error('FAIL:', err && err.message ? err.message : err);
  process.exit(1);
}
