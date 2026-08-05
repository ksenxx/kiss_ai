// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end tests for the launch tab switch, in the extension webview and in
// the remote web app.
//
// Opening a chat window while agents are still working should put the newest
// of those tasks on screen -- that is the one the user just walked away from.
// The remote web app learns about them from the `openRunningTasks` snapshot
// the server pushes after `ready`; the extension host replays a `status` for
// every tab it restored.
//
// The permission to move the user lasts only until the first real gesture, so
// a WebSocket that drops and reconnects mid-session (it says `ready` again,
// and gets the same snapshot back) cannot yank anybody off the transcript
// they are reading. See openRunningTasksNoTabSwitch.test.js for that half of
// the contract.

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

function tabIdForChat(win, posted, chatId) {
  const call = posted.find(m => m.type === 'resumeSession' && m.id === chatId);
  assert.ok(call, `chat ${chatId} must have been resumed`);
  return call.tabId;
}

function openRunningTasks(win, tasks) {
  send(win, {type: 'openRunningTasks', tasks: tasks});
}

function statusRunning(win, tabId, startTs) {
  send(win, {type: 'status', running: true, tabId: tabId, startTs: startTs});
}

// The remote web app: the snapshot names one running task, so that is where
// the window opens.
function testRemoteSingleRunningTaskTakesTheScreen() {
  const {win, posted} = makeWebview({remote: true});
  const boot = activeTabId(win);

  openRunningTasks(win, [
    {chatId: 'chat-1', taskId: 'task-1', title: 'ship the parser', startTs: 7},
  ]);

  const restored = tabIdForChat(win, posted, 'chat-1');
  assert.notStrictEqual(restored, boot, 'the task must get its own tab');
  assert.strictEqual(
    activeTabId(win),
    restored,
    'launching with a running task must put that task on screen',
  );
  win.close();
  console.log('  ok - a single running task is on screen at launch');
}

// Several running tasks: the newest wins, whatever order the rows arrive in.
function testRemoteNewestRunningTaskWins() {
  for (const order of ['ascending', 'shuffled']) {
    const rows = [
      {chatId: 'chat-old', taskId: 't-old', title: 'old', startTs: 100},
      {chatId: 'chat-new', taskId: 't-new', title: 'new', startTs: 300},
      {chatId: 'chat-mid', taskId: 't-mid', title: 'mid', startTs: 200},
    ];
    rows.sort((a, b) =>
      order === 'ascending' ? a.startTs - b.startTs : b.startTs - a.startTs,
    );
    const {win, posted} = makeWebview({remote: true});
    openRunningTasks(win, rows);
    assert.strictEqual(
      activeTabId(win),
      tabIdForChat(win, posted, 'chat-new'),
      `the newest task must win (rows ${order})`,
    );
    assert.strictEqual(
      posted.filter(m => m.type === 'resumeSession').length,
      3,
      'every running task must still be resumed into its own tab',
    );
    win.close();
  }
  console.log('  ok - the newest running task wins, in any row order');
}

// Timestamps go missing when a task predates the history row. The backend
// lists its running tasks oldest first, so the last row is still the newest.
function testRemoteMissingTimestampsFallBackToOrder() {
  const {win, posted} = makeWebview({remote: true});
  openRunningTasks(win, [
    {chatId: 'chat-first', taskId: 't1', title: 'first'},
    {chatId: 'chat-last', taskId: 't2', title: 'last'},
  ]);
  assert.strictEqual(
    activeTabId(win),
    tabIdForChat(win, posted, 'chat-last'),
    'with no timestamps the last row is the newest task',
  );
  win.close();
  console.log('  ok - rows without timestamps fall back to snapshot order');
}

// A chat the window already had a tab for is switched to, not duplicated.
function testRemoteAlreadyOpenRunningChatIsSwitchedTo() {
  const {win, posted} = makeWebview({
    remote: true,
    state: {
      tabs: [
        {title: 'reading', chatId: 'tab-read', backendChatId: 'chat-read'},
        {title: 'working', chatId: 'tab-work', backendChatId: 'chat-work'},
      ],
      activeTabIndex: 0,
      chatId: 'tab-read',
    },
  });
  assert.strictEqual(activeTabId(win), 'tab-read', 'restored on the first tab');
  const before = tabIds(win);

  openRunningTasks(win, [
    {chatId: 'chat-work', taskId: 't-work', title: 'working', startTs: 42},
  ]);

  assert.deepStrictEqual(
    tabIds(win),
    before,
    'a chat that already has a tab must not get a second one',
  );
  assert.strictEqual(
    posted.filter(m => m.type === 'resumeSession' && m.id === 'chat-work')
      .length,
    0,
    'an already-open chat must not be resumed again by the snapshot',
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
    state: {
      tabs: [
        {title: 'one', chatId: 'tab-1', backendChatId: 'chat-1'},
        {title: 'two', chatId: 'tab-2', backendChatId: 'chat-2'},
      ],
      activeTabIndex: 1,
      chatId: 'tab-2',
    },
  });
  openRunningTasks(win, []);
  send(win, {type: 'status', running: false, tabId: 'tab-1'});
  assert.strictEqual(
    activeTabId(win),
    'tab-2',
    'with nothing running the restored tab stays on screen',
  );
  win.close();
  console.log('  ok - an idle launch keeps the restored tab');
}

// Junk in the snapshot must not move anybody.
function testMalformedSnapshotDoesNotSwitch() {
  const {win, posted} = makeWebview({remote: true});
  const boot = activeTabId(win);
  openRunningTasks(win, [null, {}, {taskId: 'no-chat-id'}]);
  openRunningTasks(win, 'not-an-array');
  send(win, {type: 'openRunningTasks'});
  assert.strictEqual(activeTabId(win), boot, 'the active tab is kept');
  assert.strictEqual(
    posted.filter(m => m.type === 'resumeSession').length,
    0,
    'nothing may be resumed',
  );
  win.close();
  console.log('  ok - a malformed snapshot switches nothing');
}

// The extension host replays a `status` per restored tab; the newest of them
// takes the screen.
function testExtensionNewestRestoredStatusWins() {
  const {win} = makeWebview({
    state: {
      tabs: [
        {title: 'alpha', chatId: 'tab-a', backendChatId: 'chat-a'},
        {title: 'beta', chatId: 'tab-b', backendChatId: 'chat-b'},
        {title: 'gamma', chatId: 'tab-c', backendChatId: 'chat-c'},
      ],
      activeTabIndex: 0,
      chatId: 'tab-a',
    },
  });
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
  const {win, posted} = makeWebview({remote: true, connect: false});

  win.document.dispatchEvent(
    new win.MouseEvent('pointerdown', {bubbles: true}),
  );
  win.document.dispatchEvent(
    new win.KeyboardEvent('keydown', {key: 'p', bubbles: true}),
  );

  send(win, {type: 'daemonStatus', connected: true});
  openRunningTasks(win, [
    {chatId: 'chat-1', taskId: 'task-1', title: 'still going', startTs: 5},
  ]);

  assert.strictEqual(
    activeTabId(win),
    tabIdForChat(win, posted, 'chat-1'),
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
  const {win, posted} = makeWebview({remote: true, connect: false});

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
  // the snapshot comes back.
  send(win, {type: 'daemonStatus', connected: true});
  openRunningTasks(win, [
    {chatId: 'chat-1', taskId: 'task-1', title: 'still going', startTs: 5},
  ]);

  assert.strictEqual(
    activeTabId(win),
    tabIdForChat(win, posted, 'chat-1'),
    'typing the remote password must not spend the launch it precedes',
  );
  win.close();
  console.log('  ok - the password prompt does not spend the launch');
}

// Landing on a task means staying there when it finishes: the result is what
// the user was brought to see.
function testTaskFinishingDoesNotMoveTheUser() {
  const {win, posted} = makeWebview({remote: true});
  openRunningTasks(win, [
    {chatId: 'chat-slow', taskId: 't-slow', title: 'slow', startTs: 100},
    {chatId: 'chat-quick', taskId: 't-quick', title: 'quick', startTs: 900},
  ]);
  const slowTab = tabIdForChat(win, posted, 'chat-slow');
  const quickTab = tabIdForChat(win, posted, 'chat-quick');

  // Both replays report themselves running, so the launch is fully resolved.
  send(win, {type: 'status', running: true, tabId: slowTab, startTs: 100});
  send(win, {type: 'status', running: true, tabId: quickTab, startTs: 900});
  assert.strictEqual(activeTabId(win), quickTab, 'the newest task won');

  send(win, {type: 'status', running: false, tabId: quickTab});
  assert.strictEqual(
    activeTabId(win),
    quickTab,
    'a task finishing must leave the user on its result, not move them to ' +
      'an older task that happens to still be running',
  );
  win.close();
  console.log('  ok - a task finishing never moves the user');
}

// A task that had already finished by the time its replay landed never won the
// launch in the first place, so the still-running one takes the screen.
function testFinishedTaskIsNoLongerACandidate() {
  const {win, posted} = makeWebview({remote: true});
  openRunningTasks(win, [
    {chatId: 'chat-slow', taskId: 't-slow', title: 'slow', startTs: 100},
    {chatId: 'chat-quick', taskId: 't-quick', title: 'quick', startTs: 900},
  ]);
  const slowTab = tabIdForChat(win, posted, 'chat-slow');
  const quickTab = tabIdForChat(win, posted, 'chat-quick');
  assert.strictEqual(activeTabId(win), quickTab, 'the newest task won');

  // The quick task finishes: its replay lands the closing status, and the
  // chat drops out of the launch snapshot.
  send(win, {type: 'status', running: false, tabId: quickTab});
  send(win, {type: 'status', running: true, tabId: slowTab, startTs: 100});
  assert.strictEqual(
    activeTabId(win),
    slowTab,
    'once the newest task is done the launch falls through to the one ' +
      'still running',
  );
  win.close();
  console.log('  ok - a finished task drops out of the launch snapshot');
}

// The first tap ends the launch: a reconnect after it must not move the user.
function testPointerDownEndsTheLaunch() {
  const {win, posted} = makeWebview({remote: true});
  const boot = activeTabId(win);

  win.document.dispatchEvent(
    new win.MouseEvent('pointerdown', {bubbles: true}),
  );
  openRunningTasks(win, [
    {chatId: 'chat-1', taskId: 'task-1', title: 'background', startTs: 9},
  ]);

  assert.strictEqual(
    activeTabId(win),
    boot,
    'a tap ends the launch, so a reconnect snapshot must not switch tabs',
  );
  assert.strictEqual(
    posted.filter(m => m.type === 'resumeSession').length,
    1,
    'the task must still get a reachable background tab',
  );
  win.close();
  console.log('  ok - a tap ends the launch');
}

// So does the first keystroke.
function testKeyDownEndsTheLaunch() {
  const {win} = makeWebview({
    state: {
      tabs: [
        {title: 'alpha', chatId: 'tab-a', backendChatId: 'chat-a'},
        {title: 'beta', chatId: 'tab-b', backendChatId: 'chat-b'},
      ],
      activeTabIndex: 0,
      chatId: 'tab-a',
    },
  });
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
  const {win} = makeWebview({
    state: {
      tabs: [
        {title: 'alpha', chatId: 'tab-a', backendChatId: 'chat-a'},
        {title: 'beta', chatId: 'tab-b', backendChatId: 'chat-b'},
      ],
      activeTabIndex: 0,
      chatId: 'tab-a',
    },
  });
  // The launch runs its course and lands on the running task.
  statusRunning(win, 'tab-b', 500);
  assert.strictEqual(activeTabId(win), 'tab-b', 'the launch had its chance');

  // The user goes back to the other chat by hand.
  clickTab(win, 'tab-a');
  assert.strictEqual(activeTabId(win), 'tab-a', 'the user chose this tab');

  // The daemon drops and returns, replaying every running task.
  send(win, {type: 'daemonStatus', connected: false});
  send(win, {type: 'daemonStatus', connected: true});
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
  const {win} = makeWebview({
    state: {
      tabs: [
        {title: 'alpha', chatId: 'tab-a', backendChatId: 'chat-a'},
        {title: 'beta', chatId: 'tab-b', backendChatId: 'chat-b'},
      ],
      activeTabIndex: 0,
      chatId: 'tab-a',
    },
  });
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
  const {win} = makeWebview({
    state: {
      tabs: [{title: 'alpha', chatId: 'tab-a', backendChatId: 'chat-a'}],
      activeTabIndex: 0,
      chatId: 'tab-a',
    },
  });
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
