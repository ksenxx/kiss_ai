// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end tests for the task-info panel's getTaskUpdate poll
// (media/chat.html + media/main.js, remote desktop mode):
//
// * a task start fires exactly one immediate poll (the poll timer's
//   own first request; syncMetaInfoRunning must not add a second one);
// * the poll goes quiet when the owner chat's task ends while a content
//   tab is shown: the owner's `status running:false` is not the active
//   tab's, yet the page mirrors it onto the shared surface (clock,
//   subpanel, poll) because the content tab borrows the owner's panel;
// * a switch from a running chat to an idle one fires no poll for the
//   idle tab: restoreTab retargets the panel before it applies the
//   destination's running state, so the poll must consult the POLLED
//   tab's own running flag rather than the lagging module-level one.
//
// The rest of the protocol (reply matching, rendering, refresh) is
// test/remoteMetaPanelInfo.test.js.  The old editor-panel / Task Info
// view relays (`metaUpdate`, `metaState`, `metaRefresh`) were the VS
// Code host's and have no subject any more.

'use strict';

const assert = require('assert');
const {makePage, send, stubRaf} = require('./simplify2_harness.js');

const POLL_MS = 5000;

/** Record every setInterval so a test can fire the poll on demand. */
function captureIntervals(win) {
  const intervals = [];
  const realSetInterval = win.setInterval.bind(win);
  const realClearInterval = win.clearInterval.bind(win);
  win.setInterval = function (fn, ms, ...args) {
    const id = realSetInterval(fn, ms, ...args);
    intervals.push({id, fn, ms});
    return id;
  };
  win.clearInterval = function (id) {
    realClearInterval(id);
    const i = intervals.findIndex(entry => entry.id === id);
    if (i >= 0) intervals.splice(i, 1);
  };
  return {
    pollTimers: () => intervals.filter(entry => entry.ms === POLL_MS),
    tickPolls() {
      for (const entry of intervals.filter(e => e.ms === POLL_MS)) entry.fn();
    },
  };
}

function desktopPage() {
  let timers;
  const page = makePage({
    beforeScripts: win => {
      timers = captureIntervals(win);
      stubRaf(win);
      win.matchMedia = query => ({
        matches: query === '(min-width: 900px)',
        media: query,
        addEventListener: () => {},
        removeEventListener: () => {},
        addListener: () => {},
        removeListener: () => {},
      });
    },
  });
  assert.ok(
    page.win.document.body.classList.contains('remote-desktop'),
    'the desktop remote page carries remote-desktop',
  );
  send(page.win, {
    type: 'configData',
    config: {work_dir: '/cfg/dir', max_budget: 42},
    apiKeys: {},
  });
  return Object.assign(page, timers);
}

function polls(page) {
  return page.posted.filter(m => m.type === 'getTaskUpdate');
}

function testTaskStartFiresOneImmediatePoll() {
  const page = desktopPage();
  assert.strictEqual(polls(page).length, 0, 'an idle page must not poll');
  // Desktop mode keeps one poll timer; it only asks while a task runs.
  assert.strictEqual(page.pollTimers().length, 1, 'one poll timer');
  page.tickPolls();
  assert.strictEqual(polls(page).length, 0, 'an idle tick asks nothing');
  send(page.win, {type: 'status', running: true});
  // The 5 s interval has not ticked: only the immediate poll can exist.
  assert.strictEqual(
    polls(page).length,
    1,
    'the start of a task must fire exactly one immediate poll',
  );
  send(page.win, {type: 'status', running: false});
  page.tickPolls();
  assert.strictEqual(polls(page).length, 1, 'the end quiets the poll');
  page.win.close();
  console.log('  ok - a task start fires exactly one immediate poll');
}

function testPollGoesQuietWhenOwnerStopsBehindContentTab() {
  const page = desktopPage();
  const win = page.win;
  const root = win._testApi.getActiveTabId();
  send(win, {type: 'status', running: true, tabId: root});
  const firstPolls = polls(page);
  assert.ok(
    firstPolls.length >= 1,
    'the running task must be polling before the content tab opens',
  );
  send(win, {
    type: 'taskUpdate',
    tabId: root,
    token: firstPolls[firstPolls.length - 1].token,
    exists: true,
    sig: '9:9',
    content: '<p>live progress</p>',
    error: '',
    running: false,
    cost: 0.01,
    updatedAt: Date.UTC(2026, 8, 22, 12, 0),
  });
  const content = win.document.getElementById('meta-info-content');
  assert.ok(content.textContent.includes('live progress'));

  // A directory listing opens a content tab (the content pane on the
  // desktop split layout); the chat's task-info target is kept.
  send(win, {
    type: 'fileContent',
    path: '/cfg/dir/sub',
    name: 'sub',
    isDirectory: true,
    content: 'a.txt\nb.txt\n',
  });
  assert.ok(
    win._testApi.openTabs().some(t => t.isContentTab),
    'the listing opened a content tab',
  );
  // The owner chat's task ends while the content tab is shown.  On the
  // desktop split layout the chat stays the active tab (the listing
  // went to the content pane), so this is the active tab's own status;
  // what matters is that the panel lent to the content view clears
  // and stops polling with it.
  send(win, {type: 'status', running: false, tabId: root});
  assert.ok(
    !win.document.getElementById('meta-info').classList.contains('visible'),
    'the task update subpanel clears with the owner task',
  );
  assert.ok(
    !win.document
      .getElementById('status-text')
      .textContent.startsWith('Running'),
    'the clock must stop with the owner task',
  );
  const count = polls(page).length;
  page.tickPolls();
  page.tickPolls();
  assert.strictEqual(
    polls(page).length,
    count,
    'no further polls once the polled tab itself stopped running',
  );
  win.close();
  console.log('  ok - the poll goes quiet when the owner chat stops');
}

function testSwitchToIdleTabFiresNoPoll() {
  const page = desktopPage();
  const win = page.win;
  const pollsFor = id => polls(page).filter(m => m.tabId === id);
  const a = win._testApi.getActiveTabId();
  win._testApi.endLaunch();
  win._testApi.createNewTab();
  const b = win._testApi.getActiveTabId();
  assert.notStrictEqual(a, b, 'the second chat is a new tab');
  win._testApi.switchToTab(a);
  send(win, {type: 'status', running: true, tabId: a});
  assert.strictEqual(pollsFor(a).length, 1, 'the running chat polls once');
  assert.strictEqual(pollsFor(b).length, 0, 'the idle chat is never asked');

  // restoreTab retargets the panel (and polls) before it applies B's
  // running state: with the module-level flag still A's `true`, one
  // poll would go out under B's id for a task B does not have.
  win._testApi.switchToTab(b);
  assert.strictEqual(
    win._testApi.openTabs().find(t => t.id === b).isRunning,
    false,
    'tab B is idle',
  );
  assert.strictEqual(
    pollsFor(b).length,
    0,
    'switching to an idle chat must not poll for it',
  );
  page.tickPolls();
  assert.strictEqual(pollsFor(b).length, 0, 'nor on the next tick');

  // Back on the running chat the panel repolls right away.
  win._testApi.switchToTab(a);
  assert.ok(
    pollsFor(a).length >= 2,
    'returning to the running chat repolls it',
  );
  assert.strictEqual(pollsFor(b).length, 0, 'still nothing for B');
  win.close();
  console.log('  ok - a switch to an idle chat fires no poll');
}

function main() {
  testTaskStartFiresOneImmediatePoll();
  testPollGoesQuietWhenOwnerStopsBehindContentTab();
  testSwitchToIdleTabFiresNoPoll();
  console.log('taskUpdatePolling: all passed.');
}

try {
  main();
} catch (err) {
  console.error(err);
  process.exit(1);
}
