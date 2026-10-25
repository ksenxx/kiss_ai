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
//   tab's own running flag rather than the lagging module-level one;
// * the poll of a tab bound to a remote machine (the daemon's
//   `tabMachine` binding) is stamped with that machine, so the daemon
//   forwards it to the machine's daemon, which runs the task and owns
//   its update; the drop-down's unsent pick does not count, a tab of
//   this machine polls with an empty `machine`, and the machine's
//   reply (stamped `machine`) renders like a local one.
//
// The rest of the protocol (reply matching, rendering, refresh) is
// test/remoteMetaPanelInfo.test.js.  The old editor-panel / Task Info
// view relays (`metaUpdate`, `metaState`, `metaRefresh`) were the VS
// Code host's and have no subject any more.

'use strict';

const assert = require('assert');
const {
  makePage,
  send,
  stubRaf,
  fakeIntervals,
  fakeMatchMedia,
} = require('./simplify2_harness.js');

const POLL_MS = 5000;

/**
 * The poll timers (the 5 s intervals) among the page's live intervals
 * as `fakeIntervals` records them, listed and fired on demand.
 */
function pollControls(intervals) {
  const pollTimers = () => intervals.filter(iv => iv.ms === POLL_MS);
  return {
    pollTimers,
    tickPolls() {
      for (const iv of pollTimers()) iv.cb(...iv.args);
    },
  };
}

function desktopPage() {
  let timers;
  const page = makePage({
    beforeScripts: win => {
      timers = pollControls(fakeIntervals(win).intervals);
      stubRaf(win);
      fakeMatchMedia(win, true);
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

function testRemoteBoundTabPollsItsMachine() {
  const page = desktopPage();
  const win = page.win;
  const root = win._testApi.getActiveTabId();
  send(win, {
    type: 'machines',
    machines: [
      {name: 'laptop', local: true, status: 'reachable', address: 'x'},
      {name: 'gpu-box', status: 'reachable', address: 'ssh me@gpu'},
    ],
  });
  // The drop-down's unsent pick is not a binding: the task of a tab
  // that has not been submitted anywhere runs here when it starts.
  const select = win.document.getElementById('chat-machine');
  select.value = 'gpu-box';
  select.dispatchEvent(new win.Event('change', {bubbles: true}));
  send(win, {type: 'status', running: true, tabId: root});
  let last = polls(page)[polls(page).length - 1];
  assert.strictEqual(last.tabId, root);
  assert.strictEqual(
    last.machine,
    '',
    'an unsent drop-down pick must not redirect the poll',
  );

  // The daemon binds the tab to the machine its task runs on.
  send(win, {type: 'tabMachine', tabId: root, machine: 'gpu-box'});
  page.tickPolls();
  last = polls(page)[polls(page).length - 1];
  assert.strictEqual(last.tabId, root);
  assert.strictEqual(
    last.machine,
    'gpu-box',
    'a bound tab polls the machine that runs its task',
  );
  assert.strictEqual(typeof last.token, 'string');

  // The machine's reply, relayed by the daemon stamped `machine`,
  // shows the subpanel like a local one.
  send(win, {
    type: 'taskUpdate',
    tabId: root,
    token: last.token,
    machine: 'gpu-box',
    exists: true,
    sig: '3:1',
    content: '<p>remote progress</p>',
    error: '',
    running: false,
    cost: 0.02,
    updatedAt: Date.UTC(2026, 9, 24, 17, 0),
  });
  assert.ok(
    win.document.getElementById('meta-info').classList.contains('visible'),
    'the Task update subpanel shows for a task running on a machine',
  );
  assert.ok(
    win.document
      .getElementById('meta-info-content')
      .textContent.includes('remote progress'),
  );
  page.tickPolls();
  last = polls(page)[polls(page).length - 1];
  assert.strictEqual(last.knownSig, '3:1', 'the next poll names the sig');
  assert.strictEqual(last.machine, 'gpu-box');

  // The binding is the daemon's: unbound again, the poll stays home.
  send(win, {type: 'tabMachine', tabId: root, machine: ''});
  page.tickPolls();
  last = polls(page)[polls(page).length - 1];
  assert.strictEqual(last.machine, '', 'an unbound tab polls this machine');
  win.close();
  console.log('  ok - a remote-bound tab polls its machine for the update');
}

function main() {
  testTaskStartFiresOneImmediatePoll();
  testPollGoesQuietWhenOwnerStopsBehindContentTab();
  testSwitchToIdleTabFiresNoPoll();
  testRemoteBoundTabPollsItsMachine();
  console.log('taskUpdatePolling: all passed.');
}

try {
  main();
} catch (err) {
  console.error(err);
  process.exit(1);
}
