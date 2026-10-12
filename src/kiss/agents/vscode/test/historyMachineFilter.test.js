// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// End-to-end test of the History panel's machine filter: the daemon's
// `history` reply names the registered remote machines, every row
// carries its machine as data-machine ('' for the local daemon) with a
// chip linking to the machine's web UI, the <select id="hf-machine">
// lists All / this machine / each remote machine (and stays hidden
// while none is registered), choosing one hides the other rows, and a
// remote row's click opens the machine's web UI instead of resuming
// the chat here.

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

function makeWebview() {
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
  html = html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');

  const dom = new JSDOM(html, {
    runScripts: 'dangerously',
    pretendToBeVisual: true,
    url: 'https://localhost/',
  });
  const win = dom.window;

  win.Element.prototype.scrollIntoView = function () {};
  win.Element.prototype.scrollTo = function () {};
  win.HTMLElement.prototype.scrollTo = function () {};

  const posted = [];
  win.acquireVsCodeApi = function () {
    let state;
    return {
      postMessage: m => posted.push(m),
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };

  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));

  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function session(id, title, machine, extra) {
  return Object.assign(
    {
      id: 'chat-' + id,
      task_id: 'task-' + id,
      title,
      timestamp: 1_700_000_000 + id * 100,
      preview: title,
      has_events: false,
      failed: false,
      is_running: false,
      tokens: 0,
      cost: 0,
      steps: 0,
      is_favorite: false,
      work_dir: '',
      startTs: (1_700_000_000 + id * 100) * 1000,
      endTs: (1_700_000_000 + id * 100) * 1000 + 10_000,
    },
    machine
      ? {machine, machine_url: 'https://' + machine.trim() + ':8787'}
      : {},
    extra || {},
  );
}

const SESSIONS = [
  session(1, 'local one', ''),
  session(2, 'box task', 'box'),
  session(3, 'local two', ''),
  session(4, 'lab task', 'lab'),
  session(5, 'box task two', ' box '),
];
const MACHINES = ['box', 'lab'];

// The reply to the webview's latest getHistory request: it must echo
// that request's generation (a filter change bumps it and refetches).
function historyReply(posted, sessions, machines) {
  const requests = posted.filter(m => m && m.type === 'getHistory');
  const last = requests[requests.length - 1];
  return {
    type: 'history',
    sessions,
    offset: 0,
    generation: last ? last.generation : 0,
    machines,
    localMachine: 'myhost',
  };
}

function lastHistoryRequest(posted) {
  const requests = posted.filter(m => m && m.type === 'getHistory');
  return requests[requests.length - 1];
}

function rows(win) {
  return Array.from(
    win.document
      .getElementById('history-list')
      .querySelectorAll('.sidebar-item'),
  );
}

function visibleTitles(win) {
  return rows(win)
    .filter(r => r.style.display !== 'none')
    .map(r => r.querySelector('.sidebar-item-text').textContent)
    .sort();
}

function options(win) {
  return Array.from(win.document.getElementById('hf-machine').options).map(
    o => [o.value, o.textContent],
  );
}

function selectMachine(win, value) {
  const sel = win.document.getElementById('hf-machine');
  sel.value = value;
  sel.dispatchEvent(new win.Event('change', {bubbles: true}));
}

function testHiddenUntilAMachineIsRegistered() {
  const {win, posted} = makeWebview();
  const sel = win.document.getElementById('hf-machine');
  assert.ok(sel && sel.tagName === 'SELECT', 'hf-machine select must exist');
  assert.ok(sel.parentElement.hidden, 'hidden before any history reply');
  send(win, historyReply(posted, SESSIONS.slice(0, 1), []));
  assert.ok(sel.parentElement.hidden, 'hidden while no machine is registered');
  assert.deepStrictEqual(options(win), [
    ['', 'All'],
    ['__local__', 'myhost (this machine)'],
  ]);
  send(win, historyReply(posted, SESSIONS, MACHINES));
  assert.ok(!sel.parentElement.hidden, 'shown once a machine is registered');
  assert.deepStrictEqual(options(win), [
    ['', 'All'],
    ['__local__', 'myhost (this machine)'],
    ['box', 'box'],
    ['lab', 'lab'],
  ]);
  // A reply without the fields (an older daemon) hides the filter again.
  send(win, {type: 'history', sessions: SESSIONS.slice(0, 1), offset: 0});
  assert.ok(sel.parentElement.hidden);
  assert.deepStrictEqual(options(win), [
    ['', 'All'],
    ['__local__', 'local (this machine)'],
  ]);
  win.close();
  console.log('  ok - machine select hidden until a machine is registered');
}

function testRowsCarryTheirMachineAndChip() {
  const {win, posted} = makeWebview();
  send(win, historyReply(posted, SESSIONS, MACHINES));
  const seen = {};
  rows(win).forEach(r => {
    const chip = r.querySelector('.sidebar-item-machine');
    seen[r.querySelector('.sidebar-item-text').textContent] = {
      machine: r.dataset.machine,
      chip: chip ? chip.textContent : null,
      href: chip ? chip.href : null,
      target: chip ? chip.target : null,
      favourite: !!r.querySelector('.sidebar-item-favorite'),
      copy: !!r.querySelector('.sidebar-item-copy'),
    };
  });
  assert.deepStrictEqual(seen['local one'], {
    machine: '',
    chip: null,
    href: null,
    target: null,
    favourite: true,
    copy: true,
  });
  assert.deepStrictEqual(seen['box task'], {
    machine: 'box',
    chip: 'box',
    href: 'https://box:8787/',
    target: '_blank',
    favourite: false,
    copy: true,
  });
  assert.deepStrictEqual(seen['box task two'].machine, 'box');
  assert.deepStrictEqual(seen['lab task'].chip, 'lab');
  win.close();
  console.log('  ok - rows carry data-machine and a linking chip');
}

function testSelectingAMachineFiltersRows() {
  const {win, posted} = makeWebview();
  send(win, historyReply(posted, SESSIONS, MACHINES));
  assert.strictEqual(visibleTitles(win).length, 5, 'All shows every row');

  selectMachine(win, 'box');
  assert.deepStrictEqual(visibleTitles(win), ['box task', 'box task two']);
  // The daemon applies the machine too (its rows may live beyond the
  // loaded page): the change refetches page one with the machine.
  const request = lastHistoryRequest(posted);
  assert.strictEqual(request.machine, 'box');
  assert.strictEqual(request.offset, 0);

  selectMachine(win, 'lab');
  assert.deepStrictEqual(visibleTitles(win), ['lab task']);

  selectMachine(win, '__local__');
  assert.deepStrictEqual(visibleTitles(win), ['local one', 'local two']);

  selectMachine(win, '');
  assert.strictEqual(visibleTitles(win).length, 5, 'All restores every row');

  // The machine filter combines with the SEA filter.
  const fixture = [
    session(1, 'box paper', 'box', {sea: 'write_paper_sea'}),
    session(2, 'local paper', '', {sea: 'write_paper_sea'}),
    session(3, 'box plain', 'box'),
  ];
  send(win, historyReply(posted, fixture, ['box']));
  selectMachine(win, 'box');
  const sea = win.document.getElementById('hf-sea');
  sea.value = 'write_paper_sea';
  sea.dispatchEvent(new win.Event('change', {bubbles: true}));
  assert.deepStrictEqual(visibleTitles(win), ['box paper']);
  win.close();
  console.log('  ok - selecting a machine / this machine / All filters rows');
}

function testSelectionSurvivesRefreshesAndRemoval() {
  const {win, posted} = makeWebview();
  send(win, historyReply(posted, SESSIONS, MACHINES));
  selectMachine(win, 'lab');
  // A refresh with the same registry keeps the selection and the filter.
  send(win, historyReply(posted, SESSIONS, MACHINES));
  assert.strictEqual(win.document.getElementById('hf-machine').value, 'lab');
  assert.deepStrictEqual(visibleTitles(win), ['lab task']);
  // The local option survives too.
  selectMachine(win, '__local__');
  send(win, historyReply(posted, SESSIONS, MACHINES));
  assert.strictEqual(
    win.document.getElementById('hf-machine').value,
    '__local__',
  );
  assert.deepStrictEqual(visibleTitles(win), ['local one', 'local two']);
  // Removing the selected machine falls back to All and re-applies.
  selectMachine(win, 'lab');
  const withoutLab = SESSIONS.filter(s => s.machine !== 'lab');
  send(win, historyReply(posted, withoutLab, ['box']));
  assert.strictEqual(win.document.getElementById('hf-machine').value, '');
  assert.strictEqual(visibleTitles(win).length, 4, 'All rows after fallback');
  assert.strictEqual(
    lastHistoryRequest(posted).machine,
    '',
    'refetched for All',
  );
  win.close();
  console.log('  ok - selection survives refreshes; a removed machine resets');
}

function testClickingARemoteRowOpensItsMachine() {
  const {win, posted} = makeWebview();
  send(win, historyReply(posted, SESSIONS, MACHINES));
  const opened = [];
  // JSDOM does not navigate: record the anchor activations instead.
  win.HTMLAnchorElement.prototype.click = function () {
    opened.push(this.href);
  };
  const remoteRow = rows(win).find(r => r.dataset.machine === 'box');
  posted.length = 0;
  remoteRow.click();
  assert.deepStrictEqual(opened, ['https://box:8787/']);
  assert.ok(
    !posted.some(
      m => m && (m.type === 'openTab' || m.type === 'resumeSession'),
    ),
    'a remote row must not resume a chat on this daemon',
  );
  // A click on the chip itself opens the link once (no double open).
  opened.length = 0;
  const chip = remoteRow.querySelector('.sidebar-item-machine');
  chip.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.deepStrictEqual(
    opened,
    [],
    'the row handler leaves the chip click alone',
  );
  // A local row still opens the chat here.
  const localRow = rows(win).find(r => r.dataset.machine === '');
  posted.length = 0;
  localRow.click();
  assert.deepStrictEqual(opened, []);
  assert.ok(posted.length > 0, 'a local row talks to the daemon');
  win.close();
  console.log('  ok - a remote row opens its machine, a local row its chat');
}

function testChatHeaderOfARemoteChatOpensItsMachine() {
  const {win, posted} = makeWebview();
  // Chat "chat-r" exists only on box; chat "chat-m" has a local task
  // and a newer remote copy of the chat (a database copied over).
  const fixture = [
    session(1, 'remote chat task', 'box', {id: 'chat-r'}),
    session(2, 'mixed chat, local task', '', {id: 'chat-m'}),
    session(3, 'mixed chat, remote task', 'lab', {id: 'chat-m'}),
  ];
  send(win, historyReply(posted, fixture, MACHINES));
  const opened = [];
  win.HTMLAnchorElement.prototype.click = function () {
    opened.push(this.href);
  };
  const groups = Array.from(
    win.document.querySelectorAll('#history-list .history-chat-group'),
  );
  const groupWithTitle = title =>
    groups.find(g =>
      Array.from(g.querySelectorAll('.sidebar-item-text')).some(
        t => t.textContent === title,
      ),
    );
  const remoteGroup = groupWithTitle('remote chat task');
  const mixedGroup = groupWithTitle('mixed chat, local task');
  assert.ok(remoteGroup && mixedGroup, 'both chat groups are rendered');
  // Only the remote-only chat's header names its machine.
  const remoteChip = remoteGroup.querySelector(
    '.history-chat-header .history-chat-machine',
  );
  assert.ok(remoteChip, 'a remote-only chat header carries the machine chip');
  assert.strictEqual(remoteChip.textContent, 'box');
  assert.strictEqual(remoteChip.href, 'https://box:8787/');
  assert.strictEqual(
    mixedGroup.querySelector('.history-chat-header .history-chat-machine'),
    null,
    'a chat with a local task carries no chip on its header',
  );

  posted.length = 0;
  remoteGroup.querySelector('.history-chat-header').click();
  assert.deepStrictEqual(opened, ['https://box:8787/']);
  assert.ok(
    !posted.some(
      m => m && (m.type === 'openTab' || m.type === 'resumeSession'),
    ),
    'a remote-only chat must not resume on this daemon',
  );

  opened.length = 0;
  posted.length = 0;
  mixedGroup.querySelector('.history-chat-header').click();
  assert.deepStrictEqual(opened, [], 'a chat present here opens here');
  assert.ok(posted.length > 0, 'the local copy of the chat is resumed');
  win.close();
  console.log(
    '  ok - a chat header opens the remote machine or the local chat',
  );
}

function groups(win) {
  return Array.from(
    win.document
      .getElementById('history-list')
      .querySelectorAll(':scope > .history-chat-group'),
  );
}

function testSelectingAMachineKeepsChatsCollapsed() {
  const {win, posted} = makeWebview();
  send(win, historyReply(posted, SESSIONS, MACHINES));
  assert.ok(groups(win).length >= 5, 'one group per chat');
  assert.ok(
    groups(win).every(g => g.classList.contains('collapsed')),
    'every chat starts collapsed',
  );
  // A machine scopes the list; it is not a search: nothing unfolds.
  selectMachine(win, 'box');
  send(win, historyReply(posted, SESSIONS, MACHINES));
  assert.deepStrictEqual(visibleTitles(win), ['box task', 'box task two']);
  assert.ok(
    groups(win).every(g => g.classList.contains('collapsed')),
    'the machine filter keeps every chat collapsed',
  );
  // A chat the user unfolded stays unfolded across the filter change.
  const open = groups(win).find(g => !g.querySelector('.sidebar-item').style.display);
  open.querySelector('.history-chat-collapse').click();
  assert.ok(!open.classList.contains('collapsed'));
  selectMachine(win, '');
  send(win, historyReply(posted, SESSIONS, MACHINES));
  const same = groups(win).find(g => g.dataset.chatId === open.dataset.chatId);
  assert.ok(!same.classList.contains('collapsed'), 'the unfolded chat stays open');
  assert.strictEqual(
    groups(win).filter(g => !g.classList.contains('collapsed')).length,
    1,
    'the rest stay collapsed',
  );
  // A search text, by contrast, unfolds the matches.
  const search = win.document.getElementById('history-search');
  search.value = 'box';
  search.dispatchEvent(new win.Event('input', {bubbles: true}));
  send(win, historyReply(posted, SESSIONS, MACHINES));
  assert.ok(
    groups(win)
      .filter(g => g.style.display !== 'none')
      .every(g => !g.classList.contains('collapsed')),
    'a search unfolds its matches',
  );
  win.close();
  console.log('  ok - selecting a machine keeps the chats collapsed');
}

function runTests() {
  testHiddenUntilAMachineIsRegistered();
  testRowsCarryTheirMachineAndChip();
  testSelectingAMachineFiltersRows();
  testSelectingAMachineKeepsChatsCollapsed();
  testSelectionSurvivesRefreshesAndRemoval();
  testClickingARemoteRowOpensItsMachine();
  testChatHeaderOfARemoteChatOpensItsMachine();
}

try {
  runTests();
  console.log('\n7 passed, 0 failed');
  process.exit(0);
} catch (err) {
  console.error('FAIL:', err && err.stack ? err.stack : err);
  process.exit(1);
}
