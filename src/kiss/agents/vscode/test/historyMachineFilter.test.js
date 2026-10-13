// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// End-to-end test of the History panel's machine filter: the daemon's
// `history` reply names the registered remote machines, every row
// carries its machine as data-machine ('' for the local daemon) with a
// chip naming the machine, the <select id="hf-machine"> lists All /
// this machine / each remote machine (and stays hidden while none is
// registered), choosing one hides the other rows, and a remote row's
// click resumes the chat HERE, in a tab the daemon binds to that
// machine (`resumeSession` carries the machine and the task title):
// the machine's own web UI is never opened, so every remote machine is
// managed from this webapp.

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

// The primary-sidebar history panel of editor-tabs mode: a history
// click asks the host for the chat editor tab (`openChatPanel`).
const PANEL_MODE_BODY_ATTRS =
  ' class="editor-tab-mode history-panel-mode"' +
  ' data-kiss-tab-id="history-panel"';

function makeWebview(bodyAttrs) {
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  html = html.replace('{{BODY_CLASS_ATTR}}', bodyAttrs || '');
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
function historyReply(posted, sessions, machines, machineLabels) {
  const requests = posted.filter(m => m && m.type === 'getHistory');
  const last = requests[requests.length - 1];
  return {
    type: 'history',
    sessions,
    offset: 0,
    generation: last ? last.generation : 0,
    machines,
    machineLabels,
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
      // A label, never a link: nothing in the row leaves this webapp.
      tag: chip ? chip.tagName : null,
      favourite: !!r.querySelector('.sidebar-item-favorite'),
      copy: !!r.querySelector('.sidebar-item-copy'),
    };
  });
  assert.deepStrictEqual(seen['local one'], {
    machine: '',
    chip: null,
    tag: null,
    favourite: true,
    copy: true,
  });
  assert.deepStrictEqual(seen['box task'], {
    machine: 'box',
    chip: 'box',
    tag: 'SPAN',
    favourite: false,
    copy: true,
  });
  assert.deepStrictEqual(seen['box task two'].machine, 'box');
  assert.deepStrictEqual(seen['lab task'].chip, 'lab');
  assert.strictEqual(
    win.document.querySelector('#history-list a[target="_blank"]'),
    null,
    'no history row links out of this webapp',
  );
  win.close();
  console.log('  ok - rows carry data-machine and a machine label');
}

// The chips and the filter call a machine by the `label` of the
// daemon's `machines` reply (its own host name when it was registered
// by its address) while the rows' data-machine and the filter's values
// stay the registered name.  A label learnt after the list was drawn
// refetches it (the sidebar is open), so the chips repaint.
function testAMachinesReplyRelabelsTheChipsAndTheFilter() {
  const {win, posted} = makeWebview();
  win.document.getElementById('menu-btn').click();
  send(win, historyReply(posted, SESSIONS, MACHINES));
  const sidebar = win.document.getElementById('sidebar');
  assert.ok(sidebar.classList.contains('open'), 'the history panel is open');
  const before = posted.filter(m => m && m.type === 'getHistory').length;
  const machines = [
    {name: 'myhost', label: 'myhost', local: true, status: 'reachable'},
    {name: 'box', label: 'box-host', status: 'reachable', address: 'ssh me@box'},
    {name: 'lab', label: 'lab', status: 'unreachable', address: 'ssh me@lab'},
  ];
  send(win, {type: 'machines', machines, checkedAt: 1});
  assert.strictEqual(
    posted.filter(m => m && m.type === 'getHistory').length,
    before + 1,
    'a new label refetches the list',
  );
  send(win, historyReply(posted, SESSIONS, MACHINES));
  const chips = {};
  rows(win).forEach(r => {
    const chip = r.querySelector('.sidebar-item-machine');
    chips[r.querySelector('.sidebar-item-text').textContent] = chip
      ? [r.dataset.machine, chip.textContent, chip.title]
      : null;
  });
  assert.deepStrictEqual(chips['box task'], [
    'box',
    'box-host',
    'On machine box-host',
  ]);
  assert.deepStrictEqual(chips['lab task'], ['lab', 'lab', 'On machine lab']);
  assert.deepStrictEqual(options(win), [
    ['', 'All'],
    ['__local__', 'myhost (this machine)'],
    ['box', 'box-host'],
    ['lab', 'lab'],
  ]);
  selectMachine(win, 'box');
  assert.strictEqual(lastHistoryRequest(posted).machine, 'box');
  // The same labels again: nothing to repaint.
  const again = posted.filter(m => m && m.type === 'getHistory').length;
  send(win, {type: 'machines', machines, checkedAt: 2});
  assert.strictEqual(
    posted.filter(m => m && m.type === 'getHistory').length,
    again,
    'unchanged labels do not refetch',
  );
  win.close();
  console.log('  ok - a machines reply relabels the chips and the filter');
}

// The daemon's `history` reply itself names the labels (`machineLabels`,
// the host names its history fetches learnt), so the filter and the
// chips read `box-host` on the very reply that draws them, with no
// `machines` reply (a history panel has no Machines subpanel), and the
// chat header's drop-down follows.
function testTheHistoryReplysLabelsNameTheFilterAndTheChips() {
  const {win, posted} = makeWebview();
  send(win, {type: 'configData', config: {}, machine: 'myhost'});
  // An older daemon's `machines` rows carry no label: they say nothing
  // about it, so the Machines subpanel shows the registered name ...
  send(win, {
    type: 'machines',
    machines: [
      {name: 'myhost', local: true, status: 'reachable'},
      {name: 'box', status: 'reachable', address: 'ssh me@box'},
    ],
    checkedAt: 1,
  });
  const subpanelNames = () =>
    Array.from(
      win.document.querySelectorAll('#meta-machines-list .sidebar-panel-name'),
    ).map(el => el.textContent);
  assert.deepStrictEqual(subpanelNames(), ['myhost', 'box']);
  send(
    win,
    historyReply(posted, SESSIONS, MACHINES, {box: 'box-host', lab: 'lab'}),
  );
  // ... until the history reply names it, which repaints the subpanel.
  assert.deepStrictEqual(subpanelNames(), ['myhost', 'box-host']);
  assert.deepStrictEqual(options(win), [
    ['', 'All'],
    ['__local__', 'myhost (this machine)'],
    ['box', 'box-host'],
    ['lab', 'lab'],
  ]);
  const chips = {};
  rows(win).forEach(r => {
    const chip = r.querySelector('.sidebar-item-machine');
    if (chip) chips[r.dataset.machine] = chip.textContent;
  });
  assert.deepStrictEqual(chips, {box: 'box-host', lab: 'lab'});
  selectMachine(win, 'box');
  assert.strictEqual(
    lastHistoryRequest(posted).machine,
    'box',
    'the filter sends the registered name',
  );
  // The drop-down at the top of the chat shows the label too.
  assert.deepStrictEqual(
    Array.from(win.document.getElementById('chat-machine').options).map(o => [
      o.value,
      o.textContent,
    ]),
    [
      ['', 'myhost'],
      ['box', 'box-host'],
    ],
  );
  win.close();
  console.log("  ok - the history reply's labels name the filter and the chips");
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

function resumes(posted) {
  return posted.filter(m => m && m.type === 'resumeSession');
}

// A posted message as a plain object of this realm (JSDOM objects
// have another prototype, which deepStrictEqual minds).
function plain(message) {
  return JSON.parse(JSON.stringify(message));
}

// The daemon's replay of a chat into a tab, after its announcement of
// the tab's binding to *machine* ('' for a local chat): it binds the
// tab to the chat client-side.
function replay(win, tabId, chatId, taskId, task, machine) {
  send(win, {type: 'tabMachine', tabId, machine: machine || ''});
  send(win, {
    type: 'task_events',
    tabId,
    chat_id: chatId,
    task_id: taskId,
    task,
    events: [],
    extra: {},
  });
}

function testClickingARemoteRowResumesTheChatHere() {
  const {win, posted} = makeWebview();
  send(win, historyReply(posted, SESSIONS, MACHINES));
  const opened = [];
  // JSDOM does not navigate: record any anchor activation instead.
  win.HTMLAnchorElement.prototype.click = function () {
    opened.push(this.href);
  };
  const remoteRow = rows(win).find(
    r => r.querySelector('.sidebar-item-text').textContent === 'box task',
  );
  posted.length = 0;
  remoteRow.click();
  assert.deepStrictEqual(opened, [], "nothing opens the machine's web UI");
  // The chat is resumed in a fresh tab here; the resume names the
  // machine (the daemon binds the tab to it) and the task (the tab's
  // title until the replay arrives).  The row has no persisted events
  // and is not running: a remote row is resumed by chat id regardless.
  const first = resumes(posted);
  assert.strictEqual(first.length, 1, 'one resume per click');
  const tabId = first[0].tabId;
  assert.ok(tabId, 'the resume names the fresh tab');
  assert.deepStrictEqual(plain(first[0]), {
    type: 'resumeSession',
    id: 'chat-2',
    taskId: 'task-2',
    tabId,
    machine: 'box',
    title: 'box task',
  });
  assert.ok(
    posted.some(m => m && m.type === 'openTab' && m.tabId === tabId),
    'the fresh tab is announced to the daemon',
  );
  // The machine's replay binds the tab to the chat; a second click on
  // the chat (here on its chip, which is part of the row) reveals that
  // tab and, the clicked task not being in the transcript (the replay
  // showed an earlier task of the chat), replays it at the task, still
  // naming the machine: without it the daemon would unbind the tab and
  // look the chat up here.
  replay(win, tabId, 'chat-2', 'task-0', 'earlier box task', 'box');
  posted.length = 0;
  remoteRow
    .querySelector('.sidebar-item-machine')
    .dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  const again = resumes(posted);
  assert.strictEqual(again.length, 1);
  assert.deepStrictEqual(plain(again[0]), {
    type: 'resumeSession',
    id: 'chat-2',
    taskId: 'task-2',
    tabId,
    machine: 'box',
    // The tab keeps its title across the replay (the replay sets it).
    title: 'earlier box task',
  });
  assert.ok(
    !posted.some(m => m && m.type === 'openTab'),
    'the open tab is reused',
  );
  // The machine name is trimmed as the filter trims it.
  const padded = rows(win).find(
    r => r.querySelector('.sidebar-item-text').textContent === 'box task two',
  );
  posted.length = 0;
  padded.click();
  assert.strictEqual(resumes(posted)[0].machine, 'box');
  // A local row's resume names no machine (nothing to bind).
  const localRow = rows(win).find(r => r.dataset.machine === '');
  posted.length = 0;
  localRow.click();
  assert.deepStrictEqual(opened, []);
  assert.ok(posted.length > 0, 'a local row talks to the daemon');
  assert.ok(
    posted.every(m => !m || !('machine' in m)),
    'a local row names no machine',
  );
  win.close();
  console.log(
    '  ok - a remote row resumes its chat here, bound to the machine',
  );
}

function testPanelModeRemoteRowAsksTheHostWithTheMachine() {
  // The primary-sidebar history panel runs no chats: the host opens
  // the chat editor tab, and must know the machine the chat lives on.
  const {win, posted} = makeWebview(PANEL_MODE_BODY_ATTRS);
  send(win, historyReply(posted, SESSIONS, MACHINES));
  const remoteRow = rows(win).find(r => r.dataset.machine === 'lab');
  posted.length = 0;
  remoteRow.click();
  assert.deepStrictEqual(
    posted.filter(m => m && m.type === 'openChatPanel').map(plain),
    [
      {
        type: 'openChatPanel',
        chatId: 'chat-4',
        taskId: 'task-4',
        title: 'lab task',
        machine: 'lab',
      },
    ],
  );
  assert.strictEqual(resumes(posted).length, 0, 'the panel resumes nothing');
  const localRow = rows(win).find(r => r.dataset.machine === '');
  posted.length = 0;
  localRow.click();
  const local = posted.find(m => m && m.type === 'openChatPanel');
  assert.ok(local, 'a local row asks for the panel too');
  assert.strictEqual(local.machine, '', 'a local chat: this machine');
  win.close();
  console.log("  ok - panel mode: the host learns the row's machine");
}

function testOpenChatFromHistoryWithAMachineResumesHere() {
  // The chat editor tab receiving the host's relay of a panel click.
  const {win, posted} = makeWebview();
  posted.length = 0;
  send(win, {
    type: 'openChatFromHistory',
    chatId: 'chat-4',
    taskId: 'task-4',
    title: 'lab task',
    machine: 'lab',
  });
  const first = resumes(posted);
  assert.strictEqual(first.length, 1);
  const tabId = first[0].tabId;
  assert.deepStrictEqual(plain(first[0]), {
    type: 'resumeSession',
    id: 'chat-4',
    taskId: 'task-4',
    tabId,
    machine: 'lab',
    title: 'lab task',
  });
  // Once the replay bound the tab, another click on the chat (at
  // another task of it) replays that tab, still naming the machine.
  replay(win, tabId, 'chat-4', 'task-4', 'lab task', 'lab');
  posted.length = 0;
  send(win, {
    type: 'openChatFromHistory',
    chatId: 'chat-4',
    taskId: 'task-9',
    title: 'lab task',
    machine: 'lab',
  });
  const again = resumes(posted);
  assert.strictEqual(again.length, 1);
  assert.strictEqual(again[0].tabId, tabId, 'the bound tab is reused');
  assert.strictEqual(again[0].taskId, 'task-9');
  assert.strictEqual(again[0].machine, 'lab');
  // A local chat's relay resumes without a machine.
  posted.length = 0;
  send(win, {
    type: 'openChatFromHistory',
    chatId: 'chat-1',
    taskId: 'task-1',
    title: 'local one',
    machine: '',
  });
  const local = resumes(posted);
  assert.strictEqual(local.length, 1);
  assert.ok(!('machine' in local[0]) && !('title' in local[0]));
  win.close();
  console.log('  ok - openChatFromHistory with a machine resumes here');
}

function testChatHeaderOfARemoteChatResumesItHere() {
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
  assert.strictEqual(remoteChip.tagName, 'SPAN', 'a label, not a link');
  assert.strictEqual(
    mixedGroup.querySelector('.history-chat-header .history-chat-machine'),
    null,
    'a chat with a local task carries no chip on its header',
  );

  posted.length = 0;
  remoteGroup.querySelector('.history-chat-header').click();
  assert.deepStrictEqual(opened, [], "the machine's web UI is not opened");
  const remote = resumes(posted);
  assert.strictEqual(remote.length, 1, 'the remote-only chat is resumed here');
  assert.strictEqual(remote[0].id, 'chat-r');
  assert.strictEqual(remote[0].machine, 'box');
  assert.strictEqual(remote[0].title, 'remote chat task');

  posted.length = 0;
  mixedGroup.querySelector('.history-chat-header').click();
  assert.deepStrictEqual(opened, []);
  const local = resumes(posted);
  assert.strictEqual(local.length, 1, 'the local copy of the chat is resumed');
  assert.strictEqual(local[0].id, 'chat-m');
  assert.ok(!('machine' in local[0]), 'a chat present here opens here');
  win.close();
  console.log(
    '  ok - a chat header resumes the remote-only chat here, bound to its machine',
  );
}

function testACopyOfTheChatOnAnotherMachineGetsItsOwnTab() {
  // Chat "chat-m" exists here and, copied, on lab: each copy is shown
  // in its own tab, bound to its machine; a click finds the right one.
  const {win, posted} = makeWebview();
  const fixture = [
    session(2, 'mixed chat, local task', '', {id: 'chat-m', has_events: true}),
    session(3, 'mixed chat, remote task', 'lab', {id: 'chat-m'}),
  ];
  send(win, historyReply(posted, fixture, MACHINES));
  const localRow = rows(win).find(r => r.dataset.machine === '');
  const remoteRow = rows(win).find(r => r.dataset.machine === 'lab');
  posted.length = 0;
  localRow.click();
  const local = resumes(posted);
  assert.strictEqual(local.length, 1);
  const localTab = local[0].tabId;
  assert.ok(!('machine' in local[0]));
  replay(win, localTab, 'chat-m', 'task-2', 'mixed chat, local task', '');
  // The remote copy, same chat id: not the local tab.
  posted.length = 0;
  remoteRow.click();
  const remote = resumes(posted);
  assert.strictEqual(remote.length, 1);
  const remoteTab = remote[0].tabId;
  assert.notStrictEqual(remoteTab, localTab, 'the copy gets its own tab');
  assert.strictEqual(remote[0].machine, 'lab');
  assert.ok(
    posted.some(m => m && m.type === 'openTab' && m.tabId === remoteTab),
  );
  // Leaving the local copy's idle tab retires it, as leaving any idle
  // chat for a history row does.
  assert.ok(
    posted.some(m => m && m.type === 'closeTab' && m.tabId === localTab),
    'the idle tab left behind is retired',
  );
  replay(win, remoteTab, 'chat-m', 'task-0', 'earlier lab task', 'lab');
  // The remote row finds its own tab again.
  posted.length = 0;
  remoteRow.click();
  const again = resumes(posted);
  assert.strictEqual(again.length, 1);
  assert.strictEqual(again[0].tabId, remoteTab);
  assert.strictEqual(again[0].machine, 'lab');
  assert.ok(!posted.some(m => m && m.type === 'openTab'), 'no new tab');
  // The local row does not take the remote copy's tab: the local copy
  // opens in a fresh tab of its own, bound to no machine.
  posted.length = 0;
  localRow.click();
  const local2 = resumes(posted);
  assert.strictEqual(local2.length, 1);
  assert.notStrictEqual(local2[0].tabId, remoteTab);
  assert.ok(!('machine' in local2[0]));
  win.close();
  console.log('  ok - a chat copied to another machine has one tab per copy');
}

function testAnUnsentMachinePickDoesNotHideTheChatsTab() {
  // The chat header's drop-down picks where the tab's NEXT task runs;
  // until submitted it says nothing about the chat the tab shows, so
  // the chat's row still finds the tab.
  const {win, posted} = makeWebview();
  send(win, {
    type: 'machines',
    machines: [
      {name: 'myhost', local: true, status: 'reachable'},
      {name: 'box', local: false, status: 'reachable'},
    ],
  });
  send(win, historyReply(posted, SESSIONS, MACHINES));
  const localRow = rows(win).find(
    r => r.querySelector('.sidebar-item-text').textContent === 'local one',
  );
  const remoteRow = rows(win).find(
    r => r.querySelector('.sidebar-item-text').textContent === 'box task',
  );
  // The remote chat shown, bound to box by the daemon.
  posted.length = 0;
  remoteRow.click();
  const remoteTab = resumes(posted)[0].tabId;
  replay(win, remoteTab, 'chat-2', 'task-0', 'earlier box task', 'box');
  // The local chat shown in its own tab (resumable: it has events).
  send(
    win,
    historyReply(
      posted,
      [session(1, 'local one', '', {has_events: true}), SESSIONS[1]],
      MACHINES,
    ),
  );
  posted.length = 0;
  rows(win)
    .find(
      r => r.querySelector('.sidebar-item-text').textContent === 'local one',
    )
    .click();
  const localTab = resumes(posted)[0].tabId;
  replay(win, localTab, 'chat-1', 'task-0', 'earlier local task', '');
  // The user picks box for the local tab's next task, submits nothing.
  const select = win.document.getElementById('chat-machine');
  select.value = 'box';
  select.dispatchEvent(new win.Event('change', {bubbles: true}));
  assert.strictEqual(select.value, 'box', 'the pick is shown');
  posted.length = 0;
  rows(win)
    .find(
      r => r.querySelector('.sidebar-item-text').textContent === 'local one',
    )
    .click();
  const again = resumes(posted);
  assert.strictEqual(again.length, 1);
  assert.strictEqual(again[0].tabId, localTab, 'the local chat keeps its tab');
  assert.ok(!('machine' in again[0]), 'a local chat: no machine');
  assert.ok(!posted.some(m => m && m.type === 'openTab'), 'no new tab');
  win.close();
  console.log("  ok - an unsent machine pick does not hide the chat's tab");
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
  const open = groups(win).find(
    g => !g.querySelector('.sidebar-item').style.display,
  );
  open.querySelector('.history-chat-collapse').click();
  assert.ok(!open.classList.contains('collapsed'));
  selectMachine(win, '');
  send(win, historyReply(posted, SESSIONS, MACHINES));
  const same = groups(win).find(g => g.dataset.chatId === open.dataset.chatId);
  assert.ok(
    !same.classList.contains('collapsed'),
    'the unfolded chat stays open',
  );
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
  testAMachinesReplyRelabelsTheChipsAndTheFilter();
  testTheHistoryReplysLabelsNameTheFilterAndTheChips();
  testSelectingAMachineFiltersRows();
  testSelectingAMachineKeepsChatsCollapsed();
  testSelectionSurvivesRefreshesAndRemoval();
  testClickingARemoteRowResumesTheChatHere();
  testPanelModeRemoteRowAsksTheHostWithTheMachine();
  testOpenChatFromHistoryWithAMachineResumesHere();
  testChatHeaderOfARemoteChatResumesItHere();
  testACopyOfTheChatOnAnotherMachineGetsItsOwnTab();
  testAnUnsentMachinePickDoesNotHideTheChatsTab();
}

try {
  runTests();
  console.log('\n13 passed, 0 failed');
  process.exit(0);
} catch (err) {
  console.error('FAIL:', err && err.stack ? err.stack : err);
  process.exit(1);
}
