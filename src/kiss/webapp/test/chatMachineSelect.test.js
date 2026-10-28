// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// The machine drop-down at the top of the chat pane (#chat-machine):
//
// * hidden until a name is known; configData.machine is its first
//   entry, the reachable registered machines of a `machines` reply
//   follow (unreachable ones are left out);
// * a pick is ONE global setting: it is sent to the daemon
//   (`setMachine`), which persists it (configData's `last_machine`)
//   and broadcasts it (`machineChanged`) to every surface; a new chat
//   shows the global pick and its submit carries it as `machine` (''
//   for this machine);
// * a chat that shows a task (running or finished) shows the machine
//   that task ran on -- the daemon's binding (`tabMachine`), this
//   machine for an unbound chat -- whatever the global pick, and
//   switching to a new chat shows the global pick again;
// * a pick made on a chat that shows a task is also where THAT chat's
//   next task goes: kept (persisted across a reload) until the submit,
//   the daemon's binding does not overwrite it; a closed tab forgets it;
// * the drop-down turns `remote` (green) for a registered machine, and
//   `down` (red) once the daemon reports that machine unreachable;
//   a pick whose machine went down stays listed so it is not lost;
// * a machine is listed by the `label` of the `machines` reply (its
//   own host name when it was registered by its address), its value
//   stays the registered name;
// * the drop-down stays enabled while the chat runs: a pick made then
//   does not move the running task (a message typed meanwhile is
//   appended to it, no `machine`), it is where the next submit goes.
'use strict';

const assert = require('assert');
const {
  makePage: makeHarnessPage,
  send,
  pickMachine: pick,
} = require('./simplify2_harness');
const {lastGetHistory, makeRow} = require('./history_helpers');

function makePage(state) {
  const {win, posted, getState} = makeHarnessPage({
    bodyClass: 'remote-chat',
    initialState: state || undefined,
  });
  return {win, posted, state: getState};
}

const MACHINES = [
  {
    name: 'laptop',
    local: true,
    status: 'reachable',
    address: 'https://10.0.0.5:8787',
  },
  {name: 'gpu-box', status: 'reachable', address: 'ssh me@gpu'},
  {
    name: 'office',
    status: 'unreachable',
    address: 'ssh me@office',
    error: 'ssh exited',
  },
];
// MACHINES after a ping round that found gpu-box unreachable.
const GPU_BOX_DOWN = MACHINES.map(m =>
  m.name === 'gpu-box' ? Object.assign({}, m, {status: 'unreachable'}) : m,
);

function select(win) {
  return win.document.getElementById('chat-machine');
}

function options(win) {
  return Array.from(select(win).options).map(o => [o.value, o.textContent]);
}

function submit(win, text) {
  win.document.getElementById('task-input').value = text;
  win.document.getElementById('send-btn').click();
}

// The persisted unsent picks, as a plain object of this realm (the
// page's objects have the page's prototypes, which deepStrictEqual
// compares).
function persistedPicks(state) {
  return JSON.parse(JSON.stringify(state().tabMachinePicks));
}

function lastSubmit(posted) {
  const submits = posted.filter(m => m && m.type === 'submit');
  return submits[submits.length - 1];
}

function setMachines(posted) {
  return posted.filter(m => m && m.type === 'setMachine').map(m => m.machine);
}

// A page that knows this machine's name and the registered machines.
function readyPage(state) {
  const page = makePage(state);
  send(page.win, {type: 'configData', config: {}, machine: 'laptop'});
  send(page.win, {type: 'machines', machines: MACHINES, checkedAt: 1});
  return page;
}

// Run a task on the active tab there and then: a `status running`
// from the daemon and its end.
function runTask(win, posted, text) {
  submit(win, text);
  const tabId = win._testApi.getActiveTabId();
  send(win, {type: 'status', running: true, tabId});
  send(win, {type: 'status', running: false, tabId});
  return lastSubmit(posted);
}

function testListsThisMachineThenTheReachableOnes() {
  const {win} = makePage();
  assert.deepStrictEqual(
    options(win),
    [],
    'empty (hidden) until a name is known',
  );
  send(win, {type: 'configData', config: {}, machine: 'laptop'});
  assert.deepStrictEqual(options(win), [['', 'laptop']]);
  assert.strictEqual(select(win).value, '');
  assert.strictEqual(
    win.document.getElementById('status-machine').textContent,
    'laptop',
  );
  send(win, {type: 'machines', machines: MACHINES, checkedAt: 1});
  assert.deepStrictEqual(options(win), [
    ['', 'laptop'],
    ['gpu-box', 'gpu-box'],
  ]);
  assert.ok(!select(win).disabled);
  assert.ok(!select(win).classList.contains('remote'));
  win.close();
  console.log(
    '  ok - lists this machine first, then the reachable registered machines',
  );
}
function testAPickIsTheGlobalSettingOfEveryNewChat() {
  const {win, posted, state} = readyPage();
  const api = win._testApi;
  const first = api.getActiveTabId();
  pick(win, 'gpu-box');
  assert.strictEqual(select(win).value, 'gpu-box');
  assert.ok(
    select(win).classList.contains('remote'),
    'a remote pick is marked',
  );
  assert.deepStrictEqual(
    setMachines(posted),
    ['gpu-box'],
    'the pick is sent to the daemon as the global setting',
  );
  assert.deepStrictEqual(
    persistedPicks(state),
    {},
    'a new chat has no pick of its own: it shows the global one',
  );

  api.createNewTab();
  const second = api.getActiveTabId();
  assert.strictEqual(
    select(win).value,
    'gpu-box',
    'a new chat shows the global pick',
  );
  assert.ok(select(win).classList.contains('remote'));
  const sub = runTask(win, posted, 'hello from the second tab');
  assert.strictEqual(sub.tabId, second);
  assert.strictEqual(sub.machine, 'gpu-box', 'the submit names the machine');
  assert.ok(!select(win).disabled, 'still enabled while the chat runs');

  // Back home in a new chat: the global pick is this machine again.
  api.createNewTab();
  assert.strictEqual(select(win).value, 'gpu-box');
  pick(win, '');
  assert.deepStrictEqual(setMachines(posted), ['gpu-box', '']);
  assert.ok(!select(win).classList.contains('remote'));
  assert.strictEqual(runTask(win, posted, 'back home').machine, '');

  // The chats keep the machine their task ran on; a new chat shows
  // the global pick.
  api.switchToTab(second);
  assert.strictEqual(
    select(win).value,
    'gpu-box',
    'the chat shows the machine its task ran on',
  );
  api.switchToTab(first);
  assert.strictEqual(select(win).value, '', 'a new chat: the global pick');
  api.createNewTab();
  assert.strictEqual(select(win).value, '');
  win.close();
  console.log('  ok - a pick is the global setting of every new chat');
}

function testAPickOnAChatShowingATaskGoesToItsNextTask() {
  const {win, posted, state} = readyPage();
  const api = win._testApi;
  const first = api.getActiveTabId();
  assert.strictEqual(runTask(win, posted, 'run here').machine, '');
  // The chat shows a task of this machine: a pick of gpu-box is the
  // global setting AND where this chat's next task goes.
  pick(win, 'gpu-box');
  assert.strictEqual(select(win).value, 'gpu-box');
  assert.deepStrictEqual(setMachines(posted), ['gpu-box']);
  assert.deepStrictEqual(
    persistedPicks(state),
    {[first]: 'gpu-box'},
    'the unsent pick is kept for the chat',
  );
  api.createNewTab();
  const second = api.getActiveTabId();
  assert.strictEqual(select(win).value, 'gpu-box', 'the global pick');
  pick(win, '');
  assert.deepStrictEqual(
    persistedPicks(state),
    {[first]: 'gpu-box'},
    'a pick on a new chat is the global setting only',
  );
  api.switchToTab(first);
  assert.strictEqual(
    select(win).value,
    'gpu-box',
    'the unsent pick stands whatever the global setting',
  );
  // The daemon's binding (replayed on a reconnect, say) does not
  // overwrite the unsent pick either...
  send(win, {type: 'tabMachine', tabId: first, machine: ''});
  assert.strictEqual(select(win).value, 'gpu-box');
  // ...the submit carries it, and the chat's machine is gpu-box from
  // then on: the daemon's binding confirms it and takes the pick over.
  const sub = runTask(win, posted, 'now on the gpu');
  assert.strictEqual(sub.tabId, first);
  assert.strictEqual(sub.machine, 'gpu-box');
  assert.deepStrictEqual(
    persistedPicks(state),
    {[first]: 'gpu-box'},
    'sent: the pick stands until the daemon binds the chat',
  );
  assert.strictEqual(select(win).value, 'gpu-box');
  send(win, {type: 'tabMachine', tabId: first, machine: 'gpu-box'});
  assert.deepStrictEqual(persistedPicks(state), {}, 'bound: no pick left');
  assert.strictEqual(select(win).value, 'gpu-box');
  // Re-picking the chat's own machine is no pick of the chat's.
  pick(win, 'gpu-box');
  assert.deepStrictEqual(persistedPicks(state), {});
  // The daemon bringing the chat home (the run refused, say) shows.
  send(win, {type: 'tabMachine', tabId: first, machine: ''});
  assert.strictEqual(select(win).value, '', 'the binding rules');
  api.switchToTab(second);
  assert.strictEqual(select(win).value, 'gpu-box', 'the global pick');
  win.close();
  console.log(
    "  ok - a pick on a chat showing a task goes to the chat's next task",
  );
}

function testTheDaemonsBindingIsTheChatsMachine() {
  const {win, state} = readyPage();
  const api = win._testApi;
  const first = api.getActiveTabId();
  api.createNewTab();
  const second = api.getActiveTabId();
  // Another surface bound the first (background) tab: it shows that
  // machine once switched to; the on-screen tab is untouched.
  send(win, {type: 'tabMachine', tabId: first, machine: 'gpu-box'});
  assert.strictEqual(select(win).value, '');
  assert.deepStrictEqual(persistedPicks(state), {}, 'a binding is no pick');
  api.switchToTab(first);
  assert.strictEqual(
    select(win).value,
    'gpu-box',
    'the bound tab shows the machine',
  );
  // ...and the on-screen tab, bound then brought home.
  api.switchToTab(second);
  send(win, {type: 'tabMachine', tabId: second, machine: 'gpu-box'});
  assert.strictEqual(select(win).value, 'gpu-box');
  send(win, {type: 'tabMachine', tabId: second, machine: ''});
  assert.strictEqual(select(win).value, '');
  send(win, {type: 'tabMachine', tabId: '', machine: 'gpu-box'});
  assert.strictEqual(select(win).value, '', 'no tab, no binding');
  // A pick on a bound chat is the chat's until submitted.
  api.switchToTab(first);
  pick(win, '');
  assert.deepStrictEqual(persistedPicks(state), {[first]: ''});
  send(win, {type: 'tabMachine', tabId: first, machine: 'gpu-box'});
  assert.strictEqual(select(win).value, '', 'the unsent pick stands');
  send(win, {type: 'tabMachine', tabId: first, machine: ''});
  assert.strictEqual(select(win).value, '');
  assert.deepStrictEqual(
    persistedPicks(state),
    {},
    'the binding caught up with the pick',
  );
  win.close();
  console.log("  ok - the daemon's binding is the machine the chat shows");
}

// A remote pick is painted like the Machines subpanel's dot: `remote`
// alone (green) while its machine answers the daemon's pings, `down`
// too (red) once a `machines` reply has shown it unreachable, and
// never `down` before any reply (the global pick restored from the
// daemon's config at load is not presumed dead).  The pick itself stays
// listed and selected while its machine is down.
function testARemotePickIsGreenUntilTheDaemonReportsItDown() {
  const {win} = readyPage();
  pick(win, 'gpu-box');
  assert.ok(select(win).classList.contains('remote'));
  assert.ok(!select(win).classList.contains('down'), 'reachable: not down');
  send(win, {type: 'machines', machines: GPU_BOX_DOWN, checkedAt: 2});
  assert.ok(select(win).classList.contains('remote'));
  assert.ok(select(win).classList.contains('down'), 'reported down');
  assert.deepStrictEqual(options(win), [
    ['', 'laptop'],
    ['gpu-box', 'gpu-box'],
  ]);
  assert.strictEqual(
    select(win).value,
    'gpu-box',
    'the pick is kept while the machine is down',
  );
  send(win, {type: 'machines', machines: MACHINES, checkedAt: 3});
  assert.ok(!select(win).classList.contains('down'), 'back up');
  pick(win, '');
  assert.ok(!select(win).classList.contains('remote'));
  assert.ok(!select(win).classList.contains('down'));
  win.close();

  const again = makePage();
  send(again.win, {
    type: 'configData',
    config: {last_machine: 'gpu-box'},
    machine: 'laptop',
  });
  assert.strictEqual(select(again.win).value, 'gpu-box');
  assert.ok(select(again.win).classList.contains('remote'));
  assert.ok(
    !select(again.win).classList.contains('down'),
    'no machines reply yet: the restored pick is not presumed down',
  );
  send(again.win, {type: 'machines', machines: GPU_BOX_DOWN, checkedAt: 4});
  assert.ok(select(again.win).classList.contains('down'));
  again.win.close();
  console.log('  ok - a remote pick is green until the daemon reports it down');
}

// A machine registered by its address (its name is the address's
// host) is listed under the label of the daemon's `machines` reply --
// the machine's own host name -- while its value stays the registered
// name, which is what the submit carries.  The label is kept for a
// pick whose machine is down, and is the name until the first reply.
function testAnAddressNamedMachineIsListedByItsHostName() {
  const rows = MACHINES.concat([
    {
      name: '10.0.0.7',
      label: 'rack-7',
      status: 'reachable',
      address: 'ssh me@10.0.0.7',
    },
  ]);
  const {win, posted} = makePage();
  send(win, {type: 'configData', config: {}, machine: 'laptop'});
  send(win, {type: 'machines', machines: rows, checkedAt: 1});
  assert.deepStrictEqual(options(win), [
    ['', 'laptop'],
    ['gpu-box', 'gpu-box'],
    ['10.0.0.7', 'rack-7'],
  ]);
  pick(win, '10.0.0.7');
  assert.strictEqual(
    runTask(win, posted, 'hello rack').machine,
    '10.0.0.7',
    'the submit carries the registered name',
  );
  const down = rows.map(m =>
    m.name === '10.0.0.7' ? Object.assign({}, m, {status: 'unreachable'}) : m,
  );
  send(win, {type: 'machines', machines: down, checkedAt: 2});
  assert.deepStrictEqual(
    options(win),
    [
      ['', 'laptop'],
      ['gpu-box', 'gpu-box'],
      ['10.0.0.7', 'rack-7'],
    ],
    'the down pick keeps its label',
  );
  win.close();

  const again = makePage();
  send(again.win, {
    type: 'configData',
    config: {last_machine: '10.0.0.7'},
    machine: 'laptop',
  });
  assert.deepStrictEqual(
    options(again.win),
    [
      ['', 'laptop'],
      ['10.0.0.7', '10.0.0.7'],
    ],
    'the registered name until the first machines reply',
  );
  send(again.win, {type: 'machines', machines: rows, checkedAt: 3});
  assert.deepStrictEqual(options(again.win), [
    ['', 'laptop'],
    ['gpu-box', 'gpu-box'],
    ['10.0.0.7', 'rack-7'],
  ]);
  again.win.close();
  console.log('  ok - an address-named machine is listed by its host name');
}
function testAClosedTabForgetsAndAReloadRestores() {
  const {win, posted, state} = readyPage();
  const api = win._testApi;
  const first = api.getActiveTabId();
  runTask(win, posted, 'one');
  pick(win, 'gpu-box');
  api.createNewTab();
  const second = api.getActiveTabId();
  pick(win, '');
  runTask(win, posted, 'two');
  pick(win, 'gpu-box');
  assert.deepStrictEqual(persistedPicks(state), {
    [first]: 'gpu-box',
    [second]: 'gpu-box',
  });
  win.document
    .querySelector(`[data-tab-id="${second}"] .chat-tab-close`)
    .click();
  assert.deepStrictEqual(
    persistedPicks(state),
    {[first]: 'gpu-box'},
    'a closed tab forgets its pick',
  );
  const persisted = state();
  win.close();

  // The reload hands the unsent pick back: the registry snapshot
  // re-opens the tab under its id, and the tab finds its pick again.
  // The pick was never submitted, so the daemon's binding replayed on
  // the reconnect does not override it either.  The global pick comes
  // back from the daemon's config.
  const again = makePage(persisted);
  send(again.win, {
    type: 'configData',
    config: {last_machine: 'gpu-box'},
    machine: 'laptop',
  });
  send(again.win, {type: 'machines', machines: MACHINES, checkedAt: 1});
  send(again.win, {
    type: 'tabs_state',
    tabs: [{tabId: first, title: 'one'}],
  });
  again.win._testApi.switchToTab(first);
  assert.strictEqual(again.win._testApi.getActiveTabId(), first);
  assert.strictEqual(
    select(again.win).value,
    'gpu-box',
    'restored after the reload',
  );
  assert.ok(select(again.win).classList.contains('remote'));
  send(again.win, {type: 'tabMachine', tabId: first, machine: ''});
  assert.strictEqual(
    select(again.win).value,
    'gpu-box',
    'the unsent pick survives the reload',
  );
  again.win._testApi.createNewTab();
  assert.strictEqual(
    select(again.win).value,
    'gpu-box',
    "a new chat shows the daemon's global pick",
  );
  again.win.close();
  console.log(
    '  ok - a closed tab forgets its pick; a reload restores the picks',
  );
}

function testReachingForTheDropDownAsksForAFreshList() {
  const {win, posted} = makePage();
  send(win, {type: 'configData', config: {}, machine: 'laptop'});
  posted.length = 0;
  select(win).dispatchEvent(new win.Event('focus'));
  assert.deepStrictEqual(
    posted.map(m => m.type),
    ['getMachines'],
    'a focus asks the daemon for the machines again',
  );
  win.close();
  console.log('  ok - reaching for the drop-down asks for a fresh list');
}

function testTheGlobalPickFollowsTheDaemon() {
  const {win, posted, state} = readyPage();
  const api = win._testApi;
  const first = api.getActiveTabId();
  runTask(win, posted, 'run here');
  api.createNewTab();
  // Another surface picked gpu-box: this surface's new chats go there.
  send(win, {type: 'machineChanged', machine: 'gpu-box'});
  assert.strictEqual(select(win).value, 'gpu-box');
  assert.ok(select(win).classList.contains('remote'));
  assert.deepStrictEqual(setMachines(posted), [], 'not a pick: not echoed');
  assert.deepStrictEqual(persistedPicks(state), {});
  // A chat that shows a task keeps that task's machine.
  api.switchToTab(first);
  assert.strictEqual(select(win).value, '');
  send(win, {type: 'machineChanged', machine: ''});
  api.createNewTab();
  assert.strictEqual(select(win).value, '');
  // This daemon's configData carries it too; a config without one is
  // this machine.  A registered machine's config (forwarded for its
  // settings sheet) says nothing about it.
  send(win, {
    type: 'configData',
    config: {last_machine: 'gpu-box'},
    machine: 'laptop',
  });
  assert.strictEqual(select(win).value, 'gpu-box');
  send(win, {
    type: 'configData',
    forwarded: true,
    machine: 'office',
    config: {last_machine: ''},
  });
  assert.strictEqual(select(win).value, 'gpu-box');
  send(win, {type: 'configData', config: {}, machine: 'laptop'});
  assert.strictEqual(select(win).value, '');
  send(win, {type: 'machineChanged', machine: 42});
  assert.strictEqual(select(win).value, '', 'junk is this machine');
  win.close();
  console.log(
    "  ok - the global pick follows the daemon's broadcast and config",
  );
}

function testAPickMadeWhileTheChatRunsGoesToTheNextTask() {
  const {win, posted, state} = readyPage();
  const api = win._testApi;
  const tab = api.getActiveTabId();
  submit(win, 'run here');
  assert.strictEqual(lastSubmit(posted).machine, '', 'the task runs here');
  send(win, {type: 'status', running: true, tabId: tab});
  assert.ok(!select(win).disabled, 'the drop-down stays usable mid-run');

  // The user picks gpu-box while the local task runs: the pick is
  // taken and persisted...
  pick(win, 'gpu-box');
  assert.strictEqual(select(win).value, 'gpu-box');
  assert.ok(select(win).classList.contains('remote'));
  assert.deepStrictEqual(persistedPicks(state), {[tab]: 'gpu-box'});

  // ...a message typed meanwhile still goes to the running task, as a
  // follow-up without a machine (the daemon keeps it where it runs)...
  posted.length = 0;
  submit(win, 'and also do this');
  assert.deepStrictEqual(
    posted.filter(m => m.type === 'submit'),
    [],
    'no new task while one runs',
  );
  const appended = posted.find(m => m.type === 'appendUserMessage');
  assert.ok(appended, 'the text is appended to the running task');
  assert.strictEqual(appended.tabId, tab);
  assert.ok(!('machine' in appended), 'the follow-up names no machine');
  assert.strictEqual(select(win).value, 'gpu-box', 'the pick is kept');

  // ...the daemon's binding of the running task (replayed on a
  // reconnect, say) does not override the unsent pick...
  send(win, {type: 'tabMachine', tabId: tab, machine: ''});
  assert.strictEqual(select(win).value, 'gpu-box');

  // ...and once the task is over, the next submit carries it.
  send(win, {type: 'status', running: false, tabId: tab});
  assert.strictEqual(select(win).value, 'gpu-box');
  posted.length = 0;
  submit(win, 'now on the gpu');
  const sub = lastSubmit(posted);
  assert.strictEqual(sub.tabId, tab);
  assert.strictEqual(sub.machine, 'gpu-box', 'the next task goes there');
  win.close();
  console.log('  ok - a pick made while the chat runs goes to the next task');
}

// A chat's transcript replayed into `tabId` (the chat is this
// machine's unless `machine` names one).
function replay(win, tabId, chatId, task, machine) {
  send(win, {type: 'tabMachine', tabId, machine: machine || ''});
  send(win, {
    type: 'task_events',
    tabId,
    chat_id: chatId,
    task_id: 'task-' + chatId,
    task,
    events: [],
    extra: {},
  });
}

function localHistory(win, posted, chatId, title) {
  win.document.getElementById('sidebar').classList.add('open');
  const last = lastGetHistory(posted);
  send(win, {
    type: 'history',
    offset: 0,
    generation: last ? last.generation : 0,
    localMachine: 'laptop',
    machines: [],
    sessions: [
      makeRow({id: chatId, task_id: 'task-' + chatId, title, preview: title}),
    ],
  });
  return win.document.querySelector('#history-list .sidebar-item');
}

// A chat that comes to show a task without a pick or a binding of its
// own -- a history row's task opened in a fresh tab, another surface's
// submit replayed into a mirrored tab -- is this machine's: the
// drop-down leaves the global pick for it.
function testAChatThatComesToShowATaskLeavesTheGlobalPick() {
  const {win, posted} = readyPage();
  const api = win._testApi;
  pick(win, 'gpu-box');
  const row = localHistory(win, posted, 'chat-1', 'local one');
  assert.ok(row, 'the history row is listed');
  row.click();
  const opened = api.getActiveTabId();
  assert.strictEqual(
    select(win).value,
    '',
    "a history row's task shows the machine it ran on",
  );
  assert.ok(!select(win).classList.contains('remote'));
  assert.strictEqual(runTask(win, posted, 'go on').machine, '');
  api.createNewTab();
  assert.strictEqual(
    select(win).value,
    'gpu-box',
    'a new chat: the global pick',
  );
  // Another surface submitted a task of this machine on this tab: its
  // replay shows the task, and the drop-down follows.
  replay(win, api.getActiveTabId(), 'chat-2', 'typed elsewhere');
  assert.strictEqual(select(win).value, '');
  api.switchToTab(opened);
  assert.strictEqual(select(win).value, '');
  win.close();
  console.log('  ok - a chat that comes to show a task leaves the global pick');
}

// A submit's machine stays the chat's unsent pick until the daemon's
// binding confirms it: a refused run (the machine down, say) leaves
// the pick standing for the retry and the chat on the machine it was
// on -- a history row of the chat still finds its tab.
function testARefusedRunKeepsTheChatsMachineAndThePick() {
  const {win, posted, state} = readyPage();
  const api = win._testApi;
  const tab = api.getActiveTabId();
  replay(win, tab, 'chat-1', 'run here');
  pick(win, 'gpu-box');
  submit(win, 'now on the gpu');
  assert.strictEqual(lastSubmit(posted).machine, 'gpu-box');
  assert.deepStrictEqual(
    persistedPicks(state),
    {[tab]: 'gpu-box'},
    'the pick stands until the daemon binds the chat',
  );
  send(win, {
    type: 'error',
    code: 'prompt_refused',
    tabId: tab,
    text: 'gpu-box is unreachable',
  });
  send(win, {type: 'status', running: false, tabId: tab});
  assert.strictEqual(
    select(win).value,
    'gpu-box',
    'the pick stands for a retry',
  );
  assert.deepStrictEqual(persistedPicks(state), {[tab]: 'gpu-box'});
  const row = localHistory(win, posted, 'chat-1', 'run here');
  posted.length = 0;
  row.click();
  assert.strictEqual(
    api.getActiveTabId(),
    tab,
    "the chat is still this machine's: its tab is found",
  );
  assert.ok(!posted.some(m => m && m.type === 'openTab'), 'no second tab');
  // The retry goes through: the daemon's binding confirms the pick.
  submit(win, 'try again');
  assert.strictEqual(lastSubmit(posted).machine, 'gpu-box');
  send(win, {type: 'tabMachine', tabId: tab, machine: 'gpu-box'});
  assert.deepStrictEqual(persistedPicks(state), {}, 'bound: no pick left');
  assert.strictEqual(select(win).value, 'gpu-box');
  win.close();
  console.log("  ok - a refused run keeps the chat's machine and the pick");
}

// A task on a machine fans out: the daemon announces the sub-agent's
// tab (`new_tab`) and, once that tab joins the stream, its binding
// (`tabMachine`), as it did the parent's.  Switching to the sub-agent
// tab shows the machine the sub-agent runs on, not this machine.
function testASubAgentTabShowsItsParentsMachine() {
  const {win, state} = readyPage();
  const api = win._testApi;
  const tab = api.getActiveTabId();
  replay(win, tab, 'chat-1', 'fan out', 'gpu-box');
  send(win, {type: 'status', running: true, tabId: tab});
  assert.strictEqual(select(win).value, 'gpu-box');
  send(win, {
    type: 'new_tab',
    parent_tab_id: tab,
    task_id: 'child-task',
    task: 'the helper',
  });
  const sub = tab + '__sub_child-task';
  assert.ok(
    api.openTabs().some(t => t.id === sub && t.isSubagentTab),
    'the sub-agent tab opened',
  );
  assert.strictEqual(api.getActiveTabId(), tab, 'in the background');
  send(win, {type: 'tabMachine', tabId: sub, machine: 'gpu-box'});
  send(win, {type: 'status', running: true, tabId: sub});
  assert.strictEqual(select(win).value, 'gpu-box', 'the parent is unmoved');
  api.switchToTab(sub);
  assert.strictEqual(
    select(win).value,
    'gpu-box',
    'the sub-agent tab shows the machine it runs on',
  );
  assert.deepStrictEqual(persistedPicks(state), {}, 'a binding is no pick');
  api.switchToTab(tab);
  assert.strictEqual(select(win).value, 'gpu-box');
  // The sub-agent ends and its tab closes: nothing of it is left over.
  send(win, {type: 'subagentDone', tab_id: sub});
  assert.ok(
    !api.openTabs().some(t => t.id === sub),
    'the sub-agent tab closed',
  );
  assert.strictEqual(api.getActiveTabId(), tab);
  assert.strictEqual(select(win).value, 'gpu-box');
  win.close();
  console.log("  ok - a sub-agent tab shows its parent's machine");
}

try {
  testListsThisMachineThenTheReachableOnes();
  testASubAgentTabShowsItsParentsMachine();
  testAChatThatComesToShowATaskLeavesTheGlobalPick();
  testARefusedRunKeepsTheChatsMachineAndThePick();
  testAPickIsTheGlobalSettingOfEveryNewChat();
  testAPickOnAChatShowingATaskGoesToItsNextTask();
  testTheDaemonsBindingIsTheChatsMachine();
  testARemotePickIsGreenUntilTheDaemonReportsItDown();
  testAnAddressNamedMachineIsListedByItsHostName();
  testAClosedTabForgetsAndAReloadRestores();
  testReachingForTheDropDownAsksForAFreshList();
  testTheGlobalPickFollowsTheDaemon();
  testAPickMadeWhileTheChatRunsGoesToTheNextTask();
  console.log('chatMachineSelect: all tests passed');
} catch (err) {
  console.error('FAIL:', err && err.stack ? err.stack : err);
  process.exit(1);
}
