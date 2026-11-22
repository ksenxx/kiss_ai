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
// * the daemon's answer to `ready` names every binding (`tabMachines`):
//   one the page holds that is not among them ended while it was away;
// * a pick of another machine than the chat's opens a new chat on the
//   picked machine (a chat that is still new is that chat); the chat
//   left keeps the machine its tasks ran on;
// * a chat's next task may be headed for another machine than the one
//   the chat shows (a chat opened for a machine, a submit the daemon
//   has not bound yet): that unsent pick is kept (persisted across a
//   reload) until the binding, which does not overwrite it; a closed
//   tab forgets it;
// * the drop-down turns `remote` (green) for a registered machine, and
//   `down` (red) once the daemon reports that machine unreachable;
//   a pick whose machine went down stays listed so it is not lost;
// * a machine is listed by the `label` of the `machines` reply (its
//   own host name when it was registered by its address), its value
//   stays the registered name;
// * the drop-down stays enabled while the chat runs: a pick made then
//   does not move the running task, which stays open behind the new
//   chat (a message typed into it is appended to it, no `machine`).
'use strict';

const assert = require('assert');
const {
  makeRemoteChatPage: makePage,
  persistedPicks,
  send,
  pickMachine: pick,
} = require('./simplify2_harness');
const {lastGetHistory, makeRow} = require('./history_helpers');

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

// The name the header shows: the label the transparent select lies over.
function shown(win) {
  return select(win).previousElementSibling.textContent;
}

function submit(win, text) {
  win.document.getElementById('task-input').value = text;
  win.document.getElementById('send-btn').click();
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
  assert.strictEqual(shown(win), '');
  send(win, {type: 'configData', config: {}, machine: 'laptop'});
  assert.deepStrictEqual(options(win), [['', 'laptop']]);
  assert.strictEqual(select(win).value, '');
  assert.strictEqual(shown(win), 'laptop');
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

function testAPickOnAChatShowingATaskOpensANewChatThere() {
  const {win, posted, state} = readyPage();
  const api = win._testApi;
  const first = api.getActiveTabId();
  assert.strictEqual(runTask(win, posted, 'run here').machine, '');
  // The chat shows a task of this machine: a pick of gpu-box is the
  // global setting and opens a new chat on gpu-box.  The chat left is
  // not moved: it is still this machine's.
  posted.length = 0;
  pick(win, 'gpu-box');
  const second = api.getActiveTabId();
  assert.notStrictEqual(second, first, 'a new chat is on screen');
  assert.ok(
    posted.some(m => m.type === 'newChat' && m.tabId === second),
    'the new chat is announced to the daemon',
  );
  assert.strictEqual(select(win).value, 'gpu-box');
  assert.ok(select(win).classList.contains('remote'));
  assert.deepStrictEqual(setMachines(posted), ['gpu-box']);
  assert.deepStrictEqual(
    persistedPicks(state),
    {},
    'the new chat follows the global pick; the chat left has no pick',
  );
  api.switchToTab(first);
  assert.strictEqual(select(win).value, '', 'the chat left keeps its machine');
  api.switchToTab(second);
  // A chat that is still new is the new chat: a pick only moves it.
  pick(win, '');
  assert.strictEqual(api.getActiveTabId(), second, 'the new chat is kept');
  assert.strictEqual(select(win).value, '');
  pick(win, 'gpu-box');
  assert.strictEqual(api.getActiveTabId(), second);
  // Its task goes to the picked machine, which is the chat's from
  // then on: the daemon's binding confirms it.
  const sub = runTask(win, posted, 'now on the gpu');
  assert.strictEqual(sub.tabId, second);
  assert.strictEqual(sub.machine, 'gpu-box');
  assert.deepStrictEqual(
    persistedPicks(state),
    {[second]: 'gpu-box'},
    'sent: the machine stands until the daemon binds the chat',
  );
  send(win, {type: 'tabMachine', tabId: second, machine: 'gpu-box'});
  assert.deepStrictEqual(persistedPicks(state), {}, 'bound: no pick left');
  assert.strictEqual(select(win).value, 'gpu-box');
  // Picking the machine the chat shows moves nothing: no new chat.
  pick(win, 'gpu-box');
  assert.strictEqual(api.getActiveTabId(), second);
  assert.deepStrictEqual(persistedPicks(state), {});
  // Home from the gpu-box chat: again a new chat, on this machine.
  pick(win, '');
  const third = api.getActiveTabId();
  assert.ok(third !== second && third !== first, 'a third chat');
  assert.strictEqual(select(win).value, '');
  assert.strictEqual(runTask(win, posted, 'back home').machine, '');
  api.switchToTab(second);
  assert.strictEqual(select(win).value, 'gpu-box', 'the gpu chat is unmoved');
  win.close();
  console.log(
    '  ok - a pick on a chat showing a task opens a new chat on that machine',
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
  // A pick on a chat the daemon bound before it showed a task is the
  // chat's until submitted.
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
  assert.strictEqual(shown(win), 'rack-7', 'the pick is the name shown');
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
  assert.strictEqual(shown(again.win), '10.0.0.7');
  send(again.win, {type: 'machines', machines: rows, checkedAt: 3});
  assert.deepStrictEqual(options(again.win), [
    ['', 'laptop'],
    ['gpu-box', 'gpu-box'],
    ['10.0.0.7', 'rack-7'],
  ]);
  assert.strictEqual(shown(again.win), 'rack-7', 'the name shown follows');
  again.win.close();
  console.log('  ok - an address-named machine is listed by its host name');
}
function testAClosedTabForgetsAndAReloadRestores() {
  const {win, posted, state} = readyPage();
  const api = win._testApi;
  // Two chats sent to gpu-box that the daemon has not bound yet: each
  // holds its machine as an unsent pick.
  pick(win, 'gpu-box');
  const first = api.getActiveTabId();
  submit(win, 'one');
  api.createNewTab();
  const second = api.getActiveTabId();
  // The daemon takes this chat's prompt (its echo, then the run's
  // `clear`): once the task has ended nothing is going on in the chat.
  submit(win, 'two');
  send(win, {type: 'setTaskText', text: 'two', tabId: second});
  send(win, {type: 'clear', tabId: second});
  send(win, {type: 'status', running: true, tabId: second});
  send(win, {type: 'status', running: false, tabId: second});
  assert.deepStrictEqual(persistedPicks(state), {
    [first]: 'gpu-box',
    [second]: 'gpu-box',
  });
  // Leaving the idle chat for a new one closes its tab.
  win.document.getElementById('new-chat-btn').click();
  assert.ok(!api.openTabs().some(t => t.id === second), 'the chat is closed');
  assert.deepStrictEqual(
    persistedPicks(state),
    {[first]: 'gpu-box'},
    'a closed tab forgets its pick',
  );
  const persisted = state();
  win.close();

  // The reload hands the unsent pick back: the registry snapshot
  // re-opens the tab under its id, and the tab finds its pick again.
  // The daemon never bound the chat, so the binding replayed on the
  // reconnect (none) does not override it either.  The global pick
  // comes back from the daemon's config.
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

// A phone opens the native picker on that focus, and iOS dismisses (or
// leaves stale) a picker whose options are replaced under it: the
// reply must leave the option nodes alone when it changes nothing, and
// repaint only what a changed list (a machine down, relabelled, new)
// needs.
function testReachingForTheDropDownAsksForAFreshList() {
  const {win, posted} = readyPage();
  posted.length = 0;
  select(win).dispatchEvent(new win.Event('focus'));
  assert.deepStrictEqual(
    posted.map(m => m.type),
    ['getMachines'],
    'a focus asks the daemon for the machines again',
  );
  const before = Array.from(select(win).options);
  send(win, {type: 'machines', machines: MACHINES, checkedAt: 2});
  send(win, {type: 'configData', config: {}, machine: 'laptop'});
  const after = Array.from(select(win).options);
  assert.ok(
    after.length === before.length && after.every((o, i) => o === before[i]),
    'an unchanged list keeps the option nodes (the open picker) as they are',
  );
  assert.deepStrictEqual(options(win), [
    ['', 'laptop'],
    ['gpu-box', 'gpu-box'],
  ]);
  const relabelled = MACHINES.map(m =>
    m.name === 'gpu-box' ? Object.assign({}, m, {label: 'gpu-host'}) : m,
  );
  send(win, {type: 'machines', machines: relabelled, checkedAt: 3});
  assert.deepStrictEqual(
    options(win),
    [
      ['', 'laptop'],
      ['gpu-box', 'gpu-host'],
    ],
    'a relabelled machine is repainted',
  );
  send(win, {type: 'machines', machines: GPU_BOX_DOWN, checkedAt: 4});
  assert.deepStrictEqual(
    options(win),
    [['', 'laptop']],
    'a machine that went down (not picked) leaves the list',
  );
  win.close();
  console.log(
    '  ok - reaching for the drop-down asks for a fresh list without ' +
      'repainting an unchanged one',
  );
}

function testTheGlobalPickFollowsTheDaemon() {
  const {win, posted, state} = readyPage();
  const api = win._testApi;
  const first = api.getActiveTabId();
  runTask(win, posted, 'run here');
  api.createNewTab();
  const fresh = api.getActiveTabId();
  const openBefore = api.openTabs().length;
  // Another surface picked gpu-box: this surface's new chats go there.
  send(win, {type: 'machineChanged', machine: 'gpu-box'});
  assert.strictEqual(select(win).value, 'gpu-box');
  assert.ok(select(win).classList.contains('remote'));
  assert.deepStrictEqual(setMachines(posted), [], 'not a pick: not echoed');
  assert.deepStrictEqual(persistedPicks(state), {});
  assert.strictEqual(api.getActiveTabId(), fresh);
  // A chat that shows a task keeps that task's machine, and stays on
  // screen: only a pick in this drop-down opens a new chat.
  api.switchToTab(first);
  assert.strictEqual(select(win).value, '');
  send(win, {type: 'machineChanged', machine: ''});
  send(win, {type: 'machineChanged', machine: 'gpu-box'});
  assert.strictEqual(
    api.getActiveTabId(),
    first,
    "another surface's pick opens no chat here",
  );
  assert.strictEqual(api.openTabs().length, openBefore);
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

// A pick made while the chat on screen runs a task opens a new chat on
// the picked machine and leaves the run alone: the running chat stays
// open on its own machine, takes follow-ups, and its next task goes
// where its tasks ran.
function testAPickMadeWhileTheChatRunsOpensANewChatAndLeavesTheRun() {
  const {win, posted, state} = readyPage();
  const api = win._testApi;
  const tab = api.getActiveTabId();
  submit(win, 'run here');
  assert.strictEqual(lastSubmit(posted).machine, '', 'the task runs here');
  send(win, {type: 'status', running: true, tabId: tab});
  assert.ok(!select(win).disabled, 'the drop-down stays usable mid-run');

  pick(win, 'gpu-box');
  const fresh = api.getActiveTabId();
  assert.notStrictEqual(fresh, tab, 'a new chat is on screen');
  assert.ok(
    api.openTabs().some(t => t.id === tab),
    'the running chat stays open',
  );
  assert.strictEqual(select(win).value, 'gpu-box');
  assert.ok(select(win).classList.contains('remote'));
  assert.deepStrictEqual(
    persistedPicks(state),
    {},
    'no pick is kept for the running chat',
  );

  // Back on the running chat: it is still this machine's, and a message
  // typed there goes to the running task as a follow-up without a
  // machine (the daemon keeps it where it runs).
  api.switchToTab(tab);
  assert.strictEqual(select(win).value, '', 'the run was not re-targeted');
  assert.ok(!select(win).classList.contains('remote'));
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

  // Once the task is over, the chat's next task runs where it ran.
  send(win, {type: 'status', running: false, tabId: tab});
  posted.length = 0;
  submit(win, 'go on here');
  const here = lastSubmit(posted);
  assert.strictEqual(here.tabId, tab);
  assert.strictEqual(here.machine, '', 'the chat left keeps its machine');

  // The new chat's task goes to the picked machine.
  api.switchToTab(fresh);
  assert.strictEqual(select(win).value, 'gpu-box');
  posted.length = 0;
  submit(win, 'now on the gpu');
  const there = lastSubmit(posted);
  assert.strictEqual(there.tabId, fresh);
  assert.strictEqual(there.machine, 'gpu-box', 'the new chat runs there');
  win.close();
  console.log(
    '  ok - a pick made while the chat runs opens a new chat and leaves the run',
  );
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
// the pick standing for the retry.  The chat is the new one the
// drop-down pick opened.
function testARefusedRunKeepsTheChatsMachineAndThePick() {
  const {win, posted, state} = readyPage();
  const api = win._testApi;
  const left = api.getActiveTabId();
  replay(win, left, 'chat-1', 'run here');
  pick(win, 'gpu-box');
  const tab = api.getActiveTabId();
  assert.notStrictEqual(tab, left, 'the pick opened a new chat');
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

// A page that was away when a tab came home (another surface ran a task
// of this machine in it, its machine was lost) still holds the tab's
// binding.  The daemon's answer to its `ready` names every binding
// there is, and the page ends where a freshly loaded one starts.  Until
// the answer is in, what the page held stands: a prompt kept in the
// composer during the outage is sent the moment the banner goes, and
// still runs where its chat lives.
function testAReadyAnswerNamesEveryBinding() {
  const {win, posted, state} = readyPage();
  const api = win._testApi;
  send(win, {type: 'daemonStatus', connected: true});
  const home = api.getActiveTabId();
  replay(win, home, 'chat-1', 'ran on the gpu', 'gpu-box');
  api.createNewTab();
  const stays = api.getActiveTabId();
  replay(win, stays, 'chat-2', 'runs on the gpu', 'gpu-box');
  pick(win, 'gpu-box');
  // A new chat for this machine while the global pick is the gpu: its
  // tab holds an unsent pick of this machine.
  api.createNewTab('');
  const fresh = api.getActiveTabId();
  assert.deepStrictEqual(persistedPicks(state), {[fresh]: ''});

  send(win, {type: 'daemonStatus', connected: false, reconnecting: true});
  send(win, {type: 'daemonStatus', connected: true});
  api.switchToTab(stays);
  assert.strictEqual(
    runTask(win, posted, 'sent as the banner went').machine,
    'gpu-box',
    "the daemon's answer is not in yet: the binding held stands",
  );

  api.switchToTab(home);
  assert.strictEqual(select(win).value, 'gpu-box');
  send(win, {type: 'tabMachines', machines: {[stays]: 'gpu-box'}});
  assert.strictEqual(
    select(win).value,
    '',
    'the tab on screen came home while the page was away',
  );
  assert.strictEqual(runTask(win, posted, 'go on here').machine, '');
  api.switchToTab(stays);
  assert.strictEqual(select(win).value, 'gpu-box', 'named again: it stands');
  api.switchToTab(fresh);
  assert.strictEqual(select(win).value, '');
  assert.deepStrictEqual(
    persistedPicks(state),
    {[fresh]: ''},
    "a new chat's unsent pick of this machine is not the daemon's to drop",
  );

  // A pick in the drop-down on a chat that shows a task does not
  // re-target that chat: the new chat comes on screen, on the picked
  // machine, and no pick is kept for either of them.
  api.switchToTab(home);
  pick(win, 'gpu-box');
  assert.strictEqual(
    api.getActiveTabId(),
    fresh,
    'the chat that is still new is the new chat of the pick',
  );
  assert.strictEqual(select(win).value, 'gpu-box');
  assert.deepStrictEqual(
    persistedPicks(state),
    {},
    'the new chat follows the picked machine; the chat left is not re-targeted',
  );
  // An answer that binds nothing for the new chat leaves it on the pick.
  send(win, {type: 'tabMachines', machines: {[stays]: 'gpu-box'}});
  assert.strictEqual(select(win).value, 'gpu-box');
  assert.deepStrictEqual(persistedPicks(state), {});
  // A daemon with no binding at all (it restarted) names none: a chat
  // that shows a task is back on this machine, a new chat stays on the
  // global pick.
  send(win, {type: 'tabMachines', machines: {}});
  assert.strictEqual(select(win).value, 'gpu-box');
  api.switchToTab(stays);
  assert.strictEqual(select(win).value, '');
  win.close();
  console.log("  ok - the daemon's answer to ready names every binding");
}

try {
  testListsThisMachineThenTheReachableOnes();
  testAReadyAnswerNamesEveryBinding();
  testASubAgentTabShowsItsParentsMachine();
  testAChatThatComesToShowATaskLeavesTheGlobalPick();
  testARefusedRunKeepsTheChatsMachineAndThePick();
  testAPickIsTheGlobalSettingOfEveryNewChat();
  testAPickOnAChatShowingATaskOpensANewChatThere();
  testTheDaemonsBindingIsTheChatsMachine();
  testARemotePickIsGreenUntilTheDaemonReportsItDown();
  testAnAddressNamedMachineIsListedByItsHostName();
  testAClosedTabForgetsAndAReloadRestores();
  testReachingForTheDropDownAsksForAFreshList();
  testTheGlobalPickFollowsTheDaemon();
  testAPickMadeWhileTheChatRunsOpensANewChatAndLeavesTheRun();
  console.log('chatMachineSelect: all tests passed');
} catch (err) {
  console.error('FAIL:', err && err.stack ? err.stack : err);
  process.exit(1);
}
