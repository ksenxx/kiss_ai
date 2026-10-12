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
// * the pick is per tab: it survives a tab switch, a submit carries it
//   as `machine` ('' for this machine), a closed tab forgets it, a
//   reload restores it from the persisted state;
// * the drop-down is disabled while the chat runs and turns `remote`
//   for a registered machine; a pick whose machine went down stays
//   listed so it is not lost;
// * the daemon's `tabMachine` (the binding it made for a submit from
//   any surface) sets or clears the tab's pick, so every surface shows
//   the same machine for the tab.
'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

function makeWebview(state) {
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  html = html.replace('{{BODY_CLASS_ATTR}}', ' class="remote-chat"');
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
  let saved = state || undefined;
  win.acquireVsCodeApi = function () {
    return {
      postMessage: msg => posted.push(msg),
      getState: () => saved,
      setState: s => {
        saved = s;
      },
    };
  };
  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
  return {win, posted, state: () => saved};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
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

function select(win) {
  return win.document.getElementById('chat-machine');
}

function options(win) {
  return Array.from(select(win).options).map(o => [o.value, o.textContent]);
}

function pick(win, value) {
  select(win).value = value;
  select(win).dispatchEvent(new win.Event('change', {bubbles: true}));
}

function submit(win, text) {
  win.document.getElementById('task-input').value = text;
  win.document.getElementById('send-btn').click();
}

// The persisted picks, as a plain object of this realm (the page's
// objects have the page's prototypes, which deepStrictEqual compares).
function persistedPicks(state) {
  return JSON.parse(JSON.stringify(state().tabMachines));
}

function lastSubmit(posted) {
  const submits = posted.filter(m => m && m.type === 'submit');
  return submits[submits.length - 1];
}

function testListsThisMachineThenTheReachableOnes() {
  const {win} = makeWebview();
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

function testThePickIsPerTabAndTravelsWithTheSubmit() {
  const {win, posted, state} = makeWebview();
  const api = win._testApi;
  send(win, {type: 'configData', config: {}, machine: 'laptop'});
  send(win, {type: 'machines', machines: MACHINES, checkedAt: 1});
  const first = api.getActiveTabId();
  pick(win, 'gpu-box');
  assert.strictEqual(select(win).value, 'gpu-box');
  assert.ok(
    select(win).classList.contains('remote'),
    'a remote pick is marked',
  );
  assert.deepStrictEqual(
    persistedPicks(state),
    {[first]: 'gpu-box'},
    'persisted',
  );

  api.createNewTab();
  const second = api.getActiveTabId();
  assert.strictEqual(select(win).value, '', 'a new tab runs here');
  assert.ok(!select(win).classList.contains('remote'));
  submit(win, 'hello from the second tab');
  assert.strictEqual(lastSubmit(posted).machine, '', 'this machine: empty');
  send(win, {type: 'status', running: false, tabId: second});

  api.switchToTab(first);
  assert.strictEqual(
    select(win).value,
    'gpu-box',
    'the first tab keeps its pick',
  );
  submit(win, 'hello from gpu-box');
  const sub = lastSubmit(posted);
  assert.strictEqual(sub.tabId, first);
  assert.strictEqual(sub.machine, 'gpu-box', 'the submit names the machine');
  send(win, {type: 'status', running: true, tabId: first});
  assert.ok(select(win).disabled, 'disabled while the chat runs');
  send(win, {type: 'status', running: false, tabId: first});
  assert.ok(!select(win).disabled, 'enabled again once the chat is idle');

  // Back home: the next submit carries no machine.
  pick(win, '');
  assert.deepStrictEqual(persistedPicks(state), {}, 'the pick is forgotten');
  submit(win, 'back home');
  assert.strictEqual(lastSubmit(posted).machine, '');
  win.close();
  console.log('  ok - the pick is per tab and travels with the submit');
}

function testADownMachineKeepsThePickListed() {
  const {win} = makeWebview();
  send(win, {type: 'configData', config: {}, machine: 'laptop'});
  send(win, {type: 'machines', machines: MACHINES, checkedAt: 1});
  pick(win, 'gpu-box');
  const down = MACHINES.map(m =>
    m.name === 'gpu-box' ? Object.assign({}, m, {status: 'unreachable'}) : m,
  );
  send(win, {type: 'machines', machines: down, checkedAt: 2});
  assert.deepStrictEqual(options(win), [
    ['', 'laptop'],
    ['gpu-box', 'gpu-box'],
  ]);
  assert.strictEqual(
    select(win).value,
    'gpu-box',
    'the pick is kept while the machine is down',
  );
  win.close();
  console.log('  ok - a pick whose machine is down stays listed');
}

function testAClosedTabForgetsAndAReloadRestores() {
  const {win, state} = makeWebview();
  const api = win._testApi;
  send(win, {type: 'configData', config: {}, machine: 'laptop'});
  send(win, {type: 'machines', machines: MACHINES, checkedAt: 1});
  const first = api.getActiveTabId();
  pick(win, 'gpu-box');
  api.createNewTab();
  const second = api.getActiveTabId();
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

  // The reload hands the picks back: the registry snapshot re-opens
  // the tab under its id, and the tab finds its pick again.  The pick
  // was never submitted, so the daemon's binding replayed on the
  // reconnect does not override it either.
  const again = makeWebview(persisted);
  send(again.win, {type: 'configData', config: {}, machine: 'laptop'});
  send(again.win, {type: 'machines', machines: MACHINES, checkedAt: 1});
  send(again.win, {
    type: 'tabs_state',
    tabs: [{tabId: first, title: 'gpu chat'}],
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
  again.win.close();
  console.log(
    '  ok - a closed tab forgets its pick; a reload restores the picks',
  );
}

function testReachingForTheDropDownAsksForAFreshList() {
  const {win, posted} = makeWebview();
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

function testTheDaemonsBindingSyncsThePick() {
  const {win, state} = makeWebview();
  const api = win._testApi;
  send(win, {type: 'configData', config: {}, machine: 'laptop'});
  send(win, {type: 'machines', machines: MACHINES, checkedAt: 1});
  const first = api.getActiveTabId();
  api.createNewTab();
  const second = api.getActiveTabId();
  // Another surface bound the first (background) tab: the pick is
  // stored and persisted; the on-screen tab is untouched.
  send(win, {type: 'tabMachine', tabId: first, machine: 'gpu-box'});
  assert.strictEqual(select(win).value, '');
  assert.deepStrictEqual(persistedPicks(state), {[first]: 'gpu-box'});
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
  assert.deepStrictEqual(persistedPicks(state), {[first]: 'gpu-box'});
  send(win, {type: 'tabMachine', tabId: '', machine: 'gpu-box'});
  assert.deepStrictEqual(
    persistedPicks(state),
    {[first]: 'gpu-box'},
    'no tab, no pick',
  );
  // A pick made here and not submitted yet stands against the daemon's
  // binding (a reconnect replays it): the next task goes where the
  // user said.  Once submitted, the binding is the pick again.
  api.switchToTab(first);
  pick(win, '');
  send(win, {type: 'tabMachine', tabId: first, machine: 'gpu-box'});
  assert.strictEqual(select(win).value, '', 'the unsent pick stands');
  send(win, {type: 'tabMachine', tabId: first, machine: ''});
  assert.strictEqual(select(win).value, '');
  pick(win, 'gpu-box');
  submit(win, 'go');
  send(win, {type: 'status', running: false, tabId: first});
  send(win, {type: 'tabMachine', tabId: first, machine: ''});
  assert.strictEqual(
    select(win).value,
    '',
    'after the submit the binding rules',
  );
  win.close();
  console.log("  ok - the daemon's tabMachine syncs the pick across surfaces");
}

try {
  testListsThisMachineThenTheReachableOnes();
  testTheDaemonsBindingSyncsThePick();
  testThePickIsPerTabAndTravelsWithTheSubmit();
  testADownMachineKeepsThePickListed();
  testAClosedTabForgetsAndAReloadRestores();
  testReachingForTheDropDownAsksForAFreshList();
  console.log('\n6 passed, 0 failed');
  process.exit(0);
} catch (err) {
  console.error('FAIL:', err && err.stack ? err.stack : err);
  process.exit(1);
}
