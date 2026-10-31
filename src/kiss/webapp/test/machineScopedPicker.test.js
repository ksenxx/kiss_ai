// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// The model picker and the settings sheet belong to the machine the chat
// header's drop-down (#chat-machine) shows, like the right sidebar:
//
// * the picker lists that machine's models and shows its default: the page
//   asks for them with `getModels` stamped `machine`, the daemon forwards it
//   there and the reply comes back `forwarded`, stamped `machine`;
// * a pick goes to that machine only (`selectModel` stamped `machine`;
//   unstamped for this machine), so another machine's default never moves;
// * a reply of another machine, or of a machine the picker left, repaints
//   nothing; tabs on the replying machine still showing its old default
//   follow the new one, a tab with its own pick keeps it;
// * an agent's model override belongs to the machine its task runs on;
// * a reconnect asks the picked machine's list again (the daemon's `ready`
//   fan-out brings this daemon's only);
// * the open settings sheet saves its edits and reloads for the machine the
//   drop-down moves to;
// * a task for a registered machine runs with that machine's own web-tools
//   setting, not this daemon's toggle.
//
// Real page: media/chat.html plus media/main.js in jsdom, fed the events the
// daemon emits.

'use strict';

const assert = require('assert');
const {
  makePage,
  send,
  stubRaf,
  pickMachine,
  click,
  byId,
  ofType,
  lastMsg,
} = require('./simplify2_harness');
const {MACHINES} = require('./settings_helpers');
const {test, report} = require('./ui_antipattern_harness').makeRunner();

const LAPTOP_MODELS = [
  {name: 'laptop-a', inp: 1, out: 2, uses: 0, vendor: 'anthropic'},
  {name: 'laptop-b', inp: 3, out: 4, uses: 0, vendor: 'openai'},
];
const GPU_MODELS = [
  {name: 'gpu-a', inp: 0, out: 0, uses: 0, vendor: 'Custom'},
  {name: 'gpu-b', inp: 0, out: 0, uses: 0, vendor: 'Custom'},
  {name: 'gpu-c', inp: 0, out: 0, uses: 0, vendor: 'Custom'},
];

/**
 * A page whose chat header lists this daemon (`laptop`) and `gpu-box`,
 * with this daemon's model list (default `laptop-a`) delivered.
 */
function pickerPage() {
  const page = makePage({
    modelName: 'laptop-a',
    beforeScripts: stubRaf,
    sourceUrl: 'machinescope-main.js',
  });
  send(page.win, {type: 'daemonStatus', connected: true});
  send(page.win, {
    type: 'configData',
    config: {},
    apiKeys: {},
    machine: 'laptop',
  });
  send(page.win, {type: 'machines', machines: MACHINES, checkedAt: 1});
  send(page.win, {
    type: 'models',
    models: LAPTOP_MODELS,
    selected: 'laptop-a',
  });
  return page;
}

/** gpu-box's forwarded `models` reply. */
function gpuModels(win, selected, models = GPU_MODELS) {
  send(win, {
    type: 'models',
    forwarded: true,
    machine: 'gpu-box',
    models: models,
    selected: selected,
  });
}

/** The model the picker's pill shows. */
function label(win) {
  return byId(win, 'model-name').textContent.replace(/\u200e/g, '');
}

/** The models the open picker offers, in order; the picker is closed again. */
function offered(win) {
  click(win, byId(win, 'model-btn'));
  const names = Array.from(
    win.document.querySelectorAll('#model-list .model-item-name'),
  ).map(el => el.textContent.replace(/\u200e/g, ''));
  click(win, byId(win, 'model-btn'));
  return names;
}

/** Open the picker and click *model*. */
function pick(win, model) {
  click(win, byId(win, 'model-btn'));
  const item = Array.from(
    win.document.querySelectorAll('#model-list .model-item'),
  ).find(
    el =>
      el.querySelector('.model-item-name').textContent.replace(/\u200e/g, '') ===
      model,
  );
  assert.ok(item, 'the picker offers ' + model);
  click(win, item);
}

/** The `machine` field of every `type` message posted since index *from*. */
function machinesOf(posted, from, type) {
  return ofType(posted.slice(from), type).map(m =>
    'machine' in m ? m.machine : '(none)',
  );
}

/** Submit *text* and return the posted `submit` message. */
function submit(win, posted, text) {
  byId(win, 'task-input').value = text;
  click(win, byId(win, 'send-btn'));
  const sent = lastMsg(posted, 'submit');
  assert.ok(sent && sent.prompt === text, 'the page posted the submit');
  return sent;
}

test("the picker lists the picked machine's models; a pick changes only that machine", () => {
  const {win, posted} = pickerPage();
  const tabId = win._testApi.getActiveTabId();
  assert.strictEqual(label(win), 'laptop-a');
  assert.deepStrictEqual(offered(win), ['laptop-a', 'laptop-b']);

  // A move to gpu-box asks gpu-box; until it answers the picker shows
  // none of this daemon's models.
  let from = posted.length;
  pickMachine(win, 'gpu-box');
  assert.deepStrictEqual(machinesOf(posted, from, 'getModels'), ['gpu-box']);
  assert.strictEqual(label(win), '');
  assert.deepStrictEqual(offered(win), []);
  gpuModels(win, 'gpu-a');
  assert.strictEqual(label(win), 'gpu-a');
  assert.deepStrictEqual(offered(win), ['gpu-a', 'gpu-b', 'gpu-c']);

  // This daemon's own reply (its default moved on another surface) is
  // noted for later but repaints nothing while the picker shows gpu-box.
  send(win, {type: 'models', models: LAPTOP_MODELS, selected: 'laptop-b'});
  assert.strictEqual(label(win), 'gpu-a');
  assert.deepStrictEqual(offered(win), ['gpu-a', 'gpu-b', 'gpu-c']);

  // A pick is gpu-box's: the daemon forwards it there.
  from = posted.length;
  pick(win, 'gpu-b');
  assert.strictEqual(label(win), 'gpu-b');
  const gpuPick = ofType(posted.slice(from), 'selectModel');
  assert.deepStrictEqual(
    gpuPick.map(m => [m.model, m.tabId, m.machine]),
    [['gpu-b', tabId, 'gpu-box']],
  );

  // Back on this machine: its own list and its default, asked anew.
  from = posted.length;
  pickMachine(win, '');
  assert.deepStrictEqual(machinesOf(posted, from, 'getModels'), ['(none)']);
  assert.strictEqual(label(win), 'laptop-b');
  assert.deepStrictEqual(offered(win), ['laptop-a', 'laptop-b']);
  from = posted.length;
  pick(win, 'laptop-a');
  assert.deepStrictEqual(machinesOf(posted, from, 'selectModel'), ['(none)']);
  assert.strictEqual(lastMsg(posted, 'selectModel').model, 'laptop-a');

  // gpu-box again: the pick made there is what it shows at once.
  from = posted.length;
  pickMachine(win, 'gpu-box');
  assert.strictEqual(label(win), 'gpu-b');
  assert.deepStrictEqual(offered(win), ['gpu-a', 'gpu-b', 'gpu-c']);
  assert.deepStrictEqual(machinesOf(posted, from, 'getModels'), ['gpu-box']);

  // A task submitted now runs on gpu-box with gpu-box's model.
  const sent = submit(win, posted, 'train it');
  assert.strictEqual(sent.machine, 'gpu-box');
  assert.strictEqual(sent.model, 'gpu-b');
  win.close();
});

test('a machine that does not answer is reported; the picker keeps its list', () => {
  const {win} = pickerPage();
  pickMachine(win, 'gpu-box');
  gpuModels(win, 'gpu-a');
  send(win, {
    type: 'models',
    forwarded: true,
    machine: 'gpu-box',
    models: [],
    selected: '',
    error: 'gpu-box (ssh me@gpu) did not answer',
  });
  const notice = win.document.querySelector(
    '.kiss-notification[data-notification-id="model-machine-error"]',
  );
  assert.ok(notice, 'the error is shown as a notification');
  assert.ok(
    /gpu-box: gpu-box \(ssh me@gpu\) did not answer/.test(notice.textContent),
    notice.textContent,
  );
  assert.strictEqual(label(win), 'gpu-a');
  assert.deepStrictEqual(offered(win), ['gpu-a', 'gpu-b', 'gpu-c']);
  win.close();
});

test('an open picker follows a pick made on another surface', () => {
  const {win, posted} = pickerPage();
  pickMachine(win, 'gpu-box');
  gpuModels(win, 'gpu-a');
  pickMachine(win, '');
  click(win, byId(win, 'model-btn'));
  const listed = () =>
    Array.from(win.document.querySelectorAll('#model-list .model-item-name')).map(
      el => el.textContent.replace(/\u200e/g, ''),
    );
  assert.deepStrictEqual(listed(), ['laptop-a', 'laptop-b']);
  const from = posted.length;
  send(win, {type: 'machineChanged', machine: 'gpu-box'});
  assert.strictEqual(byId(win, 'chat-machine').value, 'gpu-box');
  assert.ok(byId(win, 'model-dropdown').classList.contains('open'));
  assert.deepStrictEqual(listed(), ['gpu-a', 'gpu-b', 'gpu-c']);
  assert.strictEqual(label(win), 'gpu-a');
  assert.deepStrictEqual(machinesOf(posted, from, 'getModels'), ['gpu-box']);
  win.close();
});

test("a machine's new default moves only its tabs that still showed the old one", () => {
  // Both machines list a model of the same name.
  const both = {name: 'claude-opus-5', inp: 5, out: 25, uses: 0, vendor: 'anthropic'};
  const laptopList = LAPTOP_MODELS.concat([both]);
  const gpuList = GPU_MODELS.concat([both]);
  const {win, posted} = pickerPage();
  const api = win._testApi;
  // A chat on this machine (it shows a task) whose model is that name.
  const onLaptop = api.getActiveTabId();
  send(win, {type: 'models', models: laptopList, selected: both.name});
  send(win, {
    type: 'task_events',
    events: [],
    task: 'local job',
    tabId: onLaptop,
    chat_id: 'chat-local',
  });
  assert.strictEqual(label(win), both.name);

  // New chats on gpu-box: one keeps gpu-a, one picks the shared name
  // (now gpu-box's default), one more shows that default.
  api.createNewTab();
  const ownModel = api.getActiveTabId();
  pickMachine(win, 'gpu-box');
  gpuModels(win, 'gpu-a', gpuList);
  api.createNewTab();
  const follower = api.getActiveTabId();
  pick(win, both.name);
  api.createNewTab();
  assert.strictEqual(label(win), both.name);

  // Another surface makes gpu-c gpu-box's default.
  gpuModels(win, 'gpu-c', gpuList);
  assert.strictEqual(label(win), 'gpu-c', 'the shown tab followed it');
  api.switchToTab(follower);
  assert.strictEqual(label(win), 'gpu-c', 'so did a hidden one');
  api.switchToTab(ownModel);
  assert.strictEqual(label(win), 'gpu-a', 'a tab on another model kept it');
  const from = posted.length;
  api.switchToTab(onLaptop);
  assert.strictEqual(label(win), both.name, "this machine's chat kept its model");
  assert.deepStrictEqual(offered(win), ['laptop-a', 'laptop-b', both.name]);
  assert.deepStrictEqual(machinesOf(posted, from, 'getModels'), ['(none)']);

  // A reply naming no default moves no tab and keeps the default known.
  api.switchToTab(follower);
  gpuModels(win, '', gpuList.slice(0, 2));
  assert.strictEqual(label(win), 'gpu-c');
  assert.deepStrictEqual(offered(win), ['gpu-a', 'gpu-b']);
  api.createNewTab();
  pickMachine(win, '');
  pickMachine(win, 'gpu-box');
  assert.strictEqual(label(win), 'gpu-c', "a chat moved to gpu-box takes its default");
  win.close();
});

test("an agent's model stays with the machine its task runs on", () => {
  const {win} = pickerPage();
  const tabId = win._testApi.getActiveTabId();
  // The daemon bound the tab to gpu-box (its task runs there).
  send(win, {type: 'tabMachine', tabId: tabId, machine: 'gpu-box'});
  assert.strictEqual(byId(win, 'chat-machine').value, 'gpu-box');
  gpuModels(win, 'gpu-a');
  assert.strictEqual(label(win), 'gpu-a');
  send(win, {type: 'modelPick', tabId: tabId, model: 'gpu-b', source: 'agent'});
  assert.strictEqual(label(win), 'gpu-b');

  // The drop-down moves to this machine (the chat's next task): the
  // picker shows this machine's default, and gpu-box's agent news no
  // longer repaints it.
  pickMachine(win, '');
  assert.strictEqual(label(win), 'laptop-a');
  send(win, {type: 'modelPick', tabId: tabId, model: 'gpu-c', source: 'agent'});
  assert.strictEqual(label(win), 'laptop-a');
  send(win, {type: 'modelPick', tabId: tabId, model: 'gpu-a', source: 'restore'});
  assert.strictEqual(label(win), 'laptop-a');

  // Back on gpu-box the agent's news counts again.
  pickMachine(win, 'gpu-box');
  assert.strictEqual(label(win), 'gpu-a');
  send(win, {type: 'modelPick', tabId: tabId, model: 'gpu-c', source: 'agent'});
  assert.strictEqual(label(win), 'gpu-c');
  win.close();
});

test("a reconnect asks the picked machine's models again", () => {
  const {win, posted} = pickerPage();
  pickMachine(win, 'gpu-box');
  gpuModels(win, 'gpu-a');
  send(win, {type: 'daemonStatus', connected: false, reconnecting: true});
  // Moves while the daemon is away ask nothing (the request would be
  // lost); the picker shows what it last knew of each machine.
  let from = posted.length;
  pickMachine(win, '');
  assert.strictEqual(label(win), 'laptop-a');
  pickMachine(win, 'gpu-box');
  assert.strictEqual(label(win), 'gpu-a');
  assert.deepStrictEqual(machinesOf(posted, from, 'getModels'), []);
  from = posted.length;
  send(win, {type: 'daemonStatus', connected: true});
  assert.deepStrictEqual(machinesOf(posted, from, 'getModels'), ['gpu-box']);

  // On this machine the daemon's `ready` fan-out brings the list.
  pickMachine(win, '');
  send(win, {type: 'daemonStatus', connected: false, reconnecting: true});
  from = posted.length;
  send(win, {type: 'daemonStatus', connected: true});
  assert.deepStrictEqual(machinesOf(posted, from, 'getModels'), []);
  win.close();
});

/** Type *value* into the settings field *id* as the user would. */
function edit(win, id, value) {
  byId(win, id).value = value;
  byId(win, id).dispatchEvent(new win.Event('input', {bubbles: true}));
}

test('the open settings sheet follows the drop-down, saving its edits where they were made', () => {
  const {win, posted} = pickerPage();
  let from = posted.length;
  click(win, byId(win, 'settings-btn'));
  assert.deepStrictEqual(machinesOf(posted, from, 'getConfig'), ['(none)']);
  send(win, {
    type: 'configData',
    config: {max_budget: 3},
    apiKeys: {},
    machine: 'laptop',
  });
  assert.strictEqual(byId(win, 'cfg-max-budget').value, '3');
  edit(win, 'cfg-max-budget', '5');

  // The move saves this daemon's edit, then loads gpu-box's settings.
  from = posted.length;
  pickMachine(win, 'gpu-box');
  let saved = ofType(posted.slice(from), 'saveConfig');
  assert.deepStrictEqual(
    saved.map(m => [m.machine, m.config.max_budget]),
    [[undefined, 5]],
  );
  assert.deepStrictEqual(machinesOf(posted, from, 'getConfig'), ['gpu-box']);
  assert.deepStrictEqual(machinesOf(posted, from, 'getMyModels'), ['gpu-box']);
  assert.strictEqual(byId(win, 'settings-machine').textContent, 'Settings of gpu-box');
  assert.strictEqual(byId(win, 'cfg-max-budget').value, '', "no laptop value on show");
  send(win, {
    type: 'configData',
    forwarded: true,
    machine: 'gpu-box',
    config: {max_budget: 42},
    apiKeys: {},
  });
  assert.strictEqual(byId(win, 'cfg-max-budget').value, '42');

  // A repaint that leaves the machine where it is reloads nothing.
  from = posted.length;
  pickMachine(win, 'gpu-box');
  assert.deepStrictEqual(machinesOf(posted, from, 'getConfig'), []);
  assert.deepStrictEqual(ofType(posted.slice(from), 'saveConfig'), []);

  // Back to this machine: the gpu-box edit goes to gpu-box.
  edit(win, 'cfg-max-budget', '43');
  from = posted.length;
  pickMachine(win, '');
  saved = ofType(posted.slice(from), 'saveConfig');
  assert.deepStrictEqual(
    saved.map(m => [m.machine, m.config.max_budget]),
    [['gpu-box', 43]],
  );
  assert.deepStrictEqual(machinesOf(posted, from, 'getConfig'), ['(none)']);
  assert.strictEqual(byId(win, 'settings-machine').hidden, true);

  // A closed sheet follows nothing.
  click(win, byId(win, 'settings-panel-close'));
  from = posted.length;
  pickMachine(win, 'gpu-box');
  assert.deepStrictEqual(machinesOf(posted, from, 'getConfig'), []);
  win.close();
});

test("a task for a registered machine runs with that machine's web-tools setting", () => {
  const {win, posted} = pickerPage();
  // This daemon's settings came (pickerPage): the toggle is known.
  const local = submit(win, posted, 'here');
  assert.strictEqual(local.machine, '');
  assert.strictEqual(local.useWebTools, true);
  send(win, {type: 'status', running: false});
  win._testApi.createNewTab();
  pickMachine(win, 'gpu-box');
  const remote = submit(win, posted, 'there');
  assert.strictEqual(remote.machine, 'gpu-box');
  assert.ok(!('useWebTools' in remote), 'gpu-box applies its own setting');
  win.close();
});

/** A `models` reply answering the getModels *asked* (echoing its generation). */
function answer(win, asked, selected, models) {
  const reply = {type: 'models', models, selected, generation: asked.generation};
  if (asked.machine) Object.assign(reply, {forwarded: true, machine: asked.machine});
  send(win, reply);
}

test('a list asked before a pick does not undo it', () => {
  const {win, posted} = pickerPage();
  pickMachine(win, 'gpu-box');
  const first = lastMsg(posted, 'getModels');
  answer(win, first, 'gpu-a', GPU_MODELS);
  assert.strictEqual(label(win), 'gpu-a');

  // Away and back: gpu-box is asked again, and the user picks before
  // that reply -- gpu-box's default as it was then -- comes.
  pickMachine(win, '');
  pickMachine(win, 'gpu-box');
  const asked = lastMsg(posted, 'getModels');
  assert.strictEqual(asked.machine, 'gpu-box');
  assert.ok(asked.generation > first.generation, 'every request is numbered');
  pick(win, 'gpu-b');
  answer(win, asked, 'gpu-a', GPU_MODELS.slice(0, 2));
  assert.strictEqual(label(win), 'gpu-b');
  assert.deepStrictEqual(offered(win), ['gpu-a', 'gpu-b'], 'its list still counts');

  // A list asked after the pick is the machine's word again: another
  // surface has made gpu-c the default since.
  pickMachine(win, '');
  pickMachine(win, 'gpu-box');
  assert.strictEqual(label(win), 'gpu-b');
  answer(win, lastMsg(posted, 'getModels'), 'gpu-c', GPU_MODELS);
  assert.strictEqual(label(win), 'gpu-c');

  // This machine too: a pick made while its list is on the way stands.
  pickMachine(win, '');
  const local = lastMsg(posted, 'getModels');
  assert.ok(!('machine' in local));
  pick(win, 'laptop-b');
  answer(win, local, 'laptop-a', LAPTOP_MODELS);
  assert.strictEqual(label(win), 'laptop-b');
  // A list nobody asked for (this daemon's own, after a settings save)
  // is current.
  send(win, {type: 'models', models: LAPTOP_MODELS, selected: 'laptop-a'});
  assert.strictEqual(label(win), 'laptop-a');
  assert.strictEqual(submit(win, posted, 'go').model, 'laptop-a');
  win.close();
});

test("saving a machine's settings asks for its models again", () => {
  const {win, posted} = pickerPage();
  const saveAndAsk = from =>
    posted
      .slice(from)
      .filter(m => m.type === 'saveConfig' || m.type === 'getModels')
      .map(m => m.type + ':' + (m.machine || ''));
  pickMachine(win, 'gpu-box');
  click(win, byId(win, 'settings-btn'));
  send(win, {
    type: 'configData',
    forwarded: true,
    machine: 'gpu-box',
    config: {max_budget: 42},
    apiKeys: {},
  });
  edit(win, 'cfg-max-budget', '43');
  let from = posted.length;
  click(win, byId(win, 'settings-panel-close'));
  assert.deepStrictEqual(saveAndAsk(from), ['saveConfig:gpu-box', 'getModels:gpu-box']);

  // This daemon answers its own save with its list: nothing more.
  pickMachine(win, '');
  click(win, byId(win, 'settings-btn'));
  send(win, {type: 'configData', config: {max_budget: 3}, apiKeys: {}, machine: 'laptop'});
  edit(win, 'cfg-max-budget', '4');
  from = posted.length;
  click(win, byId(win, 'settings-panel-close'));
  assert.deepStrictEqual(saveAndAsk(from), ['saveConfig:']);

  // A move saves the edit to the machine left behind; the picker shows
  // the new machine and asks for that one's list only.
  pickMachine(win, 'gpu-box');
  click(win, byId(win, 'settings-btn'));
  send(win, {
    type: 'configData',
    forwarded: true,
    machine: 'gpu-box',
    config: {max_budget: 43},
    apiKeys: {},
  });
  edit(win, 'cfg-max-budget', '44');
  from = posted.length;
  pickMachine(win, '');
  assert.deepStrictEqual(saveAndAsk(from), ['getModels:', 'saveConfig:gpu-box']);
  win.close();
});

test("a hidden tab bound to a machine shows its agent's model when brought forward", () => {
  const {win} = pickerPage();
  const api = win._testApi;
  const hidden = api.getActiveTabId();
  api.createNewTab();
  // Another surface ran the hidden tab's task on gpu-box; its agent
  // switched models.
  send(win, {type: 'tabMachine', tabId: hidden, machine: 'gpu-box'});
  send(win, {type: 'modelPick', tabId: hidden, model: 'gpu-b', source: 'agent'});
  assert.strictEqual(label(win), 'laptop-a', 'the shown tab is untouched');
  api.switchToTab(hidden);
  assert.strictEqual(byId(win, 'chat-machine').value, 'gpu-box');
  assert.strictEqual(label(win), 'gpu-b');
  gpuModels(win, 'gpu-a');
  assert.strictEqual(label(win), 'gpu-b', 'the override stays on show');
  send(win, {type: 'modelPick', tabId: hidden, model: 'gpu-c', source: 'restore'});
  assert.strictEqual(label(win), 'gpu-c');
  win.close();
});

test("a file tab's settings and picker stay on the machine the file is on", () => {
  const {win, posted} = pickerPage();
  pickMachine(win, 'gpu-box');
  gpuModels(win, 'gpu-a');
  const from = posted.length;
  send(win, {
    type: 'fileContent',
    path: '/gpu/notes.txt',
    name: 'notes.txt',
    content: 'hi',
    machine: 'gpu-box',
  });
  const file = win._testApi.openTabs().find(t => t.isContentTab);
  assert.strictEqual(file.ownerBrowseMachine, 'gpu-box');
  win._testApi.switchToTab(file.id);
  assert.strictEqual(win._testApi.getActiveTabId(), file.id);
  assert.deepStrictEqual(machinesOf(posted, from, 'getModels'), [], 'no needless ask');
  click(win, byId(win, 'settings-btn'));
  assert.deepStrictEqual(machinesOf(posted, from, 'getConfig'), ['gpu-box']);
  assert.strictEqual(byId(win, 'settings-machine').textContent, 'Settings of gpu-box');
  win.close();
});

report('machineScopedPicker');
process.exit(0);
