// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end (JSDOM) tests: the chat header's working directory
// drop-down follows the MACHINE the active tab runs its tasks on.
//
// * a tab bound to a registered machine lists THAT machine's working
//   directory and the directories selected there so far (the daemon's
//   machineWorkDirs reply to getMachineWorkDirs), not this daemon's;
//   this machine's list is back once the tab is brought home;
// * a pick of one of the machine's directories is checked by a listDir
//   carrying `machine` and then sent as setWorkDir with `machine`: this
//   client's own working directory is untouched and the drop-down
//   follows the daemon's `set` reply (or reports its error);
// * "Choose ..." for a machine opens the in-page folder picker on the
//   machine's folders; the pick goes to the machine;
// * the daemon's tabMachine binding (a chat of the machine opened from
//   CHATS) asks for the machine's directories and repaints;
// * reaching for the drop-down asks the machine's list again;
// * a task's end asks for the machines again, and a `machines` reply
//   naming a new one unfolds the Machines subpanel (and opens the
//   task-info drawer on the phone layout).

/* global require, __dirname, console, process */

'use strict';

const assert = require('assert');
const {
  makePage,
  closeWindows,
  send,
  byId,
  msgs,
  assertJsonEqual,
  CHOOSE,
  optionLabels,
  currentDir,
  pick,
  errorNotifications,
} = require('./workdir_helpers');

function pickMachine(win, value) {
  const select = byId(win, 'chat-machine');
  select.value = value;
  select.dispatchEvent(new win.Event('change', {bubbles: true}));
}

const NOW = Date.now() / 1000;
const LOCAL_RECENTS = [
  {path: '/work/ws', ts: NOW},
  {path: '/home/u/older', ts: NOW - 100},
];
const MACHINES = [
  {
    name: 'laptop',
    local: true,
    status: 'reachable',
    address: 'https://10.0.0.5:8787',
  },
  {
    name: 'gpu-box',
    label: 'gpu.lab',
    status: 'reachable',
    address: 'https://gpu:8787',
  },
];
const BOX_DIRS = {
  type: 'machineWorkDirs',
  machine: 'gpu-box',
  workDir: '/srv/proj',
  recentWorkDirs: [
    {path: '/srv/other', ts: NOW - 50},
    {path: '/srv/proj', ts: NOW},
  ],
  error: '',
};

/** A page whose daemon knows its own directories and the machines. */
function boot() {
  const made = makePage();
  send(made.win, {
    type: 'configData',
    config: {work_dir: '/work/ws', recent_work_dirs: LOCAL_RECENTS},
    machine: 'laptop',
  });
  send(made.win, {type: 'machines', machines: MACHINES, checkedAt: 1});
  made.posted.length = 0;
  return made;
}

function testAMachinePickListsTheMachinesDirectories() {
  const {win, posted} = boot();
  assert.deepStrictEqual(optionLabels(win), [
    '/work/ws',
    '/home/u/older',
    'Choose ...',
  ]);
  pickMachine(win, 'gpu-box');
  assertJsonEqual(msgs(posted, 'getMachineWorkDirs'), [
    {machine: 'gpu-box', type: 'getMachineWorkDirs'},
  ]);
  assert.deepStrictEqual(
    optionLabels(win),
    ['No working directory', 'Choose ...'],
    "nothing of the machine's is known until its reply",
  );
  assert.strictEqual(currentDir(win), '');
  send(win, BOX_DIRS);
  assert.deepStrictEqual(
    optionLabels(win),
    ['/srv/proj', '/srv/other', 'Choose ...'],
    "the machine's current directory first, then its recents",
  );
  assert.strictEqual(currentDir(win), '/srv/proj');
  assert.ok(
    byId(win, 'chat-workdir').title.includes('/srv/proj on gpu.lab'),
    'the title names the machine by its label',
  );
  // Home again: this daemon's own list, nothing asked of the daemon.
  posted.length = 0;
  pickMachine(win, '');
  assert.deepStrictEqual(optionLabels(win), [
    '/work/ws',
    '/home/u/older',
    'Choose ...',
  ]);
  assert.strictEqual(currentDir(win), '/work/ws');
  assert.strictEqual(msgs(posted, 'getMachineWorkDirs').length, 0);
  assert.strictEqual(msgs(posted, 'getConfig').length, 0);
  // Back on the machine: the kept list shows at once, asked again.
  pickMachine(win, 'gpu-box');
  assert.deepStrictEqual(optionLabels(win), [
    '/srv/proj',
    '/srv/other',
    'Choose ...',
  ]);
  assert.strictEqual(msgs(posted, 'getMachineWorkDirs').length, 1);
  // A reply for a machine that is not the active tab's is kept quietly.
  send(win, {
    ...BOX_DIRS,
    machine: 'office',
    workDir: '/o',
    recentWorkDirs: [],
  });
  assert.strictEqual(currentDir(win), '/srv/proj');
  // A read that failed leaves the list as it was.
  send(win, {
    ...BOX_DIRS,
    workDir: '',
    recentWorkDirs: [],
    error: 'gpu-box: down',
  });
  assert.strictEqual(currentDir(win), '/srv/proj');
  assert.deepStrictEqual(errorNotifications(win), []);
}

function testAPickOnTheMachineGoesToTheMachine() {
  const {win, posted} = boot();
  pickMachine(win, 'gpu-box');
  send(win, BOX_DIRS);
  posted.length = 0;
  pick(win, '/srv/other');
  const listDirs = msgs(posted, 'listDir');
  assert.strictEqual(listDirs.length, 1, 'the pick is checked on the machine');
  assert.strictEqual(listDirs[0].machine, 'gpu-box');
  assert.strictEqual(listDirs[0].path, '/srv/other');
  assert.strictEqual(listDirs[0].token, 'workdir:1');
  assert.strictEqual(msgs(posted, 'openWorkDir').length, 0);
  assert.strictEqual(
    currentDir(win),
    '/srv/proj',
    'springs back until the daemon answers',
  );
  send(win, {
    type: 'dirListing',
    token: 'workdir:1',
    path: '/srv/other',
    machine: 'gpu-box',
    entries: [],
  });
  assertJsonEqual(msgs(posted, 'setWorkDir'), [
    {workDir: '/srv/other', machine: 'gpu-box', type: 'setWorkDir'},
  ]);
  assert.strictEqual(
    currentDir(win),
    '/srv/proj',
    "the drop-down follows the machine's answer, not the wish",
  );
  send(win, {
    ...BOX_DIRS,
    workDir: '/srv/other',
    recentWorkDirs: [
      {path: '/srv/other', ts: NOW + 1},
      {path: '/srv/proj', ts: NOW},
    ],
    set: true,
  });
  assert.deepStrictEqual(optionLabels(win), [
    '/srv/other',
    '/srv/proj',
    'Choose ...',
  ]);
  assert.strictEqual(currentDir(win), '/srv/other');
  // This daemon's own directory and list are untouched.
  pickMachine(win, '');
  assert.deepStrictEqual(optionLabels(win), [
    '/work/ws',
    '/home/u/older',
    'Choose ...',
  ]);
  assert.strictEqual(currentDir(win), '/work/ws');
  // A refused pick is reported; a refused check too.
  pickMachine(win, 'gpu-box');
  posted.length = 0;
  pick(win, '/srv/proj');
  let token = msgs(posted, 'listDir')[0].token;
  send(win, {
    type: 'dirListing',
    token: token,
    path: '/srv/proj',
    machine: 'gpu-box',
  });
  assert.strictEqual(msgs(posted, 'setWorkDir').length, 1);
  send(win, {
    ...BOX_DIRS,
    workDir: '',
    recentWorkDirs: [],
    set: true,
    error: 'gpu-box: down',
  });
  assert.deepStrictEqual(
    errorNotifications(win).map(t => t.includes('gpu-box: down')),
    [true],
  );
  assert.strictEqual(currentDir(win), '/srv/other');
  pick(win, '/srv/proj');
  token = msgs(posted, 'listDir')[1].token;
  send(win, {
    type: 'dirListing',
    token: token,
    machine: 'gpu-box',
    error: 'Not a directory',
  });
  assert.strictEqual(
    msgs(posted, 'setWorkDir').length,
    1,
    'a refused check sends nothing',
  );
  assert.strictEqual(errorNotifications(win).length, 2);
}

function testChooseOnTheMachineBrowsesItsFolders() {
  const {win, posted} = boot();
  pickMachine(win, 'gpu-box');
  send(win, BOX_DIRS);
  posted.length = 0;
  pick(win, CHOOSE);
  const picker = byId(win, 'folder-picker');
  assert.ok(!picker.hidden, 'the in-page picker opened');
  assert.strictEqual(
    byId(win, 'folder-picker-title').textContent,
    'Open Folder as Working Directory on gpu.lab',
  );
  let listDirs = msgs(posted, 'listDir');
  assert.strictEqual(listDirs.length, 1);
  assert.strictEqual(listDirs[0].machine, 'gpu-box');
  assert.strictEqual(
    listDirs[0].path,
    '/srv/proj',
    "starts at the machine's working directory",
  );
  assert.strictEqual(listDirs[0].token, 'picker:1');
  send(win, {
    type: 'dirListing',
    token: 'picker:1',
    path: '/srv/proj',
    machine: 'gpu-box',
    entries: [{name: 'sub', path: '/srv/proj/sub', isDir: true}],
  });
  const item = picker.querySelector(
    '.folder-picker-item[data-path="/srv/proj/sub"]',
  );
  assert.ok(item, "the machine's subfolder is listed");
  item.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  picker
    .querySelector('.folder-picker-select')
    .dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  // A listed folder needs no second check: the pick goes to the machine.
  assertJsonEqual(msgs(posted, 'setWorkDir'), [
    {workDir: '/srv/proj/sub', machine: 'gpu-box', type: 'setWorkDir'},
  ]);
  assert.ok(picker.hidden, 'the picker closed on the pick');
  // A typed folder is listed on the machine first, then picked.
  pick(win, CHOOSE);
  const input = picker.querySelector('.folder-picker-input');
  input.value = '/srv/typed';
  picker
    .querySelector('.folder-picker-select')
    .dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  listDirs = msgs(posted, 'listDir');
  const check = listDirs[listDirs.length - 1];
  assert.strictEqual(check.machine, 'gpu-box');
  assert.strictEqual(check.path, '/srv/typed');
  send(win, {
    type: 'dirListing',
    token: check.token,
    path: '/srv/typed',
    machine: 'gpu-box',
    entries: [],
  });
  assert.strictEqual(msgs(posted, 'setWorkDir').length, 2);
  assert.strictEqual(msgs(posted, 'setWorkDir')[1].workDir, '/srv/typed');
  assert.strictEqual(msgs(posted, 'setWorkDir')[1].machine, 'gpu-box');
  assert.ok(picker.hidden);
  // This machine's "Choose ..." is as before: the in-page picker on
  // this daemon's folders.
  pickMachine(win, '');
  posted.length = 0;
  pick(win, CHOOSE);
  assert.ok(!picker.hidden);
  assert.strictEqual(
    byId(win, 'folder-picker-title').textContent,
    'Open Folder as Working Directory',
  );
  assert.strictEqual(msgs(posted, 'listDir')[0].machine, undefined);
}

function testTheDaemonsBindingAsksTheMachine() {
  const {win, posted} = boot();
  const api = win._testApi;
  const tab = api.getActiveTabId();
  // A chat of the machine opened from CHATS: the daemon binds the tab.
  send(win, {type: 'tabMachine', tabId: tab, machine: 'gpu-box'});
  assert.strictEqual(byId(win, 'chat-machine').value, 'gpu-box');
  assertJsonEqual(msgs(posted, 'getMachineWorkDirs'), [
    {machine: 'gpu-box', type: 'getMachineWorkDirs'},
  ]);
  assert.deepStrictEqual(optionLabels(win), [
    'No working directory',
    'Choose ...',
  ]);
  send(win, BOX_DIRS);
  assert.strictEqual(currentDir(win), '/srv/proj');
  // A binding of a background tab asks nothing; the list is at hand
  // when the tab comes on screen.
  api.createNewTab();
  posted.length = 0;
  send(win, {type: 'tabMachine', tabId: tab, machine: 'gpu-box'});
  assert.strictEqual(msgs(posted, 'getMachineWorkDirs').length, 0);
  assert.strictEqual(currentDir(win), '/work/ws');
  api.switchToTab(tab);
  assert.strictEqual(currentDir(win), '/srv/proj');
  assert.strictEqual(msgs(posted, 'getMachineWorkDirs').length, 0);
  // Brought home by the daemon: this daemon's list.
  send(win, {type: 'tabMachine', tabId: tab, machine: ''});
  assert.strictEqual(currentDir(win), '/work/ws');
}

function testReachingForTheDropDownAsksTheMachine() {
  const {win, posted} = boot();
  byId(win, 'chat-workdir').dispatchEvent(new win.Event('focus'));
  assert.deepStrictEqual(
    posted.map(m => m.type),
    ['getConfig'],
  );
  pickMachine(win, 'gpu-box');
  send(win, BOX_DIRS);
  posted.length = 0;
  byId(win, 'chat-workdir').dispatchEvent(new win.Event('focus'));
  assert.deepStrictEqual(
    posted.map(m => m.type),
    ['getMachineWorkDirs'],
  );
}

function testANewMachineUnfoldsTheMachinesSubpanel() {
  const {win, posted} = boot();
  const section = byId(win, 'meta-machines');
  const toggle = section.querySelector('.meta-section-toggle');
  if (!section.classList.contains('collapsed'))
    toggle.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.ok(section.classList.contains('collapsed'), 'folded');
  const metaPanel = byId(win, 'meta-panel');
  metaPanel.classList.remove('open');
  // The same machines again: nothing moves.
  send(win, {type: 'machines', machines: MACHINES, checkedAt: 2});
  assert.ok(section.classList.contains('collapsed'));
  // A task ended: the daemon is asked for the machines (a `/machine
  // add` task may have registered one).
  posted.length = 0;
  send(win, {
    type: 'task_done',
    tabId: win._testApi.getActiveTabId(),
    success: true,
  });
  assert.deepStrictEqual(msgs(posted, 'getMachines').length, 1);
  send(win, {
    type: 'machines',
    machines: MACHINES.concat([
      {name: 'office', status: 'reachable', address: 'https://o:8787'},
    ]),
    checkedAt: 3,
  });
  assert.ok(
    !section.classList.contains('collapsed'),
    'unfolded for the new machine',
  );
  assert.ok(section.querySelector('.machine-row[data-machine="office"]'));
  // The phone layout (not desktop) shows the task-info panel as a
  // drawer: it opens.
  assert.strictEqual(metaPanel.classList.contains('open'), true);
  // Folded again by the user, a reply without news leaves it folded.
  toggle.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  send(win, {
    type: 'machines',
    machines: MACHINES.concat([
      {name: 'office', status: 'reachable', address: 'https://o:8787'},
    ]),
    checkedAt: 4,
  });
  assert.ok(section.classList.contains('collapsed'));
}

function testAMachineDirectoryNobodySelectedIsNotListed() {
  const {win} = boot();
  pickMachine(win, 'gpu-box');
  // The machine's daemon runs in /opt/boot (its start-up folder), but
  // only /srv/other was ever selected there: the drop-down offers the
  // selected folders and "Choose ...", nothing else, and the
  // placeholder stays selected while the title still says where the
  // next task runs.
  send(win, {
    ...BOX_DIRS,
    workDir: '/opt/boot',
    recentWorkDirs: [{path: '/srv/other', ts: NOW - 50}],
  });
  assert.deepStrictEqual(optionLabels(win), [
    'No working directory',
    '/srv/other',
    'Choose ...',
  ]);
  assert.strictEqual(currentDir(win), '');
  assert.ok(byId(win, 'chat-workdir').title.includes('/opt/boot on gpu.lab'));
  // A machine with no selected folder at all: just "Choose ...".
  send(win, {...BOX_DIRS, workDir: '/opt/boot', recentWorkDirs: []});
  assert.deepStrictEqual(optionLabels(win), [
    'No working directory',
    'Choose ...',
  ]);
  // This daemon's own drop-down still offers its current directory
  // even when it was never selected.
  pickMachine(win, '');
  send(win, {
    type: 'configData',
    config: {work_dir: '/work/boot', recent_work_dirs: LOCAL_RECENTS},
    machine: 'laptop',
  });
  assert.strictEqual(optionLabels(win)[0], '/work/boot');
  assert.strictEqual(currentDir(win), '/work/boot');
}

const tests = [
  [
    "a machine pick lists the machine's directories",
    testAMachinePickListsTheMachinesDirectories,
  ],
  [
    'a machine directory nobody selected is not listed',
    testAMachineDirectoryNobodySelectedIsNotListed,
  ],
  [
    'a pick on the machine goes to the machine',
    testAPickOnTheMachineGoesToTheMachine,
  ],
  [
    '"Choose ..." on the machine browses its folders',
    testChooseOnTheMachineBrowsesItsFolders,
  ],
  [
    "the daemon's binding asks the machine",
    testTheDaemonsBindingAsksTheMachine,
  ],
  [
    'reaching for the drop-down asks the machine',
    testReachingForTheDropDownAsksTheMachine,
  ],
  [
    'a new machine unfolds the Machines subpanel',
    testANewMachineUnfoldsTheMachinesSubpanel,
  ],
];

let failed = 0;
for (const [name, fn] of tests) {
  try {
    fn();
    console.log('ok - ' + name);
  } catch (e) {
    failed += 1;
    console.log('not ok - ' + name);
    console.log(e && e.stack ? e.stack : String(e));
  } finally {
    closeWindows();
  }
}
console.log(`\n${tests.length - failed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
