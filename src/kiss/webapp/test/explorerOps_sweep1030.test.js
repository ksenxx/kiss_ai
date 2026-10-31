// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end (jsdom) tests for the Explorer's file operations and the
// working-directory pickers of media/main.js:
//
// * an editor tab follows its file when Cut + Paste moves it;
// * Rename... of a folder renames it in its parent, not into itself;
// * Cut and Set as Working Directory act on the machine the menu was
//   opened on, even when the tree turned to another machine meanwhile;
// * the folder dialog refuses a typed path the daemon spells as a
//   file-system root ("/tmp/.." is "/");
// * repaints before a machine's directories arrive ask it only once.

'use strict';

const assert = require('assert');
const {
  makePage: buildPage,
  send,
  click,
  ofType,
  sleep,
  pickMachine,
  stubRaf,
  fakeMatchMedia,
} = require('./simplify2_harness');
const {
  WD,
  makePage,
  makeRunner,
  byId,
  key,
  menuItem,
  rightClick,
  runMenuItem,
  openExplorer,
  explorerRow,
  showSection,
} = require('./ui_antipattern_harness');

const {test, report} = makeRunner();

const MACHINES = [
  {
    name: 'laptop',
    local: true,
    status: 'reachable',
    address: 'https://10.0.0.5:8787',
  },
  {name: 'gpu-box', status: 'reachable', address: 'https://gpu:8787'},
];
// gpu-box's workspace has the same path as this machine's, so a path
// cut here names an entry there too.
const BOX_DIRS = {
  type: 'machineWorkDirs',
  machine: 'gpu-box',
  workDir: WD,
  recentWorkDirs: [{path: WD, ts: 1}],
  error: '',
};

/** Answer the last listDir for *p* on *machine* ('' for this one). */
function answerListing(win, posted, p, machine, entries) {
  const req = ofType(posted, 'listDir')
    .filter(m => m.path === p && (m.machine || '') === machine)
    .pop();
  assert.ok(req, 'a listDir for ' + p + ' was sent');
  const reply = {
    type: 'dirListing',
    token: req.token,
    path: p,
    root: req.workDir,
    entries,
  };
  if (machine) reply.machine = machine;
  send(win, reply);
}

/**
 * A page on this machine's WD (with /extra added to the Explorer
 * earlier) whose daemon also reaches gpu-box.
 */
function machinePage() {
  const page = buildPage({
    bodyClass: 'remote-chat',
    extraScripts: ['contentContextMenu.js', 'treeContextMenu.js'],
    beforeScripts(w) {
      stubRaf(w);
      fakeMatchMedia(w, true);
      w.localStorage.setItem('kiss-explorer-roots', '["/extra"]');
    },
  });
  const {win, posted} = page;
  send(win, {type: 'configData', config: {work_dir: WD}, machine: 'laptop'});
  send(win, {type: 'machines', machines: MACHINES, checkedAt: 1});
  showSection(win, 'meta-explorer');
  answerListing(win, posted, WD, '', [
    {name: 'foo.py', path: WD + '/foo.py', isDir: false},
    {name: 'sub', path: WD + '/sub', isDir: true},
  ]);
  return page;
}

/** Turn the active chat (and so the Explorer) to gpu-box. */
function bindToBox(win) {
  send(win, {
    type: 'tabMachine',
    tabId: win._testApi.getActiveTabId(),
    machine: 'gpu-box',
  });
  send(win, BOX_DIRS);
}

async function main() {
  await test('an editor tab follows its file through Cut + Paste', async () => {
    const {win, posted} = makePage();
    openExplorer(win, posted, [
      {name: 'a.txt', path: WD + '/a.txt', isDir: false},
      {name: 'dir', path: WD + '/dir', isDir: true},
    ]);
    send(win, {
      type: 'fileContent',
      tabId: win._testApi.getActiveTabId(),
      path: WD + '/a.txt',
      name: 'a.txt',
      content: 'text',
      version: 'v1',
    });
    await sleep(0);
    runMenuItem(win, explorerRow(win, WD + '/a.txt'), 'Cut');
    runMenuItem(win, explorerRow(win, WD + '/dir'), 'Paste');
    const move = ofType(posted, 'fsAction').pop();
    assert.strictEqual(move.action, 'move');
    send(win, {
      type: 'fsResult',
      token: move.token,
      ok: true,
      path: WD + '/dir/a.txt',
    });
    const tab = win._testApi.openTabs().find(t => t.isContentTab);
    assert.strictEqual(tab.contentPath, WD + '/dir/a.txt');
    assert.strictEqual(tab.title, 'a.txt');
    win.close();
  });

  await test('Rename... of a folder keeps it in its parent', async () => {
    const {win, posted} = makePage();
    openExplorer(win, posted, [{name: 'src', path: WD + '/src', isDir: true}]);
    const row = explorerRow(win, WD + '/src');
    runMenuItem(win, row, 'Rename...');
    await sleep(5);
    const input = row.querySelector('input.explorer-input');
    input.value = 'lib';
    key(win, input, 'Enter');
    const [rename] = ofType(posted, 'fsAction');
    assert.strictEqual(rename.action, 'rename');
    assert.strictEqual(rename.path, WD + '/src');
    assert.strictEqual(rename.dest, WD + '/lib');
    win.close();
  });

  await test('Cut keeps the machine its menu was opened on', () => {
    const {win, posted} = machinePage();
    rightClick(win, explorerRow(win, WD + '/foo.py'));
    bindToBox(win);
    click(win, menuItem(win, 'Cut'));
    answerListing(win, posted, WD, 'gpu-box', [
      {name: 'foo.py', path: WD + '/foo.py', isDir: false},
      {name: 'pkg', path: WD + '/pkg', isDir: true},
    ]);
    const before = ofType(posted, 'fsAction').length;
    runMenuItem(win, explorerRow(win, WD + '/pkg'), 'Paste');
    assert.strictEqual(
      ofType(posted, 'fsAction').length,
      before,
      "this machine's file is not moved on gpu-box",
    );
    win.close();
  });

  await test("Set as Working Directory keeps the menu's machine", () => {
    const {win, posted} = machinePage();
    rightClick(win, explorerRow(win, '/extra'));
    bindToBox(win);
    click(win, menuItem(win, 'Set as Working Directory'));
    const check = ofType(posted, 'listDir')
      .filter(m => String(m.token).startsWith('workdir:'))
      .pop();
    assert.strictEqual(check.path, '/extra');
    assert.strictEqual(check.machine, undefined, 'checked on this machine');
    win.close();
  });

  await test('the folder dialog refuses a path the daemon spells as a root', () => {
    const {win, posted} = makePage();
    openExplorer(win, posted, []);
    click(win, byId(win, 'explorer-pick-folder'));
    const picker = byId(win, 'folder-picker');
    answerListing(win, posted, WD, '', []);
    picker.querySelector('.folder-picker-input').value = '/tmp/..';
    click(win, picker.querySelector('.folder-picker-select'));
    const req = ofType(posted, 'listDir').pop();
    assert.strictEqual(req.path, '/tmp/..');
    send(win, {
      type: 'dirListing',
      token: req.token,
      path: '/',
      root: '/',
      entries: [],
    });
    assert.strictEqual(ofType(posted, 'setWorkDir').length, 0);
    assert.ok(!picker.hidden, 'the dialog stays open');
    assert.ok(
      picker
        .querySelector('.folder-picker-note')
        .textContent.includes('file-system root'),
    );
    assert.ok(explorerRow(win, WD), 'the Explorer keeps the workspace');
    win.close();
  });

  await test('repaints before the reply ask a machine for its directories once', () => {
    const {win, posted} = makePage();
    send(win, {type: 'configData', config: {work_dir: WD}, machine: 'laptop'});
    send(win, {type: 'machines', machines: MACHINES, checkedAt: 1});
    const asks = () => ofType(posted, 'getMachineWorkDirs').length;
    pickMachine(win, 'gpu-box');
    assert.strictEqual(asks(), 1);
    for (let i = 0; i < 3; i++) send(win, {type: 'status', running: false});
    assert.strictEqual(asks(), 1, 'the repaints ask nothing more');
    send(win, BOX_DIRS);
    send(win, {type: 'status', running: false});
    assert.strictEqual(asks(), 1);
    // An outage swallows a reply: the reconnected page asks again.
    pickMachine(win, '');
    send(win, {type: 'daemonStatus', connected: false});
    send(win, {type: 'daemonStatus', connected: true});
    send(win, {type: 'machines', machines: MACHINES, checkedAt: 2});
    const sent = asks();
    pickMachine(win, 'gpu-box');
    assert.ok(asks() > sent, 'asked again after the outage');
    win.close();
  });

  report('explorerOps_sweep1030');
}

main();
