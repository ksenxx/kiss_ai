// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end (JSDOM) tests, remote webapp (main.js audit W2, 2026-10-19):
//
// * W2-1: the Schedule / Machines in-place forms are rebuilt on every
//   `cronJobs` / `machines` reply (the 30 s poll, another row's action).
//   The rebuild must not steal the keyboard: a field the user is typing
//   in keeps focus and its caret, and focus elsewhere (the chat
//   composer) stays there.  Only the user's click on Edit / `+` focuses
//   the form's first field.
// * W2-2: an fsResult rename / delete follows only the content tabs of
//   the machine the action ran on (the same path on two machines is two
//   files), and a rename repaints the editor's save-bar path.
// * W2-7: "Compare with Selected" follows a rename of the selected file,
//   is withdrawn when the file is deleted, and is not offered on another
//   machine's tree.

/* global require, console, process */

'use strict';

const assert = require('assert');
const path = require('path');
const {
  makePage: makeChatPage,
  send,
  byId,
  waitFor,
  stubRaf,
  fakeMatchMedia,
  stubMonaco,
} = require('./simplify2_harness');

const openWindows = [];

function makePage() {
  let editors;
  const {win, posted} = makeChatPage({
    bodyClass: 'remote-chat',
    extraScripts: ['treeContextMenu.js'],
    beforeScripts(win) {
      openWindows.push(win);
      stubRaf(win);
      // The polls are recorded, not run: a test delivers replies by hand.
      win.setInterval = function () {
        return 1;
      };
      win.clearInterval = function () {};
      fakeMatchMedia(win, true);
      editors = stubMonaco(win);
    },
  });
  return {win, posted, editors};
}

function click(win, el) {
  assert.ok(el, 'the element to click exists');
  el.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

function msgs(posted, type) {
  return posted.filter(m => m && m.type === type);
}

function all(win, sel) {
  return Array.from(win.document.querySelectorAll(sel));
}

function showSection(win, id) {
  const section = byId(win, id);
  if (section.classList.contains('collapsed'))
    click(win, section.querySelector('.meta-section-toggle'));
}

function pickMachine(win, value) {
  const select = byId(win, 'chat-machine');
  select.value = value;
  select.dispatchEvent(new win.Event('change', {bubbles: true}));
}

/** Answer the last listDir for *p* (stamped *machine*, '' for none). */
function answerListing(win, posted, p, machine, entries) {
  const reqs = msgs(posted, 'listDir').filter(
    m => m.path === p && (m.machine || '') === machine,
  );
  const req = reqs[reqs.length - 1];
  assert.ok(
    req,
    `a listDir for ${p} on '${machine || 'this machine'}' was sent`,
  );
  const reply = {
    type: 'dirListing',
    token: req.token,
    path: p,
    root: req.workDir,
    entries,
  };
  if (machine) reply.machine = machine;
  send(win, reply);
  return req;
}

function rightClick(win, el) {
  assert.ok(el, 'the row to right-click exists');
  el.dispatchEvent(
    new win.MouseEvent('contextmenu', {
      bubbles: true,
      cancelable: true,
      clientX: 10,
      clientY: 10,
    }),
  );
}

function menuItem(win, label) {
  return all(win, '#sidebar-context-menu .tree-ctx-item').find(
    el => el.querySelector('.tree-ctx-label').textContent === label,
  );
}

function closeMenu(win) {
  win.document.dispatchEvent(
    new win.KeyboardEvent('keydown', {key: 'Escape', bubbles: true}),
  );
}

function rowFor(win, p) {
  return all(win, '.explorer-row[data-explorer-path]').find(
    r => r.dataset.explorerPath === p,
  );
}

/** Rename *p* through the Explorer's inline box; returns the fsAction sent. */
function renameInExplorer(win, posted, p, newName) {
  rightClick(win, rowFor(win, p));
  click(win, menuItem(win, 'Rename...'));
  const input = win.document.querySelector('input.explorer-input');
  assert.ok(input, 'the rename box');
  input.value = newName;
  posted.length = 0;
  input.dispatchEvent(
    new win.KeyboardEvent('keydown', {
      key: 'Enter',
      bubbles: true,
      cancelable: true,
    }),
  );
  const sent = msgs(posted, 'fsAction');
  assert.strictEqual(sent.length, 1, 'one rename sent');
  assert.strictEqual(sent[0].action, 'rename');
  return sent[0];
}

/** Delete *p* through the Explorer's menu and its toast; returns the fsAction sent. */
function deleteInExplorer(win, posted, p) {
  rightClick(win, rowFor(win, p));
  click(win, menuItem(win, 'Delete'));
  posted.length = 0;
  const toast = win.document.querySelector(
    '[data-notification-id="fs-delete"]',
  );
  assert.ok(toast, 'the delete confirmation');
  click(
    win,
    Array.from(toast.querySelectorAll('.kiss-notification-action')).find(
      b => b.textContent.trim() === 'Delete',
    ),
  );
  const sent = msgs(posted, 'fsAction');
  assert.strictEqual(sent.length, 1, 'one delete sent');
  assert.strictEqual(sent[0].action, 'delete');
  return sent[0];
}

const NOW = Date.now() / 1000;
const MACHINES = [
  {
    name: 'laptop',
    local: true,
    status: 'reachable',
    address: 'https://10.0.0.5:8787',
    ip: '10.0.0.5',
    port: 8787,
  },
  {
    name: 'gpu-box',
    label: 'gpu.lab',
    status: 'reachable',
    address: 'https://gpu:8787',
    ip: '10.0.0.9',
    port: 8787,
  },
];
const BOX_DIRS = {
  type: 'machineWorkDirs',
  machine: 'gpu-box',
  workDir: '/srv/proj',
  recentWorkDirs: [{path: '/srv/proj', ts: NOW}],
  error: '',
};
const HOME_FILES = [
  {name: 'a.py', path: '/work/ws/a.py', isDir: false},
  {name: 'b.py', path: '/work/ws/b.py', isDir: false},
];
const BOX_FILES = [
  {name: 'a.py', path: '/srv/proj/a.py', isDir: false},
  {name: 'b.py', path: '/srv/proj/b.py', isDir: false},
];

function job(fields) {
  return Object.assign(
    {
      id: 'a',
      name: 'Nightly',
      schedule: 'every 1h',
      kind: 'prompt',
      what: 'summarize',
      enabled: true,
      running: false,
      nextRunAt: '',
      lastRunAt: '',
      lastStatus: '',
    },
    fields,
  );
}

/** A page on this daemon's /work/ws with the Explorer expanded and loaded. */
function boot() {
  const made = makePage();
  const {win, posted} = made;
  send(win, {
    type: 'configData',
    config: {
      work_dir: '/work/ws',
      recent_work_dirs: [{path: '/work/ws', ts: NOW}],
    },
    machine: 'laptop',
  });
  send(win, {type: 'machines', machines: MACHINES, checkedAt: 1});
  showSection(win, 'meta-explorer');
  answerListing(win, posted, '/work/ws', '', HOME_FILES);
  posted.length = 0;
  return made;
}

/** Bind the active chat to gpu-box and list its folder. */
function goToBox(win, posted) {
  pickMachine(win, 'gpu-box');
  send(win, BOX_DIRS);
  answerListing(win, posted, '/srv/proj', 'gpu-box', BOX_FILES);
  posted.length = 0;
}

function focusWithCaret(win, field, at) {
  field.focus();
  field.setSelectionRange(at, at);
  assert.strictEqual(win.document.activeElement, field);
}

function assertCaret(win, field, at, why) {
  assert.strictEqual(win.document.activeElement, field, why + ': focus');
  assert.strictEqual(field.selectionStart, at, why + ': selectionStart');
  assert.strictEqual(field.selectionEnd, at, why + ': selectionEnd');
}

function testScheduleEditorKeepsTheKeyboardAcrossAPoll() {
  const {win} = boot();
  showSection(win, 'meta-schedule');
  const jobs = [job({id: 'a'}), job({id: 'b', name: 'Weekly'})];
  send(win, {type: 'cronJobs', jobs});
  const list = byId(win, 'meta-schedule-list');
  // The user's click on Edit focuses the first field.
  click(win, list.children[0].querySelector('.sidebar-item-edit'));
  let form = list.children[0].querySelector('form');
  assert.strictEqual(
    win.document.activeElement,
    form.elements.name,
    'Edit focuses Name',
  );
  // Typing in Task when the poll lands: the caret stays in Task.
  form.elements.what.value = 'summarize my email';
  focusWithCaret(win, form.elements.what, 2);
  send(win, {type: 'cronJobs', jobs});
  form = list.children[0].querySelector('form');
  assert.ok(form, 'the editor survived the poll');
  assert.strictEqual(
    form.elements.what.value,
    'summarize my email',
    'with the draft',
  );
  assertCaret(win, form.elements.what, 2, 'the poll');
  // Another row's action reply lands the same way.
  focusWithCaret(win, form.elements.schedule, 3);
  send(win, {type: 'cronJobs', jobs, action: 'pause', error: ''});
  form = list.children[0].querySelector('form');
  assertCaret(win, form.elements.schedule, 3, "another row's action");
  // Focus in the chat composer (the editor left open) stays there.
  const composer = byId(win, 'task-input');
  composer.focus();
  send(win, {type: 'cronJobs', jobs});
  assert.ok(list.children[0].querySelector('form'), 'the editor is still open');
  assert.strictEqual(
    win.document.activeElement,
    composer,
    'the composer kept focus',
  );
  // A refused update re-opens the editor; the user was in Task.
  form = list.children[0].querySelector('form');
  focusWithCaret(win, form.elements.what, 5);
  click(win, form.querySelector('.sidebar-inline-form-save'));
  send(win, {type: 'cronJobs', jobs, action: 'update', error: 'never fires'});
  form = list.children[0].querySelector('form');
  assertCaret(win, form.elements.what, 5, 'a refusal');
}

function testScheduleAddFormKeepsTheKeyboardAcrossAPoll() {
  const {win} = boot();
  showSection(win, 'meta-schedule');
  const jobs = [job({id: 'a'})];
  send(win, {type: 'cronJobs', jobs});
  const list = byId(win, 'meta-schedule-list');
  click(win, byId(win, 'meta-schedule-add'));
  let form = list.querySelector('.sched-add-row form');
  assert.strictEqual(
    win.document.activeElement,
    form.elements.name,
    '+ focuses Name',
  );
  form.elements.what.value = 'echo night';
  focusWithCaret(win, form.elements.what, 4);
  send(win, {type: 'cronJobs', jobs});
  form = list.querySelector('.sched-add-row form');
  assert.ok(form, 'the add form survived the poll');
  assertCaret(win, form.elements.what, 4, 'the poll');
  // The kind drop-down keeps focus too (no caret to restore).
  form.elements.kind.focus();
  send(win, {type: 'cronJobs', jobs});
  form = list.querySelector('.sched-add-row form');
  assert.strictEqual(
    win.document.activeElement,
    form.elements.kind,
    'the select kept focus',
  );
  const composer = byId(win, 'task-input');
  composer.focus();
  send(win, {type: 'cronJobs', jobs});
  assert.ok(
    list.querySelector('.sched-add-row form'),
    'the add form is still open',
  );
  assert.strictEqual(
    win.document.activeElement,
    composer,
    'the composer kept focus',
  );
}

function testMachinesAddFormKeepsTheKeyboardAcrossAPoll() {
  const {win} = boot();
  showSection(win, 'meta-machines');
  const list = byId(win, 'meta-machines-list');
  click(win, byId(win, 'meta-machines-add'));
  let form = list.querySelector('.machine-add-row form');
  assert.strictEqual(
    win.document.activeElement,
    form.elements.name,
    '+ focuses Name',
  );
  form.elements.address.value = 'https://lab:8787';
  focusWithCaret(win, form.elements.address, 6);
  send(win, {type: 'machines', machines: MACHINES, checkedAt: 2});
  form = list.querySelector('.machine-add-row form');
  assert.ok(form, 'the add form survived the poll');
  assert.strictEqual(
    form.elements.address.value,
    'https://lab:8787',
    'with the draft',
  );
  assertCaret(win, form.elements.address, 6, 'the poll');
  // A refused add re-opens the form; the user was in Address.
  click(win, form.querySelector('.sidebar-inline-form-save'));
  send(win, {
    type: 'machines',
    machines: MACHINES,
    checkedAt: 3,
    action: 'add',
    error: 'lab: unreachable',
  });
  form = list.querySelector('.machine-add-row form');
  assertCaret(win, form.elements.address, 6, 'a refusal');
  const composer = byId(win, 'task-input');
  composer.focus();
  send(win, {type: 'machines', machines: MACHINES, checkedAt: 4});
  assert.ok(
    list.querySelector('.machine-add-row form'),
    'the add form is still open',
  );
  assert.strictEqual(
    win.document.activeElement,
    composer,
    'the composer kept focus',
  );
}

/** Open the same path as a file of this machine and of gpu-box. */
async function openOnBothMachines(win, posted, editors, p) {
  const chat = win._testApi.getActiveTabId();
  const before = editors.length;
  send(win, {
    type: 'fileContent',
    tabId: chat,
    path: p,
    name: path.basename(p),
    content: 'home\n',
    version: 'h1',
  });
  await waitFor(() => editors.length === before + 1, 'the home file opened');
  send(win, {
    type: 'fileContent',
    tabId: chat,
    machine: 'gpu-box',
    path: p,
    name: path.basename(p),
    content: 'box\n',
    version: 'b1',
  });
  await waitFor(
    () => editors.length === before + 2,
    "the machine's file opened",
  );
  const tabs = win._testApi
    .openTabs()
    .filter(t => t.isContentTab && t.contentPath === p);
  const home = tabs.find(t => t.ownerBrowseMachine === '');
  const box = tabs.find(t => t.ownerBrowseMachine === 'gpu-box');
  assert.ok(home && box, 'one tab per machine');
  return {home, box};
}

function savePathOf(win, tabId) {
  const view = win.document.querySelector(
    '.content-tab-view[data-tab-id="' + tabId + '"] .content-save-path',
  );
  assert.ok(view, 'the save bar of ' + tabId);
  return view.title;
}

async function testRenameFollowsOnlyTheMachinesTabs() {
  const {win, posted, editors} = boot();
  goToBox(win, posted);
  const {home, box} = await openOnBothMachines(
    win,
    posted,
    editors,
    '/srv/proj/a.py',
  );
  // The chat still browses gpu-box: rename a.py there.
  win._testApi.switchToTab(win._testApi.openTabs()[0].id);
  const req = renameInExplorer(win, posted, '/srv/proj/a.py', 'c.py');
  assert.strictEqual(req.machine, 'gpu-box');
  send(win, {
    type: 'fsResult',
    token: req.token,
    action: 'rename',
    path: '/srv/proj/c.py',
    machine: 'gpu-box',
  });
  const after = win._testApi.openTabs();
  const homeAfter = after.find(t => t.id === home.id);
  const boxAfter = after.find(t => t.id === box.id);
  assert.strictEqual(
    boxAfter.contentPath,
    '/srv/proj/c.py',
    "the machine's editor follows",
  );
  assert.strictEqual(boxAfter.title, 'c.py');
  assert.strictEqual(
    homeAfter.contentPath,
    '/srv/proj/a.py',
    "this machine's editor stays",
  );
  assert.strictEqual(homeAfter.title, 'a.py');
  assert.strictEqual(
    savePathOf(win, box.id),
    '/srv/proj/c.py',
    'the save bar repainted',
  );
  assert.strictEqual(savePathOf(win, home.id), '/srv/proj/a.py');
}

async function testDeleteClosesOnlyTheMachinesTabs() {
  const {win, posted, editors} = boot();
  goToBox(win, posted);
  const {home, box} = await openOnBothMachines(
    win,
    posted,
    editors,
    '/srv/proj/a.py',
  );
  win._testApi.switchToTab(win._testApi.openTabs()[0].id);
  const req = deleteInExplorer(win, posted, '/srv/proj/a.py');
  assert.strictEqual(req.machine, 'gpu-box');
  send(win, {
    type: 'fsResult',
    token: req.token,
    action: 'delete',
    path: '/srv/proj/a.py',
    machine: 'gpu-box',
  });
  const ids = win._testApi.openTabs().map(t => t.id);
  assert.ok(!ids.includes(box.id), "the machine's editor closed");
  assert.ok(ids.includes(home.id), "this machine's editor stays open");
}

function testCompareWithSelectedFollowsRenameAndDelete() {
  const {win, posted} = boot();
  rightClick(win, rowFor(win, '/work/ws/a.py'));
  click(win, menuItem(win, 'Select for Compare'));
  rightClick(win, rowFor(win, '/work/ws/b.py'));
  assert.ok(
    menuItem(win, 'Compare with Selected'),
    'offered after a selection',
  );
  closeMenu(win);
  // Renamed: the item compares against the new path.
  let req = renameInExplorer(win, posted, '/work/ws/a.py', 'c.py');
  send(win, {
    type: 'fsResult',
    token: req.token,
    action: 'rename',
    path: '/work/ws/c.py',
  });
  answerListing(win, posted, '/work/ws', '', [
    {name: 'c.py', path: '/work/ws/c.py', isDir: false},
    HOME_FILES[1],
  ]);
  rightClick(win, rowFor(win, '/work/ws/b.py'));
  posted.length = 0;
  click(win, menuItem(win, 'Compare with Selected'));
  const compare = msgs(posted, 'fsAction');
  assert.strictEqual(compare.length, 1);
  assert.strictEqual(compare[0].action, 'compare');
  assert.strictEqual(compare[0].path, '/work/ws/c.py', 'the renamed path');
  assert.strictEqual(compare[0].dest, '/work/ws/b.py');
  // Deleted: the item is withdrawn.
  req = deleteInExplorer(win, posted, '/work/ws/c.py');
  send(win, {
    type: 'fsResult',
    token: req.token,
    action: 'delete',
    path: '/work/ws/c.py',
  });
  answerListing(win, posted, '/work/ws', '', [HOME_FILES[1]]);
  rightClick(win, rowFor(win, '/work/ws/b.py'));
  assert.ok(
    !menuItem(win, 'Compare with Selected'),
    'withdrawn after the delete',
  );
  assert.ok(menuItem(win, 'Select for Compare'));
  closeMenu(win);
}

function testCompareWithSelectedStaysOnItsMachine() {
  const {win, posted} = boot();
  rightClick(win, rowFor(win, '/work/ws/a.py'));
  click(win, menuItem(win, 'Select for Compare'));
  goToBox(win, posted);
  rightClick(win, rowFor(win, '/srv/proj/b.py'));
  assert.ok(
    !menuItem(win, 'Compare with Selected'),
    "not offered on another machine's tree",
  );
  closeMenu(win);
  // Back home, the selection is still good.
  pickMachine(win, '');
  answerListing(win, posted, '/work/ws', '', HOME_FILES);
  rightClick(win, rowFor(win, '/work/ws/b.py'));
  assert.ok(menuItem(win, 'Compare with Selected'), 'offered again at home');
  closeMenu(win);
  // A folder deleted on the machine does not touch this machine's pick
  // at the same path.
  goToBox(win, posted);
  rightClick(win, rowFor(win, '/srv/proj/a.py'));
  click(win, menuItem(win, 'Select for Compare'));
  const req = deleteInExplorer(win, posted, '/srv/proj/a.py');
  send(win, {
    type: 'fsResult',
    token: req.token,
    action: 'delete',
    path: '/srv/proj/a.py',
    machine: 'gpu-box',
  });
  answerListing(win, posted, '/srv/proj', 'gpu-box', [BOX_FILES[1]]);
  rightClick(win, rowFor(win, '/srv/proj/b.py'));
  assert.ok(
    !menuItem(win, 'Compare with Selected'),
    'withdrawn on the machine',
  );
  closeMenu(win);
}

function testCompareWithSelectedReadsTheSelectionWhenClicked() {
  const {win, posted} = boot();
  const errors = [];
  win.addEventListener('error', e => {
    errors.push(e.message);
    e.preventDefault();
  });
  rightClick(win, rowFor(win, '/work/ws/a.py'));
  click(win, menuItem(win, 'Select for Compare'));
  // The delete of the selected file is in flight when b.py's menu
  // opens, so the item is still offered...
  const req = deleteInExplorer(win, posted, '/work/ws/a.py');
  rightClick(win, rowFor(win, '/work/ws/b.py'));
  const compare = menuItem(win, 'Compare with Selected');
  assert.ok(compare, 'offered while the delete is pending');
  // ...and the reply withdraws the selection under the open menu.
  send(win, {
    type: 'fsResult',
    token: req.token,
    action: 'delete',
    path: '/work/ws/a.py',
  });
  assert.ok(compare.isConnected, 'the menu stayed open');
  posted.length = 0;
  click(win, compare);
  assert.deepStrictEqual(errors, [], 'the click must not throw');
  assert.deepStrictEqual(
    msgs(posted, 'fsAction'),
    [],
    'nothing is compared against a deleted file',
  );
  // A rename landing under the open menu is followed the same way.
  rightClick(win, rowFor(win, '/work/ws/a.py'));
  click(win, menuItem(win, 'Select for Compare'));
  const rename = renameInExplorer(win, posted, '/work/ws/a.py', 'c.py');
  rightClick(win, rowFor(win, '/work/ws/b.py'));
  const compareRenamed = menuItem(win, 'Compare with Selected');
  assert.ok(compareRenamed, 'offered while the rename is pending');
  send(win, {
    type: 'fsResult',
    token: rename.token,
    action: 'rename',
    path: '/work/ws/c.py',
  });
  posted.length = 0;
  click(win, compareRenamed);
  const sent = msgs(posted, 'fsAction');
  assert.strictEqual(sent.length, 1, 'one compare sent');
  assert.strictEqual(sent[0].path, '/work/ws/c.py', 'against the new path');
  assert.strictEqual(sent[0].dest, '/work/ws/b.py');
  assert.deepStrictEqual(errors, []);
}

/** Open the Delete confirmation of *row* and return its Cancel button. */
function openDeleteConfirmation(win, row) {
  click(win, row.querySelector('.sidebar-item-delete'));
  const cancel = row.querySelector('.sidebar-confirm-no');
  assert.strictEqual(
    row.querySelector('.sidebar-item-confirm').style.display,
    '',
    'the confirmation is shown',
  );
  assert.strictEqual(win.document.activeElement, cancel, 'Cancel has focus');
  return cancel;
}

function assertConfirmationOpen(win, row, why) {
  assert.strictEqual(
    row.querySelector('.sidebar-item-confirm').style.display,
    '',
    why + ': the confirmation is still shown',
  );
  assert.strictEqual(
    row.querySelector('.sidebar-item-delete').style.display,
    'none',
    why + ': the Delete button stays hidden',
  );
}

function testScheduleDeleteConfirmationSurvivesAPoll() {
  const {win} = boot();
  showSection(win, 'meta-schedule');
  const jobs = [job({id: 'a'}), job({id: 'b', name: 'Weekly'})];
  send(win, {type: 'cronJobs', jobs});
  const list = byId(win, 'meta-schedule-list');
  openDeleteConfirmation(win, list.children[1]);
  send(win, {type: 'cronJobs', jobs});
  let row = list.children[1];
  assertConfirmationOpen(win, row, 'the poll');
  assert.strictEqual(
    list.children[0].querySelector('.sidebar-item-confirm').style.display,
    'none',
    "the other row's confirmation stays closed",
  );
  assert.strictEqual(
    win.document.activeElement,
    row.querySelector('.sidebar-confirm-no'),
    'Cancel kept the keyboard',
  );
  // The row's order may change between polls: the confirmation follows
  // the job, and the Delete button keeps focus when it had it.
  row.querySelector('.sidebar-confirm-yes').focus();
  send(win, {type: 'cronJobs', jobs: [jobs[1], jobs[0]]});
  row = list.children[0];
  assert.strictEqual(
    row.querySelector('.sidebar-panel-name').textContent,
    'Weekly',
  );
  assertConfirmationOpen(win, row, 'a reordered poll');
  assert.strictEqual(
    win.document.activeElement,
    row.querySelector('.sidebar-confirm-yes'),
    'Delete kept the keyboard',
  );
  // Focus elsewhere stays there; the confirmation is still open.
  const composer = byId(win, 'task-input');
  composer.focus();
  send(win, {type: 'cronJobs', jobs: [jobs[1], jobs[0]]});
  row = list.children[0];
  assertConfirmationOpen(win, row, 'a poll with the composer focused');
  assert.strictEqual(win.document.activeElement, composer);
  // Cancel in the rebuilt row still closes it.
  click(win, row.querySelector('.sidebar-confirm-no'));
  assert.strictEqual(
    row.querySelector('.sidebar-item-confirm').style.display,
    'none',
  );
  assert.strictEqual(
    win.document.activeElement,
    row.querySelector('.sidebar-item-delete'),
  );
  send(win, {type: 'cronJobs', jobs});
  assert.ok(
    all(win, '#meta-schedule-list .sidebar-item-confirm').every(
      w => w.style.display === 'none',
    ),
    'a cancelled confirmation is not revived',
  );
  // The job removed under an open confirmation: nothing to restore.
  openDeleteConfirmation(win, list.children[1]);
  send(win, {type: 'cronJobs', jobs: [jobs[0]]});
  assert.strictEqual(list.children.length, 1);
  assert.strictEqual(
    list.children[0].querySelector('.sidebar-item-confirm').style.display,
    'none',
  );
}

function machineRow(win, name) {
  return byId(win, 'meta-machines-list').querySelector(
    '.machine-row[data-machine="' + name + '"]',
  );
}

function testMachinesDeleteConfirmationSurvivesAPoll() {
  const {win, posted} = boot();
  showSection(win, 'meta-machines');
  openDeleteConfirmation(win, machineRow(win, 'gpu-box'));
  send(win, {type: 'machines', machines: MACHINES, checkedAt: 2});
  assertConfirmationOpen(win, machineRow(win, 'gpu-box'), 'the poll');
  assert.strictEqual(
    win.document.activeElement,
    machineRow(win, 'gpu-box').querySelector('.sidebar-confirm-no'),
    'Cancel kept the keyboard',
  );
  assert.strictEqual(
    machineRow(win, 'laptop').querySelector('.sidebar-item-confirm').style
      .display,
    'none',
    "the local row's confirmation stays closed",
  );
  // Confirming in the rebuilt row sends the removal.
  posted.length = 0;
  click(win, machineRow(win, 'gpu-box').querySelector('.sidebar-confirm-yes'));
  const sent = msgs(posted, 'machineAction');
  assert.strictEqual(sent.length, 1, 'one removal sent');
  assert.strictEqual(sent[0].action, 'remove');
  assert.strictEqual(sent[0].name, 'gpu-box');
}

const tests = [
  [
    'Schedule: the editor keeps the keyboard across a poll',
    testScheduleEditorKeepsTheKeyboardAcrossAPoll,
  ],
  [
    'Schedule: an open Delete confirmation survives a poll',
    testScheduleDeleteConfirmationSurvivesAPoll,
  ],
  [
    'Machines: an open Delete confirmation survives a poll',
    testMachinesDeleteConfirmationSurvivesAPoll,
  ],
  [
    'Compare with Selected reads the selection when clicked',
    testCompareWithSelectedReadsTheSelectionWhenClicked,
  ],
  [
    'Schedule: the add form keeps the keyboard across a poll',
    testScheduleAddFormKeepsTheKeyboardAcrossAPoll,
  ],
  [
    'Machines: the add form keeps the keyboard across a poll',
    testMachinesAddFormKeepsTheKeyboardAcrossAPoll,
  ],
  [
    "a rename follows only the machine's editors",
    testRenameFollowsOnlyTheMachinesTabs,
  ],
  [
    "a delete closes only the machine's editors",
    testDeleteClosesOnlyTheMachinesTabs,
  ],
  [
    'Compare with Selected follows a rename and a delete',
    testCompareWithSelectedFollowsRenameAndDelete,
  ],
  [
    'Compare with Selected stays on its machine',
    testCompareWithSelectedStaysOnItsMachine,
  ],
];

(async () => {
  let failed = 0;
  for (const [name, fn] of tests) {
    try {
      await fn();
      console.log('ok - ' + name);
    } catch (e) {
      failed += 1;
      console.log('not ok - ' + name);
      console.log(e && e.stack ? e.stack : String(e));
    } finally {
      while (openWindows.length) openWindows.pop().close();
    }
  }
  console.log(`\n${tests.length - failed} passed, ${failed} failed`);
  process.exit(failed ? 1 : 0);
})();
