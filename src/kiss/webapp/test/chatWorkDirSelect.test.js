// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) tests for the working directory drop-down of the
// chat header ("machine: directory" at the top of the chat pane): it
// lists the directory the next task runs in (selected), every other
// directory selected so far (configData.recent_work_dirs, most recent
// first) and a final "Choose ..." entry that opens a folder picker.
// The "..." menu has no "Working directory" item any more.
//
// The working directory is ONE global value (config.json work_dir) that
// every task runs in; a pick changes it for every tab.  A listed
// directory is checked through the daemon's listDir ('workdir:<n>'
// token) and then sent as setWorkDir (never saveConfig); "Choose ..."
// opens the in-page folder browser.  `submit` carries no workDir (the
// daemon runs the task in the global value), a running task never blocks
// a pick (it keeps the folder it started in), a refusal is an error
// notification, and the daemon's workDirChanged broadcast re-scopes the
// page to a pick made in another browser.

/* global require, __dirname, console, process */

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
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
const {click} = require('./simplify2_harness');

const MEDIA = path.join(__dirname, '..', 'media');

const NOW = Date.now() / 1000;
const RECENTS = [
  {path: '/home/u/older', ts: NOW - 3 * 3600},
  {path: '/home/u/newest', ts: NOW - 30},
  {path: '/home/u/middle', ts: NOW - 2 * 86400},
];

/** Submit *text* from the composer; the posted `submit` message. */
function submitPrompt(win, posted, text) {
  const before = msgs(posted, 'submit').length;
  byId(win, 'task-input').value = text;
  click(win, byId(win, 'send-btn'));
  const sent = msgs(posted, 'submit');
  assert.strictEqual(sent.length, before + 1, 'one submit is posted');
  return sent[sent.length - 1];
}

/** The ids of every open tab (a background chat has no strip entry, so
 *  the registry, not the DOM, lists them). */
function openTabIds(win) {
  return Array.from(win._testApi.openTabs(), t => t.id);
}

/** Bring a background chat on screen the way the Chats panel's pick does. */
function switchToChat(win, tabId) {
  assert.ok(openTabIds(win).includes(tabId), 'tab ' + tabId + ' is open');
  win._testApi.switchToTab(tabId);
}

// ---------------------------------------------------------------------------

function testHeaderShowsMachineThenDirectory() {
  const {win} = makePage();
  const header = byId(win, 'chat-header');
  const sep = byId(win, 'chat-header-sep');
  assert.strictEqual(
    Array.from(header.children, el => el.id).join(','),
    'chat-machine-wrap,chat-header-sep,chat-workdir-wrap',
    'machine, ":", directory -- in that order',
  );
  assert.strictEqual(sep.textContent, ':');
  // Each select sits alone in a .chat-header-select span: its ::after
  // is the down chevron that makes the plain text read as a drop-down
  // (a bordered square turned 45deg in the current colour, over the
  // select's right padding, letting clicks through to the select).
  ['chat-machine', 'chat-workdir'].forEach(id => {
    const wrap = byId(win, id + '-wrap');
    assert.ok(wrap.classList.contains('chat-header-select'), id + ' wrapped');
    assert.strictEqual(wrap.children.length, 1);
    assert.strictEqual(wrap.firstElementChild.id, id);
  });
  const css = fs.readFileSync(path.join(MEDIA, 'main.css'), 'utf8');
  const chevron = css.match(/\.chat-header-select::after\s*\{([^}]*)\}/);
  assert.ok(chevron, 'the chevron rule');
  assert.ok(/content:\s*''/.test(chevron[1]));
  assert.ok(/position:\s*absolute/.test(chevron[1]));
  assert.ok(/right:/.test(chevron[1]));
  assert.ok(/border-right:\s*2px solid currentcolor/.test(chevron[1]));
  assert.ok(/border-bottom:\s*2px solid currentcolor/.test(chevron[1]));
  assert.ok(/rotate\(45deg\)/.test(chevron[1]));
  assert.ok(
    /pointer-events:\s*none/.test(chevron[1]),
    'clicks reach the select',
  );
  const selects = css.match(/#chat-machine,\s*#chat-workdir\s*\{([^}]*)\}/);
  assert.ok(selects, 'the shared select rule');
  assert.ok(/padding:\s*0 1\.1em 0 0/.test(selects[1]), 'room for the chevron');
  assert.ok(/appearance:\s*none/.test(selects[1]), 'no native arrow');
  // A remote pick colours the name and its chevron green (red once
  // the daemon reports the machine down); an empty machine list hides
  // the span, chevron included.
  assert.ok(
    /#chat-machine-wrap:has\(> \.remote\)\s*\{\s*color:\s*var\(--status-ok\)/.test(
      css,
    ),
  );
  assert.ok(
    /#chat-machine-wrap:has\(> \.remote\.down\)\s*\{\s*color:\s*var\(--status-fail\)/.test(
      css,
    ),
  );
  assert.ok(
    /#chat-machine-wrap:has\(> #chat-machine:empty\)\s*\{\s*display:\s*none/.test(
      css,
    ),
  );
  // Nothing known yet: no machine name, no ":", the placeholder
  // selected and "Choose ..." a pick away.
  assert.strictEqual(byId(win, 'chat-machine').options.length, 0);
  assert.ok(sep.hidden, 'no machine name yet: no ":"');
  assert.deepStrictEqual(optionLabels(win), [
    'No working directory',
    'Choose ...',
  ]);
  assert.strictEqual(currentDir(win), '');
  assert.ok(byId(win, 'chat-workdir-none').selected);
  assert.ok(
    byId(win, 'chat-workdir-none').disabled,
    'the placeholder is no pick',
  );

  // The "..." menu has no "Working directory" item any more, and the
  // panel it used to open is gone.
  assert.strictEqual(win.document.getElementById('workdir-btn'), null);
  assert.strictEqual(win.document.getElementById('workdir-panel'), null);
  const labels = Array.from(
    win.document.querySelectorAll('#more-menu .more-item-label'),
    el => el.textContent.trim(),
  );
  assert.ok(labels.length >= 1, 'the "..." menu still has its items');
  assert.ok(!labels.includes('Working directory'), labels.join(', '));

  // The directory alone, still without the ":".
  send(win, {type: 'configData', config: {work_dir: '/work/ws'}});
  assert.ok(sep.hidden, 'no machine name yet: no ":"');
  assert.deepStrictEqual(optionLabels(win), ['/work/ws', 'Choose ...']);
  assert.strictEqual(currentDir(win), '/work/ws');
  assert.strictEqual(
    byId(win, 'chat-workdir').title,
    'The next task runs in /work/ws',
  );

  // The machine name joins it, separated by the ":".
  send(win, {
    type: 'configData',
    machine: 'box',
    config: {work_dir: '/work/ws'},
  });
  assert.strictEqual(byId(win, 'chat-machine').value, '');
  assert.strictEqual(byId(win, 'chat-machine').options[0].textContent, 'box');
  assert.ok(!sep.hidden, 'both known: "box: /work/ws"');
}

function testMachineAloneOffersChoose() {
  const {win} = makePage();
  send(win, {type: 'configData', machine: 'box', config: {}});
  assert.ok(
    !byId(win, 'chat-header-sep').hidden,
    '"box: No working directory"',
  );
  assert.deepStrictEqual(optionLabels(win), [
    'No working directory',
    'Choose ...',
  ]);
  assert.strictEqual(currentDir(win), '');
  assert.strictEqual(
    byId(win, 'chat-workdir').title,
    'Choose the working directory the next task runs in',
  );
  // With nothing known "Choose ..." is still a different entry from
  // the selected one, so picking it fires change and opens the picker.
  pick(win, CHOOSE);
  assert.ok(!byId(win, 'folder-picker').hidden, 'the folder browser opens');
  assert.ok(byId(win, 'chat-workdir-none').selected, 'the placeholder is back');
}

function testRecentsListNewestFirstAfterTheCurrentOne() {
  const {win} = makePage();
  send(win, {type: 'configData', config: {recent_work_dirs: RECENTS}});
  assert.deepStrictEqual(optionLabels(win), [
    'No working directory',
    '/home/u/newest',
    '/home/u/older',
    '/home/u/middle',
    'Choose ...',
  ]);
  assert.strictEqual(currentDir(win), '', 'no global directory yet');
  assert.ok(byId(win, 'chat-workdir-none').selected);

  // The current directory is always listed, first when the daemon's
  // list does not carry it, selected when it does.
  send(win, {
    type: 'configData',
    config: {work_dir: '/work/ws', recent_work_dirs: RECENTS},
  });
  assert.deepStrictEqual(optionLabels(win), [
    '/work/ws',
    '/home/u/newest',
    '/home/u/older',
    '/home/u/middle',
    'Choose ...',
  ]);
  assert.strictEqual(currentDir(win), '/work/ws');
  send(win, {
    type: 'configData',
    config: {work_dir: '/home/u/older/', recent_work_dirs: RECENTS},
  });
  assert.deepStrictEqual(optionLabels(win), [
    '/home/u/newest',
    '/home/u/older',
    '/home/u/middle',
    'Choose ...',
  ]);
  assert.strictEqual(currentDir(win), '/home/u/older');

  // A later configData (another surface selected a folder) repaints
  // the list; junk rows are dropped.
  send(win, {
    type: 'configData',
    config: {
      work_dir: '/home/u/brand-new',
      recent_work_dirs: [
        {path: '/home/u/brand-new', ts: NOW},
        {path: '', ts: NOW},
        {path: '/home/u/no-ts'},
        'junk',
        RECENTS[1],
      ],
    },
  });
  assert.deepStrictEqual(optionLabels(win), [
    '/home/u/brand-new',
    '/home/u/newest',
    'Choose ...',
  ]);

  // A configData without the list leaves the rows alone.
  send(win, {type: 'configData', config: {work_dir: '/home/u/brand-new'}});
  assert.deepStrictEqual(optionLabels(win), [
    '/home/u/brand-new',
    '/home/u/newest',
    'Choose ...',
  ]);
}

function testRemotePickIsCheckedThenAdopted() {
  const {win, posted} = makePage();
  send(win, {
    type: 'configData',
    config: {work_dir: '/work/ws', recent_work_dirs: RECENTS},
  });
  pick(win, '/home/u/older');
  let listDirs = msgs(posted, 'listDir');
  assert.strictEqual(listDirs.length, 1, 'the pick asks the daemon to list it');
  assert.strictEqual(listDirs[0].path, '/home/u/older');
  assert.strictEqual(listDirs[0].workDir, '/home/u/older');
  assert.strictEqual(listDirs[0].token, 'workdir:1');
  assert.strictEqual(
    currentDir(win),
    '/work/ws',
    'the drop-down springs back until the daemon answers',
  );

  // A stale token (some other listing) is ignored.
  send(win, {type: 'dirListing', token: 'workdir:0', error: 'nope'});
  assert.deepStrictEqual(errorNotifications(win), []);

  // Not a folder: an error notification, nothing is sent.
  send(win, {
    type: 'dirListing',
    token: 'workdir:1',
    error: 'Not a directory: /home/u/older',
  });
  assert.deepStrictEqual(
    errorNotifications(win).map(t =>
      t.includes('Not a directory: /home/u/older'),
    ),
    [true],
  );
  assert.strictEqual(msgs(posted, 'setWorkDir').length, 0);
  assert.strictEqual(currentDir(win), '/work/ws');

  // Second try: the daemon lists it (with its canonical spelling), so
  // it becomes the global working directory through setWorkDir -- the
  // settings file is not written from here.
  pick(win, '/home/u/middle');
  listDirs = msgs(posted, 'listDir');
  assert.strictEqual(listDirs.length, 2);
  assert.strictEqual(listDirs[1].token, 'workdir:2');
  send(win, {
    type: 'dirListing',
    token: 'workdir:2',
    path: '/home/u/middle',
    entries: [],
  });
  // api.js spreads the fields before `type`, hence the key order.
  assertJsonEqual(msgs(posted, 'setWorkDir'), [
    {workDir: '/home/u/middle', type: 'setWorkDir'},
  ]);
  assert.strictEqual(msgs(posted, 'saveConfig').length, 0);
  assert.strictEqual(currentDir(win), '/home/u/middle');

  // The client re-scopes at once, without waiting for the broadcast,
  // and the next task carries no directory of its own.
  const sub = submitPrompt(win, posted, 'list the files');
  assert.strictEqual(sub.workDir, undefined);

  // The reply to a superseded pick is ignored; the latest one's adopts.
  pick(win, '/home/u/older');
  pick(win, '/home/u/newest');
  listDirs = msgs(posted, 'listDir');
  assert.strictEqual(listDirs.length, 4);
  send(win, {
    type: 'dirListing',
    token: listDirs[2].token,
    path: '/home/u/older',
  });
  assert.strictEqual(msgs(posted, 'setWorkDir').length, 1);
  send(win, {
    type: 'dirListing',
    token: listDirs[3].token,
    path: '/home/u/newest',
  });
  assert.strictEqual(msgs(posted, 'setWorkDir').length, 2);
  assert.strictEqual(currentDir(win), '/home/u/newest');
}

function testRemoteRootIsRefused() {
  const {win, posted} = makePage();
  send(win, {
    type: 'configData',
    config: {
      recent_work_dirs: [
        {path: '/', ts: NOW},
        {path: '/tmp/..', ts: NOW - 1},
      ],
    },
  });
  // A file-system root is refused before asking the daemon.
  pick(win, '/');
  assert.strictEqual(msgs(posted, 'listDir').length, 0);
  assert.ok(/root/.test(errorNotifications(win).join(' ')));

  // "/tmp/.." passes the lexical root check; the daemon's canonical
  // spelling of it is "/", which must not be adopted.
  pick(win, '/tmp/..');
  const check = msgs(posted, 'listDir').pop();
  assert.strictEqual(check.path, '/tmp/..');
  send(win, {type: 'dirListing', token: check.token, path: '/', entries: []});
  assert.strictEqual(msgs(posted, 'saveConfig').length, 0);
  assert.strictEqual(msgs(posted, 'setWorkDir').length, 0);
  assert.strictEqual(errorNotifications(win).length, 2);
  assert.strictEqual(currentDir(win), '');
}

function testTwoRefusalsInOneMillisecondAreTwoToasts() {
  // Both refusals raise an error toast without an id of its own.  Two
  // of them within one millisecond (the clock is frozen so the second
  // pick cannot land in the next one) must be two toasts, not one toast
  // rewritten.
  const {win, posted} = makePage();
  const now = win.Date.now();
  win.Date.now = () => now;
  send(win, {
    type: 'configData',
    config: {
      recent_work_dirs: [
        {path: '/', ts: NOW},
        {path: '/tmp/..', ts: NOW - 1},
      ],
    },
  });
  pick(win, '/tmp/..');
  const check = msgs(posted, 'listDir').pop();
  send(win, {type: 'dirListing', token: check.token, path: '/', entries: []});
  pick(win, '/');
  const toasts = Array.from(
    win.document.querySelectorAll('.kiss-notification-error'),
    el => el.dataset.notificationId,
  );
  assert.strictEqual(toasts.length, 2);
  assert.notStrictEqual(toasts[0], toasts[1]);
}

function testRemoteChooseOpensThePicker() {
  const {win, posted} = makePage();
  send(win, {type: 'configData', config: {work_dir: '/work/ws'}});
  assert.strictEqual(win.document.getElementById('folder-picker'), null);
  pick(win, CHOOSE);
  assert.strictEqual(
    currentDir(win),
    '/work/ws',
    '"Choose ..." is never the shown value',
  );
  const picker = byId(win, 'folder-picker');
  assert.ok(!picker.hidden, 'the folder browser opens');
  assert.strictEqual(
    picker.querySelector('#folder-picker-title').textContent,
    'Open Folder as Working Directory',
  );
  // The browser lists through the daemon.
  const listing = msgs(posted, 'listDir').pop();
  assert.ok(/^picker:/.test(listing.token));

  // A pick made in the browser closes it and is the directory shown.
  send(win, {
    type: 'dirListing',
    token: listing.token,
    path: listing.path,
    entries: [{name: 'app', isDir: true}],
  });
  click(win, picker.querySelector('.folder-picker-item'));
  click(win, picker.querySelector('.folder-picker-select'));
  assert.ok(picker.hidden, 'the folder browser closes on a pick');
  const sets = msgs(posted, 'setWorkDir');
  assert.ok(sets.length >= 1);
  assert.ok(/\/app$/.test(sets[sets.length - 1].workDir));
  assert.strictEqual(msgs(posted, 'saveConfig').length, 0);
  assert.ok(/\/app$/.test(currentDir(win)));
  assert.strictEqual(optionLabels(win)[0], currentDir(win));
}

function testRemotePickerChoiceSupersedesAPendingCheck() {
  // A listed directory is picked and the daemon is slow to list it;
  // meanwhile "Choose ..." picks another folder in the browser.  The
  // late reply to the first pick belongs to a superseded request and
  // must not undo the newer choice.
  const {win, posted} = makePage();
  send(win, {
    type: 'configData',
    config: {work_dir: '/work/ws', recent_work_dirs: RECENTS},
  });
  pick(win, '/home/u/older');
  const held = msgs(posted, 'listDir').pop();
  assert.strictEqual(held.token, 'workdir:1');

  pick(win, CHOOSE);
  const picker = byId(win, 'folder-picker');
  const listing = msgs(posted, 'listDir').pop();
  send(win, {
    type: 'dirListing',
    token: listing.token,
    path: listing.path,
    entries: [{name: 'new-choice', isDir: true}],
  });
  click(win, picker.querySelector('.folder-picker-item'));
  click(win, picker.querySelector('.folder-picker-select'));
  assert.ok(picker.hidden);
  assert.ok(/\/new-choice$/.test(currentDir(win)));
  assert.strictEqual(msgs(posted, 'setWorkDir').length, 1);

  send(win, {type: 'dirListing', token: held.token, path: '/home/u/older'});
  assert.strictEqual(
    msgs(posted, 'setWorkDir').length,
    1,
    'the superseded reply adopts nothing',
  );
  assert.ok(/\/new-choice$/.test(currentDir(win)), 'the newer choice stays');
}

function testReachingForTheDropDownAsksForAFreshList() {
  const {win, posted} = makePage();
  send(win, {type: 'configData', config: {work_dir: '/work/ws'}});
  const before = msgs(posted, 'getConfig').length;
  byId(win, 'chat-workdir').dispatchEvent(new win.Event('focus'));
  assert.strictEqual(
    msgs(posted, 'getConfig').length,
    before + 1,
    'focusing the drop-down refreshes the list from the daemon',
  );
}

/** Pick *dir* through the daemon round trip (drop-down -> listDir ->
 *  dirListing -> setWorkDir). */
function pickViaDaemon(win, posted, dir) {
  const asked = msgs(posted, 'listDir').length;
  pick(win, dir);
  const checks = msgs(posted, 'listDir');
  assert.strictEqual(checks.length, asked + 1);
  assert.strictEqual(checks[asked].path, dir);
  send(win, {
    type: 'dirListing',
    token: checks[asked].token,
    path: dir,
    entries: [],
  });
  const sets = msgs(posted, 'setWorkDir');
  assertJsonEqual(sets[sets.length - 1], {workDir: dir, type: 'setWorkDir'});
  assert.strictEqual(currentDir(win), dir, 'the verified pick is shown');
}

function testPickIsGlobal() {
  const {win, posted} = makePage();

  // The daemon's global directory arrives untouched in configData.
  send(win, {
    type: 'configData',
    config: {
      work_dir: '/work/ws',
      recent_work_dirs: RECENTS.concat([
        {path: '/elsewhere/repo', ts: NOW - 5},
        {path: '/elsewhere/other', ts: NOW - 6},
      ]),
    },
  });
  assert.strictEqual(
    currentDir(win),
    '/work/ws',
    'the drop-down names the directory the next task would run in',
  );
  const firstTab = win._testApi.getActiveTabId();

  // The daemon listed the picked folder: it is the global directory
  // now, set through setWorkDir and never the settings file.
  pickViaDaemon(win, posted, '/elsewhere/repo');
  assert.strictEqual(msgs(posted, 'saveConfig').length, 0);
  assert.strictEqual(msgs(posted, 'setWorkDir').length, 1);

  // The next task carries no directory: the daemon runs it in the
  // global value.  No workspace scope travels with it either.
  let sub = submitPrompt(win, posted, 'list the files');
  assert.strictEqual(sub.tabId, firstTab);
  assert.strictEqual(sub.workDir, undefined);
  assert.strictEqual(sub.tabScopeWorkDir, undefined);

  // Every other tab sees the same directory: it is not the tab's.
  win._testApi.createNewTab();
  const secondTab = win._testApi.getActiveTabId();
  assert.notStrictEqual(secondTab, firstTab);
  assert.deepStrictEqual(openTabIds(win).sort(), [firstTab, secondTab].sort());
  assert.strictEqual(currentDir(win), '/elsewhere/repo');
  sub = submitPrompt(win, posted, 'third');
  assert.strictEqual(sub.tabId, secondTab);
  assert.strictEqual(sub.workDir, undefined);

  // A running task does not block a pick: it keeps the folder it
  // started in, the next task runs in the new one.
  send(win, {type: 'status', running: true, tabId: secondTab});
  assert.ok(!byId(win, 'chat-workdir').disabled, 'a running chat can pick');
  pick(win, '/elsewhere/other');
  const checks = msgs(posted, 'listDir');
  assert.strictEqual(checks[checks.length - 1].path, '/elsewhere/other');
  assert.deepStrictEqual(errorNotifications(win), [], 'no refusal');
  pick(win, CHOOSE);
  assert.ok(
    !byId(win, 'folder-picker').hidden,
    'the browser opens while running',
  );
  click(win, byId(win, 'folder-picker').querySelector('.folder-picker-close'));
  send(win, {
    type: 'dirListing',
    token: checks[checks.length - 1].token,
    path: '/elsewhere/other',
    entries: [],
  });
  assert.strictEqual(
    currentDir(win),
    '/elsewhere/other',
    'lands while running',
  );
  send(win, {type: 'status', running: false, tabId: secondTab});
  assert.strictEqual(currentDir(win), '/elsewhere/other');
  switchToChat(win, firstTab);
  assert.strictEqual(win._testApi.getActiveTabId(), firstTab);
  assert.strictEqual(
    currentDir(win),
    '/elsewhere/other',
    'the first tab follows the same global directory',
  );
  assert.ok(
    msgs(posted, 'setWorkDir').every(m => m.tabId === undefined),
    'no pick ever named a tab',
  );
}

function testGlobalDirWinsOverReplayedTaskDir() {
  const {win, posted} = makePage();
  send(win, {type: 'configData', config: {work_dir: '/work/ws'}});
  const tabA = win._testApi.getActiveTabId();

  // A replay of the chat's previous task (its own work dir in `extra`)
  // reaches the tab: the global directory still wins for the next run.
  send(win, {
    type: 'task_events',
    tabId: tabA,
    chat_id: 'chat-a',
    task_id: 41,
    task: 'the earlier task',
    events: [],
    extra: JSON.stringify({work_dir: '/old/task/dir', startTs: 1, endTs: 2}),
  });
  assert.strictEqual(currentDir(win), '/work/ws');
  let sub = submitPrompt(win, posted, 'run it');
  assert.strictEqual(sub.workDir, undefined);

  // Only while the daemon reports no global directory does the
  // drop-down fall back to the tab's last task directory.
  send(win, {type: 'status', running: false, tabId: tabA});
  send(win, {type: 'configData', config: {}});
  assert.strictEqual(
    currentDir(win),
    '/old/task/dir',
    'without a global directory the tab shows its last task directory',
  );
  // A root as the global value is no directory to run in either.
  send(win, {type: 'configData', config: {work_dir: '/'}});
  assert.strictEqual(
    currentDir(win),
    '/old/task/dir',
    'a root-valued global directory falls back the same way',
  );
  // A fresh tab has no last task: nothing is current.
  win._testApi.createNewTab();
  assert.strictEqual(currentDir(win), '');
  switchToChat(win, tabA);
  assert.strictEqual(currentDir(win), '/old/task/dir');

  // A pick made in another browser arrives as the daemon's broadcast
  // and re-scopes this window too.
  send(win, {type: 'workDirChanged', workDir: '/picked/elsewhere'});
  assert.strictEqual(currentDir(win), '/picked/elsewhere');
  sub = submitPrompt(win, posted, 'once more');
  assert.strictEqual(sub.workDir, undefined);
  assert.strictEqual(msgs(posted, 'setWorkDir').length, 0);
  assert.strictEqual(msgs(posted, 'saveConfig').length, 0);
}

function testRemoteWorkDirChangedRescopes() {
  const {win, posted} = makePage();
  // A directory an older build left in sessionStorage is not this
  // browser tab's own working directory any more: the daemon's value
  // wins, and configData is not echoed back as setWorkDir.
  win.sessionStorage.setItem('sorcar-work-dir', '/stale/per-tab');
  send(win, {type: 'configData', config: {work_dir: '/work/ws'}});
  assert.strictEqual(msgs(posted, 'setWorkDir').length, 0, 'no echo');
  assert.strictEqual(currentDir(win), '/work/ws');
  pick(win, CHOOSE);
  const picker = byId(win, 'folder-picker');
  assert.ok(!picker.hidden);

  // Another browser picked a folder: the daemon's broadcast re-scopes
  // this client -- the drop-down follows; the folder browser this user
  // is in stays (a pick made here still applies, audit1020_workdir_picker
  // covers it).  Nothing is echoed back to the daemon (no setWorkDir
  // loop, no saveConfig).
  send(win, {type: 'workDirChanged', workDir: '/picked/elsewhere'});
  assert.strictEqual(currentDir(win), '/picked/elsewhere');
  assert.ok(!picker.hidden, 'the folder browser stays open');
  assert.strictEqual(msgs(posted, 'setWorkDir').length, 0);
  assert.strictEqual(msgs(posted, 'saveConfig').length, 0);

  // The same value again, or a non-string, changes nothing.
  send(win, {type: 'workDirChanged', workDir: '/picked/elsewhere'});
  send(win, {type: 'workDirChanged', workDir: 42});
  assert.ok(!picker.hidden, 'an unchanged value leaves the browser alone');
  assert.strictEqual(currentDir(win), '/picked/elsewhere');
  click(win, picker.querySelector('.folder-picker-close'));
  assert.ok(picker.hidden);
  const sub = submitPrompt(win, posted, 'go');
  assert.strictEqual(sub.workDir, undefined);
}

function testConfigDataRepaintsTheDropDown() {
  // A configData that carries a new work_dir (a reconnect, a settings
  // save) repaints the drop-down, exactly like the workDirChanged
  // broadcast does; the task-info panel's Workdir item follows too.
  const {win} = makePage();
  send(win, {type: 'configData', config: {work_dir: '/old'}});
  assert.strictEqual(currentDir(win), '/old');
  send(win, {type: 'configData', config: {work_dir: '/new'}});
  assert.strictEqual(currentDir(win), '/new');
  assert.strictEqual(byId(win, 'meta-workdir').textContent, '/new');
}

function testOrphanedFileTabBrowsesTheGlobalDir() {
  // A file tab remembers the folder of the chat it was opened from so
  // the views keep browsing it after that chat closes -- but only as a
  // fallback: once the daemon reports a global directory, the Explorer
  // and Source Control views of the orphaned file tab browse THAT.
  const {win, posted} = makePage({desktop: true});
  send(win, {type: 'configData', config: {work_dir: '/old'}});
  const owner = win._testApi.getActiveTabId();
  send(win, {
    type: 'fileContent',
    path: '/old/a.txt',
    name: 'a.txt',
    content: 'x',
    tabId: owner,
  });
  // The split layout lists content tabs on the content pane's own row.
  const fileTab = win.document.querySelector(
    '#content-tab-list .chat-tab.content-tab',
  );
  assert.ok(fileTab, 'the file opened as a content tab');
  click(win, fileTab);
  click(
    win,
    win.document.querySelector(
      '#tab-list .chat-tab[data-tab-id="' + owner + '"] .chat-tab-close',
    ),
  );
  assert.ok(!openTabIds(win).includes(owner), 'the owning chat is closed');
  // Opening the file slid the task-info panel off the desktop layout;
  // its views reload the moment the drawer brings the panel back.
  assert.ok(
    win.document.body.classList.contains('meta-hidden'),
    'the file open hid the panel',
  );
  const before = msgs(posted, 'listDir').length;
  send(win, {type: 'workDirChanged', workDir: '/new'});
  assert.strictEqual(
    msgs(posted, 'listDir').length,
    before,
    'a hidden Explorer is only marked stale',
  );
  // The Explorer and Source Control sections of the task-info panel
  // follow the new workspace on their own.
  click(win, byId(win, 'meta-drawer'));
  const listing = msgs(posted, 'listDir').pop();
  assert.strictEqual(listing.path, '/new');
  assert.strictEqual(listing.workDir, '/new');
  const status = msgs(posted, 'gitStatus').pop();
  assert.strictEqual(status.workDir, '/new');
  // The desktop layout polls the task info every 5 s for as long as the
  // page lives; closing the window drops that timer so node can exit.
  win.close();
}

const tests = [
  ['header: machine, ":", directory', testHeaderShowsMachineThenDirectory],
  ['header: nothing known offers "Choose ..."', testMachineAloneOffersChoose],
  [
    'recents newest first after the current one',
    testRecentsListNewestFirstAfterTheCurrentOne,
  ],
  [
    'a pick is checked by the daemon, then adopted',
    testRemotePickIsCheckedThenAdopted,
  ],
  ['a root is refused', testRemoteRootIsRefused],
  ['"Choose ..." opens the in-page browser', testRemoteChooseOpensThePicker],
  [
    'a browser choice supersedes a pending check',
    testRemotePickerChoiceSupersedesAPendingCheck,
  ],
  ['focus asks for a fresh list', testReachingForTheDropDownAsksForAFreshList],
  ['a pick is the global directory of every tab', testPickIsGlobal],
  [
    'the global directory wins over a replayed task directory',
    testGlobalDirWinsOverReplayedTaskDir,
  ],
  [
    'workDirChanged from the daemon re-scopes the client',
    testRemoteWorkDirChangedRescopes,
  ],
  ['configData repaints the drop-down', testConfigDataRepaintsTheDropDown],
  [
    'an orphaned file tab browses the global directory',
    testOrphanedFileTabBrowsesTheGlobalDir,
  ],
  [
    'two refusals in one millisecond are two toasts',
    testTwoRefusalsInOneMillisecondAreTwoToasts,
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
if (failed) {
  console.log(`${failed} of ${tests.length} tests failed`);
  process.exit(1);
}
console.log(`all ${tests.length} chatWorkDirSelect tests passed`);
process.exit(0);
