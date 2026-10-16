// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// E2E test (jsdom page): a filesystem-root work dir must never be
// adopted by a tab or stamped on a tab-scoped command.
//
// A pre-guard daemon (or the task history it persisted) can hand the
// page `work_dir: '/'` -- via configData or a task_events replay --
// which used to root every daemon-bound command at the whole disk.
// `workDirForTab()` and the two replay adoption sites ignore roots
// (isRootDir in media/main.js).
//
// workDirForTab() resolves the GLOBAL working directory (configData
// work_dir) first and a tab's own folder (learned from a replay) only
// while there is no usable global one, so the replay guard is observed
// with an empty global value, and each case also checks that a real
// global value wins over whatever the replay recorded.
//
// The probe is the `@` file picker: typing `@a` posts a getFiles command
// stamped with the active tab's work dir.

'use strict';

const assert = require('assert');
const {makePage, send} = require('./simplify2_harness.js');

function lastMsg(posted, type) {
  for (let i = posted.length - 1; i >= 0; i -= 1) {
    if (posted[i] && posted[i].type === type) return posted[i];
  }
  return null;
}

// The work dir the active tab stamps on the file picker's getFiles.
function pickerWorkDir(win, posted) {
  const inp = win.document.getElementById('task-input');
  inp.value = '';
  const before = posted.length;
  for (const ch of 'see @a') {
    inp.value += ch;
    inp.setSelectionRange(inp.value.length, inp.value.length);
    inp.dispatchEvent(new win.Event('input', {bubbles: true}));
  }
  const cmd = lastMsg(posted.slice(before), 'getFiles');
  assert.ok(cmd, 'typing an @-mention must send getFiles');
  inp.value = '';
  inp.dispatchEvent(new win.Event('input', {bubbles: true}));
  return cmd.workDir;
}

function setConfigWorkDir(win, workDir) {
  send(win, {type: 'configData', config: {work_dir: workDir}});
}

function replay(win, workDir, tabFields) {
  send(win, {
    type: 'task_events',
    ...(tabFields || {}),
    task: 'replayed task',
    events: [],
    extra: JSON.stringify({work_dir: workDir}),
  });
}

// Chat tabs have no row of their own: the Chats panel's pick is
// switchToTab.
function activateTab(win, tabId) {
  assert.ok(
    win._testApi.openTabs().some(t => t.id === tabId),
    `the ${tabId} tab must be open`,
  );
  win._testApi.switchToTab(tabId);
  assert.strictEqual(win._testApi.getActiveTabId(), tabId);
}

const BG_TABS = [
  {tabId: 'tab-a', chatId: 'chat-a', title: 'a', workDir: ''},
  {tabId: 'tab-bg', chatId: 'chat-bg', title: 'bg', workDir: ''},
];
const BG_FIELDS = {tabId: 'tab-bg', chat_id: 'chat-bg'};

// A root global work dir (a pre-guard daemon whose fallback had
// degenerated to '/') must not be stamped on tab-scoped commands ...
function testRootConfigWorkDirNotStamped() {
  const {win, posted} = makePage();
  setConfigWorkDir(win, '/');
  assert.strictEqual(
    pickerWorkDir(win, posted),
    '',
    'a root global work dir must be treated as no work dir',
  );
  win.close();
  console.log('ok - root config work dir is not stamped on commands');
}

// ... and, being treated as none, it leaves the tab's own folder as the
// fallback instead of hiding it.
function testRootConfigWorkDirFallsThroughToTabFolder() {
  const {win, posted} = makePage();
  setConfigWorkDir(win, '/');
  replay(win, '/proj/repo');
  assert.strictEqual(
    pickerWorkDir(win, posted),
    '/proj/repo',
    'a root global work dir must fall through to the tab folder',
  );
  win.close();
  console.log('ok - root config work dir falls through to the tab folder');
}

// A replayed task that truthfully recorded work_dir '/' (run before
// the daemon guard existed) must not poison the live tab: the folder
// the tab already learned survives (workDirForTab() would hide a stored
// root on its own, so a surviving real folder is what proves the replay
// never stored it), and a real global one is untouched by the replay.
function testReplayRootWorkDirNotAdopted() {
  const {win, posted} = makePage();
  setConfigWorkDir(win, '');
  replay(win, '/proj/repo');
  replay(win, '/');
  assert.strictEqual(
    pickerWorkDir(win, posted),
    '/proj/repo',
    'a replayed root work_dir must not displace the tab folder',
  );
  setConfigWorkDir(win, '/x/y');
  assert.strictEqual(
    pickerWorkDir(win, posted),
    '/x/y',
    'the global work dir is stamped once the daemon reports one',
  );
  win.close();
  console.log('ok - replayed root work_dir is not adopted by the tab');
}

// The same replay poisoning through the BACKGROUND-tab branch of
// `task_events` (a hidden tab's transcript restored behind the active
// one): the root must not stick to the hidden tab either.
function testBgReplayRootWorkDirNotAdopted() {
  const {win, posted} = makePage();
  setConfigWorkDir(win, '');
  send(win, {type: 'tabs_state', tabs: BG_TABS});
  replay(win, '/proj/bg', BG_FIELDS);
  replay(win, '/', BG_FIELDS);
  activateTab(win, 'tab-bg');
  assert.strictEqual(
    pickerWorkDir(win, posted),
    '/proj/bg',
    'a bg-replayed root work_dir must not displace the hidden tab folder',
  );
  setConfigWorkDir(win, '/x/y');
  assert.strictEqual(
    pickerWorkDir(win, posted),
    '/x/y',
    'the global work dir is stamped once the daemon reports one',
  );
  win.close();
  console.log('ok - bg-replayed root work_dir is not adopted by the tab');
}

testRootConfigWorkDirNotStamped();
testRootConfigWorkDirFallsThroughToTabFolder();
testReplayRootWorkDirNotAdopted();
testBgReplayRootWorkDirNotAdopted();
console.log('all rootWorkDirTabGuard tests passed');
