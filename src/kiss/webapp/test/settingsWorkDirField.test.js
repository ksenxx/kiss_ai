// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) tests for the Settings panel having NO "Working
// directory" (and no "Memory directory") field, and for how the config
// reply carries the ONE global working directory instead.
//
// The working directory is a single daemon-wide value (config.json
// work_dir) that every task runs in.  The web client treats
// `configData.config.work_dir` this way:
//
//   * it is adopted as-is -- no `setWorkDir` echo back to the daemon
//     (that used to re-pin the connection; there is no connection pin
//     any more);
//   * the Settings form never carries `work_dir` in `saveConfig`, because
//     the folder is changed only through the chat header's working
//     directory drop-down (a `listDir` check, then `setWorkDir`;
//     the daemon persists it and broadcasts `workDirChanged`);
//   * a `workDirChanged` broadcast from the daemon (a pick made in another
//     browser tab) re-scopes this client without any echo.
//
// What the sheet does have is "Default working directory:", a drop-down
// of the chat header's folders for the folder a NEW chat starts in
// (config.json default_work_dir).  The last three tests pin it: the pick
// is saved with the sheet and adopted by a chat no task has touched, the
// folder picked in the chat header overrides it, "Choose ..." picks
// through the folder dialog, and a registered machine's sheet has no
// such row.

'use strict';

const assert = require('assert');
const {
  openSettings,
  closeSettings,
  typeInto,
  MACHINES,
} = require('./settings_helpers');
const {chatTabs} = require('./history_helpers');
const {
  makePage: makeSharedPage,
  send,
  click,
  byId,
  lastMsg,
  ofType,
  pickMachine,
} = require('./simplify2_harness');
const {installTokenisedCss} = require('./designTokens');
const {
  currentDir,
  pick: pickHeaderDir,
  answerWorkDirCheck,
} = require('./workdir_helpers');

// The served web page carries body.remote-chat (web_server.py injects the
// class).  The page script is named for projects.coverage.js, which
// measures the default-working-directory code this suite runs.
function makePage() {
  return makeSharedPage({
    bodyClass: 'remote-chat',
    sourceUrl: 'projects-main.js',
  });
}

function assertNoDirectoryFields(win) {
  const labels = Array.from(
    win.document.querySelectorAll('#settings-panel .config-label'),
  ).map(l => l.textContent.trim());
  assert.ok(
    !labels.some(t => /^(Working|Memory) directory/.test(t)),
    'no settings label may read "Working directory" or "Memory directory"',
  );
}

function tabEntry(tabId, workDir) {
  return {tabId, chatId: '', title: tabId, workDir};
}

function setWorkDirs(posted) {
  return posted.filter(m => m && m.type === 'setWorkDir').map(m => m.workDir);
}

// The directory the active tab's daemon-bound commands actually carry:
// typing an @-mention posts getFiles stamped with workDirForTab().
function mentionWorkDir(win, posted) {
  const before = posted.length;
  typeInto(win, 'task-input', '@readm');
  const msg = lastMsg(posted.slice(before), 'getFiles');
  assert.ok(msg, 'typing an @-mention must post a getFiles command');
  typeInto(win, 'task-input', '');
  return msg.workDir;
}

// The directory the chat header's working directory drop-down shows.
function headerWorkDir(win) {
  return win.document.getElementById('chat-workdir').value;
}

function testRemoteSettingsHaveNoDirectoryFieldsAndSaveNoWorkDir() {
  const {win, posted} = makePage();
  openSettings(win);
  send(win, {type: 'configData', config: {work_dir: '/srv/project'}});
  assertNoDirectoryFields(win);

  typeInto(win, 'cfg-max-budget', '77');
  closeSettings(win);

  const saved = lastMsg(posted, 'saveConfig');
  assert.ok(saved, 'closing the settings panel must save the form');
  assert.strictEqual(saved.config.max_budget, 77, 'the edit is saved');
  assert.ok(
    !('work_dir' in saved.config) && !('memory_dir' in saved.config),
    'the form has no directory boxes, so it must not invent either key ' +
      '(the daemon merges the payload; a blank would wipe the stored value)',
  );
  win.close();
  console.log('  ok - the web client settings carry no directory fields');
}

function testRemoteBlankWhenNothingIsStored() {
  const {win, posted} = makePage();
  openSettings(win);
  send(win, {type: 'configData', config: {}});
  closeSettings(win);

  assert.strictEqual(
    lastMsg(posted, 'saveConfig'),
    null,
    'closing an untouched settings panel saves nothing (so no work_dir)',
  );
  assert.deepStrictEqual(setWorkDirs(posted), [], 'nothing to announce');
  assert.strictEqual(
    mentionWorkDir(win, posted),
    '',
    'with no global directory yet, commands carry none and the daemon ' +
      'resolves its own fallback',
  );
  win.close();
  console.log('  ok - a blank stored working directory announces nothing');
}

function testRemoteAdoptsGlobalWorkDirWithoutEcho() {
  const {win, posted} = makePage();
  send(win, {type: 'configData', config: {work_dir: '/srv/project'}});
  assert.deepStrictEqual(
    setWorkDirs(posted),
    [],
    'the config reply reports the global value; echoing it back as ' +
      'setWorkDir would persist and re-broadcast what the daemon just said',
  );
  assert.strictEqual(
    mentionWorkDir(win, posted),
    '/srv/project',
    'commands from this page run in the global working directory',
  );
  assert.strictEqual(headerWorkDir(win), '/srv/project');
  send(win, {
    type: 'tabs_state',
    tabs: [tabEntry('p-1', '/srv/project'), tabEntry('q-1', '/srv/other')],
  });
  // There is no row of chat tabs any more (the Chats panel lists them),
  // so the internal tab list is the subject.  Array.from: the list comes
  // from the JSDOM realm, and deepStrictEqual compares prototypes too.
  assert.deepStrictEqual(
    Array.from(chatTabs(win), t => t.id),
    ['p-1', 'q-1'],
    'the tab bar still shows every registry tab: the directory is where ' +
      'tasks run, not a filter',
  );
  win.close();
  console.log('  ok - the web client adopts the global directory without echo');
}

// Changing the folder through the chat header's drop-down (listDir
// confirm, then setWorkDir, never saveConfig, re-scoped at once) and
// following the daemon's workDirChanged broadcast are pinned by
// chatWorkDirSelect.test.js.

// ---- "Default working directory" ---------------------------------------

// The folders selected so far on the daemon, as configData lists them.
const FOLDERS = [
  {path: '/srv/project', ts: 2},
  {path: '/srv/other', ts: 1},
];

/**
 * A page of a daemon that runs in /srv/project, has selected /srv/other
 * too and stores `defaultDir` as its default working directory.
 */
function makeFolderPage(defaultDir) {
  const page = makePage();
  sendConfig(page.win, '/srv/project', defaultDir);
  return page;
}

/** The daemon's configData: it runs in `workDir`, its default is `defaultDir`. */
function sendConfig(win, workDir, defaultDir) {
  send(win, {
    type: 'configData',
    config: {
      work_dir: workDir,
      default_work_dir: defaultDir,
      recent_work_dirs: FOLDERS,
    },
  });
}

/** The texts of the Settings drop-down's entries, in order. */
function defaultDirEntries(win) {
  return Array.from(byId(win, 'cfg-default-work-dir').options, o => o.textContent);
}

/** Pick the entry valued `value` in the Settings drop-down, as the user does. */
function pickDefaultDir(win, value) {
  const select = byId(win, 'cfg-default-work-dir');
  select.value = value;
  select.dispatchEvent(new win.Event('change', {bubbles: true}));
}

/** How many folders the page has asked the daemon to check before adopting them. */
function workDirChecks(posted) {
  return ofType(posted, 'listDir').filter(m =>
    String(m.token).startsWith('workdir:'),
  ).length;
}

/** Have the chat on screen run a task, then press "+": a new chat. */
function runTaskThenOpenNewChat(win, chatId) {
  send(win, {
    type: 'task_events',
    events: [],
    task: 'write it',
    tabId: win._testApi.getActiveTabId(),
    chat_id: chatId,
  });
  click(win, byId(win, 'new-chat-btn'));
}

function testDefaultWorkDirIsSavedAdoptedByNewChatsAndOverriddenByTheHeader() {
  const {win, posted} = makeFolderPage('');
  openSettings(win);
  const row = byId(win, 'cfg-default-work-dir-row');
  const select = byId(win, 'cfg-default-work-dir');
  assert.ok(row.textContent.trim().startsWith('Default working directory:'));
  assert.strictEqual(select.parentNode, row, 'the label is followed by the drop-down');
  assert.ok(
    row.compareDocumentPosition(byId(win, 'cfg-max-budget')) &
      win.Node.DOCUMENT_POSITION_FOLLOWING,
    'ahead of Max budget',
  );
  // The folders of the chat header's drop-down, after an entry for none.
  assert.deepStrictEqual(defaultDirEntries(win), [
    'None',
    '/srv/project',
    '/srv/other',
    'Choose ...',
  ]);
  assert.strictEqual(select.value, '');

  // The pick is saved with the sheet, alone...
  pickDefaultDir(win, '/srv/other');
  closeSettings(win);
  assert.deepStrictEqual(
    JSON.parse(JSON.stringify(lastMsg(posted, 'saveConfig').config)),
    {default_work_dir: '/srv/other'},
  );
  // ...and the chat on screen, which no task has touched, starts
  // there: the folder is checked, then adopted, as a pick in the chat
  // header is.
  answerWorkDirCheck(win, posted, '/srv/other');
  assert.deepStrictEqual(setWorkDirs(posted), ['/srv/other']);
  assert.strictEqual(currentDir(win), '/srv/other');

  // The folder picked at the top of the chat overrides the default:
  // neither the save's echo nor the sheet puts the default back.
  pickHeaderDir(win, '/srv/project');
  answerWorkDirCheck(win, posted, '/srv/project');
  const checks = workDirChecks(posted);
  sendConfig(win, '/srv/project', '/srv/other');
  openSettings(win);
  assert.strictEqual(select.value, '/srv/other', 'the sheet shows the default');
  closeSettings(win);
  assert.strictEqual(workDirChecks(posted), checks);
  assert.strictEqual(currentDir(win), '/srv/project');
  assert.deepStrictEqual(setWorkDirs(posted), ['/srv/other', '/srv/project']);

  // The next new chat starts in the default again.
  runTaskThenOpenNewChat(win, 'c1');
  answerWorkDirCheck(win, posted, '/srv/other');
  assert.strictEqual(currentDir(win), '/srv/other');
  // "+" on a chat that is still new opens none: its folder stays.
  pickHeaderDir(win, '/srv/project');
  answerWorkDirCheck(win, posted, '/srv/project');
  click(win, byId(win, 'new-chat-btn'));
  assert.strictEqual(workDirChecks(posted), checks + 2);
  assert.strictEqual(currentDir(win), '/srv/project');
  // "+" from a chat with a task goes to a new chat that is open
  // already, and that one starts in the default too.
  win._testApi.createNewTab();
  send(win, {
    type: 'task_events',
    events: [],
    task: 'write it',
    tabId: win._testApi.getActiveTabId(),
    chat_id: 'c2',
  });
  click(win, byId(win, 'new-chat-btn'));
  assert.strictEqual(workDirChecks(posted), checks + 3);
  answerWorkDirCheck(win, posted, '/srv/other');
  assert.strictEqual(currentDir(win), '/srv/other');
  win.close();
  console.log('  ok - a default working directory is saved, adopted by new chats and overridden by the header');
}

function testDefaultWorkDirLeavesChatsWithTasksAndOtherMachinesAlone() {
  // A default the header's list does not have (selected on another
  // surface, say) is listed, first.
  const {win, posted} = makeFolderPage('/srv/elsewhere');
  openSettings(win);
  assert.deepStrictEqual(defaultDirEntries(win), [
    'None',
    '/srv/elsewhere',
    '/srv/project',
    '/srv/other',
    'Choose ...',
  ]);
  assert.strictEqual(byId(win, 'cfg-default-work-dir').value, '/srv/elsewhere');
  closeSettings(win);

  // A chat that has run a task keeps the folder it is in when the
  // default changes...
  runTaskThenOpenNewChat(win, 'c1');
  answerWorkDirCheck(win, posted, '/srv/elsewhere');
  send(win, {
    type: 'task_events',
    events: [],
    task: 'again',
    tabId: win._testApi.getActiveTabId(),
    chat_id: 'c2',
  });
  const checks = workDirChecks(posted);
  openSettings(win);
  pickDefaultDir(win, '/srv/other');
  closeSettings(win);
  assert.strictEqual(lastMsg(posted, 'saveConfig').config.default_work_dir, '/srv/other');
  assert.strictEqual(workDirChecks(posted), checks);
  assert.strictEqual(currentDir(win), '/srv/elsewhere');

  // ...a default that is the working directory already is not asked
  // for again, and "None" leaves a new chat where the last one was.
  sendConfig(win, '/srv/elsewhere', '/srv/elsewhere');
  click(win, byId(win, 'new-chat-btn'));
  openSettings(win);
  pickDefaultDir(win, '');
  closeSettings(win);
  assert.strictEqual(lastMsg(posted, 'saveConfig').config.default_work_dir, '');
  runTaskThenOpenNewChat(win, 'c3');
  assert.strictEqual(workDirChecks(posted), checks);

  // The default is a folder of this machine: a chat whose tasks run on
  // a registered machine does not start in it.
  sendConfig(win, '/srv/elsewhere', '/srv/other');
  send(win, {type: 'machines', machines: MACHINES});
  pickMachine(win, 'gpu-box');
  runTaskThenOpenNewChat(win, 'c4');
  assert.strictEqual(workDirChecks(posted), checks);
  win.close();
  console.log('  ok - a default working directory leaves chats with tasks and other machines alone');
}

function testChooseOpensTheFolderDialogAndAMachinesSheetHasNoDefault() {
  const {win, posted} = makeFolderPage('/srv/other');
  installTokenisedCss(win, 'main.css');
  openSettings(win);
  const row = byId(win, 'cfg-default-work-dir-row');
  const select = byId(win, 'cfg-default-work-dir');
  assert.notStrictEqual(win.getComputedStyle(row).display, 'none');

  // "Choose ..." is not a value: the drop-down springs back, the sheet
  // hears of no edit, and the folder dialog opens at the default.
  Array.from(select.options).find(o => o.dataset.choose).selected = true;
  select.dispatchEvent(new win.Event('change', {bubbles: true}));
  assert.strictEqual(select.value, '/srv/other');
  const dialog = byId(win, 'folder-picker');
  assert.ok(!dialog.hidden);
  assert.strictEqual(
    byId(win, 'folder-picker-title').textContent,
    'Choose the Default Working Directory',
  );
  const listing = ofType(posted, 'listDir').pop();
  assert.strictEqual(listing.path, '/srv/other');
  click(win, dialog.querySelector('.folder-picker-cancel'));
  closeSettings(win);
  assert.strictEqual(lastMsg(posted, 'saveConfig'), null, 'nothing was edited');

  // The folder picked in the dialog is selected, as an edit of the
  // sheet: a repaint by the daemon's configData keeps it, and closing
  // the sheet saves it.
  openSettings(win);
  Array.from(select.options).find(o => o.dataset.choose).selected = true;
  select.dispatchEvent(new win.Event('change', {bubbles: true}));
  const typed = dialog.querySelector('.folder-picker-input');
  typed.value = '/srv/new';
  click(win, dialog.querySelector('.folder-picker-select'));
  send(win, {
    type: 'dirListing',
    token: ofType(posted, 'listDir').pop().token,
    path: '/srv/new',
    entries: [],
  });
  assert.ok(dialog.hidden);
  assert.strictEqual(select.value, '/srv/new');
  assert.strictEqual(defaultDirEntries(win)[1], '/srv/new');
  sendConfig(win, '/srv/project', '/srv/other');
  assert.strictEqual(select.value, '/srv/new', 'the unsaved pick is kept');
  closeSettings(win);
  assert.strictEqual(lastMsg(posted, 'saveConfig').config.default_work_dir, '/srv/new');

  // The sheet of a registered machine has no such row: the default is
  // a folder of this machine.
  send(win, {type: 'machines', machines: MACHINES});
  pickMachine(win, 'gpu-box');
  openSettings(win);
  assert.ok(byId(win, 'settings-panel').classList.contains('for-machine'));
  assert.strictEqual(win.getComputedStyle(row).display, 'none');
  win.close();
  console.log('  ok - Choose ... picks a default through the folder dialog; a machine\u2019s sheet has no default');
}

function main() {
  testRemoteSettingsHaveNoDirectoryFieldsAndSaveNoWorkDir();
  testRemoteBlankWhenNothingIsStored();
  testRemoteAdoptsGlobalWorkDirWithoutEcho();
  testDefaultWorkDirIsSavedAdoptedByNewChatsAndOverriddenByTheHeader();
  testDefaultWorkDirLeavesChatsWithTasksAndOtherMachinesAlone();
  testChooseOpensTheFolderDialogAndAMachinesSheetHasNoDefault();
  console.log('settingsWorkDirField.test.js: all tests passed');
}

main();
