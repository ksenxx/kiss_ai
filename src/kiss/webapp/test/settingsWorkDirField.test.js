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

'use strict';

const assert = require('assert');
const {openSettings, closeSettings, typeInto} = require('./settings_helpers');
const {
  makePage: makeSharedPage,
  send,
  lastMsg,
} = require('./simplify2_harness');

// The served web page carries body.remote-chat (web_server.py injects the
// class).
function makePage() {
  return makeSharedPage({bodyClass: 'remote-chat'});
}

function assertNoDirectoryFields(win) {
  const labels = Array.from(
    win.document.querySelectorAll('#config-form .config-label'),
  ).map(l => l.textContent.trim());
  assert.ok(
    !labels.some(t => /^(Working|Memory) directory/.test(t)),
    'no settings label may read "Working directory" or "Memory directory"',
  );
}

// The chat tabs mirrored from the registry, in tab order. There is no
// row of chat tabs any more (the Chats panel lists them), so the
// internal tab list is the subject.
function tabBarIds(win) {
  // Array.from: the list comes from the JSDOM realm; deepStrictEqual
  // compares prototypes too.
  return Array.from(win._testApi.openTabs())
    .filter(t => !t.isSubagentTab && !t.isContentTab)
    .map(t => t.id);
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
  assert.deepStrictEqual(
    tabBarIds(win),
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

function main() {
  testRemoteSettingsHaveNoDirectoryFieldsAndSaveNoWorkDir();
  testRemoteBlankWhenNothingIsStored();
  testRemoteAdoptsGlobalWorkDirWithoutEcho();
  console.log('settingsWorkDirField.test.js: all tests passed');
}

main();
