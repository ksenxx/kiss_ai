// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Three main.js races / stale-state bugs, each driven through the real
// chat.html + api.js + main.js in jsdom (nothing mocked):
//
//  1. a finishing task must not pull the keyboard out of a content
//     view (the editor / terminal / browser screen) into the composer;
//  2. closing an untouched settings sheet posts nothing, and a touched
//     one posts only the touched fields (the daemon merges the payload,
//     so re-sending the loaded values would overwrite what another
//     surface saved meanwhile);
//  3. a file tab whose Monaco load failed keeps its "Reload from disk"
//     button (it used to leave with the Save bar).

'use strict';

const assert = require('assert');
const {
  makePage: makeChatPage,
  send,
  readyTabId,
  lastMsg,
  stubMonaco,
} = require('./simplify2_harness');
const {
  openSettings,
  closeSettings,
  typeInto,
} = require('./settings_helpers');

function openFile(win, tabId, name) {
  send(win, {
    type: 'fileContent',
    tabId: tabId,
    path: '/ws/' + name,
    name: name,
    content: 'hello',
  });
  const tab = win._testApi
    .openTabs()
    .find(t => t.isContentTab && t.contentPath === '/ws/' + name);
  assert.ok(tab, name + ' opened a content tab');
  return tab.id;
}

function testTaskDoneLeavesTheContentViewFocused() {
  const {win, posted} = makeChatPage({bodyClass: 'remote-desktop'});
  const chatId = readyTabId(posted);
  send(win, {type: 'status', running: true, tabId: chatId, startTs: Date.now()});
  openFile(win, chatId, 'notes.txt');
  // The user is working in the content pane (a focusable screen such
  // as the terminal's or the browser's, which is not a text field).
  const view = win.document.querySelector(
    '#content-tab-area .content-tab-view',
  );
  assert.ok(view, 'the content pane shows the file view');
  const field = win.document.createElement('div');
  field.tabIndex = 0;
  view.appendChild(field);
  field.focus();
  assert.strictEqual(win.document.activeElement, field, 'setup: field focused');

  send(win, {type: 'task_done', success: true, tabId: chatId});

  assert.strictEqual(
    win.document.activeElement,
    field,
    'BUG - task_done must not pull the keyboard out of the content pane',
  );
  win.close();
  console.log('  ok - a finishing task leaves the content view focused');
}

const STORED = {
  max_budget: 100,
  remote_password: 'pw',
  custom_endpoint: 'https://old.example/v1',
  custom_api_key: 'old-custom',
  custom_headers: 'X-Old: 1',
};

function testUntouchedPopulatedSettingsSheetSavesNothing() {
  const {win, posted} = makeChatPage();
  openSettings(win);
  send(win, {type: 'configData', config: STORED, apiKeys: {OPENAI_API_KEY: 'k'}});
  const before = posted.length;
  closeSettings(win);
  assert.ok(
    !posted.slice(before).some(m => m.type === 'saveConfig'),
    'BUG - closing a populated sheet nobody touched must not post the form',
  );
  win.close();
  console.log('  ok - an untouched populated settings sheet saves nothing');
}

// The daemon merges the payload, so every untouched key box, empty or
// populated, is left out: sending it would re-save (or, as "", delete)
// a key another window may have changed between populate and close.
function testTouchedSettingsSheetSavesOnlyTheTouchedFields() {
  const {win, posted} = makeChatPage();
  openSettings(win);
  send(win, {type: 'configData', config: STORED, apiKeys: {OPENAI_API_KEY: 'k'}});
  typeInto(win, 'cfg-custom-headers', 'X-New: 1');
  closeSettings(win);
  const save = lastMsg(posted, 'saveConfig');
  assert.ok(save, 'the touched field is saved');
  assert.deepStrictEqual(
    Object.keys(save.config),
    ['custom_headers'],
    'BUG - only the touched field may be sent; got ' +
      JSON.stringify(save.config),
  );
  assert.strictEqual(save.config.custom_headers, 'X-New: 1');
  assert.deepStrictEqual(
    Object.keys(save.apiKeys),
    [],
    'no untouched key is re-sent',
  );
  win.close();
  console.log('  ok - a touched settings sheet saves only the touched fields');
}

async function testMonacoFailureKeepsTheReloadButton() {
  const {win} = makeChatPage({
    bodyClass: 'remote-chat',
    beforeScripts: w => stubMonaco(w, false),
  });
  send(win, {type: 'configData', config: {work_dir: '/ws'}, apiKeys: {}});
  send(win, {
    type: 'fileContent',
    path: '/ws/b.py',
    name: 'b.py',
    content: 'print(1)\n',
    version: '100:8',
  });
  const view = win.document.querySelector('#content-tab-area .content-tab-view');
  assert.ok(view.querySelector('.content-save-bar .content-refresh-btn'));
  const loader = win.document.head.querySelector('script[src*="vs/loader.js"]');
  assert.ok(loader, 'the CDN loader is requested');
  loader.dispatchEvent(new win.Event('error'));
  await new Promise(r => setTimeout(r, 0));
  assert.ok(view.querySelector('.content-code-fallback'), 'the <pre> fallback');
  assert.ok(!view.querySelector('.content-save-bar'), 'the Save bar is gone');
  const btn = view.querySelector('.content-refresh-btn');
  assert.ok(btn, 'BUG - the Reload from disk button must survive the Save bar');
  assert.ok(btn.classList.contains('content-refresh-overlay'), 'it floats');
  win.close();
  console.log('  ok - a failed Monaco load keeps the Reload from disk button');
}

async function main() {
  console.log('f12_main_races.test.js');
  await testMonacoFailureKeepsTheReloadButton();
  testTaskDoneLeavesTheContentViewFocused();
  testUntouchedPopulatedSettingsSheetSavesNothing();
  testTouchedSettingsSheetSavesOnlyTheTouchedFields();
  console.log('f12_main_races.test.js: all tests passed');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
