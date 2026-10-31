// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Four main.js races / stale-state bugs, each driven through the real
// chat.html + api.js + main.js in jsdom (nothing mocked):
//
//  1. a finishing task must not pull the keyboard out of a content
//     view (the editor / terminal / browser screen) into the composer;
//  2. closing the running chat on screen onto a content tab (stacked
//     surface) must leave no running state behind: the timer, the
//     spinner and the closed chat's transcript are gone;
//  3. a finished history row with no recorded end shows no duration
//     (it used to tick upwards from its start forever);
//  4. closing an untouched settings sheet posts nothing, and a touched
//     one posts only the touched fields (the daemon merges the payload,
//     so re-sending the loaded values would overwrite what another
//     surface saved meanwhile);
//  5. a file tab whose Monaco load failed keeps its "Reload from disk"
//     button (it used to leave with the Save bar).

'use strict';

const assert = require('assert');
const {
  makePage: makeChatPage,
  send,
  readyTabId,
  rowsByTitle,
  click,
  byId,
  lastMsg,
  stubMonaco,
} = require('./simplify2_harness');
const {freezeClockAndStyle, makeRow} = require('./history_helpers');
const {
  openSettings,
  closeSettings,
  typeInto,
} = require('./settings_helpers');

function activeTabId(win) {
  return win._testApi.getActiveTabId();
}

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

function testCloseRunningChatOntoContentTabClearsRunningState() {
  const {win, posted} = makeChatPage({bodyClass: 'remote-chat'});
  const chatId = readyTabId(posted);
  send(win, {type: 'status', running: true, tabId: chatId, startTs: Date.now()});
  send(win, {type: 'prompt', text: 'do things', tabId: chatId});
  const fileId = openFile(win, chatId, 'notes.txt');
  // The stacked surface shows the file in place of the chat; go back.
  assert.strictEqual(activeTabId(win), fileId, 'the file took the screen');
  win._testApi.switchToTab(chatId);
  assert.strictEqual(activeTabId(win), chatId, 'back on the chat');
  const statusText = byId(win, 'status-text');
  assert.match(statusText.textContent, /^Running/, 'setup: timer ticking');

  // The user closes the running chat: the file tab next to it is the
  // successor, so the chat surface goes off screen with nothing to
  // replace its transcript.
  click(
    win,
    win.document.querySelector(
      `#tab-list .chat-tab[data-tab-id="${chatId}"] .chat-tab-close`,
    ),
  );
  assert.strictEqual(activeTabId(win), fileId, 'the file tab took over');
  assert.ok(
    !win._testApi.openTabs().some(t => t.id === chatId),
    'the chat is gone',
  );
  assert.doesNotMatch(
    statusText.textContent,
    /^Running/,
    'BUG - the closed chat\'s timer must stop',
  );
  assert.strictEqual(
    byId(win, 'output').children.length,
    0,
    'BUG - the closed chat\'s transcript must leave #output',
  );
  assert.strictEqual(
    byId(win, 'stop-btn').style.display,
    'none',
    'nothing is running: no Stop button',
  );
  win.close();
  console.log('  ok - closing the running chat onto a file leaves no running state');
}

const FROZEN_NOW_MS = 1_700_500_000_000;

function testFinishedRowWithoutEndShowsNoDuration() {
  const {win} = makeChatPage({
    beforeScripts: w => freezeClockAndStyle(w, FROZEN_NOW_MS),
  });
  send(win, {type: 'configData', config: {work_dir: ''}, apiKeys: {}});
  send(win, {
    type: 'history',
    offset: 0,
    sessions: [
      makeRow({
        task_id: 1,
        title: 'crashed without an end',
        is_running: false,
        startTs: FROZEN_NOW_MS - 3_600_000,
        endTs: 0,
      }),
      makeRow({
        task_id: 2,
        title: 'still running',
        is_running: true,
        startTs: FROZEN_NOW_MS - 65_000,
        endTs: 0,
      }),
    ],
  });
  const rows = rowsByTitle(win);
  const crashed = rows['crashed without an end'].querySelector(
    '.running-item-metrics',
  ).textContent;
  assert.doesNotMatch(
    crashed,
    /\d\d:\d\d:\d\d/,
    'BUG - a finished row with no recorded end must show no duration; got: ' +
      crashed,
  );
  const running = rows['still running'].querySelector(
    '.running-item-metrics',
  ).textContent;
  assert.match(running, /00:01:05\b/, 'a running row still ticks: ' + running);
  win.close();
  console.log('  ok - a finished row without an end shows no duration');
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
  testCloseRunningChatOntoContentTabClearsRunningState();
  testFinishedRowWithoutEndShowsNoDuration();
  testUntouchedPopulatedSettingsSheetSavesNothing();
  testTouchedSettingsSheetSavesOnlyTheTouchedFields();
  console.log('f12_main_races.test.js: all tests passed');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
