// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) tests for EDITOR-TABS MODE in the chat webview
// (media/main.js): the webview is hosted in a VS Code editor tab
// (WebviewPanel) pinned to a single root chat tab, marked by the
// `editor-tab-mode` body class and `data-kiss-*` attributes the
// extension host stamps on <body> (SorcarTab.editorTabBodyAttrs).
//
// Covered behavior:
//  - boot adopts the host-provided root tab id/title and announces it
//    in `ready`;
//  - the internal tab bar stays hidden for a single conversation, has
//    no '+' / settings buttons, and appears only when a sub-agent tab
//    joins the root tab;
//  - the root tab's renames reach the host as `panelTitle` messages;
//  - `tabs_state` reconciliation follows ONLY the root tab's entry
//    (title/chat binding), never adopts other clients' tabs, and posts
//    `closePanel` exactly once when the registered root tab vanishes
//    from a later snapshot;
//  - sub-agent announcements for foreign parents are ignored;
//  - `clearChat` / createNewTab ask the host for a new editor tab
//    (`openChatPanel`) instead of stacking an internal chat tab;
//  - the host's `openSettings` message opens the settings panel;
//  - the settings panel's editor-tabs toggle is visible, checked, and
//    posts `setEditorTabsMode` on change;
//  - the persisted webview state carries `editorRootTabId` for the
//    panel serializer;
//  - `data-kiss-resume-*` attributes trigger a resumeSession right
//    after `ready`.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

function makeWebview(bodyAttrs) {
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace('{{BODY_CLASS_ATTR}}', bodyAttrs);
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
  html = html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');

  const dom = new JSDOM(html, {
    runScripts: 'dangerously',
    pretendToBeVisual: true,
    url: 'https://localhost/',
  });
  const win = dom.window;

  win.Element.prototype.scrollIntoView = function () {};
  win.Element.prototype.scrollTo = function () {};
  win.HTMLElement.prototype.scrollTo = function () {};

  const posted = [];
  let state;
  win.acquireVsCodeApi = function () {
    return {
      postMessage: msg => posted.push(msg),
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };

  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(
    fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8') +
      '\n//# sourceURL=editor-tabs-main.js',
  );

  return {
    win,
    posted,
    getState: () => state,
  };
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function byType(posted, type) {
  return posted.filter(m => m.type === type);
}

function chatTabStrips(win) {
  return win.document.querySelectorAll('#tab-list .chat-tab');
}

const ROOT = 'root-tab-0001';

function editorAttrs(extra) {
  return (
    ' class="editor-tab-mode"' +
    ` data-kiss-tab-id="${ROOT}"` +
    ' data-kiss-tab-title="My chat"' +
    (extra || '')
  );
}

function testBootAdoptsRootTab() {
  const {win, posted, getState} = makeWebview(editorAttrs());
  const readies = byType(posted, 'ready');
  assert.strictEqual(readies.length, 1, 'exactly one ready');
  assert.strictEqual(
    readies[0].tabId,
    ROOT,
    'ready must announce the host-provided root tab id',
  );

  const bar = win.document.getElementById('tab-bar');
  assert.strictEqual(
    bar.style.display,
    'none',
    'single conversation: internal tab bar hidden',
  );
  // The + and settings controls live in the input footer on every
  // surface now.  In editor-tabs mode the + must open a NEW EDITOR TAB
  // (createNewTab posts openChatPanel) instead of stacking a second
  // internal chat tab, and the "..." menu's settings item must open
  // the settings sheet.
  const addBtn = win.document.getElementById('new-chat-btn');
  assert.ok(addBtn, 'footer + button must exist in editor-tabs mode');
  const tabsBefore = chatTabStrips(win).length;
  addBtn.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.strictEqual(
    byType(posted, 'openChatPanel').length,
    1,
    'the + button must ask the host for a new editor tab',
  );
  assert.strictEqual(
    chatTabStrips(win).length,
    tabsBefore,
    'the + button must not create an internal chat tab in editor-tabs mode',
  );
  const settingsBtn = win.document.getElementById('settings-btn');
  assert.ok(settingsBtn, 'footer settings button must exist');
  settingsBtn.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.ok(
    win.document
      .getElementById('settings-panel')
      .classList.contains('open'),
    'the settings menu item must open the settings sheet',
  );

  const titles = byType(posted, 'panelTitle');
  assert.ok(titles.length >= 1, 'boot reports the panel title');
  assert.strictEqual(titles[titles.length - 1].title, 'My chat');
  assert.strictEqual(titles[titles.length - 1].tabId, ROOT);

  assert.ok(getState(), 'boot must persist webview state immediately');
  assert.strictEqual(
    getState().editorRootTabId,
    ROOT,
    'the serializer can re-adopt the chat even after an instant reload',
  );
  return {win, posted, getState};
}

function testReconcileFollowsOwnEntryOnly(win, posted, getState) {
  send(win, {
    type: 'tabs_state',
    tabs: [
      {tabId: ROOT, chatId: 'chat-1', title: 'Renamed chat', workDir: ''},
      {tabId: 'other-tab', chatId: 'chat-2', title: 'Other', workDir: ''},
    ],
  });
  assert.strictEqual(
    chatTabStrips(win).length,
    1,
    'another client tab must NOT be adopted into this panel',
  );
  const titles = byType(posted, 'panelTitle');
  assert.strictEqual(
    titles[titles.length - 1].title,
    'Renamed chat',
    'root entry title must be followed and reported to the host',
  );
  assert.strictEqual(
    byType(posted, 'closePanel').length,
    0,
    'no closePanel while the root tab is listed',
  );
  assert.ok(getState(), 'webview state persisted');
  assert.strictEqual(
    getState().editorRootTabId,
    ROOT,
    'persisted state must carry the root tab id for the serializer',
  );
}

function testClosePanelOnceWhenRootVanishes(win, posted) {
  send(win, {type: 'tabs_state', tabs: []});
  assert.strictEqual(
    byType(posted, 'closePanel').length,
    1,
    'registered root tab vanished: exactly one closePanel',
  );
  send(win, {type: 'tabs_state', tabs: []});
  assert.strictEqual(
    byType(posted, 'closePanel').length,
    1,
    'a snapshot storm must not repeat closePanel',
  );
}

function testSubagentTabs() {
  const {win, posted} = makeWebview(editorAttrs());
  send(win, {
    type: 'openSubagentTab',
    tab_id: 'sub-1',
    parent_tab_id: 'not-in-this-panel',
    description: 'foreign sub-agent',
    task_id: 'task-x',
  });
  assert.strictEqual(
    chatTabStrips(win).length,
    1,
    "another chat's sub-agent must be ignored",
  );

  send(win, {
    type: 'openSubagentTab',
    tab_id: 'sub-2',
    parent_tab_id: ROOT,
    description: 'own sub-agent',
    task_id: 'task-y',
  });
  assert.strictEqual(
    chatTabStrips(win).length,
    2,
    "the root chat's sub-agent gets an internal tab",
  );
  assert.strictEqual(
    win.document.getElementById('tab-bar').style.display,
    '',
    'tab bar appears when a sub-agent tab joins the root tab',
  );

  // Closing the ROOT strip closes the whole panel (retiring the chat)
  // instead of stacking a replacement chat.
  const rootClose = win.document.querySelector(
    '#tab-list .chat-tab .chat-tab-close',
  );
  assert.ok(rootClose, 'root strip has a close control');
  rootClose.dispatchEvent(
    new win.MouseEvent('click', {bubbles: true, cancelable: true}),
  );
  const closes = byType(posted, 'closePanel');
  assert.strictEqual(closes.length, 1, 'root close asks the host to close');
  assert.strictEqual(closes[0].retire, true, 'user close retires the chat');
  assert.strictEqual(
    byType(posted, 'openChatPanel').length,
    0,
    'root close must NOT open a replacement panel',
  );
  assert.strictEqual(
    byType(posted, 'closeTab').length,
    0,
    'the retire travels via closePanel, not a doomed queued closeTab',
  );
}

function testParentlessNewTabIgnored() {
  const {win, posted} = makeWebview(editorAttrs());
  send(win, {type: 'new_tab', task_id: 'task-orphan', parent_tab_id: ''});
  assert.strictEqual(
    chatTabStrips(win).length,
    1,
    'a parentless spawn broadcast must not be adopted by a panel',
  );
  assert.strictEqual(
    byType(posted, 'resumeSession').length,
    0,
    'no duplicate resumeSession for the orphan spawn',
  );
}

function testRegistryBornPanelClosesOnFirstEmptySnapshot() {
  const {win, posted} = makeWebview(editorAttrs(' data-kiss-in-registry="1"'));
  send(win, {type: 'tabs_state', tabs: []});
  assert.strictEqual(
    byType(posted, 'closePanel').length,
    1,
    'a registry-born panel whose tab is gone from the FIRST snapshot ' +
      'must close',
  );
  assert.notStrictEqual(
    byType(posted, 'closePanel')[0].retire,
    true,
    'the registry already dropped the tab: no re-retire',
  );
}

function testClearChatAsksHostForNewPanel() {
  const {win, posted} = makeWebview(editorAttrs());
  // A used tab (welcome hidden) so clearChat takes the createNewTab path.
  win._testApi.hideWelcome();
  send(win, {type: 'clearChat'});
  const opens = byType(posted, 'openChatPanel');
  assert.strictEqual(opens.length, 1, 'clearChat asks the host for a panel');
  assert.strictEqual(opens[0].chatId, undefined, 'fresh chat: no chatId');
  assert.strictEqual(
    chatTabStrips(win).length,
    1,
    'no internal chat tab may be stacked in editor-tabs mode',
  );
}

function testOpenSettingsMessage() {
  const {win} = makeWebview(editorAttrs());
  const panel = win.document.getElementById('settings-panel');
  assert.ok(!panel.classList.contains('open'));
  send(win, {type: 'openSettings'});
  assert.ok(
    panel.classList.contains('open'),
    "the host's openSettings message opens the settings panel",
  );
}

function testSettingsToggle() {
  const {win, posted} = makeWebview(editorAttrs());
  const label = win.document.getElementById('cfg-editor-tabs-mode-label');
  const box = win.document.getElementById('cfg-editor-tabs-mode');
  assert.strictEqual(
    label.style.display,
    '',
    'toggle visible in the VS Code webview',
  );
  assert.strictEqual(box.checked, true, 'reflects the active mode');
  box.checked = false;
  box.dispatchEvent(new win.Event('change', {bubbles: true}));
  const msgs = byType(posted, 'setEditorTabsMode');
  assert.strictEqual(msgs.length, 1);
  assert.strictEqual(msgs[0].enabled, false);
}

function testResumeOnBoot() {
  const {posted} = makeWebview(
    editorAttrs(
      ' data-kiss-resume-chat-id="chat-77" data-kiss-resume-task-id="42"',
    ),
  );
  const readies = byType(posted, 'ready');
  const resumes = byType(posted, 'resumeSession');
  assert.strictEqual(resumes.length, 1, 'one resumeSession on boot');
  assert.strictEqual(resumes[0].id, 'chat-77');
  assert.strictEqual(resumes[0].taskId, 42, 'numeric task id restored');
  assert.strictEqual(resumes[0].tabId, ROOT);
  assert.ok(
    posted.indexOf(readies[0]) < posted.indexOf(resumes[0]),
    'resume must follow ready',
  );
}

function main() {
  const {win, posted, getState} = testBootAdoptsRootTab();
  testReconcileFollowsOwnEntryOnly(win, posted, getState);
  testClosePanelOnceWhenRootVanishes(win, posted);
  testSubagentTabs();
  testParentlessNewTabIgnored();
  testRegistryBornPanelClosesOnFirstEmptySnapshot();
  testClearChatAsksHostForNewPanel();
  testOpenSettingsMessage();
  testSettingsToggle();
  testResumeOnBoot();
  console.log('editorTabsModeWebview: all tests passed');
}

main();
