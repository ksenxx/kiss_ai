// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) tests for EDITOR-TABS MODE in the chat webview
// (media/main.js): the webview is hosted in the ONE chat editor tab
// (WebviewPanel), marked by the `editor-tab-mode` body class the
// extension host stamps on <body> (SorcarTab.EDITOR_TAB_BODY_ATTRS).
// Its tab model is the sidebar view's: the daemon's whole tab registry
// is mirrored, chats are switched in place, and only the surface
// differs (the shown chat's title goes to the host for the editor tab).
//
// Covered behavior:
//  - boot announces one new chat in `ready` (no `singleTabId`: the
//    daemon replays every bound tab) and reports its title;
//  - `tabs_state` reconciliation adopts every registry tab;
//  - `panelTitle` follows the chat ON SCREEN: a switch, a rename and
//    the task status all reach the host;
//  - `clearChat` opens a new chat in place and never asks the host for
//    another editor tab (`openChatPanel`);
//  - the host's `openChatFromHistory` resumes a chat in place;
//  - sub-agent tabs: an own chat's sub-agent gets an internal tab, a
//    parentless spawn is adopted by the chat surface but ignored by the
//    chatless history / Task Info panels, which also ignore
//    `tabs_state` and name their root as the only tab to replay;
//  - the host's `openSettings` message opens the settings panel;
//  - the settings panel's editor-tabs toggle is visible, checked, and
//    posts `setEditorTabsMode` on change.

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

const EDITOR = ' class="editor-tab-mode"';

function lastTitle(posted) {
  const titles = byType(posted, 'panelTitle');
  return titles[titles.length - 1];
}

function testBootAnnouncesOneNewChat() {
  const {win, posted} = makeWebview(EDITOR);
  const readies = byType(posted, 'ready');
  assert.strictEqual(readies.length, 1, 'exactly one ready');
  assert.strictEqual(readies[0].tabId, win._testApi.getActiveTabId());
  assert.strictEqual(
    readies[0].singleTabId,
    undefined,
    'the chat editor tab mirrors the whole registry: no single tab',
  );
  assert.strictEqual(win._testApi.openTabs().length, 1);
  assert.strictEqual(lastTitle(posted).title, 'new chat');
  assert.strictEqual(lastTitle(posted).state, '');
  win.close();
}

function testReconcileAdoptsEveryRegistryTab() {
  const {win, posted} = makeWebview(EDITOR);
  const boot = win._testApi.getActiveTabId();
  send(win, {
    type: 'tabs_state',
    tabs: [
      {tabId: boot, chatId: 'chat-1', title: 'Renamed chat', workDir: ''},
      {tabId: 'other-tab', chatId: 'chat-2', title: 'Other', workDir: ''},
    ],
  });
  const ids = win._testApi.openTabs().map(t => String(t.id));
  assert.strictEqual(
    ids.sort().join(','),
    [boot, 'other-tab'].sort().join(','),
    "another client's tab is adopted, like the sidebar view does",
  );
  assert.strictEqual(
    lastTitle(posted).title,
    'Renamed chat',
    'the registry title of the chat on screen reaches the host',
  );

  // The panel title follows the chat ON SCREEN.
  win._testApi.switchToTab('other-tab');
  assert.strictEqual(lastTitle(posted).title, 'Other');
  send(win, {type: 'status', running: true, tabId: 'other-tab', taskId: 7});
  assert.strictEqual(lastTitle(posted).state, 'running');
  win._testApi.switchToTab(boot);
  assert.strictEqual(lastTitle(posted).title, 'Renamed chat');
  assert.notStrictEqual(lastTitle(posted).state, 'running');

  // A snapshot without the other tab drops it (closed elsewhere); the
  // chat on screen stays.
  send(win, {
    type: 'tabs_state',
    tabs: [{tabId: boot, chatId: 'chat-1', title: 'Renamed chat', workDir: ''}],
  });
  assert.strictEqual(
    win._testApi.openTabs().map(t => String(t.id)).join(','),
    boot,
  );
  assert.strictEqual(byType(posted, 'closePanel').length, 0);
  win.close();
}

function testClearChatOpensAChatInPlace() {
  const {win, posted} = makeWebview(EDITOR);
  const boot = win._testApi.getActiveTabId();
  send(win, {type: 'setTaskText', text: 'first task'});
  send(win, {type: 'clearChat'});
  assert.notStrictEqual(
    win._testApi.getActiveTabId(),
    boot,
    'Cmd+T on a used chat opens a new chat in this webview',
  );
  assert.strictEqual(
    byType(posted, 'openChatPanel').length,
    0,
    'the chat editor tab never asks the host for another editor tab',
  );
  assert.strictEqual(lastTitle(posted).title, 'new chat');
  win.close();
}

function testOpenChatFromHistoryResumesInPlace() {
  const {win, posted} = makeWebview(EDITOR);
  send(win, {
    type: 'openChatFromHistory',
    chatId: 'chat-far',
    taskId: 7,
    title: 'an older task',
  });
  const resumes = byType(posted, 'resumeSession');
  assert.strictEqual(resumes.length, 1, 'the history click resumes the chat');
  assert.strictEqual(resumes[0].id, 'chat-far');
  assert.strictEqual(resumes[0].taskId, 7);
  assert.strictEqual(
    resumes[0].tabId,
    win._testApi.getActiveTabId(),
    'in the tab now on screen',
  );
  assert.strictEqual(byType(posted, 'openChatPanel').length, 0);
  win.close();
}

function testSubagentTabs() {
  const {win, posted} = makeWebview(EDITOR);
  const boot = win._testApi.getActiveTabId();
  send(win, {
    type: 'openSubagentTab',
    tab_id: 'sub-1',
    parent_tab_id: 'not-in-this-webview',
    description: 'foreign sub-agent',
    task_id: 'task-x',
  });
  assert.strictEqual(
    win._testApi.openTabs().length,
    1,
    "an unknown chat's sub-agent is ignored",
  );
  send(win, {
    type: 'openSubagentTab',
    tab_id: 'sub-2',
    parent_tab_id: boot,
    description: 'own sub-agent',
    task_id: 'task-y',
  });
  assert.strictEqual(
    win._testApi.openTabs().filter(t => t.isSubagentTab).length,
    1,
    "the chat's sub-agent gets an internal tab",
  );
  assert.strictEqual(
    win.document.getElementById('tab-bar').style.display,
    '',
    'tab bar appears when a sub-agent tab joins the chat',
  );
  // A parentless spawn (a run_agent sub-task) is the chat surface's
  // to adopt.
  send(win, {type: 'new_tab', task_id: 'task-orphan', parent_tab_id: ''});
  assert.strictEqual(
    win._testApi.openTabs().filter(t => t.isSubagentTab).length,
    2,
    'the chat surface adopts a parentless spawn',
  );
  assert.ok(byType(posted, 'resumeSession').length >= 1);
  win.close();
}

function testChatlessPanelsStayOutOfTheRegistry() {
  for (const mode of ['history-panel-mode', 'meta-panel-mode']) {
    const root = mode + '-root';
    const {win, posted} = makeWebview(
      ` class="editor-tab-mode ${mode}" data-kiss-tab-id="${root}"`,
    );
    const ready = byType(posted, 'ready');
    assert.strictEqual(ready[0].tabId, root, 'boots on the stamped root');
    assert.strictEqual(ready[0].singleTabId, root, 'nothing to replay');
    send(win, {
      type: 'tabs_state',
      tabs: [{tabId: 'other-tab', chatId: 'chat-2', title: 'Other', workDir: ''}],
    });
    assert.strictEqual(
      win._testApi.openTabs().map(t => String(t.id)).join(','),
      root,
      mode + ' adopts no registry tab',
    );
    send(win, {type: 'new_tab', task_id: 'task-orphan', parent_tab_id: ''});
    assert.strictEqual(
      win._testApi.openTabs().length,
      1,
      mode + ' adopts no parentless spawn',
    );
    assert.strictEqual(byType(posted, 'resumeSession').length, 0);
    win.close();
  }
}

function testOpenSettingsMessage() {
  const {win} = makeWebview(EDITOR);
  const panel = win.document.getElementById('settings-panel');
  assert.ok(!panel.classList.contains('open'));
  send(win, {type: 'openSettings'});
  assert.ok(
    panel.classList.contains('open'),
    "the host's openSettings message opens the settings panel",
  );
  win.close();
}

function testSettingsToggle() {
  const {win, posted} = makeWebview(EDITOR);
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
  win.close();
}

function main() {
  const tests = [
    testBootAnnouncesOneNewChat,
    testReconcileAdoptsEveryRegistryTab,
    testClearChatOpensAChatInPlace,
    testOpenChatFromHistoryResumesInPlace,
    testSubagentTabs,
    testChatlessPanelsStayOutOfTheRegistry,
    testOpenSettingsMessage,
    testSettingsToggle,
  ];
  for (const t of tests) {
    t();
    console.log(`  ok - ${t.name}`);
  }
  console.log('editorTabsModeWebview: all tests passed');
}

main();
