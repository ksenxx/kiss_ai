// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) tests for the tab strip (renderTabBar in media/main.js)
// now that chats are picked from the Chats panel rather than a row of tabs:
//
//   * the strip (#tab-list) lists the chat on screen with its sub-agents;
//   * content tabs share that strip on the VS Code sidebar and mobile file
//     screen, but stay off the mobile chat screen;
//   * it appears only when it has more than one entry;
//   * a chat that is not on screen has no entry: it is brought back the
//     way the Chats panel's pick does it (win._testApi.switchToTab);
//   * in editor-tabs mode the strip holds the panel's one group.
//
// Covered for the sidebar webview and the remote webapp.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

function makeWebview(opts) {
  opts = opts || {};
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
  html = html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');
  if (opts.editorRoot) {
    html = html.replace(
      '<body',
      `<body class="editor-tab-mode" data-kiss-tab-id="${opts.editorRoot}"` +
        ' data-kiss-tab-title="My chat"',
    );
  }
  const dom = new JSDOM(html, {
    runScripts: 'dangerously',
    pretendToBeVisual: true,
    url: 'https://localhost/',
  });
  const win = dom.window;
  if (opts.remote) win.document.body.classList.add('remote-chat');
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
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function entry(tabId, title) {
  return {tabId, chatId: 'chat-' + tabId, title: title || tabId, workDir: ''};
}

// Every open tab, in tab order (copied into this realm's Array so
// deepStrictEqual compares values, not realms' prototypes).
function openIds(win) {
  return Array.from(win._testApi.openTabs(), t => t.id);
}

function stripIds(win) {
  return Array.from(
    win.document.querySelectorAll('#tab-list .chat-tab[data-tab-id]'),
  ).map(el => el.dataset.tabId);
}

function stripActive(win) {
  const el = win.document.querySelector('#tab-list .chat-tab.active');
  return el ? el.dataset.tabId : null;
}

function stripShown(win) {
  return win.document.getElementById('tab-bar').style.display !== 'none';
}

function click(win, el, what) {
  assert.ok(el, `cannot click a missing ${what}`);
  el.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

function stripEl(win, tabId) {
  return win.document.querySelector(
    `#tab-list .chat-tab[data-tab-id="${tabId}"]`,
  );
}

// The Chats panel's pick of a chat that is not on screen.
function pick(win, tabId) {
  assert.ok(openIds(win).includes(tabId), `cannot pick a closed tab ${tabId}`);
  win._testApi.switchToTab(tabId);
}

function pressKey(win, el, key) {
  el.focus();
  el.dispatchEvent(
    new win.KeyboardEvent('keydown', {key, bubbles: true, cancelable: true}),
  );
}

function spawnSubagent(win, parentId, subId, title) {
  send(win, {
    type: 'openSubagentTab',
    tab_id: subId,
    parent_tab_id: parentId,
    description: title || subId,
    task_id: 'task-' + subId,
    taskIndex: 0,
    isSubagentTab: true,
  });
}

function openFile(win, ownerId, name) {
  send(win, {
    type: 'fileContent',
    tabId: ownerId,
    path: '/ws/' + name,
    name: name,
    content: 'content of ' + name,
  });
  // A background file remains open even when the mobile chat omits its tab.
  const tab = win._testApi.openTabs().find(t => t.isContentTab && t.title === name);
  return tab ? tab.id : null;
}

// ---- Tests ---------------------------------------------------------

function testLoneChatShowsNoStrip(opts, label) {
  const {win} = makeWebview(opts);
  send(win, {type: 'tabs_state', tabs: [entry('a'), entry('b')]});
  assert.deepStrictEqual(openIds(win), ['a', 'b'], `${label}: both chats open`);
  assert.ok(!stripShown(win), `${label}: a lone chat needs no strip`);
  assert.deepStrictEqual(
    stripIds(win),
    ['a'],
    `${label}: the strip holds the chat on screen only`,
  );
  assert.strictEqual(stripActive(win), 'a');
  assert.strictEqual(
    stripEl(win, 'a').getAttribute('aria-selected'),
    'true',
    `${label}: aria-selected follows the highlight`,
  );
  assert.strictEqual(
    win.document.getElementById('tab-list').getAttribute('role'),
    'tablist',
  );
  assert.strictEqual(
    stripEl(win, 'b'),
    null,
    `${label}: a chat that is not on screen has no strip entry`,
  );

  pick(win, 'b');
  assert.strictEqual(
    stripActive(win),
    'b',
    `${label}: the Chats panel's pick switches chats`,
  );
  assert.deepStrictEqual(stripIds(win), ['b']);
  assert.ok(!stripShown(win));
  // Let queued DOM observers finish before destroying their document.
  queueMicrotask(win.close.bind(win));
  console.log(`  ok - [${label}] a lone chat shows no strip`);
}

function testSubagentsAndFilesFormTheGroupStrip(opts, label) {
  const {win} = makeWebview(opts);
  send(win, {type: 'tabs_state', tabs: [entry('a'), entry('b')]});
  spawnSubagent(win, 'a', 'a-sub-1', 'index the code');
  const fileId = openFile(win, 'a', 'notes.md');
  assert.ok(fileId, `${label}: the file opened as a content tab`);

  assert.ok(
    stripShown(win),
    `${label}: the strip appears once the group grows`,
  );
  assert.deepStrictEqual(
    stripIds(win),
    ['a', 'a-sub-1', fileId],
    `${label}: the file screen lists the chat, its sub-agent and its file`,
  );
  // A file opened for the chat on screen comes forward.
  assert.strictEqual(
    stripActive(win),
    fileId,
    `${label}: the file is on screen`,
  );
  assert.ok(stripEl(win, fileId).classList.contains('content-tab'));
  click(win, stripEl(win, 'a'), 'chat a strip entry');
  assert.strictEqual(stripActive(win), 'a');
  assert.ok(stripEl(win, 'a-sub-1').classList.contains('subagent-tab'));
  assert.deepStrictEqual(
    stripIds(win),
    opts.remote ? ['a', 'a-sub-1'] : ['a', 'a-sub-1', fileId],
    `${label}: mobile chat shows its sub-agent but no editor tabs`,
  );

  click(win, stripEl(win, 'a-sub-1'), 'sub-agent strip entry');
  assert.strictEqual(
    stripActive(win),
    'a-sub-1',
    `${label}: the strip highlights the tab on screen`,
  );
  assert.strictEqual(
    stripEl(win, 'a-sub-1').getAttribute('aria-selected'),
    'true',
  );
  assert.strictEqual(stripEl(win, 'a').getAttribute('aria-selected'), 'false');
  assert.deepStrictEqual(
    stripIds(win),
    opts.remote ? ['a', 'a-sub-1'] : ['a', 'a-sub-1', fileId],
  );

  // Another chat: its own group replaces a's; content tabs stay off
  // the mobile chat screen even when they belong to a different chat.
  pick(win, 'b');
  assert.strictEqual(stripActive(win), 'b');
  assert.deepStrictEqual(
    stripIds(win),
    opts.remote ? ['b'] : ['b', fileId],
    `${label}: chat b's strip excludes other chats' sub-agents`,
  );
  assert.strictEqual(stripShown(win), !opts.remote);

  // Back to a: the pick lands on the chat itself.
  pick(win, 'a');
  assert.strictEqual(
    stripActive(win),
    'a',
    `${label}: picking the chat brings the conversation back`,
  );
  assert.deepStrictEqual(
    stripIds(win),
    opts.remote ? ['a', 'a-sub-1'] : ['a', 'a-sub-1', fileId],
  );
  assert.strictEqual(
    win.document.getElementById('output').style.display,
    '',
    `${label}: the chat surface is on screen`,
  );
  // Let queued DOM observers finish before destroying their document.
  queueMicrotask(win.close.bind(win));
  console.log(
    `  ok - [${label}] sub-agents and files follow the screen's tab strip`,
  );
}

function testBackgroundChatsFileNeverStealsFocus(opts, label) {
  const {win} = makeWebview(opts);
  send(win, {type: 'tabs_state', tabs: [entry('a'), entry('b')]});
  // b's task opens a file while the user reads a: it stays in the
  // background and must not add an editor tab to the mobile chat.
  const fileId = openFile(win, 'b', 'report.html');
  assert.ok(fileId, `${label}: the background file is open`);
  assert.strictEqual(
    stripActive(win),
    'a',
    `${label}: a background file never steals focus`,
  );
  assert.deepStrictEqual(stripIds(win), opts.remote ? ['a'] : ['a', fileId]);
  assert.strictEqual(stripShown(win), !opts.remote);

  pick(win, 'b');
  assert.strictEqual(
    stripActive(win),
    'b',
    `${label}: a never-viewed chat opens on itself`,
  );
  assert.deepStrictEqual(stripIds(win), opts.remote ? ['b'] : ['b', fileId]);
  pick(win, fileId);
  assert.ok(stripEl(win, fileId).classList.contains('content-tab'));
  assert.ok(stripEl(win, fileId).textContent.indexOf('report.html') >= 0);
  // Let queued DOM observers finish before destroying their document.
  queueMicrotask(win.close.bind(win));
  console.log(
    `  ok - [${label}] a background chat's file waits without stealing focus`,
  );
}

function testPendingQuestionFlagsTheSubagentEntry(opts, label) {
  const {win} = makeWebview(opts);
  send(win, {type: 'tabs_state', tabs: [entry('a'), entry('b')]});
  spawnSubagent(win, 'a', 'a-sub-1');
  // A new question pulls the user onto the asking tab.
  send(win, {type: 'askUser', question: 'Continue?', tabId: 'a-sub-1'});
  assert.strictEqual(stripActive(win), 'a-sub-1');
  assert.ok(
    !win.document.querySelector('.chat-tab-attention'),
    `${label}: no flag while the asking tab is on screen`,
  );

  click(win, stripEl(win, 'a'), 'chat a strip entry');
  assert.ok(
    stripEl(win, 'a-sub-1').querySelector('.chat-tab-attention'),
    `${label}: on the strip the sub-agent itself carries the flag`,
  );
  assert.ok(!stripEl(win, 'a').querySelector('.chat-tab-attention'));

  // Another chat on screen: the asking sub-agent is off the strip, and
  // the record still says it waits.
  pick(win, 'b');
  assert.deepStrictEqual(stripIds(win), ['b']);
  assert.ok(
    win._testApi.openTabs().find(t => t.id === 'a-sub-1').askPending,
    `${label}: the sub-agent still waits for its answer`,
  );

  pick(win, 'a');
  assert.strictEqual(stripActive(win), 'a');
  assert.ok(
    stripEl(win, 'a-sub-1').querySelector('.chat-tab-attention'),
    `${label}: back in the group, the flag is on the strip again`,
  );
  click(win, stripEl(win, 'a-sub-1'), 'the asking sub-agent');
  assert.ok(
    !win.document.querySelector('.chat-tab-attention'),
    `${label}: the flag goes once the asking tab is on screen`,
  );
  // Let queued DOM observers finish before destroying their document.
  queueMicrotask(win.close.bind(win));
  console.log(
    `  ok - [${label}] a sub-agent's pending question flags its strip entry`,
  );
}

function testFinishedSubagentsFileMovesUpToTheChat(opts, label) {
  const {win} = makeWebview(opts);
  send(win, {type: 'tabs_state', tabs: [entry('a')]});
  spawnSubagent(win, 'a', 'a-sub-1');
  spawnSubagent(win, 'a-sub-1', 'a-sub-1-sub', 'nested');
  const fileId = openFile(win, 'a-sub-1-sub', 'deep.md');
  assert.deepStrictEqual(
    stripIds(win),
    opts.remote
      ? ['a', 'a-sub-1', 'a-sub-1-sub']
      : ['a', 'a-sub-1', 'a-sub-1-sub', fileId],
  );
  // The outer sub-agent finishes while the nested one still runs: the
  // nested one moves up under the chat and keeps its file.
  send(win, {type: 'subagentDone', tab_id: 'a-sub-1'});
  assert.deepStrictEqual(
    openIds(win),
    ['a', 'a-sub-1-sub', fileId],
    `${label}: only the finished sub-agent is gone`,
  );
  assert.deepStrictEqual(
    stripIds(win),
    opts.remote ? ['a', 'a-sub-1-sub'] : ['a', 'a-sub-1-sub', fileId],
    `${label}: the running nested sub-agent stays in the chat's group`,
  );
  send(win, {type: 'subagentDone', tab_id: 'a-sub-1-sub'});
  assert.deepStrictEqual(openIds(win), ['a', fileId]);
  assert.deepStrictEqual(
    stripIds(win),
    opts.remote ? ['a'] : ['a', fileId],
    `${label}: the finished sub-agent's file stays off the mobile chat`,
  );
  pick(win, fileId);
  assert.deepStrictEqual(stripIds(win), ['a', fileId]);
  // Let queued DOM observers finish before destroying their document.
  queueMicrotask(win.close.bind(win));
  console.log(
    `  ok - [${label}] a finished sub-agent's file moves up to its chat`,
  );
}

function testClosingAChatClosesItsSubagentsAndKeepsItsFiles(opts, label) {
  const {win} = makeWebview(opts);
  send(win, {type: 'tabs_state', tabs: [entry('a'), entry('b')]});
  spawnSubagent(win, 'a', 'a-sub-1');
  const fileId = openFile(win, 'a', 'kept.md');
  click(win, stripEl(win, 'a').querySelector('.chat-tab-close'), 'close a');
  assert.deepStrictEqual(
    openIds(win),
    ['b', fileId],
    `${label}: the closed chat's sub-agent went with it; its file stays open`,
  );
  assert.strictEqual(
    stripActive(win),
    fileId,
    `${label}: the file the user was reading stays on screen`,
  );
  assert.deepStrictEqual(
    stripIds(win),
    [fileId],
    `${label}: a top-level file is its own group`,
  );
  assert.ok(!stripShown(win));
  pick(win, 'b');
  assert.strictEqual(stripActive(win), 'b');
  assert.deepStrictEqual(
    stripIds(win),
    opts.remote ? ['b'] : ['b', fileId],
    `${label}: the orphaned file stays off the mobile chat screen`,
  );
  assert.strictEqual(stripShown(win), !opts.remote);
  pick(win, fileId);
  assert.strictEqual(stripActive(win), fileId);
  // Let queued DOM observers finish before destroying their document.
  queueMicrotask(win.close.bind(win));
  console.log(
    `  ok - [${label}] closing a chat closes its sub-agents and keeps its files`,
  );
}

function testArrowKeysMoveAlongTheStrip(opts, label) {
  const {win} = makeWebview(opts);
  send(win, {type: 'tabs_state', tabs: [entry('a'), entry('b'), entry('c')]});
  spawnSubagent(win, 'a', 'a-sub-1');
  spawnSubagent(win, 'a', 'a-sub-2');
  // a -> a-sub-1 -> a-sub-2 -> a, End/Home; the other chats are not on
  // the strip, so the keys never reach them.
  pressKey(win, stripEl(win, 'a'), 'ArrowRight');
  assert.strictEqual(
    win.document.activeElement,
    stripEl(win, 'a-sub-1'),
    `${label}: ArrowRight along the strip`,
  );
  pressKey(win, win.document.activeElement, 'End');
  assert.strictEqual(win.document.activeElement, stripEl(win, 'a-sub-2'));
  pressKey(win, win.document.activeElement, 'ArrowRight');
  assert.strictEqual(
    win.document.activeElement,
    stripEl(win, 'a'),
    `${label}: the strip wraps around`,
  );
  pressKey(win, win.document.activeElement, 'ArrowLeft');
  assert.strictEqual(win.document.activeElement, stripEl(win, 'a-sub-2'));
  pressKey(win, win.document.activeElement, 'Home');
  assert.strictEqual(win.document.activeElement, stripEl(win, 'a'));
  // Roving tabindex.
  assert.strictEqual(stripEl(win, 'a').getAttribute('tabindex'), '0');
  assert.strictEqual(stripEl(win, 'a-sub-1').getAttribute('tabindex'), '-1');
  pressKey(win, stripEl(win, 'a-sub-1'), 'Enter');
  assert.strictEqual(
    stripActive(win),
    'a-sub-1',
    `${label}: Enter activates a strip entry`,
  );
  assert.strictEqual(stripEl(win, 'a-sub-1').getAttribute('tabindex'), '0');
  // Let queued DOM observers finish before destroying their document.
  queueMicrotask(win.close.bind(win));
  console.log(`  ok - [${label}] arrow keys move along the strip`);
}

function testBrowserTabIsListedButOpensInTheBackground(opts, label) {
  const {win} = makeWebview(opts);
  win.eval(fs.readFileSync(path.join(MEDIA, 'browserTab.js'), 'utf8'));
  send(win, {type: 'tabs_state', tabs: [entry('a')]});
  spawnSubagent(win, 'a', 'a-sub-1');
  send(win, {
    type: 'openBrowserTab',
    tab_id: 'browser__1',
    url: 'https://example.com',
    title: 'Example',
  });
  assert.deepStrictEqual(
    stripIds(win),
    opts.remote ? ['a', 'a-sub-1'] : ['a', 'a-sub-1', 'browser__1'],
    `${label}: a background browser tab stays off the mobile chat screen`,
  );
  assert.strictEqual(
    stripActive(win),
    'a',
    `${label}: it opens in the background`,
  );
  pick(win, 'browser__1');
  assert.strictEqual(stripActive(win), 'browser__1');
  assert.deepStrictEqual(
    stripIds(win),
    ['browser__1'],
    `${label}: owned by no chat, it is its own group`,
  );
  assert.ok(!stripShown(win));
  // Let queued DOM observers finish before destroying their document.
  queueMicrotask(win.close.bind(win));
  console.log(
    `  ok - [${label}] the daemon's browser tab opens in the background`,
  );
}

function testEditorTabsModeKeepsOnlyTheStrip() {
  const {win} = makeWebview({editorRoot: 'root-1'});
  assert.ok(
    !stripShown(win),
    'editor-tabs mode: a lone conversation shows no strip',
  );
  spawnSubagent(win, 'root-1', 'root-1-sub');
  assert.ok(
    stripShown(win),
    'editor-tabs mode: the strip appears with the sub-agent',
  );
  assert.deepStrictEqual(stripIds(win), ['root-1', 'root-1-sub']);
  // Let queued DOM observers finish before destroying their document.
  queueMicrotask(win.close.bind(win));
  console.log('  ok - [editor] editor-tabs mode keeps only the strip');
}

function main() {
  const surfaces = [
    [{}, 'sidebar'],
    [{remote: true}, 'remote'],
  ];
  for (const [opts, label] of surfaces) {
    testLoneChatShowsNoStrip(opts, label);
    testSubagentsAndFilesFormTheGroupStrip(opts, label);
    testBackgroundChatsFileNeverStealsFocus(opts, label);
    testPendingQuestionFlagsTheSubagentEntry(opts, label);
    testFinishedSubagentsFileMovesUpToTheChat(opts, label);
    testClosingAChatClosesItsSubagentsAndKeepsItsFiles(opts, label);
    testArrowKeysMoveAlongTheStrip(opts, label);
    testBrowserTabIsListedButOpensInTheBackground(opts, label);
  }
  testEditorTabsModeKeepsOnlyTheStrip();
  console.log('tabGroupsTwoRows.test.js: all tests passed');
}

main();
