// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) tests for the two tab rows every surface shows
// (renderTabBar in media/main.js), the mechanism editor-tabs mode has
// always had with VS Code's editor tabs above the webview's own strip:
//
//   * the MAIN row (#main-tab-list) lists the chats and any tab nobody
//     owns (the daemon's browser tab), highlighting the group on screen;
//   * the GROUP strip (#tab-list) lists the chat on screen with the
//     sub-agents it spawned and the files its task opened, and appears
//     only when that group has more than one tab;
//   * a main-row entry returns to the tab last viewed in its group;
//   * in editor-tabs mode the main row stays hidden (VS Code's editor
//     tabs are the main row) and the strip holds the panel's one group.
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

function ids(win, listId) {
  return Array.from(
    win.document.querySelectorAll(`#${listId} .chat-tab[data-tab-id]`),
  ).map(el => el.dataset.tabId);
}

function mainIds(win) {
  return ids(win, 'main-tab-list');
}

function stripIds(win) {
  return ids(win, 'tab-list');
}

function mainActive(win) {
  const el = win.document.querySelector('#main-tab-list .chat-tab.active');
  return el ? el.dataset.tabId : null;
}

function stripActive(win) {
  const el = win.document.querySelector('#tab-list .chat-tab.active');
  return el ? el.dataset.tabId : null;
}

function stripShown(win) {
  return win.document.getElementById('tab-bar').style.display !== 'none';
}

function mainShown(win) {
  return win.document.getElementById('main-tab-bar').style.display !== 'none';
}

function click(win, el, what) {
  assert.ok(el, `cannot click a missing ${what}`);
  el.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

function mainEl(win, tabId) {
  return win.document.querySelector(
    `#main-tab-list .chat-tab[data-tab-id="${tabId}"]`,
  );
}

function stripEl(win, tabId) {
  return win.document.querySelector(
    `#tab-list .chat-tab[data-tab-id="${tabId}"]`,
  );
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
  const tab = Array.from(
    win.document.querySelectorAll('.chat-tab.content-tab[data-tab-id]'),
  ).find(el => el.textContent.indexOf(name) >= 0);
  return tab ? tab.dataset.tabId : null;
}

// ---- Tests ---------------------------------------------------------

function testChatsOnMainRowStripHiddenForLoneChat(opts, label) {
  const {win} = makeWebview(opts);
  send(win, {type: 'tabs_state', tabs: [entry('a'), entry('b')]});
  assert.ok(mainShown(win), `${label}: the main row is on screen`);
  assert.deepStrictEqual(
    mainIds(win),
    ['a', 'b'],
    `${label}: chats on the main row`,
  );
  assert.strictEqual(
    mainActive(win),
    'a',
    `${label}: the chat on screen is highlighted`,
  );
  assert.strictEqual(
    mainEl(win, 'a').getAttribute('aria-selected'),
    'true',
    `${label}: aria-selected follows the highlight`,
  );
  assert.strictEqual(
    win.document.getElementById('main-tab-list').getAttribute('role'),
    'tablist',
  );
  assert.ok(!stripShown(win), `${label}: a lone chat needs no strip`);
  assert.deepStrictEqual(
    stripIds(win),
    ['a'],
    `${label}: the strip holds the group anyway`,
  );

  click(win, mainEl(win, 'b'), 'chat b');
  assert.strictEqual(
    mainActive(win),
    'b',
    `${label}: clicking a main-row entry switches chats`,
  );
  assert.strictEqual(stripActive(win), 'b');
  assert.ok(!stripShown(win));
  win.close();
  console.log(
    `  ok - [${label}] chats live on the main row; a lone chat shows no strip`,
  );
}

function testSubagentsAndFilesFormTheGroupStrip(opts, label) {
  const {win} = makeWebview(opts);
  send(win, {type: 'tabs_state', tabs: [entry('a'), entry('b')]});
  spawnSubagent(win, 'a', 'a-sub-1', 'index the code');
  const fileId = openFile(win, 'a', 'notes.md');
  assert.ok(fileId, `${label}: the file opened as a content tab`);

  assert.deepStrictEqual(
    mainIds(win),
    ['a', 'b'],
    `${label}: sub-agents and files stay off the main row`,
  );
  assert.ok(
    stripShown(win),
    `${label}: the strip appears once the group grows`,
  );
  assert.deepStrictEqual(
    stripIds(win),
    ['a', 'a-sub-1', fileId],
    `${label}: the strip lists the chat, its sub-agent and its file`,
  );
  // A file opened for the chat on screen comes forward; the main row
  // keeps highlighting the chat it belongs to.
  assert.strictEqual(
    stripActive(win),
    fileId,
    `${label}: the file is on screen`,
  );
  assert.strictEqual(
    mainActive(win),
    'a',
    `${label}: its chat stays highlighted`,
  );
  click(win, stripEl(win, 'a'), 'chat a strip entry');
  assert.strictEqual(stripActive(win), 'a');
  assert.ok(stripEl(win, 'a-sub-1').classList.contains('subagent-tab'));
  assert.ok(stripEl(win, fileId).classList.contains('content-tab'));

  // Viewing the sub-agent keeps the chat highlighted on the main row.
  click(win, stripEl(win, 'a-sub-1'), 'sub-agent strip entry');
  assert.strictEqual(
    stripActive(win),
    'a-sub-1',
    `${label}: the strip highlights the tab on screen`,
  );
  assert.strictEqual(
    mainActive(win),
    'a',
    `${label}: the main row highlights the group`,
  );
  assert.strictEqual(mainEl(win, 'a').getAttribute('aria-selected'), 'true');
  assert.strictEqual(stripEl(win, 'a').getAttribute('aria-selected'), 'false');

  // Another chat: its group (just itself) replaces the strip.
  click(win, mainEl(win, 'b'), 'chat b');
  assert.strictEqual(mainActive(win), 'b');
  assert.deepStrictEqual(stripIds(win), ['b']);
  assert.ok(!stripShown(win), `${label}: chat b has no group to show`);

  // Back to a: the main-row entry lands on the tab last viewed there.
  click(win, mainEl(win, 'a'), 'chat a');
  assert.strictEqual(
    stripActive(win),
    'a-sub-1',
    `${label}: the main-row entry returns to the last viewed tab of its group`,
  );
  assert.strictEqual(mainActive(win), 'a');
  assert.ok(stripShown(win));
  assert.deepStrictEqual(stripIds(win), ['a', 'a-sub-1', fileId]);

  // The strip entry of the chat itself brings the conversation back.
  click(win, stripEl(win, 'a'), 'chat a strip entry');
  assert.strictEqual(stripActive(win), 'a');
  assert.strictEqual(
    win.document.getElementById('output').style.display,
    '',
    `${label}: the chat surface is on screen`,
  );
  win.close();
  console.log(
    `  ok - [${label}] sub-agents and files form the group strip under the chat`,
  );
}

function testBackgroundChatsFilesStayInTheirGroup(opts, label) {
  const {win} = makeWebview(opts);
  send(win, {type: 'tabs_state', tabs: [entry('a'), entry('b')]});
  // b's task opens a file while the user reads a: no row shows it yet.
  assert.strictEqual(openFile(win, 'b', 'report.html'), null);
  assert.strictEqual(
    stripActive(win),
    'a',
    `${label}: a background file never steals focus`,
  );
  assert.deepStrictEqual(
    mainIds(win),
    ['a', 'b'],
    `${label}: the file is not a top-level tab`,
  );
  assert.deepStrictEqual(
    stripIds(win),
    ['a'],
    `${label}: the file is not in a's group`,
  );
  assert.ok(!stripShown(win));

  click(win, mainEl(win, 'b'), 'chat b');
  assert.strictEqual(
    stripActive(win),
    'b',
    `${label}: a never-viewed group opens on its chat`,
  );
  const strip = stripIds(win);
  assert.strictEqual(strip.length, 2, `${label}: the file waits in b's group`);
  assert.strictEqual(strip[0], 'b');
  assert.ok(stripEl(win, strip[1]).classList.contains('content-tab'));
  assert.ok(stripEl(win, strip[1]).textContent.indexOf('report.html') >= 0);
  assert.ok(stripShown(win));
  win.close();
  console.log(
    `  ok - [${label}] a background chat's file waits in that chat's group`,
  );
}

function testPendingQuestionOfAGroupShowsOnItsMainRowEntry(opts, label) {
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

  click(win, mainEl(win, 'b'), 'chat b');
  assert.ok(
    mainEl(win, 'a').querySelector('.chat-tab-attention'),
    `${label}: a question waiting in a's sub-agent flags a on the main row`,
  );
  assert.ok(!mainEl(win, 'b').querySelector('.chat-tab-attention'));

  click(win, mainEl(win, 'a'), 'chat a');
  assert.strictEqual(
    stripActive(win),
    'a-sub-1',
    `${label}: the main-row entry returns to the asking tab`,
  );
  assert.ok(
    !win.document.querySelector('.chat-tab-attention'),
    `${label}: the flag goes once the asking tab is on screen`,
  );

  click(win, stripEl(win, 'a'), 'chat a strip entry');
  assert.ok(
    stripEl(win, 'a-sub-1').querySelector('.chat-tab-attention'),
    `${label}: on the strip the sub-agent itself carries the flag`,
  );
  assert.ok(!stripEl(win, 'a').querySelector('.chat-tab-attention'));
  assert.ok(
    mainEl(win, 'a').querySelector('.chat-tab-attention'),
    `${label}: the main-row entry flags any waiting question in its group`,
  );
  win.close();
  console.log(
    `  ok - [${label}] a group's pending question flags its main-row entry`,
  );
}

function testFinishedSubagentsFileMovesUpToTheChat(opts, label) {
  const {win} = makeWebview(opts);
  send(win, {type: 'tabs_state', tabs: [entry('a')]});
  spawnSubagent(win, 'a', 'a-sub-1');
  spawnSubagent(win, 'a-sub-1', 'a-sub-1-sub', 'nested');
  const fileId = openFile(win, 'a-sub-1-sub', 'deep.md');
  assert.deepStrictEqual(stripIds(win), [
    'a',
    'a-sub-1',
    'a-sub-1-sub',
    fileId,
  ]);
  // The outer sub-agent finishes while the nested one still runs: the
  // nested one moves up under the chat and keeps its file.
  send(win, {type: 'subagentDone', tab_id: 'a-sub-1'});
  assert.deepStrictEqual(
    mainIds(win),
    ['a'],
    `${label}: nothing became top-level`,
  );
  assert.deepStrictEqual(
    stripIds(win),
    ['a', 'a-sub-1-sub', fileId],
    `${label}: the running nested sub-agent and its file stay in the chat's group`,
  );
  send(win, {type: 'subagentDone', tab_id: 'a-sub-1-sub'});
  assert.deepStrictEqual(
    mainIds(win),
    ['a'],
    `${label}: the file is still the chat's, not top-level`,
  );
  assert.deepStrictEqual(
    stripIds(win),
    ['a', fileId],
    `${label}: the finished sub-agent's file moved up into the chat's group`,
  );
  win.close();
  console.log(
    `  ok - [${label}] a finished sub-agent's file moves up to its chat`,
  );
}

function testClosingAChatLeavesItsFilesTopLevel(opts, label) {
  const {win} = makeWebview(opts);
  send(win, {type: 'tabs_state', tabs: [entry('a'), entry('b')]});
  spawnSubagent(win, 'a', 'a-sub-1');
  const fileId = openFile(win, 'a', 'kept.md');
  click(win, mainEl(win, 'a').querySelector('.chat-tab-close'), 'close a');
  assert.deepStrictEqual(
    mainIds(win),
    ['b', fileId],
    `${label}: the closed chat's sub-agent went with it; its file stays reachable at top level`,
  );
  assert.ok(!stripShown(win));
  click(win, mainEl(win, fileId), 'the orphaned file');
  assert.strictEqual(
    mainActive(win),
    fileId,
    `${label}: a top-level file is its own group`,
  );
  assert.deepStrictEqual(stripIds(win), [fileId]);
  win.close();
  console.log(
    `  ok - [${label}] closing a chat closes its sub-agents and keeps its files top-level`,
  );
}

function testArrowKeysStayWithinARow(opts, label) {
  const {win} = makeWebview(opts);
  send(win, {type: 'tabs_state', tabs: [entry('a'), entry('b'), entry('c')]});
  spawnSubagent(win, 'a', 'a-sub-1');
  // Main row: a -> b -> c, End/Home, never onto the strip.
  pressKey(win, mainEl(win, 'a'), 'ArrowRight');
  assert.strictEqual(
    win.document.activeElement,
    mainEl(win, 'b'),
    `${label}: ArrowRight along the main row`,
  );
  pressKey(win, win.document.activeElement, 'End');
  assert.strictEqual(win.document.activeElement, mainEl(win, 'c'));
  pressKey(win, win.document.activeElement, 'ArrowRight');
  assert.strictEqual(
    win.document.activeElement,
    mainEl(win, 'a'),
    `${label}: the main row wraps around`,
  );
  // Strip: a -> a-sub-1 -> a, never onto the main row.
  pressKey(win, stripEl(win, 'a'), 'ArrowRight');
  assert.strictEqual(
    win.document.activeElement,
    stripEl(win, 'a-sub-1'),
    `${label}: ArrowRight along the strip`,
  );
  pressKey(win, win.document.activeElement, 'ArrowRight');
  assert.strictEqual(
    win.document.activeElement,
    stripEl(win, 'a'),
    `${label}: the strip wraps around`,
  );
  // Roving tabindex per row.
  assert.strictEqual(mainEl(win, 'a').getAttribute('tabindex'), '0');
  assert.strictEqual(mainEl(win, 'b').getAttribute('tabindex'), '-1');
  assert.strictEqual(stripEl(win, 'a').getAttribute('tabindex'), '0');
  assert.strictEqual(stripEl(win, 'a-sub-1').getAttribute('tabindex'), '-1');
  pressKey(win, stripEl(win, 'a-sub-1'), 'Enter');
  assert.strictEqual(
    stripActive(win),
    'a-sub-1',
    `${label}: Enter activates a strip entry`,
  );
  assert.strictEqual(stripEl(win, 'a-sub-1').getAttribute('tabindex'), '0');
  assert.strictEqual(
    mainEl(win, 'a').getAttribute('tabindex'),
    '0',
    `${label}: the group's main-row entry stays the Tab stop`,
  );
  win.close();
  console.log(`  ok - [${label}] arrow keys move within one row`);
}

function testBrowserTabIsTopLevel(opts, label) {
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
    mainIds(win),
    ['a', 'browser__1'],
    `${label}: the daemon's browser tab, owned by no chat, is a top-level tab`,
  );
  assert.deepStrictEqual(
    stripIds(win),
    ['a', 'a-sub-1'],
    `${label}: it is not in the chat's group`,
  );
  assert.strictEqual(
    stripActive(win),
    'a',
    `${label}: it opens in the background`,
  );
  click(win, mainEl(win, 'browser__1'), 'browser tab');
  assert.strictEqual(mainActive(win), 'browser__1');
  assert.ok(!stripShown(win));
  win.close();
  console.log(`  ok - [${label}] the daemon's browser tab is a top-level tab`);
}

function testEditorTabsModeHidesTheMainRow() {
  const {win} = makeWebview({editorRoot: 'root-1'});
  assert.ok(
    !mainShown(win),
    "editor-tabs mode: VS Code's editor tabs are the main row",
  );
  assert.ok(
    !stripShown(win),
    'editor-tabs mode: a lone conversation shows no strip',
  );
  spawnSubagent(win, 'root-1', 'root-1-sub');
  assert.ok(!mainShown(win));
  assert.ok(
    stripShown(win),
    'editor-tabs mode: the strip appears with the sub-agent',
  );
  assert.deepStrictEqual(stripIds(win), ['root-1', 'root-1-sub']);
  assert.deepStrictEqual(
    mainIds(win),
    [],
    'editor-tabs mode: nothing is rendered on the main row',
  );
  win.close();
  console.log('  ok - [editor] editor-tabs mode keeps only the strip');
}

function testTabsStateReplaceKeepsGroupBookmarkValid(opts, label) {
  // The bookmark a main-row entry follows must point into its own
  // group; a tab that left (a sub-agent closed by the daemon) is skipped.
  const {win} = makeWebview(opts);
  send(win, {type: 'tabs_state', tabs: [entry('a'), entry('b')]});
  spawnSubagent(win, 'a', 'a-sub-1');
  click(win, stripEl(win, 'a-sub-1'), 'sub-agent');
  click(win, mainEl(win, 'b'), 'chat b');
  send(win, {type: 'closeSubagentTab', tab_id: 'a-sub-1'});
  click(win, mainEl(win, 'a'), 'chat a');
  assert.strictEqual(
    stripActive(win),
    'a',
    `${label}: a bookmark on a closed tab falls back to the chat itself`,
  );
  win.close();
  console.log(
    `  ok - [${label}] a stale group bookmark falls back to the chat`,
  );
}

function main() {
  const surfaces = [
    [{}, 'sidebar'],
    [{remote: true}, 'remote'],
  ];
  for (const [opts, label] of surfaces) {
    testChatsOnMainRowStripHiddenForLoneChat(opts, label);
    testSubagentsAndFilesFormTheGroupStrip(opts, label);
    testBackgroundChatsFilesStayInTheirGroup(opts, label);
    testPendingQuestionOfAGroupShowsOnItsMainRowEntry(opts, label);
    testFinishedSubagentsFileMovesUpToTheChat(opts, label);
    testClosingAChatLeavesItsFilesTopLevel(opts, label);
    testArrowKeysStayWithinARow(opts, label);
    testBrowserTabIsTopLevel(opts, label);
    testTabsStateReplaceKeepsGroupBookmarkValid(opts, label);
  }
  testEditorTabsModeHidesTheMainRow();
  console.log('tabGroupsTwoRows.test.js: all tests passed');
}

main();
