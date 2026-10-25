// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) tests for the tab strip (renderTabBar in media/main.js).
// The file name predates the Chats panel: chats are picked there rather
// than from a two-row strip, and the strip only groups what is on screen:
//
//   * the strip (#tab-list) lists the chat on screen with its sub-agents;
//   * content tabs share that strip on the mobile file screen, but stay
//     off the mobile chat screen;
//   * it appears only when it has more than one entry;
//   * a chat that is not on screen has no entry: it is brought back the
//     way the Chats panel's pick does it (win._testApi.switchToTab).
//
// Covered for the remote webapp (phone layout: no matchMedia, so no
// desktop split).

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {
  makePage: makeChatPage,
  send,
  click,
  clickTab,
  MEDIA,
} = require('./simplify2_harness');

function makePage() {
  return makeChatPage({bodyClass: 'remote-chat'});
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
  // A background file remains open even when the mobile chat omits its tab.
  const tab = win._testApi
    .openTabs()
    .find(t => t.isContentTab && t.title === name);
  return tab ? tab.id : null;
}

// ---- Tests ---------------------------------------------------------

function testLoneChatShowsNoStrip() {
  const {win} = makePage();
  send(win, {type: 'tabs_state', tabs: [entry('a'), entry('b')]});
  assert.deepStrictEqual(openIds(win), ['a', 'b'], 'both chats open');
  assert.ok(!stripShown(win), 'a lone chat needs no strip');
  assert.deepStrictEqual(
    stripIds(win),
    ['a'],
    'the strip holds the chat on screen only',
  );
  assert.strictEqual(stripActive(win), 'a');
  assert.strictEqual(
    stripEl(win, 'a').getAttribute('aria-selected'),
    'true',
    'aria-selected follows the highlight',
  );
  assert.strictEqual(
    win.document.getElementById('tab-list').getAttribute('role'),
    'tablist',
  );
  assert.strictEqual(
    stripEl(win, 'b'),
    null,
    'a chat that is not on screen has no strip entry',
  );

  clickTab(win, 'b');
  assert.strictEqual(
    stripActive(win),
    'b',
    "the Chats panel's pick switches chats",
  );
  assert.deepStrictEqual(stripIds(win), ['b']);
  assert.ok(!stripShown(win));
  // Let queued DOM observers finish before destroying their document.
  queueMicrotask(win.close.bind(win));
  console.log('  ok - a lone chat shows no strip');
}

function testSubagentsAndFilesFormTheGroupStrip() {
  const {win} = makePage();
  send(win, {type: 'tabs_state', tabs: [entry('a'), entry('b')]});
  spawnSubagent(win, 'a', 'a-sub-1', 'index the code');
  const fileId = openFile(win, 'a', 'notes.md');
  assert.ok(fileId, 'the file opened as a content tab');

  assert.ok(stripShown(win), 'the strip appears once the group grows');
  assert.deepStrictEqual(
    stripIds(win),
    ['a', 'a-sub-1', fileId],
    'the file screen lists the chat, its sub-agent and its file',
  );
  // A file opened for the chat on screen comes forward.
  assert.strictEqual(stripActive(win), fileId, 'the file is on screen');
  assert.ok(stripEl(win, fileId).classList.contains('content-tab'));
  click(win, stripEl(win, 'a'));
  assert.strictEqual(stripActive(win), 'a');
  assert.ok(stripEl(win, 'a-sub-1').classList.contains('subagent-tab'));
  assert.deepStrictEqual(
    stripIds(win),
    ['a', 'a-sub-1'],
    'mobile chat shows its sub-agent but no editor tabs',
  );

  click(win, stripEl(win, 'a-sub-1'));
  assert.strictEqual(
    stripActive(win),
    'a-sub-1',
    'the strip highlights the tab on screen',
  );
  assert.strictEqual(
    stripEl(win, 'a-sub-1').getAttribute('aria-selected'),
    'true',
  );
  assert.strictEqual(stripEl(win, 'a').getAttribute('aria-selected'), 'false');
  assert.deepStrictEqual(stripIds(win), ['a', 'a-sub-1']);

  // Another chat: its own group replaces a's; content tabs stay off
  // the mobile chat screen even when they belong to a different chat.
  clickTab(win, 'b');
  assert.strictEqual(stripActive(win), 'b');
  assert.deepStrictEqual(
    stripIds(win),
    ['b'],
    "chat b's strip excludes other chats' sub-agents",
  );
  assert.ok(!stripShown(win));

  // Back to a: the pick lands on the chat itself.
  clickTab(win, 'a');
  assert.strictEqual(
    stripActive(win),
    'a',
    'picking the chat brings the conversation back',
  );
  assert.deepStrictEqual(stripIds(win), ['a', 'a-sub-1']);
  assert.strictEqual(
    win.document.getElementById('output').style.display,
    '',
    'the chat surface is on screen',
  );
  // Let queued DOM observers finish before destroying their document.
  queueMicrotask(win.close.bind(win));
  console.log("  ok - sub-agents and files follow the screen's tab strip");
}

function testBackgroundChatsFileNeverStealsFocus() {
  const {win} = makePage();
  send(win, {type: 'tabs_state', tabs: [entry('a'), entry('b')]});
  // b's task opens a file while the user reads a: it stays in the
  // background and must not add an editor tab to the mobile chat.
  const fileId = openFile(win, 'b', 'report.html');
  assert.ok(fileId, 'the background file is open');
  assert.strictEqual(
    stripActive(win),
    'a',
    'a background file never steals focus',
  );
  assert.deepStrictEqual(stripIds(win), ['a']);
  assert.ok(!stripShown(win));

  clickTab(win, 'b');
  assert.strictEqual(
    stripActive(win),
    'b',
    'a never-viewed chat opens on itself',
  );
  assert.deepStrictEqual(stripIds(win), ['b']);
  clickTab(win, fileId);
  assert.ok(stripEl(win, fileId).classList.contains('content-tab'));
  assert.ok(stripEl(win, fileId).textContent.indexOf('report.html') >= 0);
  // Let queued DOM observers finish before destroying their document.
  queueMicrotask(win.close.bind(win));
  console.log("  ok - a background chat's file waits without stealing focus");
}

function testPendingQuestionFlagsTheSubagentEntry() {
  const {win} = makePage();
  send(win, {type: 'tabs_state', tabs: [entry('a'), entry('b')]});
  spawnSubagent(win, 'a', 'a-sub-1');
  // A new question pulls the user onto the asking tab.
  send(win, {type: 'askUser', question: 'Continue?', tabId: 'a-sub-1'});
  assert.strictEqual(stripActive(win), 'a-sub-1');
  assert.ok(
    !win.document.querySelector('.chat-tab-attention'),
    'no flag while the asking tab is on screen',
  );

  click(win, stripEl(win, 'a'));
  assert.ok(
    stripEl(win, 'a-sub-1').querySelector('.chat-tab-attention'),
    'on the strip the sub-agent itself carries the flag',
  );
  assert.ok(!stripEl(win, 'a').querySelector('.chat-tab-attention'));

  // Another chat on screen: the asking sub-agent is off the strip, and
  // the record still says it waits.
  clickTab(win, 'b');
  assert.deepStrictEqual(stripIds(win), ['b']);
  assert.ok(
    win._testApi.openTabs().find(t => t.id === 'a-sub-1').askPending,
    'the sub-agent still waits for its answer',
  );

  clickTab(win, 'a');
  assert.strictEqual(stripActive(win), 'a');
  assert.ok(
    stripEl(win, 'a-sub-1').querySelector('.chat-tab-attention'),
    'back in the group, the flag is on the strip again',
  );
  click(win, stripEl(win, 'a-sub-1'));
  assert.ok(
    !win.document.querySelector('.chat-tab-attention'),
    'the flag goes once the asking tab is on screen',
  );
  // Let queued DOM observers finish before destroying their document.
  queueMicrotask(win.close.bind(win));
  console.log("  ok - a sub-agent's pending question flags its strip entry");
}

function testFinishedSubagentsFileMovesUpToTheChat() {
  const {win} = makePage();
  send(win, {type: 'tabs_state', tabs: [entry('a')]});
  spawnSubagent(win, 'a', 'a-sub-1');
  spawnSubagent(win, 'a-sub-1', 'a-sub-1-sub', 'nested');
  const fileId = openFile(win, 'a-sub-1-sub', 'deep.md');
  assert.deepStrictEqual(stripIds(win), ['a', 'a-sub-1', 'a-sub-1-sub']);
  // The outer sub-agent finishes while the nested one still runs: the
  // nested one moves up under the chat and keeps its file.
  send(win, {type: 'subagentDone', tab_id: 'a-sub-1'});
  assert.deepStrictEqual(
    openIds(win),
    ['a', 'a-sub-1-sub', fileId],
    'only the finished sub-agent is gone',
  );
  assert.deepStrictEqual(
    stripIds(win),
    ['a', 'a-sub-1-sub'],
    "the running nested sub-agent stays in the chat's group",
  );
  send(win, {type: 'subagentDone', tab_id: 'a-sub-1-sub'});
  assert.deepStrictEqual(openIds(win), ['a', fileId]);
  assert.deepStrictEqual(
    stripIds(win),
    ['a'],
    "the finished sub-agent's file stays off the mobile chat",
  );
  clickTab(win, fileId);
  assert.deepStrictEqual(stripIds(win), ['a', fileId]);
  // Let queued DOM observers finish before destroying their document.
  queueMicrotask(win.close.bind(win));
  console.log("  ok - a finished sub-agent's file moves up to its chat");
}

function testClosingAChatClosesItsSubagentsAndKeepsItsFiles() {
  const {win} = makePage();
  send(win, {type: 'tabs_state', tabs: [entry('a'), entry('b')]});
  spawnSubagent(win, 'a', 'a-sub-1');
  const fileId = openFile(win, 'a', 'kept.md');
  click(win, stripEl(win, 'a').querySelector('.chat-tab-close'));
  assert.deepStrictEqual(
    openIds(win),
    ['b', fileId],
    "the closed chat's sub-agent went with it; its file stays open",
  );
  assert.strictEqual(
    stripActive(win),
    fileId,
    'the file the user was reading stays on screen',
  );
  assert.deepStrictEqual(
    stripIds(win),
    [fileId],
    'a top-level file is its own group',
  );
  assert.ok(!stripShown(win));
  clickTab(win, 'b');
  assert.strictEqual(stripActive(win), 'b');
  assert.deepStrictEqual(
    stripIds(win),
    ['b'],
    'the orphaned file stays off the mobile chat screen',
  );
  assert.ok(!stripShown(win));
  clickTab(win, fileId);
  assert.strictEqual(stripActive(win), fileId);
  // Let queued DOM observers finish before destroying their document.
  queueMicrotask(win.close.bind(win));
  console.log(
    '  ok - closing a chat closes its sub-agents and keeps its files',
  );
}

function testArrowKeysMoveAlongTheStrip() {
  const {win} = makePage();
  send(win, {type: 'tabs_state', tabs: [entry('a'), entry('b'), entry('c')]});
  spawnSubagent(win, 'a', 'a-sub-1');
  spawnSubagent(win, 'a', 'a-sub-2');
  // a -> a-sub-1 -> a-sub-2 -> a, End/Home; the other chats are not on
  // the strip, so the keys never reach them.
  pressKey(win, stripEl(win, 'a'), 'ArrowRight');
  assert.strictEqual(
    win.document.activeElement,
    stripEl(win, 'a-sub-1'),
    'ArrowRight along the strip',
  );
  pressKey(win, win.document.activeElement, 'End');
  assert.strictEqual(win.document.activeElement, stripEl(win, 'a-sub-2'));
  pressKey(win, win.document.activeElement, 'ArrowRight');
  assert.strictEqual(
    win.document.activeElement,
    stripEl(win, 'a'),
    'the strip wraps around',
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
    'Enter activates a strip entry',
  );
  assert.strictEqual(stripEl(win, 'a-sub-1').getAttribute('tabindex'), '0');
  // Let queued DOM observers finish before destroying their document.
  queueMicrotask(win.close.bind(win));
  console.log('  ok - arrow keys move along the strip');
}

function testBrowserTabIsListedButOpensInTheBackground() {
  const {win} = makePage();
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
    ['a', 'a-sub-1'],
    'a background browser tab stays off the mobile chat screen',
  );
  assert.strictEqual(stripActive(win), 'a', 'it opens in the background');
  clickTab(win, 'browser__1');
  assert.strictEqual(stripActive(win), 'browser__1');
  assert.deepStrictEqual(
    stripIds(win),
    ['browser__1'],
    'owned by no chat, it is its own group',
  );
  assert.ok(!stripShown(win));
  // Let queued DOM observers finish before destroying their document.
  queueMicrotask(win.close.bind(win));
  console.log("  ok - the daemon's browser tab opens in the background");
}

function main() {
  testLoneChatShowsNoStrip();
  testSubagentsAndFilesFormTheGroupStrip();
  testBackgroundChatsFileNeverStealsFocus();
  testPendingQuestionFlagsTheSubagentEntry();
  testFinishedSubagentsFileMovesUpToTheChat();
  testClosingAChatClosesItsSubagentsAndKeepsItsFiles();
  testArrowKeysMoveAlongTheStrip();
  testBrowserTabIsListedButOpensInTheBackground();
  console.log('tabGroupsTwoRows.test.js: all tests passed');
}

main();
