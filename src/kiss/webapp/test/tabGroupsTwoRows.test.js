// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) tests for the tab strip (renderTabBar in media/main.js).
// The file name predates the Chats panel: chats are picked there rather
// than from a two-row strip, and the strip only groups what is on screen:
//
//   * on the chat screen the strip (#tab-list) lists the chat with its
//     sub-agents, and appears only when it has more than one entry;
//   * on the mobile file screen it lists the content tabs and nothing
//     else, and is on view even for a single one;
//   * a chat that is not on screen has no entry: it is brought back the
//     way the Chats panel's pick does it (win._testApi.switchToTab);
//   * a swipe to the right on the streamed browser page turns back to
//     the chat only once the daemon says that the page had nothing to
//     scroll that way under the finger.
//
// Covered for the remote webapp (phone layout: no matchMedia, so no
// desktop split).

'use strict';

const assert = require('assert');
const {
  makePage: makeChatPage,
  send,
  click,
  clickTab,
  snapshotEntry,
  tabIds: openIds,
} = require('./simplify2_harness');
const {tabEl: stripEl} = require('./subagent_tab_helpers');

function makePage(extraScripts) {
  return makeChatPage({bodyClass: 'remote-chat', extraScripts});
}

function entry(tabId) {
  return snapshotEntry(tabId, tabId, 'chat-' + tabId);
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

  assert.ok(stripShown(win), 'the file screen has its strip');
  assert.deepStrictEqual(
    stripIds(win),
    [fileId],
    'the file screen lists the file alone: no chat, no sub-agent',
  );
  assert.strictEqual(
    win.document.getElementById('tab-list').getAttribute('aria-label'),
    'Open files',
  );
  // A file opened for the chat on screen comes forward.
  assert.strictEqual(stripActive(win), fileId, 'the file is on screen');
  assert.ok(stripEl(win, fileId).classList.contains('content-tab'));
  clickTab(win, 'a');
  assert.strictEqual(stripActive(win), 'a');
  assert.strictEqual(
    win.document.getElementById('tab-list').getAttribute('aria-label'),
    'Chat tabs',
  );
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
  assert.strictEqual(
    win._testApi.openTabs().find(t => t.id === fileId).rootId,
    'a',
    'the file now belongs to the chat',
  );
  clickTab(win, fileId);
  assert.deepStrictEqual(stripIds(win), [fileId]);
  // Let queued DOM observers finish before destroying their document.
  queueMicrotask(win.close.bind(win));
  console.log("  ok - a finished sub-agent's file moves up to its chat");
}

function testClosingAChatClosesItsSubagentsAndKeepsItsFiles() {
  const {win} = makePage();
  send(win, {type: 'tabs_state', tabs: [entry('a'), entry('b')]});
  spawnSubagent(win, 'a', 'a-sub-1');
  const fileId = openFile(win, 'a', 'kept.md');
  // The chat's own entry is on the chat screen.
  clickTab(win, 'a');
  click(win, stripEl(win, 'a').querySelector('.chat-tab-close'));
  assert.deepStrictEqual(
    openIds(win),
    ['b', fileId],
    "the closed chat's sub-agent went with it; its file stays open",
  );
  assert.strictEqual(stripActive(win), 'b');
  assert.deepStrictEqual(
    stripIds(win),
    ['b'],
    'the orphaned file stays off the mobile chat screen',
  );
  assert.ok(!stripShown(win));
  clickTab(win, fileId);
  assert.strictEqual(stripActive(win), fileId);
  assert.deepStrictEqual(stripIds(win), [fileId]);
  assert.ok(stripShown(win), 'a lone file shows its strip');
  // Let queued DOM observers finish before destroying their document.
  queueMicrotask(win.close.bind(win));
  console.log(
    '  ok - closing a chat closes its sub-agents and keeps its files',
  );
}

// Keyboard navigation along the strip (arrows, Home/End, the roving
// tabindex, Enter) is pinned by tabBarAccessibility.test.js.

/** One finger dragged across *target* from x = *from* to x = *to*;
 *  *during* runs before the finger lifts. */
function drag(win, target, from, to, during) {
  const at = x => [{clientX: x, clientY: 100}];
  const fire = (type, touches, changed) => {
    const e = new win.Event(type, {bubbles: true, cancelable: true});
    Object.defineProperty(e, 'touches', {value: touches});
    Object.defineProperty(e, 'changedTouches', {value: changed});
    target.dispatchEvent(e);
  };
  fire('touchstart', at(from), at(from));
  fire('touchmove', at(to), at(to));
  if (during) during();
  fire('touchend', [], at(to));
}

function testBrowserTabIsListedButOpensInTheBackground() {
  const {win, posted} = makePage(['browserTab.js', 'mobileSwipe.js']);
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
  assert.deepStrictEqual(stripIds(win), ['browser__1']);
  assert.ok(stripShown(win), 'a lone browser tab shows its strip');

  // A drag on the streamed page scrolls the remote page.  A swipe to
  // the right also turns back to the chat once the daemon says that
  // nothing under the finger had anything to scroll that way.
  const page = win.document.querySelector('.browser-screen');
  /** The daemon's word on the finger last put down on *tabId*'s page. */
  const answer = (scrollsLeft, tabId) => {
    const asked = posted
      .filter(m => m.type === 'browserInput' && m.event.kind === 'touch')
      .pop();
    send(win, {
      type: 'browserTouch',
      tab_id: tabId || asked.tab_id,
      id: asked.event.id,
      scrollsLeft,
    });
  };
  drag(win, page, 100, 300);
  assert.strictEqual(stripActive(win), 'browser__1', 'no word yet');
  answer(false, 'browser__404');
  assert.strictEqual(stripActive(win), 'browser__1', 'word on a closed tab');
  answer(true);
  assert.strictEqual(stripActive(win), 'browser__1', 'the page scrolls');
  drag(win, page, 100, 300);
  answer(false);
  assert.strictEqual(stripActive(win), 'a', 'nothing to scroll: the chat');
  clickTab(win, 'browser__1');
  drag(win, page, 100, 300, () => answer(false));
  assert.strictEqual(stripActive(win), 'a', 'said before the finger lifted');
  clickTab(win, 'browser__1');
  // The user moved on to the task info before the daemon answered.
  drag(win, page, 100, 300);
  click(win, win.document.getElementById('meta-drawer-btn'));
  answer(false);
  assert.ok(
    win.document.getElementById('meta-panel').classList.contains('open'),
    'a late word leaves the screen the user went to',
  );
  click(win, win.document.getElementById('meta-drawer-btn'));
  assert.strictEqual(stripActive(win), 'browser__1');
  drag(win, page, 300, 100, () => answer(false));
  assert.strictEqual(stripActive(win), 'browser__1', 'leftwards: the page');
  drag(win, win.document.querySelector('.browser-url'), 100, 300);
  assert.strictEqual(stripActive(win), 'browser__1', 'the address field');
  drag(win, win.document.querySelector('.browser-toolbar'), 100, 300);
  assert.strictEqual(stripActive(win), 'a', 'the toolbar swipes too');
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
  testBrowserTabIsListedButOpensInTheBackground();
  console.log('tabGroupsTwoRows.test.js: all tests passed');
}

main();
