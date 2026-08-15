// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end test for the `activeTabChanged` notification the chat webview
// sends to its host (the VS Code extension, or the remote web app server).
//
// The host keeps the last reported id in `_activeTabId` and uses it to
// attribute tab-scoped requests to the chat tab the user is looking at —
// e.g. routing ghost-text completions (SorcarSidebarView.ts:
// `this._getApi().complete({..., tabId: message.tabId || this._activeTabId
// ...})`).  So the invariant this file pins down is simple and absolute:
//
//   after ANY tab activation, the last `activeTabChanged` the host received
//   must name the chat tab that is actually on screen, and that tab must
//   still exist.
//
// The bug: only `switchToTab()` used to emit.  Creating a tab with `+`
// (`createNewTab`, which assigns `activeTabId` itself) and closing the active
// tab (`closeTab`/`closeContentTab` -> `activateAdjacentTab` -> `restoreTab`)
// both moved the webview without telling the host, leaving `_activeTabId`
// pointing at the previous — or at an outright deleted — tab.
//
// Content tabs are deliberately exempt: `_activeTabId` is only ever compared
// against CHAT tab ids, so viewing a file must not overwrite the host's idea
// of which chat owns the screen.  The tests below encode that too.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

function makeWebview() {
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
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
  win.requestAnimationFrame = function (cb) {
    cb();
    return 0;
  };

  const sent = [];
  win.acquireVsCodeApi = function () {
    let state;
    return {
      postMessage: msg => {
        sent.push(msg);
      },
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };

  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'contentContextMenu.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));

  win._sentMessages = sent;
  return win;
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function tabEl(win, tabId) {
  return win.document.querySelector(
    `.chat-tab[data-tab-id=${JSON.stringify(tabId)}]`,
  );
}

function tabIds(win) {
  return Array.from(
    win.document.querySelectorAll('.chat-tab[data-tab-id]'),
  ).map(el => el.getAttribute('data-tab-id'));
}

function click(win, el, what) {
  assert.ok(el, `cannot click a missing ${what}`);
  el.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

function clickTab(win, tabId) {
  click(win, tabEl(win, tabId), `tab ${tabId}`);
}

function clickNewTabButton(win) {
  click(win, win.document.querySelector('.chat-tab-add'), 'new-tab button');
}

function closeTabByButton(win, tabId) {
  const el = tabEl(win, tabId);
  assert.ok(el, `tab ${tabId} must exist to be closed`);
  click(win, el.querySelector('.chat-tab-close'), `close button of ${tabId}`);
}

function openContentTab(win, name) {
  send(win, {
    type: 'fileContent',
    name: name,
    path: '/tmp/' + name,
    content: '<h1>' + name + '</h1>',
  });
}

// Everything the host learned about which tab is on screen, in order.  The
// initial `ready` seeds `_activeTabId` on the host exactly like an
// `activeTabChanged` would, so it counts as a report.
function reports(win) {
  return win._sentMessages
    .filter(m => m && (m.type === 'activeTabChanged' || m.type === 'ready'))
    .map(m => m.tabId);
}

function lastReport(win) {
  const all = reports(win);
  assert.ok(all.length > 0, 'the host must have been told about a tab');
  return all[all.length - 1];
}

// The one assertion this whole file exists for.
function assertHostIsUpToDate(win, why) {
  const active = win._testApi.getActiveTabId();
  const reported = lastReport(win);
  assert.ok(
    tabIds(win).indexOf(reported) >= 0,
    `${why}: the host was last told about tab ${reported}, which no longer ` +
      `exists (tabs are ${JSON.stringify(tabIds(win))})`,
  );
  assert.strictEqual(
    reported,
    active,
    `${why}: the host thinks ${reported} is on screen but ${active} is`,
  );
}

// A content tab must not overwrite the host's chat-tab id, but the host's id
// must still name a live chat tab.
function assertHostKeepsChatTab(win, expectedChatTab, why) {
  const reported = lastReport(win);
  assert.strictEqual(
    reported,
    expectedChatTab,
    `${why}: a content tab must not change the host's chat tab ` +
      `(expected ${expectedChatTab}, got ${reported})`,
  );
  assert.ok(
    tabIds(win).indexOf(reported) >= 0,
    `${why}: the host's chat tab ${reported} must still exist`,
  );
}

// Booting the webview tells the host which tab it is showing.
function testInitialLoadReportsTheActiveTab() {
  const win = makeWebview();
  assertHostIsUpToDate(win, 'after initial load');
  const readyMsgs = win._sentMessages.filter(m => m && m.type === 'ready');
  assert.strictEqual(readyMsgs.length, 1, 'exactly one ready on boot');
  win.close();
  console.log('  ok - the initial load reports the active tab');
}

// The `+` button: the most common way to change tabs, and the one that used
// to leave the host pointing at the previous chat.
function testNewTabButtonReportsTheNewTab() {
  const win = makeWebview();
  const first = win._testApi.getActiveTabId();

  clickNewTabButton(win);

  const second = win._testApi.getActiveTabId();
  assert.notStrictEqual(second, first, 'the + button must open a new tab');
  assertHostIsUpToDate(win, 'after clicking the + button');
  win.close();
  console.log('  ok - the + button reports the newly created tab');
}

// The programmatic entry point behind the button.
function testCreateNewTabApiReportsTheNewTab() {
  const win = makeWebview();
  const first = win._testApi.getActiveTabId();

  win._testApi.createNewTab();
  assertHostIsUpToDate(win, 'after _testApi.createNewTab()');

  win._testApi.createNewTab();
  const third = win._testApi.getActiveTabId();
  assert.strictEqual(tabIds(win).length, 3, 'three chat tabs must exist');
  assert.notStrictEqual(third, first);
  assertHostIsUpToDate(win, 'after a second _testApi.createNewTab()');
  win.close();
  console.log('  ok - createNewTab() reports every tab it opens');
}

// Clicking between tabs — the path that already worked — must keep working,
// and must not spam the host with duplicates.
function testClickingTabsReportsEachActivationOnce() {
  const win = makeWebview();
  const first = win._testApi.getActiveTabId();
  win._testApi.createNewTab();
  const second = win._testApi.getActiveTabId();

  const before = reports(win).length;
  clickTab(win, first);
  assertHostIsUpToDate(win, 'after clicking back to the first tab');
  assert.strictEqual(
    reports(win).length,
    before + 1,
    'switching tabs must notify the host exactly once',
  );

  clickTab(win, first);
  assert.strictEqual(
    reports(win).length,
    before + 1,
    're-clicking the tab that is already active must not notify again',
  );

  clickTab(win, second);
  assertHostIsUpToDate(win, 'after clicking the second tab');
  win.close();
  console.log('  ok - clicking tabs reports each activation exactly once');
}

// Closing the active tab used to leave the host naming a deleted tab.
function testClosingTheActiveTabReportsItsReplacement() {
  const win = makeWebview();
  const first = win._testApi.getActiveTabId();
  win._testApi.createNewTab();
  const second = win._testApi.getActiveTabId();

  closeTabByButton(win, second);

  assert.strictEqual(
    win._testApi.getActiveTabId(),
    first,
    'closing the active tab must fall back to its neighbour',
  );
  assert.deepStrictEqual(tabIds(win), [first], 'only the first tab is left');
  assertHostIsUpToDate(win, 'after closing the active tab');
  win.close();
  console.log('  ok - closing the active tab reports its replacement');
}

// Closing a background tab must not move the webview, so it must not move
// the host either.
function testClosingABackgroundTabKeepsTheReport() {
  const win = makeWebview();
  const first = win._testApi.getActiveTabId();
  win._testApi.createNewTab();
  const second = win._testApi.getActiveTabId();

  const before = reports(win).length;
  closeTabByButton(win, first);

  assert.strictEqual(
    win._testApi.getActiveTabId(),
    second,
    'closing a background tab must not switch tabs',
  );
  assert.strictEqual(
    reports(win).length,
    before,
    'closing a background tab must not notify the host',
  );
  assertHostIsUpToDate(win, 'after closing a background tab');
  win.close();
  console.log('  ok - closing a background tab leaves the report alone');
}

// Closing the very last tab auto-creates a fresh one; the host must hear
// about that fresh tab, not about the tab that was just destroyed.
function testClosingTheLastTabReportsTheReplacementTab() {
  const win = makeWebview();
  const only = win._testApi.getActiveTabId();

  closeTabByButton(win, only);

  const now = win._testApi.getActiveTabId();
  assert.notStrictEqual(now, only, 'a replacement tab must be created');
  assert.deepStrictEqual(tabIds(win), [now]);
  assertHostIsUpToDate(win, 'after closing the last remaining tab');
  win.close();
  console.log('  ok - closing the last tab reports the replacement tab');
}

// Opening a file opens a content tab.  Content tabs never own a merge, so
// the host must keep pointing at the chat tab underneath.
function testContentTabDoesNotOverwriteTheChatTab() {
  const win = makeWebview();
  const chat = win._testApi.getActiveTabId();

  openContentTab(win, 'report.html');
  const content = win._testApi.getActiveTabId();
  assert.notStrictEqual(content, chat, 'the file must open in its own tab');
  assertHostKeepsChatTab(win, chat, 'while a content tab is on screen');

  clickTab(win, chat);
  assertHostIsUpToDate(win, 'after returning to the chat tab');
  win.close();
  console.log('  ok - a content tab does not overwrite the chat tab');
}

// Closing a content tab activates its neighbour.  When that neighbour is a
// chat tab the host must be told; the reported id must always be alive.
function testClosingAContentTabReportsTheChatTabItFallsBackTo() {
  const win = makeWebview();
  const chat = win._testApi.getActiveTabId();

  openContentTab(win, 'report.html');
  const content = win._testApi.getActiveTabId();

  closeTabByButton(win, content);

  assert.strictEqual(
    win._testApi.getActiveTabId(),
    chat,
    'closing the content tab must fall back to the chat tab',
  );
  assert.deepStrictEqual(tabIds(win), [chat]);
  assertHostIsUpToDate(win, 'after closing the active content tab');
  win.close();
  console.log('  ok - closing a content tab reports the chat tab beneath it');
}

// Closing a content tab whose neighbour is another content tab must leave
// the host on the chat tab it already knew about — and never on the tab that
// was just deleted.
function testClosingAContentTabNextToAnotherContentTab() {
  const win = makeWebview();
  const chat = win._testApi.getActiveTabId();

  openContentTab(win, 'one.html');
  openContentTab(win, 'two.html');
  const second = win._testApi.getActiveTabId();

  closeTabByButton(win, second);

  assert.notStrictEqual(
    win._testApi.getActiveTabId(),
    second,
    'the closed content tab must not stay active',
  );
  assertHostKeepsChatTab(win, chat, 'after closing one of two content tabs');
  win.close();
  console.log('  ok - content-tab churn never strands the host');
}

// Closing the last remaining tab when it is a content tab creates a fresh
// chat tab; the host must be told about that chat tab.
function testClosingTheLastContentTabReportsTheNewChatTab() {
  const win = makeWebview();
  const chat = win._testApi.getActiveTabId();

  openContentTab(win, 'solo.html');
  const content = win._testApi.getActiveTabId();
  closeTabByButton(win, chat);
  assert.deepStrictEqual(tabIds(win), [content], 'only the file tab is left');

  closeTabByButton(win, content);

  const fresh = win._testApi.getActiveTabId();
  assert.notStrictEqual(fresh, content);
  assert.notStrictEqual(fresh, chat);
  assert.deepStrictEqual(tabIds(win), [fresh]);
  assertHostIsUpToDate(win, 'after closing the last (content) tab');
  win.close();
  console.log('  ok - closing the last content tab reports the new chat tab');
}

// Closing the only chat tab while a content tab stays on screen leaves no
// chat tab at all.  There is then nothing to name, so the host must be told
// to forget its chat tab outright: it may never keep matching merges against
// the chat the user just deleted.  The next chat that appears must be
// reported as usual.
function testClosingTheOnlyChatTabClearsTheHostsChatTab() {
  const win = makeWebview();
  const chat = win._testApi.getActiveTabId();

  openContentTab(win, 'solo.html');
  const content = win._testApi.getActiveTabId();

  closeTabByButton(win, chat);

  assert.deepStrictEqual(tabIds(win), [content], 'only the file tab is left');
  assert.strictEqual(
    win._testApi.getActiveTabId(),
    content,
    'the file tab must stay on screen',
  );
  const reported = lastReport(win);
  assert.strictEqual(
    reported,
    '',
    'with no chat tab left the host must be told to forget its chat tab ' +
      `(it was last told ${JSON.stringify(reported)}, and ${JSON.stringify(
        chat,
      )} has been deleted)`,
  );

  // A new chat must reach the host even though the previous report was the
  // empty string.
  clickNewTabButton(win);
  const fresh = win._testApi.getActiveTabId();
  assert.notStrictEqual(fresh, chat);
  assertHostIsUpToDate(win, 'after opening a chat tab again');
  win.close();
  console.log('  ok - losing the last chat tab clears the host\u2019s chat tab');
}

// A whole session of mixed activity: whatever the user does, the host's idea
// of the on-screen chat tab must never name a tab that has been deleted.
function testTheHostNeverNamesADeadTabDuringAMixedSession() {
  const win = makeWebview();
  const first = win._testApi.getActiveTabId();

  const alive = () => {
    assert.ok(
      tabIds(win).indexOf(lastReport(win)) >= 0,
      'the host must never name a deleted tab; it named ' + lastReport(win),
    );
  };

  clickNewTabButton(win);
  alive();
  const second = win._testApi.getActiveTabId();
  openContentTab(win, 'mixed.html');
  alive();
  const content = win._testApi.getActiveTabId();
  clickTab(win, first);
  alive();
  assertHostIsUpToDate(win, 'mixed session: back on the first tab');
  clickTab(win, content);
  alive();
  assertHostKeepsChatTab(win, first, 'mixed session: viewing the file');
  clickTab(win, second);
  alive();
  assertHostIsUpToDate(win, 'mixed session: on the second tab');
  closeTabByButton(win, second);
  alive();
  // The content tab is what lands on screen, so the host must fall back to
  // the surviving chat tab rather than keep naming the tab just deleted.
  assert.strictEqual(
    win._testApi.getActiveTabId(),
    content,
    'closing the second tab must leave the file tab on screen',
  );
  assertHostKeepsChatTab(win, first, 'mixed session: after closing a chat tab');
  closeTabByButton(win, content);
  alive();
  assertHostIsUpToDate(win, 'mixed session: back on the surviving chat tab');
  closeTabByButton(win, win._testApi.getActiveTabId());
  alive();
  assertHostIsUpToDate(win, 'mixed session: after closing everything');

  win.close();
  console.log('  ok - a mixed session never strands the host on a dead tab');
}

function runTests() {
  testInitialLoadReportsTheActiveTab();
  testNewTabButtonReportsTheNewTab();
  testCreateNewTabApiReportsTheNewTab();
  testClickingTabsReportsEachActivationOnce();
  testClosingTheActiveTabReportsItsReplacement();
  testClosingABackgroundTabKeepsTheReport();
  testClosingTheLastTabReportsTheReplacementTab();
  testContentTabDoesNotOverwriteTheChatTab();
  testClosingAContentTabReportsTheChatTabItFallsBackTo();
  testClosingAContentTabNextToAnotherContentTab();
  testClosingTheLastContentTabReportsTheNewChatTab();
  testClosingTheOnlyChatTabClearsTheHostsChatTab();
  testTheHostNeverNamesADeadTabDuringAMixedSession();
}

try {
  runTests();
  console.log('\n13 passed, 0 failed');
  process.exit(0);
} catch (err) {
  console.error('FAIL:', err && err.message ? err.message : err);
  process.exit(1);
}
