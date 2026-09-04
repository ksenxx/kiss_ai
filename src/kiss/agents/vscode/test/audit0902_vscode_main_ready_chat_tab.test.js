// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) tests for which tab the webview reports as the
// chat "on screen" when it re-announces `ready` after a daemon outage
// while a CONTENT tab (an opened report / file) is active.
//
// Every other path that moves that id -- restoreTab, closeTab,
// reportSurvivingChatTab -- names a visible CHAT tab only ("the host
// compares this id against chat tab ids only").  sendReady() mirrored
// `activeTabId` verbatim, so a reconnect while reading a report left
// the content tab as the reported chat: the settings drawer's Git
// Commit then targeted the content tab (whose transcript is the hidden
// shared #output) and the toast filter compared against it.
//
// Review follow-up (review-vscode.md #3): with NO visible chat tab at all
// sendReady() correctly reported '', but autocommitTargetTabId() fell
// back from that '' to activeTabId -- the content tab -- and Git Commit
// posted autocommitAction for it (the daemon accepts any tabId and starts
// the commit job).  Now Git Commit does not send in that state and shows a
// toast explaining that a chat tab must be open.

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
      '\n//# sourceURL=audit0902-readychat-main.js',
  );

  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function readies(posted) {
  return posted.filter(m => m.type === 'ready');
}

function openReport(win, name) {
  send(win, {
    type: 'fileContent',
    path: '/repo/reports/' + name,
    name: name,
    content: '<!DOCTYPE html><html><body>' + name + '</body></html>',
  });
}

function reconnect(win) {
  send(win, {type: 'daemonStatus', connected: false});
  send(win, {type: 'daemonStatus', connected: true});
}

function clickGitCommit(win) {
  const btn = win.document.getElementById('autocommit-btn');
  assert.ok(btn && !btn.disabled, 'the Git Commit button is armed');
  btn.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

function testReadyOnContentTabNamesOwningChat() {
  const {win, posted} = makeWebview();
  const chatTab = win._testApi.getActiveTabId();
  assert.strictEqual(readies(posted).length, 1, 'boot sends one ready');
  assert.strictEqual(readies(posted)[0].tabId, chatTab);

  openReport(win, 'audit.html');
  const contentTab = win._testApi.getActiveTabId();
  assert.notStrictEqual(contentTab, chatTab, 'the report opened in its tab');

  reconnect(win);
  const again = readies(posted);
  assert.strictEqual(again.length, 2, 'the reconnect re-announces ready');
  assert.strictEqual(
    again[1].tabId,
    chatTab,
    'ready names the chat that owns the report, never the content tab',
  );

  // The reported chat is what the settings drawer's Git Commit acts on
  // while a content tab is on screen.
  clickGitCommit(win);
  const commit = posted.filter(m => m.type === 'autocommitAction').pop();
  assert.ok(commit, 'Git Commit posts autocommitAction');
  assert.strictEqual(
    commit.tabId,
    chatTab,
    'the commit targets the owning chat tab, not the content tab',
  );
  win.close();
  console.log('  ok - ready on a content tab names the owning chat');
}

function testReadyOnChatTabIsUnchanged() {
  const {win, posted} = makeWebview();
  const first = win._testApi.getActiveTabId();
  win._testApi.createNewTab();
  const second = win._testApi.getActiveTabId();
  assert.notStrictEqual(first, second);
  reconnect(win);
  const again = readies(posted);
  assert.strictEqual(again.length, 2);
  assert.strictEqual(again[1].tabId, second, 'a chat tab reports itself');
  clickGitCommit(win);
  const commit = posted.filter(m => m.type === 'autocommitAction').pop();
  assert.strictEqual(commit.tabId, second);
  win.close();
  console.log('  ok - ready on a chat tab is unchanged');
}

function testReadyOnOrphanContentTabFallsBackToAnyChat() {
  const {win, posted} = makeWebview();
  const chatA = win._testApi.getActiveTabId();
  win._testApi.createNewTab();
  const chatB = win._testApi.getActiveTabId();
  openReport(win, 'orphan.html');
  const contentTab = win._testApi.getActiveTabId();
  // The owning chat (B) is closed from another client; the content tab
  // survives on screen and B's transcript is gone.  B's own `openTab`
  // shields it from the first snapshots that predate its registration
  // (PENDING_OPEN_MAX_MISSES), so the registry has to miss it thrice.
  for (let i = 0; i < 3; i++) {
    send(win, {
      type: 'tabs_state',
      tabs: [{tabId: chatA, chatId: '', title: 'new chat', workDir: ''}],
    });
  }
  // The registry snapshot never lists content tabs, so the report stays.
  assert.ok(
    win.document.querySelector(
      `.chat-tab[data-tab-id=${JSON.stringify(contentTab)}]`,
    ),
    'the content tab survives the snapshot',
  );
  assert.ok(
    !win.document.querySelector(
      `.chat-tab[data-tab-id=${JSON.stringify(chatB)}]`,
    ),
    'chat B is gone',
  );
  reconnect(win);
  const again = readies(posted);
  assert.strictEqual(
    again[again.length - 1].tabId,
    chatA,
    'with the owner gone, ready falls back to a surviving chat tab',
  );
  win.close();
  console.log('  ok - ready on an orphaned content tab falls back to a chat');
}

function testReadyWithNoVisibleChatReportsNone() {
  const {win, posted} = makeWebview();
  const chatA = win._testApi.getActiveTabId();
  // A is unpinned, so the report it opens is visible in every workspace.
  openReport(win, 'everywhere.html');
  const contentTab = win._testApi.getActiveTabId();
  // The registry replaces the local placeholder A with a chat pinned to
  // another workspace; the content tab (orphaned, scope frozen at '')
  // stays on screen.
  send(win, {
    type: 'tabs_state',
    tabs: [{tabId: 'pinned', chatId: 'c1', title: 'other', workDir: '/other'}],
  });
  assert.ok(
    !win.document.querySelector(
      `.chat-tab[data-tab-id=${JSON.stringify(chatA)}]`,
    ),
    'the placeholder chat is gone',
  );
  assert.strictEqual(win._testApi.getActiveTabId(), contentTab);
  // This window is scoped to /ws: the pinned chat hides, the content
  // tab does not, so no visible chat is left to represent the window.
  send(win, {type: 'workspaceWorkDir', workDir: '/ws'});
  assert.ok(
    !win.document.querySelector('.chat-tab[data-tab-id="pinned"]'),
    'the other workspace\u2019s chat is hidden here',
  );
  const lastChanged = posted.filter(m => m.type === 'activeTabChanged').pop();
  assert.strictEqual(
    lastChanged.tabId,
    '',
    'the host is told no chat tab is on screen',
  );
  reconnect(win);
  const again = readies(posted);
  assert.strictEqual(
    again[again.length - 1].tabId,
    '',
    'ready reports no chat rather than the content tab',
  );

  // Git Commit from the settings drawer has no chat tab to act for: it
  // must not fall back to the content tab, and the user must be told.
  const commitsBefore = posted.filter(m => m.type === 'autocommitAction');
  clickGitCommit(win);
  const commits = posted.filter(m => m.type === 'autocommitAction');
  assert.strictEqual(
    commits.length,
    commitsBefore.length,
    `Git Commit posted autocommitAction for ${JSON.stringify(
      (commits[commits.length - 1] || {}).tabId,
    )} although no chat tab is visible`,
  );
  const toasts = Array.from(
    win.document.querySelectorAll('.kiss-notification-warning'),
  ).map(el => el.textContent);
  assert.strictEqual(toasts.length, 1, `expected one warning toast: ${toasts}`);
  assert.match(toasts[0], /no chat tab/i);
  assert.match(toasts[0], /Git Commit/);
  const commitBtn = win.document.getElementById('autocommit-btn');
  assert.ok(!commitBtn.disabled, 'the button is not left stuck in-flight');
  win.close();
  console.log(
    '  ok - ready with no visible chat reports none and blocks Git Commit',
  );
}

function main() {
  testReadyOnContentTabNamesOwningChat();
  testReadyOnChatTabIsUnchanged();
  testReadyOnOrphanContentTabFallsBackToAnyChat();
  testReadyWithNoVisibleChatReportsNone();
  console.log('all audit0902 ready-chat-tab tests passed');
}

main();
