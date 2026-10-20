// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end tests: the welcome screen must always appear scrolled to
// the top.  It lives inside the scrolling chat container (#output), so
// the scroll offset left behind by the conversation it replaces (a
// finished task is parked at its bottom) would otherwise hide the
// greeting and the first suggestions.  Both ways of showing it are
// covered -- the backend's `showWelcome` (new chat) and a tab
// activation that restores a tab whose welcome screen is visible.

'use strict';

const assert = require('assert');
const {makePage: makeChatPage, send} = require('./simplify2_harness');

function makePage() {
  return makeChatPage({
    bodyClass: 'remote-chat',
    placeholders: {PRODUCT_NAME: 'KISS Sorcar'},
  });
}

function fakeGeometry(el, geo) {
  Object.defineProperty(el, 'scrollHeight', {
    get: () => geo.sh,
    configurable: true,
  });
  Object.defineProperty(el, 'clientHeight', {
    get: () => geo.ch,
    configurable: true,
  });
}

const label = 'remote webapp';

// Runs a task in the active tab until its output is parked at the
// bottom of the chat, which is the state a real finished conversation
// leaves behind.  Returns the tab id.
function runScrolledConversation(win, posted, geo) {
  const ready = posted.find(m => m.type === 'ready');
  assert.ok(ready && ready.tabId, 'page must post ready with a tabId');
  win._testApi.hideWelcome();
  send(win, {
    type: 'status',
    running: true,
    tabId: ready.tabId,
    startTs: Date.now() - 2000,
  });
  send(win, {type: 'tool_call', name: 'Bash', command: 'make -j'});
  send(win, {type: 'system_output', text: 'x'.repeat(200) + '\n'});
  send(win, {type: 'result', summary: 'all done', success: true});
  send(win, {type: 'status', running: false, tabId: ready.tabId});
  const O = win.document.getElementById('output');
  assert.strictEqual(
    O.scrollTop,
    Math.max(0, geo.sh - geo.ch),
    'setup: the finished conversation must be parked at the chat bottom',
  );
  return ready.tabId;
}

function assertWelcomeAtTop(win, why) {
  const O = win.document.getElementById('output');
  const welcome = win.document.getElementById('welcome');
  assert.notStrictEqual(
    welcome.style.display,
    'none',
    why + ': the welcome screen must be visible',
  );
  assert.strictEqual(
    welcome.parentNode,
    O,
    why + ': the welcome screen must sit inside the chat container',
  );
  assert.strictEqual(O.scrollTop, 0, why);
}

// --------------------------------------------------------------------
// New chat: the backend's showWelcome replaces a scrolled conversation.
// --------------------------------------------------------------------

async function testShowWelcomeScrollsToTop() {
  const {win, posted} = makePage();
  const O = win.document.getElementById('output');
  const geo = {sh: 3000, ch: 500};
  fakeGeometry(O, geo);
  const tabId = runScrolledConversation(win, posted, geo);

  send(win, {type: 'showWelcome', tabId: tabId});
  assertWelcomeAtTop(
    win,
    'BUG (' +
      label +
      '): showWelcome left the chat scrolled where the replaced ' +
      'conversation was',
  );
  assert.ok(
    O.textContent.includes('Welcome to KISS Sorcar'),
    'the welcome greeting must be back in the chat',
  );
  win.close();
  console.log('  ok - showWelcome lands at the top (' + label + ')');
}

// --------------------------------------------------------------------
// Tab activation: a new tab (and switching back to it) shows the
// welcome screen through restoreTab.
// --------------------------------------------------------------------

async function testWelcomeTabActivationScrollsToTop() {
  const {win, posted} = makePage();
  const O = win.document.getElementById('output');
  const geo = {sh: 3000, ch: 500};
  fakeGeometry(O, geo);
  const chatTabId = runScrolledConversation(win, posted, geo);

  win._testApi.endLaunch();
  win._testApi.createNewTab();
  const welcomeTabId = win._testApi.getActiveTabId();
  assert.notStrictEqual(
    welcomeTabId,
    chatTabId,
    'setup: a new tab must become active',
  );
  assertWelcomeAtTop(
    win,
    'BUG (' +
      label +
      '): a new tab kept the previous chat scroll offset instead of ' +
      'showing the welcome screen from the top',
  );

  // Switching back to the conversation (a Chats-panel pick; chats have no
  // tab row of their own any more) must still land at its end...
  const openIds = () => win._testApi.openTabs().map(t => t.id);
  assert.ok(
    openIds().includes(chatTabId),
    'the conversation chat must still be open',
  );
  win._testApi.switchToTab(chatTabId);
  assert.strictEqual(win._testApi.getActiveTabId(), chatTabId);
  assert.strictEqual(
    O.scrollTop,
    Math.max(0, geo.sh - geo.ch),
    'BUG (' +
      label +
      '): switching back to a conversation no longer lands at its end',
  );

  // ...and switching to the welcome tab again must land at the top.
  assert.ok(
    openIds().includes(welcomeTabId),
    'the welcome chat must still be open',
  );
  win._testApi.switchToTab(welcomeTabId);
  assert.strictEqual(win._testApi.getActiveTabId(), welcomeTabId);
  assertWelcomeAtTop(
    win,
    'BUG (' +
      label +
      '): switching back to a welcome tab kept the conversation scroll ' +
      'offset',
  );
  win.close();
  console.log('  ok - welcome tab activation lands at the top (' + label + ')');
}

async function main() {
  await testShowWelcomeScrollsToTop();
  await testWelcomeTabActivationScrollsToTop();
  console.log('welcomeScrollTop.test.js: all tests passed');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
