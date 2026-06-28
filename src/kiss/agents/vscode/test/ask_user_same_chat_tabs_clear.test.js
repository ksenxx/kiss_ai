// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end regression test: when the same backend chat/task is open
// in multiple local webview tabs, answering an ask-user prompt in any
// one of those tabs must close the ask-user modal for every local tab
// that has the same backend chat id.  Otherwise switching to a sibling
// tab after answering exposes a stale ask window whose answer would be
// submitted to an already-resolved question.
//
// Run directly with ``node``:
//
//     node src/kiss/agents/vscode/test/ask_user_same_chat_tabs_clear.test.js

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

/**
 * Build a jsdom window running the production chat webview: the real
 * ``chat.html`` body (placeholders blanked), ``panelCopy.js`` and
 * ``main.js`` evaluated in the window, and a recording
 * ``acquireVsCodeApi`` stub (the only host API the webview has).
 */
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

  const posted = [];
  win.acquireVsCodeApi = function () {
    let state;
    return {
      postMessage: msg => posted.push(msg),
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };

  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));

  return {win, posted};
}

/** Dispatch a backend→webview event exactly like the extension does. */
function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

/** Click a real tab-bar element to drive the production switchToTab flow. */
function clickTab(win, tabId) {
  const tabEl = win.document.querySelector(
    `.chat-tab[data-tab-id="${tabId}"]`,
  );
  assert.ok(tabEl, `tab ${tabId} must exist in the tab bar`);
  tabEl.click();
}

function visibleAskText(win) {
  const modal = win.document.getElementById('ask-user-modal');
  if (!modal || modal.style.display !== 'flex') return '';
  return modal.textContent || '';
}

function testAnswerClearsSiblingTabsWithSameBackendChatId() {
  const {win, posted} = makeWebview();
  const api = win._demoApi;
  assert.ok(api, '_demoApi must be exposed by main.js');

  const firstTab = api.getActiveTabId();
  assert.ok(firstTab, 'initial tab id must exist');

  send(win, {type: 'clear', chat_id: 'shared-chai-id', tabId: firstTab});
  api.createNewTab();
  const secondTab = api.getActiveTabId();
  assert.ok(secondTab && secondTab !== firstTab, 'second tab must be active');
  send(win, {type: 'clear', chat_id: 'shared-chai-id', tabId: secondTab});

  send(win, {
    type: 'askUser',
    question: 'Question from the shared chat?',
    tabId: firstTab,
  });
  assert.strictEqual(api.getActiveTabId(), firstTab, 'first ask switches active tab');
  assert.ok(
    visibleAskText(win).includes('Question from the shared chat?'),
    'first tab must show its ask-user prompt',
  );

  send(win, {
    type: 'askUser',
    question: 'Same shared chat question in sibling tab?',
    tabId: secondTab,
  });
  assert.strictEqual(api.getActiveTabId(), secondTab, 'second ask switches active tab');

  const modal = win.document.getElementById('ask-user-modal');
  const input = modal.querySelector('.ask-user-input');
  assert.ok(input, 'ask-user input must be mounted for the active tab');
  input.value = 'yes, proceed';
  modal.querySelector('.ask-user-submit').click();

  assert.ok(
    posted.some(
      msg =>
        msg.type === 'userAnswer' &&
        msg.tabId === secondTab &&
        msg.answer === 'yes, proceed',
    ),
    'submitting in the sibling tab must post the answer for that tab',
  );
  assert.notStrictEqual(
    modal.style.display,
    'flex',
    'the answering tab ask-user modal must close immediately',
  );

  clickTab(win, firstTab);
  assert.strictEqual(api.getActiveTabId(), firstTab, 'clicking first tab must switch back');
  assert.notStrictEqual(
    modal.style.display,
    'flex',
    'BUG: answering in one tab must close stale ask windows in all tabs with the same chai/chat id',
  );
  assert.strictEqual(
    visibleAskText(win),
    '',
    'no stale sibling ask-user prompt may be remounted after switching tabs',
  );

  win.close();
  console.log('  ok - answer clears ask windows for same-chai-id sibling tabs');
}

function testAnswerKeepsDifferentBackendChatIdPromptOpen() {
  const {win} = makeWebview();
  const api = win._demoApi;
  const firstTab = api.getActiveTabId();

  send(win, {type: 'clear', chat_id: 'chat-a', tabId: firstTab});
  api.createNewTab();
  const secondTab = api.getActiveTabId();
  send(win, {type: 'clear', chat_id: 'chat-b', tabId: secondTab});

  send(win, {
    type: 'askUser',
    question: 'Question for chat A',
    tabId: firstTab,
  });
  send(win, {
    type: 'askUser',
    question: 'Question for chat B',
    tabId: secondTab,
  });

  const modal = win.document.getElementById('ask-user-modal');
  const input = modal.querySelector('.ask-user-input');
  input.value = 'answer B';
  modal.querySelector('.ask-user-submit').click();

  clickTab(win, firstTab);
  assert.strictEqual(
    modal.style.display,
    'flex',
    'answering a different backend chat id must not close this tab prompt',
  );
  assert.ok(
    visibleAskText(win).includes('Question for chat A'),
    'the unrelated chat prompt should still be visible after switching back',
  );

  win.close();
  console.log('  ok - answer does not clear ask windows for different chat ids');
}

function runTests() {
  testAnswerClearsSiblingTabsWithSameBackendChatId();
  testAnswerKeepsDifferentBackendChatIdPromptOpen();
}

try {
  runTests();
  console.log('\n2 passed, 0 failed');
  process.exit(0);
} catch (err) {
  console.error('FAIL:', err && err.message ? err.message : err);
  process.exit(1);
}
