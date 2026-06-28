// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end regression test: when a running task in a background chat
// tab asks the user a question, the real webview must switch to that
// tab immediately so the modal is visible and answerable.
//
// Run directly with ``node``:
//
//     node src/kiss/agents/vscode/test/ask_user_switches_to_tab.test.js

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

function testAskUserSwitchesToQuestionTab() {
  const {win, posted} = makeWebview();
  const api = win._demoApi;
  assert.ok(api, '_demoApi must be exposed by main.js');

  const questionTab = api.getActiveTabId();
  assert.ok(questionTab, 'initial tab id must exist');

  // Open a second tab.  The initial tab is now a background tab whose
  // running task is about to ask the user a question.
  api.createNewTab();
  const otherTab = api.getActiveTabId();
  assert.ok(otherTab && otherTab !== questionTab, 'second tab must be active');

  // The background task asks a question.  This must switch the active
  // webview tab to the task's tab, otherwise the modal remains hidden
  // behind the wrong tab and the user does not see the question.
  send(win, {
    type: 'askUser',
    question: 'Please provide the deployment token.',
    tabId: questionTab,
  });

  assert.strictEqual(
    api.getActiveTabId(),
    questionTab,
    'BUG: askUser for a background running task must switch to that tab',
  );

  const activeEl = win.document.querySelector('.chat-tab.active');
  assert.ok(activeEl, 'one tab must be marked active in the DOM');
  assert.strictEqual(
    activeEl.dataset.tabId,
    questionTab,
    'active tab DOM class must move to the question tab',
  );

  const modal = win.document.getElementById('ask-user-modal');
  assert.ok(modal, 'ask-user modal must exist');
  assert.strictEqual(
    modal.style.display,
    'flex',
    'ask-user modal must be visible after switching to the question tab',
  );
  assert.ok(
    modal.textContent.includes('Please provide the deployment token.'),
    'ask-user modal must show the question text',
  );

  const input = modal.querySelector('.ask-user-input');
  assert.ok(input, 'ask-user input must be mounted for the active tab');
  input.value = 'tok_live_123';
  modal.querySelector('.ask-user-submit').click();
  assert.ok(
    posted.some(
      msg =>
        msg.type === 'userAnswer' &&
        msg.tabId === questionTab &&
        msg.answer === 'tok_live_123',
    ),
    'submitting the visible modal must answer the question tab',
  );

  win.close();
  console.log('  ok - askUser switches to the question tab');
}

function testAskUserForUnknownTabIsIgnored() {
  const {win} = makeWebview();
  const api = win._demoApi;
  const activeBefore = api.getActiveTabId();

  send(win, {
    type: 'askUser',
    question: 'This belongs to another VS Code window.',
    tabId: 'foreign-window-tab',
  });

  assert.strictEqual(
    api.getActiveTabId(),
    activeBefore,
    'askUser for an unknown foreign tab must not switch local tabs',
  );
  const modal = win.document.getElementById('ask-user-modal');
  assert.notStrictEqual(
    modal.style.display,
    'flex',
    'askUser for an unknown foreign tab must not show a modal locally',
  );

  win.close();
  console.log('  ok - foreign-window askUser is ignored');
}

function runTests() {
  testAskUserSwitchesToQuestionTab();
  testAskUserForUnknownTabIsIgnored();
}

try {
  runTests();
  console.log('\n2 passed, 0 failed');
  process.exit(0);
} catch (err) {
  console.error('FAIL:', err && err.message ? err.message : err);
  process.exit(1);
}
