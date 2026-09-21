// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

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

  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));

  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function clickTab(win, tabId) {
  const tabEl = win.document.querySelector(`.chat-tab[data-tab-id="${tabId}"]`);
  assert.ok(tabEl, `tab ${tabId} must exist in the tab bar`);
  tabEl.click();
}

// The composer is in answer mode while the tab on screen has a question.
function answering(win) {
  return win.document.body.classList.contains('ask-answering');
}

// The Question panel of the tab on screen while it is awaiting an answer.
function visibleAskText(win) {
  if (!answering(win)) return '';
  const panel = win.document.querySelector(
    '#output .tc-question.tc-question-pending',
  );
  return panel ? panel.textContent || '' : '';
}

function askQuestionCall(win, question, tabId) {
  send(win, {
    type: 'tool_call',
    name: 'ask_user_question',
    extras: {question},
    callId: 7,
    tabId,
    ts: Date.now(),
  });
}

function submitAnswer(win, answer) {
  win.document.getElementById('task-input').value = answer;
  win.document.getElementById('send-btn').click();
}

function testAnswerClearsSiblingTabsWithSameBackendChatId() {
  const {win, posted} = makeWebview();
  const api = win._testApi;
  assert.ok(api, '_testApi must be exposed by main.js');

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
  assert.strictEqual(
    api.getActiveTabId(),
    secondTab,
    'a background ask must not steal the active tab',
  );
  assert.strictEqual(
    visibleAskText(win),
    '',
    'a background ask must not put the active tab composer in answer mode',
  );

  send(win, {
    type: 'askUser',
    question: 'Same shared chat question in sibling tab?',
    tabId: secondTab,
  });
  assert.strictEqual(
    api.getActiveTabId(),
    secondTab,
    'an ask for the active tab keeps it active',
  );

  assert.ok(answering(win), 'the active tab composer must be in answer mode');
  submitAnswer(win, 'yes, proceed');

  assert.ok(
    posted.some(
      msg =>
        msg.type === 'userAnswer' &&
        msg.tabId === secondTab &&
        msg.answer === 'yes, proceed',
    ),
    'submitting in the sibling tab must post the answer for that tab',
  );
  assert.ok(
    !answering(win),
    'the answering tab composer must leave answer mode immediately',
  );

  clickTab(win, firstTab);
  assert.strictEqual(
    api.getActiveTabId(),
    firstTab,
    'clicking first tab must switch back',
  );
  assert.ok(
    !answering(win),
    'BUG: answering in one tab must retire stale questions in all tabs with the same chat id',
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
  const api = win._testApi;
  const firstTab = api.getActiveTabId();

  send(win, {type: 'clear', chat_id: 'chat-a', tabId: firstTab});
  api.createNewTab();
  const secondTab = api.getActiveTabId();
  send(win, {type: 'clear', chat_id: 'chat-b', tabId: secondTab});

  askQuestionCall(win, 'Question for chat A', firstTab);
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

  submitAnswer(win, 'answer B');

  clickTab(win, firstTab);
  assert.ok(
    answering(win),
    'answering a different backend chat id must not retire this tab question',
  );
  assert.ok(
    visibleAskText(win).includes('Question for chat A'),
    'the unrelated chat prompt should still be visible after switching back',
  );

  win.close();
  console.log(
    '  ok - answer does not clear ask windows for different chat ids',
  );
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
