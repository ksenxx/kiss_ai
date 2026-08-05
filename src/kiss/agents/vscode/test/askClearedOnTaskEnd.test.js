// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// A question is only answerable while its task is alive. The server emits
// `askUserDone` only after an answer is accepted, so a task that ends while a
// question is outstanding (error, stop, interrupt, or a plain done) would
// otherwise leave a dead modal on screen and a '?' badge that never clears.

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

  win.acquireVsCodeApi = function () {
    let state;
    return {
      postMessage: () => {},
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };

  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));

  return win;
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function tabElement(win, tabId) {
  return win.document.querySelector(
    `.chat-tab[data-tab-id=${JSON.stringify(tabId)}]`,
  );
}

function clickTab(win, tabId) {
  const el = tabElement(win, tabId);
  assert.ok(el, `tab ${tabId} must exist in the tab bar`);
  el.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

function askModalVisible(win) {
  const modal = win.document.getElementById('ask-user-modal');
  return !!modal && modal.style.display === 'flex';
}

function attentionGlyph(win, tabId) {
  const el = tabElement(win, tabId);
  assert.ok(el, `tab ${tabId} must exist in the tab bar`);
  const marker = el.querySelector('.chat-tab-attention');
  return marker ? marker.textContent : '';
}

const TERMINAL_EVENTS = [
  'task_done',
  'task_error',
  'task_interrupted',
  'task_stopped',
];

// A task that ends while its own tab is on screen must take its modal away.
function testActiveTabAskClearedOnEachTerminalEvent() {
  for (const type of TERMINAL_EVENTS) {
    const win = makeWebview();
    const api = win._testApi;
    const tabId = api.getActiveTabId();

    send(win, {type: 'askUser', question: 'Which branch?', tabId: tabId});
    assert.ok(
      askModalVisible(win),
      `${type}: the question must be showing before the task ends`,
    );

    send(win, {type: type, tabId: tabId, success: true});

    assert.strictEqual(
      askModalVisible(win),
      false,
      `${type} must take the dead question modal away`,
    );
    assert.strictEqual(
      attentionGlyph(win, tabId),
      '',
      `${type} must clear the attention indicator`,
    );

    win.close();
  }
  console.log('  ok - an ending task clears its own visible question');
}

// The badge on a background tab must clear too, and the user must not be
// left with a stale question when they later visit that tab.
function testBackgroundAskClearedOnEachTerminalEvent() {
  for (const type of TERMINAL_EVENTS) {
    const win = makeWebview();
    const api = win._testApi;
    const askTab = api.getActiveTabId();
    api.createNewTab();
    const userTab = api.getActiveTabId();

    send(win, {type: 'askUser', question: 'Which branch?', tabId: askTab});
    assert.strictEqual(
      attentionGlyph(win, askTab),
      '?',
      `${type}: the waiting background tab must be flagged first`,
    );

    send(win, {type: type, tabId: askTab, success: true});

    assert.strictEqual(
      attentionGlyph(win, askTab),
      '',
      `${type} must clear the background tab's attention indicator`,
    );
    assert.strictEqual(
      askModalVisible(win),
      false,
      `${type} must not pop a modal over the tab the user is on`,
    );

    clickTab(win, askTab);
    assert.strictEqual(
      askModalVisible(win),
      false,
      `${type}: visiting the finished tab must not resurrect the question`,
    );
    assert.notStrictEqual(
      api.getActiveTabId(),
      userTab,
      `${type}: the click must have switched to the finished tab`,
    );

    win.close();
  }
  console.log('  ok - an ending task clears its background question badge');
}

// Ending one task must not silence a question that another chat is still
// waiting on.
function testUnrelatedTabKeepsItsQuestion() {
  const win = makeWebview();
  const api = win._testApi;
  const askTab = api.getActiveTabId();
  api.createNewTab();
  const otherTab = api.getActiveTabId();
  api.createNewTab();

  send(win, {type: 'askUser', question: 'Which branch?', tabId: askTab});
  send(win, {type: 'askUser', question: 'Which remote?', tabId: otherTab});

  send(win, {type: 'task_done', tabId: askTab, success: true});

  assert.strictEqual(
    attentionGlyph(win, askTab),
    '',
    "the finished task's badge must clear",
  );
  assert.strictEqual(
    attentionGlyph(win, otherTab),
    '?',
    'an unrelated waiting task must keep its badge',
  );
  // A task that has FINISHED is allowed to pull focus - that is the one
  // case the no-auto-switch rule deliberately keeps.
  assert.strictEqual(
    api.getActiveTabId(),
    askTab,
    'a finished task may still be focused',
  );

  clickTab(win, otherTab);
  assert.ok(
    askModalVisible(win),
    'the unrelated question must still be answerable',
  );

  win.close();
  console.log('  ok - ending one task leaves other questions alone');
}

// Two tabs can be bound to the same backend chat (history click, resumed
// session). Each still runs its own task, so the end of ONE task may only
// retire that task's question - the sibling's question is still live and the
// backend is still blocked on it.
function testSameChatIdSiblingKeepsItsQuestion() {
  const win = makeWebview();
  const api = win._testApi;
  const doneTab = api.getActiveTabId();
  send(win, {type: 'clear', chat_id: 'shared-chat-id', tabId: doneTab});
  api.createNewTab();
  const siblingTab = api.getActiveTabId();
  send(win, {type: 'clear', chat_id: 'shared-chat-id', tabId: siblingTab});
  // The user sits on a third, uninvolved tab so nothing here is on screen.
  api.createNewTab();

  send(win, {type: 'askUser', question: 'Which branch?', tabId: doneTab});
  send(win, {type: 'askUser', question: 'Which remote?', tabId: siblingTab});
  assert.strictEqual(
    attentionGlyph(win, siblingTab),
    '?',
    'the sibling tab must be flagged before the other task ends',
  );

  send(win, {type: 'task_done', tabId: doneTab, success: true});

  assert.strictEqual(
    attentionGlyph(win, doneTab),
    '',
    "the finished task's own question must be retired",
  );
  assert.strictEqual(
    attentionGlyph(win, siblingTab),
    '?',
    'a sibling tab sharing the backend chat id must keep its own question',
  );

  clickTab(win, siblingTab);
  assert.ok(
    askModalVisible(win),
    "the sibling's question must still be answerable",
  );
  const modal = win.document.getElementById('ask-user-modal');
  assert.ok(
    modal.textContent.includes('Which remote?'),
    'the sibling must still show its own question text',
  );

  clickTab(win, doneTab);
  assert.strictEqual(
    askModalVisible(win),
    false,
    'the finished tab must not resurrect its dead question',
  );

  win.close();
  console.log('  ok - ending one task keeps a same-chat sibling question');
}

// Submitting an answer is a property of the backend CHAT, not of one task, so
// it must still retire the prompt in every tab bound to that chat.
function testSubmittingAnswerStillClearsSameChatIdSiblings() {
  const win = makeWebview();
  const api = win._testApi;
  const firstTab = api.getActiveTabId();
  send(win, {type: 'clear', chat_id: 'shared-chat-id', tabId: firstTab});
  api.createNewTab();
  const secondTab = api.getActiveTabId();
  send(win, {type: 'clear', chat_id: 'shared-chat-id', tabId: secondTab});

  send(win, {type: 'askUser', question: 'Which branch?', tabId: firstTab});
  send(win, {type: 'askUser', question: 'Which remote?', tabId: secondTab});

  const modal = win.document.getElementById('ask-user-modal');
  modal.querySelector('.ask-user-input').value = 'origin';
  modal.querySelector('.ask-user-submit').click();

  assert.strictEqual(
    attentionGlyph(win, firstTab),
    '',
    'answering must still clear a sibling tab sharing the backend chat id',
  );
  clickTab(win, firstTab);
  assert.strictEqual(
    askModalVisible(win),
    false,
    'no stale sibling prompt may be remounted after an accepted answer',
  );

  win.close();
  console.log('  ok - answering still clears same-chat sibling prompts');
}

// A terminal event for a tab with no outstanding question changes nothing.
function testTerminalEventWithoutQuestionIsHarmless() {
  const win = makeWebview();
  const api = win._testApi;
  const askTab = api.getActiveTabId();
  api.createNewTab();
  const userTab = api.getActiveTabId();

  send(win, {type: 'askUser', question: 'Which branch?', tabId: askTab});
  send(win, {type: 'task_done', tabId: userTab, success: true});

  assert.strictEqual(
    attentionGlyph(win, askTab),
    '?',
    'a terminal event for another tab must not clear this badge',
  );
  assert.strictEqual(
    askModalVisible(win),
    false,
    'no modal may appear for a tab with no question',
  );

  send(win, {type: 'task_done', success: true});
  assert.strictEqual(
    attentionGlyph(win, askTab),
    '?',
    'a terminal event without a tabId must not clear another tab',
  );

  win.close();
  console.log('  ok - terminal events without a question are harmless');
}

function runTests() {
  testActiveTabAskClearedOnEachTerminalEvent();
  testBackgroundAskClearedOnEachTerminalEvent();
  testUnrelatedTabKeepsItsQuestion();
  testSameChatIdSiblingKeepsItsQuestion();
  testSubmittingAnswerStillClearsSameChatIdSiblings();
  testTerminalEventWithoutQuestionIsHarmless();
}

try {
  runTests();
  console.log('\n6 passed, 0 failed');
  process.exit(0);
} catch (err) {
  console.error('FAIL:', err && err.message ? err.message : err);
  process.exit(1);
}
