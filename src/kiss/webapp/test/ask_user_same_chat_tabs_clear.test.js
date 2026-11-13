// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const {
  makePage,
  send,
  stubRaf,
  answering,
  clickTab,
} = require('./simplify2_harness');
const {typeAndSend} = require('./attachment_helpers');

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

function testAnswerClearsSiblingTabsWithSameBackendChatId() {
  const {win, posted} = makePage({beforeScripts: stubRaf});
  const api = win._testApi;
  assert.ok(api, '_testApi must be exposed by main.js');

  const firstTab = api.getActiveTabId();
  assert.ok(firstTab, 'initial tab id must exist');

  send(win, {type: 'clear', chat_id: 'shared-chat-id', tabId: firstTab});
  api.createNewTab();
  const secondTab = api.getActiveTabId();
  assert.ok(secondTab && secondTab !== firstTab, 'second tab must be active');
  send(win, {type: 'clear', chat_id: 'shared-chat-id', tabId: secondTab});

  // Both tabs show the same chat asking the same question (the daemon's
  // question id tells it apart from another ask of the same words):
  // answering it in one tab answers it in the other.
  send(win, {
    type: 'askUser',
    question: 'Question from the shared chat?',
    questionId: 'q-shared',
    tabId: firstTab,
  });
  assert.strictEqual(
    api.getActiveTabId(),
    firstTab,
    'a question switches the client to its tab',
  );
  // Back on the sibling tab: it shares the backend chat but has no
  // question of its own.
  clickTab(win, secondTab);
  assert.strictEqual(api.getActiveTabId(), secondTab);
  assert.ok(
    !answering(win),
    'a background ask must not put the active tab composer in answer mode',
  );

  send(win, {
    type: 'askUser',
    question: 'Question from the shared chat?',
    questionId: 'q-shared',
    tabId: secondTab,
  });
  assert.strictEqual(
    api.getActiveTabId(),
    secondTab,
    'an ask for the active tab keeps it active',
  );

  assert.ok(answering(win), 'the active tab composer must be in answer mode');
  typeAndSend(win, 'yes, proceed');

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
  const firstRec = api.openTabs().find(t => t.id === firstTab);
  assert.ok(
    firstRec && !firstRec.askPending,
    'answering must also drop the "?" attention mark of the off-screen sibling',
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

  win.close();
  console.log('  ok - answer clears ask windows for same-chat-id sibling tabs');
}

function testAnswerKeepsDifferentBackendChatIdPromptOpen() {
  const {win} = makePage({beforeScripts: stubRaf});
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

  typeAndSend(win, 'answer B');

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

// Audit 2026-11-08 (W7 F3): two sub-agents of one task share its chat
// id but ask their own questions, each waiting on its own answer: the
// answer to one (given here, or on another surface: askUserDone names
// its tab) leaves the other asking.
function testAnswerKeepsSiblingsAskingTheirOwnQuestions() {
  const {win, posted} = makePage({beforeScripts: stubRaf});
  const api = win._testApi;
  const childA = api.getActiveTabId();
  send(win, {type: 'clear', chat_id: 'shared-chat', tabId: childA});
  api.createNewTab();
  const childB = api.getActiveTabId();
  send(win, {type: 'clear', chat_id: 'shared-chat', tabId: childB});
  askQuestionCall(win, 'Deploy A?', childA);
  send(win, {type: 'askUser', question: 'Deploy A?', questionId: 'q-a', tabId: childA});
  askQuestionCall(win, 'Deploy B?', childB);
  send(win, {type: 'askUser', question: 'Deploy B?', questionId: 'q-b', tabId: childB});
  const askPending = id => !!api.openTabs().find(t => t.id === id).askPending;
  assert.ok(askPending(childA) && askPending(childB), 'both children ask');

  // Another surface answered A: only A's question is done.
  send(win, {type: 'askUserDone', tabId: childA});
  assert.ok(!askPending(childA), 'A is answered');
  assert.ok(
    askPending(childB),
    'BUG - answering one sub-agent retired its sibling\'s own question',
  );
  clickTab(win, childB);
  assert.ok(answering(win), 'B is still being answered');
  assert.ok(visibleAskText(win).includes('Deploy B?'));

  // A asks again; answering B here leaves A asking.
  askQuestionCall(win, 'Deploy A again?', childA);
  send(win, {type: 'askUser', question: 'Deploy A again?', questionId: 'q-a2', tabId: childA});
  clickTab(win, childB);
  typeAndSend(win, 'yes for B');
  assert.ok(
    posted.some(m => m.type === 'userAnswer' && m.tabId === childB && m.questionId === 'q-b'),
    'the answer names B\'s question',
  );
  assert.ok(!answering(win), 'B leaves answer mode');
  assert.ok(askPending(childA), 'A still asks');
  clickTab(win, childA);
  assert.ok(answering(win) && visibleAskText(win).includes('Deploy A again?'));

  win.close();
  console.log('  ok - an answer leaves siblings asking their own questions');
}

function runTests() {
  testAnswerClearsSiblingTabsWithSameBackendChatId();
  testAnswerKeepsDifferentBackendChatIdPromptOpen();
  testAnswerKeepsSiblingsAskingTheirOwnQuestions();
}

try {
  runTests();
  console.log('\n3 passed, 0 failed');
  process.exit(0);
} catch (err) {
  console.error('FAIL:', err && err.message ? err.message : err);
  process.exit(1);
}
