// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// The daemon names every ask_user_question with a fresh `questionId`
// (server/task_runner.py `_publish_ask`) and drops an answer whose id is
// not the pending ask's (server/commands.py `_cmd_user_answer`).  The
// question text alone cannot tell apart two asks of the same text: a
// late surface's answer to the first would answer the second.  So the
// page sends back the id of the ask it shows, takes a new ask's id even
// when its text is the same, and keeps the id across a replay.

const assert = require('assert');
const {makePage, send, stubRaf, readyTabId} = require('./simplify2_harness');
const {test, report} = require('./ui_antipattern_harness').makeRunner();

const Q = 'Deploy to staging?';

/** A surface showing a running task on its tab. */
function surface() {
  const {win, posted} = makePage({beforeScripts: stubRaf});
  const tab = readyTabId(posted);
  send(win, {type: 'clear', chat_id: 'chat-qid', tabId: tab});
  send(win, {type: 'status', running: true, tabId: tab, startTs: Date.now()});
  send(win, {type: 'daemonStatus', connected: true});
  return {win, posted, tab};
}

/** Type *text* into the composer and press Enter. */
function answer(win, text) {
  const inp = win.document.getElementById('task-input');
  inp.value = text;
  inp.dispatchEvent(
    new win.KeyboardEvent('keydown', {key: 'Enter', bubbles: true}),
  );
}

/** The page's userAnswer frames as plain objects. */
function answers(posted) {
  return JSON.parse(
    JSON.stringify(posted.filter(m => m.type === 'userAnswer')),
  );
}

/** Send the ask *id* of the question Q to surface *s*. */
function ask(s, id) {
  send(s.win, {type: 'askUser', question: Q, questionId: id, tabId: s.tab});
}

test('an answer names the id of the ask it answers', () => {
  const s = surface();
  ask(s, 'id-1');
  answer(s.win, 'yes');
  assert.deepStrictEqual(answers(s.posted), [
    {answer: 'yes', question: Q, questionId: 'id-1', tabId: s.tab, type: 'userAnswer'},
  ]);
  // The agent asks the same text again: the next answer names the new ask.
  ask(s, 'id-2');
  answer(s.win, 'yes again');
  assert.strictEqual(answers(s.posted)[1].questionId, 'id-2');
  s.win.close();
});

test('a new ask of the same text takes its id while the old one is still open', () => {
  const s = surface();
  ask(s, 'id-1');
  // The old ask was answered on another surface and the agent asked the
  // same text again before this surface heard the first was answered.
  ask(s, 'id-2');
  answer(s.win, 'no');
  assert.strictEqual(answers(s.posted)[0].questionId, 'id-2');
  s.win.close();
});

test('a replay of the pending ask keeps its id and the typed answer', () => {
  const s = surface();
  ask(s, 'id-7');
  const inp = s.win.document.getElementById('task-input');
  inp.value = 'half typed';
  ask(s, 'id-7');
  assert.strictEqual(inp.value, 'half typed');
  answer(s.win, 'done');
  assert.strictEqual(answers(s.posted)[0].questionId, 'id-7');
  s.win.close();
});

test('an ask without an id (an older daemon) is answered without one', () => {
  const s = surface();
  ask(s, 'id-1');
  answer(s.win, 'first');
  send(s.win, {type: 'askUser', question: Q, tabId: s.tab});
  answer(s.win, 'second');
  assert.deepStrictEqual(answers(s.posted)[1], {
    answer: 'second',
    question: Q,
    tabId: s.tab,
    type: 'userAnswer',
  });
  s.win.close();
});

report('askAnswerNamesQuestionId_wave3');
