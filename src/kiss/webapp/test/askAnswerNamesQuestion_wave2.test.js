// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// Two surfaces show the same ask_user_question.  One answers it; the
// agent can ask its next question before the other surface hears that
// the first was answered.  An answer typed on the late surface must not
// become the answer to the next question, so every `userAnswer` names
// the question it answers, verbatim, and the daemon
// (server/commands.py `_cmd_user_answer`) drops an answer whose
// question is no longer the pending one.

const assert = require('assert');
const {makePage, send, stubRaf, readyTabId} = require('./simplify2_harness');
const {test, report} = require('./ui_antipattern_harness').makeRunner();

const Q1 = 'Deploy to **staging** or production?\n  (say which)';
const Q2 = 'Run the migrations too?';

/** A surface showing a running task on its tab. */
function surface() {
  const {win, posted} = makePage({beforeScripts: stubRaf});
  const tab = readyTabId(posted);
  send(win, {type: 'clear', chat_id: 'chat-q', tabId: tab});
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

test('an answer names the question it answers, verbatim', () => {
  const a = surface();
  const b = surface();
  for (const s of [a, b]) {
    send(s.win, {type: 'askUser', question: Q1, tabId: s.tab});
  }
  answer(a.win, 'staging');
  assert.deepStrictEqual(answers(a.posted), [
    {answer: 'staging', question: Q1, tabId: a.tab, type: 'userAnswer'},
  ]);

  // The daemon delivered A's answer and the agent asked Q2, but B has
  // not heard yet: B's answer still names Q1, which the daemon no
  // longer has pending.
  answer(b.win, 'production');
  assert.deepStrictEqual(answers(b.posted), [
    {answer: 'production', question: Q1, tabId: b.tab, type: 'userAnswer'},
  ]);

  // A answers the next question under that question's name.
  send(a.win, {type: 'askUser', question: Q2, tabId: a.tab});
  answer(a.win, 'yes');
  assert.deepStrictEqual(answers(a.posted)[1], {
    answer: 'yes',
    question: Q2,
    tabId: a.tab,
    type: 'userAnswer',
  });
  a.win.close();
  b.win.close();
});

report('askAnswerNamesQuestion_wave2');
