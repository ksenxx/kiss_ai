// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// ask_user_question no longer opens a floating modal. Its tool_call renders
// as the "Question" transcript panel (translucent red header), the daemon's
// live askUser event marks that panel pending and puts the composer into
// answer mode, the text typed into the composer is posted as the userAnswer,
// and the tool_result (the answer) is shown as a separate user message.

'use strict';

const assert = require('assert');
const {
  makePage: makeChatPage,
  send,
  stubRaf,
  clickTab,
  answering,
  readyTabId,
  typeInto,
  ofType,
} = require('./simplify2_harness');
const {makeRunner, key, pickTabMenuItem} = require('./ui_antipattern_harness');
const {paste, settle, typeAndSend} = require('./attachment_helpers');
const {tabEl} = require('./subagent_tab_helpers');
const {
  VOICE_CFG,
  voicePageState,
  stubVoiceSurface,
  startPipeline,
  dictate,
} = require('./voice_page_helpers');

const pages = [];

// The composer placeholder chat.html gets from {{INPUT_PLACEHOLDER}}; main.js
// reads it from the element at load, so it is set before the scripts run.
function placeholderAndRaf(win) {
  win.document.getElementById('task-input').placeholder = 'Ask anything';
  stubRaf(win);
}

// `voice` (a voicePageState) puts the page over the stubbed voice surface,
// for a test that dictates.
function makePage(withMarked, voice) {
  const {win, posted, getState} = makeChatPage({
    beforeScripts(win) {
      placeholderAndRaf(win);
      if (!voice) return;
      stubVoiceSurface(win, voice);
      win.__VOICE__ = Object.assign({}, VOICE_CFG);
    },
    extraScripts: withMarked ? ['marked.min.js', 'voice.js'] : ['voice.js'],
  });
  pages.push(win);
  if (voice) Object.assign(voice, {win, posted});

  const tab = readyTabId(posted);
  send(win, {type: 'clear', chat_id: 'chat-q', tabId: tab});
  send(win, {type: 'status', running: true, tabId: tab, startTs: Date.now()});
  send(win, {type: 'daemonStatus', connected: true});
  return {win, posted, tab, getState};
}

function askQuestionCall(win, tab, question) {
  send(win, {
    type: 'tool_call',
    name: 'ask_user_question',
    extras: {question},
    callId: 41,
    tabId: tab,
    ts: Date.now(),
  });
}

function pressEnter(win) {
  key(win, win.document.getElementById('task-input'), 'Enter');
}

const runner = makeRunner();
// An async test (fn returns a promise) reports when it settles; the
// summary waits for all of them.
const pending = [];

function test(name, fn) {
  const settled = runner.test(name, fn);
  if (settled) pending.push(settled);
}

test('the ask_user_question tool call renders as the Question panel', () => {
  const {win, tab} = makePage(true);
  askQuestionCall(win, tab, 'Deploy to **staging** or production?');
  const panel = win.document.querySelector('#output .ev.tc.tc-question');
  assert.ok(panel, 'a .tc-question panel must be in the transcript');
  const hdr = panel.querySelector(':scope > .tc-h');
  assert.ok(hdr.classList.contains('tc-h-question'), 'red header class');
  assert.ok(hdr.textContent.endsWith('Question'), 'header reads Question');
  const body = panel.querySelector('.tc-question-body');
  assert.ok(body, 'the question body is rendered');
  assert.ok(body.classList.contains('md-body'), 'question is markdown');
  assert.ok(
    body.querySelector('strong') &&
      body.textContent.includes('Deploy to staging or production?'),
    'markdown is rendered: ' + body.innerHTML,
  );
  assert.strictEqual(
    body.dataset.rawText,
    'Deploy to **staging** or production?',
  );
  assert.ok(
    !panel.querySelector('.tc-b'),
    'the raw argument dump is not shown for a question',
  );
  assert.ok(
    !panel.classList.contains('tc-question-pending'),
    'not pending until the daemon asks',
  );
});

test('without a markdown renderer the question is shown as plain text', () => {
  const {win, tab} = makePage(false);
  askQuestionCall(win, tab, 'Plain <b>text</b>?');
  const body = win.document.querySelector('#output .tc-question-body');
  assert.strictEqual(body.textContent, 'Plain <b>text</b>?');
  assert.ok(
    !body.querySelector('b'),
    'HTML in the question is not interpreted',
  );
});

test('a tool call without a question still renders an empty body', () => {
  const {win, tab} = makePage(true);
  send(win, {
    type: 'tool_call',
    name: 'ask_user_question',
    callId: 42,
    tabId: tab,
    ts: Date.now(),
  });
  const body = win.document.querySelector('#output .tc-question-body');
  assert.ok(body, 'body exists');
  assert.strictEqual(body.textContent, '');
});

test('askUser marks the panel pending and puts the composer into answer mode', () => {
  const {win, tab} = makePage(true);
  const inp = win.document.getElementById('task-input');
  assert.strictEqual(inp.placeholder, 'Ask anything');
  askQuestionCall(win, tab, 'Which branch?');
  send(win, {type: 'askUser', question: 'Which branch?', tabId: tab});
  const panel = win.document.querySelector('#output .tc-question');
  assert.ok(panel.classList.contains('tc-question-pending'));
  const hint = panel.querySelector('.tc-h .tc-question-hint');
  assert.ok(hint, 'the header tells the user where to answer');
  assert.ok(hint.textContent.includes('answer in the chat box below'));
  assert.ok(answering(win), 'body.ask-answering is set');
  assert.strictEqual(inp.placeholder, 'Type your answer and press Enter');
  assert.strictEqual(win.document.activeElement, inp, 'composer is focused');
  // (A duplicate replay of the same question is
  // ask_user_duplicate_replay_preserves_draft.test.js.)
});

test('Enter in the composer posts the text as the userAnswer, not a prompt', () => {
  const {win, posted, tab} = makePage(true);
  const inp = win.document.getElementById('task-input');
  askQuestionCall(win, tab, 'Which branch?');
  send(win, {type: 'askUser', question: 'Which branch?', tabId: tab});
  posted.length = 0;
  inp.value = '  main  ';
  pressEnter(win);
  const answers = ofType(posted, 'userAnswer');
  assert.deepStrictEqual(
    answers.map(a => [a.answer, a.tabId]),
    [['main', tab]],
    JSON.stringify(posted),
  );
  assert.strictEqual(
    posted.filter(m => m.type === 'submit' || m.type === 'appendUserMessage')
      .length,
    0,
    'the running task must not get a follow-up prompt',
  );
  assert.strictEqual(inp.value, '', 'the composer is cleared');
  assert.ok(!answering(win), 'answer mode ends as soon as the answer is sent');
  assert.strictEqual(inp.placeholder, 'Ask anything');
  const panel = win.document.querySelector('#output .tc-question');
  assert.ok(
    !panel.classList.contains('tc-question-pending'),
    'the panel is no longer pending',
  );
  assert.ok(!panel.querySelector('.tc-question-hint'), 'the hint is gone');

  // The tool returns the answer as a separate user message.
  send(win, {
    type: 'tool_result',
    tool_name: 'ask_user_question',
    content: 'main',
    tabId: tab,
    ts: Date.now(),
  });
  const ans = panel.nextElementSibling;
  assert.ok(
    ans.classList.contains('user-msg'),
    'the response is a user message',
  );
  assert.ok(
    ans.querySelector('.task-panel-h').textContent.includes('Response'),
  );
  assert.strictEqual(ans.querySelector('.task-panel-text').textContent, 'main');
  assert.strictEqual(ans.dataset.rawText, 'main');
  assert.ok(
    !panel.querySelector('.bash-panel'),
    'no generic tool output block is added for a question',
  );

  // Once answered, the composer is a prompt box again.
  posted.length = 0;
  inp.value = 'now run the tests';
  pressEnter(win);
  assert.strictEqual(ofType(posted, 'userAnswer').length, 0);
  assert.strictEqual(
    ofType(posted, 'appendUserMessage').length,
    1,
    JSON.stringify(posted),
  );
});

test('an empty composer sends nothing while a question is pending', () => {
  const {win, posted, tab} = makePage(true);
  askQuestionCall(win, tab, 'Which branch?');
  send(win, {type: 'askUser', question: 'Which branch?', tabId: tab});
  posted.length = 0;
  win.document.getElementById('task-input').value = '   ';
  pressEnter(win);
  assert.strictEqual(ofType(posted, 'userAnswer').length, 0);
  assert.ok(answering(win), 'the question stays open');
});

test('askUserDone from another client clears the pending mark and answer mode', () => {
  const {win, tab} = makePage(true);
  askQuestionCall(win, tab, 'Which branch?');
  send(win, {type: 'askUser', question: 'Which branch?', tabId: tab});
  send(win, {type: 'askUserDone', tabId: tab});
  const panel = win.document.querySelector('#output .tc-question');
  assert.ok(!panel.classList.contains('tc-question-pending'));
  assert.ok(!panel.querySelector('.tc-question-hint'));
  assert.ok(!answering(win));
  // Clearing again is harmless (no hint to remove).
  send(win, {type: 'askUserDone', tabId: tab});
  assert.ok(!panel.querySelector('.tc-question-hint'));
});

// The answer was given on another surface and its askUserDone was
// missed: the tool result alone retires the question still pending here.
test('the answer of a pending question shows under it and ends answer mode', () => {
  const {win, tab} = makePage(true);
  askQuestionCall(win, tab, 'Which branch?');
  send(win, {type: 'askUser', question: 'Which branch?', tabId: tab});
  const panel = win.document.querySelector('#output .tc-question');
  assert.ok(panel.classList.contains('tc-question-pending'));
  assert.ok(answering(win));
  send(win, {
    type: 'tool_result',
    tool_name: 'ask_user_question',
    content: 'release',
    tabId: tab,
    ts: Date.now(),
  });
  assert.ok(
    panel
      .querySelector('.tc-question-body')
      .textContent.includes('Which branch?'),
  );
  assert.strictEqual(
    panel.nextElementSibling.querySelector('.task-panel-text').textContent,
    'release',
  );
  assert.ok(!panel.classList.contains('tc-question-pending'));
  assert.ok(!answering(win));
});

test('a failed tool_result on a question renders the error, not an answer', () => {
  const {win, tab} = makePage(true);
  askQuestionCall(win, tab, 'Which branch?');
  send(win, {
    type: 'tool_result',
    tool_name: 'ask_user_question',
    content: 'interrupted',
    is_error: true,
    tabId: tab,
    ts: Date.now(),
  });
  const panel = win.document.querySelector('#output .tc-question');
  assert.ok(!panel.querySelector('.tc-question-answer'), 'no answer block');
  assert.ok(panel.querySelector('.tr'), 'the error result block is shown');
});

test('a tool_result with no preceding tool call panel falls back to the plain path', () => {
  const {win, tab} = makePage(true);
  send(win, {
    type: 'tool_result',
    tool_name: 'ask_user_question',
    content: 'orphan',
    tabId: tab,
    ts: Date.now(),
  });
  assert.ok(!win.document.querySelector('.tc-question-answer'));
  assert.ok(win.document.querySelector('#output .bash-panel'));
});

test('askUser before any tool call panel still enters answer mode', () => {
  const {win, posted, tab} = makePage(true);
  send(win, {type: 'askUser', question: 'No panel yet?', tabId: tab});
  assert.ok(answering(win));
  posted.length = 0;
  win.document.getElementById('task-input').value = 'fine';
  pressEnter(win);
  assert.strictEqual(ofType(posted, 'userAnswer').length, 1);
});

test('the question of a background tab is answered from that tab, not the one on screen', () => {
  const {win, posted, tab} = makePage(true);
  const inp = win.document.getElementById('task-input');
  win._testApi.createNewTab();
  const other = win._testApi.getActiveTabId();
  assert.notStrictEqual(other, tab);
  askQuestionCall(win, tab, 'Background question?');
  send(win, {type: 'askUser', question: 'Background question?', tabId: tab});
  // The question pulls the user over to its tab; they go back to the
  // other tab to carry on there.
  assert.strictEqual(win._testApi.getActiveTabId(), tab);
  clickTab(win, other);
  assert.strictEqual(win._testApi.getActiveTabId(), other);
  assert.ok(!answering(win), 'the tab on screen has no question');
  assert.strictEqual(inp.placeholder, 'Ask anything');
  posted.length = 0;
  inp.value = 'a prompt for the other tab';
  pressEnter(win);
  assert.strictEqual(
    ofType(posted, 'userAnswer').length,
    0,
    'typing on the other tab is a prompt: ' + JSON.stringify(posted),
  );
  assert.strictEqual(ofType(posted, 'submit').length, 1);

  // Switching to the asking tab: its panel is pending and the composer
  // answers it.
  clickTab(win, tab);
  assert.ok(answering(win));
  const panel = win.document.querySelector('#output .tc-question');
  assert.ok(panel && panel.classList.contains('tc-question-pending'));
  posted.length = 0;
  inp.value = 'yes';
  pressEnter(win);
  assert.deepStrictEqual(
    ofType(posted, 'userAnswer').map(a => [a.answer, a.tabId]),
    [['yes', tab]],
  );
});

test('a file tab is never in answer mode; back on the chat tab it is again', async () => {
  const voice = voicePageState();
  const {win, posted, tab} = makePage(true, voice);
  await startPipeline(voice);
  askQuestionCall(win, tab, 'Owner question?');
  send(win, {type: 'askUser', question: 'Owner question?', tabId: tab});
  assert.ok(answering(win));
  send(win, {
    type: 'fileContent',
    name: 'README.md',
    path: '/w/README.md',
    content: '<h1>readme</h1>',
    tabId: tab,
  });
  const fileTab = win._testApi.getActiveTabId();
  assert.notStrictEqual(fileTab, tab, 'file tab is on screen');
  assert.ok(
    !answering(win),
    'the file tab hides the text box, so it is not the answer box',
  );
  // Voice dictation while the file tab is up switches back to the chat
  // and answers there.
  posted.length = 0;
  dictate(voice, 'from the file tab');
  assert.strictEqual(win._testApi.getActiveTabId(), tab);
  assert.deepStrictEqual(
    ofType(posted, 'userAnswer').map(a => [a.answer, a.tabId]),
    [['from the file tab', tab]],
    JSON.stringify(posted),
  );
  assert.ok(!answering(win));
});

test('opening a file tab and coming back keeps the question answerable', () => {
  const {win, tab} = makePage(true);
  send(win, {
    type: 'fileContent',
    name: 'a.md',
    path: '/w/a.md',
    content: '<h1>a</h1>',
    tabId: tab,
  });
  const fileTab = win._testApi.getActiveTabId();
  // On the phone layout the file screen's strip lists the file tabs and
  // the chat screen's the chat alone, so the chat is picked from the
  // Chats panel.
  clickTab(win, tab);
  askQuestionCall(win, tab, 'Q?');
  send(win, {type: 'askUser', question: 'Q?', tabId: tab});
  assert.ok(answering(win));
  // Activating the EXISTING file tab leaves answer mode ...
  clickTab(win, fileTab);
  assert.strictEqual(win._testApi.getActiveTabId(), fileTab);
  assert.ok(!answering(win), 'answer mode is off on the file tab');
  // ... and closing the asking chat while its file tab is up leaves
  // nothing to answer. With the file tab up the strip lists the file
  // alone: its menu closes every other tab.
  const fileEl = tabEl(win, fileTab);
  assert.ok(fileEl, 'the file tab is on its strip');
  pickTabMenuItem(win, fileEl, 'Close Others');
  assert.ok(
    !win._testApi.openTabs().some(t => t.id === tab),
    'the asking chat is closed',
  );
  assert.strictEqual(win._testApi.getActiveTabId(), fileTab, 'file tab stays');
  assert.ok(!answering(win));
  assert.strictEqual(
    win.document.getElementById('task-input').placeholder,
    'Ask anything',
  );
});

test('a prompt typed before the question is parked, not sent as the answer', () => {
  const {win, posted, tab} = makePage(true);
  const inp = win.document.getElementById('task-input');
  inp.value = 'run the deployment';
  askQuestionCall(win, tab, 'Which region?');
  send(win, {type: 'askUser', question: 'Which region?', tabId: tab});
  assert.strictEqual(inp.value, '', 'the composer starts empty for the answer');
  posted.length = 0;
  pressEnter(win);
  assert.strictEqual(
    ofType(posted, 'userAnswer').length,
    0,
    'an empty composer sends nothing',
  );
  inp.value = 'eu-west';
  pressEnter(win);
  assert.deepStrictEqual(
    ofType(posted, 'userAnswer').map(a => a.answer),
    ['eu-west'],
  );
  assert.strictEqual(
    inp.value,
    'run the deployment',
    'the parked prompt is back once the answer is sent',
  );
  assert.ok(!answering(win));
});

test('an answer being typed when another client answers is kept, above the parked prompt', () => {
  const {win, posted, tab} = makePage(true);
  const inp = win.document.getElementById('task-input');
  inp.value = 'old prompt';
  askQuestionCall(win, tab, 'Which region?');
  send(win, {type: 'askUser', question: 'Which region?', tabId: tab});
  inp.value = 'half an ans';
  send(win, {type: 'askUserDone', tabId: tab});
  assert.strictEqual(inp.value, 'half an ans\nold prompt');
  posted.length = 0;
  pressEnter(win);
  assert.strictEqual(ofType(posted, 'userAnswer').length, 0);
  assert.strictEqual(
    ofType(posted, 'appendUserMessage').length,
    1,
    'without a question the text is a prompt again',
  );
});

// A27_astra-1: another surface answered the question while this one was
// away (its askUserDone was missed), so the replay that follows the
// reconnect carries the question's tool_result.  The tab leaves answer
// mode with it, as the daemon re-announces only a question still
// pending: the next message is a prompt again, not an answer to a
// question that is gone (the daemon drops such an answer).
test('a question answered elsewhere while away is retired by its replayed result', () => {
  const {win, posted, tab} = makePage(true);
  const inp = win.document.getElementById('task-input');
  inp.value = 'old prompt';
  askQuestionCall(win, tab, 'Proceed?');
  send(win, {type: 'askUser', question: 'Proceed?', questionId: 'q1', tabId: tab});
  inp.value = 'half an ans';
  send(win, {type: 'daemonStatus', connected: false});
  send(win, {type: 'daemonStatus', connected: true});
  send(win, {
    type: 'task_events',
    tabId: tab,
    task: 'deploy',
    task_id: 't1',
    chat_id: 'chat-q',
    events: [
      {
        type: 'tool_call',
        name: 'ask_user_question',
        extras: {question: 'Proceed?'},
        callId: 41,
        ts: 1,
      },
      {type: 'tool_result', callId: 41, content: 'yes, go', ts: 2},
      {type: 'text', content: 'Deploying.', ts: 3},
    ],
  });
  assert.ok(!answering(win), 'BUG - still answering a question that was answered');
  assert.strictEqual(inp.value, 'half an ans\nold prompt', 'nothing typed is lost');
  assert.ok(
    !win.document.querySelector('.tc-question-pending'),
    'the replayed question panel is not pending',
  );
  posted.length = 0;
  pressEnter(win);
  assert.strictEqual(ofType(posted, 'userAnswer').length, 0);
  assert.strictEqual(
    ofType(posted, 'appendUserMessage').length,
    1,
    'the message is a prompt again',
  );
  // A question still open at replay time stays open (the daemon
  // re-announces it right after the replay; the typed answer stands).
  inp.value = 'next prompt';
  askQuestionCall(win, tab, 'Region?');
  send(win, {type: 'askUser', question: 'Region?', questionId: 'q2', tabId: tab});
  inp.value = 'eu';
  send(win, {
    type: 'task_events',
    tabId: tab,
    task: 'deploy',
    task_id: 't1',
    chat_id: 'chat-q',
    events: [
      {
        type: 'tool_call',
        name: 'ask_user_question',
        extras: {question: 'Region?'},
        callId: 42,
        ts: 4,
      },
    ],
  });
  send(win, {type: 'askUser', question: 'Region?', questionId: 'q2', tabId: tab});
  assert.ok(answering(win), 'the open question is still being answered');
  assert.strictEqual(inp.value, 'eu', 'the typed answer stands');
  // Audit 2026-11-08 (W6 F2): the agent asked the same words twice
  // (calls 43 and 44); the first was answered, the second is the open
  // one.  Its replayed answer belongs to the earlier call, not to the
  // question being answered here: answer mode, the typed answer and
  // the parked prompt all stand, and only the last panel is pending.
  // (The daemon re-announces the open question under the id it was
  // asked with; another id is another ask, whose composer starts
  // empty: askAnswerNamesQuestionId_wave3.test.js.)
  send(win, {
    type: 'task_events',
    tabId: tab,
    task: 'deploy',
    task_id: 't1',
    chat_id: 'chat-q',
    events: [
      {
        type: 'tool_call',
        name: 'ask_user_question',
        extras: {question: 'Region?'},
        callId: 43,
        ts: 5,
      },
      {type: 'tool_result', callId: 43, content: 'us', ts: 6},
      {
        type: 'tool_call',
        name: 'ask_user_question',
        extras: {question: 'Region?'},
        callId: 44,
        ts: 7,
      },
    ],
  });
  send(win, {type: 'askUser', question: 'Region?', questionId: 'q2', tabId: tab});
  assert.ok(answering(win), 'BUG - the earlier call\'s answer retired the open question');
  assert.strictEqual(inp.value, 'eu', 'the typed answer stands');
  const panels = win.document.querySelectorAll('.tc-question');
  const pendingPanels = win.document.querySelectorAll('.tc-question-pending');
  assert.strictEqual(pendingPanels.length, 1, 'one pending panel');
  assert.strictEqual(pendingPanels[0], panels[panels.length - 1], 'the last one');
  send(win, {type: 'askUserDone', tabId: tab});
  assert.strictEqual(
    inp.value,
    'eu\nnext prompt',
    'the parked prompt comes back under the answer once the question closes',
  );
});

// A27_astra-2: the question changed under this surface (another one
// answered Q1 and the agent asked Q2; Q1's close was missed).  Entering
// answer mode for Q2 must not take the Q1 answer draft for a prompt and
// overwrite the prompt parked before Q1: the old mode is retired first,
// losslessly, so both come back once Q2 is answered.
test('a replacement question keeps the parked prompt and the old answer draft', () => {
  const {win, posted, tab} = makePage(true);
  const inp = win.document.getElementById('task-input');
  inp.value = 'ORIGINAL PARKED PROMPT';
  askQuestionCall(win, tab, 'Q1?');
  send(win, {type: 'askUser', question: 'Q1?', questionId: 'q1', tabId: tab});
  inp.value = 'DRAFT ANSWER FOR Q1';
  askQuestionCall(win, tab, 'Q2?');
  send(win, {type: 'askUser', question: 'Q2?', questionId: 'q2', tabId: tab});
  assert.ok(answering(win));
  assert.strictEqual(inp.value, '', 'the composer starts empty for Q2');
  posted.length = 0;
  inp.value = 'answer 2';
  pressEnter(win);
  const answers = ofType(posted, 'userAnswer');
  assert.deepStrictEqual(
    answers.map(a => [a.answer, a.questionId]),
    [['answer 2', 'q2']],
  );
  assert.strictEqual(
    inp.value,
    'DRAFT ANSWER FOR Q1\nORIGINAL PARKED PROMPT',
    'BUG - the parked prompt was lost to the replacement question',
  );
  assert.ok(!answering(win));
});

test('a background tab parks its prompt too and shows it again once answered', () => {
  const {win, posted, tab} = makePage(true);
  const inp = win.document.getElementById('task-input');
  inp.value = 'draft for the first tab';
  win._testApi.createNewTab();
  askQuestionCall(win, tab, 'Q?');
  send(win, {type: 'askUser', question: 'Q?', tabId: tab});
  clickTab(win, tab);
  assert.strictEqual(inp.value, '', 'the parked prompt is not the answer');
  assert.ok(answering(win));
  posted.length = 0;
  inp.value = 'yes';
  pressEnter(win);
  assert.deepStrictEqual(
    ofType(posted, 'userAnswer').map(a => a.answer),
    ['yes'],
  );
  assert.strictEqual(inp.value, 'draft for the first tab');
});

test('the task ending gives the parked prompt back', () => {
  const {win, tab} = makePage(true);
  const inp = win.document.getElementById('task-input');
  inp.value = 'next step';
  send(win, {type: 'askUser', question: 'Q?', tabId: tab});
  assert.strictEqual(inp.value, '');
  send(win, {type: 'task_done', tabId: tab, success: true});
  assert.ok(!answering(win));
  assert.strictEqual(inp.value, 'next step');
});

test('the parked prompt and the answer in progress both survive a reload', () => {
  const {win, tab, getState} = makePage(true);
  win._testApi.endLaunch();
  const inp = win.document.getElementById('task-input');
  inp.value = 'parked prompt';
  send(win, {type: 'askUser', question: 'Q?', tabId: tab});
  inp.value = 'typed answer';
  win.dispatchEvent(new win.Event('pagehide'));
  const state = getState();
  assert.strictEqual(
    JSON.stringify(state.inputDrafts),
    JSON.stringify({[tab]: 'parked prompt'}),
  );
  assert.strictEqual(
    JSON.stringify(state.askDrafts),
    JSON.stringify({[tab]: {question: 'Q?', answer: 'typed answer'}}),
  );
});

// Paste a HEIC photo whose decode (the createImageBitmap the page asks
// the browser for) is parked forever, so its attachment chip stays busy.
async function pasteHeic(win) {
  win.createImageBitmap = () => new Promise(() => {});
  const inp = win.document.getElementById('task-input');
  const heic = new win.File([new Uint8Array([1, 2, 3])], 'IMG_0001.HEIC', {
    type: 'image/heic',
  });
  paste(win, heic);
  await settle(() => win.document.querySelector('.file-chip'));
  return {inp, heic};
}

test('a send that waits for an attachment keeps its route', async () => {
  const {win, posted, tab} = makePage(true);
  win._testApi.endLaunch();
  const {inp, heic} = await pasteHeic(win);
  // A prompt parked behind the conversion: a question arriving during
  // the wait must not turn it into an answer.
  inp.value = 'a prompt with a photo';
  posted.length = 0;
  pressEnter(win);
  await settle();
  send(win, {type: 'askUser', question: 'Q?', tabId: tab});
  assert.ok(answering(win));
  // The stuck chip is removed, releasing the parked send.
  win.document
    .querySelector('.file-chip .fc-rm')
    .dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  await settle(() => !win.document.querySelector('.file-chip'));
  assert.strictEqual(
    posted.filter(
      m => m.type === 'userAnswer' || m.type === 'appendUserMessage',
    ).length,
    0,
    'the released send posts nothing: the prompt is parked: ' +
      JSON.stringify(posted),
  );
  assert.strictEqual(inp.value, '', 'the composer is the empty answer box');
  // An answer never waits for a photo: it is text only, the chip stays
  // for the next prompt, and the parked prompt comes back.
  paste(win, heic);
  await settle(() => win.document.querySelector('.file-chip'));
  posted.length = 0;
  inp.value = 'the answer';
  pressEnter(win);
  assert.deepStrictEqual(
    ofType(posted, 'userAnswer').map(a => a.answer),
    ['the answer'],
  );
  assert.ok(win.document.querySelector('.file-chip'), 'the photo chip stays');
  assert.strictEqual(inp.value, 'a prompt with a photo');
});

test('the send button answers too', () => {
  const {win, posted, tab} = makePage(true);
  askQuestionCall(win, tab, 'Button?');
  send(win, {type: 'askUser', question: 'Button?', tabId: tab});
  posted.length = 0;
  typeAndSend(win, 'clicked');
  assert.strictEqual(ofType(posted, 'userAnswer').length, 1);
});

test('a question is never folded by the streaming collapse pass, answered or not', () => {
  const {win, tab} = makePage(true);
  function bash(callId) {
    send(win, {
      type: 'tool_call',
      name: 'Bash',
      command: 'echo ' + callId,
      callId,
      tabId: tab,
      ts: Date.now(),
    });
  }
  bash(1);
  askQuestionCall(win, tab, 'Still readable?');
  send(win, {type: 'askUser', question: 'Still readable?', tabId: tab});
  bash(2);
  bash(3);
  bash(4);
  const panels = win.document.querySelectorAll('#output .ev.tc');
  assert.strictEqual(panels.length, 5);
  assert.ok(
    panels[0].classList.contains('collapsed'),
    'an older Bash panel folds as the stream goes on',
  );
  const question = win.document.querySelector('#output .tc-question');
  assert.ok(
    !question.classList.contains('collapsed'),
    'the unanswered question stays open',
  );
  // Answered: the question stays open while the stream goes on (the
  // user's side of the conversation is never folded away; see
  // askPanelsNeverCollapsed.test.js for the answer and the replays).
  send(win, {type: 'askUserDone', tabId: tab});
  bash(5);
  bash(6);
  bash(7);
  assert.ok(
    !question.classList.contains('collapsed'),
    'the answered question stays open too',
  );
  assert.ok(
    !question.classList.contains('tc-question-pending'),
    'and is no longer marked pending',
  );
  assert.ok(
    win.document
      .querySelectorAll('#output .ev.tc')[2]
      .classList.contains('collapsed'),
    'while plain panels around it still fold',
  );
});

test('answers stay out of the prompt history, and history recall is off while answering', () => {
  const {win, posted, tab} = makePage(true);
  const inp = win.document.getElementById('task-input');
  // A prompt in the history first.
  inp.value = 'previous task prompt';
  pressEnter(win);
  send(win, {type: 'askUser', question: 'Q?', tabId: tab});
  key(win, inp, 'ArrowUp');
  assert.strictEqual(
    inp.value,
    '',
    'no prompt is recalled into the answer box',
  );
  inp.value = 'secret answer';
  pressEnter(win);
  assert.deepStrictEqual(
    ofType(posted, 'userAnswer').map(a => a.answer),
    ['secret answer'],
  );
  key(win, inp, 'ArrowUp');
  assert.strictEqual(
    inp.value,
    'previous task prompt',
    'history recall offers the prompt, never the answer',
  );
});

test('the Send button works for an answer while a photo is still converting', async () => {
  const {win, posted, tab} = makePage(true);
  win._testApi.endLaunch();
  const {inp} = await pasteHeic(win);
  const sendBtn = win.document.getElementById('send-btn');
  assert.ok(sendBtn.disabled, 'a prompt waits for the photo');
  send(win, {type: 'askUser', question: 'Q?', tabId: tab});
  assert.ok(!sendBtn.disabled, 'an answer does not');
  posted.length = 0;
  inp.value = 'yes';
  sendBtn.click();
  assert.deepStrictEqual(
    ofType(posted, 'userAnswer').map(a => a.answer),
    ['yes'],
  );
  assert.ok(
    sendBtn.disabled,
    'back to waiting for the photo for the next prompt',
  );
});

test('the clear button reflects the parked prompt that comes back', () => {
  const {win, tab} = makePage(true);
  const inp = win.document.getElementById('task-input');
  const clearBtn = win.document.getElementById('input-clear-btn');
  typeInto(win, inp, 'next prompt');
  assert.notStrictEqual(clearBtn.style.display, 'none');
  send(win, {type: 'askUser', question: 'Q?', tabId: tab});
  assert.strictEqual(clearBtn.style.display, 'none', 'empty answer box');
  inp.value = 'yes';
  pressEnter(win);
  assert.strictEqual(inp.value, 'next prompt');
  assert.notStrictEqual(
    clearBtn.style.display,
    'none',
    'text is back, so is the button',
  );
});

Promise.all(pending).then(() => {
  for (const win of pages) win.close();
  runner.report('askQuestionPanel');
});
