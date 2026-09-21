// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// ask_user_question no longer opens a floating modal. Its tool_call renders
// as the "Question" transcript panel (translucent red header), the daemon's
// live askUser event marks that panel pending and puts the composer into
// answer mode, the text typed into the composer is posted as the userAnswer,
// and the tool_result (the answer) is shown inside the same panel.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

function makeWebview(withMarked) {
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  html = html.replace(/\{\{INPUT_PLACEHOLDER\}\}/g, 'Ask anything');
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
      postMessage: m => posted.push(m),
      getState: () => state,
      setState: s => {
        state = s;
        win._vscodeState = s;
      },
    };
  };

  if (withMarked) {
    win.eval(fs.readFileSync(path.join(MEDIA, 'marked.min.js'), 'utf8'));
  }
  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'voice.js'), 'utf8'));

  const ready = posted.find(m => m.type === 'ready');
  assert.ok(ready && ready.tabId, 'webview must post ready with a tabId');
  const tab = ready.tabId;
  send(win, {type: 'clear', chat_id: 'chat-q', tabId: tab});
  send(win, {type: 'status', running: true, tabId: tab, startTs: Date.now()});
  send(win, {type: 'daemonStatus', connected: true});
  return {win, posted, tab};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
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
  const inp = win.document.getElementById('task-input');
  inp.dispatchEvent(
    new win.KeyboardEvent('keydown', {key: 'Enter', bubbles: true}),
  );
}

function answering(win) {
  return win.document.body.classList.contains('ask-answering');
}

let passed = 0;
const failures = [];
const pending = [];

function pass(name) {
  passed++;
  console.log(`  \u2713 ${name}`);
}

function fail(name, e) {
  failures.push({name, error: e});
  console.log(`  \u2717 ${name}`);
  console.log(`      ${e.message}`);
}

// Synchronous tests report at once; an async test (fn returns a promise)
// reports when it settles, and the summary waits for all of them.
function test(name, fn) {
  let r;
  try {
    r = fn();
  } catch (e) {
    fail(name, e);
    return;
  }
  if (r && typeof r.then === 'function') {
    pending.push(
      r.then(
        () => pass(name),
        e => fail(name, e),
      ),
    );
  } else {
    pass(name);
  }
}

test('the ask_user_question tool call renders as the Question panel', () => {
  const {win, tab} = makeWebview(true);
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
  const {win, tab} = makeWebview(false);
  askQuestionCall(win, tab, 'Plain <b>text</b>?');
  const body = win.document.querySelector('#output .tc-question-body');
  assert.strictEqual(body.textContent, 'Plain <b>text</b>?');
  assert.ok(
    !body.querySelector('b'),
    'HTML in the question is not interpreted',
  );
});

test('a tool call without a question still renders an empty body', () => {
  const {win, tab} = makeWebview(true);
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
  const {win, tab} = makeWebview(true);
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
  // A duplicate replay of the same question changes nothing.
  send(win, {type: 'askUser', question: 'Which branch?', tabId: tab});
  assert.strictEqual(panel.querySelectorAll('.tc-question-hint').length, 1);
});

test('Enter in the composer posts the text as the userAnswer, not a prompt', () => {
  const {win, posted, tab} = makeWebview(true);
  const inp = win.document.getElementById('task-input');
  askQuestionCall(win, tab, 'Which branch?');
  send(win, {type: 'askUser', question: 'Which branch?', tabId: tab});
  posted.length = 0;
  inp.value = '  main  ';
  pressEnter(win);
  const answers = posted.filter(m => m.type === 'userAnswer');
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

  // The tool returns the answer: it is shown inside the Question panel.
  send(win, {
    type: 'tool_result',
    tool_name: 'ask_user_question',
    content: 'main',
    tabId: tab,
    ts: Date.now(),
  });
  const ans = panel.querySelector('.tc-question-answer');
  assert.ok(ans, 'the answer block is inside the Question panel');
  assert.strictEqual(
    ans.querySelector('.tc-question-answer-label').textContent,
    'Answer',
  );
  assert.strictEqual(
    ans.querySelector('.tc-question-answer-text').textContent,
    'main',
  );
  assert.strictEqual(ans.dataset.rawText, 'Answer: main');
  assert.ok(
    !panel.querySelector('.bash-panel'),
    'no generic tool output block is added for a question',
  );

  // Once answered, the composer is a prompt box again.
  posted.length = 0;
  inp.value = 'now run the tests';
  pressEnter(win);
  assert.strictEqual(posted.filter(m => m.type === 'userAnswer').length, 0);
  assert.strictEqual(
    posted.filter(m => m.type === 'appendUserMessage').length,
    1,
    JSON.stringify(posted),
  );
});

test('an empty composer sends nothing while a question is pending', () => {
  const {win, posted, tab} = makeWebview(true);
  askQuestionCall(win, tab, 'Which branch?');
  send(win, {type: 'askUser', question: 'Which branch?', tabId: tab});
  posted.length = 0;
  win.document.getElementById('task-input').value = '   ';
  pressEnter(win);
  assert.strictEqual(posted.filter(m => m.type === 'userAnswer').length, 0);
  assert.ok(answering(win), 'the question stays open');
});

test('askUserDone from another client clears the pending mark and answer mode', () => {
  const {win, tab} = makeWebview(true);
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

test('a replayed answered question shows question and answer, never pending', () => {
  const {win, tab} = makeWebview(true);
  askQuestionCall(win, tab, 'Which branch?');
  send(win, {
    type: 'tool_result',
    tool_name: 'ask_user_question',
    content: 'release',
    tabId: tab,
    ts: Date.now(),
  });
  const panel = win.document.querySelector('#output .tc-question');
  assert.ok(
    panel
      .querySelector('.tc-question-body')
      .textContent.includes('Which branch?'),
  );
  assert.strictEqual(
    panel.querySelector('.tc-question-answer-text').textContent,
    'release',
  );
  assert.ok(!panel.classList.contains('tc-question-pending'));
  assert.ok(!answering(win));
});

test('a failed tool_result on a question renders the error, not an answer', () => {
  const {win, tab} = makeWebview(true);
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
  const {win, tab} = makeWebview(true);
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
  const {win, posted, tab} = makeWebview(true);
  send(win, {type: 'askUser', question: 'No panel yet?', tabId: tab});
  assert.ok(answering(win));
  posted.length = 0;
  win.document.getElementById('task-input').value = 'fine';
  pressEnter(win);
  assert.strictEqual(posted.filter(m => m.type === 'userAnswer').length, 1);
});

test('the question of a background tab is answered from that tab, not the one on screen', () => {
  const {win, posted, tab} = makeWebview(true);
  const inp = win.document.getElementById('task-input');
  win._testApi.createNewTab();
  const other = win._testApi.getActiveTabId();
  assert.notStrictEqual(other, tab);
  askQuestionCall(win, tab, 'Background question?');
  send(win, {type: 'askUser', question: 'Background question?', tabId: tab});
  assert.ok(!answering(win), 'the tab on screen has no question');
  assert.strictEqual(inp.placeholder, 'Ask anything');
  posted.length = 0;
  inp.value = 'a prompt for the other tab';
  pressEnter(win);
  assert.strictEqual(
    posted.filter(m => m.type === 'userAnswer').length,
    0,
    'typing on the other tab is a prompt: ' + JSON.stringify(posted),
  );
  assert.strictEqual(posted.filter(m => m.type === 'submit').length, 1);

  // Switching to the asking tab: its panel is pending and the composer
  // answers it.
  const tabEl = win.document.querySelector(
    `.chat-tab[data-tab-id=${JSON.stringify(tab)}]`,
  );
  tabEl.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.ok(answering(win));
  const panel = win.document.querySelector('#output .tc-question');
  assert.ok(panel && panel.classList.contains('tc-question-pending'));
  posted.length = 0;
  inp.value = 'yes';
  pressEnter(win);
  assert.deepStrictEqual(
    posted.filter(m => m.type === 'userAnswer').map(a => [a.answer, a.tabId]),
    [['yes', tab]],
  );
});

test('a file tab is never in answer mode; back on the chat tab it is again', () => {
  const {win, posted, tab} = makeWebview(true);
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
  send(win, {type: 'voiceSpeech', text: 'from the file tab'});
  assert.strictEqual(win._testApi.getActiveTabId(), tab);
  assert.deepStrictEqual(
    posted.filter(m => m.type === 'userAnswer').map(a => [a.answer, a.tabId]),
    [['from the file tab', tab]],
    JSON.stringify(posted),
  );
  assert.ok(!answering(win));
});

test('opening a file tab and coming back keeps the question answerable', () => {
  const {win, tab} = makeWebview(true);
  send(win, {
    type: 'fileContent',
    name: 'a.md',
    path: '/w/a.md',
    content: '<h1>a</h1>',
    tabId: tab,
  });
  const fileTab = win._testApi.getActiveTabId();
  const chatEl = win.document.querySelector(
    `.chat-tab[data-tab-id=${JSON.stringify(tab)}]`,
  );
  chatEl.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  askQuestionCall(win, tab, 'Q?');
  send(win, {type: 'askUser', question: 'Q?', tabId: tab});
  assert.ok(answering(win));
  // Activating the EXISTING file tab leaves answer mode ...
  const fileEl = win.document.querySelector(
    `.chat-tab[data-tab-id=${JSON.stringify(fileTab)}]`,
  );
  fileEl.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.ok(!answering(win), 'answer mode is off on the file tab');
  // ... and closing the asking chat while its file tab is up leaves
  // nothing to answer.
  const closeBtn = chatEl.querySelector('.chat-tab-close');
  assert.ok(closeBtn, 'the chat tab has a close button');
  closeBtn.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.strictEqual(win._testApi.getActiveTabId(), fileTab, 'file tab stays');
  assert.ok(!answering(win));
  assert.strictEqual(
    win.document.getElementById('task-input').placeholder,
    'Ask anything',
  );
});

test('a prompt typed before the question is parked, not sent as the answer', () => {
  const {win, posted, tab} = makeWebview(true);
  const inp = win.document.getElementById('task-input');
  inp.value = 'run the deployment';
  askQuestionCall(win, tab, 'Which region?');
  send(win, {type: 'askUser', question: 'Which region?', tabId: tab});
  assert.strictEqual(inp.value, '', 'the composer starts empty for the answer');
  posted.length = 0;
  pressEnter(win);
  assert.strictEqual(
    posted.filter(m => m.type === 'userAnswer').length,
    0,
    'an empty composer sends nothing',
  );
  inp.value = 'eu-west';
  pressEnter(win);
  assert.deepStrictEqual(
    posted.filter(m => m.type === 'userAnswer').map(a => a.answer),
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
  const {win, posted, tab} = makeWebview(true);
  const inp = win.document.getElementById('task-input');
  inp.value = 'old prompt';
  askQuestionCall(win, tab, 'Which region?');
  send(win, {type: 'askUser', question: 'Which region?', tabId: tab});
  inp.value = 'half an ans';
  send(win, {type: 'askUserDone', tabId: tab});
  assert.strictEqual(inp.value, 'half an ans\nold prompt');
  posted.length = 0;
  pressEnter(win);
  assert.strictEqual(posted.filter(m => m.type === 'userAnswer').length, 0);
  assert.strictEqual(
    posted.filter(m => m.type === 'appendUserMessage').length,
    1,
    'without a question the text is a prompt again',
  );
});

test('a background tab parks its prompt too and shows it again once answered', () => {
  const {win, posted, tab} = makeWebview(true);
  const inp = win.document.getElementById('task-input');
  inp.value = 'draft for the first tab';
  win._testApi.createNewTab();
  askQuestionCall(win, tab, 'Q?');
  send(win, {type: 'askUser', question: 'Q?', tabId: tab});
  const tabEl = win.document.querySelector(
    `.chat-tab[data-tab-id=${JSON.stringify(tab)}]`,
  );
  tabEl.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.strictEqual(inp.value, '', 'the parked prompt is not the answer');
  assert.ok(answering(win));
  posted.length = 0;
  inp.value = 'yes';
  pressEnter(win);
  assert.deepStrictEqual(
    posted.filter(m => m.type === 'userAnswer').map(a => a.answer),
    ['yes'],
  );
  assert.strictEqual(inp.value, 'draft for the first tab');
});

test('the task ending gives the parked prompt back', () => {
  const {win, tab} = makeWebview(true);
  const inp = win.document.getElementById('task-input');
  inp.value = 'next step';
  send(win, {type: 'askUser', question: 'Q?', tabId: tab});
  assert.strictEqual(inp.value, '');
  send(win, {type: 'task_done', tabId: tab, success: true});
  assert.ok(!answering(win));
  assert.strictEqual(inp.value, 'next step');
});

test('the parked prompt and the answer in progress both survive a reload', () => {
  const {win, tab} = makeWebview(true);
  win._testApi.endLaunch();
  const inp = win.document.getElementById('task-input');
  inp.value = 'parked prompt';
  send(win, {type: 'askUser', question: 'Q?', tabId: tab});
  inp.value = 'typed answer';
  win.dispatchEvent(new win.Event('pagehide'));
  const state = win._vscodeState;
  assert.strictEqual(
    JSON.stringify(state.inputDrafts),
    JSON.stringify({[tab]: 'parked prompt'}),
  );
  assert.strictEqual(
    JSON.stringify(state.askDrafts),
    JSON.stringify({[tab]: {question: 'Q?', answer: 'typed answer'}}),
  );
});

test('a send that waits for an attachment keeps its route', async () => {
  const {win, posted, tab} = makeWebview(true);
  win._testApi.endLaunch();
  // The HEIC decoder the page asks the browser for, parked forever.
  win.createImageBitmap = () => new Promise(() => {});
  const inp = win.document.getElementById('task-input');
  const heic = new win.File([new Uint8Array([1, 2, 3])], 'IMG_0001.HEIC', {
    type: 'image/heic',
  });
  const paste = new win.Event('paste', {bubbles: true, cancelable: true});
  Object.defineProperty(paste, 'clipboardData', {
    value: {items: [{kind: 'file', getAsFile: () => heic}]},
  });
  inp.dispatchEvent(paste);
  await new Promise(r => setTimeout(r, 50));
  // A prompt parked behind the conversion: a question arriving during
  // the wait must not turn it into an answer.
  inp.value = 'a prompt with a photo';
  posted.length = 0;
  pressEnter(win);
  await new Promise(r => setTimeout(r, 20));
  send(win, {type: 'askUser', question: 'Q?', tabId: tab});
  assert.ok(answering(win));
  // The stuck chip is removed, releasing the parked send.
  win.document
    .querySelector('.file-chip .fc-rm')
    .dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  await new Promise(r => setTimeout(r, 50));
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
  inp.dispatchEvent(paste);
  await new Promise(r => setTimeout(r, 50));
  posted.length = 0;
  inp.value = 'the answer';
  pressEnter(win);
  assert.deepStrictEqual(
    posted.filter(m => m.type === 'userAnswer').map(a => a.answer),
    ['the answer'],
  );
  assert.ok(win.document.querySelector('.file-chip'), 'the photo chip stays');
  assert.strictEqual(inp.value, 'a prompt with a photo');
});

test('the send button answers too', () => {
  const {win, posted, tab} = makeWebview(true);
  askQuestionCall(win, tab, 'Button?');
  send(win, {type: 'askUser', question: 'Button?', tabId: tab});
  posted.length = 0;
  win.document.getElementById('task-input').value = 'clicked';
  win.document.getElementById('send-btn').click();
  assert.strictEqual(posted.filter(m => m.type === 'userAnswer').length, 1);
});

test('a pending question is never folded by the streaming collapse pass', () => {
  const {win, tab} = makeWebview(true);
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
  // Answered: the next events fold it like any other panel.
  send(win, {type: 'askUserDone', tabId: tab});
  bash(5);
  assert.ok(question.classList.contains('collapsed'));
});

test('answers stay out of the prompt history, and history recall is off while answering', () => {
  const {win, posted, tab} = makeWebview(true);
  const inp = win.document.getElementById('task-input');
  function key(k) {
    inp.dispatchEvent(
      new win.KeyboardEvent('keydown', {key: k, bubbles: true}),
    );
  }
  // A prompt in the history first.
  inp.value = 'previous task prompt';
  pressEnter(win);
  send(win, {type: 'askUser', question: 'Q?', tabId: tab});
  key('ArrowUp');
  assert.strictEqual(
    inp.value,
    '',
    'no prompt is recalled into the answer box',
  );
  inp.value = 'secret answer';
  pressEnter(win);
  assert.deepStrictEqual(
    posted.filter(m => m.type === 'userAnswer').map(a => a.answer),
    ['secret answer'],
  );
  key('ArrowUp');
  assert.strictEqual(
    inp.value,
    'previous task prompt',
    'history recall offers the prompt, never the answer',
  );
});

test('the Send button works for an answer while a photo is still converting', async () => {
  const {win, posted, tab} = makeWebview(true);
  win._testApi.endLaunch();
  win.createImageBitmap = () => new Promise(() => {});
  const inp = win.document.getElementById('task-input');
  const heic = new win.File([new Uint8Array([1, 2, 3])], 'IMG_0001.HEIC', {
    type: 'image/heic',
  });
  const paste = new win.Event('paste', {bubbles: true, cancelable: true});
  Object.defineProperty(paste, 'clipboardData', {
    value: {items: [{kind: 'file', getAsFile: () => heic}]},
  });
  inp.dispatchEvent(paste);
  await new Promise(r => setTimeout(r, 50));
  const sendBtn = win.document.getElementById('send-btn');
  assert.ok(sendBtn.disabled, 'a prompt waits for the photo');
  send(win, {type: 'askUser', question: 'Q?', tabId: tab});
  assert.ok(!sendBtn.disabled, 'an answer does not');
  posted.length = 0;
  inp.value = 'yes';
  sendBtn.click();
  assert.deepStrictEqual(
    posted.filter(m => m.type === 'userAnswer').map(a => a.answer),
    ['yes'],
  );
  assert.ok(
    sendBtn.disabled,
    'back to waiting for the photo for the next prompt',
  );
});

test('the clear button reflects the parked prompt that comes back', () => {
  const {win, tab} = makeWebview(true);
  const inp = win.document.getElementById('task-input');
  const clearBtn = win.document.getElementById('input-clear-btn');
  inp.value = 'next prompt';
  inp.dispatchEvent(new win.Event('input', {bubbles: true}));
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

test('the floating modal is gone from the markup', () => {
  const {win} = makeWebview(true);
  assert.strictEqual(win.document.getElementById('ask-user-modal'), null);
  assert.strictEqual(win.document.querySelector('.ask-user-input'), null);
});

Promise.all(pending).then(() => {
  console.log(`\n${passed} passed, ${failures.length} failed`);
  process.exit(failures.length > 0 ? 1 : 0);
});
