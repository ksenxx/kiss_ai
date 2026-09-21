// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// The `/ask` side channel: the answering sub-agent runs in a nested tab
// that is closed the moment it finishes (subagentDone), so the daemon
// delivers the reply as a separate `ask_answer` event into the OWNER
// task's transcript.  These tests pin that the event renders as its own
// "Answer" panel -- live in the active tab, live in a background tab,
// and on a task_events replay -- and that the answer HTML is sanitized.

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

  return {win};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

const QUESTION = 'why did step 3 fail?';
const ANSWER_HTML =
  '<p>Step 3 failed because <code>config.toml</code> was missing.</p>' +
  '<script>window.__xss = 1</script>';

function askAnswerEvent(extra) {
  return Object.assign(
    {
      type: 'ask_answer',
      question: QUESTION,
      text: ANSWER_HTML,
      success: true,
    },
    extra || {},
  );
}

function assertAnswerPanel(win, panel) {
  assert.ok(panel, 'an .ev.ask-answer panel must be rendered');
  const header = panel.querySelector('.ask-answer-h');
  assert.ok(header, 'panel must have a header');
  assert.strictEqual(
    header.querySelector('.ask-answer-label').textContent,
    'Answer',
  );
  assert.strictEqual(
    header.querySelector('.ask-answer-q').textContent,
    QUESTION,
    'the header must quote the question',
  );
  const body = panel.querySelector('.ask-answer-body');
  assert.ok(body, 'panel must have a body');
  assert.ok(
    body.textContent.includes('Step 3 failed because'),
    'the answer text must be in the body',
  );
  assert.ok(
    body.querySelector('code'),
    'the answer HTML must be rendered as HTML, not escaped',
  );
  assert.strictEqual(
    body.querySelector('script'),
    null,
    'script tags in the answer must be sanitized away',
  );
  assert.strictEqual(win.__xss, undefined, 'sanitized script must not run');
  assert.ok(
    panel.classList.contains('collapsible'),
    'panel must be collapsible like a prompt panel',
  );
  assert.ok(
    panel.querySelector(':scope > .panel-copy-btn'),
    'panel must have a copy button',
  );
  assert.ok(
    panel.dataset.rawText.includes('Step 3 failed because config.toml'),
    'copy text must be the formatted answer, got: ' + panel.dataset.rawText,
  );
  assert.ok(
    !panel.dataset.rawText.includes('<p>'),
    'copy text must not contain raw HTML markup',
  );
}

function testLiveAnswerRendersInActiveTab() {
  const {win} = makeWebview();
  const api = win._testApi;
  const tab = api.getActiveTabId();
  send(win, {type: 'status', running: true, tabId: tab});
  send(win, {type: 'text_delta', text: 'Working on it.', tabId: tab});
  send(win, {type: 'prompt', text: '/ask ' + QUESTION, tabId: tab});
  send(win, askAnswerEvent({tabId: tab, taskId: '42'}));

  const output = win.document.getElementById('output');
  const panels = output.querySelectorAll('.ev.ask-answer');
  assert.strictEqual(panels.length, 1, 'exactly one answer panel');
  assertAnswerPanel(win, panels[0]);
  assert.ok(
    !panels[0].classList.contains('failed'),
    'a successful answer must not be marked failed',
  );
  // The panel is a top-level transcript entry (not nested inside the
  // agent's thoughts panel), so it stays visible as its own event.
  assert.strictEqual(
    panels[0].parentElement,
    output,
    'answer panel must be a direct child of the transcript',
  );
  win.close();
  console.log(
    '  ok - live ask_answer renders as its own panel in the active tab',
  );
}

function testFailedAnswerIsMarked() {
  const {win} = makeWebview();
  const tab = win._testApi.getActiveTabId();
  send(
    win,
    askAnswerEvent({
      tabId: tab,
      success: false,
      text: 'The /ask agent failed: daemon is down',
    }),
  );
  const panel = win.document.querySelector('#output .ev.ask-answer');
  assert.ok(panel, 'failed answer must still render');
  assert.ok(panel.classList.contains('failed'), 'must carry .failed');
  assert.strictEqual(
    panel.querySelector('.ask-answer-label').textContent,
    'Answer (failed)',
  );
  assert.ok(
    panel
      .querySelector('.ask-answer-body')
      .textContent.includes('daemon is down'),
  );
  win.close();
  console.log('  ok - a failed ask_answer is marked as failed');
}

function testAnswerStaysOpenWhileTaskStreams() {
  const {win} = makeWebview();
  const tab = win._testApi.getActiveTabId();
  send(win, {type: 'status', running: true, tabId: tab});
  send(win, {type: 'prompt', text: '/ask ' + QUESTION, tabId: tab});
  send(win, askAnswerEvent({tabId: tab, taskId: '42'}));
  // The running task keeps going: its next events fold older panels
  // away (collapseOlderPanels) -- the prompt echo folds, the answer
  // the user asked for must not.
  send(win, {type: 'text_delta', text: 'Continuing with the fix.', tabId: tab});
  send(win, {type: 'text_end', tabId: tab});
  send(win, {type: 'tool_call', name: 'Bash', command: 'ls', tabId: tab});
  const prompt = win.document.querySelector('#output .ev.prompt');
  assert.ok(prompt.classList.contains('collapsed'), 'prompt echo folds');
  const panel = win.document.querySelector('#output .ev.ask-answer');
  assert.ok(
    !panel.classList.contains('collapsed'),
    'the answer must stay open while the task streams on',
  );
  // The user can still fold it by hand.
  panel.querySelector('.ask-answer-h').click();
  assert.ok(panel.classList.contains('collapsed'), 'manual collapse works');
  win.close();
  console.log('  ok - the answer stays open while the task keeps streaming');
}

function testHeaderClickCollapsesBody() {
  const {win} = makeWebview();
  const tab = win._testApi.getActiveTabId();
  send(win, askAnswerEvent({tabId: tab}));
  const panel = win.document.querySelector('#output .ev.ask-answer');
  assert.ok(!panel.classList.contains('collapsed'), 'starts expanded');
  panel.querySelector('.ask-answer-h').click();
  assert.ok(panel.classList.contains('collapsed'), 'header click collapses');
  panel.querySelector('.ask-answer-h').click();
  assert.ok(!panel.classList.contains('collapsed'), 'second click expands');
  win.close();
  console.log('  ok - header click toggles the answer panel');
}

function testBackgroundTabAnswerSurvivesTabSwitch() {
  const {win} = makeWebview();
  const api = win._testApi;
  const tab1 = api.getActiveTabId();
  api.createNewTab();
  const tab2 = api.getActiveTabId();
  assert.ok(tab2 && tab2 !== tab1, 'a fresh second tab must be active');

  send(win, askAnswerEvent({tabId: tab1, taskId: '42'}));

  assert.strictEqual(
    win.document.querySelector('#output .ev.ask-answer'),
    null,
    'a background tab answer must not render in the active tab',
  );

  win.document.querySelector('.chat-tab[data-tab-id="' + tab1 + '"]').click();
  assert.strictEqual(api.getActiveTabId(), tab1);
  const panel = win.document.querySelector('#output .ev.ask-answer');
  assertAnswerPanel(win, panel);
  win.close();
  console.log(
    '  ok - background-tab ask_answer is shown when the tab is restored',
  );
}

function testReplayRendersPersistedAnswer() {
  const {win} = makeWebview();
  send(win, {
    type: 'task_events',
    task: 'fix the bug',
    task_id: 42,
    events: [
      {type: 'text_delta', text: 'Working on it.'},
      {type: 'text_end'},
      {type: 'prompt', text: '/ask ' + QUESTION},
      // Persisted copy: tabId stripped, taskId + ts kept.
      askAnswerEvent({taskId: 42, ts: 1700000000000}),
      {type: 'result', text: 'summary: done', total_tokens: 10, cost: '$0.01'},
    ],
  });
  const panels = win.document.querySelectorAll('#output .ev.ask-answer');
  assert.strictEqual(panels.length, 1, 'replay must render the answer once');
  assertAnswerPanel(win, panels[0]);
  win.close();
  console.log('  ok - a persisted ask_answer renders on task_events replay');
}

function testForeignTabAnswerDropped() {
  const {win} = makeWebview();
  send(win, askAnswerEvent({tabId: 'some-other-window-tab', taskId: '7'}));
  assert.strictEqual(
    win.document.querySelector('#output .ev.ask-answer'),
    null,
    'an answer for an unknown tab must not leak into this window',
  );
  win.close();
  console.log('  ok - ask_answer for a foreign tab is dropped');
}

function testStaleAnswerForReusedTabIsDropped() {
  // The tab ran task 100, asked a question, then started task 200 in
  // the same tab.  The answer to the task-100 question arrives now.
  const {win} = makeWebview();
  const api = win._testApi;
  const tab = api.getActiveTabId();
  send(win, {type: 'status', running: true, tabId: tab});
  send(win, {
    type: 'text_delta',
    text: 'task 100 output',
    tabId: tab,
    taskId: '100',
  });
  send(win, {type: 'text_end', tabId: tab, taskId: '100'});
  send(win, {type: 'task_done', tabId: tab, taskId: '100'});
  send(win, {type: 'status', running: false, tabId: tab});
  send(win, {type: 'status', running: true, tabId: tab});
  send(win, {
    type: 'text_delta',
    text: 'task 200 output',
    tabId: tab,
    taskId: '200',
  });
  send(win, {type: 'text_end', tabId: tab, taskId: '200'});
  // Precondition: the tab owns task 200 -- an event addressed only by
  // that task id (no tabId) is accepted on task identity alone.
  send(win, {type: 'prompt', text: 'owned by 200 PRE', taskId: '200'});
  const output = win.document.getElementById('output');
  assert.ok(
    output.textContent.includes('owned by 200 PRE'),
    'tab owns task 200',
  );

  send(win, askAnswerEvent({tabId: tab, taskId: '100'}));

  assert.strictEqual(
    output.querySelector('.ev.ask-answer'),
    null,
    'a stale answer for the finished task must not enter the new task',
  );
  // Had the stale answer re-bound the tab to task 100, this task-only
  // addressed event of the real task would now be rejected.
  send(win, {type: 'prompt', text: 'still task 200 QZX', taskId: '200'});
  assert.ok(
    output.textContent.includes('still task 200 QZX'),
    'task-200 output after the stale answer must still render ' +
      '(the stale answer must not re-bind the tab to task 100)',
  );
  // And an answer for the CURRENT task renders as usual.
  send(win, askAnswerEvent({tabId: tab, taskId: '200'}));
  assert.strictEqual(
    win.document.querySelectorAll('#output .ev.ask-answer').length,
    1,
    'an answer for the current task still renders',
  );
  win.close();
  console.log(
    '  ok - a stale answer for a reused tab is dropped without re-binding it',
  );
}

function testStaleAnswerForReusedBackgroundTabIsDropped() {
  const {win} = makeWebview();
  const api = win._testApi;
  const tab1 = api.getActiveTabId();
  send(win, {
    type: 'text_delta',
    text: 'task 100 output',
    tabId: tab1,
    taskId: '100',
  });
  send(win, {type: 'text_end', tabId: tab1, taskId: '100'});
  send(win, {
    type: 'text_delta',
    text: 'task 200 output',
    tabId: tab1,
    taskId: '200',
  });
  send(win, {type: 'text_end', tabId: tab1, taskId: '200'});
  api.createNewTab();
  assert.notStrictEqual(api.getActiveTabId(), tab1);

  send(win, askAnswerEvent({tabId: tab1, taskId: '100'}));
  win.document.querySelector('.chat-tab[data-tab-id="' + tab1 + '"]').click();
  assert.strictEqual(api.getActiveTabId(), tab1);
  assert.strictEqual(
    win.document.querySelector('#output .ev.ask-answer'),
    null,
    'a stale answer must not be buffered into a background tab either',
  );
  send(win, {type: 'prompt', text: 'still task 200 BGX', taskId: '200'});
  assert.ok(
    win.document
      .getElementById('output')
      .textContent.includes('still task 200 BGX'),
    'the restored tab must still own task 200',
  );
  win.close();
  console.log('  ok - a stale answer for a reused background tab is dropped');
}

function testStaleAnswerInPreAdoptionWindowIsDropped() {
  // Task 100 ended; the user submitted task 200 in the same tab.  The
  // daemon has sent setTaskText + clear but not yet an event carrying
  // task 200's id, so the tab still carries task 100's id.
  const {win} = makeWebview();
  const api = win._testApi;
  const tab = api.getActiveTabId();
  send(win, {
    type: 'text_delta',
    text: 'task 100 output',
    tabId: tab,
    taskId: '100',
  });
  send(win, {type: 'text_end', tabId: tab, taskId: '100'});
  send(win, {type: 'task_done', tabId: tab, taskId: '100'});
  send(win, {type: 'setTaskText', text: 'task 200', tabId: tab});
  send(win, {type: 'clear', tabId: tab, chat_id: 'chat-1'});

  send(win, askAnswerEvent({tabId: tab, taskId: '100'}));
  assert.strictEqual(
    win.document.querySelector('#output .ev.ask-answer'),
    null,
    'an answer for the replaced task must not enter the cleared transcript',
  );
  send(win, {
    type: 'text_delta',
    text: 'task 200 output',
    tabId: tab,
    taskId: '200',
  });
  send(win, {type: 'text_end', tabId: tab, taskId: '200'});
  send(win, askAnswerEvent({tabId: tab, taskId: '200'}));
  assert.strictEqual(
    win.document.querySelectorAll('#output .ev.ask-answer').length,
    1,
    'an answer for the new task renders once it is bound',
  );
  win.close();
  console.log(
    "  ok - a stale answer in the replacement run's pre-adoption window is dropped",
  );
}

function runTests() {
  testLiveAnswerRendersInActiveTab();
  testFailedAnswerIsMarked();
  testAnswerStaysOpenWhileTaskStreams();
  testHeaderClickCollapsesBody();
  testBackgroundTabAnswerSurvivesTabSwitch();
  testReplayRendersPersistedAnswer();
  testForeignTabAnswerDropped();
  testStaleAnswerForReusedTabIsDropped();
  testStaleAnswerForReusedBackgroundTabIsDropped();
  testStaleAnswerInPreAdoptionWindowIsDropped();
}

try {
  runTests();
  console.log('\n10 passed, 0 failed');
  process.exit(0);
} catch (err) {
  console.error('FAIL:', err && err.stack ? err.stack : err);
  process.exit(1);
}
