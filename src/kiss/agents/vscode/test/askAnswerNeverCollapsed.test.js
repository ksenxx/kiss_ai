// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// The `/ask` answer panel (an `ask_answer` event) is never folded or
// hidden by any automatic pass of the chat webview, on any surface:
//
//   * collapseOlderPanels  -- the streaming sweep of a running task;
//   * collapseAllExceptResult -- a task_events replay (reload, reattach,
//     background tab, neighbouring task) and the share export;
//   * applyChevronState -- the finished-task digest that takes every
//     plain panel off screen (chv-hidden);
//   * the `summary` tool call, which adopts the panels before it into a
//     collapsed .summary-sub.
//
// Only the user folds it, by clicking its header.  The share export is
// built from the same replay path, so the exported page shows the
// answer open too.

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
  win.cancelAnimationFrame = function () {};

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

const QUESTION = 'why did step 3 fail?';
const ANSWER = '<p>Step 3 failed because <code>config.toml</code> was missing.</p>';

function askAnswerEvent(extra) {
  return Object.assign(
    {type: 'ask_answer', question: QUESTION, text: ANSWER, success: true},
    extra || {},
  );
}

/** Persisted events of a finished task with an answer in the middle. */
function finishedTaskEvents(taskId) {
  const events = [{type: 'prompt', text: 'fix the bug'}];
  for (let i = 0; i < 3; i++) {
    events.push({type: 'tool_call', name: 'Read', path: '/tmp/r' + i});
    events.push({type: 'tool_result', name: 'Read', content: 'x' + i});
  }
  events.push({type: 'prompt', text: '/ask ' + QUESTION});
  events.push(askAnswerEvent({taskId: taskId, ts: 1700000000000}));
  events.push({type: 'tool_call', name: 'Read', path: '/tmp/last'});
  events.push({type: 'tool_result', name: 'Read', content: 'last'});
  events.push({type: 'result', text: 'done', summary: 'done', success: true});
  return events;
}

function answerPanelIn(root) {
  const panels = root.querySelectorAll('.ev.ask-answer');
  assert.strictEqual(panels.length, 1, 'exactly one answer panel renders');
  return panels[0];
}

function assertOpenAndOnScreen(panel, where) {
  assert.ok(
    !panel.classList.contains('collapsed'),
    'BUG: the answer panel was auto-collapsed ' + where,
  );
  assert.ok(
    !panel.classList.contains('chv-hidden'),
    'BUG: the answer panel was hidden ' + where,
  );
  assert.ok(
    !panel.closest('.summary-sub'),
    'BUG: the answer panel was swallowed by a summary ' + where,
  );
}

function testReplayOfFinishedTaskKeepsAnswerOpen() {
  const {win} = makeWebview();
  send(win, {
    type: 'task_events',
    task: 'fix the bug',
    task_id: 42,
    events: finishedTaskEvents(42),
  });
  const out = win.document.getElementById('output');
  const tools = out.querySelectorAll('.tc');
  assert.strictEqual(tools.length, 4, 'four tool panels replay');
  assert.ok(
    Array.from(tools).every(p => p.classList.contains('collapsed')),
    'plain replayed tool panels are folded',
  );
  assert.ok(
    Array.from(tools).every(p => p.classList.contains('chv-hidden')),
    'plain replayed tool panels of a finished task are taken off screen',
  );
  assertOpenAndOnScreen(answerPanelIn(out), 'on a finished-task replay');
  win.close();
  console.log('  ok - a finished-task replay keeps the answer open and visible');
}

function testReplayOfRunningTaskKeepsAnswerOpen() {
  // A client reconnecting to a task that is still running: the replay
  // folds everything but the result; the answer must survive both the
  // replay pass and the streaming sweep that follows.
  const {win} = makeWebview();
  const tab = win._testApi.getActiveTabId();
  send(win, {type: 'status', running: true, tabId: tab});
  const events = finishedTaskEvents(43);
  events.pop(); // still running: no result yet
  send(win, {
    type: 'task_events',
    task: 'fix the bug',
    task_id: 43,
    tabId: tab,
    events: events,
  });
  const out = win.document.getElementById('output');
  assertOpenAndOnScreen(answerPanelIn(out), 'on a mid-run replay');
  send(win, {type: 'tool_call', name: 'Bash', command: 'ls', tabId: tab});
  send(win, {type: 'tool_result', content: 'a', tool_name: 'Bash', tabId: tab});
  send(win, {type: 'tool_call', name: 'Bash', command: 'pwd', tabId: tab});
  assertOpenAndOnScreen(answerPanelIn(out), 'after the run streamed on');
  win.close();
  console.log('  ok - a mid-run replay keeps the answer open through the stream');
}

function testBackgroundTabRestoreKeepsAnswerOpen() {
  // The answer arrives in a tab that is not on screen; the tab is
  // later restored, which runs the collapse pass over its fragment.
  const {win} = makeWebview();
  const api = win._testApi;
  const tab1 = api.getActiveTabId();
  send(win, {type: 'status', running: true, tabId: tab1});
  send(win, {type: 'task_settings', task_id: '42', tabId: tab1});
  api.createNewTab();
  const tab2 = api.getActiveTabId();
  assert.notStrictEqual(tab2, tab1);
  send(win, {type: 'prompt', text: '/ask ' + QUESTION, tabId: tab1});
  send(win, askAnswerEvent({tabId: tab1, taskId: '42'}));
  for (let i = 0; i < 3; i++) {
    send(win, {type: 'tool_call', name: 'Bash', command: 'ls ' + i, tabId: tab1});
    send(win, {type: 'tool_result', content: 'a', tool_name: 'Bash', tabId: tab1});
  }
  win.document.querySelector('.chat-tab[data-tab-id="' + tab1 + '"]').click();
  assert.strictEqual(api.getActiveTabId(), tab1);
  const out = win.document.getElementById('output');
  const prompt = out.querySelector('.ev.prompt');
  assert.ok(prompt.classList.contains('collapsed'), 'the prompt echo folds');
  assertOpenAndOnScreen(answerPanelIn(out), 'when a background tab came back');
  win.close();
  console.log('  ok - restoring a background tab keeps the answer open');
}

function testSummaryToolLeavesAnswerOnTranscript() {
  const {win} = makeWebview();
  const tab = win._testApi.getActiveTabId();
  send(win, {type: 'status', running: true, tabId: tab});
  send(win, {type: 'tool_call', name: 'Bash', command: 'ls', tabId: tab});
  send(win, {type: 'tool_result', content: 'a', tool_name: 'Bash', tabId: tab});
  send(win, askAnswerEvent({tabId: tab, taskId: '42'}));
  send(win, {type: 'tool_call', name: 'Bash', command: 'pwd', tabId: tab});
  send(win, {type: 'tool_result', content: '/', tool_name: 'Bash', tabId: tab});
  send(win, {type: 'tool_call', name: 'summary', description: 'so far', tabId: tab});
  const out = win.document.getElementById('output');
  const summary = out.querySelector('.tc-summary');
  assert.ok(summary, 'the summary panel renders');
  assert.ok(summary.classList.contains('collapsed'), 'the summary folds');
  const adopted = summary.querySelectorAll('.summary-sub > .tc');
  assert.strictEqual(
    adopted.length,
    2,
    'the summary adopts the tool panels on both sides of the answer',
  );
  const panel = answerPanelIn(out);
  assertOpenAndOnScreen(panel, 'when a summary adopted its neighbours');
  assert.strictEqual(
    panel.nextElementSibling,
    summary,
    'the answer sits on the transcript right before the summary',
  );
  win.close();
  console.log('  ok - the summary tool leaves the answer on the transcript');
}

function testReplayedSummaryLeavesAnswerVisible() {
  // The same on a finished replay, where the chevron pass also hides
  // every plain panel and re-collapses the summary.
  const {win} = makeWebview();
  const events = finishedTaskEvents(44);
  events.splice(events.length - 1, 0, {
    type: 'tool_call',
    name: 'summary',
    description: 'so far',
  });
  send(win, {type: 'task_events', task: 'fix the bug', task_id: 44, events});
  const out = win.document.getElementById('output');
  const summary = out.querySelector('.tc-summary');
  assert.ok(summary.classList.contains('collapsed'), 'the summary folds');
  assert.ok(
    !summary.classList.contains('chv-hidden'),
    'the summary stays on screen',
  );
  // The adoption walks back over the answer to the `/ask` prompt echo,
  // which (like any prompt) bounds it: one tool panel is adopted.
  assert.strictEqual(
    summary.querySelectorAll('.summary-sub > .tc').length,
    1,
    'the summary adopts the tool panel after the answer, not the answer',
  );
  assertOpenAndOnScreen(answerPanelIn(out), 'on a replay with a summary');
  win.close();
  console.log('  ok - a replayed summary leaves the answer visible');
}

function testShareExportKeepsAnswerOpen() {
  const {win, posted} = makeWebview();
  const tab = win._testApi.getActiveTabId();
  send(win, {type: 'clear', chat_id: 'chat-1', tabId: tab});
  send(win, {type: 'status', running: true, tabId: tab});
  send(win, {type: 'setTaskText', text: 'fix the bug', tabId: tab});
  send(win, {type: 'tool_call', name: 'Bash', command: 'ls', tabId: tab, taskId: 'task-1'});
  send(win, {type: 'result', text: 'done', tabId: tab, taskId: 'task-1'});
  send(win, {type: 'status', running: false, tabId: tab});
  win.document.getElementById('share-btn').dispatchEvent(
    new win.MouseEvent('click', {bubbles: true, cancelable: true}),
  );
  const req = posted.filter(m => m.type === 'shareChatTasks').pop();
  assert.ok(req, 'share asks the daemon for the chat tasks');
  // The on-screen task is exported as it stands; an earlier task of the
  // chat is rebuilt from its persisted events -- the replay path the
  // export shares with a reload.
  send(win, {
    type: 'share_tasks',
    tabId: req.tabId,
    chatId: req.chatId,
    tasks: [
      {task: 'earlier task', task_id: 'task-0', events: finishedTaskEvents('task-0')},
      {task: 'fix the bug', task_id: 'task-1', events: []},
    ],
    truncated: false,
  });
  const msg = posted.filter(m => m.type === 'shareChat').pop();
  assert.ok(msg, 'the share_tasks reply produces a shareChat command');
  const page = new JSDOM('<!doctype html><html><body>' + msg.html + '</body></html>');
  const panel = answerPanelIn(page.window.document.body);
  const out = panel.closest('[id="output"]');
  assert.ok(out, 'the exported answer sits in a transcript');
  // The on-screen task's live `ls` panel shares the transcript; the
  // earlier task's four Read panels are the replayed ones.
  const tools = Array.from(out.querySelectorAll('.tc')).filter(p =>
    p.textContent.includes('/tmp/'),
  );
  assert.strictEqual(tools.length, 4, 'the earlier task exports its tools');
  assert.ok(
    tools.every(p => p.classList.contains('collapsed')),
    'exported tool panels are folded',
  );
  assertOpenAndOnScreen(panel, 'in the share export');
  win.close();
  console.log('  ok - the share export shows the answer open');
}

function testUserStillFoldsByHand() {
  const {win} = makeWebview();
  const tab = win._testApi.getActiveTabId();
  send(win, askAnswerEvent({tabId: tab}));
  const panel = answerPanelIn(win.document.getElementById('output'));
  panel.querySelector('.ask-answer-h').click();
  assert.ok(panel.classList.contains('collapsed'), 'a header click folds it');
  panel.querySelector('.ask-answer-h').click();
  assert.ok(!panel.classList.contains('collapsed'), 'and unfolds it again');
  win.close();
  console.log('  ok - the user still folds the answer by its header');
}

const tests = [
  testReplayOfFinishedTaskKeepsAnswerOpen,
  testReplayOfRunningTaskKeepsAnswerOpen,
  testBackgroundTabRestoreKeepsAnswerOpen,
  testSummaryToolLeavesAnswerOnTranscript,
  testReplayedSummaryLeavesAnswerVisible,
  testShareExportKeepsAnswerOpen,
  testUserStillFoldsByHand,
];

try {
  for (const t of tests) t();
  console.log('\n' + tests.length + ' passed, 0 failed');
} catch (e) {
  console.error(e);
  process.exit(1);
}
