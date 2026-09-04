// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) tests for the chat tab's status dot after a task
// whose Result panel says "Status: FAILED".
//
// The daemon's terminal `task_done` event carries no `success` field
// (server/task_runner.py broadcasts `{type, tabId, startTs, endTs}`);
// the verdict lives on the `result` event that precedes it.  main.js
// recorded that verdict (`tab.lastTaskFailed = true` in streamEnd) and
// then, one event later, `markTabDone(tabId, ev.success === false)`
// overwrote it with `false` -- so a live failed task ended with a
// green dot while the same task reopened from history (a `task_events`
// replay, which no `task_done` follows) showed a red one.  These tests
// pin the live sequence, the replay, and the reset of the flag when a
// tab replays a DIFFERENT task than the one it last flagged.

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

  const posted = [];
  let state;
  win.acquireVsCodeApi = function () {
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
  win.eval(
    fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8') +
      '\n//# sourceURL=audit0902-faildot-main.js',
  );

  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function tabStrip(win, tabId) {
  const el = win.document.querySelector(
    `.chat-tab[data-tab-id=${JSON.stringify(tabId)}]`,
  );
  assert.ok(el, `tab ${tabId} must exist in the tab bar`);
  return el;
}

function dotOf(win, tabId) {
  const strip = tabStrip(win, tabId);
  if (strip.querySelector('.chat-tab-fail')) return 'fail';
  if (strip.querySelector('.chat-tab-ok')) return 'ok';
  if (strip.querySelector('.chat-tab-spinner')) return 'running';
  return 'none';
}

// The daemon's live sequence for one task in *tabId*, exactly as
// commands.py / task_runner.py / json_printer.py broadcast it.
function runTask(win, tabId, taskId, success) {
  send(win, {type: 'setTaskText', text: 'do the thing', tabId});
  send(win, {type: 'clear', chat_id: 'chat-' + tabId, tabId});
  send(win, {
    type: 'status',
    running: true,
    tabId,
    startTs: Date.now() - 1000,
    taskId,
  });
  assert.strictEqual(dotOf(win, tabId), 'running');
  send(win, {type: 'text_delta', text: 'working', tabId, taskId});
  send(win, {type: 'text_end', tabId, taskId});
  send(win, {
    type: 'result',
    text: success ? 'done' : 'it broke',
    summary: success ? '<p>done</p>' : '<p>it broke</p>',
    success,
    is_continue: false,
    total_tokens: 10,
    cost: '$0.0001',
    step_count: 1,
    tabId,
    taskId,
  });
  // task_done carries NO success field.
  send(win, {
    type: 'task_done',
    tabId,
    startTs: Date.now() - 1000,
    endTs: Date.now(),
  });
  send(win, {type: 'status', running: false, tabId, taskId});
}

function testLiveFailedTaskShowsRedDot() {
  const {win} = makeWebview();
  const tabId = win._testApi.getActiveTabId();
  runTask(win, tabId, 'task-1', false);
  assert.ok(
    win.document.querySelector('#output .rc-status-fail'),
    'precondition: the Result panel says FAILED',
  );
  assert.strictEqual(
    dotOf(win, tabId),
    'fail',
    'a task whose result reports success:false ends with the red dot',
  );
  win.close();
  console.log('  ok - live failed task shows the red dot');
}

function testLiveSuccessfulTaskShowsGreenDot() {
  const {win} = makeWebview();
  const tabId = win._testApi.getActiveTabId();
  runTask(win, tabId, 'task-2', true);
  assert.strictEqual(dotOf(win, tabId), 'ok');
  // A later failed task in the same tab flips it; a later success
  // flips it back (the `clear` of each new task resets the flag).
  runTask(win, tabId, 'task-3', false);
  assert.strictEqual(dotOf(win, tabId), 'fail');
  runTask(win, tabId, 'task-4', true);
  assert.strictEqual(dotOf(win, tabId), 'ok');
  win.close();
  console.log('  ok - live successful task shows the green dot');
}

function testBackgroundFailedTaskShowsRedDot() {
  const {win} = makeWebview();
  const tabA = win._testApi.getActiveTabId();
  win._testApi.createNewTab();
  const tabB = win._testApi.getActiveTabId();
  // Tab A runs and fails while the user reads tab B (its task_done then
  // brings the user to it, as a finished task may).
  assert.strictEqual(win._testApi.getActiveTabId(), tabB);
  runTask(win, tabA, 'task-5', false);
  assert.strictEqual(
    dotOf(win, tabA),
    'fail',
    'a background tab keeps its failed verdict too',
  );
  assert.strictEqual(dotOf(win, tabB), 'none', 'tab B never ran a task');
  win.close();
  console.log('  ok - background failed task shows the red dot');
}

function testTerminalErrorStillWins() {
  const {win} = makeWebview();
  const tabId = win._testApi.getActiveTabId();
  send(win, {type: 'setTaskText', text: 'x', tabId});
  send(win, {type: 'clear', chat_id: 'chat-x', tabId});
  send(win, {type: 'status', running: true, tabId, startTs: Date.now()});
  send(win, {
    type: 'task_error',
    tabId,
    startTs: Date.now(),
    endTs: Date.now(),
  });
  send(win, {type: 'status', running: false, tabId});
  assert.strictEqual(
    dotOf(win, tabId),
    'fail',
    'task_error flags the tab without any result event',
  );
  win.close();
  console.log('  ok - task_error still flags the tab');
}

function testReplayMirrorsTheReplayedTask() {
  const {win} = makeWebview();
  const tabId = win._testApi.getActiveTabId();
  runTask(win, tabId, 'task-6', false);
  assert.strictEqual(dotOf(win, tabId), 'fail');

  // The user reopens an OLDER, successful task of the chat in this
  // tab: the dot must describe what the tab now shows.
  send(win, {
    type: 'task_events',
    tabId,
    chat_id: 'chat-' + tabId,
    task_id: 'task-0',
    task: 'older task',
    events: [
      {type: 'text_delta', text: 'fine'},
      {type: 'text_end'},
      {type: 'result', text: 'done', success: true},
    ],
  });
  send(win, {type: 'status', running: false, tabId});
  assert.strictEqual(
    dotOf(win, tabId),
    'ok',
    'replaying a successful task clears the previous failure flag',
  );

  // And replaying a failed one flags it.
  send(win, {
    type: 'task_events',
    tabId,
    chat_id: 'chat-' + tabId,
    task_id: 'task-6',
    task: 'do the thing',
    events: [{type: 'result', text: 'it broke', success: false}],
  });
  send(win, {type: 'status', running: false, tabId});
  assert.strictEqual(dotOf(win, tabId), 'fail');

  // Background twin: a hidden tab replays a successful task.
  win._testApi.createNewTab();
  const tabB = win._testApi.getActiveTabId();
  send(win, {
    type: 'task_events',
    tabId,
    chat_id: 'chat-' + tabId,
    task_id: 'task-0',
    task: 'older task',
    events: [{type: 'result', text: 'done', success: true}],
  });
  send(win, {type: 'status', running: false, tabId});
  assert.strictEqual(win._testApi.getActiveTabId(), tabB);
  assert.strictEqual(
    dotOf(win, tabId),
    'ok',
    'a background replay of a successful task clears the flag too',
  );
  win.close();
  console.log('  ok - a replay mirrors the replayed task');
}

function main() {
  testLiveFailedTaskShowsRedDot();
  testLiveSuccessfulTaskShowsGreenDot();
  testBackgroundFailedTaskShowsRedDot();
  testTerminalErrorStillWins();
  testReplayMirrorsTheReplayedTask();
  console.log('all audit0902 tab-fail-dot tests passed');
}

main();
