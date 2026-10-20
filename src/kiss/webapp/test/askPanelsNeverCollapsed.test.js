// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// The two ask panels -- the `/ask` answer panel (an `ask_answer` event)
// and the ask_user_question "Question" panel with its separate user
// response once the tool returns -- are never folded or hidden by any
// automatic pass of the chat page, wherever the tab is loaded (the
// remote webapp, a share export):
//
//   * collapseOlderPanels  -- the streaming sweep of a running task;
//   * collapseAllExceptResult -- a task_events replay (reload, reattach,
//     background tab, neighbouring task) and the share export;
//   * applyChevronState -- the finished-task digest that folds every
//     plain panel;
//   * the `summary` tool call, which adopts the panels before it into a
//     collapsed .summary-sub.
//
// Only the user folds them, by clicking their header.  The share export
// is built from the same replay path, so the exported page shows them
// open too.  Every scenario runs once per panel kind (the KINDS table).

'use strict';

const assert = require('assert');
const {JSDOM} = require('jsdom');
const {makePage, send, stubRaf} = require('./simplify2_harness');

// One event-loop turn before closing: the theme MutationObserver that a
// tab switch queues must run while the document still exists, so
// win.close() tears down a quiet window.
async function closeQuietly(win) {
  await new Promise(r => setTimeout(r, 0));
  win.close();
}

const ASK_QUESTION = 'why did step 3 fail?';
const ASK_ANSWER =
  '<p>Step 3 failed because <code>config.toml</code> was missing.</p>';
const USER_QUESTION = 'Which branch should I push to?';
const USER_ANSWER = 'release/2026.10';

function askAnswerEvent(extra) {
  return Object.assign(
    {
      type: 'ask_answer',
      question: ASK_QUESTION,
      text: ASK_ANSWER,
      success: true,
    },
    extra || {},
  );
}

function questionCall(extra) {
  return Object.assign(
    {
      type: 'tool_call',
      name: 'ask_user_question',
      callId: 'ask-1',
      extras: {question: USER_QUESTION},
    },
    extra || {},
  );
}

function questionResult(extra) {
  return Object.assign(
    {type: 'tool_result', tool_name: 'ask_user_question', content: USER_ANSWER},
    extra || {},
  );
}

/** The `/ask` answer panel: the `/ask` prompt echo and the answer. */
const ANSWER_KIND = {
  label: 'answer',
  task: 'fix the bug',
  selector: '.ev.ask-answer',
  header: '.ask-answer-h',
  persistedEvents: taskId => [
    {type: 'prompt', text: '/ask ' + ASK_QUESTION},
    askAnswerEvent({taskId: taskId, ts: 1700000000000}),
  ],
  // The /ask prompt echo folds under a summary next to the task prompt.
  promptsUnderSummary: 2,
  live(win, tabId, taskId) {
    send(win, askAnswerEvent({tabId, taskId}));
  },
  assertExtra() {},
};

/** The ask_user_question panel: the tool call and its answered result. */
const QUESTION_KIND = {
  label: 'question',
  task: 'ship it',
  selector: '.tc-question',
  header: '.tc-h',
  persistedEvents: taskId => [questionCall({taskId}), questionResult({taskId})],
  promptsUnderSummary: 1,
  // `otherTab` (optional): the question pulls the user over to its tab;
  // the user then picks `otherTab` and the answer lands in the background.
  live(win, tabId, taskId, otherTab) {
    const api = win._testApi;
    send(win, questionCall({tabId, taskId}));
    send(win, {type: 'askUser', question: USER_QUESTION, tabId});
    if (otherTab) {
      assert.strictEqual(
        api.getActiveTabId(),
        tabId,
        'the question pulls the user over',
      );
      // Chats are picked from the sidebar's Chats panel; switchToTab is that pick.
      api.switchToTab(otherTab);
      assert.strictEqual(api.getActiveTabId(), otherTab);
    }
    send(win, {type: 'askUserDone', tabId});
    send(win, questionResult({tabId, taskId}));
  },
  assertExtra(panel, where) {
    const response = panel.nextElementSibling;
    assert.ok(
      response.classList.contains('user-msg'),
      'the response is a user message ' + where,
    );
    assert.ok(
      !response.classList.contains('collapsed'),
      'the response stays open ' + where,
    );
    assert.strictEqual(
      response.querySelector('.task-panel-text').textContent,
      USER_ANSWER,
    );
    assert.ok(
      !panel.classList.contains('tc-question-pending'),
      'an answered question is not marked pending ' + where,
    );
  },
};

/** Persisted events of a finished task with the kind's panel in the middle. */
function finishedTaskEvents(kind, taskId) {
  const events = [{type: 'prompt', text: kind.task}];
  for (let i = 0; i < 3; i++) {
    events.push({type: 'tool_call', name: 'Read', path: '/tmp/r' + i});
    events.push({type: 'tool_result', name: 'Read', content: 'x' + i});
  }
  events.push(...kind.persistedEvents(taskId));
  events.push({type: 'tool_call', name: 'Read', path: '/tmp/last'});
  events.push({type: 'tool_result', name: 'Read', content: 'last'});
  events.push({type: 'result', text: 'done', summary: 'done', success: true});
  return events;
}

function panelIn(kind, root) {
  const panels = root.querySelectorAll(kind.selector);
  assert.strictEqual(
    panels.length,
    1,
    'exactly one ' + kind.label + ' panel renders',
  );
  return panels[0];
}

function assertOpenAndOnScreen(kind, panel, where) {
  assert.ok(
    !panel.classList.contains('collapsed'),
    'BUG: the ' + kind.label + ' panel was auto-collapsed ' + where,
  );
  assert.ok(
    !panel.classList.contains('chv-hidden'),
    'BUG: the ' + kind.label + ' panel was hidden by the digest ' + where,
  );
  assert.ok(
    !panel.closest('.summary-sub'),
    'BUG: the ' + kind.label + ' panel was swallowed by a summary ' + where,
  );
  kind.assertExtra(panel, where);
}

async function testReplayOfFinishedTaskKeepsPanelOpen(kind) {
  const {win} = makePage({beforeScripts: stubRaf});
  send(win, {
    type: 'task_events',
    task: kind.task,
    task_id: 42,
    events: finishedTaskEvents(kind, 42),
  });
  const out = win.document.getElementById('output');
  const tools = out.querySelectorAll('.tc:not(.tc-question)');
  assert.strictEqual(tools.length, 4, 'four plain tool panels replay');
  assert.ok(
    Array.from(tools).every(p => p.classList.contains('collapsed')),
    'plain replayed tool panels are folded',
  );
  assertOpenAndOnScreen(kind, panelIn(kind, out), 'on a finished-task replay');
  await closeQuietly(win);
  console.log(
    '  ok - a finished-task replay keeps the ' +
      kind.label +
      ' open and visible',
  );
}

async function testReplayOfRunningTaskKeepsPanelOpen(kind) {
  // A client reconnecting to a task that is still running: the replay
  // folds everything but the result; the panel must survive both the
  // replay pass and the streaming sweep that follows.
  const {win} = makePage({beforeScripts: stubRaf});
  const tab = win._testApi.getActiveTabId();
  send(win, {type: 'status', running: true, tabId: tab});
  const events = finishedTaskEvents(kind, 43);
  events.pop(); // still running: no result yet
  send(win, {
    type: 'task_events',
    task: kind.task,
    task_id: 43,
    tabId: tab,
    events: events,
  });
  const out = win.document.getElementById('output');
  assertOpenAndOnScreen(kind, panelIn(kind, out), 'on a mid-run replay');
  for (let i = 0; i < 3; i++) {
    send(win, {
      type: 'tool_call',
      name: 'Bash',
      command: 'ls ' + i,
      tabId: tab,
    });
    send(win, {
      type: 'tool_result',
      content: 'a',
      tool_name: 'Bash',
      tabId: tab,
    });
  }
  send(win, {type: 'tool_call', name: 'Bash', command: 'pwd', tabId: tab});
  assertOpenAndOnScreen(kind, panelIn(kind, out), 'after the run streamed on');
  await closeQuietly(win);
  console.log(
    '  ok - a mid-run replay keeps the ' +
      kind.label +
      ' open through the stream',
  );
}

async function testBackgroundTabRestoreKeepsPanelOpen(kind) {
  // The panel lands in a tab that is not on screen; the tab is later
  // restored, which runs the collapse pass over its fragment -- the same
  // path a tab takes when another surface loads it.
  const {win} = makePage({beforeScripts: stubRaf});
  const api = win._testApi;
  const tab1 = api.getActiveTabId();
  send(win, {type: 'status', running: true, tabId: tab1});
  send(win, {type: 'task_settings', task_id: '42', tabId: tab1});
  api.createNewTab();
  const tab2 = api.getActiveTabId();
  assert.notStrictEqual(tab2, tab1);
  send(win, {type: 'prompt', text: kind.task, tabId: tab1});
  kind.live(win, tab1, '42', tab2);
  for (let i = 0; i < 3; i++) {
    send(win, {
      type: 'tool_call',
      name: 'Bash',
      command: 'ls ' + i,
      tabId: tab1,
    });
    send(win, {
      type: 'tool_result',
      content: 'a',
      tool_name: 'Bash',
      tabId: tab1,
    });
  }
  // The Chats-panel pick brings the chat back on screen.
  api.switchToTab(tab1);
  assert.strictEqual(api.getActiveTabId(), tab1);
  const out = win.document.getElementById('output');
  const prompt = out.querySelector('.ev.prompt');
  assert.ok(prompt.classList.contains('collapsed'), 'the prompt echo folds');
  assertOpenAndOnScreen(
    kind,
    panelIn(kind, out),
    'when a background tab came back',
  );
  await closeQuietly(win);
  console.log(
    '  ok - restoring a background tab keeps the ' + kind.label + ' open',
  );
}

async function testSummaryToolLeavesPanelOnTranscript(kind) {
  const {win} = makePage({beforeScripts: stubRaf});
  const tab = win._testApi.getActiveTabId();
  send(win, {type: 'status', running: true, tabId: tab});
  send(win, {type: 'tool_call', name: 'Bash', command: 'ls', tabId: tab});
  send(win, {type: 'tool_result', content: 'a', tool_name: 'Bash', tabId: tab});
  kind.live(win, tab, '42');
  send(win, {type: 'tool_call', name: 'Bash', command: 'pwd', tabId: tab});
  send(win, {type: 'tool_result', content: '/', tool_name: 'Bash', tabId: tab});
  send(win, {
    type: 'tool_call',
    name: 'summary',
    description: 'so far',
    tabId: tab,
  });
  const out = win.document.getElementById('output');
  const summary = out.querySelector('.tc-summary');
  assert.ok(summary, 'the summary panel renders');
  assert.ok(summary.classList.contains('collapsed'), 'the summary folds');
  const adopted = summary.querySelectorAll('.summary-sub > .tc');
  assert.strictEqual(
    adopted.length,
    2,
    'the summary adopts the tool panels on both sides of the ' + kind.label,
  );
  const panel = panelIn(kind, out);
  assertOpenAndOnScreen(kind, panel, 'when a summary adopted its neighbours');
  assert.strictEqual(
    panel.previousElementSibling,
    summary,
    'the ' + kind.label + ' sits on the transcript right after the summary',
  );
  await closeQuietly(win);
  console.log(
    '  ok - the summary tool leaves the ' + kind.label + ' on the transcript',
  );
}

async function testReplayedSummaryLeavesPanelVisible(kind) {
  // The same on a finished replay, where the chevron pass also hides
  // every plain panel and re-collapses the summary.
  const {win} = makePage({beforeScripts: stubRaf});
  const events = finishedTaskEvents(kind, 44);
  events.splice(events.length - 1, 0, {
    type: 'tool_call',
    name: 'summary',
    description: 'so far',
  });
  send(win, {type: 'task_events', task: kind.task, task_id: 44, events});
  const out = win.document.getElementById('output');
  const summary = out.querySelector('.tc-summary');
  assert.ok(summary.classList.contains('collapsed'), 'the summary folds');
  // The run's first summary walks back over the panel and the prompt
  // echoes: every tool panel and every prompt panel are adopted, the
  // ask panel is not.
  assert.strictEqual(
    summary.querySelectorAll('.summary-sub > .tc').length,
    4,
    'the summary adopts the plain tool panels, never the ' + kind.label,
  );
  assert.strictEqual(
    summary.querySelectorAll('.summary-sub > .ev.prompt').length,
    kind.promptsUnderSummary,
    'the prompt echoes fold under the summary',
  );
  assertOpenAndOnScreen(kind, panelIn(kind, out), 'on a replay with a summary');
  await closeQuietly(win);
  console.log(
    '  ok - a replayed summary leaves the ' + kind.label + ' visible',
  );
}

async function testShareExportKeepsPanelOpen(kind) {
  const {win, posted} = makePage({beforeScripts: stubRaf});
  const tab = win._testApi.getActiveTabId();
  send(win, {type: 'clear', chat_id: 'chat-1', tabId: tab});
  send(win, {type: 'status', running: true, tabId: tab});
  send(win, {type: 'setTaskText', text: kind.task, tabId: tab});
  send(win, {
    type: 'tool_call',
    name: 'Bash',
    command: 'ls',
    tabId: tab,
    taskId: 'task-1',
  });
  send(win, {type: 'result', text: 'done', tabId: tab, taskId: 'task-1'});
  send(win, {type: 'status', running: false, tabId: tab});
  win.document
    .getElementById('share-btn')
    .dispatchEvent(
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
      {
        task: 'earlier task',
        task_id: 'task-0',
        events: finishedTaskEvents(kind, 'task-0'),
      },
      {task: kind.task, task_id: 'task-1', events: []},
    ],
    truncated: false,
  });
  const msg = posted.filter(m => m.type === 'shareChat').pop();
  assert.ok(msg, 'the share_tasks reply produces a shareChat command');
  const page = new JSDOM(
    '<!doctype html><html><body>' + msg.html + '</body></html>',
  );
  const panel = panelIn(kind, page.window.document.body);
  const out = panel.closest('[id="output"]');
  assert.ok(out, 'the exported ' + kind.label + ' sits in a transcript');
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
  assertOpenAndOnScreen(kind, panel, 'in the share export');
  await closeQuietly(win);
  console.log('  ok - the share export shows the ' + kind.label + ' open');
}

async function testUserStillFoldsByHand(kind) {
  const {win} = makePage({beforeScripts: stubRaf});
  const tab = win._testApi.getActiveTabId();
  kind.live(win, tab, undefined);
  const panel = panelIn(kind, win.document.getElementById('output'));
  panel.querySelector(kind.header).click();
  assert.ok(panel.classList.contains('collapsed'), 'a header click folds it');
  panel.querySelector(kind.header).click();
  assert.ok(!panel.classList.contains('collapsed'), 'and unfolds it again');
  await closeQuietly(win);
  console.log(
    '  ok - the user still folds the ' + kind.label + ' by its header',
  );
}

const tests = [
  testReplayOfFinishedTaskKeepsPanelOpen,
  testReplayOfRunningTaskKeepsPanelOpen,
  testBackgroundTabRestoreKeepsPanelOpen,
  testSummaryToolLeavesPanelOnTranscript,
  testReplayedSummaryLeavesPanelVisible,
  testShareExportKeepsPanelOpen,
  testUserStillFoldsByHand,
];

(async () => {
  let n = 0;
  for (const kind of [ANSWER_KIND, QUESTION_KIND]) {
    console.log('--- ' + kind.label + ' panel');
    for (const t of tests) {
      await t(kind);
      n++;
    }
  }
  console.log('\n' + n + ' passed, 0 failed');
})().catch(e => {
  console.error(e);
  process.exit(1);
});
