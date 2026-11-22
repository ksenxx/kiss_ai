// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// The chat page loads a transcript in pieces (`lazy: true` on its
// `ready`, `resumeSession` and `getAdjacentTask` commands, see
// server/lazy_replay.py).  A finished task arrives as its text, an
// empty "Trajectory N events" panel and its result; opening that panel
// asks the daemon for the task's outline (`getTaskEvents`), whose
// `summary` panels come empty too and are asked for when they are
// opened in turn.  A running task arrives as its outline at once and
// goes on streaming.  The events the daemon left out of a piece are
// stood in for by `lazy_gap` events, so the panels of a piece are laid
// out as the whole transcript would lay them out.
//
// These tests drive the real page end to end: the daemon's frames come
// in as window messages, the user's part is a click (a panel header,
// the share button, a history row) or a prompt typed into the
// composer, and what is checked is the page (its panels, its tabs,
// its composer) and the commands it posts.  The frames are the
// daemon's own pieces of the transcripts below, copied from
// server/lazy_replay.py's output; what the page builds from pieces
// the daemon makes on the spot is compared with a whole replay in
// tests/webapp/test_lazy_chat_loading_jsdom.py.

'use strict';

const assert = require('assert');
const {JSDOM} = require('jsdom');
const {
  makePage: makeChatPage,
  send,
  click,
  readyTabId,
  ofType,
  lastMsg,
  statusMetrics,
  typeInto,
  byId,
  wheel,
  fakeGeometry,
  getAdjacent,
  answering,
  plain,
  stubClipboard,
  recordTimers,
  fireTimers,
  PNG_B64,
} = require('./simplify2_harness');
const {loadHistory, historyRows, makeRow} = require('./history_helpers');
const {shareWithTasks} = require('./share_helpers');

const TASK = 'task-1';
const CHAT = 'chat-1';
// The daemon's clock when the task started (event stamps are ms).
const T0 = 1794000000000;
// The last stored event of a finished task when its first piece was
// made: the handles of its panels name it, so that every piece is
// read from that same transcript.  A running task's handles name none.
const UPTO = 33;

function makePage() {
  return makeChatPage({sourceUrl: 'lazypanel-main.js'});
}

// ---- The transcripts, as the daemon cuts them up ----------------------
//
// The task most scenarios show reads, whole:
//
//   prompt, thoughts, Bash ls (output), run_parallel [one, two],
//   thoughts, Read a.png (a picture), a typed message,
//   summary 0 "- listed the files"     (folds everything above but the
//                                       picture and the message)
//   thoughts, Bash false (fails),
//   summary 1 "- failed, noted"        (folds those two)
//   thoughts, finish, the result card.
//
// Its Trajectory panel holds six panels: the two summaries, the
// picture, the message, the last thoughts and finish.

function call(name, ts, fields) {
  return Object.assign({type: 'tool_call', name, ts}, fields || {});
}

function done(name, ts, content, fields) {
  return Object.assign(
    {type: 'tool_result', tool_name: name, content, is_error: false, ts},
    fields || {},
  );
}

function thoughts(text, ts) {
  return [
    {type: 'thinking_start', ts},
    {type: 'thinking_delta', text, ts: ts + 100},
    {type: 'thinking_end', ts: ts + 200},
  ];
}

/** The stand-in of events of type *of* that a piece leaves out. */
function gap(of, fields) {
  return Object.assign({type: 'lazy_gap', of}, fields || {});
}

const SETTINGS = {
  type: 'task_settings',
  settings: {model: 'm', work_dir: '/w'},
};
const PROMPT = {type: 'prompt', text: 'Do the thing.', ts: T0 + 100};
// A message typed into the task while it ran, recorded under the id
// its prompt was sent with.
const STEER = {
  type: 'prompt',
  steer: true,
  text: 'also check b.txt',
  promptIds: ['prompt-1'],
};
// The same message as a daemon without prompt ids recorded it.
const OLD_STEER = {type: 'prompt', steer: true, text: 'also check b.txt'};
const RESULT_CARD = {
  type: 'result',
  text: 'success: true',
  summary: '<p>All done.</p>',
  success: true,
  step_count: 4,
  total_tokens: 100,
  cost: '$0.05',
  ts: T0 + 12000,
};
const USAGE = {
  type: 'usage_info',
  text: 'Steps: 4/100, Context: 1/2 tokens',
  total_tokens: 100,
  cost: '$0.05',
  ts: T0 + 12100,
};

/**
 * The steps summary 0 folded: what `getTaskEvents {seg: 0}` answers.
 * *after* stands in for what follows them (the finished task's by
 * default: the Read call, then the result card).
 */
function segment0(after) {
  return [
    PROMPT,
    ...thoughts('Let me look.', T0 + 200),
    call('Bash', T0 + 1000, {command: 'ls', description: 'list the files'}),
    {type: 'system_output', text: 'a.txt\nb.txt\n', ts: T0 + 1500},
    done('Bash', T0 + 2000, ''),
    call('run_parallel', T0 + 3000, {
      description: 'fan out',
      extras: {tasks: '["one", "two"]'},
    }),
    done('run_parallel', T0 + 6000, 'both done'),
    ...thoughts('A picture next.', T0 + 6100),
    ...(after || [gap('tool_call', {ts: T0 + 6500}), gap('result')]),
  ];
}

/** The steps summary 1 folded: what `getTaskEvents {seg: 1}` answers. */
function segment1() {
  return [
    // Before them: tool calls, a fan-out among them, and the result of
    // summary 0, after which the agent thought again.
    gap('tool_call', {ts: T0 + 1000, fanouts: 1}),
    gap('tool_result', {ts: T0 + 7000}),
    ...thoughts('Next one.', T0 + 7500),
    call('Bash', T0 + 9000, {command: 'false', description: 'fail'}),
    done('Bash', T0 + 9500, 'exit 1', {is_error: true}),
    gap('tool_call', {ts: T0 + 10000}),
    gap('result'),
  ];
}

/**
 * How the outline of a task begins, finished or running: *handle*
 * names the transcript its summaries are read from, *steer* is the
 * message typed into the task.
 */
function outlineStart(handle, steer) {
  return [
    // The Bash call and the fan-out that summary 0 folded.
    gap('tool_call', {ts: T0 + 1000, fanouts: 1}),
    call('Read', T0 + 6500, {path: '/w/a.png'}),
    done('Read', T0 + 6600, 'a picture', {
      images: [
        {
          mime: 'image/png',
          path: 'a.png',
          lazy: {taskId: handle.taskId, index: 0},
        },
      ],
    }),
    Object.assign({ts: T0 + 6700}, steer || STEER),
    call('summary', T0 + 6800, {
      description: '- listed the files',
      lazy: Object.assign({}, handle, {seg: 0}),
    }),
    done('summary', T0 + 7000, 'ok'),
  ];
}

/** What `getTaskEvents` without `seg` answers: the Trajectory's inside. */
function outline(taskId, steer) {
  const handle = {taskId, upto: UPTO};
  return [
    ...outlineStart(handle, steer),
    gap('tool_call', {ts: T0 + 9000}),
    call('summary', T0 + 10000, {
      description: '- failed, noted',
      lazy: Object.assign({}, handle, {seg: 1}),
    }),
    done('summary', T0 + 10100, 'ok'),
    ...thoughts('Wrapping up.', T0 + 10200),
    call('finish', T0 + 11000, {extras: {success: 'True'}}),
    done('finish', T0 + 11100, 'finished'),
    gap('result'),
  ];
}

/**
 * What a lazy replay sends first of the finished task *taskId*; the
 * stub names the messages typed into the task (*steers*).
 */
function finished(taskId, steers) {
  return [
    SETTINGS,
    gap('tool_call', {ts: T0 + 1000, fanouts: 1}),
    {
      type: 'trajectory',
      count: 6,
      lazy: {taskId, upto: UPTO},
      steers: steers || [STEER],
    },
    RESULT_CARD,
    USAGE,
    {type: 'followup_suggestion', text: 'Do more.'},
    {type: 'lazy_steps', steps: 4},
  ];
}

/**
 * What a lazy replay sends of the task while it runs, stopped after
 * the thoughts that follow summary 0 by a second fan-out still out.
 * *steer* is the message typed into it.
 */
function running(taskId, steer) {
  return [
    SETTINGS,
    ...outlineStart({taskId}, steer),
    ...thoughts('Next one.', T0 + 7500),
    call('run_parallel', T0 + 9000, {
      description: 'fan out again',
      extras: {tasks: '["three"]'},
    }),
    {type: 'lazy_steps', steps: 3},
  ];
}

/** The whole transcript, as a page that does not load in pieces gets it. */
function whole() {
  const sent = ev => ev.type !== 'lazy_gap';
  return [
    SETTINGS,
    ...segment0().filter(sent),
    call('Read', T0 + 6500, {path: '/w/a.png'}),
    done('Read', T0 + 6600, 'a picture', {
      images: [{mime: 'image/png', path: 'a.png', b64: PNG_B64}],
    }),
    Object.assign({ts: T0 + 6700}, STEER),
    call('summary', T0 + 6800, {description: '- listed the files'}),
    done('summary', T0 + 7000, 'ok'),
    ...segment1().filter(sent),
    call('summary', T0 + 10000, {description: '- failed, noted'}),
    done('summary', T0 + 10100, 'ok'),
    ...thoughts('Wrapping up.', T0 + 10200),
    call('finish', T0 + 11000, {extras: {success: 'True'}}),
    done('finish', T0 + 11100, 'finished'),
    RESULT_CARD,
    USAGE,
    {type: 'task_done', success: true},
    {type: 'followup_suggestion', text: 'Do more.'},
  ];
}

// A task stopped while its first command ran: prompt, thoughts, the
// Bash call that never reported back.
function stoppedFinished(taskId) {
  return [
    SETTINGS,
    {type: 'usage_info', text: 'Steps: 1/100', total_tokens: 10, cost: '$0.01'},
    gap('tool_call', {ts: T0 + 1000}),
    {type: 'trajectory', count: 3, lazy: {taskId, upto: UPTO}},
    {type: 'lazy_steps', steps: 1},
  ];
}

function stoppedOutline() {
  return [
    PROMPT,
    ...thoughts('Let me look.', T0 + 200),
    call('Bash', T0 + 1000, {command: 'sleep 100', description: 'wait'}),
  ];
}

// A task of two sessions: prompt, thoughts, Bash ls, the first
// session's result card; thoughts, Bash pwd, a summary of those two,
// thoughts, finish, the second result card.
const FIRST_CARD = {
  type: 'result',
  text: 'to be continued',
  summary: '<p>Half done.</p>',
  success: false,
  is_continue: true,
  step_count: 1,
  total_tokens: 40,
  cost: '$0.02',
  ts: T0 + 3000,
};

function sessionsFinished(taskId) {
  return [
    SETTINGS,
    gap('tool_call', {ts: T0 + 1000}),
    {type: 'trajectory', count: 6, lazy: {taskId, upto: UPTO}},
    FIRST_CARD,
    gap('tool_call', {ts: T0 + 5000}),
    Object.assign({}, RESULT_CARD, {step_count: 3}),
    {type: 'lazy_steps', steps: 3},
  ];
}

function sessionsOutline(taskId) {
  return [
    PROMPT,
    ...thoughts('First half.', T0 + 200),
    call('Bash', T0 + 1000, {command: 'ls', description: 'list the files'}),
    done('Bash', T0 + 2000, 'a.txt'),
    gap('result'),
    gap('tool_call', {ts: T0 + 5000}),
    call('summary', T0 + 6000, {
      description: '- finished the second half',
      lazy: {taskId, upto: UPTO, seg: 0},
    }),
    done('summary', T0 + 6100, 'ok'),
    ...thoughts('Wrapping up.', T0 + 6200),
    call('finish', T0 + 7000, {extras: {success: 'True'}}),
    done('finish', T0 + 7100, 'finished'),
    gap('result'),
  ];
}

function sessionsSegment() {
  return [
    gap('tool_call', {ts: T0 + 1000}),
    gap('result'),
    ...thoughts('Second half.', T0 + 4000),
    call('Bash', T0 + 5000, {command: 'pwd', description: 'where'}),
    done('Bash', T0 + 5500, '/w'),
    gap('tool_call', {ts: T0 + 6000}),
    gap('result'),
  ];
}

// ---- Driving the page -------------------------------------------------

/** The daemon's replay of a task into *tabId* (a fresh copy of *events*). */
function replay(win, tabId, events, fields) {
  send(
    win,
    Object.assign(
      {
        type: 'task_events',
        tabId,
        chat_id: CHAT,
        task_id: TASK,
        task: 'Do the thing.',
        events: JSON.parse(JSON.stringify(events)),
      },
      fields || {},
    ),
  );
}

/**
 * A page on an announced connection, as the transport's always is:
 * its ready tab is *page.tabId*, its transcript *page.O*, its text box
 * *page.composer*.
 */
function openPage() {
  const page = makePage();
  send(page.win, {type: 'daemonStatus', connected: true});
  page.tabId = readyTabId(page.posted);
  page.O = page.win.document.getElementById('output');
  page.composer = byId(page.win, 'task-input');
  return page;
}

/** A page showing the lazily loaded finished task (*page.stub*). */
function finishedPage(steers) {
  const page = openPage();
  replay(page.win, page.tabId, finished(TASK, steers));
  page.stub = page.O.querySelector(':scope > .trajectory');
  assert.ok(page.stub, 'the finished task shows its Trajectory panel');
  return page;
}

/**
 * A page showing the lazily loaded task while it runs; *steer* is the
 * message typed into it.
 */
function runningPage(steer) {
  const page = openPage();
  send(page.win, {type: 'status', running: true, tabId: page.tabId});
  replay(page.win, page.tabId, running(TASK, steer));
  return page;
}

/** The user types *text* into the composer and sends it. */
function sendPrompt(page, text) {
  typeInto(page.win, page.composer, text);
  click(page.win, byId(page.win, 'send-btn'));
}

/** The connection drops and comes back, with nothing still on its way. */
function outage(win) {
  send(win, {type: 'daemonStatus', connected: false, reconnecting: true});
  send(win, {type: 'daemonStatus', connected: true, unconfirmed: []});
}

/** The daemon brings the tab up to date: its task, over by now. */
function resync(page, events) {
  const tabId = page.tabId;
  send(page.win, {
    type: 'tabs_state',
    tabs: [{tabId, title: 'Do the thing.', chatId: CHAT}],
  });
  send(page.win, {type: 'status', running: false, tabId});
  replay(page.win, tabId, events);
}

/**
 * Scroll past the top of the transcript: the task above the one on
 * screen arrives as *events*.  Returns its container.
 */
function loadTaskAbove(page, events) {
  const {win, posted, tabId, O} = page;
  fakeGeometry(O, {sh: 3000, ch: 500});
  O.scrollTop = 0;
  wheel(win, O, -50, 10);
  const asked = getAdjacent(posted).pop();
  assert.ok(asked, 'the overscroll asks for the task above');
  send(win, {
    type: 'adjacent_task_events',
    tabId,
    direction: 'prev',
    adjacent_to: asked.taskId,
    task: 'The task before.',
    task_id: 'task-0',
    events: JSON.parse(JSON.stringify(events)),
  });
  const above = O.querySelector('.adjacent-task');
  assert.ok(above, 'the task above is on the page');
  return above;
}

/** The task's live stream goes on with *events*. */
function stream(page, events) {
  for (const ev of events)
    send(page.win, Object.assign({tabId: page.tabId}, ev));
}

function header(panel) {
  return panel.querySelector(':scope > .collapse-header');
}

/** What a folded panel's header says after its title. */
function preview(panel) {
  return header(panel).querySelector('.collapse-preview').textContent;
}

/** The note a panel shows while its events are away, or null. */
function note(panel) {
  const el = panel.querySelector(':scope > .lazy-note');
  return el ? el.textContent : null;
}

/** The `getTaskEvents` commands the page posted so far, in order. */
function requests(posted) {
  return ofType(posted, 'getTaskEvents').map(plain);
}

/**
 * Click the header of *panel* and return the `getTaskEvents` commands
 * that click posted.
 */
function toggle(page, panel) {
  const from = requests(page.posted).length;
  click(page.win, header(panel));
  return requests(page.posted).slice(from);
}

/** Open the unloaded *panel*: the one request its click posts. */
function open(page, panel) {
  const asked = toggle(page, panel);
  assert.strictEqual(asked.length, 1, 'opening the panel asks once');
  return asked[0];
}

/** The daemon's answer to the request *req*: the piece's *events*. */
function answer(win, req, events) {
  send(win, {
    type: 'taskEventsPiece',
    taskId: req.taskId,
    seg: req.seg === undefined ? -1 : req.seg,
    token: req.token,
    events: JSON.parse(JSON.stringify(events)),
  });
}

/** Open *panel* and fill it with *events*; returns the request. */
function load(page, panel, events) {
  const req = open(page, panel);
  answer(page.win, req, events);
  assert.ok(!panel.classList.contains('lazy-unloaded'), 'the answer fills');
  return req;
}

function body(panel) {
  return panel.querySelector(':scope > .summary-sub, :scope > .trajectory-sub');
}

/** One word for what the element *el* of a transcript is. */
function kind(el) {
  const is = cls => el.classList.contains(cls);
  if (is('task-panel')) return 'task';
  if (is('trajectory')) return 'trajectory';
  if (is('rc')) return 'result';
  if (is('followup-bar')) return 'suggestion';
  if (is('tc-summary')) return 'summary';
  if (is('tc-run-parallel')) return 'fan-out';
  if (is('tc-bash')) return 'bash';
  if (is('llm-panel')) return 'thoughts';
  if (is('user-msg')) return 'message';
  if (is('ask-answer')) return 'answer';
  if (is('prompt')) return 'prompt';
  if (is('tc')) return el.querySelector('.tc-h-name').textContent.toLowerCase();
  return el.className;
}

/** What the panels in *panel*'s body are, in order. */
function kinds(panel) {
  return Array.from(body(panel).children).map(kind);
}

/** What the transcript *root* shows at its top level, in order. */
function shown(root) {
  return Array.from(root.children).map(kind);
}

function summaries(root) {
  return Array.from(root.querySelectorAll('.tc-summary'));
}

/** How long a panel's footer says it took. */
function elapsed(panel) {
  return panel.querySelector('.panel-elapsed').textContent;
}

// ---- Scenarios ----------------------------------------------------------

// Opening a chat: every way the page asks for a transcript says it
// takes it in pieces, and a finished task is its text, a folded empty
// Trajectory panel that says how many panels it holds, and its result.
function testFinishedTaskFirstLoad() {
  const {win, posted} = makePage();
  const O = win.document.getElementById('output');
  assert.strictEqual(
    posted.find(m => m.type === 'ready').lazy,
    true,
    'ready asks for lazily loaded transcripts',
  );
  // A click on the task's row in the Chats panel.
  loadHistory(win, [
    makeRow({id: CHAT, task_id: TASK, title: 'Do the thing.', has_events: true}),
  ]);
  click(win, historyRows(win)[0]);
  const resume = lastMsg(posted, 'resumeSession');
  assert.strictEqual(resume.lazy, true, 'resumeSession asks lazily');
  assert.strictEqual(resume.chatId, CHAT);
  assert.strictEqual(resume.taskId, TASK);
  fakeGeometry(O, {sh: 3000, ch: 500});
  replay(win, resume.tabId, finished(TASK));

  assert.deepStrictEqual(
    shown(O),
    ['task', 'trajectory', 'result', 'suggestion'],
    'the task text, the Trajectory panel, the result and the suggestion',
  );
  const stub = O.querySelector(':scope > .trajectory');
  assert.ok(stub.classList.contains('collapsed'), 'the Trajectory is folded');
  assert.ok(stub.classList.contains('lazy-unloaded'), 'and not loaded');
  assert.strictEqual(body(stub).children.length, 0, 'it holds no panel yet');
  assert.ok(header(stub).textContent.startsWith('Trajectory'));
  assert.strictEqual(preview(stub), '6 events', "the daemon's count");
  assert.strictEqual(note(stub), null, 'nothing is being loaded');
  assert.strictEqual(
    O.querySelector('.rc .rc-body').textContent.trim(),
    'All done.',
    'the result is on the page',
  );
  assert.deepStrictEqual(
    statusMetrics(win),
    {tokens: 'Tokens: 100', budget: 'Cost: $0.05', steps: 'Steps: 4'},
    'the status row of the whole task',
  );
  assert.strictEqual(requests(posted).length, 0, 'no piece asked for unasked');

  // Scrolling past the top asks for the neighbouring task, lazily too.
  O.scrollTop = 0;
  wheel(win, O, -50, 10);
  const adjacent = getAdjacent(posted);
  assert.strictEqual(adjacent.length, 1, 'the overscroll asks once');
  assert.strictEqual(adjacent[0].lazy, true, 'getAdjacentTask asks lazily');
  win.close();

  // A task stopped before its result: the stub closes the transcript,
  // and the steps are the daemon's count (no result card tells them).
  const stopped = openPage();
  replay(stopped.win, stopped.tabId, stoppedFinished(TASK));
  assert.deepStrictEqual(shown(stopped.O), ['task', 'trajectory']);
  assert.strictEqual(preview(stopped.O.lastElementChild), '3 events');
  assert.strictEqual(statusMetrics(stopped.win).steps, 'Steps: 1');
  stopped.win.close();
  console.log('  ok - a finished task loads as text, Trajectory stub, result');
}

// Opening the Trajectory panel asks for the task's outline once, shows
// that it is loading, and the answer fills it: the count on the header
// stands, and the summaries inside are folded and empty in turn.
function testTrajectoryClickLoadsTheOutline() {
  const page = finishedPage();
  const {win, posted, tabId, O, stub} = page;

  const req = open(page, stub);
  assert.deepStrictEqual(
    req,
    {type: 'getTaskEvents', taskId: TASK, tabId, token: req.token, upto: UPTO},
    "the outline is asked for by the stub's handle, for this tab, with no seg",
  );
  assert.ok(req.token, 'the request carries a token');
  assert.ok(!stub.classList.contains('collapsed'), 'the panel is open');
  assert.strictEqual(note(stub), 'Loading…');
  // Folding and opening it again while the answer is on its way.
  assert.deepStrictEqual(toggle(page, stub), [], 'folding asks nothing');
  assert.strictEqual(preview(stub), '6 events', 'folded, it still counts');
  assert.deepStrictEqual(toggle(page, stub), [], 'one request is enough');
  assert.strictEqual(note(stub), 'Loading…');

  answer(win, req, outline(TASK));
  assert.strictEqual(note(stub), null, 'the note goes with the answer');
  assert.ok(!stub.classList.contains('lazy-unloaded'), 'the panel is loaded');
  assert.deepStrictEqual(
    kinds(stub),
    ['summary', 'read', 'message', 'summary', 'thoughts', 'finish'],
    'the panels no summary folded, where a whole replay puts them',
  );
  const [first, second] = summaries(stub);
  for (const [panel, seg] of [
    [first, '0'],
    [second, '1'],
  ]) {
    assert.ok(panel.classList.contains('collapsed'), 'a summary is folded');
    assert.ok(panel.classList.contains('lazy-unloaded'), 'and not loaded');
    assert.strictEqual(panel.dataset.lazySeg, seg, 'it names its piece');
    assert.strictEqual(body(panel).children.length, 0, 'and holds nothing');
  }
  assert.strictEqual(
    first.querySelector('.tc-summary-desc').textContent,
    '- listed the files',
    'what the summary says is on the page',
  );
  // The picture of the outline is asked for as any lazily sent one.
  assert.deepStrictEqual(
    ofType(posted, 'getReplayImage').map(m => [m.taskId, m.index]),
    [[TASK, 0]],
  );
  // Folded again the header counts the same panels; opened again
  // nothing is asked.
  assert.deepStrictEqual(toggle(page, stub), []);
  assert.strictEqual(preview(stub), '6 events', 'the count is unchanged');
  assert.deepStrictEqual(toggle(page, stub), [], 'a loaded panel asks nothing');
  assert.strictEqual(requests(posted).length, 1);
  assert.deepStrictEqual(shown(O), ['task', 'trajectory', 'result', 'suggestion']);
  win.close();
  console.log('  ok - opening the Trajectory loads the outline once');
}

// Opening a summary asks for the steps it folded, by its place among
// the transcript's summaries.  The stand-ins of the events around them
// make the steps read as in a whole replay: a Thinking panel ends when
// the tool call left out began, and begins with the result left out.
function testSummaryClickLoadsItsSteps() {
  const page = finishedPage();
  const {win, posted, tabId, stub} = page;
  load(page, stub, outline(TASK));
  const [first, second] = summaries(stub);

  const req = open(page, first);
  assert.deepStrictEqual(
    req,
    {
      type: 'getTaskEvents',
      taskId: TASK,
      tabId,
      token: req.token,
      upto: UPTO,
      seg: 0,
    },
    'the summary names its piece of the same transcript',
  );
  assert.strictEqual(note(first), 'Loading…');
  assert.strictEqual(note(second), null, 'only the opened summary waits');
  answer(win, req, segment0());
  assert.strictEqual(note(first), null);
  assert.ok(!first.classList.contains('lazy-unloaded'));
  assert.strictEqual(first.dataset.lazySeg, undefined, 'no piece to ask for');
  assert.deepStrictEqual(
    kinds(first),
    ['prompt', 'thoughts', 'bash', 'fan-out', 'thoughts'],
    'the steps the summary folded',
  );
  assert.strictEqual(
    body(first).querySelector('.tc-bash .bash-panel-content').textContent,
    'a.txt\nb.txt\n',
    "the command's output is in its panel",
  );
  assert.strictEqual(
    elapsed(body(first).lastElementChild),
    '500ms',
    "the last thoughts ran from the fan-out's result to the Read call (6500 ms)",
  );
  assert.ok(second.classList.contains('lazy-unloaded'), 'the other waits');

  const req1 = open(page, second);
  assert.strictEqual(req1.seg, 1);
  assert.notStrictEqual(req1.token, req.token, 'a token per request');
  answer(win, req1, segment1());
  assert.deepStrictEqual(kinds(second), ['thoughts', 'bash']);
  const thinking = body(second).firstElementChild;
  assert.strictEqual(
    thinking.dataset.startTs,
    String(T0 + 7000),
    "the thoughts began with summary 0's result",
  );
  assert.strictEqual(elapsed(thinking), '2.0s', 'until the command (9000 ms)');
  assert.ok(
    body(second).querySelector('.tc-bash .tr.err'),
    'the failed command shows its failure',
  );
  // The header of the Trajectory counts its own panels only, and the
  // status row the steps of the task, not the panels of its pieces.
  toggle(page, stub);
  assert.strictEqual(preview(stub), '6 events');
  assert.strictEqual(statusMetrics(win).steps, 'Steps: 4');
  assert.strictEqual(requests(posted).length, 3, 'one request per panel');
  win.close();
  console.log('  ok - opening a summary loads the steps it folded');
}

// A result card left out of a piece still ends the reach of the
// summary after it, and the Thinking panel a tool result opened on
// spec is withdrawn where the card stood.
function testResultCardsLeftOut() {
  const page = openPage();
  const {win, tabId, O} = page;
  replay(win, tabId, sessionsFinished(TASK));
  assert.deepStrictEqual(
    shown(O),
    ['task', 'trajectory', 'result', 'result'],
    'both sessions show their result',
  );
  assert.strictEqual(statusMetrics(win).steps, 'Steps: 3');
  const stub = O.querySelector(':scope > .trajectory');
  load(page, stub, sessionsOutline(TASK));
  assert.deepStrictEqual(
    kinds(stub),
    ['prompt', 'thoughts', 'bash', 'summary', 'thoughts', 'finish'],
    "the first session's steps stay out of the second one's summary",
  );
  const summary = summaries(stub)[0];
  load(page, summary, sessionsSegment());
  assert.deepStrictEqual(kinds(summary), ['thoughts', 'bash']);
  assert.ok(!O.querySelector('.lazy-stop'), 'no stand-in is left on the page');
  toggle(page, stub);
  assert.strictEqual(preview(stub), '6 events');
  win.close();
  console.log('  ok - a result card left out still ends a summary');
}

// The daemon cannot read the events: the panel says why, stays
// unloaded, and opening it again asks again.  What it says is no part
// of the transcript: the panel's copy button leaves it out.
function testErrorReplyThenRetry() {
  const page = finishedPage();
  const {win, stub} = page;
  const copied = stubClipboard(win).writes;
  const req = open(page, stub);
  const reason = 'Failed to load the events: database is locked';
  send(win, {
    type: 'taskEventsPiece',
    taskId: TASK,
    seg: -1,
    token: req.token,
    error: reason,
  });
  assert.strictEqual(note(stub), reason, 'the reason is shown');
  assert.ok(stub.classList.contains('lazy-unloaded'), 'still not loaded');
  assert.strictEqual(body(stub).children.length, 0);
  click(win, stub.querySelector(':scope > .panel-copy-btn'));
  assert.deepStrictEqual(copied, ['Trajectory'], 'the reason is not copied');

  assert.deepStrictEqual(toggle(page, stub), [], 'folding asks nothing');
  const retry = open(page, stub);
  assert.notStrictEqual(retry.token, req.token, 'a new request');
  assert.strictEqual(note(stub), 'Loading…', 'the reason gives way');
  // An answer with neither events nor a reason.
  send(win, {type: 'taskEventsPiece', taskId: TASK, seg: -1, token: retry.token});
  assert.strictEqual(note(stub), 'Failed to load the events.');

  toggle(page, stub);
  load(page, stub, outline(TASK));
  assert.strictEqual(note(stub), null);
  assert.strictEqual(body(stub).children.length, 6, 'the third try loads');
  win.close();
  console.log('  ok - a failed load says why and is retried by re-opening');
}

// An answer nobody waits for changes nothing: one with a token the
// page never sent, a second answer to a request, and the answer for a
// panel of a transcript that a replay replaced meanwhile.
function testStaleAnswersAreDropped() {
  const page = finishedPage();
  const {win, posted, tabId, O, stub} = page;
  answer(win, {taskId: TASK, token: 'lazy-999'}, outline(TASK));
  assert.ok(stub.classList.contains('lazy-unloaded'), 'unknown token: dropped');
  assert.strictEqual(body(stub).children.length, 0);
  assert.strictEqual(note(stub), null);

  const req = open(page, stub);
  // The daemon replays the tab (another surface asked, a reconnect):
  // the transcript on screen is a new one.
  replay(win, tabId, finished(TASK));
  const fresh = O.querySelector(':scope > .trajectory');
  assert.notStrictEqual(fresh, stub, 'the replay rebuilt the transcript');
  assert.ok(!O.contains(stub), 'the old panel is off the page');
  answer(win, req, outline(TASK));
  assert.strictEqual(body(stub).children.length, 0, 'the old panel: dropped');
  assert.ok(fresh.classList.contains('lazy-unloaded'), 'the new one: untouched');
  assert.strictEqual(note(fresh), null);
  assert.strictEqual(preview(fresh), '6 events');

  // The new panel loads by its own request; a second answer to it is
  // dropped as well.
  const again = load(page, fresh, outline(TASK));
  answer(win, again, segment0());
  assert.strictEqual(body(fresh).children.length, 6, 'answered once');
  assert.strictEqual(requests(posted).length, 2);
  win.close();
  console.log('  ok - answers nobody waits for are dropped');
}

// A panel folded again while its piece is on the way is filled all the
// same, and its header then counts the panels it holds: the daemon
// leaves out of a piece the events deleted since it counted them.
function testPieceForAPanelFoldedAgain() {
  const page = finishedPage();
  const {win, stub} = page;
  const req = open(page, stub);
  assert.deepStrictEqual(toggle(page, stub), [], 'folding asks nothing');
  assert.strictEqual(preview(stub), '6 events', 'as the daemon counted');

  answer(win, req, outline(TASK).slice(0, -3));
  assert.ok(stub.classList.contains('collapsed'), 'it stays folded');
  assert.ok(!stub.classList.contains('lazy-unloaded'), 'and is loaded');
  assert.deepStrictEqual(
    kinds(stub),
    ['summary', 'read', 'message', 'summary', 'thoughts'],
    'the task lost its last call since',
  );
  assert.strictEqual(preview(stub), '5 events', 'what the panel holds');
  assert.deepStrictEqual(toggle(page, stub), [], 'opening it asks nothing');
  win.close();
  console.log('  ok - a panel folded while waiting counts what it was sent');
}

// The connection drops while a panel waits: its answer never comes,
// so it asks again (the request goes out once the daemon is back).
// The answer to the lost request is no answer any more, and the panel
// of a transcript thrown away asks nothing.
function testDisconnectAsksAgain() {
  const page = finishedPage();
  const {win, posted, tabId, O, stub} = page;
  const req = open(page, stub);

  send(win, {type: 'daemonStatus', connected: false, reconnecting: true});
  const asked = requests(posted);
  assert.strictEqual(asked.length, 2, 'the waiting panel asks again');
  assert.deepStrictEqual(
    asked[1],
    Object.assign({}, req, {token: asked[1].token}),
    'the same piece',
  );
  assert.notStrictEqual(asked[1].token, req.token, 'under a new token');
  assert.strictEqual(note(stub), 'Loading…');

  answer(win, req, outline(TASK));
  assert.ok(stub.classList.contains('lazy-unloaded'), 'the lost one: ignored');
  assert.strictEqual(note(stub), 'Loading…');
  answer(win, asked[1], outline(TASK));
  assert.strictEqual(body(stub).children.length, 6, 'the new one fills');
  assert.strictEqual(note(stub), null);

  // A summary waits when the daemon replays the tab: its transcript is
  // thrown away, and the next outage asks nothing for it.
  const summary = summaries(stub)[0];
  const lost = open(page, summary);
  replay(win, tabId, finished(TASK));
  assert.ok(!O.contains(summary), 'the replay replaced the transcript');
  send(win, {type: 'daemonStatus', connected: false, reconnecting: true});
  assert.strictEqual(requests(posted).length, 3, 'nothing is asked again');
  answer(win, lost, segment0());
  assert.strictEqual(body(summary).children.length, 0, 'nor is it filled');
  // Nor does a later outage ask for it once more.
  send(win, {type: 'daemonStatus', connected: false, reconnecting: true});
  assert.strictEqual(requests(posted).length, 3);
  win.close();
  console.log('  ok - a waiting panel asks again after a disconnect');
}

// A running task arrives as its outline: no Trajectory panel, its
// summary folded and empty, the step count the daemon's.  The live
// stream goes on from there; the task's end folds everything into a
// Trajectory panel, and the summary inside still loads.
function testRunningTaskOutlineThenLiveStream() {
  const page = runningPage();
  const {win, posted, O} = page;
  assert.deepStrictEqual(
    shown(O),
    ['task', 'summary', 'read', 'message', 'thoughts', 'fan-out'],
    'the outline is on the page as it is, in no Trajectory panel',
  );
  const lazy = summaries(O)[0];
  assert.ok(lazy.classList.contains('collapsed'), 'the summary is folded');
  assert.ok(lazy.classList.contains('lazy-unloaded'), 'and not loaded');
  assert.strictEqual(lazy.dataset.lazySeg, '0');
  assert.strictEqual(
    statusMetrics(win).steps,
    'Steps: 3',
    "the daemon's count, not the one step the outline shows",
  );
  assert.strictEqual(requests(posted).length, 0);

  // The fan-out returns, the agent thinks, a command fails, and a
  // summary folds the steps since the last one: all of them here.
  stream(page, [
    done('run_parallel', T0 + 12000, 'done'),
    ...thoughts('Then a command.', T0 + 12100),
  ]);
  assert.strictEqual(statusMetrics(win).steps, 'Steps: 4', 'the count goes on');
  stream(page, [
    call('Bash', T0 + 13000, {command: 'false', description: 'fail'}),
    done('Bash', T0 + 13500, 'exit 1', {is_error: true}),
    call('summary', T0 + 14000, {description: '- failed, noted'}),
    done('summary', T0 + 14100, 'ok'),
  ]);
  const live = summaries(O)[1];
  assert.deepStrictEqual(
    kinds(live),
    ['thoughts', 'fan-out', 'thoughts', 'bash'],
    'a summary of steps on the page folds them',
  );
  assert.ok(!live.classList.contains('lazy-unloaded'));
  assert.deepStrictEqual(toggle(page, live), [], 'and asks for nothing');
  assert.ok(lazy.classList.contains('lazy-unloaded'), 'the first still waits');

  stream(page, [
    Object.assign({}, RESULT_CARD, {ts: T0 + 15000}),
    {type: 'task_done', success: true},
  ]);
  send(win, {type: 'status', running: false, tabId: page.tabId});
  assert.deepStrictEqual(shown(O), ['task', 'trajectory', 'result']);
  const traj = O.querySelector(':scope > .trajectory');
  assert.ok(traj.classList.contains('collapsed'), 'the end folds the task');
  assert.ok(!traj.classList.contains('lazy-unloaded'), 'its panels are here');
  assert.deepStrictEqual(kinds(traj), ['summary', 'read', 'message', 'summary']);
  assert.strictEqual(preview(traj), '4 events');
  assert.deepStrictEqual(toggle(page, traj), [], 'opening it asks nothing');
  const req = open(page, lazy);
  assert.deepStrictEqual(
    req,
    {
      type: 'getTaskEvents',
      taskId: TASK,
      tabId: page.tabId,
      token: req.token,
      seg: 0,
    },
    'the summary inside is still asked for, from the transcript as it is',
  );
  answer(
    win,
    req,
    segment0([gap('tool_call', {ts: T0 + 6500, fanouts: 1}), gap('result')]),
  );
  assert.deepStrictEqual(
    kinds(lazy),
    ['prompt', 'thoughts', 'bash', 'fan-out', 'thoughts'],
  );
  win.close();
  console.log('  ok - a running task is its outline, then its live stream');
}

// The task above the one on screen, loaded by scrolling past the top,
// comes in pieces as well: its Trajectory panel, in the neighbour's
// own container, loads on a click by the neighbour's handle, and the
// task on screen is left as it was.
function testNeighbourTaskLoadsItsOwnPieces() {
  const page = finishedPage();
  const {win, tabId, stub} = page;
  const above = loadTaskAbove(page, sessionsFinished('task-0'));
  assert.deepStrictEqual(
    shown(above),
    ['task', 'trajectory', 'result', 'result'],
    'the neighbour is its text, its Trajectory panel and its results',
  );
  const theirs = above.querySelector(':scope > .trajectory');
  assert.ok(theirs.classList.contains('lazy-unloaded'));
  assert.strictEqual(preview(theirs), '6 events');
  // The status row follows the task in view, here the neighbour.
  const row = statusMetrics(win);
  assert.strictEqual(row.steps, 'Steps: 3', "the daemon's count of its steps");

  const req = open(page, theirs);
  assert.deepStrictEqual(
    req,
    {
      type: 'getTaskEvents',
      taskId: 'task-0',
      tabId,
      token: req.token,
      upto: UPTO,
    },
    "the neighbour's task, for the tab that shows it",
  );
  answer(win, req, sessionsOutline('task-0'));
  assert.deepStrictEqual(
    kinds(theirs),
    ['prompt', 'thoughts', 'bash', 'summary', 'thoughts', 'finish'],
  );
  const summary = summaries(theirs)[0];
  const inner = open(page, summary);
  assert.strictEqual(inner.taskId, 'task-0');
  assert.strictEqual(inner.seg, 0);
  answer(win, inner, sessionsSegment());
  assert.deepStrictEqual(kinds(summary), ['thoughts', 'bash']);

  assert.deepStrictEqual(statusMetrics(win), row, 'a piece counts no step');
  assert.ok(stub.classList.contains('lazy-unloaded'), 'the task below waits');
  assert.strictEqual(body(stub).children.length, 0);
  assert.strictEqual(preview(stub), '6 events');
  win.close();

  // The task above a running one is over all the same: the command it
  // was stopped in is not at work, as the fan-out of the task below is.
  const busy = runningPage();
  const stopped = loadTaskAbove(busy, stoppedFinished('task-0'));
  const stoppedStub = stopped.querySelector(':scope > .trajectory');
  load(busy, stoppedStub, stoppedOutline());
  const command = body(stoppedStub).querySelector('.tc-bash');
  assert.ok(!command.classList.contains('panel-active'), 'no pulsing header');
  assert.ok(!command.querySelector('.panel-elapsed'), 'and no running time');
  const atWork = busy.O.querySelector(':scope > .tc-run-parallel');
  assert.ok(atWork.classList.contains('panel-active'), 'the live call runs');
  busy.win.close();
  console.log('  ok - a neighbouring task loads its own pieces');
}

// A panel belongs to the tab whose transcript holds it, on screen or
// not: its piece is asked for that tab, from the machine the tab's
// task ran on, and the answer fills the transcript where it is parked.
function testPanelOfABackgroundTab() {
  const page = runningPage();
  const {win, posted, tabId, O} = page;
  send(win, {type: 'tabMachine', tabId, machine: 'gpu-box'});
  const lazy = summaries(O)[0];
  const req = open(page, lazy);
  assert.deepStrictEqual(
    req,
    {
      type: 'getTaskEvents',
      taskId: TASK,
      tabId,
      token: req.token,
      seg: 0,
      machine: 'gpu-box',
    },
    'asked of the machine the tab is bound to',
  );
  // The user starts a new chat: the running task keeps its tab, which
  // is no longer the one on screen.
  click(win, byId(win, 'new-chat-btn'));
  assert.notStrictEqual(win._testApi.getActiveTabId(), tabId);
  assert.ok(!O.contains(lazy), "the task's transcript is parked");

  // The connection drops: the parked panel asks again, for its own
  // tab and machine (the chat on screen is bound to none).
  send(win, {type: 'daemonStatus', connected: false, reconnecting: true});
  const again = requests(posted)[1];
  assert.deepStrictEqual(again, Object.assign({}, req, {token: again.token}));
  assert.notStrictEqual(again.token, req.token);
  answer(
    win,
    again,
    segment0([gap('tool_call', {ts: T0 + 6500, fanouts: 1})]),
  );

  win._testApi.switchToTab(tabId);
  assert.ok(O.contains(lazy), 'back on screen');
  assert.ok(!lazy.classList.contains('lazy-unloaded'), 'the panel is loaded');
  assert.strictEqual(note(lazy), null);
  assert.deepStrictEqual(
    kinds(lazy),
    ['prompt', 'thoughts', 'bash', 'fan-out', 'thoughts'],
    'filled while its tab was in the background',
  );
  win.close();
  console.log('  ok - a panel of a background tab is asked for and filled there');
}

/**
 * The Trajectory panel of the task on screen in the share export
 * *cmd* of *page*, which the daemon then wrote.
 */
function exportedTrajectory(page, cmd) {
  assert.ok(cmd, 'the export is sent');
  // The daemon wrote the page: the share button is free again.
  send(page.win, {
    type: 'share_done',
    ok: true,
    path: '/w/reports/chat.html',
    tabId: page.tabId,
  });
  const doc = new JSDOM('<!DOCTYPE html><body>' + cmd.html + '</body>').window
    .document;
  assert.strictEqual(
    doc.querySelectorAll('.lazy-unloaded, .lazy-note').length,
    0,
    'nothing in the export waits for the daemon',
  );
  return doc.querySelector('.share-task:last-child > .trajectory');
}

/**
 * The Trajectory panel of the task in the share export of *page*, the
 * daemon having answered the export's request with *tasks*.
 */
function sharedTrajectory(page, tasks) {
  return exportedTrajectory(page, shareWithTasks(page, tasks));
}

/** Fail unless *traj*, an exported Trajectory panel, holds every panel of the task. */
function assertEveryPanel(traj, what) {
  assert.deepStrictEqual(
    kinds(traj),
    ['summary', 'read', 'message', 'summary', 'thoughts', 'finish'],
    what,
  );
  const [first, second] = summaries(traj);
  assert.deepStrictEqual(
    kinds(first),
    ['prompt', 'thoughts', 'bash', 'fan-out', 'thoughts'],
    what,
  );
  assert.deepStrictEqual(kinds(second), ['thoughts', 'bash'], what);
}

/** ... and the picture, taken from the daemon's transcript of the task. */
function assertWhole(traj, what) {
  assertEveryPanel(traj, what);
  assert.ok(
    traj.querySelector('img').src.startsWith('data:image/png;base64,'),
    'the picture is in the export',
  );
}

// The share export copies the transcript on screen.  What its panels
// have not loaded comes from the daemon's whole transcript of the
// task, which the export is sent in any case: the copy of each such
// panel takes the inside of its twin there.  The page loads nothing.
function testShareExportFillsUnloadedPanels() {
  const tasks = [{task_id: TASK, task: 'Do the thing.', events: whole()}];
  // Nothing is loaded.
  const page = finishedPage();
  const {win, posted, stub} = page;
  let traj = sharedTrajectory(page, tasks);
  assertWhole(traj, 'an unloaded Trajectory');
  assert.ok(traj.classList.contains('collapsed'), 'folded as on the page');
  assert.strictEqual(preview(traj), '6 events', 'and counting what it holds');
  assert.ok(stub.classList.contains('lazy-unloaded'), 'the page is as it was');
  assert.strictEqual(requests(posted).length, 0, 'and asked for nothing');

  // The Trajectory panel is open and waits for its outline.
  const req = open(page, stub);
  traj = sharedTrajectory(page, tasks);
  assertWhole(traj, 'a Trajectory that waits');
  assert.ok(!traj.classList.contains('collapsed'), 'open as on the page');
  assert.strictEqual(note(stub), 'Loading…', 'the page still waits');

  // The outline is here, its summaries are not.
  answer(win, req, outline(TASK));
  assertWhole(sharedTrajectory(page, tasks), 'unloaded summaries');
  assert.ok(
    summaries(stub).every(p => p.classList.contains('lazy-unloaded')),
    'the page loaded nothing for the export',
  );
  assert.strictEqual(requests(posted).length, 1);

  // Everything is loaded: the export is the transcript on screen.
  const [first, second] = summaries(stub);
  load(page, first, segment0());
  load(page, second, segment1());
  assertWhole(sharedTrajectory(page, tasks), 'a loaded transcript');
  win.close();

  // The daemon lists the task without its events (its database lost
  // them): the copy of the panel has no twin to take from.  It waits
  // for nothing all the same, and its header counts what it holds.
  const bare = finishedPage();
  const empty = sharedTrajectory(bare, [
    {task_id: TASK, task: 'Do the thing.', events: []},
  ]);
  assert.strictEqual(body(empty).children.length, 0);
  assert.strictEqual(preview(empty), '0 events');
  bare.win.close();
  console.log('  ok - the share export fills what the page did not load');
}

// The daemon's answer to an export does not always hold the task on
// screen: a task too large for one answer is left out of it.  The
// export then copies the screen, so the page first asks for what the
// screen's panels left with the daemon and exports with the last
// answer.
function testShareExportLoadsTheScreenFirst() {
  const others = [{task_id: 'task-9', task: 'Another.', events: [PROMPT]}];
  const page = finishedPage();
  const {win, posted, tabId, stub} = page;
  // The task above is on screen too, its panel open and waiting.  The
  // export copies the tab's own task only.
  const above = loadTaskAbove(page, sessionsFinished('task-0'));
  const theirs = open(page, above.querySelector(':scope > .trajectory'));

  assert.strictEqual(shareWithTasks(page, others), undefined, 'the export waits');
  const first = requests(posted).pop();
  assert.deepStrictEqual(
    first,
    {type: 'getTaskEvents', taskId: TASK, tabId, token: first.token, upto: UPTO},
    "the screen's own Trajectory is asked for",
  );
  assert.ok(stub.classList.contains('collapsed'), 'and stays folded');
  // The neighbour's panel cannot be read: nothing the export waits for.
  send(win, {
    type: 'taskEventsPiece',
    taskId: 'task-0',
    seg: -1,
    token: theirs.token,
    error: 'Failed to load the events: database is locked',
  });
  // The outline holds two summaries that fold more: both are asked for.
  answer(win, first, outline(TASK));
  const inner = requests(posted).slice(-2);
  assert.deepStrictEqual(
    inner.map(req => [req.taskId, req.seg]),
    [
      [TASK, 0],
      [TASK, 1],
    ],
  );
  answer(win, inner[0], segment0());
  assert.ok(!lastMsg(posted, 'shareChat'), 'one panel is still away');
  answer(win, inner[1], segment1());
  const traj = exportedTrajectory(page, lastMsg(posted, 'shareChat'));
  assertEveryPanel(traj, 'the export holds what the page loaded for it');
  assert.ok(traj.classList.contains('collapsed'), 'folded as on the page');
  assert.strictEqual(ofType(posted, 'shareChat').length, 1, 'sent once');
  assert.strictEqual(requests(posted).length, 4, 'each panel asked for once');
  win.close();

  // A panel of the screen that cannot be read fails the export: the
  // saved page would lack it.  The button works again.
  const failing = finishedPage();
  assert.strictEqual(shareWithTasks(failing, [null]), undefined);
  const lost = requests(failing.posted).pop();
  send(failing.win, {
    type: 'taskEventsPiece',
    taskId: TASK,
    seg: -1,
    token: lost.token,
    error: 'Failed to load the events: database is locked',
  });
  assert.ok(!lastMsg(failing.posted, 'shareChat'), 'nothing is exported');
  assert.ok(byId(failing.win, 'share-btn').classList.contains('share-err'));
  assert.ok(
    failing.O.textContent.includes(
      'Share failed: the transcript on screen could not be loaded',
    ),
    'and the page says why',
  );
  assert.strictEqual(shareWithTasks(failing, []), undefined, 'a new export');
  assert.strictEqual(requests(failing.posted).length, 2, 'asks again');
  failing.win.close();

  // The export is of the screen it was asked on.  By the time a panel
  // is answered the tab shows another chat: dropped, the button works.
  const moved = finishedPage();
  assert.strictEqual(shareWithTasks(moved, []), undefined);
  replay(moved.win, moved.tabId, finished('task-7'), {
    chat_id: 'chat-2',
    task_id: 'task-7',
  });
  load(moved, moved.O.querySelector(':scope > .trajectory'), outline('task-7'));
  assert.ok(!lastMsg(moved.posted, 'shareChat'), 'the old export is dropped');
  assert.strictEqual(shareWithTasks(moved, []), undefined, 'a new export');
  assert.strictEqual(ofType(moved.posted, 'shareChatTasks').length, 2);
  moved.win.close();

  // ... or another tab is on screen (a running task keeps its own):
  // the answer fills the parked panel and exports nothing.
  const away = runningPage();
  assert.strictEqual(shareWithTasks(away, []), undefined);
  const parked = requests(away.posted).pop();
  const lazy = summaries(away.O)[0];
  click(away.win, byId(away.win, 'new-chat-btn'));
  answer(
    away.win,
    parked,
    segment0([gap('tool_call', {ts: T0 + 6500, fanouts: 1})]),
  );
  assert.ok(!lazy.classList.contains('lazy-unloaded'), 'filled all the same');
  assert.ok(!lastMsg(away.posted, 'shareChat'), 'the export is dropped');
  shareWithTasks(away, []);
  assert.strictEqual(ofType(away.posted, 'shareChatTasks').length, 2);
  away.win.close();

  // The chat is shown again with a panel's request still out: another
  // screen.  The answer finds no panel to fill, and the export held
  // for that panel is dropped with it.
  const again = finishedPage();
  assert.strictEqual(shareWithTasks(again, []), undefined);
  const stale = requests(again.posted).pop();
  replay(again.win, again.tabId, finished(TASK));
  answer(again.win, stale, outline(TASK));
  assert.strictEqual(shareWithTasks(again, []), undefined, 'a new export');
  assert.strictEqual(ofType(again.posted, 'shareChatTasks').length, 2);
  again.win.close();

  // ... and so does an answer for the screen that took its place: it
  // does not bring the old export back.
  const anew = finishedPage();
  assert.strictEqual(shareWithTasks(anew, []), undefined);
  replay(anew.win, anew.tabId, stoppedFinished(TASK));
  load(anew, anew.O.querySelector(':scope > .trajectory'), stoppedOutline());
  assert.ok(!lastMsg(anew.posted, 'shareChat'), 'the old export is dropped');
  assert.ok(shareWithTasks(anew, []), 'the next click exports');
  anew.win.close();

  // No answer within the failsafe: the button works again and the
  // export is over.  The answer that comes late fills its panel and
  // sends nothing; the next click exports, once.
  const slow = openPage();
  replay(slow.win, slow.tabId, stoppedFinished(TASK));
  const failsafe = recordTimers(slow.win, 120000);
  assert.strictEqual(shareWithTasks(slow, []), undefined);
  fireTimers(failsafe);
  answer(slow.win, requests(slow.posted).pop(), stoppedOutline());
  assert.ok(!lastMsg(slow.posted, 'shareChat'), 'the export is over');
  assert.ok(shareWithTasks(slow, []), 'the next click exports');
  assert.strictEqual(ofType(slow.posted, 'shareChat').length, 1, 'once');
  slow.win.close();

  // An export given up on can still be answered: the history of a
  // chat on a registered machine is forwarded in the background, and
  // may come after the failsafe.  By then the tab shows a chat of this
  // daemon whose own export waits for its panel.  The late answer is
  // not that export's, which is sent, once, when its panel is loaded.
  // (Audit 2026-11-21, W11 F1.)
  const late = openPage();
  send(late.win, {type: 'tabMachine', tabId: late.tabId, machine: 'box'});
  replay(late.win, late.tabId, [SETTINGS, PROMPT], {
    chat_id: 'chat-on-box',
    task_id: 'task-on-box',
  });
  const givenUp = recordTimers(late.win, 120000);
  click(late.win, byId(late.win, 'share-btn'));
  const old = lastMsg(late.posted, 'shareChatTasks');
  assert.strictEqual(old.machine, 'box', "the machine's history is asked for");
  fireTimers(givenUp);
  send(late.win, {type: 'tabMachine', tabId: late.tabId, machine: ''});
  replay(late.win, late.tabId, stoppedFinished(TASK));
  assert.strictEqual(shareWithTasks(late, []), undefined, 'the export waits');
  send(late.win, {
    type: 'share_tasks',
    tabId: old.tabId,
    chatId: old.chatId,
    machine: 'box',
    tasks: [],
    truncated: false,
  });
  answer(late.win, requests(late.posted).pop(), stoppedOutline());
  const exports = ofType(late.posted, 'shareChat');
  assert.strictEqual(
    exports.length,
    1,
    'BUG - the late answer to an export given up on ended the one held',
  );
  assert.strictEqual(exports[0].chatId, CHAT, 'the chat on screen');
  assert.ok(exports[0].html.includes('Let me look.'), 'with the loaded panel');
  late.win.close();
  console.log('  ok - the share export loads the screen the daemon left out');
}

// What a finished task is sent after its first piece was made (an
// answer that came late) is shown as it arrives.  The next fold puts
// it into the Trajectory panel, loaded or not: the header counts it
// with the panels the daemon holds, and loading those keeps it last.
function testLatePanelJoinsTheUnloadedTrajectory() {
  const page = finishedPage();
  const {win, tabId, O, stub} = page;
  send(win, {
    type: 'ask_answer',
    tabId,
    question: 'Why two files?',
    text: '<p>One per run.</p>',
    success: true,
  });
  assert.deepStrictEqual(
    shown(O),
    ['task', 'trajectory', 'result', 'suggestion', 'answer'],
    'the late answer shows where it arrives',
  );
  assert.strictEqual(preview(stub), '6 events');
  send(win, {type: 'status', running: false, tabId});
  assert.deepStrictEqual(shown(O), ['task', 'trajectory', 'result', 'suggestion']);
  assert.ok(stub.classList.contains('lazy-unloaded'), 'still not loaded');
  assert.strictEqual(
    preview(stub),
    '7 events',
    "the daemon's six and the one on the page",
  );

  load(page, stub, outline(TASK));
  assert.deepStrictEqual(
    kinds(stub),
    ['summary', 'read', 'message', 'summary', 'thoughts', 'finish', 'answer'],
    'the late panel stays after the ones loaded',
  );
  toggle(page, stub);
  assert.strictEqual(preview(stub), '7 events', 'the count is unchanged');
  win.close();
  console.log('  ok - a late panel joins the unloaded Trajectory');
}

// A message typed into the running task is lost in an outage, and the
// task is over when the daemon is back.  Whether the message ran is
// told by the messages the Trajectory panel holds back (the stub's
// `steers`): one recorded under the prompt's id or, from a daemon that
// records no ids, one more reading the same than the page showed.
function testLostMessageIsSettledByTheStub() {
  for (const [ran, back] of [
    [true, ''],
    [false, 'one more thing'],
  ]) {
    const page = runningPage();
    sendPrompt(page, 'one more thing');
    const sent = lastMsg(page.posted, 'appendUserMessage');
    assert.strictEqual(sent.prompt, 'one more thing', 'typed into the task');
    outage(page.win);
    const mine = {
      type: 'prompt',
      steer: true,
      text: 'one more thing',
      promptIds: [sent.promptId],
    };
    resync(page, finished(TASK, ran ? [STEER, mine] : [STEER]));
    assert.strictEqual(
      page.composer.value,
      back,
      ran ? 'recorded under its id: it ran' : 'not recorded: offered again',
    );
    page.win.close();
  }
  for (const [held, back] of [
    [[OLD_STEER, OLD_STEER], ''],
    [[OLD_STEER], OLD_STEER.text],
  ]) {
    // The page shows one message reading like the one about to be sent.
    const page = runningPage(OLD_STEER);
    sendPrompt(page, OLD_STEER.text);
    assert.strictEqual(ofType(page.posted, 'appendUserMessage').length, 1);
    outage(page.win);
    resync(page, finished(TASK, held));
    assert.strictEqual(
      page.composer.value,
      back,
      held.length + ' such message(s) in the finished task',
    );
    page.win.close();
  }
  console.log('  ok - a lost message is settled by the messages the stub holds');
}

// A prompt sent from a finished task's page is lost in an outage.  A
// message of that task reading the same is nothing new in the replay
// that follows, whether the page showed it or its unloaded Trajectory
// panel held it back: the prompt comes back into the composer.
function testLostPromptReadingLikeAHeldMessage() {
  for (const loaded of [false, true]) {
    const page = finishedPage([OLD_STEER]);
    // The task above holds back no such message, and none of this task.
    loadTaskAbove(page, finished('task-0', []));
    if (loaded) load(page, page.stub, outline(TASK, OLD_STEER));
    sendPrompt(page, OLD_STEER.text);
    assert.strictEqual(
      lastMsg(page.posted, 'submit').prompt,
      OLD_STEER.text,
      'a new task is asked for',
    );
    outage(page.win);
    resync(page, finished(TASK, [OLD_STEER]));
    assert.strictEqual(
      page.composer.value,
      OLD_STEER.text,
      'offered again, the Trajectory ' + (loaded ? 'loaded' : 'not loaded'),
    );
    page.win.close();
  }
  console.log('  ok - a lost prompt reading like a held message comes back');
}

// The task asked a question and ended while the page was away.  The
// replay of the finished task keeps its questions in the Trajectory
// panel it does not send: none is open, and the composer stops being
// the answer box.
function testQuestionOfAFinishedTaskIsRetired() {
  const page = runningPage();
  const {win, posted, tabId} = page;
  send(win, {type: 'askUser', tabId, question: 'Which one?', questionId: 'q-1'});
  assert.ok(answering(win), 'the composer answers the question');
  replay(win, tabId, running(TASK));
  assert.ok(answering(win), 'a running task replayed keeps its question');

  send(win, {type: 'status', running: false, tabId});
  replay(win, tabId, finished(TASK));
  assert.ok(!answering(win), 'the finished task asks nothing');
  sendPrompt(page, 'Next task.');
  assert.strictEqual(ofType(posted, 'userAnswer').length, 0);
  assert.strictEqual(
    lastMsg(posted, 'submit').prompt,
    'Next task.',
    'what is typed is a prompt again',
  );
  win.close();
  console.log('  ok - a question of a finished task is retired');
}

testFinishedTaskFirstLoad();
testTrajectoryClickLoadsTheOutline();
testSummaryClickLoadsItsSteps();
testResultCardsLeftOut();
testErrorReplyThenRetry();
testStaleAnswersAreDropped();
testPieceForAPanelFoldedAgain();
testDisconnectAsksAgain();
testRunningTaskOutlineThenLiveStream();
testNeighbourTaskLoadsItsOwnPieces();
testPanelOfABackgroundTab();
testShareExportFillsUnloadedPanels();
testShareExportLoadsTheScreenFirst();
testLatePanelJoinsTheUnloadedTrajectory();
testLostMessageIsSettledByTheStub();
testLostPromptReadingLikeAHeldMessage();
testQuestionOfAFinishedTaskIsRetired();
console.log('lazyChatLoading: all tests passed');
