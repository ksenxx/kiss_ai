// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) tests for the time a task that is over leaves in
// its tab's status label (#status-text) and so in the Task Info
// panel's Time row (#meta-time): "2m 5s", whichever way the task
// ended and whether the page saw it end or replays its history row.
//
// The daemon stamps `startTs` / `endTs` on the terminal event and the
// time is computed from that span.  When the span is missing the page
// falls back to the start time it recorded itself -- and that fallback
// used the module-level `t0`, i.e. the VISIBLE tab's clock, even when
// the event named a background tab, so a task that ran for a minute in
// a hidden tab could report the few seconds the tab on screen had been
// running.  The label is also built by the same helper the timer uses
// (doneLabelFor) instead of a second copy of the arithmetic.
//
// A replayed history row used to leave the label alone: a finished
// task opened from the history list, or repainted by a reload, read
// "Time: Ready".

'use strict';

const assert = require('assert');
const harness = require('./simplify2_harness');

const {send, flushMicrotasks} = harness;

// The page's clock stands still, so every label is exact: at NOW
// unless a test moves `clock.now`.
const NOW = 1_700_000_000_000;
const clock = {now: NOW};
const DASH = '\u2014';

// The sourceURL is what audit0902_main_done_label.coverage.js keys on.
function makePage() {
  clock.now = NOW;
  return harness.makePage({
    sourceUrl: 'audit0902-donelabel-main.js',
    beforeScripts(win) {
      win.Date.now = () => clock.now;
    },
  });
}

// The Chats-panel pick: chat tabs have no row of their own any more.
// The pick is a click, which main.js counts as the user moving on
// since the submit (so a finishing background task no longer pulls
// them over); the gesture is dispatched before the switch.
function clickTab(win, tabId) {
  assert.ok(
    win._testApi.openTabs().some(t => t.id === tabId),
    `tab ${tabId} must be open`,
  );
  win.document.body.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  win._testApi.switchToTab(tabId);
}

function statusText(win) {
  return win.document.getElementById('status-text').textContent;
}

/** The Task Info panel's Time row, once the page's observers have run. */
async function timeRow(win) {
  await flushMicrotasks();
  return win.document.getElementById('meta-time').textContent;
}

/**
 * The daemon's replay of history row *taskId* into *tabId*: the task's
 * text, one event and the row's `extra` (its start / end stamps).
 */
function replay(tabId, taskId, extra) {
  return {
    type: 'task_events',
    tabId,
    chat_id: 'chat-' + tabId,
    task_id: taskId,
    task: 'task ' + taskId,
    events: [{type: 'system_output', text: 'output of ' + taskId + '\n'}],
    extra: JSON.stringify(extra),
  };
}

function startTask(win, tabId, startTs) {
  send(win, {type: 'setTaskText', text: 'work', tabId});
  send(win, {type: 'clear', chat_id: 'chat-' + tabId, tabId});
  send(win, {type: 'status', running: true, tabId, startTs});
}

function testSpanOnTheEventWins() {
  const {win} = makePage();
  const tabId = win._testApi.getActiveTabId();
  const now = NOW;
  startTask(win, tabId, now - 3000);
  send(win, {
    type: 'task_done',
    tabId,
    startTs: now - 125000,
    endTs: now,
  });
  assert.strictEqual(
    statusText(win),
    '2m 5s',
    'the label comes from the span stamped on the event',
  );
  // A span that ends before it starts is no span: the tab's own clock
  // is used instead of a clamped zero.
  startTask(win, tabId, now - 30000);
  send(win, {type: 'task_done', tabId, startTs: now, endTs: now - 1000});
  assert.strictEqual(
    statusText(win),
    '30s',
    "an inverted span falls back to the tab's clock",
  );
  win.close();
  console.log('  ok - the span on the event wins');
}

function testFallbackUsesTheFinishedTabsClock() {
  const {win} = makePage();
  const tabA = win._testApi.getActiveTabId();
  win._testApi.createNewTab();
  const tabB = win._testApi.getActiveTabId();
  const now = NOW;
  // A has been running for a minute in the background; B, on screen,
  // started five seconds ago.
  startTask(win, tabA, now - 60000);
  clickTab(win, tabB);
  startTask(win, tabB, now - 5000);
  assert.strictEqual(win._testApi.getActiveTabId(), tabB);

  // A finishes; the event carries no span.  The user clicked B since
  // submitting, so its task_done leaves them on B (focusFinishedTab
  // only undoes an agent-made switch); they go to A themselves.
  send(win, {type: 'task_done', tabId: tabA});
  assert.strictEqual(win._testApi.getActiveTabId(), tabB);
  clickTab(win, tabA);
  assert.strictEqual(
    statusText(win),
    '1m 0s',
    "a background tab's label is measured from ITS start",
  );
  // B is still running its own five-second-old task.
  clickTab(win, tabB);
  assert.strictEqual(
    statusText(win),
    'Running 5s',
    'the visible tab keeps its own clock',
  );
  win.close();
  console.log('  ok - the fallback uses the finished tab\u2019s clock');
}

function testFallbackOnTheVisibleTab() {
  const {win} = makePage();
  const tabId = win._testApi.getActiveTabId();
  startTask(win, tabId, NOW - 70000);
  send(win, {type: 'task_done', tabId});
  assert.strictEqual(
    statusText(win),
    '1m 10s',
    'the visible tab measures from its own start',
  );
  // A tab id the page does not know is ignored: the label on screen
  // stays as it is rather than throwing.
  send(win, {type: 'task_done', tabId: 'no-such-tab'});
  assert.strictEqual(
    statusText(win),
    '1m 10s',
    'an unknown tab changes nothing',
  );
  win.close();
  console.log('  ok - the fallback on the visible tab');
}

function testFallbackWithoutAnyStart() {
  const {win} = makePage();
  const tabId = win._testApi.getActiveTabId();
  // No status/startTs ever arrived for this tab.
  send(win, {type: 'clear', chat_id: 'c', tabId});
  send(win, {type: 'task_done', tabId});
  assert.strictEqual(
    statusText(win),
    '0s',
    'with no start recorded the task reads as an instant finish',
  );
  win.close();
  console.log('  ok - the fallback without any start');
}

// The reported bug: a finished task opened from the history list (or
// repainted by a reload) is replayed from its history row, and the
// Time row stayed on "Ready".
async function testReplayedFinishedTaskShowsItsTime() {
  const {win} = makePage();
  const tabId = win._testApi.getActiveTabId();
  const end = NOW - 3_600_000;
  // The daemon settles the tab's running flag before the transcript.
  send(win, {type: 'status', running: false, tabId});
  send(win, replay(tabId, 7, {startTs: end - 125_000, endTs: end}));
  assert.strictEqual(statusText(win), '2m 5s');
  assert.strictEqual(
    await timeRow(win),
    '2m 5s',
    'the Time row reads the time the task took, not Ready',
  );
  // A row that records no end (its daemon was killed mid-task): the
  // time is unknown, and it is not the time of the task shown before.
  send(win, replay(tabId, 8, {startTs: end}));
  assert.strictEqual(await timeRow(win), DASH, 'no end: no time');
  win.close();
  console.log('  ok - a replayed finished task shows its time');
}

// A chat without a history row replays as no task at all: the tab has
// nothing whose time could be shown, and does not keep the time of the
// task it showed before.
async function testTabWithoutATaskStaysReady() {
  const {win} = makePage();
  const tabId = win._testApi.getActiveTabId();
  const empty = {
    type: 'task_events',
    tabId,
    chat_id: 'chat-empty',
    task: '',
    events: [],
    extra: '',
  };
  send(win, {type: 'status', running: false, tabId});
  send(win, {...empty, task_id: null});
  assert.strictEqual(await timeRow(win), 'Ready');
  send(win, replay(tabId, 7, {startTs: NOW - 5000, endTs: NOW}));
  assert.strictEqual(await timeRow(win), '5s');
  // A replay that names no task at all is one without a task too.
  send(win, empty);
  assert.strictEqual(await timeRow(win), 'Ready');
  win.close();
  console.log('  ok - a tab without a task stays Ready');
}

// A running task's replay leaves the clock ticking, on screen and in
// a background tab.
async function testReplayedRunningTaskKeepsItsClock() {
  const {win} = makePage();
  const tabA = win._testApi.getActiveTabId();
  const running = {type: 'status', running: true, attached: true};
  send(win, {...running, tabId: tabA, startTs: NOW - 65_000});
  send(win, replay(tabA, 3, {startTs: NOW - 65_000}));
  assert.strictEqual(await timeRow(win), 'Running 1m 5s');

  win._testApi.createNewTab();
  const tabB = win._testApi.getActiveTabId();
  clickTab(win, tabA);
  send(win, {...running, tabId: tabB, startTs: NOW - 30_000});
  send(win, replay(tabB, 4, {startTs: NOW - 30_000}));
  clickTab(win, tabB);
  assert.strictEqual(await timeRow(win), 'Running 30s');
  win.close();
  console.log('  ok - a replayed running task keeps its clock');
}

// A background tab's replay of a row without an end shows no time
// either, instead of whatever the tab said before.
async function testBackgroundReplayWithoutAnEnd() {
  const {win} = makePage();
  const tabA = win._testApi.getActiveTabId();
  win._testApi.createNewTab();
  send(win, {type: 'status', running: false, tabId: tabA});
  send(win, replay(tabA, 5, {startTs: NOW - 90_000}));
  clickTab(win, tabA);
  assert.strictEqual(await timeRow(win), DASH);
  win.close();
  console.log('  ok - a background replay without an end shows no time');
}

// A task that failed, was stopped or was interrupted is over too: the
// row shows how long it ran, by the event's span or the tab's clock.
async function testEveryEndShowsTheTime() {
  for (const type of ['task_error', 'task_stopped', 'task_interrupted']) {
    const {win} = makePage();
    const tabId = win._testApi.getActiveTabId();
    startTask(win, tabId, NOW - 3000);
    send(win, {type, tabId, startTs: NOW - 65_000, endTs: NOW});
    assert.strictEqual(await timeRow(win), '1m 5s', type + ' with a span');
    startTask(win, tabId, NOW - 42_000);
    send(win, {type, tabId});
    assert.strictEqual(await timeRow(win), '42s', type + ' without a span');
    win.close();
  }
  console.log('  ok - every end shows the time the task took');
}

// An end the page learns of from a bare `status` (a start that failed,
// an end it missed) stops the clock when that status arrives.
async function testStatusOnlyEndStopsTheClock() {
  const {win} = makePage();
  const tabA = win._testApi.getActiveTabId();
  startTask(win, tabA, NOW - 42_000);
  send(win, {type: 'status', running: false, tabId: tabA});
  assert.strictEqual(await timeRow(win), '42s', 'on screen');

  // In a background tab: once last seen running, once started unseen.
  startTask(win, tabA, NOW - 90_000);
  win._testApi.createNewTab();
  const tabB = win._testApi.getActiveTabId();
  win._testApi.createNewTab();
  send(win, {type: 'status', running: true, tabId: tabB, startTs: NOW - 10_000});
  clickTab(win, win._testApi.openTabs().pop().id);
  send(win, {type: 'status', running: false, tabId: tabA});
  send(win, {type: 'status', running: false, tabId: tabB});
  // The reader comes back a minute later: the tasks ended back then.
  clock.now += 60_000;
  clickTab(win, tabA);
  assert.strictEqual(await timeRow(win), '1m 30s', 'last seen running');
  clickTab(win, tabB);
  assert.strictEqual(await timeRow(win), '10s', 'started unseen');
  win.close();
  console.log('  ok - a status-only end stops the clock');
}

// What the tab's clock already knows wins over the moment the status
// arrived: an end a replay brought is kept, and a tab whose start the
// page never learned has no time to show.
async function testStatusOnlyEndReadsTheTabsClock() {
  const {win} = makePage();
  const tabA = win._testApi.getActiveTabId();
  const span = {startTs: NOW - 300_000, endTs: NOW - 175_000};
  const stopped = {type: 'status', running: false};
  // On screen: the replayed row records the end before the status does.
  startTask(win, tabA, span.startTs);
  send(win, replay(tabA, 7, span));
  send(win, {...stopped, tabId: tabA});
  assert.strictEqual(await timeRow(win), '2m 5s', 'on screen');

  // The same in a background tab, beside one told to run without a start.
  win._testApi.createNewTab();
  const tabB = win._testApi.getActiveTabId();
  win._testApi.createNewTab();
  const tabC = win._testApi.getActiveTabId();
  clickTab(win, tabC);
  startTask(win, tabA, span.startTs);
  send(win, replay(tabA, 8, span));
  send(win, {type: 'status', running: true, tabId: tabB});
  assert.strictEqual(win._testApi.getActiveTabId(), tabC, 'both run unseen');
  send(win, {...stopped, tabId: tabA});
  send(win, {...stopped, tabId: tabB});
  clickTab(win, tabA);
  assert.strictEqual(await timeRow(win), '2m 5s', 'in a background tab');
  clickTab(win, tabB);
  assert.strictEqual(await timeRow(win), DASH, 'no start: no time');
  win.close();
  console.log("  ok - a status-only end reads the tab's clock");
}

// A file opened from a chat covers it (the stacked layout) and has no
// task of its own, so the Task Info panel keeps describing the chat
// behind it.  When that chat's task ends, the row shows the recorded
// time there too, whichever way the page learns of the end.
async function testFileViewShowsTheTimeOfTheChatBehindIt() {
  const span = {startTs: NOW - 180_000, endTs: NOW - 60_000};
  const stopped = {type: 'status', running: false};
  const flows = {
    'the end, then the status': () => [{type: 'task_done', ...span}, stopped],
    'the status, then the end': () => [stopped, {type: 'task_stopped', ...span}],
    'a reconnect that replays the finished row': chat => [
      stopped,
      replay(chat, 7, span),
    ],
  };
  for (const [flow, events] of Object.entries(flows)) {
    const {win} = makePage();
    const chat = win._testApi.getActiveTabId();
    startTask(win, chat, span.startTs);
    send(win, {
      type: 'fileContent',
      tabId: chat,
      name: 'notes.txt',
      path: '/tmp/notes.txt',
      content: 'notes',
    });
    const file = win._testApi.getActiveTabId();
    assert.notStrictEqual(file, chat, 'the file is in front');
    // A click: the reader moved on since the submit, so the end of the
    // task leaves the file in front.
    win.document.body.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
    assert.strictEqual(await timeRow(win), 'Running 3m 0s', flow);
    events(chat).forEach(ev => send(win, {...ev, tabId: chat}));
    assert.strictEqual(win._testApi.getActiveTabId(), file, flow);
    assert.strictEqual(await timeRow(win), '2m 0s', flow);
    clickTab(win, chat);
    assert.strictEqual(await timeRow(win), '2m 0s', flow + ': back in the chat');
    win.close();
  }
  console.log('  ok - a file view shows the time of the chat behind it');
}

// While the reader is parked on an earlier task of the thread the
// panel's rows describe that task, and so does the Time row.  jsdom
// lays nothing out, so the first region of the thread is the one on
// screen: the neighbour spliced in above.
async function testNeighbourTaskShowsItsOwnTime() {
  const {win} = makePage();
  const tabId = win._testApi.getActiveTabId();
  const end = NOW - 3_600_000;
  const own = replay(tabId, 7, {startTs: end - 125_000, endTs: end});
  send(win, {type: 'status', running: false, tabId});
  send(win, own);
  const neighbour = {
    type: 'adjacent_task_events',
    tabId,
    direction: 'prev',
    events: [{type: 'system_output', text: 'earlier output\n'}],
  };
  send(win, {
    ...neighbour,
    task: 'task 6',
    task_id: '6',
    extra: JSON.stringify({startTs: end - 400_000, endTs: end - 355_000}),
  });
  assert.strictEqual(await timeRow(win), '45s', "the neighbour's time");
  assert.strictEqual(
    win.document.getElementById('meta-time').style.color,
    'var(--dim)',
    'in the colour of a task that is over',
  );
  assert.strictEqual(statusText(win), '2m 5s', "the tab's label is its own");
  // A neighbour whose reply carries no span has no time to show.
  send(win, {...neighbour, task: 'task 5', task_id: '5'});
  assert.strictEqual(await timeRow(win), DASH);
  // The tab's next replay takes the rows back, though its label reads
  // the same as before.
  send(win, own);
  assert.strictEqual(await timeRow(win), '2m 5s');
  win.close();
  console.log('  ok - a neighbouring task shows its own time');
}

async function main() {
  testSpanOnTheEventWins();
  testFallbackUsesTheFinishedTabsClock();
  testFallbackOnTheVisibleTab();
  testFallbackWithoutAnyStart();
  await testReplayedFinishedTaskShowsItsTime();
  await testTabWithoutATaskStaysReady();
  await testReplayedRunningTaskKeepsItsClock();
  await testBackgroundReplayWithoutAnEnd();
  await testEveryEndShowsTheTime();
  await testStatusOnlyEndStopsTheClock();
  await testStatusOnlyEndReadsTheTabsClock();
  await testFileViewShowsTheTimeOfTheChatBehindIt();
  await testNeighbourTaskShowsItsOwnTime();
  console.log('all audit0902 done-label tests passed');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
