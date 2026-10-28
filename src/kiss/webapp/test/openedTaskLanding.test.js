// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end tests for where an opened chat lands (main.js
// landOpenedTranscript): a chat whose task is running lands at the end
// of its transcript, where the live output arrives; a chat whose task
// is finished lands with its own task panel at the top of the viewport
// and the result right below it.  Both when the daemon replays the
// task into the visible tab (opening a chat from the history) and when
// the user switches to a tab that holds the transcript.  A reconnect
// replay of a finished task is preceded by the daemon's status
// running:false, which settles a tab that missed the task's end; a
// deferred reconnect replay (a tab not on screen) re-sends the tab's
// running state, so a background tab that missed a run's start spins.
//
// jsdom lays nothing out, so a layout model stands in: every top-level
// child of #output is ROW_H px tall except the result panel (RESULT_H),
// stacked from the top; the viewport is VIEW_H px tall; rects follow
// #output's scrollTop.  The result is taller than the viewport, so a
// transcript landed at its bottom hides the task panel and one landed
// on the task panel cuts the result: the two landings are distinct.

'use strict';

const assert = require('assert');
const {
  makePage,
  send,
  readyTabId,
  clickTab,
  twoTabs,
  ofType,
} = require('./simplify2_harness');
const {historyRows, makeRow} = require('./history_helpers');

const ROW_H = 100;
const RESULT_H = 1000;
const VIEW_H = 500;

function isOutput(el) {
  return el.id === 'output';
}

function rowHeight(el) {
  if (el.id === 'welcome' || el.id === 'adjacent-loader') return 0;
  if (el.classList.contains('rc')) return RESULT_H;
  return ROW_H;
}

/** Install the layout model into `win` before the page scripts run. */
function stubLayout(win) {
  const proto = win.Element.prototype;
  function rows(O) {
    return Array.from(O.children);
  }
  function totalHeight(O) {
    return rows(O).reduce((h, el) => h + rowHeight(el), 0);
  }
  function rowTop(O, el) {
    let y = 0;
    for (const row of rows(O)) {
      if (row === el) return y;
      y += rowHeight(row);
    }
    throw new Error('not a transcript row');
  }
  Object.defineProperty(proto, 'scrollHeight', {
    configurable: true,
    get() {
      return isOutput(this) ? totalHeight(this) : 0;
    },
  });
  Object.defineProperty(proto, 'clientHeight', {
    configurable: true,
    get() {
      return isOutput(this) ? VIEW_H : 0;
    },
  });
  proto.getBoundingClientRect = function () {
    if (isOutput(this)) {
      return {top: 0, bottom: VIEW_H, left: 0, right: 400, height: VIEW_H};
    }
    const O = this.parentNode;
    if (O && O.nodeType === 1 && isOutput(O)) {
      const top = rowTop(O, this) - O.scrollTop;
      const h = rowHeight(this);
      return {top, bottom: top + h, left: 0, right: 400, height: h};
    }
    return {top: 0, bottom: 0, left: 0, right: 0, height: 0};
  };
}

function setupPage() {
  const {win, posted} = makePage({
    bodyClass: 'remote-chat',
    beforeScripts: stubLayout,
    sourceUrl: 'opened-task-landing-main.js',
  });
  const tabId = readyTabId(posted);
  win._testApi.hideWelcome();
  const O = win.document.getElementById('output');
  return {win, posted, tabId, O};
}

/** A finished task's transcript: a few tool calls and a long result. */
function finishedEvents() {
  return [
    {type: 'tool_call', name: 'Read', args: {file_path: '/tmp/a'}},
    {type: 'tool_result', name: 'Read', text: 'contents of a\n'},
    {type: 'tool_call', name: 'Read', args: {file_path: '/tmp/b'}},
    {type: 'tool_result', name: 'Read', text: 'contents of b\n'},
    {type: 'result', text: 'The result.\n'},
  ];
}

function replay(win, tabId, taskId, task) {
  send(win, {
    type: 'task_events',
    tabId,
    chat_id: 'chat-1',
    task_id: taskId,
    task,
    events: finishedEvents(),
  });
}

/** Click the Chats-drawer row of task `taskId` of chat 'chat-1'. */
function clickHistoryRow(win, taskId, title) {
  send(win, {
    type: 'history',
    offset: 0,
    generation: 0,
    sessions: [
      makeRow({id: 'chat-1', task_id: taskId, title, has_events: true}),
    ],
  });
  historyRows(win)[0].click();
}

function spliceOlderTask(win, tabId) {
  send(win, {
    type: 'adjacent_task_events',
    tabId,
    direction: 'prev',
    task: 'Older task',
    task_id: '41',
    events: [
      {type: 'tool_call', name: 'Read', args: {file_path: '/tmp/x'}},
      {type: 'tool_result', name: 'Read', text: 'x\n'},
      {type: 'result', text: 'Older result.\n'},
    ],
  });
  const older = win.document.querySelector('.adjacent-task[data-task]');
  assert.ok(older, 'the older task is spliced into the transcript');
  return older;
}

function bottomOf(O) {
  return Math.max(0, O.scrollHeight - O.clientHeight);
}

/** Whether the page holds `tabId` running; its strip entry, when rendered, agrees. */
function isRunning(win, tabId) {
  const tab = win._testApi.openTabs().find(t => t.id === tabId);
  assert.ok(tab, `tab ${tabId} must be open`);
  const onStrip = win.document.querySelector(
    `#tab-list .chat-tab[data-tab-id=${JSON.stringify(tabId)}]`,
  );
  if (onStrip) {
    assert.strictEqual(
      Boolean(onStrip.querySelector('.chat-tab-spinner')),
      tab.isRunning,
      `the strip entry of ${tabId} shows its running state`,
    );
  }
  return tab.isRunning;
}

function taskPanel(O) {
  const panel = O.querySelector('.task-panel');
  assert.ok(panel, 'the transcript starts with its task panel');
  return panel;
}

function assertLandedOnTaskPanel(O, label) {
  const panel = taskPanel(O);
  assert.strictEqual(
    panel.getBoundingClientRect().top,
    0,
    label + ': the task panel sits at the top of the viewport',
  );
  const result = O.querySelector('.rc');
  assert.ok(result, label + ': the result panel is rendered');
  const rect = result.getBoundingClientRect();
  assert.ok(
    rect.top < VIEW_H && rect.top >= 0,
    label + ': the result starts inside the viewport (top ' + rect.top + ')',
  );
  assert.notStrictEqual(
    O.scrollTop,
    bottomOf(O),
    label + ': the transcript is not at its bottom',
  );
}

// --------------------------------------------------------------------
// Opening a finished chat (the daemon replays the task into the visible
// tab) lands on the task panel; the result follows right below it.
function testFinishedReplayLandsOnTaskPanel() {
  const {win, tabId, O} = setupPage();
  // A stale scroll offset from before the replay must not survive it.
  O.scrollTop = 33;
  replay(win, tabId, '42', 'Finished task');
  assert.ok(bottomOf(O) > ROW_H, 'the transcript is longer than the viewport');
  assertLandedOnTaskPanel(O, 'finished replay');
  assert.strictEqual(O.scrollTop, 0, 'the task panel is the first row');
  win.close();
  console.log('  ok - finished replay lands on the task panel');
}

// --------------------------------------------------------------------
// Opening a running chat lands at the end of the transcript, where the
// live output arrives.
function testRunningReplayLandsAtEnd() {
  const {win, tabId, O} = setupPage();
  send(win, {type: 'status', running: true, tabId, startTs: Date.now()});
  O.scrollTop = 33;
  replay(win, tabId, '42', 'Running task');
  assert.strictEqual(
    O.scrollTop,
    bottomOf(O),
    'a running task lands at the bottom of its transcript',
  );
  assert.ok(
    taskPanel(O).getBoundingClientRect().bottom < 0,
    'the task panel scrolled out above',
  );
  win.close();
  console.log('  ok - running replay lands at the end');
}

// --------------------------------------------------------------------
// Switching to a finished tab lands on ITS task panel, even when a
// neighbouring task was spliced in above it (the panel is then not the
// first row), and the tab label names that task.
function testSwitchToFinishedTabLandsOnOwnTaskPanel() {
  const {win, tabId, O} = setupPage();
  replay(win, tabId, '42', 'Own task');
  const older = spliceOlderTask(win, tabId);
  assert.strictEqual(O.firstElementChild, older, 'spliced in above');
  const olderHeight = older.getBoundingClientRect().height;
  // Lose the position: read the older task, then leave the tab.
  O.scrollTop = 0;
  const {second} = twoTabs(win);
  assert.strictEqual(win._testApi.getActiveTabId(), second);
  clickTab(win, tabId);
  assertLandedOnTaskPanel(O, 'switch back to a finished tab');
  assert.strictEqual(
    O.scrollTop,
    olderHeight,
    'landed on the own task panel below the spliced-in older task',
  );
  assert.strictEqual(
    older.getBoundingClientRect().bottom,
    0,
    'the older task ends where the viewport begins',
  );
  const label = win.document.querySelector('#tab-list .chat-tab.active');
  assert.ok(label, 'the strip shows the active tab');
  assert.ok(
    label.textContent.includes('Own task'),
    'the tab label names the own task; got ' + label.textContent,
  );
  win.close();
  console.log('  ok - switching to a finished tab lands on its task panel');
}

// --------------------------------------------------------------------
// Switching to a running tab lands at the end of its transcript, which
// may have grown while the tab was hidden.
function testSwitchToRunningTabLandsAtEnd() {
  const {win, tabId, O} = setupPage();
  send(win, {type: 'status', running: true, tabId, startTs: Date.now()});
  replay(win, tabId, '42', 'Running task');
  O.scrollTop = 0;
  const {second} = twoTabs(win);
  assert.strictEqual(win._testApi.getActiveTabId(), second);
  // Output streams into the hidden tab meanwhile.
  send(win, {type: 'system_output', text: 'more\n', tabId});
  send(win, {type: 'system_output', text: 'and more\n', tabId});
  clickTab(win, tabId);
  assert.ok(bottomOf(O) > 0, 'the transcript is longer than the viewport');
  assert.strictEqual(
    O.scrollTop,
    bottomOf(O),
    'a running tab comes back at the bottom of its transcript',
  );
  win.close();
  console.log('  ok - switching to a running tab lands at the end');
}

// --------------------------------------------------------------------
// Switching to a fresh chat on its welcome screen keeps the welcome
// screen at the top: there is no transcript to land.
function testSwitchToWelcomeTabStaysAtTop() {
  const {win, tabId, O} = setupPage();
  replay(win, tabId, '42', 'Own task');
  const {second} = twoTabs(win);
  clickTab(win, tabId);
  clickTab(win, second);
  assert.strictEqual(win._testApi.getActiveTabId(), second);
  assert.strictEqual(O.querySelector('.task-panel'), null, 'no task panel');
  const welcome = win.document.getElementById('welcome');
  assert.notStrictEqual(welcome.style.display, 'none', 'welcome shown');
  assert.strictEqual(O.scrollTop, 0, 'the welcome screen sits at the top');
  win.close();
  console.log('  ok - switching to a welcome tab stays at the top');
}

// --------------------------------------------------------------------
// A replay with nothing to show (no task text, no events) has no task
// region to land on; it lands at its end like a running one, without
// throwing.
function testEmptyReplayLandsAtEnd() {
  const {win, tabId, O} = setupPage();
  O.scrollTop = 33;
  send(win, {type: 'task_events', tabId, chat_id: 'chat-1', events: []});
  assert.strictEqual(O.querySelector('.task-panel'), null, 'no task panel');
  assert.strictEqual(O.scrollTop, bottomOf(O), 'landed at the end');
  win.close();
  console.log('  ok - an empty replay lands at its end');
}

// --------------------------------------------------------------------
// A history click on the running task the tab already shows opens it
// at the end of its transcript, not on the task's text.
function testHistoryClickOnRunningTaskLandsAtEnd() {
  const {win, posted, tabId, O} = setupPage();
  send(win, {type: 'status', running: true, tabId, startTs: Date.now()});
  replay(win, tabId, '42', 'Running task');
  O.scrollTop = 0;
  O.dispatchEvent(new win.Event('scroll'));
  const resumes = ofType(posted, 'resumeSession').length;
  clickHistoryRow(win, 42, 'Running task');
  assert.strictEqual(win._testApi.getActiveTabId(), tabId, 'same tab');
  assert.strictEqual(
    ofType(posted, 'resumeSession').length,
    resumes,
    'the task is on screen already: no replay is asked for',
  );
  assert.strictEqual(
    O.scrollTop,
    bottomOf(O),
    'the running task opens at the end of its transcript',
  );
  win.close();
  console.log('  ok - a history click on a running task lands at the end');
}

// --------------------------------------------------------------------
// A history click on the finished task the tab shows puts its task
// panel at the top, however far down the reader was.
function testHistoryClickOnFinishedTaskLandsOnTaskPanel() {
  const {win, tabId, O} = setupPage();
  replay(win, tabId, '42', 'Finished task');
  O.scrollTop = bottomOf(O);
  O.dispatchEvent(new win.Event('scroll'));
  clickHistoryRow(win, 42, 'Finished task');
  assertLandedOnTaskPanel(O, 'history click on a finished task');
  win.close();
  console.log('  ok - a history click on a finished task lands on its panel');
}

// --------------------------------------------------------------------
// A replay the page asked for lands afresh.  The reader left a finished
// tab landed below a spliced-in older task (a position away from the
// bottom, which engages the scroll lock a reconnect replay honours);
// a history click on another task of the chat replaces the transcript,
// and that replay must land on the new task's panel, not keep the old
// offset.
function testRequestedReplayLandsAfresh() {
  const {win, posted, tabId, O} = setupPage();
  replay(win, tabId, '42', 'Own task');
  const older = spliceOlderTask(win, tabId);
  const {second} = twoTabs(win);
  assert.strictEqual(win._testApi.getActiveTabId(), second);
  clickTab(win, tabId);
  assert.strictEqual(O.scrollTop, older.getBoundingClientRect().height);
  // The browser reports the landing as a scroll: the lock engages.
  O.dispatchEvent(new win.Event('scroll'));
  const resumes = ofType(posted, 'resumeSession').length;
  clickHistoryRow(win, 43, 'Newer task');
  const asked = ofType(posted, 'resumeSession');
  assert.strictEqual(asked.length, resumes + 1, 'the unloaded task is fetched');
  assert.strictEqual(String(asked[asked.length - 1].taskId), '43');
  replay(win, tabId, '43', 'Newer task');
  assert.strictEqual(
    O.querySelector('.adjacent-task[data-task]'),
    null,
    'the replay replaced the transcript, older task included',
  );
  assertLandedOnTaskPanel(O, 'requested replay');
  assert.strictEqual(O.scrollTop, 0, 'the new task panel is the first row');
  win.close();
  console.log('  ok - a requested replay lands afresh');
}

// --------------------------------------------------------------------
// A reconnect replay of the task on screen that arrives while another
// task's replay is on its way neither answers that request nor lands
// afresh: it keeps the reading position, as a reconnect replay does,
// and the requested replay still lands on its own task panel.
function testUnrelatedReplayKeepsTheRequestPending() {
  const {win, posted, tabId, O} = setupPage();
  replay(win, tabId, '42', 'Own task');
  // Reading the result: a position away from the bottom, lock engaged.
  O.scrollTop = 150;
  O.dispatchEvent(new win.Event('scroll'));
  clickHistoryRow(win, 43, 'Newer task');
  const asked = ofType(posted, 'resumeSession');
  assert.strictEqual(String(asked[asked.length - 1].taskId), '43');
  // The reconnect replay of the task on screen comes first.
  replay(win, tabId, '42', 'Own task');
  assert.strictEqual(
    O.scrollTop,
    150,
    'the reconnect replay keeps the reading position',
  );
  O.scrollTop = 250;
  O.dispatchEvent(new win.Event('scroll'));
  replay(win, tabId, '43', 'Newer task');
  assertLandedOnTaskPanel(O, 'requested replay after a reconnect replay');
  assert.strictEqual(O.scrollTop, 0, 'the new task panel is the first row');
  win.close();
  console.log('  ok - an unrelated replay keeps the request pending');
}

// --------------------------------------------------------------------
// A deferred fetch of a chat (no task named) that brings no transcript
// back must not make a later reconnect replay land afresh: the new
// task that started meanwhile replaced the transcript, and its
// reconnect replay keeps the reading position.
function testNewTaskDropsThePendingRequest() {
  const {win, posted, tabId, O} = setupPage();
  replay(win, tabId, '42', 'Own task');
  const {second} = twoTabs(win);
  // The daemon withholds the first tab's transcript on a reconnect …
  send(win, {type: 'replay_deferred', tabId, chatId: 'chat-1'});
  const resumes = ofType(posted, 'resumeSession').length;
  clickTab(win, tabId);
  const asked = ofType(posted, 'resumeSession');
  assert.strictEqual(asked.length, resumes + 1, '… and the tab fetches it');
  assert.strictEqual(asked[asked.length - 1].tabId, tabId);
  assert.ok(!('taskId' in asked[asked.length - 1]), 'the chat, no task');
  // Nothing comes back (no persisted task); a new task starts instead.
  send(win, {type: 'clear', tabId});
  send(win, {type: 'status', running: true, tabId, startTs: Date.now()});
  send(win, {type: 'system_output', text: 'new run\n', tabId});
  send(win, {type: 'result', text: 'New result.\n', tabId});
  send(win, {type: 'status', running: false, tabId});
  O.scrollTop = 100;
  O.dispatchEvent(new win.Event('scroll'));
  // Its reconnect replay, unasked for, keeps the reading position.
  replay(win, tabId, '44', 'New task');
  assert.strictEqual(O.scrollTop, 100, 'the reading position is kept');
  assert.strictEqual(win._testApi.getActiveTabId(), tabId);
  assert.ok(second, 'the other tab is still open');
  win.close();
  console.log('  ok - a new task drops the pending request');
}

// --------------------------------------------------------------------
// A tab that saw its task's running:true, lost its connection and
// missed the task's end reconnects with a stale running flag.  The
// daemon's reconnect replay of the finished task now settles the flag
// with a status running:false ahead of the transcript, so the replay
// lands on the task panel, not at the end, and the tab stops running.
function testReconnectReplaySettlesStaleRunningFlag() {
  const {win, posted, tabId, O} = setupPage();
  send(win, {type: 'daemonStatus', connected: true});
  send(win, {type: 'status', running: true, tabId, startTs: Date.now()});
  send(win, {type: 'system_output', text: 'working\n', tabId});
  assert.ok(isRunning(win, tabId), 'the tab shows its task running');
  const readies = ofType(posted, 'ready').length;
  // The outage: the task ends while the page is away.
  send(win, {type: 'daemonStatus', connected: false});
  send(win, {type: 'daemonStatus', connected: true});
  assert.strictEqual(
    ofType(posted, 'ready').length,
    readies + 1,
    'the reconnect re-sends ready',
  );
  assert.ok(isRunning(win, tabId), 'the flag is stale until the daemon speaks');
  // The daemon's answer: the idle status, then the finished transcript.
  send(win, {type: 'status', running: false, tabId});
  replay(win, tabId, '42', 'Finished while away');
  assert.ok(!isRunning(win, tabId), 'the tab is settled as finished');
  assertLandedOnTaskPanel(O, 'reconnect replay of a finished task');
  win.close();
  console.log('  ok - a reconnect replay settles a stale running flag');
}

// --------------------------------------------------------------------
// The other half: a background tab whose task is still running when
// the page reconnects.  The page showed the tab idle (its run started
// from another window during the outage, so the page never saw the
// run's running:true).  The daemon's deferred answer for a tab not on
// screen is its running state plus a replay_deferred offer, no
// transcript: the spinner comes on, the tab on screen keeps the
// window (the launch switch was spent on the earlier task's news, and
// the status is `attached`), and opening the tab fetches the
// transcript, which lands at the end as a running task's does.
function testDeferredReconnectResendsRunningToBackgroundTab() {
  const {win, posted, tabId, O} = setupPage();
  send(win, {type: 'daemonStatus', connected: true});
  // An earlier task ran to its end on the tab while the page watched.
  send(win, {type: 'status', running: true, tabId, startTs: Date.now()});
  send(win, {type: 'status', running: false, tabId});
  replay(win, tabId, '42', 'Earlier task');
  const {second} = twoTabs(win);
  assert.strictEqual(win._testApi.getActiveTabId(), second);
  assert.ok(!isRunning(win, tabId), 'the background tab shows no run');
  const readies = ofType(posted, 'ready').length;
  // The outage: a run starts on the background tab while the page is away.
  send(win, {type: 'daemonStatus', connected: false});
  send(win, {type: 'daemonStatus', connected: true});
  assert.strictEqual(
    ofType(posted, 'ready').length,
    readies + 1,
    'the reconnect re-sends ready',
  );
  assert.ok(!isRunning(win, tabId), 'the flag is stale until the daemon speaks');
  // The daemon's deferred answer: the running state, then the offer.
  const startTs = Date.now() - 5000;
  send(win, {type: 'status', running: true, tabId, startTs, attached: true});
  send(win, {type: 'replay_deferred', tabId, chatId: 'chat-1', taskId: '43'});
  assert.ok(isRunning(win, tabId), 'the background tab shows its run');
  assert.strictEqual(
    win._testApi.getActiveTabId(),
    second,
    'a re-sent running state does not steal the window',
  );
  // Opening the tab fetches the transcript it was offered.
  const resumes = ofType(posted, 'resumeSession').length;
  clickTab(win, tabId);
  const resume = ofType(posted, 'resumeSession').slice(resumes).pop();
  assert.ok(resume, 'opening the offered tab asks for its transcript');
  assert.strictEqual(resume.tabId, tabId);
  send(win, {
    type: 'task_events',
    tabId,
    chat_id: 'chat-1',
    task_id: '43',
    task: 'Started while away',
    events: finishedEvents().slice(0, 4),
  });
  assert.ok(isRunning(win, tabId), 'the replay keeps the tab running');
  assert.strictEqual(
    O.scrollTop,
    bottomOf(O),
    'a running tab lands at the end of its transcript',
  );
  win.close();
  console.log('  ok - a deferred reconnect re-sends running to a background tab');
}

testReconnectReplaySettlesStaleRunningFlag();
testDeferredReconnectResendsRunningToBackgroundTab();
testHistoryClickOnRunningTaskLandsAtEnd();
testHistoryClickOnFinishedTaskLandsOnTaskPanel();
testRequestedReplayLandsAfresh();
testUnrelatedReplayKeepsTheRequestPending();
testNewTaskDropsThePendingRequest();
testFinishedReplayLandsOnTaskPanel();
testRunningReplayLandsAtEnd();
testSwitchToFinishedTabLandsOnOwnTaskPanel();
testSwitchToRunningTabLandsAtEnd();
testSwitchToWelcomeTabStaysAtTop();
testEmptyReplayLandsAtEnd();
console.log('PASS openedTaskLanding');
