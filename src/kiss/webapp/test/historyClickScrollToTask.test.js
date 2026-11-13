// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Clicking a task in the Task History sidebar must land the reader ON that
// task: after switching to (or loading) the tab that shows the task's chat,
// the transcript is scrolled so the clicked task's region is what the reader
// sees, the tab's label names the clicked task, and
// the chat shows the clicked task's events.
//
// Four scenarios:
//   1. The clicked task is already spliced into the open tab's transcript as
//      an `.adjacent-task` neighbour -> scroll to it, no resumeSession.
//   2. The clicked task's events are NOT loaded in the open tab -> the tab is
//      replayed at that task via resumeSession({id, taskId, tabId}).
//   3. The clicked task IS the tab's own task -> scroll back to it, and no
//      resumeSession is issued (parity with historyClickNoInputCopy).
//   4. The tab's task is RUNNING and the clicked task is not loaded -> the
//      tab stays on its running task (replayed at an older one it would show
//      that task as running and take the live output into it); the thread
//      is read back, one getAdjacentTask('prev') per answer, until the
//      clicked task is spliced in, also when the tab's own transcript is
//      still to be fetched (replay_deferred).

'use strict';

const assert = require('assert');
const {
  makePage,
  send,
  sendEmptyWorkDir,
  readyTabId,
  ofType,
  twoTabs,
  sendTabsState,
  plain,
} = require('./simplify2_harness');
// chatTabs: the open chats.  The group strip renders only the chat on
// screen, so a DOM count of `.chat-tab` is 1 whatever their number.
const {historyRows, makeRow, chatTabs} = require('./history_helpers');

/** The active tab's label: it names the task the reader is looking at. */
function shownTask(win) {
  const el = win.document.querySelector(
    '.chat-tab[aria-selected="true"] .chat-tab-label',
  );
  return el ? el.textContent : '';
}

function resumeMessages(posted) {
  return ofType(posted, 'resumeSession');
}

function historyRow(win, idx) {
  return historyRows(win)[idx];
}

function historySession(id, taskId, title) {
  return makeRow({id, task_id: taskId, title, has_events: true});
}

// jsdom computes no layout, so give the transcript a fake geometry that
// tracks O.scrollTop: the neighbour occupies content rows [0, 1000) and the
// tab's own task rows [1000, 2000), with a 500px viewport.
function fakeRegionGeometry(O) {
  Object.defineProperty(O, 'scrollHeight', {value: 2000, configurable: true});
  Object.defineProperty(O, 'clientHeight', {value: 500, configurable: true});
  O.getBoundingClientRect = () => ({top: 0, bottom: 500, left: 0, right: 400});
  const container = O.querySelector('.adjacent-task');
  container.getBoundingClientRect = () => ({
    top: 0 - O.scrollTop,
    bottom: 1000 - O.scrollTop,
    left: 0,
    right: 400,
  });
  for (const el of O.children) {
    if (el === container) continue;
    el.getBoundingClientRect = () => ({
      top: 1000 - O.scrollTop,
      bottom: 2000 - O.scrollTop,
      left: 0,
      right: 400,
    });
  }
}

// One open tab on chat-1 showing its own "Task B" (id 42) with a previous
// "Task A" (id 41) spliced in above it as an adjacent region.
function setupTabWithNeighbour() {
  const {win, posted} = makePage();
  sendEmptyWorkDir(win);
  const tabId = readyTabId(posted);
  win._testApi.hideWelcome();
  send(win, {
    type: 'task_events',
    tabId,
    chat_id: 'chat-1',
    task_id: '42',
    task: 'Task B',
    events: [{type: 'system_output', text: 'output of task B\n'}],
  });
  send(win, {
    type: 'adjacent_task_events',
    tabId,
    direction: 'prev',
    task: 'Task A',
    task_id: '41',
    events: [{type: 'system_output', text: 'output of task A\n'}],
  });
  const O = win.document.getElementById('output');
  assert.ok(
    O.querySelector('.adjacent-task[data-task-id="41"]'),
    'sanity: the neighbour task must be spliced into the transcript',
  );
  // Park the reader on the tab's own task, as after a real splice.
  fakeRegionGeometry(O);
  O.scrollTop = 1200;
  O.dispatchEvent(new win.Event('scroll'));
  assert.strictEqual(
    shownTask(win),
    'Task B',
    'sanity: the panel names the tab own task before the history click',
  );
  return {win, posted, tabId, O};
}

function testClickScrollsToSplicedNeighbour() {
  const {win, posted, tabId, O} = setupTabWithNeighbour();
  const resumeBefore = resumeMessages(posted).length;

  clickTask(win, 41, 'Task A');

  assert.strictEqual(
    chatTabs(win).length,
    1,
    'clicking a task of the already-open chat must not open a new tab',
  );
  assert.strictEqual(
    win._testApi.getActiveTabId(),
    tabId,
    'the open tab showing the chat must stay active',
  );
  assert.strictEqual(
    resumeMessages(posted).length,
    resumeBefore,
    'a task already spliced into the transcript needs no resumeSession',
  );
  assert.strictEqual(
    O.scrollTop,
    0,
    'the transcript must scroll the clicked task region to the top',
  );
  assert.strictEqual(
    shownTask(win),
    'Task A',
    'the tab label must name the clicked task after the scroll',
  );
  win.close();
  console.log('PASS history click scrolls to the spliced neighbour task');
}

function testClickScrollsBackToOwnTask() {
  const {win, posted, tabId, O} = setupTabWithNeighbour();
  // Park the reader on the neighbour first, as a real scroll would.
  send(win, {
    type: 'history',
    offset: 0,
    generation: 0,
    sessions: [
      historySession('chat-1', 41, 'Task A'),
      historySession('chat-1', 42, 'Task B'),
    ],
  });
  historyRow(win, 0).click();
  assert.strictEqual(shownTask(win), 'Task A', 'sanity: parked on Task A');
  const resumeBefore = resumeMessages(posted).length;

  historyRow(win, 1).click();

  assert.strictEqual(
    resumeMessages(posted).length,
    resumeBefore,
    'the tab own task is already loaded: no resumeSession',
  );
  assert.strictEqual(
    win._testApi.getActiveTabId(),
    tabId,
    'the open tab showing the chat must stay active',
  );
  assert.strictEqual(
    O.scrollTop,
    1000,
    'the transcript must scroll the own task region to the top',
  );
  assert.strictEqual(
    shownTask(win),
    'Task B',
    'clicking the tab own task must scroll back and rename the panel',
  );
  win.close();
  console.log('PASS history click scrolls back to the tab own task');
}

function testClickLoadsTaskNotInTranscript() {
  const {win, posted, tabId} = setupTabWithNeighbour();
  const resumeBefore = resumeMessages(posted).length;

  clickTask(win, 40, 'Task Z');

  assert.strictEqual(
    chatTabs(win).length,
    1,
    'loading another task of the open chat must reuse its tab',
  );
  const resumes = resumeMessages(posted);
  assert.strictEqual(
    resumes.length,
    resumeBefore + 1,
    'a task whose events are not in the transcript must be fetched',
  );
  const resume = resumes[resumes.length - 1];
  assert.strictEqual(resume.chatId, 'chat-1', 'resumeSession must name the chat');
  assert.strictEqual(resume.taskId, 40, 'resumeSession must name the task');
  assert.strictEqual(
    resume.tabId,
    tabId,
    'resumeSession must replay into the tab already showing the chat',
  );

  // The daemon answers with the clicked task's events; the tab must show
  // them and the panel must name the clicked task.
  send(win, {
    type: 'task_events',
    tabId,
    chat_id: 'chat-1',
    task_id: '40',
    task: 'Task Z',
    events: [{type: 'system_output', text: 'output of task Z\n'}],
  });
  assert.strictEqual(
    shownTask(win),
    'Task Z',
    'the tab label must name the loaded task',
  );
  const out = win.document.getElementById('output').textContent;
  assert.ok(
    out.includes('output of task Z'),
    'the chat must show the loaded task events',
  );
  win.close();
  console.log('PASS history click loads a task missing from the transcript');
}

function testClickRowWithoutTaskIdJustSwitches() {
  const {win, posted, tabId} = setupTabWithNeighbour();
  const resumeBefore = resumeMessages(posted).length;

  clickTask(win, null, 'Untitled chat');

  assert.strictEqual(
    win._testApi.getActiveTabId(),
    tabId,
    'a row without a task id still switches to the open chat tab',
  );
  assert.strictEqual(
    resumeMessages(posted).length,
    resumeBefore,
    'a row without a task id has nothing to fetch: no resumeSession',
  );
  win.close();
  console.log('PASS history row without a task id only switches tabs');
}

// The tab's own task produced no output, so it has no rendered region,
// while a spliced-in neighbour fills the transcript and the reader has
// scrolled into it.  Clicking the own task's history row must reclaim
// the static panel and the status row from the neighbour — without
// issuing a resumeSession, since the own task is already loaded.
function testClickOwnTaskWithoutRegionReclaimsPanel() {
  const {win, posted} = makePage();
  sendEmptyWorkDir(win);
  const tabId = readyTabId(posted);
  win._testApi.hideWelcome();
  // The tab's own "Task B" (id 42) recorded no output at all.
  send(win, {
    type: 'task_events',
    tabId,
    chat_id: 'chat-1',
    task_id: '42',
    task: 'Task B',
    events: [],
  });
  send(win, {
    type: 'adjacent_task_events',
    tabId,
    direction: 'prev',
    task: 'Task A',
    task_id: '41',
    events: [{type: 'system_output', text: 'output of task A\n'}],
  });
  const O = win.document.getElementById('output');
  const container = O.querySelector('.adjacent-task[data-task-id="41"]');
  assert.ok(
    container,
    'sanity: the neighbour task must be spliced into the transcript',
  );
  // The neighbour is the only rendered region: fake its geometry and give
  // it borrowable metrics, then scroll into it as a real reader would.
  Object.defineProperty(O, 'scrollHeight', {value: 1000, configurable: true});
  Object.defineProperty(O, 'clientHeight', {value: 500, configurable: true});
  O.getBoundingClientRect = () => ({top: 0, bottom: 500, left: 0, right: 400});
  container.getBoundingClientRect = () => ({
    top: 0 - O.scrollTop,
    bottom: 1000 - O.scrollTop,
    left: 0,
    right: 400,
  });
  container.dataset.metricTokens = 'Tokens: 999';
  O.scrollTop = 100;
  O.dispatchEvent(new win.Event('scroll'));
  assert.strictEqual(
    shownTask(win),
    'Task A',
    'sanity: the panel names the neighbour once the reader scrolls into it',
  );
  const statusTokens = win.document.getElementById('status-tokens');
  assert.strictEqual(
    statusTokens.textContent,
    'Tokens: 999',
    'sanity: the status row is lent to the neighbour before the click',
  );
  const resumeBefore = resumeMessages(posted).length;

  clickTask(win, 42, 'Task B');

  assert.strictEqual(
    chatTabs(win).length,
    1,
    'clicking the own task of the open chat must not open a new tab',
  );
  assert.strictEqual(
    win._testApi.getActiveTabId(),
    tabId,
    'the open tab showing the chat must stay active',
  );
  assert.strictEqual(
    resumeMessages(posted).length,
    resumeBefore,
    'the tab own task is already loaded: no resumeSession',
  );
  assert.strictEqual(
    shownTask(win),
    'Task B',
    'the tab label must be reclaimed for the clicked own task',
  );
  assert.strictEqual(
    statusTokens.textContent,
    '',
    'the status row must show the own task metrics again after the click',
  );
  win.close();
  console.log('PASS history click reclaims the label for a region-less task');
}

/** Click the history row of task `taskId` of chat-1. */
function clickTask(win, taskId, title) {
  send(win, {
    type: 'history',
    offset: 0,
    generation: 0,
    sessions: [historySession('chat-1', taskId, title)],
  });
  historyRow(win, 0).click();
}

/** The daemon's answer to the last getAdjacentTask: `task` (null: none). */
function answerPrev(win, posted, tabId, task) {
  const asked = ofType(posted, 'getAdjacentTask').pop();
  send(win, {
    type: 'adjacent_task_events',
    tabId,
    direction: 'prev',
    adjacent_to: String(asked.taskId),
    task: task ? task.title : '',
    task_id: task ? task.id : null,
    events: task ? [{type: 'system_output', text: task.title + '\n'}] : [],
  });
}

/** The (taskId, direction) of every getAdjacentTask from index `from`. */
function askedSince(posted, from) {
  return ofType(posted, 'getAdjacentTask')
    .slice(from)
    .map(m => String(m.taskId) + ' ' + m.direction);
}

function isRunning(win, tabId) {
  return win._testApi.openTabs().find(t => t.id === tabId).isRunning;
}

function testClickOnOlderTaskOfRunningChatReadsBackToIt() {
  const {win, posted, tabId} = setupTabWithNeighbour();
  send(win, {type: 'status', running: true, tabId, startTs: Date.now()});
  const resumeBefore = resumeMessages(posted).length;
  const asked = ofType(posted, 'getAdjacentTask').length;

  clickTask(win, 39, 'Task Y');
  // The reader sat near the bottom, so the task below is on its way
  // (none: the running task is the chat's last); its answer starts
  // the reading back.
  assert.deepStrictEqual(askedSince(posted, asked), []);
  send(win, {
    type: 'adjacent_task_events',
    tabId,
    direction: 'next',
    adjacent_to: '42',
    task: '',
    task_id: null,
    events: [],
  });
  assert.deepStrictEqual(
    askedSince(posted, asked),
    ['41 prev'],
    'the thread is read back from its oldest task',
  );
  answerPrev(win, posted, tabId, {id: '40', title: 'Task Z'});
  assert.deepStrictEqual(
    askedSince(posted, asked),
    ['41 prev', '40 prev'],
    'the task before is not the clicked one: the thread reads on',
  );
  answerPrev(win, posted, tabId, {id: '39', title: 'Task Y'});
  assert.deepStrictEqual(
    askedSince(posted, asked),
    ['41 prev', '40 prev'],
    'the clicked task is in the transcript: nothing more is fetched',
  );
  assert.strictEqual(shownTask(win), 'Task Y', 'the reader lands on it');

  // A task the chat does not hold above: the search ends at the chat's
  // first task.
  clickTask(win, 30, 'Task of another time');
  answerPrev(win, posted, tabId, null);
  assert.deepStrictEqual(
    askedSince(posted, asked),
    ['41 prev', '40 prev', '39 prev'],
    'no older task: the search stops',
  );

  assert.strictEqual(
    resumeMessages(posted).length,
    resumeBefore,
    'a running chat is never replayed at an older task',
  );
  assert.ok(isRunning(win, tabId), 'the tab keeps its running task');
  win.close();
  console.log('PASS history click reads a running chat back to the task');
}

// A later click is what the reader wants: it ends the search an
// earlier click began.  Task 42 finishes with the search for task 39
// still waiting; task 40, clicked then, is replayed and stays on screen.
function testLaterClickEndsTheReadBack() {
  const {win, posted, tabId} = setupTabWithNeighbour();
  send(win, {type: 'status', running: true, tabId, startTs: Date.now()});
  clickTask(win, 39, 'Task Y');
  send(win, {type: 'status', running: false, tabId});
  const asked = ofType(posted, 'getAdjacentTask').length;

  clickTask(win, 40, 'Task Z');
  send(win, {
    type: 'task_events',
    tabId,
    chat_id: 'chat-1',
    task_id: '40',
    task: 'Task Z',
    events: [{type: 'system_output', text: 'output of task Z\n'}],
  });
  assert.deepStrictEqual(
    askedSince(posted, asked),
    [],
    'the search for the task clicked first is over',
  );
  assert.strictEqual(shownTask(win), 'Task Z');
  win.close();
  console.log('PASS a later history click ends the read-back');
}

function testClickOnOlderTaskOfRunningChatFetchesItsTranscriptFirst() {
  const {win, posted, tabId} = setupTabWithNeighbour();
  send(win, {type: 'daemonStatus', connected: true});
  twoTabs(win);
  // A reconnect: the daemon offers the background tab's transcript and
  // says its task (43) is running.  The reader is at work in the other
  // tab (the launch is over), so that news moves nobody.
  send(win, {type: 'daemonStatus', connected: false});
  send(win, {type: 'daemonStatus', connected: true});
  win._testApi.endLaunch();
  send(win, {
    type: 'status',
    running: true,
    tabId,
    startTs: Date.now(),
    attached: true,
  });
  send(win, {type: 'replay_deferred', tabId, chatId: 'chat-1', taskId: '43'});
  assert.notStrictEqual(win._testApi.getActiveTabId(), tabId);
  const resumeBefore = resumeMessages(posted).length;
  const asked = ofType(posted, 'getAdjacentTask').length;

  clickTask(win, 42, 'Task B');
  assert.strictEqual(win._testApi.getActiveTabId(), tabId);
  const resumes = resumeMessages(posted).slice(resumeBefore);
  assert.deepStrictEqual(
    resumes.map(r => [r.tabId, String(r.taskId)]),
    [[tabId, '43']],
    'the tab fetches the transcript it was offered, not the clicked task',
  );
  assert.deepStrictEqual(askedSince(posted, asked), [], 'nothing to read back from yet');

  send(win, {
    type: 'task_events',
    tabId,
    chat_id: 'chat-1',
    task_id: '43',
    task: 'Task C',
    events: [{type: 'system_output', text: 'output of task C\n'}],
  });
  assert.deepStrictEqual(
    askedSince(posted, asked),
    ['43 prev'],
    'the transcript is there: the thread is read back from it',
  );
  answerPrev(win, posted, tabId, {id: '42', title: 'Task B'});
  assert.strictEqual(shownTask(win), 'Task B', 'the reader lands on it');
  assert.ok(isRunning(win, tabId), 'the tab keeps its running task');
  win.close();
  console.log('PASS history click fetches a running chat, then reads back');
}

function testClickOnOlderTaskOfRunningChatWithoutTab() {
  const {win, posted} = setupTabWithNeighbour();
  const resumeBefore = resumeMessages(posted).length;
  const asked = ofType(posted, 'getAdjacentTask').length;
  // chat-9 runs task 53; the user closed its tab.  The click is on its
  // older task 51, the only row of the chat a search lists: the row
  // itself says its chat is running.
  send(win, {
    type: 'history',
    offset: 0,
    generation: 0,
    sessions: [
      makeRow({
        id: 'chat-9',
        task_id: 51,
        title: 'Task P',
        has_events: true,
        chat_running: true,
      }),
    ],
  });
  historyRow(win, 0).click();

  const tabId = win._testApi.getActiveTabId();
  const resumes = resumeMessages(posted).slice(resumeBefore);
  assert.deepStrictEqual(
    resumes.map(r => [r.chatId, r.tabId, r.taskId]),
    [['chat-9', tabId, undefined]],
    'a running chat opens on its running task, not on the clicked one',
  );

  send(win, {type: 'status', running: true, tabId, startTs: Date.now()});
  send(win, {
    type: 'task_events',
    tabId,
    chat_id: 'chat-9',
    task_id: '53',
    task: 'Task R',
    events: [{type: 'system_output', text: 'output of task R\n'}],
  });
  assert.deepStrictEqual(
    askedSince(posted, asked),
    ['53 prev'],
    'the transcript is there: the thread is read back from it',
  );
  answerPrev(win, posted, tabId, {id: '52', title: 'Task Q'});
  answerPrev(win, posted, tabId, {id: '51', title: 'Task P'});
  assert.deepStrictEqual(askedSince(posted, asked), ['53 prev', '52 prev']);
  assert.strictEqual(shownTask(win), 'Task P', 'the reader lands on it');
  assert.ok(isRunning(win, tabId), 'the tab shows the chat running');
  win.close();
  console.log('PASS history click opens a running chat, then reads back');
}

/**
 * The daemon's answer to a resume of `chatId` into `tabId`: the tab's
 * state, then the transcript (its `chat_id` binds the tab here, as the
 * registry snapshot before them does).
 */
function answerResume(win, tabId, chatId, taskId, title, running) {
  send(win, {type: 'status', running, tabId, attached: running});
  send(win, {
    type: 'task_events',
    tabId,
    chat_id: chatId,
    task_id: String(taskId),
    task: title,
    events: [{type: 'system_output', text: 'output of ' + title + '\n'}],
  });
}

// W1 audit 2026-11-12: a chat opened from the history is on its way
// until the daemon says whether its task runs.  Leaving it meanwhile
// closed its tab (it looked idle), and the close reached the daemon
// after the resume: a running task lost its tab on every surface.
function testLeavingAChatOnItsWayClosesNothing() {
  const {win, posted} = makePage();
  sendEmptyWorkDir(win);
  send(win, {
    type: 'history',
    offset: 0,
    generation: 0,
    sessions: [
      makeRow({id: 'chat-9', task_id: 53, title: 'Task R', is_running: true}),
      historySession('chat-1', 42, 'Task B'),
    ],
  });
  historyRow(win, 0).click();
  const running = win._testApi.getActiveTabId();
  historyRow(win, 1).click();
  const settled = win._testApi.getActiveTabId();
  assert.notStrictEqual(settled, running, 'the other chat has its own tab');
  assert.ok(
    chatTabs(win).some(t => t.id === running),
    'BUG - the chat still on its way was closed when the user left it',
  );
  assert.deepStrictEqual(
    ofType(posted, 'closeTab').filter(m => m.tabId === running),
    [],
    'the daemon is not told to close it',
  );

  answerResume(win, running, 'chat-9', 53, 'Task R', true);
  assert.ok(isRunning(win, running), 'the answer: its task runs');
  // A chat whose answer says nothing runs is retired when left, as ever.
  answerResume(win, settled, 'chat-1', 42, 'Task B', false);
  historyRow(win, 0).click();
  assert.strictEqual(win._testApi.getActiveTabId(), running);
  assert.deepStrictEqual(
    plain(chatTabs(win).map(t => t.id)),
    [running],
    'the settled chat left behind is retired',
  );
  win.close();
  console.log('PASS leaving a chat on its way closes nothing');
}

// W1 audit 2026-11-12: one chat, one tab.  A second click on a chat
// still on its way opened a second tab for it (no tab was bound to the
// chat yet) and resumed the chat a second time.
function testSecondClickOnAChatOnItsWayShowsItsTab() {
  const {win, posted} = makePage();
  sendEmptyWorkDir(win);
  const older = {task_id: 51, has_events: true, chat_running: true};
  send(win, {
    type: 'history',
    offset: 0,
    generation: 0,
    sessions: [
      makeRow({id: 'chat-9', title: 'Task P', ...older}),
      // The chat's copy on another machine is another chat.
      makeRow({id: 'chat-9', title: 'Task P on box', machine: 'box', ...older}),
    ],
  });
  historyRow(win, 1).click();
  const twin = win._testApi.getActiveTabId();
  historyRow(win, 0).click();
  const tabId = win._testApi.getActiveTabId();
  assert.notStrictEqual(tabId, twin, 'the copy on box is not this chat');
  const open = chatTabs(win).length;
  historyRow(win, 0).click();
  assert.strictEqual(
    win._testApi.getActiveTabId(),
    tabId,
    'BUG - the second click opened a second tab for the chat',
  );
  assert.strictEqual(chatTabs(win).length, open, 'one tab for the chat');

  // The answer: task 53 runs.  The first click's task is still sought.
  answerResume(win, tabId, 'chat-9', 53, 'Task R', true);
  assert.deepStrictEqual(
    askedSince(posted, 0),
    ['53 prev'],
    'the thread is read back to the task clicked first',
  );
  historyRow(win, 1).click();
  assert.strictEqual(win._testApi.getActiveTabId(), twin, 'still on its way');
  // box took the chat: the tab is on its stream and its task runs.  The
  // registry never names a machine's chat, so the chat is on its way
  // until its transcript is here.
  send(win, {type: 'tabMachine', tabId: twin, machine: 'box'});
  send(win, {type: 'status', running: true, tabId: twin, attached: true});
  historyRow(win, 0).click();
  historyRow(win, 1).click();
  assert.strictEqual(
    win._testApi.getActiveTabId(),
    twin,
    'BUG - a chat whose transcript is on its way got a second tab',
  );
  assert.strictEqual(chatTabs(win).length, open, 'one tab per chat');
  assert.deepStrictEqual(
    resumeMessages(posted).map(r => [r.chatId, r.tabId, r.taskId, r.machine]),
    [
      ['chat-9', twin, undefined, 'box'],
      ['chat-9', tabId, undefined, undefined],
    ],
    'each chat is resumed once, whatever the clicks',
  );

  // The stream to box is lost: nothing is on its way any more and the
  // tab is bound to no chat.  The next click opens the chat anew.
  send(win, {type: 'status', running: false, tabId: twin});
  send(win, {
    type: 'error',
    code: 'prompt_refused',
    text: 'no box',
    tabId: twin,
  });
  send(win, {type: 'tabMachine', tabId: twin, machine: ''});
  historyRow(win, 0).click();
  historyRow(win, 1).click();
  const retry = win._testApi.getActiveTabId();
  assert.deepStrictEqual(
    resumeMessages(posted)
      .slice(2)
      .map(r => [r.chatId, r.tabId, r.taskId, r.machine]),
    [['chat-9', retry, undefined, 'box']],
    'BUG - the refused chat is not asked for as a chat opened anew is',
  );
  assert.notStrictEqual(retry, twin, 'in a tab of its own');
  assert.ok(
    !chatTabs(win).some(t => t.id === twin),
    'the refused tab is retired',
  );
  win.close();
  console.log('PASS a second click on a chat on its way shows its tab');
}

// Review 2026-11-12: the last click wins.  A click on another task of
// a chat still on its way was dropped, and the first click's task
// landed.
function testClickOnAnotherTaskOfAChatOnItsWay() {
  const {win, posted} = makePage();
  sendEmptyWorkDir(win);
  send(win, {
    type: 'history',
    offset: 0,
    generation: 0,
    sessions: [
      makeRow({id: 'chat-9', task_id: 53, title: 'Task R', is_running: true}),
      makeRow({
        id: 'chat-9',
        task_id: 51,
        title: 'Task P',
        has_events: true,
        chat_running: true,
      }),
      historySession('chat-1', 42, 'Task B'),
      historySession('chat-1', 41, 'Task A'),
    ],
  });
  // A chat that does not run is asked for again, at the task clicked.
  historyRow(win, 3).click();
  const idle = win._testApi.getActiveTabId();
  historyRow(win, 2).click();
  historyRow(win, 2).click(); // the task already asked for: nothing more
  assert.strictEqual(win._testApi.getActiveTabId(), idle, 'one tab');
  assert.deepStrictEqual(
    resumeMessages(posted).map(r => [r.chatId, r.tabId, r.taskId]),
    [
      ['chat-1', idle, 41],
      ['chat-1', idle, 42],
    ],
    'BUG - the click on another task of the chat was dropped',
  );
  answerResume(win, idle, 'chat-1', 41, 'Task A', false);
  answerResume(win, idle, 'chat-1', 42, 'Task B', false);
  assert.strictEqual(shownTask(win), 'Task B', 'the reader lands on it');

  // A running chat is never resumed at an older task: the request for
  // its running task stands and the thread is read back.
  const asked = ofType(posted, 'getAdjacentTask').length;
  historyRow(win, 0).click();
  const running = win._testApi.getActiveTabId();
  historyRow(win, 1).click();
  assert.strictEqual(win._testApi.getActiveTabId(), running, 'one tab');
  assert.deepStrictEqual(
    resumeMessages(posted)
      .slice(2)
      .map(r => [r.chatId, r.tabId, r.taskId]),
    [['chat-9', running, 53]],
    'the running chat is asked for once, at its running task',
  );
  answerResume(win, running, 'chat-9', 53, 'Task R', true);
  assert.deepStrictEqual(
    askedSince(posted, asked),
    ['53 prev'],
    'the thread is read back to the task clicked last',
  );
  win.close();
  console.log('PASS a click on another task of a chat on its way wins');
}

// W1 audit 2026-11-12: a resume that died with the connection is never
// answered, so the chat it was bringing stayed on its way for good.
// The snapshot after the reconnect lists the tab bound to no chat: the
// daemon never saw the request, and it is asked again, once.
function testChatOnItsWayIsAskedAgainAfterAnOutage() {
  const {win, posted} = makePage();
  sendEmptyWorkDir(win);
  send(win, {type: 'daemonStatus', connected: true});
  clickTask(win, 42, 'Task B');
  const tabId = win._testApi.getActiveTabId();
  const unbound = [{tabId, title: 'Task B'}];
  // The snapshot answering the tab's own opening: the resume is behind it.
  sendTabsState(win, unbound);
  assert.strictEqual(resumeMessages(posted).length, 1, 'nothing is lost yet');

  send(win, {type: 'daemonStatus', connected: false});
  send(win, {type: 'daemonStatus', connected: true});
  sendTabsState(win, unbound);
  const resumes = resumeMessages(posted);
  assert.strictEqual(
    resumes.length,
    2,
    'BUG - the request lost with the connection was not asked again',
  );
  assert.deepStrictEqual(plain(resumes[1]), plain(resumes[0]));
  sendTabsState(win, unbound);
  assert.strictEqual(resumeMessages(posted).length, 2, 'asked again once');

  // Answered: after the next outage the daemon replays the bound tab.
  answerResume(win, tabId, 'chat-1', 42, 'Task B', false);
  send(win, {type: 'daemonStatus', connected: false});
  send(win, {type: 'daemonStatus', connected: true});
  sendTabsState(win, [{tabId, title: 'Task B', chatId: 'chat-1'}]);
  assert.strictEqual(resumeMessages(posted).length, 2, 'nothing on its way');
  win.close();
  console.log('PASS a chat on its way is asked again after an outage');
}

testClickScrollsToSplicedNeighbour();
testClickScrollsBackToOwnTask();
testClickLoadsTaskNotInTranscript();
testClickRowWithoutTaskIdJustSwitches();
testClickOwnTaskWithoutRegionReclaimsPanel();
testClickOnOlderTaskOfRunningChatReadsBackToIt();
testLaterClickEndsTheReadBack();
testClickOnOlderTaskOfRunningChatFetchesItsTranscriptFirst();
testClickOnOlderTaskOfRunningChatWithoutTab();
testLeavingAChatOnItsWayClosesNothing();
testSecondClickOnAChatOnItsWayShowsItsTab();
testClickOnAnotherTaskOfAChatOnItsWay();
testChatOnItsWayIsAskedAgainAfterAnOutage();
console.log('All historyClickScrollToTask tests passed');
