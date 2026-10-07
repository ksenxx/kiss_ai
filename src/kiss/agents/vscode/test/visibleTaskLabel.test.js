// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// The active tab's label names the task the reader is looking at: the
// tab's own task, or the neighbouring task scrolled into view. Each task
// of the thread opens with its own task panel in the transcript, so
// the label and the status row are what follow the reader. These
// end-to-end tests drive the real webview (chat.html + media/main.js) in
// JSDOM, stub a deterministic layout for #output, scroll it like a user
// would, and assert the label plus the token/budget/step metrics.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

const VIEWPORT = 500;
const DEFAULT_HEIGHT = 400;

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
  win.eval(
    fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8') +
      '\n//# sourceURL=visibletask-main.js',
  );
  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function heightOf(el) {
  const h = el.dataset ? el.dataset.testHeight : '';
  if (h) return Number(h);
  // The "Loading … task" strip is one thin line of chrome.
  return el.id === 'adjacent-loader' ? 0 : DEFAULT_HEIGHT;
}

function rect(top, height) {
  return {
    top,
    bottom: top + height,
    height,
    left: 0,
    right: 400,
    width: 400,
  };
}

// #output is a VIEWPORT-tall scroller whose children stack by their
// data-test-height (DEFAULT_HEIGHT when unset), exactly like the real
// column layout the webview renders.
function installLayout(win, O) {
  const origGBCR = win.Element.prototype.getBoundingClientRect;
  win.Element.prototype.getBoundingClientRect = function () {
    if (this === O) return rect(0, VIEWPORT);
    if (this.parentNode === O) {
      let top = -O.scrollTop;
      for (let i = 0; i < O.children.length; i++) {
        const child = O.children[i];
        const h = heightOf(child);
        if (child === this) return rect(top, h);
        top += h;
      }
    }
    return origGBCR.call(this);
  };
  Object.defineProperty(O, 'scrollHeight', {
    get: () => {
      let total = 0;
      for (let i = 0; i < O.children.length; i++)
        total += heightOf(O.children[i]);
      return total;
    },
    configurable: true,
  });
  Object.defineProperty(O, 'clientHeight', {
    value: VIEWPORT,
    configurable: true,
  });
  Object.defineProperty(O, 'clientWidth', {value: 400, configurable: true});
}

function scrollTo(win, O, top) {
  const max = Math.max(0, O.scrollHeight - VIEWPORT);
  O.scrollTop = Math.max(0, Math.min(max, top));
  O.dispatchEvent(new win.Event('scroll'));
}

function scrollToBottom(win, O) {
  scrollTo(win, O, O.scrollHeight);
}

function wheel(win, el, deltaY) {
  return el.dispatchEvent(
    new win.WheelEvent('wheel', {deltaY, bubbles: true, cancelable: true}),
  );
}

function clickTab(win, tabId) {
  const el = win.document.querySelector(
    '.chat-tab[data-tab-id=' + JSON.stringify(tabId) + ']',
  );
  assert.ok(el, 'tab ' + tabId + ' must be in the tab bar');
  el.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

/** The active tab's label: the task the reader is looking at. */
function shownTask(win) {
  const label = win.document.querySelector(
    '.chat-tab[aria-selected="true"] .chat-tab-label',
  );
  assert.ok(label, 'the active tab must be in the tab bar');
  return label.textContent;
}

/** The task panel that opens the transcript region of *name*. */
function taskPanelOf(O, name) {
  const region = name === 'Main task' ? O : taskEl(O, name);
  const panel = region.querySelector(':scope > .task-panel');
  assert.ok(panel, 'the region of ' + name + ' must open with a task panel');
  return panel;
}

function metrics(win) {
  const d = win.document;
  return {
    tokens: d.getElementById('status-tokens').textContent,
    budget: d.getElementById('status-budget').textContent,
    steps: d.getElementById('status-steps').textContent,
  };
}

function taskEvents(name) {
  return [
    {type: 'task_start', task: name},
    {type: 'system_output', text: name + ' line one\n'},
  ];
}

// Builds a tab holding one main task, then splices in the neighbouring
// tasks named by `prev` / `next` (each an array of task names, nearest
// neighbour first) the way the backend replies to an overscroll.
function setup(opts) {
  const options = opts || {};
  const {win, posted} = makeWebview();
  const tabId = posted.find(m => m.type === 'ready').tabId;
  win._testApi.hideWelcome();
  const O = win.document.getElementById('output');
  installLayout(win, O);
  send(win, {
    type: 'task_events',
    tabId,
    chat_id: 'chat-abc',
    task_id: '50',
    task: 'Main task',
    events: [
      {type: 'task_start', task: 'Main task'},
      {type: 'usage_info', total_tokens: 999, cost: '0.50', total_steps: 3},
      {type: 'system_output', text: 'main line one\n'},
    ],
  });
  (options.prev || []).forEach((name, i) => {
    send(win, {
      type: 'adjacent_task_events',
      tabId,
      direction: 'prev',
      task: name,
      task_id: String(49 - i),
      events: taskEvents(name),
    });
  });
  (options.next || []).forEach((name, i) => {
    send(win, {
      type: 'adjacent_task_events',
      tabId,
      direction: 'next',
      task: name,
      task_id: String(51 + i),
      events: taskEvents(name),
    });
  });
  return {win, posted, tabId, O};
}

function taskEl(O, name) {
  const els = O.querySelectorAll('.adjacent-task[data-task]');
  for (let i = 0; i < els.length; i++)
    if (els[i].dataset.task === name) return els[i];
  return null;
}

function setHeight(el, h) {
  el.dataset.testHeight = String(h);
}

function mainChildren(O) {
  return Array.prototype.filter.call(
    O.children,
    el =>
      !el.classList.contains('adjacent-task') && el.id !== 'adjacent-loader',
  );
}

function topOf(O, el) {
  let top = 0;
  for (let i = 0; i < O.children.length; i++) {
    if (O.children[i] === el) return top;
    top += heightOf(O.children[i]);
  }
  return -1;
}

// A task whose events fill the screen must be the one the panel names,
// whichever direction the reader came from.
function testPanelNamesTheTaskFillingTheScreen() {
  const {win, O} = setup({prev: ['Prev task'], next: ['Next task']});
  const prev = taskEl(O, 'Prev task');
  const next = taskEl(O, 'Next task');
  setHeight(prev, 1000);
  setHeight(next, 1000);

  scrollTo(win, O, topOf(O, prev));
  assert.strictEqual(shownTask(win), 'Prev task', 'top of the previous task');

  scrollTo(win, O, topOf(O, mainChildren(O)[0]));
  assert.strictEqual(shownTask(win), 'Main task', 'back on the main task');

  scrollTo(win, O, topOf(O, next));
  assert.strictEqual(shownTask(win), 'Next task', 'top of the next task');

  win.close();
  console.log('PASS the panel names the task that fills the screen');
}

// The bottom-most task can never be scrolled to the top of the viewport:
// the scroller clamps.  The panel must still name it once its events own
// most of the screen.
function testLastTaskNamedWhenScrolledToBottom() {
  const {win, O} = setup({next: ['Last task']});
  const last = taskEl(O, 'Last task');
  setHeight(last, 300);

  scrollToBottom(win, O);
  assert.strictEqual(
    shownTask(win),
    'Last task',
    'the panel must name the last task once it owns most of the viewport',
  );
  win.close();
  console.log('PASS the last task is named when the scroller hits the bottom');
}

// The mirror image: a sliver of the last task at the bottom edge must NOT
// steal the panel from the task the reader is actually reading.
function testSliverOfLastTaskDoesNotStealThePanel() {
  const {win, O} = setup({next: ['Last task']});
  const last = taskEl(O, 'Last task');
  setHeight(last, 60);

  scrollToBottom(win, O);
  assert.strictEqual(
    shownTask(win),
    'Main task',
    'a 60px sliver must not rename the panel',
  );
  win.close();
  console.log('PASS a sliver of the last task does not steal the panel');
}

// The same clamp exists at the top: a short first task cannot be pushed
// down to the 30% probe line either.
function testFirstTaskNamedWhenScrolledToTop() {
  const {win, O} = setup({prev: ['First task']});
  const first = taskEl(O, 'First task');
  setHeight(first, 300);

  scrollTo(win, O, 0);
  assert.strictEqual(
    shownTask(win),
    'First task',
    'the panel must name the first task at the very top',
  );
  win.close();
  console.log('PASS the first task is named when the scroller hits the top');
}

// Metrics belong to the task on screen, and must snap back to the live
// task's own numbers when the reader returns to it.
function testMetricsFollowTheVisibleTask() {
  const {win, O} = setup({next: ['Last task']});
  const last = taskEl(O, 'Last task');
  last.dataset.metricTokens = 'Tokens: 777';
  last.dataset.metricBudget = 'Budget: $7';
  last.dataset.metricSteps = 'Steps: 7';
  setHeight(last, 300);

  scrollToBottom(win, O);
  assert.strictEqual(shownTask(win), 'Last task');
  assert.deepStrictEqual(metrics(win), {
    tokens: 'Tokens: 777',
    budget: 'Budget: $7',
    steps: 'Steps: 7',
  });

  scrollTo(win, O, 0);
  assert.strictEqual(shownTask(win), 'Main task');
  assert.deepStrictEqual(metrics(win), {
    tokens: 'Tokens: 999',
    budget: 'Cost: $0.50',
    steps: 'Steps: 3',
  });
  win.close();
  console.log('PASS the status metrics follow the visible task');
}

// Every region of the thread opens with its own task panel: the tab's
// own transcript and each spliced-in neighbour, in reading order.
function testEveryRegionOpensWithItsTaskPanel() {
  const {win, O} = setup({prev: ['Prev task'], next: ['Next task']});
  const prevPanel = taskPanelOf(O, 'Prev task');
  const mainPanel = taskPanelOf(O, 'Main task');
  const nextPanel = taskPanelOf(O, 'Next task');
  assert.strictEqual(
    prevPanel.querySelector('.task-panel-text').textContent,
    'Prev task',
  );
  assert.strictEqual(
    mainPanel.querySelector('.task-panel-text').textContent,
    'Main task',
  );
  assert.strictEqual(
    nextPanel.querySelector('.task-panel-text').textContent,
    'Next task',
  );
  const order = Array.from(O.querySelectorAll('.task-panel'));
  assert.deepStrictEqual(order, [prevPanel, mainPanel, nextPanel]);
  assert.strictEqual(
    taskEl(O, 'Prev task').firstElementChild,
    prevPanel,
    "a neighbour's task panel is the first thing in its container",
  );
  assert.strictEqual(
    mainChildren(O)[0],
    mainPanel,
    "the tab's own task panel is the first thing in its region",
  );
  order.forEach(p =>
    assert.ok(
      !p.classList.contains('collapsed'),
      'the replay leaves every task panel open',
    ),
  );
  assert.strictEqual(
    win.document.getElementById('task-panel'),
    null,
    'no fixed task panel sits above the transcript',
  );
  win.close();
  console.log('PASS every region of the thread opens with its task panel');
}

// The thread reads on without a stop: scrolling near the top fetches
// the previous task before the reader reaches it, near the bottom the
// next one; a reply without a task closes that end for good.
function testScrollingNearAnEdgePrefetchesTheNeighbour() {
  const {win, O, posted, tabId} = setup({});
  const requests = () => posted.filter(m => m.type === 'getAdjacentTask');
  mainChildren(O).forEach(el => setHeight(el, 1000));
  scrollTo(win, O, O.scrollHeight - VIEWPORT - 100);
  assert.deepStrictEqual(
    requests().map(m => [m.direction, m.taskId]),
    [['next', '50']],
    'near the bottom the next task is fetched',
  );
  assert.strictEqual(
    O.lastElementChild.id,
    'adjacent-loader',
    'the loading strip joins the bottom of the transcript',
  );
  send(win, {type: 'adjacent_task_events', tabId, direction: 'next'});
  assert.strictEqual(
    win.document.getElementById('adjacent-loader'),
    null,
    'a reply without a task removes the strip',
  );
  scrollTo(win, O, O.scrollHeight - VIEWPORT - 50);
  assert.strictEqual(requests().length, 1, 'no next task: nothing more');

  scrollTo(win, O, 200);
  assert.deepStrictEqual(
    requests().slice(1).map(m => [m.direction, m.taskId]),
    [['prev', '50']],
    'near the top the previous task is fetched',
  );
  assert.strictEqual(O.firstElementChild.id, 'adjacent-loader');
  scrollTo(win, O, 100);
  assert.strictEqual(requests().length, 2, 'one fetch at a time');
  send(win, {
    type: 'adjacent_task_events',
    tabId,
    direction: 'prev',
    task: 'Prev task',
    task_id: '49',
    events: taskEvents('Prev task'),
  });
  assert.strictEqual(
    O.firstElementChild,
    taskEl(O, 'Prev task'),
    'the previous task is spliced in above',
  );
  assert.strictEqual(
    O.scrollTop,
    100 + heightOf(taskEl(O, 'Prev task')),
    'the reader keeps their place while the thread grows above them',
  );
  scrollTo(win, O, 0);
  assert.deepStrictEqual(
    requests().slice(2).map(m => [m.direction, m.taskId]),
    [['prev', '49']],
    'the next fetch continues from the oldest task on screen',
  );
  win.close();
  console.log('PASS scrolling near an edge prefetches the neighbour');
}

// A transcript too short to scroll never fires a scroll event: pushing
// the wheel past an edge fetches the task beyond it at once, with no
// gesture to accumulate.
function testEdgePushFetchesAtOnce() {
  const {win, O, posted, tabId} = setup({});
  const requests = () => posted.filter(m => m.type === 'getAdjacentTask');
  mainChildren(O).forEach(el => setHeight(el, 50));
  assert.ok(O.scrollHeight <= VIEWPORT, 'setup: nothing to scroll');
  // A real scroller clamps an unscrollable transcript at 0.
  O.scrollTop = 0;
  wheel(win, O, 1);
  assert.deepStrictEqual(
    requests().map(m => m.direction),
    ['next'],
    'one push past the bottom fetches the next task at once',
  );
  wheel(win, O, 1);
  assert.strictEqual(requests().length, 1, 'one fetch at a time');
  send(win, {type: 'adjacent_task_events', tabId, direction: 'next'});
  wheel(win, O, 1);
  assert.strictEqual(requests().length, 1, 'no next task: nothing more');
  wheel(win, O, -1);
  assert.deepStrictEqual(
    requests().map(m => m.direction),
    ['next', 'prev'],
    'one push past the top fetches the previous task at once',
  );
  win.close();
  console.log('PASS an edge push fetches the neighbour at once');
}

// A running task is the newest of its chat: nothing is fetched below it,
// whether the reader scrolls towards the bottom or pushes past it. The
// tasks above it still join the thread.
function testRunningTaskHasNoNextTask() {
  const {win, O, posted, tabId} = setup({});
  const requests = () => posted.filter(m => m.type === 'getAdjacentTask');
  send(win, {type: 'status', running: true, tabId, startTs: 1});
  mainChildren(O).forEach(el => setHeight(el, 1000));
  scrollToBottom(win, O);
  wheel(win, O, 1);
  assert.strictEqual(requests().length, 0, 'no next task below a running task');
  scrollTo(win, O, 0);
  wheel(win, O, -1);
  assert.deepStrictEqual(
    requests().map(m => m.direction),
    ['prev'],
    'the previous task still joins the thread above a running task',
  );
  win.close();
  console.log('PASS a running task has no next task');
}

// A history click lands the reader ON the clicked task: its region is
// scrolled to the top of the viewport and pinned there, so the label
// names it even when the region is too short to own most of the screen;
// a scroll event that moves nothing keeps the pin, a real scroll ends it.
function testHistoryClickPinsTheClickedTask() {
  const {win, O, posted} = setup({next: ['Last task']});
  const last = taskEl(O, 'Last task');
  setHeight(last, 300);
  scrollTo(win, O, 0);
  assert.strictEqual(shownTask(win), 'Main task');

  send(win, {
    type: 'openChatFromHistory',
    chatId: 'chat-abc',
    taskId: '51',
    title: 'Last task',
  });
  assert.strictEqual(
    shownTask(win),
    'Last task',
    'the clicked task is what the label names',
  );
  assert.strictEqual(
    last.getBoundingClientRect().top,
    O.getBoundingClientRect().top,
    'the clicked region sits at the top of the viewport',
  );
  assert.ok(
    !posted.some(m => m.type === 'resumeSession'),
    'a task already on screen is not replayed',
  );
  O.dispatchEvent(new win.Event('scroll'));
  assert.strictEqual(
    shownTask(win),
    'Last task',
    'a scroll event that does not move the scroller keeps the pin',
  );
  scrollTo(win, O, 0);
  assert.strictEqual(shownTask(win), 'Main task', 'a real scroll ends it');
  win.close();
  console.log('PASS a history click pins the clicked task');
}

// A history row whose chat is not open gets a fresh tab that shows the
// task panel at once; the daemon answers the tab's `newChat` with a
// `showWelcome` reset, which must not blank that panel (neither while
// the tab is active nor after the reader has moved to another tab).
function testHistoryTabSurvivesTheWelcomeReset() {
  const {win, O, posted, tabId} = setup({});
  send(win, {
    type: 'openChatFromHistory',
    chatId: 'chat-other',
    taskId: '70',
    title: 'Other task',
  });
  const newTabId = posted.filter(m => m.type === 'newChat').pop().tabId;
  assert.notStrictEqual(newTabId, tabId, 'the row opens a fresh tab');
  assert.strictEqual(shownTask(win), 'Other task');
  assert.strictEqual(
    taskPanelOf(O, 'Main task').querySelector('.task-panel-text').textContent,
    'Other task',
    'the fresh tab opens with the clicked task panel',
  );
  assert.ok(
    posted.some(m => m.type === 'resumeSession' && m.tabId === newTabId),
    'the chat is resumed into the fresh tab',
  );
  send(win, {type: 'showWelcome', tabId: newTabId, model: 'm'});
  assert.strictEqual(
    taskPanelOf(O, 'Main task').querySelector('.task-panel-text').textContent,
    'Other task',
    'the welcome reset of the active history tab keeps its panel',
  );
  assert.strictEqual(
    win.document.getElementById('welcome').style.display,
    'none',
    'the welcome screen stays hidden behind the task panel',
  );
  clickTab(win, tabId);
  send(win, {type: 'showWelcome', tabId: newTabId, model: 'm'});
  clickTab(win, newTabId);
  assert.strictEqual(
    taskPanelOf(O, 'Main task').querySelector('.task-panel-text').textContent,
    'Other task',
    'the welcome reset of a background history tab keeps its panel',
  );
  // A genuinely new tab is still reset to the welcome screen.
  send(win, {type: 'clearChat'});
  const blankTabId = posted.filter(m => m.type === 'newChat').pop().tabId;
  send(win, {type: 'showWelcome', tabId: blankTabId, model: 'm'});
  assert.strictEqual(
    win.document.getElementById('welcome').style.display,
    '',
    'a blank tab shows the welcome screen',
  );
  assert.ok(!O.querySelector('.task-panel'), 'a blank tab has no task panel');
  win.close();
  console.log('PASS a history tab survives the daemon welcome reset');
}

// The daemon echoes every submitted prompt as setTaskText, queued
// follow-ups and refused submits included. Until a run claims the echo
// with 'clear', the label and the task panel keep naming the task on
// screen — across a tab round trip too.
function testFollowupEchoDoesNotRenameTheTaskOnScreen() {
  const {win, O, tabId} = setup({});
  const panelText = () =>
    taskPanelOf(O, 'Main task').querySelector('.task-panel-text').textContent;
  send(win, {type: 'setTaskText', tabId, text: 'Queued follow-up'});
  assert.strictEqual(shownTask(win), 'Main task', 'the label keeps the task');
  assert.strictEqual(panelText(), 'Main task', 'the panel keeps the task');
  assert.strictEqual(
    O.querySelectorAll('.task-panel').length,
    1,
    'the echo puts up no second panel',
  );
  win._testApi.createNewTab();
  const otherTabId = win._testApi.getActiveTabId();
  assert.notStrictEqual(otherTabId, tabId);
  clickTab(win, tabId);
  assert.strictEqual(shownTask(win), 'Main task', 'a round trip keeps it');
  assert.strictEqual(panelText(), 'Main task');
  // A background tab's echo is held the same way.
  send(win, {type: 'setTaskText', tabId: otherTabId, text: 'Other task'});
  send(win, {type: 'clear', tabId: otherTabId});
  clickTab(win, otherTabId);
  assert.strictEqual(shownTask(win), 'Other task');
  assert.strictEqual(
    O.querySelector('.task-panel-text').textContent,
    'Other task',
    "a background tab's clear opens its transcript with the echoed text",
  );
  // The run that the echo announced opens its transcript with its text.
  clickTab(win, tabId);
  send(win, {type: 'clear', tabId});
  assert.strictEqual(shownTask(win), 'Queued follow-up');
  assert.strictEqual(panelText(), 'Queued follow-up');
  assert.strictEqual(O.querySelectorAll('.task-panel').length, 1);
  win.close();
  console.log('PASS a follow-up echo does not rename the task on screen');
}

// Walking the whole transcript one scroll step at a time: the panel must
// always name a task that actually owns pixels on the screen.
function testPanelAlwaysNamesAnOnScreenTask() {
  const {win, O} = setup({
    prev: ['Prev task'],
    next: ['Next task', 'Last task'],
  });
  setHeight(taskEl(O, 'Prev task'), 900);
  setHeight(taskEl(O, 'Next task'), 700);
  setHeight(taskEl(O, 'Last task'), 300);

  const max = O.scrollHeight - VIEWPORT;
  for (let top = 0; top <= max; top += 50) {
    scrollTo(win, O, top);
    const name = shownTask(win);
    let onScreen = false;
    for (let i = 0; i < O.children.length; i++) {
      const child = O.children[i];
      const childTop = topOf(O, child) - O.scrollTop;
      const childBottom = childTop + heightOf(child);
      if (childBottom <= 0 || childTop >= VIEWPORT) continue;
      const task = child.classList.contains('adjacent-task')
        ? child.dataset.task
        : 'Main task';
      if (task === name) onScreen = true;
    }
    assert.ok(
      onScreen,
      'at scrollTop=' +
        top +
        ' the panel named "' +
        name +
        '", which is off screen',
    );
  }
  win.close();
  console.log('PASS every scroll position names an on-screen task');
}

// Overscrolling past the end appends a "Loading next task…" strip. That
// strip is chrome, not transcript, and must not shadow the task under it.
function testLoaderIsNotTranscript() {
  const {win, O, posted} = setup({next: ['Last task']});
  setHeight(taskEl(O, 'Last task'), 300);

  scrollToBottom(win, O);
  assert.strictEqual(shownTask(win), 'Last task');
  for (let i = 0; i < 5; i++) wheel(win, O, 50);
  assert.ok(
    posted.some(m => m.type === 'getAdjacentTask' && m.direction === 'next'),
    'the overscroll must ask the backend for the following task',
  );
  const loader = win.document.getElementById('adjacent-loader');
  assert.ok(loader && loader.parentNode === O, 'the loader joined #output');
  setHeight(loader, 40);

  scrollToBottom(win, O);
  assert.strictEqual(
    shownTask(win),
    'Last task',
    'the loading strip must not shadow the task it is loading past',
  );
  win.close();
  console.log('PASS the adjacent-task loading strip is not transcript');
}

// A transcript shorter than the viewport cannot be scrolled, so splicing
// a neighbour in is the only thing that changes what is on screen. The
// panel has to follow that change on its own.
function testSplicedInTaskRenamesThePanelWithoutAScroll() {
  const {win, O, tabId} = setup({});
  mainChildren(O).forEach(el => setHeight(el, 50));
  assert.ok(
    O.scrollHeight <= VIEWPORT,
    'setup: the transcript must not be scrollable',
  );
  send(win, {
    type: 'adjacent_task_events',
    tabId,
    direction: 'next',
    task: 'Last task',
    task_id: '51',
    events: taskEvents('Last task'),
  });
  setHeight(taskEl(O, 'Last task'), 400);
  assert.strictEqual(
    shownTask(win),
    'Last task',
    'the freshly spliced-in task owns the screen and must be named',
  );
  win.close();
  console.log('PASS a spliced-in task renames the panel without a scroll');
}

// Reading a neighbour is a viewing position, not a change of identity.
// Leaving the tab and coming back must not rename the tab's own task.
function testTabRoundTripKeepsTheTabsOwnTask() {
  const {win, O} = setup({next: ['Last task']});
  setHeight(taskEl(O, 'Last task'), 600);
  const first = win._testApi.getActiveTabId();

  scrollToBottom(win, O);
  assert.strictEqual(shownTask(win), 'Last task', 'reading the neighbour');

  win._testApi.createNewTab();
  clickTab(win, first);
  installLayout(win, win.document.getElementById('output'));
  const back = win.document.getElementById('output');
  setHeight(taskEl(back, 'Last task'), 600);

  scrollTo(win, back, 0);
  assert.strictEqual(
    shownTask(win),
    'Main task',
    'back on its own events, the tab must name its own task',
  );
  assert.deepStrictEqual(metrics(win), {
    tokens: 'Tokens: 999',
    budget: 'Cost: $0.50',
    steps: 'Steps: 3',
  });
  win.close();
  console.log("PASS a tab round trip keeps the tab's own task");
}

// Coming back to a tab restores the transcript where the reader left it.
// The panel must name what is on screen then, not what the tab is called.
function testReturningToATabNamesWhatIsOnScreen() {
  const {win, O} = setup({next: ['Last task']});
  const last = taskEl(O, 'Last task');
  last.dataset.metricTokens = 'Tokens: 777';
  last.dataset.metricBudget = 'Cost: 7.00';
  last.dataset.metricSteps = 'Steps: 7';
  setHeight(last, 600);
  const first = win._testApi.getActiveTabId();

  scrollToBottom(win, O);
  assert.strictEqual(shownTask(win), 'Last task', 'reading the neighbour');

  win._testApi.createNewTab();
  clickTab(win, first);

  assert.strictEqual(
    shownTask(win),
    'Last task',
    'the reader came back to the neighbour, so the panel must name it',
  );
  assert.deepStrictEqual(metrics(win), {
    tokens: 'Tokens: 777',
    budget: 'Cost: 7.00',
    steps: 'Steps: 7',
  });
  win.close();
  console.log('PASS returning to a tab names what is on screen');
}

// The status metrics are part of the same story: they must describe the
// task on screen, and they must never be inherited from another tab.
function testMetricsDoNotLeakBetweenTabs() {
  const {win, O, posted} = setup({next: ['Last task']});
  const first = win._testApi.getActiveTabId();
  setHeight(taskEl(O, 'Last task'), 600);

  win._testApi.createNewTab();
  const second = posted[posted.length - 1] && win._testApi.getActiveTabId();
  send(win, {
    type: 'task_events',
    tabId: second,
    chat_id: 'chat-two',
    task_id: '90',
    task: 'Other task',
    events: [
      {type: 'task_start', task: 'Other task'},
      {type: 'usage_info', total_tokens: 222, cost: '2.22', total_steps: 22},
    ],
  });
  clickTab(win, first);
  const back = win.document.getElementById('output');
  installLayout(win, back);
  setHeight(taskEl(back, 'Last task'), 600);

  scrollTo(win, back, 0);
  assert.strictEqual(shownTask(win), 'Main task');
  assert.deepStrictEqual(
    metrics(win),
    {tokens: 'Tokens: 999', budget: 'Cost: $0.50', steps: 'Steps: 3'},
    "the tab must show its own metrics, not the other tab's",
  );
  win.close();
  console.log('PASS metrics do not leak between tabs');
}

// The live task keeps streaming while the reader studies a neighbour.
// Its numbers belong to its own events, not to the ones on screen.
function testLiveMetricsDoNotOverrideTheVisibleTask() {
  const {win, O} = setup({next: ['Last task']});
  const last = taskEl(O, 'Last task');
  last.dataset.metricTokens = 'Tokens: 777';
  last.dataset.metricBudget = 'Cost: 7.00';
  last.dataset.metricSteps = 'Steps: 7';
  setHeight(last, 600);

  scrollToBottom(win, O);
  assert.strictEqual(shownTask(win), 'Last task');

  win._testApi.processEvent({
    type: 'usage_info',
    total_tokens: 1234,
    cost: '1.20',
    total_steps: 12,
  });
  assert.strictEqual(shownTask(win), 'Last task', 'the panel must not flip');
  assert.deepStrictEqual(
    metrics(win),
    {tokens: 'Tokens: 777', budget: 'Cost: 7.00', steps: 'Steps: 7'},
    'the status row must keep describing the task on screen',
  );

  scrollTo(win, O, 0);
  assert.strictEqual(shownTask(win), 'Main task');
  assert.deepStrictEqual(
    metrics(win),
    {tokens: 'Tokens: 1.23K', budget: 'Cost: $1.20', steps: 'Steps: 12'},
    'the live task shows its fresh numbers once it is back on screen',
  );
  win.close();
  console.log('PASS live metrics do not override the visible task');
}

// A live event that carries only some of the numbers must not adopt the
// missing ones from the neighbour that happens to be on screen.
function testPartialLiveMetricsKeepTheLiveTasksOwnNumbers() {
  const {win, O} = setup({next: ['Last task']});
  const last = taskEl(O, 'Last task');
  last.dataset.metricTokens = 'Tokens: 777';
  last.dataset.metricBudget = 'Cost: 7.00';
  last.dataset.metricSteps = 'Steps: 7';
  setHeight(last, 600);

  scrollToBottom(win, O);
  assert.strictEqual(shownTask(win), 'Last task');
  win._testApi.processEvent({
    type: 'usage_info',
    total_tokens: 1234,
    cost: '1.20',
  });

  scrollTo(win, O, 0);
  assert.deepStrictEqual(
    metrics(win),
    {tokens: 'Tokens: 1.23K', budget: 'Cost: $1.20', steps: 'Steps: 3'},
    "an event without a step count must not adopt the neighbour's",
  );
  win.close();
  console.log('PASS partial live metrics keep the live task’s own numbers');
}

// Step counting happens on ordinary streaming events, which carry no
// metrics of their own. They must not repaint the neighbour's row either.
function testLiveStepCountDoesNotRepaintTheNeighboursRow() {
  const {win, O} = setup({next: ['Last task']});
  const last = taskEl(O, 'Last task');
  last.dataset.metricTokens = 'Tokens: 777';
  last.dataset.metricBudget = 'Cost: 7.00';
  last.dataset.metricSteps = 'Steps: 7';
  setHeight(last, 1200);

  // The reader is well above the bottom, so the live panel the step
  // appends below the neighbour does not pull the viewport along.
  scrollTo(win, O, topOf(O, last));
  win._testApi.processEvent({type: 'thinking_start'});
  assert.deepStrictEqual(
    metrics(win),
    {tokens: 'Tokens: 777', budget: 'Cost: 7.00', steps: 'Steps: 7'},
    'a live step must not show up in the neighbour’s status row',
  );

  scrollTo(win, O, 0);
  assert.notStrictEqual(
    metrics(win).steps,
    'Steps: 7',
    'the live task must show its own step count once it is back on screen',
  );
  win.close();
  console.log('PASS a live step does not repaint the neighbour’s row');
}

// Replaying a neighbour's transcript renders that task's steps. It must
// not move the live task's own counter along with it.
function testAdjacentReplayLeavesTheLiveStepCountAlone() {
  const {win, O, tabId} = setup({});
  win._testApi.processEvent({type: 'thinking_start'});
  assert.strictEqual(metrics(win).steps, 'Steps: 1', 'setup: one live step');

  send(win, {
    type: 'adjacent_task_events',
    tabId,
    direction: 'prev',
    task: 'Older task',
    task_id: '49',
    events: [
      {type: 'task_start', task: 'Older task'},
      {type: 'result', success: true, step_count: 77},
    ],
  });
  setHeight(taskEl(O, 'Older task'), 100);
  scrollTo(win, O, O.scrollHeight);
  assert.strictEqual(
    metrics(win).steps,
    'Steps: 1',
    "the neighbour's replayed step count must not show as the live one",
  );

  win._testApi.processEvent({type: 'tool_call', name: 'Bash'});
  win._testApi.processEvent({type: 'thinking_start'});
  assert.strictEqual(
    metrics(win).steps,
    'Steps: 2',
    'the live counter must carry on from its own value',
  );
  win.close();
  console.log('PASS an adjacent replay leaves the live step count alone');
}

// Another tab streaming in the background renders through the very same
// machinery. It must not touch what the reader is looking at.
function testHiddenTabStreamLeavesTheVisibleRowAlone() {
  const {win, O} = setup({next: ['Last task']});
  const last = taskEl(O, 'Last task');
  last.dataset.metricTokens = 'Tokens: 777';
  last.dataset.metricBudget = 'Cost: 7.00';
  last.dataset.metricSteps = 'Steps: 7';
  setHeight(last, 600);
  const first = win._testApi.getActiveTabId();
  win._testApi.createNewTab();
  const hidden = win._testApi.getActiveTabId();
  clickTab(win, first);
  installLayout(win, win.document.getElementById('output'));
  const back = win.document.getElementById('output');
  setHeight(taskEl(back, 'Last task'), 600);
  scrollToBottom(win, back);
  assert.strictEqual(shownTask(win), 'Last task', 'reading the neighbour');

  send(win, {
    type: 'system_output',
    tabId: hidden,
    text: 'the hidden tab talks\n',
  });
  send(win, {
    type: 'usage_info',
    tabId: hidden,
    total_tokens: 55555,
    cost: '9.99',
    total_steps: 99,
  });
  assert.deepStrictEqual(
    metrics(win),
    {tokens: 'Tokens: 777', budget: 'Cost: 7.00', steps: 'Steps: 7'},
    'a hidden tab must not write on the visible status row',
  );

  scrollTo(win, back, 0);
  assert.strictEqual(shownTask(win), 'Main task');
  assert.deepStrictEqual(
    metrics(win),
    {tokens: 'Tokens: 999', budget: 'Cost: $0.50', steps: 'Steps: 3'},
    'the visible tab keeps its own numbers while another tab streams',
  );
  win.close();
  console.log('PASS a hidden tab’s stream leaves the visible row alone');
}

// A whole transcript replayed into a hidden tab is the same story at a
// larger scale.
function testHiddenTabReplayLeavesTheVisibleRowAlone() {
  const {win, O} = setup({next: ['Last task']});
  const last = taskEl(O, 'Last task');
  last.dataset.metricTokens = 'Tokens: 777';
  last.dataset.metricBudget = 'Cost: 7.00';
  last.dataset.metricSteps = 'Steps: 7';
  setHeight(last, 600);
  const first = win._testApi.getActiveTabId();
  win._testApi.createNewTab();
  const hidden = win._testApi.getActiveTabId();
  clickTab(win, first);
  installLayout(win, win.document.getElementById('output'));
  const back = win.document.getElementById('output');
  setHeight(taskEl(back, 'Last task'), 600);
  scrollToBottom(win, back);

  send(win, {
    type: 'task_events',
    tabId: hidden,
    chat_id: 'chat-hidden',
    task_id: '80',
    task: 'Hidden task',
    events: [
      {type: 'task_start', task: 'Hidden task'},
      {type: 'usage_info', total_tokens: 4321, cost: '4.30', total_steps: 43},
      {type: 'system_output', text: 'hidden replay\n'},
    ],
  });
  assert.strictEqual(shownTask(win), 'Last task', 'the panel must not move');
  assert.deepStrictEqual(
    metrics(win),
    {tokens: 'Tokens: 777', budget: 'Cost: 7.00', steps: 'Steps: 7'},
    "a hidden tab's replay must not write on the visible status row",
  );

  scrollTo(win, back, 0);
  assert.deepStrictEqual(
    metrics(win),
    {tokens: 'Tokens: 999', budget: 'Cost: $0.50', steps: 'Steps: 3'},
    'the visible tab keeps its own numbers after the replay',
  );
  win.close();
  console.log('PASS a hidden tab’s replay leaves the visible row alone');
}

// A hidden tab's replay can close the sub-agent tab that is on screen —
// a finished run_parallel panel takes its sub-agent tabs with it — and
// put another tab there. The numbers borrowed from the tab that left
// must not land on the tab that arrived.
function testHiddenReplayThatSwitchesTabsKeepsTheNewTabsNumbers() {
  const {win, posted} = makeWebview();
  const parent = posted.find(m => m.type === 'ready').tabId;
  win._testApi.hideWelcome();
  send(win, {type: 'status', running: true, tabId: parent});
  send(win, {
    type: 'task_events',
    tabId: parent,
    chat_id: 'chat-parent',
    task_id: '60',
    task: 'Parent task',
    events: [
      {type: 'task_start', task: 'Parent task'},
      {type: 'usage_info', total_tokens: 111, cost: '1.11', total_steps: 11},
    ],
  });
  send(win, {type: 'tool_call', name: 'run_parallel', tabId: parent});
  const subId = parent + '__sub_child-task';
  send(win, {
    type: 'openSubagentTab',
    tab_id: subId,
    parent_tab_id: parent,
    description: 'child',
    task_id: 'child-task',
    taskIndex: 0,
  });
  clickTab(win, subId);
  assert.strictEqual(win._testApi.getActiveTabId(), subId, 'the sub is on');
  win._testApi.processEvent({
    type: 'usage_info',
    total_tokens: 222,
    cost: '2.22',
    total_steps: 22,
  });
  assert.deepStrictEqual(metrics(win), {
    tokens: 'Tokens: 222',
    budget: 'Cost: $2.22',
    steps: 'Steps: 22',
  });

  // The parent is hidden now. Its child finishes (subagentDone closes
  // the sub-agent tab on screen; a collapse alone never closes a running
  // sub-agent's tab), then the parent's finished transcript replays.
  send(win, {type: 'subagentDone', tab_id: subId});
  assert.strictEqual(
    win._testApi.getActiveTabId(),
    parent,
    'the child finishing must have closed its tab and shown the parent',
  );
  assert.deepStrictEqual(
    metrics(win),
    {tokens: 'Tokens: 111', budget: 'Cost: $1.11', steps: 'Steps: 11'},
    'the swapped-in parent must show its own numbers, not the child\'s',
  );
  send(win, {
    type: 'task_events',
    tabId: parent,
    chat_id: 'chat-parent',
    task_id: '60',
    task: 'Parent task',
    events: [
      {type: 'task_start', task: 'Parent task'},
      {type: 'usage_info', total_tokens: 111, cost: '1.11', total_steps: 11},
      {type: 'tool_call', name: 'run_parallel'},
      {type: 'tool_result', name: 'run_parallel', text: 'done'},
      {type: 'result', success: true},
    ],
  });
  assert.strictEqual(win._testApi.getActiveTabId(), parent, 'replay keeps the parent on');
  assert.deepStrictEqual(
    metrics(win),
    {tokens: 'Tokens: 111', budget: 'Cost: $1.11', steps: 'Steps: 11'},
    'the tab that came on screen must keep its own numbers',
  );
  win.close();
  console.log('PASS a hidden replay that switches tabs keeps the new numbers');
}

// With no neighbouring task loaded there is nothing to disambiguate: the
// panel keeps naming the live task.
function testPanelUnchangedWithoutAdjacentTasks() {
  const {win, O} = setup({});
  scrollToBottom(win, O);
  assert.strictEqual(shownTask(win), 'Main task');
  win.close();
  console.log('PASS the panel is untouched when no neighbour is loaded');
}

async function main() {
  const tests = [
    testPanelNamesTheTaskFillingTheScreen,
    testLastTaskNamedWhenScrolledToBottom,
    testSliverOfLastTaskDoesNotStealThePanel,
    testFirstTaskNamedWhenScrolledToTop,
    testMetricsFollowTheVisibleTask,
    testEveryRegionOpensWithItsTaskPanel,
    testScrollingNearAnEdgePrefetchesTheNeighbour,
    testEdgePushFetchesAtOnce,
    testRunningTaskHasNoNextTask,
    testHistoryClickPinsTheClickedTask,
    testHistoryTabSurvivesTheWelcomeReset,
    testFollowupEchoDoesNotRenameTheTaskOnScreen,
    testPanelAlwaysNamesAnOnScreenTask,
    testLoaderIsNotTranscript,
    testSplicedInTaskRenamesThePanelWithoutAScroll,
    testTabRoundTripKeepsTheTabsOwnTask,
    testReturningToATabNamesWhatIsOnScreen,
    testMetricsDoNotLeakBetweenTabs,
    testLiveMetricsDoNotOverrideTheVisibleTask,
    testPartialLiveMetricsKeepTheLiveTasksOwnNumbers,
    testLiveStepCountDoesNotRepaintTheNeighboursRow,
    testAdjacentReplayLeavesTheLiveStepCountAlone,
    testHiddenTabStreamLeavesTheVisibleRowAlone,
    testHiddenTabReplayLeavesTheVisibleRowAlone,
    testHiddenReplayThatSwitchesTabsKeepsTheNewTabsNumbers,
    testPanelUnchangedWithoutAdjacentTasks,
  ];
  const failures = [];
  for (const t of tests) {
    try {
      await t();
    } catch (e) {
      failures.push(t.name + ': ' + e.message);
      console.error('FAIL ' + t.name + ': ' + e.message);
    }
  }
  if (failures.length) {
    console.error('\n' + failures.length + ' visible-task label test(s) failed');
    process.exit(1);
  }
  console.log('\nALL visible-task label tests passed');
}

main().catch(e => {
  console.error(e);
  process.exit(1);
});
