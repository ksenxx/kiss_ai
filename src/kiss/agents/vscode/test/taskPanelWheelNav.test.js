// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end test: mouse-wheel task navigation over the FIXED task
// panel of the chat webview.
//
// The issue: the task panel (#task-panel) is pinned to the top of the
// chat webview, so wheel gestures over it went nowhere — the panel
// itself has nothing to scroll and the chat underneath did not move.
// The fix: scrolling over the fixed task panel must step the chat to
// the PREVIOUS (wheel up) or NEXT (wheel down) task of the chat and
// align that task's first event with the top of the viewport, letting
// the user rapidly flip through the tasks of a chat.  When the target
// task is not loaded yet, it is fetched via the same getAdjacentTask
// path as #output overscroll and scrolled to the top once rendered.
//
// This test exercises the real ``media/main.js`` against the real
// ``media/chat.html`` in jsdom, the same harness as
// ``adjacentTaskScroll.test.js``.  jsdom has no layout engine, so the
// tests install a dynamic ``getBoundingClientRect`` model: #output is
// a 500px-tall viewport and every direct child of #output occupies a
// 1000px-tall slab at (childIndex * 1000 - O.scrollTop).
//
// Run directly with ``node``:
//
//     node src/kiss/agents/vscode/test/taskPanelWheelNav.test.js

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

/**
 * Build a jsdom webview running the real chat.html + panelCopy.js +
 * main.js, with a stubbed acquireVsCodeApi that records every message
 * posted to the extension host.
 */
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
  // jsdom has no Element.scrollTo; main.js's rAF auto-scroll calls it.
  win.Element.prototype.scrollTo = function () {};
  const posted = [];
  win.acquireVsCodeApi = function () {
    let state;
    return {
      postMessage: (msg) => posted.push(msg),
      getState: () => state,
      setState: (s) => {
        state = s;
      },
    };
  };
  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  // The sourceURL pragma names this eval instance in V8 coverage
  // output so taskPanelWheelNav.coverage.js can locate it and enforce
  // 100% line coverage of the taskwheel-coverage regions.
  win.eval(
    fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8') +
      '\n//# sourceURL=taskwheel-main.js',
  );
  return {win, posted};
}

/** Deliver a message-event from the extension host to the webview. */
function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

/**
 * Install the fake layout model (jsdom has no layout engine):
 *   - #output is a 500px-tall, 400px-wide viewport with its top at 0;
 *   - every direct child of #output is a 1000px-tall slab whose
 *     viewport position derives from its child index and O.scrollTop;
 *   - O.scrollHeight is children*1000 and O.clientHeight is 500.
 * Newly inserted .adjacent-task containers automatically obey the
 * model, so main.js code that measures them synchronously right after
 * insertion (renderAdjacentTask's pending wheel scroll) sees real
 * geometry.
 */
function installLayout(win, O) {
  const origGBCR = win.Element.prototype.getBoundingClientRect;
  win.Element.prototype.getBoundingClientRect = function () {
    if (this === O) {
      return {top: 0, bottom: 500, height: 500, left: 0, right: 400, width: 400};
    }
    if (this.parentNode === O) {
      const idx = Array.prototype.indexOf.call(O.children, this);
      const top = idx * 1000 - O.scrollTop;
      return {top, bottom: top + 1000, height: 1000, left: 0, right: 400, width: 400};
    }
    return origGBCR.call(this);
  };
  Object.defineProperty(O, 'scrollHeight', {
    get: () => O.children.length * 1000,
    configurable: true,
  });
  Object.defineProperty(O, 'clientHeight', {value: 500, configurable: true});
  Object.defineProperty(O, 'clientWidth', {value: 400, configurable: true});
}

/** Fire one wheel event on el; returns false if default was prevented. */
function wheel(win, el, deltaY) {
  return el.dispatchEvent(
    new win.WheelEvent('wheel', {deltaY, bubbles: true, cancelable: true}),
  );
}

function getAdjacent(posted) {
  return posted.filter((m) => m.type === 'getAdjacentTask');
}

function panelText(win) {
  return win.document.getElementById('task-panel-text').textContent;
}

/** Boot a webview with one loaded history task (task_id '42'). */
function setupWithHistoryTask() {
  const {win, posted} = makeWebview();
  const tabId = posted.find((m) => m.type === 'ready').tabId;
  win._demoApi.hideWelcome();
  const O = win.document.getElementById('output');
  installLayout(win, O);
  send(win, {
    type: 'task_events',
    tabId,
    chat_id: 'chat-abc',
    task_id: '42',
    task: 'My main task',
    events: [
      {type: 'task_start', task: 'My main task'},
      {type: 'system_output', text: 'hello\n'},
    ],
  });
  const panel = win.document.getElementById('task-panel');
  return {win, posted, tabId, O, panel};
}

/**
 * Boot a webview with three loaded tasks: 'Prev task' (id 41) above,
 * the main 'My main task' (id 42), and 'Next task' (id 43) below —
 * exactly the DOM renderAdjacentTask builds after two overscrolls.
 * Returns the region start offsets in content coordinates.
 */
function setupWithThreeTasks() {
  const ctx = setupWithHistoryTask();
  const {win, posted, tabId, O} = ctx;
  // Load the previous task through the real overscroll+reply path.
  O.scrollTop = 0;
  for (let i = 0; i < 5; i++) wheel(win, O, -50);
  assert.ok(getAdjacent(posted).length >= 1, 'setup: prev request posted');
  send(win, {
    type: 'adjacent_task_events',
    tabId,
    direction: 'prev',
    task: 'Prev task',
    task_id: '41',
    events: [
      {type: 'task_start', task: 'Prev task'},
      {type: 'system_output', text: 'prev output\n'},
    ],
  });
  // Load the next task the same way.
  O.scrollTop = O.scrollHeight - O.clientHeight;
  for (let i = 0; i < 5; i++) wheel(win, O, 50);
  send(win, {
    type: 'adjacent_task_events',
    tabId,
    direction: 'next',
    task: 'Next task',
    task_id: '43',
    events: [
      {type: 'task_start', task: 'Next task'},
      {type: 'system_output', text: 'next output\n'},
    ],
  });
  const prevEl = O.querySelector('.adjacent-task[data-task-id="41"]');
  const nextEl = O.querySelector('.adjacent-task[data-task-id="43"]');
  assert.ok(prevEl && nextEl, 'setup: both adjacent tasks rendered');
  assert.strictEqual(prevEl, O.children[0], 'setup: prev task is topmost');
  assert.strictEqual(
    nextEl,
    O.children[O.children.length - 1],
    'setup: next task is bottommost',
  );
  // Content offsets of each region's first element.
  const kids = Array.prototype.slice.call(O.children);
  ctx.prevTop = kids.indexOf(prevEl) * 1000; // 0
  ctx.mainTop = 1000; // first main-task child right below prev
  ctx.nextTop = kids.indexOf(nextEl) * 1000;
  ctx.prevEl = prevEl;
  ctx.nextEl = nextEl;
  return ctx;
}

/** Put the viewport on the main task (its first child at the top). */
function scrollToMain(ctx) {
  ctx.O.scrollTop = ctx.mainTop;
}

// ---------------------------------------------------------------------------
// 1. REPRO/FIX: wheel UP over the fixed task panel steps the chat to
//    the PREVIOUS task, with that task's events at the top of the
//    viewport and the panel text showing that task.
// ---------------------------------------------------------------------------
function testWheelUpGoesToPrevTask() {
  const ctx = setupWithThreeTasks();
  const {win, panel, O} = ctx;
  scrollToMain(ctx);
  wheel(win, panel, -120);
  assert.strictEqual(
    O.scrollTop,
    ctx.prevTop,
    'wheel up over the task panel must scroll the previous task to the top',
  );
  assert.strictEqual(
    panelText(win),
    'Prev task',
    'the task panel must show the task the chat scrolled to',
  );
  win.close();
  console.log('PASS wheel up over the panel scrolls to the previous task');
}

// ---------------------------------------------------------------------------
// 2. Wheel DOWN over the panel steps to the NEXT task, events at top.
// ---------------------------------------------------------------------------
function testWheelDownGoesToNextTask() {
  const ctx = setupWithThreeTasks();
  const {win, panel, O} = ctx;
  scrollToMain(ctx);
  wheel(win, panel, 120);
  assert.strictEqual(
    O.scrollTop,
    ctx.nextTop,
    'wheel down over the task panel must scroll the next task to the top',
  );
  assert.strictEqual(panelText(win), 'Next task');
  win.close();
  console.log('PASS wheel down over the panel scrolls to the next task');
}

// ---------------------------------------------------------------------------
// 3. Rapid flipping: two consecutive wheel-up gestures from the next
//    task land on the previous task (next → main → prev).
// ---------------------------------------------------------------------------
function testRapidFlipAcrossTasks() {
  const ctx = setupWithThreeTasks();
  const {win, panel, O} = ctx;
  O.scrollTop = ctx.nextTop; // viewing the next task
  wheel(win, panel, -120);
  assert.strictEqual(O.scrollTop, ctx.mainTop, 'first step lands on main');
  assert.strictEqual(panelText(win), 'My main task');
  wheel(win, panel, -120);
  assert.strictEqual(O.scrollTop, ctx.prevTop, 'second step lands on prev');
  assert.strictEqual(panelText(win), 'Prev task');
  win.close();
  console.log('PASS rapid consecutive wheel steps flip across tasks');
}

// ---------------------------------------------------------------------------
// 4. Small trackpad deltas accumulate: below the 60px step threshold
//    nothing moves; crossing it performs exactly one step.
// ---------------------------------------------------------------------------
function testDeltaAccumulation() {
  const ctx = setupWithThreeTasks();
  const {win, panel, O} = ctx;
  scrollToMain(ctx);
  wheel(win, panel, 20);
  wheel(win, panel, 20);
  assert.strictEqual(
    O.scrollTop,
    ctx.mainTop,
    '40px of accumulated wheel delta must not navigate yet',
  );
  wheel(win, panel, 20); // 60px accumulated — one step
  assert.strictEqual(
    O.scrollTop,
    ctx.nextTop,
    'crossing the 60px threshold must perform exactly one step',
  );
  win.close();
  console.log('PASS small deltas accumulate into a single step');
}

// ---------------------------------------------------------------------------
// 5. Reversing direction resets the accumulator: 40px down then 40px
//    up must not navigate; another 40px up completes the up-step.
// ---------------------------------------------------------------------------
function testDirectionChangeResetsAccumulator() {
  const ctx = setupWithThreeTasks();
  const {win, panel, O} = ctx;
  scrollToMain(ctx);
  wheel(win, panel, 40); // down 40
  wheel(win, panel, -40); // direction flip: accumulator restarts at 40
  assert.strictEqual(
    O.scrollTop,
    ctx.mainTop,
    'a direction flip must reset the accumulator (no navigation yet)',
  );
  wheel(win, panel, -40); // 80 ≥ 60 — step up
  assert.strictEqual(O.scrollTop, ctx.prevTop);
  win.close();
  console.log('PASS direction change resets the wheel accumulator');
}

// ---------------------------------------------------------------------------
// 6. The accumulator decays after 300ms of inactivity, so stale
//    partial deltas from an old gesture never combine with a new one.
// ---------------------------------------------------------------------------
async function testAccumulatorTimeoutReset() {
  const ctx = setupWithThreeTasks();
  const {win, panel, O} = ctx;
  scrollToMain(ctx);
  wheel(win, panel, 40);
  await new Promise((r) => setTimeout(r, 350)); // inactivity reset
  wheel(win, panel, 40); // fresh gesture: 40 < 60, no step
  assert.strictEqual(
    O.scrollTop,
    ctx.mainTop,
    'stale deltas must expire after the inactivity window',
  );
  wheel(win, panel, 40); // 80 ≥ 60 — step
  assert.strictEqual(O.scrollTop, ctx.nextTop);
  win.close();
  console.log('PASS accumulator resets after inactivity');
}

// ---------------------------------------------------------------------------
// 7. deltaY:0 events (pure horizontal wheel) are ignored entirely.
// ---------------------------------------------------------------------------
function testZeroDeltaIgnored() {
  const ctx = setupWithThreeTasks();
  const {win, panel, O} = ctx;
  scrollToMain(ctx);
  for (let i = 0; i < 10; i++) wheel(win, panel, 0);
  assert.strictEqual(
    O.scrollTop,
    ctx.mainTop,
    'deltaY:0 wheel events must never navigate',
  );
  win.close();
  console.log('PASS zero-delta wheel events are ignored');
}

// ---------------------------------------------------------------------------
// 8. The wheel event over the fixed panel is consumed (preventDefault)
//    so it can not double-scroll the chat underneath.
// ---------------------------------------------------------------------------
function testWheelDefaultPrevented() {
  const ctx = setupWithThreeTasks();
  const {win, panel} = ctx;
  scrollToMain(ctx);
  const notPrevented = wheel(win, panel, 120);
  assert.strictEqual(
    notPrevented,
    false,
    'wheel over the task panel must preventDefault',
  );
  win.close();
  console.log('PASS wheel over the panel is consumed (preventDefault)');
}

// ---------------------------------------------------------------------------
// 9. Wheel UP past the topmost LOADED task requests the previous task
//    from the backend (getAdjacentTask) and — once it renders — puts
//    its events at the top of the viewport with the panel text synced.
// ---------------------------------------------------------------------------
function testWheelUpLoadsUnloadedPrevTask() {
  const {win, posted, tabId, O, panel} = setupWithHistoryTask();
  O.scrollTop = 0;
  wheel(win, panel, -120);
  const adj = getAdjacent(posted);
  assert.strictEqual(adj.length, 1, 'panel wheel must request the prev task');
  assert.strictEqual(adj[0].direction, 'prev');
  assert.strictEqual(adj[0].taskId, '42');
  assert.ok(
    win.document.getElementById('adjacent-loader'),
    'the adjacent-task loader must show while the task loads',
  );
  send(win, {
    type: 'adjacent_task_events',
    tabId,
    direction: 'prev',
    task: 'Older task',
    task_id: '41',
    events: [
      {type: 'task_start', task: 'Older task'},
      {type: 'system_output', text: 'older\n'},
    ],
  });
  const cont = O.querySelector('.adjacent-task[data-task-id="41"]');
  assert.ok(cont, 'the previous task must render after the reply');
  assert.strictEqual(
    O.scrollTop,
    0,
    'the freshly loaded previous task must be scrolled to the top',
  );
  assert.strictEqual(
    panelText(win),
    'Older task',
    'the panel must show the freshly loaded task',
  );
  win.close();
  console.log('PASS wheel up loads and scrolls to an unloaded prev task');
}

// ---------------------------------------------------------------------------
// 10. Wheel DOWN past the bottommost loaded task requests the next
//     task and scrolls to it once rendered.
// ---------------------------------------------------------------------------
function testWheelDownLoadsUnloadedNextTask() {
  const {win, posted, tabId, O, panel} = setupWithHistoryTask();
  O.scrollTop = 0;
  wheel(win, panel, 120);
  const adj = getAdjacent(posted);
  assert.strictEqual(adj.length, 1, 'panel wheel must request the next task');
  assert.strictEqual(adj[0].direction, 'next');
  assert.strictEqual(adj[0].taskId, '42');
  send(win, {
    type: 'adjacent_task_events',
    tabId,
    direction: 'next',
    task: 'Newer task',
    task_id: '43',
    events: [
      {type: 'task_start', task: 'Newer task'},
      {type: 'system_output', text: 'newer\n'},
    ],
  });
  const cont = O.querySelector('.adjacent-task[data-task-id="43"]');
  assert.ok(cont, 'the next task must render after the reply');
  const contTop = Array.prototype.indexOf.call(O.children, cont) * 1000;
  assert.strictEqual(
    O.scrollTop,
    contTop,
    'the freshly loaded next task must be scrolled to the top',
  );
  assert.strictEqual(panelText(win), 'Newer task');
  win.close();
  console.log('PASS wheel down loads and scrolls to an unloaded next task');
}

// ---------------------------------------------------------------------------
// 11. A normal #output overscroll load (NOT panel-initiated) must keep
//     the reading position exactly as before — the pending-scroll flag
//     must not leak into the plain overscroll path.
// ---------------------------------------------------------------------------
function testPlainOverscrollLoadKeepsPosition() {
  const {win, posted, tabId, O} = setupWithHistoryTask();
  O.scrollTop = 0;
  for (let i = 0; i < 5; i++) wheel(win, O, -50);
  assert.ok(getAdjacent(posted).length >= 1);
  send(win, {
    type: 'adjacent_task_events',
    tabId,
    direction: 'prev',
    task: 'Older task',
    task_id: '41',
    events: [
      {type: 'task_start', task: 'Older task'},
      {type: 'system_output', text: 'older\n'},
    ],
  });
  assert.strictEqual(
    O.scrollTop,
    1000,
    'a plain overscroll load must anchor the previous reading position',
  );
  win.close();
  console.log('PASS plain overscroll load still keeps the reading position');
}

// ---------------------------------------------------------------------------
// 12. A genuine end-of-chat reply (no previous task exists) latches
//     noPrevTask: further panel wheel-ups post nothing and stay put.
// ---------------------------------------------------------------------------
function testNoPrevTaskLatchStopsRequests() {
  const {win, posted, tabId, O, panel} = setupWithHistoryTask();
  O.scrollTop = 0;
  wheel(win, panel, -120);
  assert.strictEqual(getAdjacent(posted).length, 1);
  send(win, {
    type: 'adjacent_task_events',
    tabId,
    direction: 'prev',
    task: '',
    task_id: null,
    events: [],
  });
  wheel(win, panel, -120);
  assert.strictEqual(
    getAdjacent(posted).length,
    1,
    'after a genuine end-of-chat reply, panel wheel must not re-request',
  );
  assert.strictEqual(O.scrollTop, 0, 'the view must stay put');
  win.close();
  console.log('PASS latched noPrevTask suppresses further panel requests');
}

// ---------------------------------------------------------------------------
// 13. While an adjacent-task request is in flight, further panel wheel
//     steps must not post duplicate requests.
// ---------------------------------------------------------------------------
function testNoDuplicateRequestWhileLoading() {
  const {win, posted, O, panel} = setupWithHistoryTask();
  O.scrollTop = 0;
  wheel(win, panel, -120);
  assert.strictEqual(getAdjacent(posted).length, 1);
  wheel(win, panel, -120); // still loading — must not re-post
  assert.strictEqual(
    getAdjacent(posted).length,
    1,
    'a second panel wheel step while loading must not duplicate the request',
  );
  win.close();
  console.log('PASS no duplicate getAdjacentTask while one is in flight');
}

// ---------------------------------------------------------------------------
// 14. Sub-agent tabs render exactly one task: panel wheel must be a
//     no-op there (no scroll, no request).
// ---------------------------------------------------------------------------
function testSubagentTabIgnoresPanelWheel() {
  const ctx = setupWithThreeTasks();
  const {win, posted, tabId, O, panel} = ctx;
  scrollToMain(ctx);
  // Convert the active tab into a sub-agent tab via the real event.
  send(win, {
    type: 'openSubagentTab',
    tab_id: tabId,
    parent_tab_id: tabId,
    description: 'Sub-agent task',
    task_id: '99',
  });
  const before = getAdjacent(posted).length;
  wheel(win, panel, -120);
  assert.strictEqual(
    O.scrollTop,
    ctx.mainTop,
    'panel wheel on a sub-agent tab must not scroll anywhere',
  );
  assert.strictEqual(
    getAdjacent(posted).length,
    before,
    'panel wheel on a sub-agent tab must not request adjacent tasks',
  );
  win.close();
  console.log('PASS sub-agent tabs ignore panel wheel navigation');
}

// ---------------------------------------------------------------------------
// 15. A fresh webview with no task at all (only the welcome screen):
//     panel wheel is a safe no-op — the welcome block is not a task.
// ---------------------------------------------------------------------------
function testEmptyChatIsNoop() {
  const {win, posted} = makeWebview();
  const O = win.document.getElementById('output');
  installLayout(win, O);
  const panel = win.document.getElementById('task-panel');
  wheel(win, panel, -120);
  wheel(win, panel, 120);
  assert.strictEqual(
    getAdjacent(posted).length,
    0,
    'panel wheel on an empty chat must not post anything',
  );
  win.close();
  console.log('PASS empty chat (welcome only) is a safe no-op');
}

// ---------------------------------------------------------------------------
// 16. A task with an unknown row id (no task_id in the replay) must
//     never produce a getAdjacentTask request with an empty taskId.
// ---------------------------------------------------------------------------
function testUnknownAnchorIdNeverRequested() {
  const {win, posted} = makeWebview();
  const tabId = posted.find((m) => m.type === 'ready').tabId;
  win._demoApi.hideWelcome();
  const O = win.document.getElementById('output');
  installLayout(win, O);
  send(win, {
    type: 'task_events',
    tabId,
    chat_id: 'chat-abc',
    task: 'Untracked task',
    events: [
      {type: 'task_start', task: 'Untracked task'},
      {type: 'system_output', text: 'hello\n'},
    ],
  });
  const panel = win.document.getElementById('task-panel');
  wheel(win, panel, -120);
  wheel(win, panel, -120);
  assert.strictEqual(
    getAdjacent(posted).length,
    0,
    'panel wheel must never post getAdjacentTask with an unknown anchor id',
  );
  win.close();
  console.log('PASS unknown anchor id never produces a request');
}

// ---------------------------------------------------------------------------
// 17. Viewport parked ABOVE the first region (overscrolled top): the
//     visible region falls back to the first one, so a wheel-down
//     lands on the SECOND task.
// ---------------------------------------------------------------------------
function testViewportAboveFirstRegionFallback() {
  const ctx = setupWithThreeTasks();
  const {win, panel, O} = ctx;
  O.scrollTop = -600; // every region is below the 30% check line
  wheel(win, panel, 120);
  assert.strictEqual(
    O.scrollTop,
    ctx.mainTop,
    'from above the first task, wheel down must land on the second task',
  );
  assert.strictEqual(panelText(win), 'My main task');
  win.close();
  console.log('PASS viewport above the first task falls back to index 0');
}

// ---------------------------------------------------------------------------
// 18. Viewport parked PAST the last region (overscrolled bottom): the
//     visible region falls back to the last one, so a wheel-up lands
//     on the SECOND-TO-LAST task.
// ---------------------------------------------------------------------------
function testViewportPastLastRegionFallback() {
  const ctx = setupWithThreeTasks();
  const {win, panel, O} = ctx;
  O.scrollTop = O.scrollHeight + 600; // every region is above the line
  wheel(win, panel, -120);
  assert.strictEqual(
    O.scrollTop,
    ctx.mainTop,
    'from past the last task, wheel up must land on the second-to-last task',
  );
  assert.strictEqual(panelText(win), 'My main task');
  win.close();
  console.log('PASS viewport past the last task falls back to the last index');
}

// ---------------------------------------------------------------------------
// 19. Wheel up while already on the FIRST task and end-of-chat is NOT
//     latched requests the task before it; wheel down on the LAST task
//     requests the one after it (anchors chain off the loaded ends).
// ---------------------------------------------------------------------------
function testAnchorsChainOffLoadedEnds() {
  const ctx = setupWithThreeTasks();
  const {win, posted, panel, O} = ctx;
  O.scrollTop = ctx.prevTop; // viewing the first loaded task
  const before = getAdjacent(posted).length;
  wheel(win, panel, -120);
  let adj = getAdjacent(posted);
  assert.strictEqual(adj.length, before + 1);
  assert.strictEqual(adj[adj.length - 1].direction, 'prev');
  assert.strictEqual(
    adj[adj.length - 1].taskId,
    '41',
    'the request must chain off the OLDEST loaded task id',
  );
  win.close();
  console.log('PASS panel wheel chains requests off the loaded ends');
}

/**
 * Install a REALISTIC layout: per-element heights via `heightOf(el)`,
 * .chv-hidden children are 0px (display:none has a zero rect), and
 * O.scrollTop is CLAMPED to [0, scrollHeight - clientHeight] exactly
 * like real browser engines clamp it (the simple installLayout model
 * accepts impossible scroll positions).
 */
function installRealisticLayout(win, O, heightOf) {
  function h(el) {
    if (el.classList.contains('chv-hidden')) return 0;
    return heightOf(el);
  }
  function contentTop(el) {
    let n = 0;
    for (const c of O.children) {
      if (c === el) return n;
      n += h(c);
    }
    return n;
  }
  function sh() {
    let n = 0;
    for (const c of O.children) n += h(c);
    return n;
  }
  let st = 0;
  Object.defineProperty(O, 'scrollTop', {
    get: () => st,
    set: (v) => {
      st = Math.max(0, Math.min(v, Math.max(0, sh() - 500)));
    },
    configurable: true,
  });
  Object.defineProperty(O, 'scrollHeight', {get: sh, configurable: true});
  Object.defineProperty(O, 'clientHeight', {value: 500, configurable: true});
  win.Element.prototype.getBoundingClientRect = function () {
    if (this === O) {
      return {top: 0, bottom: 500, height: 500, left: 0, right: 400, width: 400};
    }
    if (this.parentNode === O) {
      if (this.classList.contains('chv-hidden')) {
        return {top: 0, bottom: 0, height: 0, left: 0, right: 0, width: 0};
      }
      const top = contentTop(this) - st;
      const hh = h(this);
      return {top, bottom: top + hh, height: hh, left: 0, right: 400, width: 400};
    }
    return {top: 0, bottom: 0, height: 0, left: 0, right: 0, width: 0};
  };
  return {contentTop};
}

// ---------------------------------------------------------------------------
// 20. REGRESSION (review): a SHORT previous task (shorter than 30% of
//     the viewport).  After wheel-up pins it to the top, the panel
//     must show the SHORT task (the 30% probe would resolve past it),
//     and the next wheel-down must land on the main task — not skip
//     it.  Scrolling away dissolves the pin (probe takes over again).
// ---------------------------------------------------------------------------
function testShortPrevTaskNavigation() {
  const ctx = setupWithThreeTasks();
  const {win, panel, O, prevEl} = ctx;
  const {contentTop} = installRealisticLayout(win, O, (el) =>
    el === prevEl ? 50 : 1000,
  );
  const mainFirst = Array.from(O.children).find(
    (c) => !c.classList.contains('adjacent-task'),
  );
  O.scrollTop = contentTop(mainFirst); // viewing the main task
  wheel(win, panel, -120);
  assert.strictEqual(O.scrollTop, 0, 'short prev task scrolled to top');
  assert.strictEqual(
    panelText(win),
    'Prev task',
    'a 50px task pinned at the top must own the panel text (the 30% ' +
      'probe would show the task after it)',
  );
  wheel(win, panel, 120);
  assert.strictEqual(
    O.scrollTop,
    contentTop(mainFirst),
    'wheel down from a short task must land on the MAIN task, not skip it',
  );
  assert.strictEqual(panelText(win), 'My main task');
  // A user scroll away from the pinned position dissolves the pin:
  // the 30% probe decides again, so wheel-up goes back to prev.
  O.scrollTop = contentTop(mainFirst) + 300;
  O.dispatchEvent(new win.Event('scroll'));
  wheel(win, panel, -120);
  assert.strictEqual(O.scrollTop, 0);
  assert.strictEqual(panelText(win), 'Prev task');
  win.close();
  console.log('PASS short prev task is pinned, not skipped');
}

// ---------------------------------------------------------------------------
// 21. REGRESSION (review): the main task's FIRST child can be a
//     .chv-hidden collapsible (display:none, zero rect — "Uncollapse
//     Chats" tucks replayed system-prompt panels away).  Wheel-up from
//     the next task must scroll to the main task's first VISIBLE
//     event, not to a hidden element's zero rect.
// ---------------------------------------------------------------------------
function testHiddenFirstMainChild() {
  const {win, posted} = makeWebview();
  const tabId = posted.find((m) => m.type === 'ready').tabId;
  win._demoApi.hideWelcome();
  const O = win.document.getElementById('output');
  installLayout(win, O);
  send(win, {
    type: 'task_events',
    tabId,
    chat_id: 'chat-abc',
    task_id: '42',
    task: 'My main task',
    events: [
      {type: 'task_start', task: 'My main task'},
      {type: 'system_prompt', text: 'hidden prompt'},
      {type: 'system_output', text: 'visible output\n'},
    ],
  });
  const panel = win.document.getElementById('task-panel');
  O.scrollTop = O.scrollHeight - O.clientHeight;
  for (let i = 0; i < 5; i++) wheel(win, O, 50);
  send(win, {
    type: 'adjacent_task_events',
    tabId,
    direction: 'next',
    task: 'Next task',
    task_id: '43',
    events: [
      {type: 'task_start', task: 'Next task'},
      {type: 'system_output', text: 'next\n'},
    ],
  });
  const nextEl = O.querySelector('.adjacent-task[data-task-id="43"]');
  const hidden = O.querySelector(':scope > .chv-hidden');
  assert.ok(
    hidden,
    'setup: the replay must produce a hidden (chv-hidden) main-task child',
  );
  const visFirst = Array.from(O.children).find(
    (c) => c !== nextEl && !c.classList.contains('chv-hidden'),
  );
  const {contentTop} = installRealisticLayout(win, O, () => 1000);
  O.scrollTop = contentTop(nextEl); // viewing the next task
  wheel(win, panel, -120);
  assert.strictEqual(
    O.scrollTop,
    contentTop(visFirst),
    "wheel up must scroll to the main task's first VISIBLE event " +
      '(a display:none panel has a zero rect and can not be the anchor)',
  );
  assert.strictEqual(panelText(win), 'My main task');
  win.close();
  console.log('PASS hidden first main-task child does not break navigation');
}

// ---------------------------------------------------------------------------
// 22. REGRESSION (review): a SHORT LAST task.  Real browsers clamp
//     scrollTop, so the short task can never reach the very top of the
//     viewport — the pinned target must still own the panel text, the
//     next wheel-down must fetch the task AFTER it, and a wheel-up
//     must return to the main task.
// ---------------------------------------------------------------------------
function testClampedShortLastTask() {
  const {win, posted, tabId, O, panel} = setupWithHistoryTask();
  send(win, {
    type: 'adjacent_task_events',
    tabId,
    direction: 'next',
    task: 'Short next',
    task_id: '43',
    events: [],
  });
  const nextEl = O.querySelector('.adjacent-task[data-task-id="43"]');
  assert.ok(nextEl, 'setup: the short next task rendered');
  installRealisticLayout(win, O, (el) => (el === nextEl ? 50 : 1000));
  O.scrollTop = 0;
  wheel(win, panel, 120);
  assert.strictEqual(
    O.scrollTop,
    O.scrollHeight - 500,
    'the browser clamps the scroll: the short LAST task stops at the ' +
      'maximum scroll position',
  );
  assert.strictEqual(
    panelText(win),
    'Short next',
    'the clamped target must still own the panel text',
  );
  // The pinned short task is the current one: the next wheel-down
  // must fetch the task AFTER it, not re-target it.
  wheel(win, panel, 120);
  const adj = getAdjacent(posted);
  assert.strictEqual(adj.length, 1);
  assert.strictEqual(adj[0].taskId, '43');
  assert.strictEqual(adj[0].direction, 'next');
  // And a wheel-up returns to the main task.
  wheel(win, panel, -120);
  assert.strictEqual(O.scrollTop, 0);
  assert.strictEqual(panelText(win), 'My main task');
  win.close();
  console.log('PASS clamped short last task is pinned and chains');
}

// ---------------------------------------------------------------------------
// 23. REGRESSION (review): loading a new task resets the wheel
//     accumulator — a half gesture on the OLD task must not combine
//     with a small delta on the new one into a surprise navigation.
// ---------------------------------------------------------------------------
function testAccumulatorClearedOnTaskLoad() {
  const ctx = setupWithThreeTasks();
  const {win, posted, tabId, panel} = ctx;
  scrollToMain(ctx);
  wheel(win, panel, 40); // half a gesture — below the 60px step
  const before = getAdjacent(posted).length;
  send(win, {
    type: 'task_events',
    tabId,
    chat_id: 'chat-abc',
    task_id: '52',
    task: 'Reloaded task',
    events: [
      {type: 'task_start', task: 'Reloaded task'},
      {type: 'system_output', text: 'fresh\n'},
    ],
  });
  wheel(win, panel, 40); // 40 < 60 — must NOT combine with the stale 40
  assert.strictEqual(
    getAdjacent(posted).length,
    before,
    'a stale half-gesture must not leak into the newly loaded task',
  );
  win.close();
  console.log('PASS wheel accumulator resets when a task loads');
}

// ---------------------------------------------------------------------------
// 24. REGRESSION (review): deleting the pinned task dissolves the pin
//     — the next wheel step keys off the 30% probe, not a detached
//     DOM node.
// ---------------------------------------------------------------------------
function testPinDissolvesWhenTaskDeleted() {
  const ctx = setupWithThreeTasks();
  const {win, panel, O} = ctx;
  scrollToMain(ctx);
  wheel(win, panel, -120); // pin on 'Prev task' (id 41) at the top
  assert.strictEqual(panelText(win), 'Prev task');
  send(win, {
    type: 'taskDeleted',
    chatId: 'chat-abc',
    taskId: '41',
    chatHasMoreTasks: true,
  });
  assert.strictEqual(
    O.querySelector('.adjacent-task[data-task-id="41"]'),
    null,
    'setup: taskDeleted removed the pinned container',
  );
  wheel(win, panel, 120);
  assert.strictEqual(
    panelText(win),
    'Next task',
    'after the pinned task is deleted, the probe resolves the main ' +
      'task (now topmost) and wheel-down steps to the next one',
  );
  win.close();
  console.log('PASS deleting the pinned task dissolves the pin');
}

async function main() {
  testWheelUpGoesToPrevTask();
  testWheelDownGoesToNextTask();
  testRapidFlipAcrossTasks();
  testDeltaAccumulation();
  testDirectionChangeResetsAccumulator();
  await testAccumulatorTimeoutReset();
  testZeroDeltaIgnored();
  testWheelDefaultPrevented();
  testWheelUpLoadsUnloadedPrevTask();
  testWheelDownLoadsUnloadedNextTask();
  testPlainOverscrollLoadKeepsPosition();
  testNoPrevTaskLatchStopsRequests();
  testNoDuplicateRequestWhileLoading();
  testSubagentTabIgnoresPanelWheel();
  testEmptyChatIsNoop();
  testUnknownAnchorIdNeverRequested();
  testViewportAboveFirstRegionFallback();
  testViewportPastLastRegionFallback();
  testAnchorsChainOffLoadedEnds();
  testShortPrevTaskNavigation();
  testHiddenFirstMainChild();
  testClampedShortLastTask();
  testAccumulatorClearedOnTaskLoad();
  testPinDissolvesWhenTaskDeleted();
  console.log('All taskPanelWheelNav tests passed');
}

main().then(
  () => process.exit(0),
  (err) => {
    console.error(err);
    process.exit(1);
  },
);
