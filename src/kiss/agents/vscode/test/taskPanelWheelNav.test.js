// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

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

  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(
fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8') +
      '\n//# sourceURL=taskwheel-main.js',
  );
  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

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

function setupWithHistoryTask() {
  const {win, posted} = makeWebview();
  const tabId = posted.find((m) => m.type === 'ready').tabId;
  win._testApi.hideWelcome();
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

function setupWithThreeTasks() {
  const ctx = setupWithHistoryTask();
  const {win, posted, tabId, O} = ctx;
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
  const kids = Array.prototype.slice.call(O.children);
  ctx.prevTop = kids.indexOf(prevEl) * 1000;
  ctx.mainTop = 1000;
  ctx.nextTop = kids.indexOf(nextEl) * 1000;
  ctx.prevEl = prevEl;
  ctx.nextEl = nextEl;
  return ctx;
}

function scrollToMain(ctx) {
  ctx.O.scrollTop = ctx.mainTop;
}

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

function testRapidFlipAcrossTasks() {
  const ctx = setupWithThreeTasks();
  const {win, panel, O} = ctx;
  O.scrollTop = ctx.nextTop;
  wheel(win, panel, -120);
  assert.strictEqual(O.scrollTop, ctx.mainTop, 'first step lands on main');
  assert.strictEqual(panelText(win), 'My main task');
  wheel(win, panel, -120);
  assert.strictEqual(O.scrollTop, ctx.prevTop, 'second step lands on prev');
  assert.strictEqual(panelText(win), 'Prev task');
  win.close();
  console.log('PASS rapid consecutive wheel steps flip across tasks');
}

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
  wheel(win, panel, 20);
  assert.strictEqual(
    O.scrollTop,
    ctx.nextTop,
    'crossing the 60px threshold must perform exactly one step',
  );
  win.close();
  console.log('PASS small deltas accumulate into a single step');
}

function testDirectionChangeResetsAccumulator() {
  const ctx = setupWithThreeTasks();
  const {win, panel, O} = ctx;
  scrollToMain(ctx);
  wheel(win, panel, 40);
  wheel(win, panel, -40);
  assert.strictEqual(
    O.scrollTop,
    ctx.mainTop,
    'a direction flip must reset the accumulator (no navigation yet)',
  );
  wheel(win, panel, -40);
  assert.strictEqual(O.scrollTop, ctx.prevTop);
  win.close();
  console.log('PASS direction change resets the wheel accumulator');
}

async function testAccumulatorTimeoutReset() {
  const ctx = setupWithThreeTasks();
  const {win, panel, O} = ctx;
  scrollToMain(ctx);
  wheel(win, panel, 40);
  await new Promise((r) => setTimeout(r, 350));
  wheel(win, panel, 40);
  assert.strictEqual(
    O.scrollTop,
    ctx.mainTop,
    'stale deltas must expire after the inactivity window',
  );
  wheel(win, panel, 40);
  assert.strictEqual(O.scrollTop, ctx.nextTop);
  win.close();
  console.log('PASS accumulator resets after inactivity');
}

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

function testNoDuplicateRequestWhileLoading() {
  const {win, posted, O, panel} = setupWithHistoryTask();
  O.scrollTop = 0;
  wheel(win, panel, -120);
  assert.strictEqual(getAdjacent(posted).length, 1);
  wheel(win, panel, -120);
  assert.strictEqual(
    getAdjacent(posted).length,
    1,
    'a second panel wheel step while loading must not duplicate the request',
  );
  win.close();
  console.log('PASS no duplicate getAdjacentTask while one is in flight');
}

function testSubagentTabIgnoresPanelWheel() {
  const ctx = setupWithThreeTasks();
  const {win, posted, tabId, O, panel} = ctx;
  scrollToMain(ctx);
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

function testUnknownAnchorIdNeverRequested() {
  const {win, posted} = makeWebview();
  const tabId = posted.find((m) => m.type === 'ready').tabId;
  win._testApi.hideWelcome();
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

function testViewportAboveFirstRegionFallback() {
  const ctx = setupWithThreeTasks();
  const {win, panel, O} = ctx;
  O.scrollTop = -600;
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

function testViewportPastLastRegionFallback() {
  const ctx = setupWithThreeTasks();
  const {win, panel, O} = ctx;
  O.scrollTop = O.scrollHeight + 600;
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

function testAnchorsChainOffLoadedEnds() {
  const ctx = setupWithThreeTasks();
  const {win, posted, panel, O} = ctx;
  O.scrollTop = ctx.prevTop;
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

function testShortPrevTaskNavigation() {
  const ctx = setupWithThreeTasks();
  const {win, panel, O, prevEl} = ctx;
  const {contentTop} = installRealisticLayout(win, O, (el) =>
    el === prevEl ? 50 : 1000,
  );
  const mainFirst = Array.from(O.children).find(
    (c) => !c.classList.contains('adjacent-task'),
  );
  O.scrollTop = contentTop(mainFirst);
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
  O.scrollTop = contentTop(mainFirst) + 300;
  O.dispatchEvent(new win.Event('scroll'));
  wheel(win, panel, -120);
  assert.strictEqual(O.scrollTop, 0);
  assert.strictEqual(panelText(win), 'Prev task');
  win.close();
  console.log('PASS short prev task is pinned, not skipped');
}

function testHiddenFirstMainChild() {
  const {win, posted} = makeWebview();
  const tabId = posted.find((m) => m.type === 'ready').tabId;
  win._testApi.hideWelcome();
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
  O.scrollTop = contentTop(nextEl);
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
  wheel(win, panel, 120);
  const adj = getAdjacent(posted);
  assert.strictEqual(adj.length, 1);
  assert.strictEqual(adj[0].taskId, '43');
  assert.strictEqual(adj[0].direction, 'next');
  wheel(win, panel, -120);
  assert.strictEqual(O.scrollTop, 0);
  assert.strictEqual(panelText(win), 'My main task');
  win.close();
  console.log('PASS clamped short last task is pinned and chains');
}

function testAccumulatorClearedOnTaskLoad() {
  const ctx = setupWithThreeTasks();
  const {win, posted, tabId, panel} = ctx;
  scrollToMain(ctx);
  wheel(win, panel, 40);
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
  wheel(win, panel, 40);
  assert.strictEqual(
    getAdjacent(posted).length,
    before,
    'a stale half-gesture must not leak into the newly loaded task',
  );
  win.close();
  console.log('PASS wheel accumulator resets when a task loads');
}

function testPinDissolvesWhenPinnedNodeRemoved() {
  const ctx = setupWithThreeTasks();
  const {win, panel, O} = ctx;
  scrollToMain(ctx);
  wheel(win, panel, -120);
  assert.strictEqual(panelText(win), 'Prev task');
  const pinned = O.querySelector('.adjacent-task[data-task-id="41"]');
  assert.ok(pinned, 'setup: the prev task container is pinned');
  pinned.remove();
  wheel(win, panel, 120);
  assert.strictEqual(
    panelText(win),
    'Next task',
    'after the pinned container leaves the DOM, the probe resolves the ' +
      'main task (now topmost) and wheel-down steps to the next one',
  );
  win.close();
  console.log('PASS removing the pinned container dissolves the pin');
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
  testPinDissolvesWhenPinnedNodeRemoved();
  console.log('All taskPanelWheelNav tests passed');
}

main().then(
  () => process.exit(0),
  (err) => {
    console.error(err);
    process.exit(1);
  },
);
