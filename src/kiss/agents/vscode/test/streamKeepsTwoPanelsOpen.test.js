// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// JSDOM end-to-end test for the streaming collapse pass
// (collapseOlderPanels in media/main.js).
//
// While a task streams, the transcript folds its older panels so the
// screen is not a wall of finished tool calls, but it keeps the NEWEST
// TWO open: the panel that just finished stays readable next to the one
// being streamed. It used to keep only the newest one.
//
// A Thoughts panel (the agent's own words) is never folded by the
// pass: only the tool panels older than the newest two fold.
//
// Covered here, on the visible tab and on a tab that ran while hidden:
//   * after every event, every collapsible panel but the newest two and
//     the Thoughts panels is folded; the newest two are open;
//   * a transcript of one or two panels folds nothing;
//   * a panel the user expanded by hand stays open however old it gets;
//   * the result event folds nothing, so the two panels that were open
//     when the task ended stay open.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');
const TS = 1767225600000;

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
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
  win._testApi.endLaunch();
  win._testApi.hideWelcome();
  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function clickTab(win, tabId) {
  const el = win.document.querySelector(
    `.chat-tab[data-tab-id=${JSON.stringify(tabId)}]`,
  );
  assert.ok(el, `tab ${tabId} must exist in the tab bar`);
  el.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

/** Top-level collapsible panels of the visible transcript, in order. */
/** The task's event panels: on the transcript while the task runs,
 * inside its Trajectory panel once it has ended. */
function panels(win) {
  const out = win.document.getElementById('output');
  const traj = out.querySelector(':scope > .trajectory > .trajectory-sub');
  return Array.from((traj || out).children).filter(
    el => el.classList.contains('collapsible') && !el.classList.contains('rc'),
  );
}

function isCollapsed(p) {
  return p.classList.contains('collapsed');
}

/**
 * Assert the streaming invariant: every panel but the newest two is
 * folded, the newest two are open (a user-pinned panel and a Thoughts
 * panel are open anywhere).
 */
function assertNewestTwoOpen(win, label) {
  const ps = panels(win);
  ps.forEach((p, i) => {
    const isNewestTwo = i >= ps.length - 2;
    if (p.classList.contains('llm-panel')) {
      assert.ok(
        !isCollapsed(p),
        `${label}: Thoughts panel #${i + 1} of ${ps.length} is never folded`,
      );
    } else if (isNewestTwo) {
      assert.ok(
        !isCollapsed(p),
        `${label}: panel #${i + 1} of ${ps.length} (one of the newest two) must be open`,
      );
    } else if (!p.classList.contains('user-pinned')) {
      assert.ok(
        isCollapsed(p),
        `${label}: panel #${i + 1} of ${ps.length} must be folded`,
      );
    }
  });
}

// Thoughts, a tool call, thoughts, a tool call, ... Every tool_result
// arms an (empty) Thoughts panel for the next step at once, so the run
// renders seven top-level panels by the time its last result lands.
function run() {
  return [
    {type: 'thinking_start', ts: TS},
    {type: 'thinking_delta', text: 'look at one', ts: TS},
    {type: 'thinking_end', ts: TS},
    {type: 'tool_call', name: 'Read', path: 'src/one.py', ts: TS},
    {type: 'tool_result', content: 'one', ts: TS},
    {type: 'thinking_start', ts: TS},
    {type: 'thinking_delta', text: 'look at two', ts: TS},
    {type: 'thinking_end', ts: TS},
    {type: 'tool_call', name: 'Read', path: 'src/two.py', ts: TS},
    {type: 'tool_result', content: 'two', ts: TS},
    {type: 'thinking_start', ts: TS},
    {type: 'thinking_delta', text: 'write three', ts: TS},
    {type: 'thinking_end', ts: TS},
    {type: 'tool_call', name: 'Write', path: 'src/three.py', ts: TS},
    {type: 'tool_result', content: 'written', ts: TS},
  ];
}

function testVisibleStreamKeepsNewestTwoOpen() {
  const {win} = makeWebview();
  const tab = win._testApi.getActiveTabId();
  send(win, {type: 'status', running: true, tabId: tab, startTs: TS});

  const events = run();
  events.forEach((ev, i) => {
    send(win, {...ev, tabId: tab});
    assertNewestTwoOpen(win, `after event #${i + 1} (${ev.type})`);
  });
  const ps = panels(win);
  assert.strictEqual(ps.length, 7, 'the run renders seven panels');
  assert.deepStrictEqual(
    ps.map(isCollapsed),
    [false, true, false, true, false, false, false],
    'the two older tool panels are folded, the Thoughts panels and the ' +
      'newest two are open',
  );

  // The end of the task moves the panels into the Trajectory panel as
  // they are: the same two stay open inside it.
  send(win, {
    type: 'result',
    text: 'summary: Done.\nsuccess: true\n',
    summary: 'Done.',
    success: true,
    tabId: tab,
    ts: TS,
  });
  send(win, {type: 'status', running: false, tabId: tab});
  assert.deepStrictEqual(
    panels(win).map(isCollapsed),
    [false, true, false, true, false, false, false],
    'the end of the task leaves the open panels open inside the Trajectory',
  );
  win.close();
  console.log('  ok - a visible stream keeps its newest two panels open');
}

function testOneOrTwoPanelsFoldNothing() {
  const {win} = makeWebview();
  const tab = win._testApi.getActiveTabId();
  send(win, {type: 'status', running: true, tabId: tab, startTs: TS});
  for (const ev of run().slice(0, 4)) send(win, {...ev, tabId: tab});
  const ps = panels(win);
  assert.strictEqual(ps.length, 2, 'thoughts + one tool call = two panels');
  assert.deepStrictEqual(
    ps.map(isCollapsed),
    [false, false],
    'with two panels there is nothing older than the newest two',
  );
  win.close();
  console.log('  ok - a one- or two-panel transcript folds nothing');
}

function testUserPinnedPanelStaysOpen() {
  const {win} = makeWebview();
  const tab = win._testApi.getActiveTabId();
  send(win, {type: 'status', running: true, tabId: tab, startTs: TS});
  const events = run();
  // Two steps: thoughts, Read one, thoughts, Read two, and the armed
  // Thoughts panel of the next step: five panels, the first Read (#2)
  // folds. The user reopens it.
  for (const ev of events.slice(0, 10)) send(win, {...ev, tabId: tab});
  let ps = panels(win);
  assert.strictEqual(ps.length, 5);
  assert.ok(isCollapsed(ps[1]), 'panel #2 folds once two newer panels exist');
  ps[1]
    .querySelector(':scope > .collapse-header')
    .dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.ok(!isCollapsed(ps[1]), 'the user reopened panel #2');
  assert.ok(ps[1].classList.contains('user-pinned'));

  for (const ev of events.slice(10)) send(win, {...ev, tabId: tab});
  ps = panels(win);
  assert.strictEqual(ps.length, 7);
  assert.deepStrictEqual(
    ps.map(isCollapsed),
    [false, false, false, true, false, false, false],
    'the pinned panel stays open; the other older tool panel folds',
  );
  win.close();
  console.log('  ok - a user-pinned panel is never folded by the stream');
}

function testHiddenTabComesBackWithNewestTwoOpen() {
  const {win} = makeWebview();
  const tabA = win._testApi.getActiveTabId();
  win._testApi.createNewTab();
  const tabB = win._testApi.getActiveTabId();
  clickTab(win, tabA);
  assert.strictEqual(win._testApi.getActiveTabId(), tabA);

  // The run streams into the hidden tab B (no result: still running).
  send(win, {type: 'status', running: true, tabId: tabB, startTs: TS});
  for (const ev of run()) send(win, {...ev, tabId: tabB});
  assert.strictEqual(
    win._testApi.getActiveTabId(),
    tabA,
    'a background run must not steal the screen',
  );
  clickTab(win, tabB);
  const ps = panels(win);
  assert.strictEqual(ps.length, 7, 'the hidden run renders seven panels');
  assert.deepStrictEqual(
    ps.map(isCollapsed),
    [false, true, false, true, false, false, false],
    'a tab restored mid-run shows its newest two panels and its ' +
      'Thoughts panels open',
  );
  win.close();
  console.log(
    '  ok - a tab that ran hidden comes back with its newest two open',
  );
}

function main() {
  testVisibleStreamKeepsNewestTwoOpen();
  testOneOrTwoPanelsFoldNothing();
  testUserPinnedPanelStaysOpen();
  testHiddenTabComesBackWithNewestTwoOpen();
  console.log('streamKeepsTwoPanelsOpen.test.js: all tests passed');
}

main();
