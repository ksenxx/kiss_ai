// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// JSDOM end-to-end test for the streaming collapse pass
// (collapseOlderPanels in media/main.js).
//
// While a task streams, the transcript's collapse pass runs after every
// event.  A tool-call panel starts folded the moment it renders (its
// header carries the call) and only a click of the user unfolds it,
// however new it is; a Thoughts panel (the agent's own words) is never
// folded by the pass.
//
// Covered here, on the visible tab and on a tab that ran while hidden:
//   * after every event, every tool-call panel is folded unless the
//     user pinned it, every Thoughts panel is open;
//   * a tool-call panel is folded even when it is one of only two;
//   * a panel the user expanded by hand stays open however old it gets;
//   * the result event folds nothing, so what was open when the task
//     ended stays open.

'use strict';

const assert = require('assert');
const harness = require('./simplify2_harness');
const {send, stubRaf, clickTab} = harness;

const TS = 1767225600000;

function makePage() {
  const {win} = harness.makePage({beforeScripts: stubRaf});
  win._testApi.endLaunch();
  win._testApi.hideWelcome();
  return {win};
}

/**
 * The task's top-level collapsible panels, in order: on the transcript
 * while the task runs, inside its Trajectory panel once it has ended.
 */
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
 * Assert the streaming invariant: a Thoughts panel is open anywhere and
 * a tool-call panel is folded (it starts so) unless the user pinned it.
 * The run() fixture renders no other panel kind; one would need its
 * own rule here.
 */
function assertStreamFolding(win, label) {
  const ps = panels(win);
  ps.forEach((p, i) => {
    if (p.classList.contains('llm-panel')) {
      assert.ok(
        !isCollapsed(p),
        `${label}: Thoughts panel #${i + 1} of ${ps.length} is never folded`,
      );
    } else if (p.classList.contains('user-pinned')) {
      assert.ok(
        !isCollapsed(p),
        `${label}: pinned panel #${i + 1} of ${ps.length} stays open`,
      );
    } else {
      assert.ok(
        p.classList.contains('tc'),
        `${label}: panel #${i + 1} of ${ps.length} has no rule: ${p.className}`,
      );
      assert.ok(
        isCollapsed(p),
        `${label}: tool panel #${i + 1} of ${ps.length} is folded`,
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

function testVisibleStreamFoldsToolPanels() {
  const {win} = makePage();
  const tab = win._testApi.getActiveTabId();
  send(win, {type: 'status', running: true, tabId: tab, startTs: TS});

  const events = run();
  events.forEach((ev, i) => {
    send(win, {...ev, tabId: tab});
    assertStreamFolding(win, `after event #${i + 1} (${ev.type})`);
  });
  const ps = panels(win);
  assert.strictEqual(ps.length, 7, 'the run renders seven panels');
  assert.deepStrictEqual(
    ps.map(isCollapsed),
    [false, true, false, true, false, true, false],
    'every tool panel is folded, every Thoughts panel is open',
  );

  // The end of the task moves the panels into the Trajectory panel as
  // they are: the same open panels stay open inside it.  The seventh,
  // the empty Thoughts panel opened on spec after the last tool result,
  // is withdrawn by the result (nothing was ever said into it), as
  // task_stopped would withdraw it.
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
    [false, true, false, true, false, true],
    'the end of the task leaves the open panels open inside the Trajectory',
  );
  win.close();
  console.log(
    '  ok - a visible stream folds its tool panels, keeps Thoughts open',
  );
}

function testUserPinnedPanelStaysOpen() {
  const {win} = makePage();
  const tab = win._testApi.getActiveTabId();
  send(win, {type: 'status', running: true, tabId: tab, startTs: TS});
  const events = run();
  // Two steps: thoughts, Read one, thoughts, Read two, and the armed
  // Thoughts panel of the next step: five panels, the first Read (#2)
  // folds. The user reopens it.
  for (const ev of events.slice(0, 10)) send(win, {...ev, tabId: tab});
  let ps = panels(win);
  assert.strictEqual(ps.length, 5);
  assert.ok(isCollapsed(ps[1]), 'panel #2, a tool call, started folded');
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
    [false, false, false, true, false, true, false],
    'the pinned panel stays open; the other tool panels are folded',
  );
  win.close();
  console.log('  ok - a user-pinned panel is never folded by the stream');
}

function testHiddenTabComesBackFolded() {
  const {win} = makePage();
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
    [false, true, false, true, false, true, false],
    'a tab restored mid-run shows its tool panels folded and its ' +
      'Thoughts panels open',
  );
  win.close();
  console.log(
    '  ok - a tab that ran hidden comes back with its Thoughts panels open',
  );
}

function main() {
  testVisibleStreamFoldsToolPanels();
  testUserPinnedPanelStaysOpen();
  testHiddenTabComesBackFolded();
  console.log('streamFoldsToolPanels.test.js: all tests passed');
}

main();
