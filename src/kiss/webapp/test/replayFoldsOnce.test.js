// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// JSDOM end-to-end test: a task_events replay folds its transcript ONCE.
//
// Replaying a running task's transcript (a page reconnecting to a
// long run) used to run the streaming collapse pass after EVERY replayed
// event; that pass rebuilt the preview text of every older panel from
// its DOM each time, so a 1,500-event run froze the page for ~13 s.
// The replay now folds the whole transcript once, at its end, and the
// collapse pass skips panels that are already folded.
//
// Covered here:
//   * a 150-step running-task replay (302 panels) completes well under
//     the quadratic cost (15 s in this harness before the fix) and ends
//     with every older panel folded, exactly as before;
//   * live streaming after the replay still keeps the newest two panels
//     open, so the replay-only shortcut does not leak into the stream.

/* global require, console, process */
'use strict';

const assert = require('assert');
const {makePage: makeChatPage, send, stubRaf} = require('./simplify2_harness');

const TS = 1767225600000;
const STEPS = 150;
// The pre-fix replay of STEPS steps took ~15 s here; the fixed one ~0.7 s.
// The budget bounds CPU time (see replayRunning), not wall-clock time.
const REPLAY_BUDGET_MS = 5000;

function makePage() {
  const {win} = makeChatPage({
    bodyClass: 'remote-chat',
    beforeScripts: stubRaf,
  });
  win._testApi.endLaunch();
  win._testApi.hideWelcome();
  return {win};
}

/**
 * Top-level collapsible panels of the visible transcript, in order
 * (the task panel that opens the transcript is not an event panel).
 */
function panels(win) {
  const out = win.document.getElementById('output');
  return Array.from(out.children).filter(
    el =>
      el.classList.contains('collapsible') &&
      !el.classList.contains('rc') &&
      !el.classList.contains('task-panel'),
  );
}

function isCollapsed(p) {
  return p.classList.contains('collapsed');
}

/** A Thoughts panel is never folded by any automatic pass. */
function isThoughts(p) {
  return p.classList.contains('llm-panel');
}

/** Folded, or a Thoughts panel (which no pass folds). */
function isFoldedOrThoughts(p) {
  return isThoughts(p) || isCollapsed(p);
}

/** One thinking + Read step of a run. */
function step(i) {
  return [
    {type: 'thinking_start', ts: TS},
    {type: 'thinking_delta', text: 'step ' + i + ': look at the file', ts: TS},
    {type: 'thinking_end', ts: TS},
    {type: 'tool_call', name: 'Read', path: 'src/file' + i + '.py', ts: TS},
    {type: 'tool_result', content: 'content of file ' + i, ts: TS},
  ];
}

function longRun(steps) {
  const events = [{type: 'prompt', text: 'read every file', ts: TS}];
  for (let i = 0; i < steps; i++) events.push(...step(i));
  return events;
}

/**
 * Replays `events` into a running tab and returns the CPU time (user +
 * system, in ms) the replay took.  The replay is CPU-bound, so CPU time
 * is what the budget bounds; wall-clock time would also count the time
 * the process was descheduled on a loaded machine.
 */
function replayRunning(win, tab, events) {
  send(win, {type: 'status', running: true, tabId: tab, startTs: TS});
  const c0 = process.cpuUsage();
  send(win, {
    type: 'task_events',
    tabId: tab,
    task: 'read every file',
    task_id: 'task-1',
    chat_id: 'chat-1',
    extra: '',
    events,
  });
  const c = process.cpuUsage(c0);
  return (c.user + c.system) / 1000;
}

function testLongReplayFoldsOnceAndFast() {
  const {win} = makePage();
  const tab = win._testApi.getActiveTabId();
  const elapsed = replayRunning(win, tab, longRun(STEPS));

  const ps = panels(win);
  assert.strictEqual(
    ps.length,
    STEPS * 2 + 2,
    'each step renders a thoughts + a tool panel, plus the prompt and ' +
      'the thoughts panel the last result arms',
  );
  // Every older panel is folded: the replay's final collapse pass did
  // the work the per-event pass used to redo hundreds of times.
  const older = ps.slice(0, -2);
  assert.ok(
    older.every(isFoldedOrThoughts),
    'every older tool panel of a replayed running task is folded',
  );
  assert.ok(
    ps.filter(isThoughts).length >= STEPS &&
      !ps.filter(isThoughts).some(isCollapsed),
    'no Thoughts panel of the replay is folded',
  );
  assert.ok(
    elapsed < REPLAY_BUDGET_MS,
    `replaying ${STEPS} steps took ${elapsed} ms; the per-event collapse ` +
      `pass is back (budget ${REPLAY_BUDGET_MS} ms)`,
  );
  win.close();
  console.log(
    `  ok - a ${STEPS}-step replay folds once and fast (${elapsed} ms)`,
  );
}

function testLiveStreamAfterReplayKeepsNewestTwoOpen() {
  const {win} = makePage();
  const tab = win._testApi.getActiveTabId();
  replayRunning(win, tab, longRun(3));
  const before = panels(win).length;

  // The task keeps streaming after the replay: the live collapse pass
  // must run again (the replay flag is cleared) and fold the rest.  The
  // newest two are the last step's tool panel (a tool call starts
  // folded) and the Thoughts panel its result armed (never folded).
  for (const ev of [...step(3), ...step(4)]) send(win, {...ev, tabId: tab});
  const ps = panels(win);
  assert.strictEqual(ps.length, before + 4, 'two more steps, four panels');
  const folded = ps.map(isCollapsed);
  assert.deepStrictEqual(
    folded.slice(-2),
    [true, false],
    'the newest tool panel starts folded, the armed Thoughts panel is open',
  );
  assert.ok(
    ps.slice(0, -2).every(isFoldedOrThoughts),
    'every older tool panel is folded once the stream resumes',
  );
  win.close();
  console.log('  ok - live streaming after a replay keeps folding tool panels');
}

function main() {
  testLongReplayFoldsOnceAndFast();
  testLiveStreamAfterReplayKeepsNewestTwoOpen();
}

main();
