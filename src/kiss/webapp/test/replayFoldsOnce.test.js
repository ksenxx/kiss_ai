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
//   * a 150-step running-task replay (302 panels) writes each panel's
//     preview at most once (the quadratic pass wrote it once per event)
//     and ends with every older panel folded, exactly as before;
//   * live streaming after the replay folds older panels again (a
//     finished fan-out folds once two newer panels follow it), so the
//     replay-only shortcut does not leak into the stream.

/* global require, console */
'use strict';

const assert = require('assert');
const {makePage: makeChatPage, send, stubRaf} = require('./simplify2_harness');

const TS = 1767225600000;
const STEPS = 150;

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
 * Replays `events` into a running tab and returns how many times the
 * replay wrote a folded panel's header preview.  Folding a panel
 * rewrites its `.collapse-preview` text from the panel's DOM; the
 * pre-fix per-event collapse pass redid that for every older panel on
 * every replayed event (O(events x panels), ~15 s for STEPS steps), so
 * the number of preview writes is the work the test bounds, not a
 * timing that depends on the machine's load.
 */
function replayRunning(win, tab, events) {
  send(win, {type: 'status', running: true, tabId: tab, startTs: TS});
  const out = win.document.getElementById('output');
  const observer = new win.MutationObserver(() => {});
  observer.observe(out, {childList: true, characterData: true, subtree: true});
  send(win, {
    type: 'task_events',
    tabId: tab,
    task: 'read every file',
    task_id: 'task-1',
    chat_id: 'chat-1',
    extra: '',
    events,
  });
  const previewWrites = observer
    .takeRecords()
    .filter(
      r => r.target.classList && r.target.classList.contains('collapse-preview'),
    ).length;
  observer.disconnect();
  return previewWrites;
}

function testLongReplayFoldsOnceAndFast() {
  const {win} = makePage();
  const tab = win._testApi.getActiveTabId();
  const previewWrites = replayRunning(win, tab, longRun(STEPS));

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
    previewWrites <= ps.length,
    `replaying ${STEPS} steps wrote ${previewWrites} panel previews for ` +
      `${ps.length} panels; the per-event collapse pass is back`,
  );
  win.close();
  console.log(
    `  ok - a ${STEPS}-step replay folds once (${previewWrites} preview writes)`,
  );
}

function testLiveStreamAfterReplayFoldsAgain() {
  const {win} = makePage();
  const tab = win._testApi.getActiveTabId();
  replayRunning(win, tab, longRun(3));
  const before = panels(win).length;

  // The task keeps streaming after the replay.  A finished fan-out with
  // no sub-agents starts open, and only the live collapse pass folds it,
  // once two newer panels follow it: that pass must run again (the
  // replay flag is cleared).  Read and Bash panels start folded and
  // Thoughts panels never fold, so they cannot show this.
  send(win, {type: 'tool_call', name: 'run_parallel', ts: TS, tabId: tab});
  send(win, {type: 'tool_result', content: 'no tasks', ts: TS, tabId: tab});
  const fanout = win.document.querySelector('#output > .tc-run-parallel');
  assert.ok(fanout && !isCollapsed(fanout), 'a fan-out panel starts open');
  for (const ev of [...step(3), ...step(4)]) send(win, {...ev, tabId: tab});
  assert.ok(
    isCollapsed(fanout),
    'the live collapse pass folds the fan-out once newer panels follow',
  );
  const ps = panels(win);
  assert.strictEqual(
    ps.length,
    before + 5,
    'the fan-out plus two more steps add five panels',
  );
  assert.deepStrictEqual(
    ps.slice(-2).map(isCollapsed),
    [true, false],
    'the newest tool panel starts folded, the armed Thoughts panel is open',
  );
  assert.ok(
    ps.slice(0, -2).every(isFoldedOrThoughts),
    'every older tool panel is folded once the stream resumes',
  );
  win.close();
  console.log('  ok - live streaming after a replay folds older panels again');
}

function main() {
  testLongReplayFoldsOnceAndFast();
  testLiveStreamAfterReplayFoldsAgain();
}

main();
