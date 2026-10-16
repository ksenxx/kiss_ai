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
//     open, so the replay-only shortcut does not leak into the stream;
//   * the page's `ready` names no `singleTabId`: it mirrors the whole
//     registry.

/* global require, __dirname, console, process */
'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');
const TS = 1767225600000;
const STEPS = 150;
// The pre-fix replay of STEPS steps took ~15 s here; the fixed one ~0.7 s.
// The budget bounds CPU time (see replayRunning), not wall-clock time.
const REPLAY_BUDGET_MS = 5000;

function makePage() {
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace('{{BODY_CLASS_ATTR}}', ' class="remote-chat"');
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
  win.acquireTransport = function () {
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

function testReadyNamesNoSingleTab() {
  const chat = makePage();
  const ready = chat.posted.filter(m => m.type === 'ready');
  assert.strictEqual(ready.length, 1, 'the page sends one ready');
  assert.strictEqual(
    ready[0].singleTabId,
    undefined,
    'the page mirrors every registry tab and names no single tab',
  );
  chat.win.close();
  console.log('  ok - ready names no singleTabId');
}

function main() {
  testLongReplayFoldsOnceAndFast();
  testLiveStreamAfterReplayKeepsNewestTwoOpen();
  testReadyNamesNoSingleTab();
}

main();
