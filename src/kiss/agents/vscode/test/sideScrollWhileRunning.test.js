// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end test: horizontal (side) scrolling inside the chat webview
// must keep working while a task is RUNNING.
//
// The bug: while ``isRunning`` is true, every DOM mutation of #output
// (i.e. every streamed token) triggers the auto-scroll paths:
//
//   1. the MutationObserver on ``#output`` fires ``sb()`` which calls
//      ``O.scrollTo({top: O.scrollHeight, behavior: 'instant'})``;
//   2. per-panel auto-scrolls assign ``el.scrollTop = el.scrollHeight``
//      (thinking panel, bash panel, llm panel, prompt body).
//
// A user who scrolls a wide ``pre`` / bash panel sideways has their
// horizontal scroll position fought/reset because:
//
//   (a) ``sb()``'s ``scrollTo({top: …})`` — with no ``left`` — is
//       spec'd to KEEP the current scrollLeft, but the code never
//       guarded against engines/wrappers resetting it, and more
//       importantly the wheel handler only honours VERTICAL user
//       intent: ``if (isRunning && e.deltaY < 0) _scrollLock = true``.
//       Horizontal wheel gestures (deltaX) are ignored entirely, so
//       nothing marks "the user is interacting — stop auto-scroll",
//       and the continuous rAF ``scrollTo`` storm cancels the user's
//       in-progress horizontal pan of #output's descendants on
//       platforms where a programmatic scroll on an ancestor aborts
//       the child's momentum/gesture scrolling.
//
//   (b) each per-panel ``el.scrollTop = el.scrollHeight`` assignment
//       occurs on the SAME element the user side-scrolls (bash panel,
//       think panel). A programmatic scroll on the element aborts the
//       user's in-progress horizontal smooth/momentum scroll on it.
//
// The fix adds horizontal-intent detection: a ``wheel`` event with a
// dominant deltaX while running sets ``_scrollLock`` (suspending both
// ``sb()`` and per-panel auto-scroll on the wheeled panel), exactly
// like an upward vertical wheel already does.  The lock is released by
// the existing "scrolled back to bottom" logic.
//
// This test exercises the real ``media/main.js`` against the real
// ``media/chat.html`` in jsdom, the same harness as
// ``bashHeaderCyan.test.js`` / ``bughunt2_status_timer.test.js``.
//
// Run directly with ``node``:
//
//     node src/kiss/agents/vscode/test/sideScrollWhileRunning.test.js

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

/**
 * Build a jsdom window running the real chat webview (chat.html +
 * panelCopy.js + main.js) with a RECORDING scrollTo implementation so
 * the test can observe every programmatic scroll main.js performs.
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

  // Recording scrollTo: emulate a real engine — apply the requested
  // top and (per the CSSOM spec) keep scrollLeft when ``left`` is
  // omitted from the options dictionary.  Every call is logged so the
  // test can assert whether the auto-scroller fired while the user
  // was side-scrolling.
  const scrollCalls = [];
  function recordingScrollTo(opts) {
    let left;
    let top;
    if (typeof opts === 'object' && opts !== null) {
      left = opts.left;
      top = opts.top;
    } else {
      left = arguments[0];
      top = arguments[1];
    }
    scrollCalls.push({el: this, left, top});
    if (typeof top === 'number') this.scrollTop = top;
    if (typeof left === 'number') this.scrollLeft = left;
  }
  win.Element.prototype.scrollTo = recordingScrollTo;
  win.HTMLElement.prototype.scrollTo = recordingScrollTo;

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
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
  return {win, posted, scrollCalls};
}

/** Dispatch a backend→webview event exactly like the extension does. */
function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

/**
 * jsdom performs no layout, so scrollHeight/clientHeight/scrollWidth/
 * clientWidth are always 0.  Give ``el`` believable scroll geometry:
 * content taller and wider than the viewport box.
 */
function fakeGeometry(el, {sw = 2000, cw = 400, sh = 3000, ch = 500}) {
  Object.defineProperty(el, 'scrollWidth', {value: sw, configurable: true});
  Object.defineProperty(el, 'clientWidth', {value: cw, configurable: true});
  Object.defineProperty(el, 'scrollHeight', {value: sh, configurable: true});
  Object.defineProperty(el, 'clientHeight', {value: ch, configurable: true});
}

/** Wait until pending requestAnimationFrame callbacks have run. */
function nextFrames(win, n = 3) {
  return new Promise(resolve => {
    let left = n;
    function step() {
      if (--left <= 0) return resolve();
      win.requestAnimationFrame(step);
    }
    win.requestAnimationFrame(step);
  });
}

/** Start a running task streaming a wide bash panel; returns the panel. */
function startRunningBashTask(win, posted) {
  const ready = posted.find(m => m.type === 'ready');
  assert.ok(ready && ready.tabId, 'webview must post ready with a tabId');
  // Hide the welcome screen exactly like the extension does when a
  // task starts — sb() refuses to auto-scroll while welcome is shown.
  win._demoApi.hideWelcome();
  send(win, {
    type: 'status',
    running: true,
    tabId: ready.tabId,
    startTs: Date.now() - 2000,
  });
  send(win, {
    type: 'tool_call',
    name: 'Bash',
    command: 'printf "%0.s=" {1..500}',
    description: 'very wide output',
  });
  const panel = win.document.querySelector('#output .bash-panel-content');
  assert.ok(panel, 'streaming a Bash tool_call must create a bash panel');
  return panel;
}

/**
 * The core repro: while running, the user side-scrolls a wide bash
 * panel with a horizontal wheel gesture; more output then streams in.
 * The horizontal wheel MUST engage the auto-scroll lock (main.js
 * ``_scrollLock``) so the rAF ``O.scrollTo`` storm — which aborts the
 * user's in-progress horizontal pan on real engines — stops firing.
 */
async function testHorizontalWheelSuspendsAutoScrollWhileRunning() {
  const {win, posted, scrollCalls} = makeWebview();
  const O = win.document.getElementById('output');
  fakeGeometry(O, {});

  const panel = startRunningBashTask(win, posted);
  fakeGeometry(panel, {sw: 4000, cw: 300, sh: 100, ch: 200});

  // Stream some output, let auto-scroll settle (sanity: it DOES fire
  // while running and unlocked).
  send(win, {type: 'system_output', text: 'x'.repeat(500) + '\n'});
  await nextFrames(win);
  const callsBeforeUserScroll = scrollCalls.filter(
    c => c.el === O && typeof c.top === 'number',
  ).length;
  assert.ok(
    callsBeforeUserScroll > 0,
    'sanity: auto-scroll must be active while running before user input',
  );

  // The user pans SIDEWAYS inside the wide bash panel (trackpad
  // horizontal wheel).  The event bubbles up to #output where main.js
  // listens.
  const wheel = new win.WheelEvent('wheel', {
    deltaX: 40,
    deltaY: 0,
    bubbles: true,
    cancelable: true,
  });
  panel.dispatchEvent(wheel);

  // The user is mid-pan: panel.scrollLeft moved away from 0.
  panel.scrollLeft = 900;
  O.scrollTop = 100; // user is NOT pinned at the bottom

  // More streamed output arrives (MutationObserver fires → sb()).
  const before = scrollCalls.length;
  send(win, {type: 'system_output', text: 'y'.repeat(500) + '\n'});
  await nextFrames(win);

  const autoScrolls = scrollCalls
    .slice(before)
    .filter(c => c.el === O && typeof c.top === 'number');
  assert.strictEqual(
    autoScrolls.length,
    0,
    'BUG: a horizontal wheel gesture while running must engage the ' +
      'auto-scroll lock (like an upward vertical wheel does), but ' +
      'O.scrollTo auto-scroll still fired ' +
      autoScrolls.length +
      ' time(s) — the rAF scroll storm cancels the user\u2019s ' +
      'in-progress side scroll',
  );
  assert.strictEqual(
    panel.scrollLeft,
    900,
    'the user\u2019s horizontal scroll position must be preserved',
  );
  win.close();
  console.log(
    '  ok - horizontal wheel while running suspends the auto-scroller',
  );
}

/**
 * The per-panel auto-scroll (``bashPanel.scrollTop = scrollHeight``)
 * must ALSO stand down while the user's side-scroll lock is engaged —
 * a programmatic scrollTop assignment on the very panel being panned
 * aborts the horizontal gesture just like O.scrollTo does.
 */
async function testBashPanelAutoScrollPausesDuringSidePan() {
  const {win, posted} = makeWebview();
  const O = win.document.getElementById('output');
  fakeGeometry(O, {});

  const panel = startRunningBashTask(win, posted);
  fakeGeometry(panel, {sw: 4000, cw: 300, sh: 1000, ch: 200});

  // Baseline sanity: without user interaction the bash panel tails.
  send(win, {type: 'system_output', text: 'a'.repeat(300) + '\n'});
  await nextFrames(win);
  assert.strictEqual(
    panel.scrollTop,
    panel.scrollHeight,
    'sanity: bash panel must tail its output while unlocked',
  );

  // User side-pans the panel.
  panel.dispatchEvent(
    new win.WheelEvent('wheel', {
      deltaX: -35,
      deltaY: 2, // tiny incidental vertical component of a diagonal pan
      bubbles: true,
      cancelable: true,
    }),
  );
  panel.scrollLeft = 700;
  panel.scrollTop = 100; // user also dragged up inside the panel
  O.scrollTop = 50;

  send(win, {type: 'system_output', text: 'b'.repeat(300) + '\n'});
  await nextFrames(win);

  assert.strictEqual(
    panel.scrollTop,
    100,
    'BUG: bash-panel auto-tail must pause while the user\u2019s ' +
      'side-scroll lock is engaged — the scrollTop assignment aborts ' +
      'the horizontal pan gesture',
  );
  assert.strictEqual(panel.scrollLeft, 700, 'scrollLeft must be preserved');
  win.close();
  console.log(
    '  ok - bash panel tail pauses while the side-scroll lock is engaged',
  );
}

/**
 * A dominant VERTICAL downward wheel must NOT engage the lock (that is
 * the "user wants to follow the tail" gesture) — only horizontal-
 * dominant or upward gestures do.  Guards against over-locking.
 */
async function testDownwardWheelDoesNotEngageLock() {
  const {win, posted, scrollCalls} = makeWebview();
  const O = win.document.getElementById('output');
  fakeGeometry(O, {});
  startRunningBashTask(win, posted);

  O.dispatchEvent(
    new win.WheelEvent('wheel', {
      deltaX: 1, // negligible sideways jitter of a vertical scroll
      deltaY: 60,
      bubbles: true,
      cancelable: true,
    }),
  );

  const before = scrollCalls.length;
  send(win, {type: 'system_output', text: 'z'.repeat(200) + '\n'});
  await nextFrames(win);
  const autoScrolls = scrollCalls
    .slice(before)
    .filter(c => c.el === O && typeof c.top === 'number');
  assert.ok(
    autoScrolls.length > 0,
    'a dominant downward wheel must NOT suspend auto-scroll ' +
      '(deltaX jitter of a vertical scroll must be ignored)',
  );
  win.close();
  console.log('  ok - dominant downward wheel keeps auto-scroll active');
}

/**
 * The lock engaged by a horizontal wheel must release the same way the
 * vertical lock does: when the user scrolls #output back to the
 * bottom, auto-tailing resumes.
 */
async function testLockReleasesWhenUserReturnsToBottom() {
  const {win, posted, scrollCalls} = makeWebview();
  const O = win.document.getElementById('output');
  fakeGeometry(O, {});
  const panel = startRunningBashTask(win, posted);
  fakeGeometry(panel, {sw: 4000, cw: 300, sh: 1000, ch: 200});

  // Engage the lock with a horizontal pan.
  O.dispatchEvent(
    new win.WheelEvent('wheel', {
      deltaX: 50,
      deltaY: 0,
      bubbles: true,
      cancelable: true,
    }),
  );

  // User scrolls #output back to the very bottom → 'scroll' handler
  // releases _scrollLock.
  O.scrollTop = O.scrollHeight - O.clientHeight;
  O.dispatchEvent(new win.Event('scroll'));

  const before = scrollCalls.length;
  send(win, {type: 'system_output', text: 'c'.repeat(200) + '\n'});
  await nextFrames(win);
  const autoScrolls = scrollCalls
    .slice(before)
    .filter(c => c.el === O && typeof c.top === 'number');
  assert.ok(
    autoScrolls.length > 0,
    'auto-scroll must resume after the user returns to the bottom',
  );
  win.close();
  console.log('  ok - side-scroll lock releases at the bottom, tail resumes');
}

async function runTests() {
  await testHorizontalWheelSuspendsAutoScrollWhileRunning();
  await testBashPanelAutoScrollPausesDuringSidePan();
  await testDownwardWheelDoesNotEngageLock();
  await testLockReleasesWhenUserReturnsToBottom();
}

runTests()
  .then(() => {
    console.log('\n4 passed, 0 failed');
    process.exit(0);
  })
  .catch(err => {
    console.error('FAIL:', err && err.stack ? err.stack : err);
    process.exit(1);
  });
