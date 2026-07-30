// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end tests for the chat auto-scroll ON/OFF contract, in both
// the extension webview and the remote webapp (same main.js,
// remote-chat body class):
//   * auto-scroll is ALWAYS on while a task streams — a small scroll
//     back (less than 5 lines from the bottom) must NOT pause it;
//   * a small wheel-up (less than 5 lines) must NOT pause it either,
//     and wheel-up deltas from separate gestures (more than the
//     accumulation window apart) must not add up;
//   * scrolling back at least 5 lines (by scrollbar or by wheel)
//     pauses auto-scroll so the user can read older content;
//   * auto-scroll resumes AS SOON AS the user returns to the bottom —
//     with no settle delay — via scrollbar or via wheel-down at the
//     end;
//   * a sideways pan over wide content still pauses the tail so the
//     horizontal reading position is not yanked.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

// Must exceed WHEEL_UP_WINDOW_MS (500ms) in media/main.js.
const GESTURE_GAP_MS = 600;

function makeWebview(opts) {
  opts = opts || {};
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  if (opts.remote) {
    html = html.replace('{{BODY_CLASS_ATTR}}', ' class="remote-chat"');
  }
  html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
  html = html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');

  const dom = new JSDOM(html, {
    runScripts: 'dangerously',
    pretendToBeVisual: true,
    url: 'https://localhost/',
  });
  const win = dom.window;
  win.Element.prototype.scrollIntoView = function () {};

  const scrollCalls = [];
  function engineScrollTo(arg1, arg2) {
    let top;
    if (typeof arg1 === 'object' && arg1 !== null) top = arg1.top;
    else top = arg2;
    scrollCalls.push({el: this, top});
    if (typeof top !== 'number') return;
    const max = Math.max(0, this.scrollHeight - this.clientHeight);
    const clamped = Math.min(Math.max(top, 0), max);
    if (clamped === this.scrollTop) return;
    this.scrollTop = clamped;
    this.dispatchEvent(new win.Event('scroll'));
  }
  win.Element.prototype.scrollTo = engineScrollTo;
  win.HTMLElement.prototype.scrollTo = engineScrollTo;

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
      '\n//# sourceURL=fivelines-main.js',
  );
  return {win, posted, scrollCalls};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function fakeGeometry(el, geo) {
  Object.defineProperty(el, 'scrollHeight', {
    get: () => geo.sh,
    configurable: true,
  });
  Object.defineProperty(el, 'clientHeight', {
    get: () => geo.ch,
    configurable: true,
  });
  Object.defineProperty(el, 'scrollWidth', {
    get: () => geo.sw || 800,
    configurable: true,
  });
  Object.defineProperty(el, 'clientWidth', {
    get: () => geo.cw || 400,
    configurable: true,
  });
}

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

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function userScroll(win, el, top) {
  el.scrollTop = top;
  el.dispatchEvent(new win.Event('scroll'));
}

function wheel(win, el, deltaY, deltaX, deltaMode) {
  el.dispatchEvent(
    new win.WheelEvent('wheel', {
      deltaY: deltaY,
      deltaX: deltaX || 0,
      deltaMode: deltaMode || 0,
      bubbles: true,
      cancelable: true,
    }),
  );
}

function startRunningTask(win, posted) {
  const ready = posted.find(m => m.type === 'ready');
  assert.ok(ready && ready.tabId, 'webview must post ready with a tabId');
  win._demoApi.hideWelcome();
  send(win, {
    type: 'status',
    running: true,
    tabId: ready.tabId,
    startTs: Date.now() - 2000,
  });
  return ready.tabId;
}

function autoScrollsSince(scrollCalls, el, from) {
  return scrollCalls
    .slice(from)
    .filter(c => c.el === el && typeof c.top === 'number');
}

// One "line" exactly as media/main.js computes it: the chat output's
// computed line-height; when it is not a length ('normal'), ~1.2x the
// computed font size; last-resort fallback 19px.
function linePx(win, O) {
  const cs = win.getComputedStyle(O);
  let lh = parseFloat(cs.lineHeight);
  if (isFinite(lh) && lh > 0) return lh;
  lh = 1.2 * parseFloat(cs.fontSize);
  return isFinite(lh) && lh > 0 ? lh : 19;
}

const label = remote => (remote ? 'remote webapp' : 'extension webview');

async function makeTailingChat(remote) {
  const ctx = makeWebview({remote});
  ctx.O = ctx.win.document.getElementById('output');
  ctx.geo = {sh: 3000, ch: 500};
  fakeGeometry(ctx.O, ctx.geo);
  startRunningTask(ctx.win, ctx.posted);
  send(ctx.win, {type: 'system_output', text: 'seed\n'});
  await nextFrames(ctx.win);
  assert.strictEqual(
    ctx.O.scrollTop,
    ctx.geo.sh - ctx.geo.ch,
    'sanity: the chat tails to the end (' + label(remote) + ')',
  );
  ctx.LINE = linePx(ctx.win, ctx.O);
  return ctx;
}

function assertTails(ctx, remote, what) {
  ctx.geo.sh += 300;
  const before = ctx.scrollCalls.length;
  send(ctx.win, {type: 'system_output', text: 'x'.repeat(200) + '\n'});
  return nextFrames(ctx.win).then(() => {
    assert.ok(
      autoScrollsSince(ctx.scrollCalls, ctx.O, before).length > 0,
      'BUG (' + label(remote) + '): ' + what + ' — auto-scroll went off',
    );
    assert.strictEqual(
      ctx.O.scrollTop,
      ctx.geo.sh - ctx.geo.ch,
      'BUG (' + label(remote) + '): ' + what + ' — not tailing at the end',
    );
  });
}

function assertHolds(ctx, remote, holdTop, what) {
  ctx.geo.sh += 300;
  const before = ctx.scrollCalls.length;
  send(ctx.win, {type: 'system_output', text: 'y'.repeat(200) + '\n'});
  return nextFrames(ctx.win).then(() => {
    assert.strictEqual(
      autoScrollsSince(ctx.scrollCalls, ctx.O, before).length,
      0,
      'BUG (' + label(remote) + '): ' + what + ' — auto-scroll stayed on',
    );
    assert.strictEqual(
      ctx.O.scrollTop,
      holdTop,
      'BUG (' + label(remote) + '): ' + what + ' — reading position lost',
    );
  });
}

// --------------------------------------------------------------------
// A small scroll back (< 5 lines from the bottom) must NOT pause
// auto-scroll: the tail keeps running.
// --------------------------------------------------------------------

async function testSmallScrollBackKeepsTailing(remote) {
  const ctx = await makeTailingChat(remote);
  const {win, O, geo, LINE} = ctx;

  // Scroll back only 2 lines: far less than the 5-line pause threshold.
  userScroll(win, O, geo.sh - geo.ch - 2 * LINE);

  await assertTails(
    ctx,
    remote,
    'a scroll back of 2 lines (< 5) paused auto-scroll',
  );
  win.close();
  console.log(
    '  ok - a small scroll back (< 5 lines) keeps auto-scroll on (' +
      label(remote) +
      ')',
  );
}

// --------------------------------------------------------------------
// A small wheel-up (< 5 lines) must NOT pause auto-scroll, and two
// small wheel-ups from SEPARATE gestures (more than the accumulation
// window apart) must not add up to a pause.
// --------------------------------------------------------------------

async function testSmallWheelUpKeepsTailing(remote) {
  const ctx = await makeTailingChat(remote);
  const {win, O, LINE} = ctx;

  // A single 3-line wheel-up: below the 5-line threshold.
  wheel(win, O, -3 * LINE);
  await assertTails(
    ctx,
    remote,
    'a wheel-up of 3 lines (< 5) paused auto-scroll',
  );

  // A second 3-line wheel-up after the gesture window: 3 + 3 = 6 lines
  // in total, but separate gestures must not accumulate into a pause.
  await sleep(GESTURE_GAP_MS);
  wheel(win, O, -3 * LINE);
  await assertTails(
    ctx,
    remote,
    'two small wheel-ups from separate gestures accumulated into a pause',
  );
  win.close();
  console.log(
    '  ok - small wheel-ups (< 5 lines each, separate gestures) keep ' +
      'auto-scroll on (' +
      label(remote) +
      ')',
  );
}

// --------------------------------------------------------------------
// Scrolling back at least 5 lines with the scrollbar pauses the tail;
// returning to the bottom resumes it IMMEDIATELY (no settle delay).
// --------------------------------------------------------------------

async function testScrollBackFiveLinesPausesAndBottomResumes(remote) {
  const ctx = await makeTailingChat(remote);
  const {win, O, geo, LINE} = ctx;

  // Scroll back exactly 5 lines: the pause threshold is reached.
  const readTop = geo.sh - geo.ch - 5 * LINE;
  userScroll(win, O, readTop);
  await assertHolds(
    ctx,
    remote,
    readTop,
    'a scroll back of 5 lines did not pause auto-scroll',
  );

  // Scroll straight back to the (new) bottom: auto-scroll must resume
  // as soon as the bottom is reached — the very next stream event
  // tails, with NO settle delay.
  userScroll(win, O, geo.sh - geo.ch);
  await assertTails(
    ctx,
    remote,
    'auto-scroll did not resume as soon as the user reached the bottom',
  );
  win.close();
  console.log(
    '  ok - a 5-line scroll back pauses; reaching the bottom resumes ' +
      'instantly (' +
      label(remote) +
      ')',
  );
}

// --------------------------------------------------------------------
// A wheel-up gesture totalling at least 5 lines pauses the tail even
// though streaming keeps snapping the position back down between the
// wheel events; a wheel-down at the end resumes it immediately.
// --------------------------------------------------------------------

async function testWheelUpFiveLinesPausesAndWheelDownResumes(remote) {
  const ctx = await makeTailingChat(remote);
  const {win, O, geo, LINE} = ctx;

  // One quick gesture: two wheel-ups of 3 lines each (6 lines total).
  // The position never moves (streaming snaps it back), but the intent
  // adds up past the threshold and pauses the tail.
  const holdTop = geo.sh - geo.ch;
  wheel(win, O, -3 * LINE);
  wheel(win, O, -3 * LINE);
  await assertHolds(
    ctx,
    remote,
    holdTop,
    'a 6-line wheel-up gesture did not pause auto-scroll',
  );

  // The user is now 300px above the grown bottom: wheeling further up
  // while already at least 5 lines away must keep the pause engaged.
  wheel(win, O, -1);
  await assertHolds(
    ctx,
    remote,
    holdTop,
    'a wheel-up while paused far from the bottom re-enabled the tail',
  );

  // Return to the bottom, pause again with a big wheel-up, then wheel
  // down at the end: the tail must come back immediately.
  userScroll(win, O, geo.sh - geo.ch);
  wheel(win, O, -6 * LINE);
  wheel(win, O, 2);
  await assertTails(
    ctx,
    remote,
    'a wheel-down at the end did not resume auto-scroll immediately',
  );
  win.close();
  console.log(
    '  ok - a 5-line wheel-up pauses; wheel-down at the end resumes ' +
      'instantly (' +
      label(remote) +
      ')',
  );
}

// --------------------------------------------------------------------
// A sideways pan over wide content pauses the tail (the horizontal
// reading position must not be yanked); the bottom resumes it.
// --------------------------------------------------------------------

async function testSidePanPausesTail(remote) {
  const ctx = await makeTailingChat(remote);
  const {win, O, geo} = ctx;

  const holdTop = geo.sh - geo.ch;
  wheel(win, O, 0, 40);
  await assertHolds(
    ctx,
    remote,
    holdTop,
    'a sideways pan did not pause auto-scroll',
  );

  userScroll(win, O, geo.sh - geo.ch);
  await assertTails(
    ctx,
    remote,
    'auto-scroll did not resume at the bottom after a side pan',
  );
  win.close();
  console.log(
    '  ok - a sideways pan pauses the tail; the bottom resumes it (' +
      label(remote) +
      ')',
  );
}

// --------------------------------------------------------------------
// Wheels that report their deltas in LINES or PAGES (deltaMode 1 / 2,
// e.g. Firefox running the remote webapp) must honor the same 5-line
// contract: a small line-mode wheel-up keeps tailing, a 5-line one
// pauses, and a one-page wheel-up pauses.
// --------------------------------------------------------------------

async function testLineAndPageModeWheels(remote) {
  const ctx = await makeTailingChat(remote);
  const {win, O, geo} = ctx;

  // 2 lines up in line-mode: below the threshold, keeps tailing.
  wheel(win, O, -2, 0, 1);
  await assertTails(
    ctx,
    remote,
    'a 2-line line-mode wheel-up (< 5) paused auto-scroll',
  );

  // A quick gesture totalling 6 more lines in line-mode: pauses.
  const holdTop = geo.sh - geo.ch;
  wheel(win, O, -3, 0, 1);
  wheel(win, O, -3, 0, 1);
  await assertHolds(
    ctx,
    remote,
    holdTop,
    'a 6-line line-mode wheel gesture did not pause auto-scroll',
  );

  // Back to the bottom: resumes; then a one-page wheel-up (a full
  // viewport is far more than 5 lines) pauses again.
  userScroll(win, O, geo.sh - geo.ch);
  await assertTails(
    ctx,
    remote,
    'auto-scroll did not resume at the bottom after a line-mode pause',
  );
  const holdTop2 = geo.sh - geo.ch;
  wheel(win, O, -1, 0, 2);
  await assertHolds(
    ctx,
    remote,
    holdTop2,
    'a one-page wheel-up did not pause auto-scroll',
  );
  win.close();
  console.log(
    '  ok - line-mode and page-mode wheels honor the 5-line contract (' +
      label(remote) +
      ')',
  );
}

async function runTests() {
  await testSmallScrollBackKeepsTailing(true);
  await testSmallScrollBackKeepsTailing(false);
  await testSmallWheelUpKeepsTailing(true);
  await testSmallWheelUpKeepsTailing(false);
  await testScrollBackFiveLinesPausesAndBottomResumes(true);
  await testScrollBackFiveLinesPausesAndBottomResumes(false);
  await testWheelUpFiveLinesPausesAndWheelDownResumes(true);
  await testWheelUpFiveLinesPausesAndWheelDownResumes(false);
  await testSidePanPausesTail(true);
  await testSidePanPausesTail(false);
  await testLineAndPageModeWheels(true);
  await testLineAndPageModeWheels(false);
}

runTests()
  .then(() => {
    console.log('\n12 passed, 0 failed');
    process.exit(0);
  })
  .catch(err => {
    console.error('FAIL:', err && err.stack ? err.stack : err);
    process.exit(1);
  });
