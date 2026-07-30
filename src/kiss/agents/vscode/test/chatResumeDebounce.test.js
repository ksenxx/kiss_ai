// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end tests for the DEBOUNCED resume-at-bottom contract, in both
// the extension webview and the remote webapp (same main.js, remote-chat
// body class):
//   * rapid, small scroll fluctuations near the bottom (e.g. trackpad
//     jitter) must NOT repeatedly toggle auto-scroll on and off while a
//     task is streaming — touching the bottom mid-jitter must not
//     instantly resume the tail and yank the view;
//   * once the user SETTLES at the bottom (stays there longer than the
//     debounce interval), tailing resumes;
//   * scrolling away from the bottom during the debounce window cancels
//     the pending resume.
// The same contract holds for the outer chat, for wheel gestures at the
// end, and for inner streaming panels (think / bash output).

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

// Must exceed RESUME_AT_BOTTOM_MS (150ms) in media/main.js.
const SETTLE_MS = 300;

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
      '\n//# sourceURL=resumedebounce-main.js',
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

function wheel(win, el, deltaY, deltaX) {
  el.dispatchEvent(
    new win.WheelEvent('wheel', {
      deltaY: deltaY,
      deltaX: deltaX || 0,
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

const label = remote => (remote ? 'remote webapp' : 'extension webview');

// --------------------------------------------------------------------
// Outer chat: scrollbar/touch jitter near the bottom must not toggle
// the tail on and off; settling at the bottom resumes it.
// --------------------------------------------------------------------

async function testOuterJitterDoesNotToggleTail(remote) {
  const {win, posted, scrollCalls} = makeWebview({remote});
  const O = win.document.getElementById('output');
  const geo = {sh: 3000, ch: 500};
  fakeGeometry(O, geo);
  startRunningTask(win, posted);

  send(win, {type: 'system_output', text: 'seed\n'});
  await nextFrames(win);
  const bottom = geo.sh - geo.ch;
  assert.strictEqual(O.scrollTop, bottom, 'sanity: the chat tails to end');

  // Trackpad jitter: tiny up/down fluctuations near the bottom, faster
  // than the debounce interval.  Each return-to-bottom must NOT
  // instantly re-arm the tail.
  userScroll(win, O, bottom - 8);
  userScroll(win, O, bottom);
  userScroll(win, O, bottom - 6);
  userScroll(win, O, bottom);
  userScroll(win, O, bottom); // repeated at-bottom event mid-jitter

  geo.sh += 300;
  const before = scrollCalls.length;
  send(win, {type: 'system_output', text: 'a'.repeat(200) + '\n'});
  await nextFrames(win);

  assert.strictEqual(
    autoScrollsSince(scrollCalls, O, before).length,
    0,
    'BUG (' +
      label(remote) +
      '): a mid-jitter touch of the bottom instantly resumed auto-scroll ' +
      'and yanked the view during streaming',
  );
  assert.strictEqual(
    O.scrollTop,
    bottom,
    'the view must hold its position while the jitter has not settled (' +
      label(remote) +
      ')',
  );

  // The jitter ended AT the bottom: once it settles for longer than the
  // debounce interval, the pending resume fires, re-arms the tail and
  // catches it up — even though streaming moved the bottom meanwhile.
  await sleep(SETTLE_MS);
  assert.strictEqual(
    O.scrollTop,
    geo.sh - geo.ch,
    'the settled jitter (which ended at the bottom) must resume the ' +
      'tail and catch it up (' +
      label(remote) +
      ')',
  );
  geo.sh += 200;
  const before2 = scrollCalls.length;
  send(win, {type: 'system_output', text: 'b'.repeat(200) + '\n'});
  await nextFrames(win);
  assert.ok(
    autoScrollsSince(scrollCalls, O, before2).length > 0,
    'tailing must continue after the jitter settles at the bottom (' +
      label(remote) +
      ')',
  );

  // A deliberate scroll-up DURING the debounce window cancels the
  // pending resume: the user is reading and must not be yanked later.
  userScroll(win, O, geo.sh - geo.ch - 400);
  userScroll(win, O, geo.sh - geo.ch); // touches the bottom...
  userScroll(win, O, geo.sh - geo.ch - 400); // ...but leaves again
  await sleep(SETTLE_MS);
  geo.sh += 100;
  const before3 = scrollCalls.length;
  send(win, {type: 'system_output', text: 'c'.repeat(200) + '\n'});
  await nextFrames(win);
  assert.strictEqual(
    autoScrollsSince(scrollCalls, O, before3).length,
    0,
    'a resume pending from a transient bottom touch must be cancelled ' +
      'when the user scrolls away again (' +
      label(remote) +
      ')',
  );
  assert.strictEqual(
    O.scrollTop,
    geo.sh - geo.ch - 500,
    'the reading position must be held (' + label(remote) + ')',
  );

  // Settling at the bottom again resumes tailing.
  userScroll(win, O, geo.sh - geo.ch);
  await sleep(SETTLE_MS);
  geo.sh += 200;
  const before4 = scrollCalls.length;
  send(win, {type: 'system_output', text: 'd'.repeat(200) + '\n'});
  await nextFrames(win);
  assert.ok(
    autoScrollsSince(scrollCalls, O, before4).length > 0,
    'tailing must resume after the user settles at the bottom (' +
      label(remote) +
      ')',
  );
  assert.strictEqual(
    O.scrollTop,
    geo.sh - geo.ch,
    'the chat must be tailing at the very end again (' + label(remote) + ')',
  );
  win.close();
  console.log(
    '  ok - outer scroll jitter near the bottom does not toggle the ' +
      'tail (' +
      label(remote) +
      ')',
  );
}

// --------------------------------------------------------------------
// Outer chat: alternating tiny wheel deltas at the end (trackpad
// jitter) must not toggle the tail; settling resumes it.
// --------------------------------------------------------------------

async function testWheelJitterAtEndDoesNotToggleTail(remote) {
  const {win, posted, scrollCalls} = makeWebview({remote});
  const O = win.document.getElementById('output');
  const geo = {sh: 3000, ch: 500};
  fakeGeometry(O, geo);
  startRunningTask(win, posted);

  send(win, {type: 'system_output', text: 'seed\n'});
  await nextFrames(win);
  const bottom = geo.sh - geo.ch;
  assert.strictEqual(O.scrollTop, bottom, 'sanity: the chat tails to end');

  // Trackpad wheel jitter at the very end: -3 / +3 / -3 / +3 / +3.
  wheel(win, O, -3);
  wheel(win, O, 3);
  wheel(win, O, -3);
  wheel(win, O, 3);
  wheel(win, O, 3); // repeated wheel-down at end mid-jitter

  geo.sh += 300;
  const before = scrollCalls.length;
  send(win, {type: 'system_output', text: 'a'.repeat(200) + '\n'});
  await nextFrames(win);

  assert.strictEqual(
    autoScrollsSince(scrollCalls, O, before).length,
    0,
    'BUG (' +
      label(remote) +
      '): a mid-jitter wheel-down at the end instantly re-armed the ' +
      'tail and yanked the view during streaming',
  );
  assert.strictEqual(
    O.scrollTop,
    bottom,
    'the view must hold its position during wheel jitter (' +
      label(remote) +
      ')',
  );

  // The wheel jitter ended with a wheel-down at the end: once it
  // settles past the debounce interval the tail re-arms and catches up,
  // even though streaming moved the bottom meanwhile.
  await sleep(SETTLE_MS);
  assert.strictEqual(
    O.scrollTop,
    geo.sh - geo.ch,
    'the settled wheel jitter must resume the tail and catch it up (' +
      label(remote) +
      ')',
  );
  geo.sh += 200;
  const before2 = scrollCalls.length;
  send(win, {type: 'system_output', text: 'b'.repeat(200) + '\n'});
  await nextFrames(win);
  assert.ok(
    autoScrollsSince(scrollCalls, O, before2).length > 0,
    'tailing must resume after the wheel jitter settles at the bottom (' +
      label(remote) +
      ')',
  );
  win.close();
  console.log(
    '  ok - wheel jitter at the end does not toggle the tail (' +
      label(remote) +
      ')',
  );
}

// --------------------------------------------------------------------
// Inner think panel: jitter near the panel bottom must not toggle the
// panel's own tail; settling at its bottom resumes it.
// --------------------------------------------------------------------

async function testThinkPanelJitterDoesNotToggleTail(remote) {
  const {win, posted} = makeWebview({remote});
  const O = win.document.getElementById('output');
  fakeGeometry(O, {sh: 3000, ch: 500});
  startRunningTask(win, posted);

  send(win, {type: 'thinking_start'});
  const think = O.querySelector('.ev.think');
  assert.ok(think, 'thinking_start must create a think panel');
  const gt = {sh: 1000, ch: 200};
  fakeGeometry(think, gt);

  send(win, {type: 'thinking_delta', text: 'a'.repeat(80)});
  await nextFrames(win);
  const bottom = gt.sh - gt.ch;
  assert.ok(
    think.scrollTop >= bottom,
    'sanity: the think panel must tail its own streamed text',
  );

  // Jitter inside the panel, faster than the debounce interval.
  userScroll(win, think, bottom - 7);
  userScroll(win, think, bottom);
  userScroll(win, think, bottom - 5);
  userScroll(win, think, bottom);
  userScroll(win, think, bottom); // repeated at-bottom event mid-jitter

  gt.sh += 200;
  send(win, {type: 'thinking_delta', text: 'b'.repeat(80)});
  await nextFrames(win);
  assert.strictEqual(
    think.scrollTop,
    bottom,
    'BUG (' +
      label(remote) +
      '): a mid-jitter touch of the panel bottom instantly resumed the ' +
      "panel's tail and yanked it during streaming",
  );

  // The jitter ended at the panel's bottom: once it settles past the
  // debounce interval the panel tail re-arms — even though streaming
  // moved the panel's bottom meanwhile.
  await sleep(SETTLE_MS);
  gt.sh += 200;
  send(win, {type: 'thinking_delta', text: 'c'.repeat(80)});
  await nextFrames(win);
  assert.ok(
    think.scrollTop >= gt.sh - gt.ch,
    'the think panel must resume tailing after the jitter settles at ' +
      'its bottom (' +
      label(remote) +
      ')',
  );
  win.close();
  console.log(
    '  ok - think panel jitter near its bottom does not toggle its ' +
      'tail (' +
      label(remote) +
      ')',
  );
}

// --------------------------------------------------------------------
// Inner bash panel: wheel jitter over the panel at its bottom must not
// toggle the panel's tail; settling at its bottom resumes it.
// --------------------------------------------------------------------

async function testBashPanelWheelJitterDoesNotToggleTail(remote) {
  const {win, posted} = makeWebview({remote});
  const O = win.document.getElementById('output');
  const geoO = {sh: 3000, ch: 500};
  fakeGeometry(O, geoO);
  startRunningTask(win, posted);

  send(win, {type: 'tool_call', name: 'Bash', command: 'make -j'});
  const bp = O.querySelector('.bash-panel-content');
  assert.ok(bp, 'a Bash tool_call must create a bash output panel');
  const gb = {sh: 1000, ch: 200};
  fakeGeometry(bp, gb);

  send(win, {type: 'system_output', text: 'a'.repeat(120) + '\n'});
  await nextFrames(win);
  const bottom = gb.sh - gb.ch;
  assert.ok(
    bp.scrollTop >= bottom,
    'sanity: the bash panel must tail its own streamed output',
  );
  O.scrollTop = geoO.sh - geoO.ch;

  // Trackpad wheel jitter over the panel while it sits at its bottom.
  wheel(win, bp, -3);
  wheel(win, bp, 3);
  wheel(win, bp, -3);
  wheel(win, bp, 3);

  gb.sh += 300;
  send(win, {type: 'system_output', text: 'b'.repeat(120) + '\n'});
  await nextFrames(win);
  assert.strictEqual(
    bp.scrollTop,
    bottom,
    'BUG (' +
      label(remote) +
      '): a mid-jitter wheel-down at the panel bottom instantly resumed ' +
      "the panel's tail and yanked it during streaming",
  );

  // The wheel jitter ended with a wheel-down at the panel's bottom:
  // once it settles, its tail re-arms (the outer lock engaged by the
  // wheel-up also releases, since the chat sat at its end).
  await sleep(SETTLE_MS);
  gb.sh += 100;
  send(win, {type: 'system_output', text: 'c'.repeat(120) + '\n'});
  await nextFrames(win);
  assert.ok(
    bp.scrollTop >= gb.sh - gb.ch,
    'the bash panel must resume tailing after the wheel jitter settles ' +
      'at its bottom (' +
      label(remote) +
      ')',
  );
  win.close();
  console.log(
    '  ok - bash panel wheel jitter at its bottom does not toggle its ' +
      'tail (' +
      label(remote) +
      ')',
  );
}

async function runTests() {
  await testOuterJitterDoesNotToggleTail(true);
  await testOuterJitterDoesNotToggleTail(false);
  await testWheelJitterAtEndDoesNotToggleTail(true);
  await testWheelJitterAtEndDoesNotToggleTail(false);
  await testThinkPanelJitterDoesNotToggleTail(true);
  await testThinkPanelJitterDoesNotToggleTail(false);
  await testBashPanelWheelJitterDoesNotToggleTail(true);
  await testBashPanelWheelJitterDoesNotToggleTail(false);
}

runTests()
  .then(() => {
    console.log('\n8 passed, 0 failed');
    process.exit(0);
  })
  .catch(err => {
    console.error('FAIL:', err && err.stack ? err.stack : err);
    process.exit(1);
  });
