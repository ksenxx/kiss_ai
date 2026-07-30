// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end tests for the chat "follow the tail" contract, in both the
// extension webview and the remote webapp (same main.js, remote-chat body
// class):
//   * the chat always scrolls to the end as events and texts stream in;
//   * if the user scrolls up (in the chat or INSIDE a streaming panel),
//     nothing yanks them back down on the next event/text;
//   * once the user scrolls back down to the bottom, tailing resumes.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

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
      '\n//# sourceURL=followtail-main.js',
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

// Wait past the resume-at-bottom debounce (RESUME_AT_BOTTOM_MS in
// media/main.js): a return to the bottom only resumes the tail after
// the position has settled there, so jitter cannot toggle the tail.
function settle() {
  return new Promise(resolve => setTimeout(resolve, 250));
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
// Outer chat view: wheel-up over content that cannot scroll must not
// permanently disable tailing.
// --------------------------------------------------------------------

async function testWheelUpOnUnscrollableOutputKeepsTailing(remote) {
  const {win, posted, scrollCalls} = makeWebview({remote});
  const O = win.document.getElementById('output');
  const geo = {sh: 400, ch: 500};
  fakeGeometry(O, geo);
  startRunningTask(win, posted);
  send(win, {type: 'tool_call', name: 'Bash', command: 'make'});

  // The content fits in the viewport: a wheel-up cannot scroll anything
  // and no scroll event will ever fire, so it must NOT lock the tail.
  wheel(win, O, -30);

  geo.sh = 3000;
  const before = scrollCalls.length;
  send(win, {type: 'system_output', text: 'a'.repeat(200) + '\n'});
  await nextFrames(win);

  assert.ok(
    autoScrollsSince(scrollCalls, O, before).length > 0,
    'BUG (' +
      label(remote) +
      '): a wheel-up over unscrollable content permanently disabled ' +
      'auto-scroll although the user never left the bottom',
  );
  assert.strictEqual(
    O.scrollTop,
    geo.sh - geo.ch,
    'the chat must be tailing at the very end (' + label(remote) + ')',
  );
  win.close();
  console.log(
    '  ok - wheel-up on unscrollable content keeps tailing (' +
      label(remote) +
      ')',
  );
}

async function testWheelUpOnScrollableOutputStillSuspends() {
  const {win, posted, scrollCalls} = makeWebview({remote: true});
  const O = win.document.getElementById('output');
  const geo = {sh: 3000, ch: 500};
  fakeGeometry(O, geo);
  startRunningTask(win, posted);
  send(win, {type: 'tool_call', name: 'Bash', command: 'make'});
  O.scrollTop = geo.sh - geo.ch;

  // A deliberate wheel-up of well over 5 lines (the auto-scroll pause
  // threshold in media/main.js) must suspend the tail.
  wheel(win, O, -500);

  geo.sh += 200;
  const before = scrollCalls.length;
  send(win, {type: 'system_output', text: 'b'.repeat(200) + '\n'});
  await nextFrames(win);

  assert.strictEqual(
    autoScrollsSince(scrollCalls, O, before).length,
    0,
    'a wheel-up over scrollable content must engage the lock before ' +
      'the scroll event arrives',
  );

  // ...and settling back at the bottom must resume tailing.
  userScroll(win, O, geo.sh - geo.ch);
  await settle();
  geo.sh += 200;
  const before2 = scrollCalls.length;
  send(win, {type: 'system_output', text: 'c'.repeat(200) + '\n'});
  await nextFrames(win);
  assert.ok(
    autoScrollsSince(scrollCalls, O, before2).length > 0,
    'tailing must resume when the user scrolls back to the bottom',
  );
  win.close();
  console.log('  ok - wheel-up on scrollable content still suspends + resumes');
}

// --------------------------------------------------------------------
// Inner streaming panels: thinking, bash output, thoughts (llm) panel,
// prompt bodies.  Scrolling up INSIDE one of them must stop that
// panel's tailing; returning to its bottom must resume it.
// --------------------------------------------------------------------

async function testThinkPanelHonorsUserScroll(remote) {
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
  assert.ok(
    think.scrollTop >= gt.sh - gt.ch,
    'sanity: the think panel must tail its own streamed text',
  );

  userScroll(win, think, 50);
  gt.sh += 200;
  send(win, {type: 'thinking_delta', text: 'b'.repeat(80)});
  await nextFrames(win);
  assert.strictEqual(
    think.scrollTop,
    50,
    'BUG (' +
      label(remote) +
      '): the think panel yanked the user back to its bottom although ' +
      'the user had scrolled up inside it',
  );

  userScroll(win, think, gt.sh - gt.ch);
  await settle();
  gt.sh += 200;
  send(win, {type: 'thinking_delta', text: 'c'.repeat(80)});
  await nextFrames(win);
  assert.ok(
    think.scrollTop >= gt.sh - gt.ch,
    'the think panel must resume tailing once the user returns to ' +
      'its bottom',
  );
  win.close();
  console.log(
    '  ok - think panel honors inner scroll-up and resumes (' +
      label(remote) +
      ')',
  );
}

async function testBashPanelHonorsUserScroll(remote) {
  const {win, posted} = makeWebview({remote});
  const O = win.document.getElementById('output');
  fakeGeometry(O, {sh: 3000, ch: 500});
  startRunningTask(win, posted);

  send(win, {type: 'tool_call', name: 'Bash', command: 'make -j'});
  const bp = O.querySelector('.bash-panel-content');
  assert.ok(bp, 'a Bash tool_call must create a bash output panel');
  const gb = {sh: 1000, ch: 200};
  fakeGeometry(bp, gb);

  send(win, {type: 'system_output', text: 'a'.repeat(120) + '\n'});
  await nextFrames(win);
  assert.ok(
    bp.scrollTop >= gb.sh - gb.ch,
    'sanity: the bash panel must tail its own streamed output',
  );

  userScroll(win, bp, 40);
  gb.sh += 300;
  send(win, {type: 'system_output', text: 'b'.repeat(120) + '\n'});
  await nextFrames(win);
  assert.strictEqual(
    bp.scrollTop,
    40,
    'BUG (' +
      label(remote) +
      '): the bash panel yanked the user back to its bottom although ' +
      'the user had scrolled up inside it',
  );

  userScroll(win, bp, gb.sh - gb.ch);
  await settle();
  gb.sh += 100;
  send(win, {type: 'system_output', text: 'c'.repeat(120) + '\n'});
  await nextFrames(win);
  assert.ok(
    bp.scrollTop >= gb.sh - gb.ch,
    'the bash panel must resume tailing once the user returns to ' +
      'its bottom',
  );
  win.close();
  console.log(
    '  ok - bash panel honors inner scroll-up and resumes (' +
      label(remote) +
      ')',
  );
}

async function testThoughtsPanelHonorsUserScroll() {
  const {win, posted} = makeWebview({remote: true});
  const O = win.document.getElementById('output');
  fakeGeometry(O, {sh: 3000, ch: 500});
  startRunningTask(win, posted);

  send(win, {type: 'text_delta', text: 'hello '});
  const lp = O.querySelector('.llm-panel');
  assert.ok(lp, 'streaming text must create a thoughts (llm) panel');
  const gl = {sh: 1000, ch: 300};
  fakeGeometry(lp, gl);

  send(win, {type: 'text_delta', text: 'world '});
  await nextFrames(win);
  assert.ok(
    lp.scrollTop >= gl.sh - gl.ch,
    'sanity: the thoughts panel must tail its own streamed text',
  );

  userScroll(win, lp, 60);
  gl.sh += 200;
  send(win, {type: 'text_delta', text: 'more text '});
  await nextFrames(win);
  assert.strictEqual(
    lp.scrollTop,
    60,
    'BUG: the thoughts panel yanked the user back to its bottom ' +
      'although the user had scrolled up inside it',
  );

  userScroll(win, lp, gl.sh - gl.ch);
  await settle();
  gl.sh += 200;
  send(win, {type: 'text_delta', text: 'even more '});
  await nextFrames(win);
  assert.ok(
    lp.scrollTop >= gl.sh - gl.ch,
    'the thoughts panel must resume tailing once the user returns ' +
      'to its bottom',
  );
  win.close();
  console.log('  ok - thoughts panel honors inner scroll-up and resumes');
}

async function testPanelAutoScrollDoesNotMarkUserScroll() {
  const {win, posted} = makeWebview({remote: true});
  const O = win.document.getElementById('output');
  fakeGeometry(O, {sh: 3000, ch: 500});
  startRunningTask(win, posted);

  send(win, {type: 'thinking_start'});
  const think = O.querySelector('.ev.think');
  const gt = {sh: 1000, ch: 200};
  fakeGeometry(think, gt);

  // Two consecutive streamed chunks: the scroll event produced by the
  // panel's own auto-scroll must not be mistaken for the user
  // scrolling and stop the tail.
  send(win, {type: 'thinking_delta', text: 'a'.repeat(80)});
  await nextFrames(win);
  gt.sh += 200;
  send(win, {type: 'thinking_delta', text: 'b'.repeat(80)});
  await nextFrames(win);
  assert.ok(
    think.scrollTop >= gt.sh - gt.ch,
    'the panel\u2019s own programmatic scroll must not suspend its tail',
  );
  win.close();
  console.log('  ok - a panel\u2019s own auto-scroll does not stop its tail');
}

async function testOuterScrollLockPausesInnerPanels() {
  const {win, posted} = makeWebview({remote: true});
  const O = win.document.getElementById('output');
  const geo = {sh: 3000, ch: 500};
  fakeGeometry(O, geo);
  startRunningTask(win, posted);

  send(win, {type: 'thinking_start'});
  const think = O.querySelector('.ev.think');
  const gt = {sh: 1000, ch: 200};
  fakeGeometry(think, gt);
  send(win, {type: 'thinking_delta', text: 'a'.repeat(80)});
  await nextFrames(win);

  userScroll(win, O, 100); // reading older content: global lock
  const held = think.scrollTop;
  gt.sh += 200;
  send(win, {type: 'thinking_delta', text: 'b'.repeat(80)});
  await nextFrames(win);
  assert.strictEqual(
    think.scrollTop,
    held,
    'inner panels must not tail while the user reads older chat content',
  );
  win.close();
  console.log('  ok - outer scroll lock pauses inner panel tails');
}

async function testPromptRerenderPreservesUserScroll() {
  const {win, posted} = makeWebview({remote: true});
  const O = win.document.getElementById('output');
  fakeGeometry(O, {sh: 3000, ch: 500});
  startRunningTask(win, posted);

  // A fresh prompt panel (no early placeholder) simply renders.
  send(win, {type: 'prompt', text: 'plain prompt'});
  let bodies = O.querySelectorAll('.prompt-body');
  assert.strictEqual(bodies.length, 1, 'fresh prompt panel rendered');

  // An early prompt placeholder that the user scrolls inside...
  send(win, {type: 'system_prompt', text: 'sys v1', early: true});
  const sysBody1 = O.querySelector('.system-prompt-body');
  assert.ok(sysBody1, 'early system prompt rendered');
  fakeGeometry(sysBody1, {sh: 600, ch: 150});
  userScroll(win, sysBody1, 30);

  // ...must keep the user's reading position when the final event
  // re-renders the same panel.
  send(win, {type: 'system_prompt', text: 'sys v1 final'});
  const sysBody2 = O.querySelector('.system-prompt-body');
  assert.ok(sysBody2, 're-rendered system prompt body exists');
  assert.notStrictEqual(sysBody2, sysBody1, 'body was re-rendered');
  assert.strictEqual(
    sysBody2.scrollTop,
    30,
    'BUG: re-rendering a prompt panel discarded the user\u2019s ' +
      'scroll position inside it',
  );

  // An untouched early prompt keeps auto-scrolling to its end.
  send(win, {type: 'prompt', text: 'p2', early: true});
  bodies = O.querySelectorAll('.prompt-body');
  const p2Body1 = bodies[bodies.length - 1];
  send(win, {type: 'prompt', text: 'p2 final'});
  bodies = O.querySelectorAll('.prompt-body');
  const p2Body2 = bodies[bodies.length - 1];
  assert.notStrictEqual(p2Body2, p2Body1, 'second panel re-rendered');
  assert.strictEqual(
    p2Body2.scrollTop,
    Math.max(0, p2Body2.scrollHeight - p2Body2.clientHeight),
    'an untouched prompt panel still ends scrolled to its bottom',
  );
  win.close();
  console.log('  ok - prompt re-render preserves the user\u2019s position');
}

async function testPanelWithoutScrollToStillTails() {
  const {win, posted} = makeWebview({remote: true});
  const O = win.document.getElementById('output');
  fakeGeometry(O, {sh: 3000, ch: 500});
  startRunningTask(win, posted);

  send(win, {type: 'tool_call', name: 'Bash', command: 'ls'});
  const bp = O.querySelector('.bash-panel-content');
  const gb = {sh: 1000, ch: 200};
  fakeGeometry(bp, gb);
  bp.scrollTo = undefined; // engines without Element.scrollTo

  send(win, {type: 'system_output', text: 'x'.repeat(120) + '\n'});
  await nextFrames(win);
  assert.strictEqual(
    bp.scrollTop,
    gb.sh,
    'the bash panel must still tail via the scrollTop fallback',
  );
  win.close();
  console.log('  ok - panels tail via the scrollTop fallback');
}

async function testNestedWheelPauseAndResume() {
  const {win, posted, scrollCalls} = makeWebview({remote: true});
  const O = win.document.getElementById('output');
  const go = {sh: 3000, ch: 500};
  fakeGeometry(O, go);
  startRunningTask(win, posted);

  send(win, {type: 'thinking_start'});
  const think = O.querySelector('.ev.think');
  const gt = {sh: 1000, ch: 200};
  fakeGeometry(think, gt);
  send(win, {type: 'thinking_delta', text: 'a'.repeat(80)});
  await nextFrames(win);
  assert.ok(O.scrollTop >= go.sh - go.ch, 'sanity: chat tails at the end');

  // A real wheel-up INSIDE the panel bubbles up to the chat output;
  // over 5 lines' worth, it also engages the chat lock.
  wheel(win, think, -500);
  userScroll(win, think, 50);
  gt.sh += 200;
  send(win, {type: 'thinking_delta', text: 'b'.repeat(80)});
  await nextFrames(win);
  assert.strictEqual(think.scrollTop, 50, 'panel paused after wheel-up');

  // Returning the panel to its bottom with wheel-downs must release
  // BOTH the panel pause and the chat lock the bubbled wheel set.
  think.scrollTop = gt.sh - gt.ch; // wheel moved the panel...
  wheel(win, think, 30); // ...and keeps going at its bottom
  await settle();
  gt.sh += 200;
  const before = scrollCalls.length;
  send(win, {type: 'thinking_delta', text: 'c'.repeat(80)});
  await nextFrames(win);
  assert.ok(
    think.scrollTop >= gt.sh - gt.ch,
    'BUG: wheeling a nested panel back to its bottom must resume its ' +
      'tail; got ' + think.scrollTop + ' vs ' + (gt.sh - gt.ch),
  );
  assert.ok(
    autoScrollsSince(scrollCalls, O, before).length > 0,
    'BUG: the chat lock engaged by a wheel inside a nested panel must ' +
      'release when the user wheels down at the end',
  );
  win.close();
  console.log('  ok - nested wheel-up pauses and wheel-down resumes');
}

async function testSidePanOverPanelPausesItsTail() {
  const {win, posted} = makeWebview({remote: true});
  const O = win.document.getElementById('output');
  fakeGeometry(O, {sh: 400, ch: 500}); // outer chat cannot scroll
  startRunningTask(win, posted);

  send(win, {type: 'tool_call', name: 'Bash', command: 'wide output'});
  const bp = O.querySelector('.bash-panel-content');
  const gb = {sh: 1000, ch: 200, sw: 4000, cw: 300};
  fakeGeometry(bp, gb);
  send(win, {type: 'system_output', text: 'a'.repeat(120) + '\n'});
  await nextFrames(win);
  assert.ok(bp.scrollTop >= gb.sh - gb.ch, 'sanity: bash panel tails');

  // Horizontal pan over the panel: its tail must pause even though
  // the outer chat is unscrollable (so no global lock engages).
  wheel(win, bp, 0, 40);
  bp.scrollTop = 100;
  gb.sh += 300;
  send(win, {type: 'system_output', text: 'b'.repeat(120) + '\n'});
  await nextFrames(win);
  assert.strictEqual(
    bp.scrollTop,
    100,
    'BUG: a horizontal pan inside the bash panel must pause its tail',
  );

  userScroll(win, bp, gb.sh - gb.ch);
  await settle();
  gb.sh += 100;
  send(win, {type: 'system_output', text: 'c'.repeat(120) + '\n'});
  await nextFrames(win);
  assert.ok(
    bp.scrollTop >= gb.sh - gb.ch,
    'the bash panel resumes tailing at its bottom after a side pan',
  );
  win.close();
  console.log('  ok - side pan over a panel pauses its tail and resumes');
}

async function runTests() {
  await testWheelUpOnUnscrollableOutputKeepsTailing(true);
  await testWheelUpOnUnscrollableOutputKeepsTailing(false);
  await testWheelUpOnScrollableOutputStillSuspends();
  await testThinkPanelHonorsUserScroll(true);
  await testThinkPanelHonorsUserScroll(false);
  await testBashPanelHonorsUserScroll(true);
  await testBashPanelHonorsUserScroll(false);
  await testThoughtsPanelHonorsUserScroll();
  await testPanelAutoScrollDoesNotMarkUserScroll();
  await testOuterScrollLockPausesInnerPanels();
  await testPromptRerenderPreservesUserScroll();
  await testPanelWithoutScrollToStillTails();
  await testNestedWheelPauseAndResume();
  await testSidePanOverPanelPausesItsTail();
}

runTests()
  .then(() => {
    console.log('\n14 passed, 0 failed');
    process.exit(0);
  })
  .catch(err => {
    console.error('FAIL:', err && err.stack ? err.stack : err);
    process.exit(1);
  });
