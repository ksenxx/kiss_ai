// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Negative end-to-end regression tests: the chat webview has NO
// auto-scroll, in both the extension webview and the remote webapp
// (same main.js, remote-chat body class).  Streaming chat events and
// nested-panel updates must never move the outer chat scroll position
// or any inner panel's scroll position, and must never issue a
// programmatic scrollTo — neither while the user sits at the bottom,
// nor after scrolling up, nor when a task starts.

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

  // Record every programmatic scroll so the tests can assert that the
  // streaming code paths issue none at all.
  const scrollIntoViewCalls = [];
  win.Element.prototype.scrollIntoView = function () {
    scrollIntoViewCalls.push(this);
  };
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
  win.eval(
    fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8') +
      '\n//# sourceURL=noautoscroll-api.js',
  );
  win.eval(
    fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8') +
      '\n//# sourceURL=noautoscroll-main.js',
  );
  return {win, posted, scrollCalls, scrollIntoViewCalls};
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

// Let any stray debounce/timeout-based scroller fire before asserting.
function settle() {
  return new Promise(resolve => setTimeout(resolve, 250));
}

function userScroll(win, el, top) {
  el.scrollTop = top;
  el.dispatchEvent(new win.Event('scroll'));
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

function programmaticScrolls(scrollCalls, el, from) {
  return scrollCalls
    .slice(from)
    .filter(c => c.el === el && typeof c.top === 'number');
}

const label = remote => (remote ? 'remote webapp' : 'extension webview');

// --------------------------------------------------------------------
// Outer chat: streaming events must never move the chat scroll
// position, whether the user sits at the bottom or has scrolled up,
// and a starting task must not jump to the bottom either.
// --------------------------------------------------------------------

async function testOuterChatNeverScrollsOnStream(remote) {
  const {win, posted, scrollCalls, scrollIntoViewCalls} = makeWebview({
    remote,
  });
  const O = win.document.getElementById('output');
  const geo = {sh: 3000, ch: 500};
  fakeGeometry(O, geo);

  // Starting a task must not scroll the chat.
  const before0 = scrollCalls.length;
  startRunningTask(win, posted);
  await nextFrames(win);
  assert.strictEqual(
    programmaticScrolls(scrollCalls, O, before0).length,
    0,
    'BUG (' + label(remote) + '): a starting task scrolled the chat',
  );
  assert.strictEqual(
    O.scrollTop,
    0,
    'BUG (' + label(remote) + '): a starting task moved the chat position',
  );

  // While the user sits at the bottom, streaming grows the content:
  // the chat must stay exactly where it was, not follow the tail.
  userScroll(win, O, geo.sh - geo.ch);
  const atBottom = O.scrollTop;
  const before1 = scrollCalls.length;
  send(win, {type: 'tool_call', name: 'Bash', command: 'make -j'});
  geo.sh += 400;
  send(win, {type: 'system_output', text: 'a'.repeat(200) + '\n'});
  send(win, {type: 'thinking_start'});
  send(win, {type: 'thinking_delta', text: 'b'.repeat(200)});
  send(win, {type: 'text_delta', text: 'streamed text '});
  geo.sh += 400;
  send(win, {type: 'system_output', text: 'c'.repeat(200) + '\n'});
  await nextFrames(win);
  await settle();
  assert.strictEqual(
    programmaticScrolls(scrollCalls, O, before1).length,
    0,
    'BUG (' + label(remote) + '): streaming issued a programmatic chat scroll',
  );
  assert.strictEqual(
    O.scrollTop,
    atBottom,
    'BUG (' +
      label(remote) +
      '): streaming moved the chat position while the user sat at the ' +
      'former bottom',
  );

  // After the user scrolls up into history, more streaming must not
  // yank them anywhere.
  userScroll(win, O, 120);
  const before2 = scrollCalls.length;
  geo.sh += 400;
  send(win, {type: 'system_output', text: 'd'.repeat(200) + '\n'});
  send(win, {type: 'text_delta', text: 'more streamed text '});
  await nextFrames(win);
  await settle();
  assert.strictEqual(
    programmaticScrolls(scrollCalls, O, before2).length,
    0,
    'BUG (' +
      label(remote) +
      '): streaming issued a programmatic chat scroll after the user ' +
      'scrolled up',
  );
  assert.strictEqual(
    O.scrollTop,
    120,
    'BUG (' +
      label(remote) +
      '): streaming moved the chat position after the user scrolled up',
  );
  assert.ok(
    !scrollIntoViewCalls.some(el => O.contains(el) || el === O),
    'BUG (' +
      label(remote) +
      '): streaming called scrollIntoView inside the chat',
  );
  win.close();
  console.log(
    '  ok - outer chat never scrolls on stream or task start (' +
      label(remote) +
      ')',
  );
}

// --------------------------------------------------------------------
// A second task starting after a first one ends must not re-activate
// any tail or jump to the bottom.
// --------------------------------------------------------------------

async function testSecondTaskStartDoesNotScroll(remote) {
  const {win, posted, scrollCalls} = makeWebview({remote});
  const O = win.document.getElementById('output');
  const geo = {sh: 3000, ch: 500};
  fakeGeometry(O, geo);
  const tabId = startRunningTask(win, posted);
  send(win, {type: 'system_output', text: 'x'.repeat(200) + '\n'});
  send(win, {type: 'status', running: false, tabId});
  await nextFrames(win);

  userScroll(win, O, 80);
  const before = scrollCalls.length;
  send(win, {
    type: 'status',
    running: true,
    tabId,
    startTs: Date.now(),
  });
  geo.sh += 400;
  send(win, {type: 'system_output', text: 'y'.repeat(200) + '\n'});
  await nextFrames(win);
  await settle();
  assert.strictEqual(
    programmaticScrolls(scrollCalls, O, before).length,
    0,
    'BUG (' +
      label(remote) +
      '): a second starting task issued a programmatic chat scroll',
  );
  assert.strictEqual(
    O.scrollTop,
    80,
    'BUG (' +
      label(remote) +
      '): a second starting task moved the chat position',
  );
  win.close();
  console.log(
    '  ok - a second task start does not scroll (' + label(remote) + ')',
  );
}

// --------------------------------------------------------------------
// Inner streaming panels (thinking, bash output, thoughts/llm panel):
// streamed updates must never move a panel's own scroll position.
// --------------------------------------------------------------------

async function testThinkPanelNeverAutoScrolls(remote) {
  const {win, posted, scrollCalls} = makeWebview({remote});
  const O = win.document.getElementById('output');
  fakeGeometry(O, {sh: 3000, ch: 500});
  startRunningTask(win, posted);

  send(win, {type: 'thinking_start'});
  const think = O.querySelector('.ev.think');
  assert.ok(think, 'thinking_start must create a think panel');
  const gt = {sh: 1000, ch: 200};
  fakeGeometry(think, gt);

  const before = scrollCalls.length;
  send(win, {type: 'thinking_delta', text: 'a'.repeat(80)});
  gt.sh += 200;
  send(win, {type: 'thinking_delta', text: 'b'.repeat(80)});
  await nextFrames(win);
  await settle();
  assert.strictEqual(
    programmaticScrolls(scrollCalls, think, before).length,
    0,
    'BUG (' +
      label(remote) +
      '): streamed thinking issued a programmatic panel scroll',
  );
  assert.strictEqual(
    think.scrollTop,
    0,
    'BUG (' +
      label(remote) +
      '): streamed thinking moved the think panel scroll position',
  );

  userScroll(win, think, 50);
  gt.sh += 200;
  send(win, {type: 'thinking_delta', text: 'c'.repeat(80)});
  await nextFrames(win);
  await settle();
  assert.strictEqual(
    think.scrollTop,
    50,
    'BUG (' +
      label(remote) +
      '): streamed thinking moved the think panel after a user scroll',
  );
  win.close();
  console.log('  ok - think panel never auto-scrolls (' + label(remote) + ')');
}

async function testBashPanelNeverAutoScrolls(remote) {
  const {win, posted, scrollCalls} = makeWebview({remote});
  const O = win.document.getElementById('output');
  fakeGeometry(O, {sh: 3000, ch: 500});
  startRunningTask(win, posted);

  send(win, {type: 'tool_call', name: 'Bash', command: 'make -j'});
  const bp = O.querySelector('.bash-panel-content');
  assert.ok(bp, 'a Bash tool_call must create a bash output panel');
  const gb = {sh: 1000, ch: 200};
  fakeGeometry(bp, gb);

  const before = scrollCalls.length;
  send(win, {type: 'system_output', text: 'a'.repeat(120) + '\n'});
  gb.sh += 300;
  send(win, {type: 'system_output', text: 'b'.repeat(120) + '\n'});
  await nextFrames(win);
  await settle();
  assert.strictEqual(
    programmaticScrolls(scrollCalls, bp, before).length,
    0,
    'BUG (' +
      label(remote) +
      '): streamed bash output issued a programmatic panel scroll',
  );
  assert.strictEqual(
    bp.scrollTop,
    0,
    'BUG (' +
      label(remote) +
      '): streamed bash output moved the bash panel scroll position',
  );

  userScroll(win, bp, 40);
  gb.sh += 300;
  send(win, {type: 'system_output', text: 'c'.repeat(120) + '\n'});
  await nextFrames(win);
  await settle();
  assert.strictEqual(
    bp.scrollTop,
    40,
    'BUG (' +
      label(remote) +
      '): streamed bash output moved the bash panel after a user scroll',
  );
  win.close();
  console.log('  ok - bash panel never auto-scrolls (' + label(remote) + ')');
}

async function testThoughtsPanelNeverAutoScrolls(remote) {
  const {win, posted, scrollCalls} = makeWebview({remote});
  const O = win.document.getElementById('output');
  fakeGeometry(O, {sh: 3000, ch: 500});
  startRunningTask(win, posted);

  send(win, {type: 'text_delta', text: 'hello '});
  const lp = O.querySelector('.llm-panel');
  assert.ok(lp, 'streaming text must create a thoughts (llm) panel');
  const gl = {sh: 1000, ch: 300};
  fakeGeometry(lp, gl);

  const before = scrollCalls.length;
  send(win, {type: 'text_delta', text: 'world '});
  gl.sh += 200;
  send(win, {type: 'text_delta', text: 'more text '});
  await nextFrames(win);
  await settle();
  assert.strictEqual(
    programmaticScrolls(scrollCalls, lp, before).length,
    0,
    'BUG (' +
      label(remote) +
      '): streamed text issued a programmatic thoughts panel scroll',
  );
  assert.strictEqual(
    lp.scrollTop,
    0,
    'BUG (' +
      label(remote) +
      '): streamed text moved the thoughts panel scroll position',
  );

  userScroll(win, lp, 60);
  gl.sh += 200;
  send(win, {type: 'text_delta', text: 'even more '});
  await nextFrames(win);
  await settle();
  assert.strictEqual(
    lp.scrollTop,
    60,
    'BUG (' +
      label(remote) +
      '): streamed text moved the thoughts panel after a user scroll',
  );
  win.close();
  console.log(
    '  ok - thoughts panel never auto-scrolls (' + label(remote) + ')',
  );
}

async function main() {
  for (const remote of [false, true]) {
    await testOuterChatNeverScrollsOnStream(remote);
    await testSecondTaskStartDoesNotScroll(remote);
    await testThinkPanelNeverAutoScrolls(remote);
    await testBashPanelNeverAutoScrolls(remote);
    await testThoughtsPanelNeverAutoScrolls(remote);
  }
  console.log('chatNoAutoScroll.test.js: all tests passed');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
