// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end tests: auto-scroll MUST be (re)activated the moment a task
// starts executing, in both the extension webview and the remote webapp
// (same main.js, remote-chat body class).
//
//   * If the user had scrolled up (pausing the tail) and a NEW task
//     starts running, the chat must jump to the end and follow the new
//     task's events/texts again.
//   * A redundant running=true status arriving MID-run must NOT yank a
//     user who deliberately scrolled up during that same run.
//   * After the task-start re-arm, the normal contract still holds:
//     scrolling up pauses the tail, returning to the bottom resumes it.

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
      '\n//# sourceURL=taskstart-main.js',
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

function readyTabId(posted) {
  const ready = posted.find(m => m.type === 'ready');
  assert.ok(ready && ready.tabId, 'webview must post ready with a tabId');
  return ready.tabId;
}

function startTask(win, tabId) {
  win._demoApi.hideWelcome();
  send(win, {
    type: 'status',
    running: true,
    tabId: tabId,
    startTs: Date.now() - 2000,
  });
}

function finishTask(win, tabId) {
  send(win, {type: 'status', running: false, tabId: tabId});
}

function autoScrollsSince(scrollCalls, el, from) {
  return scrollCalls
    .slice(from)
    .filter(c => c.el === el && typeof c.top === 'number');
}

const label = remote => (remote ? 'remote webapp' : 'extension webview');

// --------------------------------------------------------------------
// A new task starting must re-activate the tail even if the user had
// scrolled up (and thereby paused it) while reading the previous task.
// --------------------------------------------------------------------

async function testTaskStartReactivatesTailing(remote) {
  const {win, posted, scrollCalls} = makeWebview({remote});
  const O = win.document.getElementById('output');
  const geo = {sh: 3000, ch: 500};
  fakeGeometry(O, geo);
  const tabId = readyTabId(posted);

  // Task 1 runs, streams, and finishes while the user tails it.
  startTask(win, tabId);
  send(win, {type: 'tool_call', name: 'Bash', command: 'make'});
  send(win, {type: 'system_output', text: 'one\n'});
  await nextFrames(win);
  finishTask(win, tabId);

  // The user scrolls up to read the finished task's output: the tail
  // pauses (this is the follow-tail contract).
  userScroll(win, O, 100);

  // A NEW task starts executing in this chat.
  startTask(win, tabId);

  geo.sh += 300;
  const before = scrollCalls.length;
  send(win, {type: 'system_output', text: 'x'.repeat(200) + '\n'});
  await nextFrames(win);

  assert.ok(
    autoScrollsSince(scrollCalls, O, before).length > 0,
    'BUG (' +
      label(remote) +
      '): auto-scroll stayed paused when a new task started executing — ' +
      'the stale scroll lock from the previous task was never re-armed',
  );
  assert.strictEqual(
    O.scrollTop,
    geo.sh - geo.ch,
    'the chat must follow the new task at the very end (' + label(remote) + ')',
  );
  win.close();
  console.log(
    '  ok - a new task start re-activates the tail (' + label(remote) + ')',
  );
}

// --------------------------------------------------------------------
// The chat must jump to the end at the very moment the task starts,
// before any event of the new run arrives.
// --------------------------------------------------------------------

async function testTaskStartScrollsToEndImmediately(remote) {
  const {win, posted, scrollCalls} = makeWebview({remote});
  const O = win.document.getElementById('output');
  const geo = {sh: 3000, ch: 500};
  fakeGeometry(O, geo);
  const tabId = readyTabId(posted);

  startTask(win, tabId);
  send(win, {type: 'system_output', text: 'old output\n'});
  await nextFrames(win);
  finishTask(win, tabId);

  userScroll(win, O, 100);
  const before = scrollCalls.length;

  startTask(win, tabId);
  await nextFrames(win);

  assert.ok(
    autoScrollsSince(scrollCalls, O, before).length > 0,
    'BUG (' +
      label(remote) +
      '): task start did not scroll the chat to the end',
  );
  assert.strictEqual(
    O.scrollTop,
    geo.sh - geo.ch,
    'the chat must be at the very end when the task starts (' +
      label(remote) +
      ')',
  );
  win.close();
  console.log(
    '  ok - task start jumps to the end before the first event (' +
      label(remote) +
      ')',
  );
}

// --------------------------------------------------------------------
// A redundant running=true status arriving MID-run (reconnects and
// state re-broadcasts do this) must NOT yank a user who scrolled up
// during that same run.
// --------------------------------------------------------------------

async function testMidRunStatusDoesNotYank(remote) {
  const {win, posted, scrollCalls} = makeWebview({remote});
  const O = win.document.getElementById('output');
  const geo = {sh: 3000, ch: 500};
  fakeGeometry(O, geo);
  const tabId = readyTabId(posted);

  startTask(win, tabId);
  send(win, {type: 'system_output', text: 'streaming\n'});
  await nextFrames(win);

  // The user scrolls up while the task is still running.
  userScroll(win, O, 100);

  // A duplicate running=true status for the SAME run arrives.
  startTask(win, tabId);

  geo.sh += 200;
  const before = scrollCalls.length;
  send(win, {type: 'system_output', text: 'more\n'});
  await nextFrames(win);

  assert.strictEqual(
    autoScrollsSince(scrollCalls, O, before).length,
    0,
    'a redundant mid-run running status must not yank the user down (' +
      label(remote) +
      ')',
  );
  assert.strictEqual(
    O.scrollTop,
    100,
    'the reading position must be preserved (' + label(remote) + ')',
  );
  win.close();
  console.log(
    '  ok - redundant mid-run running status does not yank (' +
      label(remote) +
      ')',
  );
}

// --------------------------------------------------------------------
// After the task-start re-arm, the normal follow-tail contract still
// holds: scrolling up pauses, returning to the bottom resumes.
// --------------------------------------------------------------------

async function testPauseAndResumeStillWorkAfterRestart(remote) {
  const {win, posted, scrollCalls} = makeWebview({remote});
  const O = win.document.getElementById('output');
  const geo = {sh: 3000, ch: 500};
  fakeGeometry(O, geo);
  const tabId = readyTabId(posted);

  startTask(win, tabId);
  send(win, {type: 'system_output', text: 'first task\n'});
  await nextFrames(win);
  finishTask(win, tabId);
  userScroll(win, O, 100);

  // New task: tail re-armed.
  startTask(win, tabId);
  geo.sh += 200;
  send(win, {type: 'system_output', text: 'tailing\n'});
  await nextFrames(win);
  assert.strictEqual(O.scrollTop, geo.sh - geo.ch, 'tailing after restart');

  // The user scrolls up during the new run: the tail must pause.
  userScroll(win, O, 200);
  geo.sh += 200;
  let before = scrollCalls.length;
  send(win, {type: 'system_output', text: 'paused\n'});
  await nextFrames(win);
  assert.strictEqual(
    autoScrollsSince(scrollCalls, O, before).length,
    0,
    'scrolling up during the restarted run must still pause the tail (' +
      label(remote) +
      ')',
  );

  // Returning to the bottom resumes the tail.
  userScroll(win, O, geo.sh - geo.ch);
  geo.sh += 200;
  before = scrollCalls.length;
  send(win, {type: 'system_output', text: 'resumed\n'});
  await nextFrames(win);
  assert.ok(
    autoScrollsSince(scrollCalls, O, before).length > 0,
    'returning to the bottom must resume the tail (' + label(remote) + ')',
  );
  assert.strictEqual(O.scrollTop, geo.sh - geo.ch, 'back at the end');
  win.close();
  console.log(
    '  ok - pause/resume contract intact after a task restart (' +
      label(remote) +
      ')',
  );
}

async function main() {
  let passed = 0;
  let failed = 0;
  const tests = [];
  for (const remote of [false, true]) {
    tests.push(
      () => testTaskStartReactivatesTailing(remote),
      () => testTaskStartScrollsToEndImmediately(remote),
      () => testMidRunStatusDoesNotYank(remote),
      () => testPauseAndResumeStillWorkAfterRestart(remote),
    );
  }
  for (const t of tests) {
    try {
      await t();
      passed++;
    } catch (e) {
      failed++;
      console.error('  FAIL -', e.message);
    }
  }
  console.log(`\n${passed} passed, ${failed} failed`);
  if (failed) process.exit(1);
}

main().catch(e => {
  console.error(e);
  process.exit(1);
});
