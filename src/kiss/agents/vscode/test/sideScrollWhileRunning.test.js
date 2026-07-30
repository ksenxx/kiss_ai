// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

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

  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(
fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
  return {win, posted, scrollCalls};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function fakeGeometry(el, {sw = 2000, cw = 400, sh = 3000, ch = 500}) {
  Object.defineProperty(el, 'scrollWidth', {value: sw, configurable: true});
  Object.defineProperty(el, 'clientWidth', {value: cw, configurable: true});
  Object.defineProperty(el, 'scrollHeight', {value: sh, configurable: true});
  Object.defineProperty(el, 'clientHeight', {value: ch, configurable: true});
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

function startRunningBashTask(win, posted) {
  const ready = posted.find(m => m.type === 'ready');
  assert.ok(ready && ready.tabId, 'webview must post ready with a tabId');
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

async function testHorizontalWheelSuspendsAutoScrollWhileRunning() {
  const {win, posted, scrollCalls} = makeWebview();
  const O = win.document.getElementById('output');
  fakeGeometry(O, {});

  const panel = startRunningBashTask(win, posted);
  fakeGeometry(panel, {sw: 4000, cw: 300, sh: 100, ch: 200});

  send(win, {type: 'system_output', text: 'x'.repeat(500) + '\n'});
  await nextFrames(win);
  const callsBeforeUserScroll = scrollCalls.filter(
    c => c.el === O && typeof c.top === 'number',
  ).length;
  assert.ok(
    callsBeforeUserScroll > 0,
    'sanity: auto-scroll must be active while running before user input',
  );

  const wheel = new win.WheelEvent('wheel', {
    deltaX: 40,
    deltaY: 0,
    bubbles: true,
    cancelable: true,
  });
  panel.dispatchEvent(wheel);

  panel.scrollLeft = 900;
  O.scrollTop = 100;

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

async function testBashPanelAutoScrollPausesDuringSidePan() {
  const {win, posted} = makeWebview();
  const O = win.document.getElementById('output');
  fakeGeometry(O, {});

  const panel = startRunningBashTask(win, posted);
  fakeGeometry(panel, {sw: 4000, cw: 300, sh: 1000, ch: 200});

  send(win, {type: 'system_output', text: 'a'.repeat(300) + '\n'});
  await nextFrames(win);
  assert.strictEqual(
    panel.scrollTop,
    panel.scrollHeight,
    'sanity: bash panel must tail its output while unlocked',
  );

  panel.dispatchEvent(
    new win.WheelEvent('wheel', {
      deltaX: -35,
      deltaY: 2,
      bubbles: true,
      cancelable: true,
    }),
  );
  panel.scrollLeft = 700;
  panel.scrollTop = 100;
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

async function testDownwardWheelDoesNotEngageLock() {
  const {win, posted, scrollCalls} = makeWebview();
  const O = win.document.getElementById('output');
  fakeGeometry(O, {});
  startRunningBashTask(win, posted);

  O.dispatchEvent(
    new win.WheelEvent('wheel', {
      deltaX: 1,
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

async function testLockReleasesWhenUserReturnsToBottom() {
  const {win, posted, scrollCalls} = makeWebview();
  const O = win.document.getElementById('output');
  fakeGeometry(O, {});
  const panel = startRunningBashTask(win, posted);
  fakeGeometry(panel, {sw: 4000, cw: 300, sh: 1000, ch: 200});

  O.dispatchEvent(
    new win.WheelEvent('wheel', {
      deltaX: 50,
      deltaY: 0,
      bubbles: true,
      cancelable: true,
    }),
  );

  O.scrollTop = O.scrollHeight - O.clientHeight;
  O.dispatchEvent(new win.Event('scroll'));
  // Wait past the resume-at-bottom debounce (RESUME_AT_BOTTOM_MS in
  // media/main.js) so the return to the bottom settles.
  await new Promise(resolve => setTimeout(resolve, 250));

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
