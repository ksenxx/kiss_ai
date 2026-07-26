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
  const pendingScrollEvents = [];
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
    if (opts.deferScrollEvents) pendingScrollEvents.push(this);
    else this.dispatchEvent(new win.Event('scroll'));
  }
  win.Element.prototype.scrollTo = engineScrollTo;
  win.HTMLElement.prototype.scrollTo = engineScrollTo;

  const rafQueue = [];
  function flushRaf() {
    while (rafQueue.length) rafQueue.shift()();
  }
  if (opts.manualRaf) {
    win.requestAnimationFrame = function (cb) {
      rafQueue.push(cb);
      return rafQueue.length;
    };
  }

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
  return {win, posted, scrollCalls, pendingScrollEvents, flushRaf};
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

function flushTimeouts(win) {
  return new Promise(resolve => win.setTimeout(resolve, 5));
}

function userScrollTo(win, O, top) {
  O.scrollTop = top;
  O.dispatchEvent(new win.Event('scroll'));
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

async function streamOutput(win, geo, text) {
  send(win, {type: 'system_output', text: text + '\n'});
  geo.sh += 200;
  await nextFrames(win);
}

function autoScrollsSince(scrollCalls, O, from) {
  return scrollCalls
    .slice(from)
    .filter(c => c.el === O && typeof c.top === 'number');
}

async function testScrolledUpUserIsNotYankedToBottom() {
  const {win, posted, scrollCalls} = makeWebview({remote: true});
  const O = win.document.getElementById('output');
  const geo = {sh: 3000, ch: 500};
  fakeGeometry(O, geo);
  startRunningTask(win, posted);
  send(win, {type: 'tool_call', name: 'Bash', command: 'make'});

  O.scrollTop = geo.sh - geo.ch;
  await streamOutput(win, geo, 'a'.repeat(100));
  assert.strictEqual(
    O.scrollTop,
    geo.sh - geo.ch,
    'sanity: chat must tail streamed output while pinned at the end',
  );

  userScrollTo(win, O, 700);

  const before = scrollCalls.length;
  await streamOutput(win, geo, 'b'.repeat(100));
  await streamOutput(win, geo, 'c'.repeat(100));

  assert.strictEqual(
    autoScrollsSince(scrollCalls, O, before).length,
    0,
    'BUG: the chat auto-scrolled to the end although the user had ' +
      'scrolled up (touch/scrollbar scrolling never engaged the ' +
      'auto-scroll lock)',
  );
  assert.strictEqual(
    O.scrollTop,
    700,
    'the user\u2019s reading position must be preserved',
  );
  win.close();
  console.log('  ok - scrolled-up user (touch/scrollbar) is not yanked down');
}

async function testUncollapsedPanelIsNotScrolledAway(remote) {
  const {win, posted, scrollCalls} = makeWebview({remote});
  const O = win.document.getElementById('output');
  const geo = {sh: 3000, ch: 500};
  fakeGeometry(O, geo);
  startRunningTask(win, posted);

  send(win, {type: 'tool_call', name: 'Bash', command: 'ls'});
  send(win, {type: 'tool_call', name: 'Bash', command: 'pwd'});
  const panels = O.querySelectorAll(':scope .ev.tc');
  assert.strictEqual(panels.length, 2, 'two tool-call panels rendered');
  const first = panels[0];
  assert.ok(
    first.classList.contains('collapsed'),
    'older tool panel must be auto-collapsed',
  );

  O.scrollTop = geo.sh - geo.ch;

  const header = first.querySelector('.collapse-header');
  assert.ok(header, 'collapsed panel must have a clickable header');
  geo.sh += 400;
  header.dispatchEvent(
    new win.MouseEvent('click', {bubbles: true, cancelable: true}),
  );
  assert.ok(
    !first.classList.contains('collapsed'),
    'clicking the header must uncollapse the panel',
  );
  await flushTimeouts(win);

  const before = scrollCalls.length;
  const posBefore = O.scrollTop;
  await streamOutput(win, geo, 'd'.repeat(100));
  await streamOutput(win, geo, 'e'.repeat(100));

  assert.strictEqual(
    autoScrollsSince(scrollCalls, O, before).length,
    0,
    'BUG: the chat auto-scrolled to the end right after the user ' +
      'uncollapsed a panel (' +
      (remote ? 'remote webapp' : 'extension webview') +
      ')',
  );
  assert.strictEqual(
    O.scrollTop,
    posBefore,
    'the expanded panel must stay where the user is reading it',
  );
  win.close();
  console.log(
    '  ok - uncollapsed panel stays put while streaming (' +
      (remote ? 'remote webapp' : 'extension webview') +
      ')',
  );
}

async function testAutoScrollResumesAtTheVeryEnd() {
  const {win, posted, scrollCalls} = makeWebview({remote: true});
  const O = win.document.getElementById('output');
  const geo = {sh: 3000, ch: 500};
  fakeGeometry(O, geo);
  startRunningTask(win, posted);
  send(win, {type: 'tool_call', name: 'Bash', command: 'make'});

  userScrollTo(win, O, 500);
  await streamOutput(win, geo, 'a'.repeat(50));
  assert.strictEqual(O.scrollTop, 500, 'locked while scrolled up');
  userScrollTo(win, O, geo.sh - geo.ch);

  const before = scrollCalls.length;
  await streamOutput(win, geo, 'b'.repeat(50));
  assert.ok(
    autoScrollsSince(scrollCalls, O, before).length > 0,
    'auto-scroll must resume when the user scrolls all the way to the end',
  );
  assert.strictEqual(
    O.scrollTop,
    geo.sh - geo.ch,
    'chat must be tailing again at the end',
  );

  const panel = O.querySelector(':scope .ev.tc');
  send(win, {type: 'tool_call', name: 'Bash', command: 'pwd'});
  const header = panel.querySelector('.collapse-header');
  geo.sh += 400;
  header.dispatchEvent(
    new win.MouseEvent('click', {bubbles: true, cancelable: true}),
  );
  await flushTimeouts(win);
  await streamOutput(win, geo, 'c'.repeat(50));
  const held = O.scrollTop;
  assert.ok(
    held < geo.sh - geo.ch,
    'sanity: view held in place after uncollapse',
  );
  userScrollTo(win, O, geo.sh - geo.ch);
  const before2 = scrollCalls.length;
  await streamOutput(win, geo, 'e'.repeat(50));
  assert.ok(
    autoScrollsSince(scrollCalls, O, before2).length > 0,
    'auto-scroll must resume after the user returns to the end ' +
      'following an uncollapse',
  );
  win.close();
  console.log('  ok - tailing works at the end and resumes on return');
}

async function testCollapsingPanelKeepsTailing() {
  const {win, posted, scrollCalls} = makeWebview({});
  const O = win.document.getElementById('output');
  const geo = {sh: 3000, ch: 500};
  fakeGeometry(O, geo);
  startRunningTask(win, posted);
  send(win, {type: 'tool_call', name: 'Bash', command: 'ls'});
  send(win, {type: 'tool_call', name: 'Bash', command: 'pwd'});
  const first = O.querySelector(':scope .ev.tc');
  const header = first.querySelector('.collapse-header');
  header.dispatchEvent(
    new win.MouseEvent('click', {bubbles: true, cancelable: true}),
  );
  await flushTimeouts(win);
  userScrollTo(win, O, geo.sh - geo.ch);
  header.dispatchEvent(
    new win.MouseEvent('click', {bubbles: true, cancelable: true}),
  );
  assert.ok(first.classList.contains('collapsed'), 'panel re-collapsed');
  geo.sh -= 400;
  userScrollTo(win, O, geo.sh - geo.ch);
  await flushTimeouts(win);

  const before = scrollCalls.length;
  await streamOutput(win, geo, 'x'.repeat(50));
  assert.ok(
    autoScrollsSince(scrollCalls, O, before).length > 0,
    'collapsing a panel while at the end must not stop the tailing',
  );
  win.close();
  console.log('  ok - collapsing a panel at the end keeps tailing');
}

async function testProgrammaticScrollRaceDoesNotLock() {
  const {win, posted, scrollCalls, pendingScrollEvents} = makeWebview({
    remote: true,
    deferScrollEvents: true,
  });
  const O = win.document.getElementById('output');
  const geo = {sh: 3000, ch: 500};
  fakeGeometry(O, geo);
  startRunningTask(win, posted);
  send(win, {type: 'tool_call', name: 'Bash', command: 'make'});
  O.scrollTop = geo.sh - geo.ch;

  await streamOutput(win, geo, 'a'.repeat(50));
  assert.ok(pendingScrollEvents.length > 0, 'an auto-scroll happened');
  geo.sh += 300;
  while (pendingScrollEvents.length)
    pendingScrollEvents.shift().dispatchEvent(new win.Event('scroll'));

  const before = scrollCalls.length;
  await streamOutput(win, geo, 'b'.repeat(50));
  while (pendingScrollEvents.length)
    pendingScrollEvents.shift().dispatchEvent(new win.Event('scroll'));
  assert.ok(
    autoScrollsSince(scrollCalls, O, before).length > 0,
    'sb()\u2019s own racy scroll event must not be mistaken for the ' +
      'user scrolling up — tailing must continue',
  );
  win.close();
  console.log('  ok - programmatic scroll racing new content does not lock');
}

async function testRafRechecksLockAtExecutionTime() {
  const {win, posted, scrollCalls} = makeWebview({remote: true});
  const O = win.document.getElementById('output');
  const geo = {sh: 3000, ch: 500};
  fakeGeometry(O, geo);
  startRunningTask(win, posted);
  send(win, {type: 'tool_call', name: 'Bash', command: 'make'});
  O.scrollTop = geo.sh - geo.ch;
  await streamOutput(win, geo, 'a'.repeat(50));

  const before = scrollCalls.length;
  send(win, {type: 'system_output', text: 'b'.repeat(50) + '\n'});
  geo.sh += 200;
  userScrollTo(win, O, 400);
  await nextFrames(win);

  assert.strictEqual(
    autoScrollsSince(scrollCalls, O, before).length,
    0,
    'a scheduled auto-scroll must stand down when the user scrolled ' +
      'up before the frame ran',
  );
  assert.strictEqual(O.scrollTop, 400, 'user position preserved');
  win.close();
  console.log('  ok - scheduled rAF auto-scroll re-checks the lock');
}

async function testNoStaleProgrammaticMarkAfterNoopScroll() {
  const {win, posted, scrollCalls} = makeWebview({remote: true});
  const O = win.document.getElementById('output');
  const geo = {sh: 3000, ch: 500};
  fakeGeometry(O, geo);
  startRunningTask(win, posted);
  send(win, {type: 'tool_call', name: 'Bash', command: 'make'});
  O.scrollTop = geo.sh - geo.ch;

  send(win, {type: 'system_output', text: 'a'.repeat(10) + '\n'});
  await nextFrames(win);

  const oldBottom = geo.sh - geo.ch;
  geo.sh += 300;
  userScrollTo(win, O, oldBottom);
  const before = scrollCalls.length;
  await streamOutput(win, geo, 'b'.repeat(100));
  assert.strictEqual(
    autoScrollsSince(scrollCalls, O, before).length,
    0,
    'the first user scroll-up after a no-op auto-scroll must already ' +
      'suspend tailing',
  );
  assert.strictEqual(O.scrollTop, oldBottom, 'user position preserved');
  win.close();
  console.log('  ok - no stale programmatic mark after a no-op auto-scroll');
}

async function testPendingFrameStandsDownDuringHeaderClick() {
  const {win, posted, scrollCalls, flushRaf} = makeWebview({
    remote: true,
    manualRaf: true,
  });
  const O = win.document.getElementById('output');
  const geo = {sh: 3000, ch: 500};
  fakeGeometry(O, geo);
  startRunningTask(win, posted);
  send(win, {type: 'tool_call', name: 'Bash', command: 'ls'});
  send(win, {type: 'tool_call', name: 'Bash', command: 'pwd'});
  O.scrollTop = geo.sh - geo.ch;

  send(win, {type: 'system_output', text: 'a'.repeat(50) + '\n'});
  geo.sh += 200;
  await flushTimeouts(win);

  const panels = O.querySelectorAll(':scope .ev.tc');
  const header = panels[panels.length - 1].querySelector('.collapse-header');
  header.dispatchEvent(
    new win.MouseEvent('click', {bubbles: true, cancelable: true}),
  );
  const before = scrollCalls.length;
  flushRaf();
  assert.strictEqual(
    autoScrollsSince(scrollCalls, O, before).length,
    0,
    'a pending auto-scroll frame must stand down while a header ' +
      'click\u2019s _noScroll window is open',
  );

  await flushTimeouts(win);
  send(win, {type: 'system_output', text: 'b'.repeat(50) + '\n'});
  geo.sh += 200;
  await flushTimeouts(win);
  const before2 = scrollCalls.length;
  flushRaf();
  assert.ok(
    autoScrollsSince(scrollCalls, O, before2).length > 0,
    'tailing must keep working after the click window closed',
  );
  win.close();
  console.log('  ok - pending frame stands down during a header click');
}

async function testUserScrollDuringPendingSbEventLocks() {
  const {win, posted, scrollCalls, pendingScrollEvents} = makeWebview({
    remote: true,
    deferScrollEvents: true,
  });
  const O = win.document.getElementById('output');
  const geo = {sh: 3000, ch: 500};
  fakeGeometry(O, geo);
  startRunningTask(win, posted);
  send(win, {type: 'tool_call', name: 'Bash', command: 'make'});
  O.scrollTop = geo.sh - geo.ch;

  await streamOutput(win, geo, 'a'.repeat(50));
  assert.ok(pendingScrollEvents.length > 0, 'an auto-scroll happened');
  O.scrollTop = 700;
  while (pendingScrollEvents.length)
    pendingScrollEvents.shift().dispatchEvent(new win.Event('scroll'));

  const before = scrollCalls.length;
  await streamOutput(win, geo, 'b'.repeat(50));
  assert.strictEqual(
    autoScrollsSince(scrollCalls, O, before).length,
    0,
    'a coalesced scroll event below the auto-scroll target must ' +
      'count as the user scrolling up and suspend tailing',
  );
  assert.strictEqual(O.scrollTop, 700, 'user position preserved');
  win.close();
  console.log('  ok - user scroll racing a pending sb event still locks');
}

async function testExpandCollapseCycleAtEndResumesTailing() {
  const {win, posted, scrollCalls} = makeWebview({remote: true});
  const O = win.document.getElementById('output');
  const geo = {sh: 3000, ch: 500};
  fakeGeometry(O, geo);
  startRunningTask(win, posted);
  send(win, {type: 'tool_call', name: 'Bash', command: 'ls'});
  send(win, {type: 'tool_call', name: 'Bash', command: 'pwd'});
  const first = O.querySelector(':scope .ev.tc');
  const header = first.querySelector('.collapse-header');
  O.scrollTop = geo.sh - geo.ch;

  geo.sh += 400;
  header.dispatchEvent(
    new win.MouseEvent('click', {bubbles: true, cancelable: true}),
  );
  await flushTimeouts(win);
  geo.sh -= 400;
  header.dispatchEvent(
    new win.MouseEvent('click', {bubbles: true, cancelable: true}),
  );
  assert.ok(first.classList.contains('collapsed'), 'panel re-collapsed');
  await flushTimeouts(win);

  const before = scrollCalls.length;
  await streamOutput(win, geo, 'a'.repeat(50));
  assert.ok(
    autoScrollsSince(scrollCalls, O, before).length > 0,
    'the expansion hold must release when a collapse leaves the view ' +
      'at the very end — tailing resumes without an extra user scroll',
  );

  userScrollTo(win, O, 300);
  geo.sh += 400;
  header.dispatchEvent(
    new win.MouseEvent('click', {bubbles: true, cancelable: true}),
  );
  await flushTimeouts(win);
  geo.sh -= 400;
  header.dispatchEvent(
    new win.MouseEvent('click', {bubbles: true, cancelable: true}),
  );
  await flushTimeouts(win);
  const before2 = scrollCalls.length;
  await streamOutput(win, geo, 'b'.repeat(50));
  assert.strictEqual(
    autoScrollsSince(scrollCalls, O, before2).length,
    0,
    'a collapse far from the end must keep the hold',
  );
  assert.strictEqual(O.scrollTop, 300, 'user position preserved');
  win.close();
  console.log('  ok - expand/collapse cycle at the end resumes tailing');
}

async function testExpandWithNoGrowthKeepsTailing() {
  const {win, posted, scrollCalls} = makeWebview({remote: true});
  const O = win.document.getElementById('output');
  const geo = {sh: 3000, ch: 500};
  fakeGeometry(O, geo);
  startRunningTask(win, posted);
  send(win, {type: 'tool_call', name: 'Bash', command: 'ls'});
  send(win, {type: 'tool_call', name: 'Bash', command: 'pwd'});
  const first = O.querySelector(':scope .ev.tc');
  const header = first.querySelector('.collapse-header');
  O.scrollTop = geo.sh - geo.ch;

  header.dispatchEvent(
    new win.MouseEvent('click', {bubbles: true, cancelable: true}),
  );
  await flushTimeouts(win);

  const before = scrollCalls.length;
  await streamOutput(win, geo, 'a'.repeat(50));
  assert.ok(
    autoScrollsSince(scrollCalls, O, before).length > 0,
    'an expansion that leaves the view at the very end must keep tailing',
  );
  win.close();
  console.log('  ok - expansion without growth keeps tailing at the end');
}

async function runTests() {
  await testScrolledUpUserIsNotYankedToBottom();
  await testUncollapsedPanelIsNotScrolledAway(true);
  await testUncollapsedPanelIsNotScrolledAway(false);
  await testAutoScrollResumesAtTheVeryEnd();
  await testCollapsingPanelKeepsTailing();
  await testProgrammaticScrollRaceDoesNotLock();
  await testRafRechecksLockAtExecutionTime();
  await testNoStaleProgrammaticMarkAfterNoopScroll();
  await testPendingFrameStandsDownDuringHeaderClick();
  await testUserScrollDuringPendingSbEventLocks();
  await testExpandCollapseCycleAtEndResumesTailing();
  await testExpandWithNoGrowthKeepsTailing();
}

runTests()
  .then(() => {
    console.log('\n13 passed, 0 failed');
    process.exit(0);
  })
  .catch(err => {
    console.error('FAIL:', err && err.stack ? err.stack : err);
    process.exit(1);
  });
