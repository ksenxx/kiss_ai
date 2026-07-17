// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end test: the chat webview must NOT auto-scroll to the end of
// the chat while the user is reading elsewhere.  This reproduces two
// bugs reported against the remote webapp's chat webview (the same
// ``media/chat.html`` + ``media/main.js`` bundle also runs inside the
// VS Code extension webview, so both clients showed the behavior —
// the remote webapp merely serves the page with ``<body
// class="remote-chat">``, see web_server.py's BODY_CLASS_ATTR):
//
//   1. USER SCROLLED UP: the auto-scroll suspension (``_scrollLock``)
//      was only engaged by ``wheel`` events.  Scrolling up by touch
//      drag (every phone/tablet using the remote webapp), by dragging
//      the scrollbar, or with the keyboard fires only ``scroll``
//      events — which never engaged the lock — so the rAF auto-scroll
//      (``sb()`` → ``O.scrollTo({top: O.scrollHeight})``) yanked the
//      view back to the bottom on every streamed token.
//
//   2. USER UNCOLLAPSED A PANEL: clicking a collapsed event panel's
//      header expanded it, but only suppressed auto-scroll for the
//      click itself (``_noScroll`` released on a setTimeout(0)).  The
//      very next streamed token scrolled the expanded panel the user
//      was reading out of view.
//
// Auto-scroll MUST keep working while the user is scrolled all the way
// to the end, and MUST resume once the user scrolls back to the end.
//
// This test exercises the real ``media/main.js`` against the real
// ``media/chat.html`` in jsdom, the same harness as
// ``sideScrollWhileRunning.test.js`` / ``adjacentTaskScroll.test.js``.
//
// Run directly with ``node``:
//
//     node src/kiss/agents/vscode/test/chatScrollUserPosition.test.js

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

/**
 * Build a jsdom window running the real chat webview (chat.html +
 * panelCopy.js + main.js).
 *
 * Unlike jsdom's no-op scrolling, the installed ``scrollTo`` emulates a
 * real engine: it clamps the requested position to the element's
 * scrollable range, records the call, and — when the position actually
 * changed — fires a ``scroll`` event, exactly like a browser does after
 * a programmatic scroll.  Pass ``deferScrollEvents: true`` to collect
 * those events in ``pendingScrollEvents`` instead, so a test can grow
 * the content BETWEEN the programmatic scroll and its event (the
 * streaming race a live chat hits constantly).
 *
 * Pass ``manualRaf: true`` to queue ``requestAnimationFrame`` callbacks
 * instead of scheduling them, so a test can run a pending frame at a
 * precise instant (e.g. inside a click's ``_noScroll`` window) via the
 * returned ``flushRaf``.
 *
 * @param {{remote?: boolean, deferScrollEvents?: boolean,
 *          manualRaf?: boolean}} [opts]
 *   ``remote`` serves the page like the remote webapp does
 *   (``<body class="remote-chat">``).
 */
function makeWebview(opts) {
  opts = opts || {};
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  if (opts.remote) {
    // Same substitution web_server.py performs for the remote webapp.
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
    if (clamped === this.scrollTop) return; // no movement → no event
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
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
  return {win, posted, scrollCalls, pendingScrollEvents, flushRaf};
}

/** Dispatch a backend→webview event exactly like the extension does. */
function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

/**
 * jsdom performs no layout, so give ``el`` live scroll geometry backed
 * by the mutable ``geo`` object — tests grow ``geo.sh`` to emulate
 * streamed content making the chat taller.
 */
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

/** Flush setTimeout(…, 0) callbacks (addCollapse's _noScroll release). */
function flushTimeouts(win) {
  return new Promise(resolve => win.setTimeout(resolve, 5));
}

/**
 * The user scrolls #output to ``top`` WITHOUT a wheel: a touch drag,
 * scrollbar drag, Home/PageUp key… all of which fire only ``scroll``
 * events.  jsdom does not fire them on scrollTop assignment, so
 * dispatch one exactly like the engine would.
 */
function userScrollTo(win, O, top) {
  O.scrollTop = top;
  O.dispatchEvent(new win.Event('scroll'));
}

/** Boot a running task; returns the ready tabId. */
function startRunningTask(win, posted) {
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
  return ready.tabId;
}

/**
 * Stream a chunk of bash output and grow the chat's scrollHeight like
 * a real render would, then let the rAF auto-scroller run.
 */
async function streamOutput(win, geo, text) {
  send(win, {type: 'system_output', text: text + '\n'});
  geo.sh += 200;
  await nextFrames(win);
}

/** O.scrollTo calls (with a numeric top) recorded after index `from`. */
function autoScrollsSince(scrollCalls, O, from) {
  return scrollCalls
    .slice(from)
    .filter(c => c.el === O && typeof c.top === 'number');
}

/**
 * BUG 1 REPRO (remote webapp): the user scrolls up by touch/scrollbar
 * (plain ``scroll`` events, no wheel) while a task streams.  The chat
 * must NOT auto-scroll back to the end.
 */
async function testScrolledUpUserIsNotYankedToBottom() {
  const {win, posted, scrollCalls} = makeWebview({remote: true});
  const O = win.document.getElementById('output');
  const geo = {sh: 3000, ch: 500};
  fakeGeometry(O, geo);
  startRunningTask(win, posted);
  send(win, {type: 'tool_call', name: 'Bash', command: 'make'});

  // Sanity: pinned at the bottom, streaming DOES tail the chat.
  O.scrollTop = geo.sh - geo.ch;
  await streamOutput(win, geo, 'a'.repeat(100));
  assert.strictEqual(
    O.scrollTop,
    geo.sh - geo.ch,
    'sanity: chat must tail streamed output while pinned at the end',
  );

  // The user drags the view up to read an earlier message (touch or
  // scrollbar → only 'scroll' events fire; there is NO wheel event in
  // the remote webapp on a phone).
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

/**
 * BUG 2 REPRO: the user uncollapses (expands) a collapsed event panel
 * to read it.  Streaming must NOT scroll the chat to the end while the
 * panel stays expanded — in the remote webapp AND in the extension
 * webview (plain body, no remote-chat class).
 */
async function testUncollapsedPanelIsNotScrolledAway(remote) {
  const {win, posted, scrollCalls} = makeWebview({remote});
  const O = win.document.getElementById('output');
  const geo = {sh: 3000, ch: 500};
  fakeGeometry(O, geo);
  startRunningTask(win, posted);

  // Two tool calls: streaming the second auto-collapses the first.
  send(win, {type: 'tool_call', name: 'Bash', command: 'ls'});
  send(win, {type: 'tool_call', name: 'Bash', command: 'pwd'});
  const panels = O.querySelectorAll(':scope .ev.tc');
  assert.strictEqual(panels.length, 2, 'two tool-call panels rendered');
  const first = panels[0];
  assert.ok(
    first.classList.contains('collapsed'),
    'older tool panel must be auto-collapsed',
  );

  O.scrollTop = geo.sh - geo.ch; // user is at the end before expanding

  // The user clicks the collapsed panel's header to read it.
  // Expanding makes the content taller (the click handler reads the
  // grown geometry after toggling the class — layout is synchronous
  // in a real engine), so the view is no longer at the end.
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
  await flushTimeouts(win); // release addCollapse's click-scoped _noScroll

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

/**
 * REQUIRED BEHAVIOR: auto-scroll works while the user is at the very
 * end, and RESUMES once a scrolled-up user returns all the way to the
 * end (again via plain scroll events — touch/scrollbar).
 */
async function testAutoScrollResumesAtTheVeryEnd() {
  const {win, posted, scrollCalls} = makeWebview({remote: true});
  const O = win.document.getElementById('output');
  const geo = {sh: 3000, ch: 500};
  fakeGeometry(O, geo);
  startRunningTask(win, posted);
  send(win, {type: 'tool_call', name: 'Bash', command: 'make'});

  // Scroll up (lock engages), then back to the very end (lock must
  // release).
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

  // A collapsed→expanded→re-scrolled-to-end cycle must also resume.
  const panel = O.querySelector(':scope .ev.tc');
  send(win, {type: 'tool_call', name: 'Bash', command: 'pwd'});
  const header = panel.querySelector('.collapse-header');
  geo.sh += 400; // expansion grows the content (see bug-2 test)
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
  userScrollTo(win, O, geo.sh - geo.ch); // user returns to the end
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

/**
 * COLLAPSING (not expanding) a panel while pinned at the end must NOT
 * suspend tailing — the user is done reading it.
 */
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
  // Expand …
  header.dispatchEvent(
    new win.MouseEvent('click', {bubbles: true, cancelable: true}),
  );
  await flushTimeouts(win);
  // … the user reads it, scrolls to the very end (releases the hold) …
  userScrollTo(win, O, geo.sh - geo.ch);
  // … and collapses it again while staying at the end.
  header.dispatchEvent(
    new win.MouseEvent('click', {bubbles: true, cancelable: true}),
  );
  assert.ok(first.classList.contains('collapsed'), 'panel re-collapsed');
  geo.sh -= 400; // collapsed content shrinks the chat
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

/**
 * RACE: the ``scroll`` event of sb()'s own programmatic scroll fires
 * AFTER more content has already grown the chat (constant occurrence
 * while streaming fast).  That event is not user intent and must NOT
 * engage the suspension — tailing must continue.
 */
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

  // Stream: sb() scrolls to the (new) end, but its scroll event is
  // still in flight …
  await streamOutput(win, geo, 'a'.repeat(50));
  assert.ok(pendingScrollEvents.length > 0, 'an auto-scroll happened');
  // … while even more content arrives, so when the event finally
  // fires the view is no longer at the very end.
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

/**
 * The rAF scheduled by sb() must re-check the suspension when it RUNS:
 * if the user scrolls up in the same frame (mutation first, user
 * scroll second), the already-scheduled auto-scroll must stand down.
 */
async function testRafRechecksLockAtExecutionTime() {
  const {win, posted, scrollCalls} = makeWebview({remote: true});
  const O = win.document.getElementById('output');
  const geo = {sh: 3000, ch: 500};
  fakeGeometry(O, geo);
  startRunningTask(win, posted);
  send(win, {type: 'tool_call', name: 'Bash', command: 'make'});
  O.scrollTop = geo.sh - geo.ch;
  await streamOutput(win, geo, 'a'.repeat(50)); // tailing engaged

  const before = scrollCalls.length;
  // Mutation schedules the auto-scroll rAF …
  send(win, {type: 'system_output', text: 'b'.repeat(50) + '\n'});
  geo.sh += 200;
  // … and the user scrolls up BEFORE the frame runs.
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

/**
 * A no-op auto-scroll (already at the very end) must not leave a stale
 * "programmatic scroll in flight" mark that would swallow the user's
 * NEXT real scroll-up.
 */
async function testNoStaleProgrammaticMarkAfterNoopScroll() {
  const {win, posted, scrollCalls} = makeWebview({remote: true});
  const O = win.document.getElementById('output');
  const geo = {sh: 3000, ch: 500};
  fakeGeometry(O, geo);
  startRunningTask(win, posted);
  send(win, {type: 'tool_call', name: 'Bash', command: 'make'});
  O.scrollTop = geo.sh - geo.ch;

  // Mutation without growth: sb() runs but the position cannot move,
  // so the engine fires NO scroll event.  A stale "programmatic scroll
  // in flight" mark recorded here would keep pointing at this old
  // bottom position (2500).
  send(win, {type: 'system_output', text: 'a'.repeat(10) + '\n'});
  await nextFrames(win);

  // Content then grows silently (the user's position no longer rests
  // at the end) and the user scrolls up — landing exactly AT the old
  // bottom the stale mark points to.  It must still engage the
  // suspension: no auto-scroll ever happened after the no-op.
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

/**
 * An auto-scroll frame already scheduled when the user CLICKS a panel
 * header (its ``_noScroll`` click window) must stand down when it runs
 * — otherwise the click's own layout change scrolls the chat to the
 * end (e.g. collapsing a tall panel while reading near it).
 */
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

  // Streamed output schedules the auto-scroll frame …
  send(win, {type: 'system_output', text: 'a'.repeat(50) + '\n'});
  geo.sh += 200;
  await flushTimeouts(win); // let the MutationObserver deliver → sb()

  // … then the user clicks the (expanded) newest panel's header to
  // collapse it, and the scheduled frame fires INSIDE the click's
  // _noScroll window (rendering frames may precede 0ms timers).
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

  // The click window closes; tailing keeps working afterwards.
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

/**
 * COALESCED-EVENT RACE: the user scrolls up while sb()'s own scroll
 * event is still in flight — engines coalesce the two into ONE event
 * showing the USER's position.  That event is below the auto-scroll's
 * landing position, so it MUST count as the user scrolling up.
 */
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

  // sb() scrolls to the end; its scroll event is still in flight …
  await streamOutput(win, geo, 'a'.repeat(50));
  assert.ok(pendingScrollEvents.length > 0, 'an auto-scroll happened');
  // … when the user scrolls up.  The engine delivers ONE coalesced
  // event at the user's position.
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

/**
 * The task-panel "Uncollapse Chats" button is another way the user
 * uncollapses event panels: it must hold auto-scroll exactly like a
 * panel-header expansion — and NOT hold when nothing grew (view still
 * at the very end), nor when collapsing.
 */
async function testUncollapseChatsButtonHoldsScroll() {
  const {win, posted, scrollCalls} = makeWebview({remote: true});
  const O = win.document.getElementById('output');
  const geo = {sh: 3000, ch: 500};
  fakeGeometry(O, geo);
  startRunningTask(win, posted);
  send(win, {type: 'setTaskText', text: 'demo task'});
  send(win, {type: 'tool_call', name: 'Bash', command: 'ls'});
  send(win, {type: 'tool_call', name: 'Bash', command: 'pwd'});
  const btn = win.document.getElementById('task-panel-collapse-btn');
  assert.ok(btn, 'task-panel collapse button must exist');
  O.scrollTop = geo.sh - geo.ch; // user at the end

  // Click "Uncollapse Chats": panels above expand, content grows, the
  // view is no longer at the end → hold.
  geo.sh += 600;
  btn.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  const before = scrollCalls.length;
  const posBefore = O.scrollTop;
  await streamOutput(win, geo, 'a'.repeat(50));
  assert.strictEqual(
    autoScrollsSince(scrollCalls, O, before).length,
    0,
    'BUG: the chat auto-scrolled to the end right after the user ' +
      'clicked "Uncollapse Chats"',
  );
  assert.strictEqual(O.scrollTop, posBefore, 'reading position preserved');

  // Click "Collapse Chats" (second click): content shrinks back; the
  // user returns to the end → tailing resumes (no hold from collapse).
  geo.sh -= 600;
  btn.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  userScrollTo(win, O, geo.sh - geo.ch);
  const before2 = scrollCalls.length;
  await streamOutput(win, geo, 'b'.repeat(50));
  assert.ok(
    autoScrollsSince(scrollCalls, O, before2).length > 0,
    'collapsing via the button must not hold tailing',
  );

  // Click "Uncollapse Chats" again while it changes NOTHING (view
  // stays at the very end): tailing must keep working.
  btn.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  const before3 = scrollCalls.length;
  await streamOutput(win, geo, 'c'.repeat(50));
  assert.ok(
    autoScrollsSince(scrollCalls, O, before3).length > 0,
    'an expansion that leaves the view at the very end must not ' +
      'suspend tailing',
  );
  win.close();
  console.log('  ok - "Uncollapse Chats" button holds; collapse does not');
}

/**
 * Expand→collapse cycle at the end WITHOUT any intervening user
 * scroll: collapsing shrinks the content back so the view rests at
 * the very end again — the expansion hold must release by itself and
 * tailing must resume.  A collapse that still leaves the user far
 * from the end must NOT release the hold.
 */
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
  O.scrollTop = geo.sh - geo.ch; // at the end

  // Expand (content grows → hold engages) …
  geo.sh += 400;
  header.dispatchEvent(
    new win.MouseEvent('click', {bubbles: true, cancelable: true}),
  );
  await flushTimeouts(win);
  // … then collapse right away (content shrinks back → the view is at
  // the very end again → the hold releases with NO user scroll).
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

  // Counterpart: a collapse while the user is far from the end keeps
  // the hold (the user is still reading up there).
  userScrollTo(win, O, 300);
  geo.sh += 400;
  header.dispatchEvent(
    new win.MouseEvent('click', {bubbles: true, cancelable: true}),
  ); // expand again
  await flushTimeouts(win);
  geo.sh -= 400;
  header.dispatchEvent(
    new win.MouseEvent('click', {bubbles: true, cancelable: true}),
  ); // collapse — but the view is nowhere near the end
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

/**
 * Expanding a panel that does NOT grow the content (the view stays at
 * the very end) must not suspend tailing — the requirement is that
 * auto-scroll keeps working while the user is at the end.
 */
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
  O.scrollTop = geo.sh - geo.ch; // at the end

  header.dispatchEvent(
    new win.MouseEvent('click', {bubbles: true, cancelable: true}),
  ); // expansion changes nothing height-wise
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
  await testUncollapseChatsButtonHoldsScroll();
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
