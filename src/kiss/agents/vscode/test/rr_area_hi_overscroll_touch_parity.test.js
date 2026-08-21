// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// I-R3: wheel and touch overscroll share one edge handler. Chaining to
// the previous/next task must behave identically whichever input device
// crossed the edge, and a scroll that leaves the edge must reset the
// accumulator for both.

/* global require, process, console, __dirname, global, setTimeout, clearTimeout */

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
  win.Element.prototype.scrollTo = function () {};
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
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function fakeGeometry(el) {
  Object.defineProperty(el, 'scrollHeight', {value: 3000, configurable: true});
  Object.defineProperty(el, 'clientHeight', {value: 500, configurable: true});
}

function setup() {
  const {win, posted} = makeWebview();
  const tabId = posted.find(m => m.type === 'ready').tabId;
  win._testApi.hideWelcome();
  const O = win.document.getElementById('output');
  fakeGeometry(O);
  send(win, {
    type: 'task_events',
    tabId,
    chat_id: 'chat-t',
    task_id: '42',
    task: 'My old task',
    events: [
      {type: 'task_start', task: 'My old task'},
      {type: 'system_output', text: 'hello\n'},
    ],
  });
  return {win, posted, tabId, O};
}

function getAdjacent(posted) {
  return posted.filter(m => m.type === 'getAdjacentTask');
}

function touchEvent(win, type, y) {
  // jsdom has no Touch/TouchEvent constructors; the handlers only read
  // e.touches.length and e.touches[0].clientY.
  const ev = new win.Event(type, {bubbles: true});
  ev.touches = [{identifier: 1, clientY: y}];
  return ev;
}

function touchSequence(win, O, deltas) {
  // One finger: touchstart at y=300, then a move per delta (positive
  // delta scrolls towards the next task, matching wheel deltaY).
  let y = 300;
  O.dispatchEvent(touchEvent(win, 'touchstart', y));
  for (const d of deltas) {
    y -= d;
    O.dispatchEvent(touchEvent(win, 'touchmove', y));
  }
}

function testTouchPrevAtTop() {
  const {win, posted, O} = setup();
  O.scrollTop = 0;
  touchSequence(win, O, Array(10).fill(-50));
  const adj = getAdjacent(posted);
  assert.ok(adj.length > 0, 'touch overscroll at top must chain to prev');
  assert.strictEqual(adj[0].direction, 'prev');
  assert.strictEqual(adj[0].taskId, '42');
  win.close();
  console.log('  ok - touch overscroll at the top requests the prev task');
}

function testTouchNextAtBottom() {
  const {win, posted, O} = setup();
  O.scrollTop = 2500; // scrollTop + clientHeight == scrollHeight
  touchSequence(win, O, Array(10).fill(50));
  const adj = getAdjacent(posted);
  assert.ok(adj.length > 0, 'touch overscroll at bottom must chain to next');
  assert.strictEqual(adj[0].direction, 'next');
  win.close();
  console.log('  ok - touch overscroll at the bottom requests the next task');
}

function testWheelParity() {
  const {win, posted, O} = setup();
  O.scrollTop = 0;
  for (let i = 0; i < 10; i++) {
    O.dispatchEvent(
      new win.WheelEvent('wheel', {
        deltaY: -50,
        bubbles: true,
        cancelable: true,
      }),
    );
  }
  const adj = getAdjacent(posted);
  assert.ok(adj.length > 0, 'wheel overscroll at top must chain to prev');
  assert.strictEqual(adj[0].direction, 'prev');
  win.close();
  console.log('  ok - wheel overscroll behaves the same as touch');
}

function testMidScrollResetsAccumulator() {
  const {win, posted, O} = setup();
  // Almost enough overscroll at the top, then a scroll away from the
  // edge, then a bit more at the top: the accumulator must have reset,
  // so no adjacent-task request fires.
  O.scrollTop = 0;
  touchSequence(win, O, [-60, -60]);
  O.scrollTop = 100;
  touchSequence(win, O, [-10]);
  O.scrollTop = 0;
  touchSequence(win, O, [-60]);
  assert.strictEqual(
    getAdjacent(posted).length,
    0,
    'leaving the edge must reset the touch overscroll accumulator',
  );
  win.close();
  console.log('  ok - scrolling off the edge resets the accumulator');
}

function main() {
  testTouchPrevAtTop();
  testTouchNextAtBottom();
  testWheelParity();
  testMidScrollResetsAccumulator();
  console.log('rr_area_hi_overscroll_touch_parity: all tests passed');
  process.exit(0);
}

try {
  main();
} catch (err) {
  console.error(err);
  process.exit(1);
}
