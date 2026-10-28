// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// Audit 2026-10-12 (webapp media): mobileSwipe.js had no end-to-end
// test.  `MobileSwipe.install` is called once per page (main.js) and
// once per preview iframe (bootstrapHtml) and its listeners live as
// long as the document, so every rule the detector applies is driven
// here through real touch events on a jsdom document:
//
//   * a fast sideways drag of SWIPE_MIN_PX (60) or more is a swipe,
//     'left' when the finger moved right to left;
//   * a short, slow, or mostly vertical drag is not;
//   * a touch that starts inside an `ignore` element is not;
//   * a second finger (pinch) cancels the touch;
//   * a sideways scroller that moved, or a `scrollPositions()` entry
//     that changed, means the finger scrolled;
//   * a selection made during the touch is a selection;
//   * touchcancel forgets the touch;
//   * bootstrapHtml() carries the detector into a preview document and
//     that copy posts {kissMobileSwipe} to the parent.

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');
const SOURCE = fs.readFileSync(path.join(MEDIA, 'mobileSwipe.js'), 'utf8');

function makePage(html) {
  const dom = new JSDOM(html || '<!DOCTYPE html><body><div id="page"></div></body>', {
    runScripts: 'outside-only',
    pretendToBeVisual: true,
  });
  dom.window.eval(SOURCE);
  return dom.window;
}

function touchEvent(win, type, target, points, changed) {
  const e = new win.Event(type, {bubbles: true, cancelable: true});
  Object.defineProperty(e, 'touches', {value: points});
  Object.defineProperty(e, 'changedTouches', {value: changed || points});
  target.dispatchEvent(e);
}

function point(x, y) {
  return {clientX: x, clientY: y};
}

/** One finger down at (x0, y0), up at (x1, y1). */
function drag(win, target, x0, y0, x1, y1) {
  touchEvent(win, 'touchstart', target, [point(x0, y0)]);
  touchEvent(win, 'touchmove', target, [point(x1, y1)]);
  touchEvent(win, 'touchend', target, [], [point(x1, y1)]);
}

function testSwipeRules() {
  const win = makePage();
  const doc = win.document;
  const page = doc.getElementById('page');
  const swipes = [];
  win.MobileSwipe.install(doc, (dir, e) => swipes.push([dir, e.type]));

  drag(win, page, 300, 100, 100, 110);
  assert.deepStrictEqual(swipes, [['left', 'touchend']], 'right-to-left is left');
  drag(win, page, 100, 100, 200, 100);
  assert.deepStrictEqual(swipes[1], ['right', 'touchend'], 'left-to-right is right');

  drag(win, page, 100, 100, 150, 100);
  assert.strictEqual(swipes.length, 2, 'a 50px drag is too short');
  drag(win, page, 100, 100, 200, 180);
  assert.strictEqual(swipes.length, 2, 'a mostly vertical drag is a scroll');
  // A finger that lifts without a change list, or two fingers lifting.
  touchEvent(win, 'touchstart', page, [point(100, 100)]);
  touchEvent(win, 'touchend', page, [], []);
  assert.strictEqual(swipes.length, 2, 'no changed touch, no swipe');

  // A pinch cancels the touch even when one finger lifts first.
  touchEvent(win, 'touchstart', page, [point(300, 100)]);
  touchEvent(win, 'touchmove', page, [point(250, 100), point(260, 120)]);
  touchEvent(win, 'touchend', page, [], [point(100, 100)]);
  assert.strictEqual(swipes.length, 2, 'a pinch is not a swipe');
  touchEvent(win, 'touchstart', page, [point(300, 100), point(310, 100)]);
  touchEvent(win, 'touchend', page, [], [point(100, 100)]);
  assert.strictEqual(swipes.length, 2, 'a two-finger start is not a swipe');

  // touchcancel forgets the touch.
  touchEvent(win, 'touchstart', page, [point(300, 100)]);
  touchEvent(win, 'touchcancel', page, []);
  touchEvent(win, 'touchend', page, [], [point(100, 100)]);
  assert.strictEqual(swipes.length, 2, 'a cancelled touch is not a swipe');

  // A slow drag is not a swipe (the detector reads the page's own clock).
  const realNow = win.Date.now;
  let now = realNow();
  win.Date.now = () => now;
  try {
    touchEvent(win, 'touchstart', page, [point(300, 100)]);
    now += 1600;
    touchEvent(win, 'touchend', page, [], [point(100, 100)]);
    assert.strictEqual(swipes.length, 2, 'a 1.6s drag is too slow');
    touchEvent(win, 'touchstart', page, [point(300, 100)]);
    now += 1400;
    touchEvent(win, 'touchend', page, [], [point(100, 100)]);
    assert.strictEqual(swipes.length, 3, 'a 1.4s drag is in time');
  } finally {
    win.Date.now = realNow;
  }
}

function testIgnoreAndScrollers() {
  const win = makePage(
    '<!DOCTYPE html><body>' +
      '<textarea id="task-input"></textarea>' +
      '<div id="wide" style="overflow-x: auto"><span id="inner">x</span></div>' +
      '<div id="plain"></div>' +
      '</body>',
  );
  const doc = win.document;
  const swipes = [];
  let positions = [0];
  win.MobileSwipe.install(doc, dir => swipes.push(dir), {
    ignore: '#task-input',
    scrollPositions: () => positions.slice(),
  });

  drag(win, doc.getElementById('task-input'), 300, 100, 100, 100);
  assert.deepStrictEqual(swipes, [], 'a touch in an ignored element never swipes');

  // A sideways scroller under the finger that scrolled during the touch.
  const wide = doc.getElementById('wide');
  Object.defineProperty(wide, 'scrollWidth', {value: 1000});
  Object.defineProperty(wide, 'clientWidth', {value: 300});
  const inner = doc.getElementById('inner');
  touchEvent(win, 'touchstart', inner, [point(300, 100)]);
  wide.scrollLeft = 120;
  touchEvent(win, 'touchend', inner, [], [point(100, 100)]);
  assert.deepStrictEqual(swipes, [], 'a scrolled ancestor means scrolling');
  // The same scroller left alone: a swipe.
  touchEvent(win, 'touchstart', inner, [point(300, 100)]);
  touchEvent(win, 'touchend', inner, [], [point(100, 100)]);
  assert.deepStrictEqual(swipes, ['left'], 'an unscrolled scroller still swipes');

  // A scroll position the DOM cannot report (Monaco) that changed.
  const plain = doc.getElementById('plain');
  touchEvent(win, 'touchstart', plain, [point(300, 100)]);
  positions = [40];
  touchEvent(win, 'touchend', plain, [], [point(100, 100)]);
  assert.deepStrictEqual(swipes, ['left'], 'a changed scrollPositions entry means scrolling');
  touchEvent(win, 'touchstart', plain, [point(300, 100)]);
  positions = [40, 0];
  touchEvent(win, 'touchend', plain, [], [point(100, 100)]);
  assert.deepStrictEqual(swipes, ['left'], 'a new scrollPositions entry means scrolling');

  // A selection made during the touch (long-press then drag).
  const sel = win.getSelection();
  touchEvent(win, 'touchstart', plain, [point(300, 100)]);
  const range = doc.createRange();
  range.selectNodeContents(inner);
  sel.removeAllRanges();
  sel.addRange(range);
  touchEvent(win, 'touchend', plain, [], [point(100, 100)]);
  assert.deepStrictEqual(swipes, ['left'], 'selecting text is not swiping');
  // A selection that already existed before the touch does not block.
  touchEvent(win, 'touchstart', plain, [point(300, 100)]);
  touchEvent(win, 'touchend', plain, [], [point(100, 100)]);
  assert.deepStrictEqual(swipes, ['left', 'left'], 'an older selection does not block');
}

function testPreviewBootstrap() {
  const win = makePage();
  const html = win.MobileSwipe.bootstrapHtml().trim();
  assert.ok(html.startsWith('<script'), 'the bootstrap is a script tag');
  // Evaluate that script in a second document whose parent is a sink.
  const preview = new JSDOM('<!DOCTYPE html><body><p id="p">text</p></body>', {
    runScripts: 'outside-only',
  }).window;
  const posted = [];
  Object.defineProperty(preview, 'parent', {
    value: {postMessage: (msg, origin) => posted.push([msg, origin])},
  });
  const code = html.replace(/^<script[^>]*>/, '').replace(/<\/script>$/, '');
  preview.eval(code);
  drag(preview, preview.document.getElementById('p'), 300, 100, 100, 100);
  // The message is built in the preview realm: compare by value.
  assert.deepStrictEqual(JSON.parse(JSON.stringify(posted)), [
    [{kissMobileSwipe: 'left'}, '*'],
  ]);
}

testSwipeRules();
testIgnoreAndScrollers();
testPreviewBootstrap();
console.log('audit1012_media_mobile_swipe.test.js passed');
