// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (jsdom) tests of the scrollbars that stay out of sight
// until the user scrolls (media/main.js, "Scrollbars stay out of
// sight").  The page marks its root `scrollbars-autohide`, under which
// main.css paints a scroller's thumb only while the scroller carries
// `scrolling`.  A scroller carries it from a scroll the user caused (a
// wheel turn, a touch drag or the lift that ends it, a scroll key,
// inside that scroller and at most a second before) until 1.2 s after
// the last such scroll.  A scroll the page made itself, as when a
// transcript follows the agent's output, shows no bar.
//
// jsdom lays nothing out, so the press on the bar itself (a press past
// the scroller's client box) and the thumb's colours are measured in a
// real Chromium by tests/webapp/test_slim_scrollbars.py.

'use strict';

const assert = require('assert');
const {
  makePage,
  recordTimers,
  fireTimers,
  sleep,
} = require('./simplify2_harness');

const SCROLL_KEYS = [
  ' ',
  'ArrowUp',
  'ArrowDown',
  'ArrowLeft',
  'ArrowRight',
  'PageUp',
  'PageDown',
  'Home',
  'End',
];

/** Dispatch the user's input `type` on `target`, as the browser does. */
function input(win, target, type, key) {
  const event =
    type === 'keydown'
      ? new win.KeyboardEvent(type, {key, bubbles: true})
      : new win.Event(type, {bubbles: true});
  target.dispatchEvent(event);
}

/** Dispatch the scroll event of `scroller` (it does not bubble). */
function scroll(win, scroller) {
  scroller.dispatchEvent(new win.Event('scroll'));
}

function showsBar(scroller) {
  return scroller.classList.contains('scrolling');
}

async function testBarsShowOnlyWhileTheUserScrolls() {
  // The bars' 1.2 s hide timers are recorded and fired by hand.
  let timers;
  const {win} = makePage({
    beforeScripts: w => {
      timers = recordTimers(w, 1200);
    },
  });
  const doc = win.document;
  // The two scrollers of the request: the sidebar's list and the chat.
  const list = doc.getElementById('history-list');
  const chat = doc.getElementById('output');
  const inList = list.appendChild(doc.createElement('span'));
  const inChat = chat.appendChild(doc.createElement('span'));

  assert.ok(
    doc.documentElement.classList.contains('scrollbars-autohide'),
    'the page asks main.css to keep its bars out of sight',
  );

  // The page scrolls the chat itself: nobody is scrolling.
  scroll(win, chat);
  assert.ok(!showsBar(chat), 'a scroll the user did not cause shows no bar');
  assert.strictEqual(timers.length, 0, 'and arms no timer');

  // The user turns the wheel over the chat.  A key that scrolls
  // nothing and a press that is not on a bar, both in the list, are
  // not scrolls: the list, scrolled by the page just then, stays bare.
  input(win, inChat, 'wheel');
  input(win, inList, 'keydown', 'a');
  input(win, inList, 'pointerdown');
  scroll(win, list);
  scroll(win, doc); // the document's own scroll has no scroller to mark
  assert.strictEqual(doc.querySelector('.scrolling'), null);
  scroll(win, chat);
  assert.ok(showsBar(chat), 'the scroller the user scrolls shows its bar');
  assert.strictEqual(doc.querySelectorAll('.scrolling').length, 1);

  // Scrolling on keeps the bar: each scroll starts the 1.2 s again.
  scroll(win, chat);
  assert.deepStrictEqual(
    timers.map(timer => timer.cleared),
    [true, false],
    'a further scroll replaces the pending hide timer',
  );
  fireTimers(timers);
  assert.ok(!showsBar(chat), 'the bar goes 1.2 s after the last scroll');

  // Every way of scrolling counts.  Each input lands in the scroller
  // the previous one did not, so only that input can explain the bar.
  const inputs = [['touchmove'], ['touchend']].concat(
    SCROLL_KEYS.map(key => ['keydown', key]),
  );
  inputs.forEach(([type, key], i) => {
    const [scroller, target] = i % 2 ? [chat, inChat] : [list, inList];
    const what = key === undefined ? type : `the "${key}" key`;
    input(win, target, type, key);
    scroll(win, scroller);
    assert.ok(showsBar(scroller), `${what} shows the scrolled bar`);
    fireTimers(timers);
    assert.ok(!showsBar(scroller), `${what}: the bar goes again`);
  });

  // More than a second after the user's last input a scroll is the
  // page's own again.
  await sleep(1050);
  scroll(win, chat);
  scroll(win, list);
  assert.strictEqual(
    doc.querySelector('.scrolling'),
    null,
    'a scroll long after the input shows no bar',
  );
  win.close();
  console.log('  ok - a bar shows only while the user scrolls its scroller');
}

async function main() {
  await testBarsShowOnlyWhileTheUserScrolls();
  console.log('scrollbarAutoHide.test.js: all passed.');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
