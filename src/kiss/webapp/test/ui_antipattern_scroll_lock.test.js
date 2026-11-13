// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end (jsdom) test for anti-pattern A7 (audit-main-2 M4) in
// media/main.js: while the user has scrolled up (scroll lock) and new
// output arrives below, a "New output ↓" button appears; it jumps to the
// bottom and releases the lock, and it hides once the user scrolls to
// the bottom themselves.  jsdom has no layout, so the transcript's
// scroll geometry is given fixed values here.  Fails on the old code.
'use strict';

const assert = require('assert');
const h = require('./ui_antipattern_harness');

const {test, report} = h.makeRunner();

/** Give #output a 1000 px tall content in a 200 px viewport; the test
 *  grows the content through the returned `sh`. */
function giveScrollGeometry(out) {
  const geo = {sh: 1000};
  Object.defineProperty(out, 'scrollHeight', {
    get: () => geo.sh,
    configurable: true,
  });
  Object.defineProperty(out, 'clientHeight', {value: 200, configurable: true});
  let top = 0;
  Object.defineProperty(out, 'scrollTop', {
    get: () => top,
    set: v => {
      top = v;
    },
    configurable: true,
  });
  return geo;
}

function userScrollTo(win, out, top) {
  out.scrollTop = top;
  out.dispatchEvent(new win.Event('scroll'));
}

function newOutput(win, tabId, text) {
  h.send(win, {type: 'result', summary: text, success: true, tabId});
}

async function main() {
  await test('"New output ↓" appears under the lock, jumps to the bottom, hides at the bottom', () => {
    const {win} = h.makePage();
    const tabId = win._testApi.getActiveTabId();
    const out = h.byId(win, 'output');
    const geo = giveScrollGeometry(out);
    const btn = out.nextElementSibling;
    assert.ok(
      btn && btn.classList.contains('new-output-btn'),
      'the button sits right after the transcript, outside its scroller',
    );
    assert.strictEqual(btn.textContent, 'New output ↓');
    assert.ok(btn.hidden, 'hidden while nothing is pending');

    // The user scrolls up to read; output keeps arriving below.
    userScrollTo(win, out, 100);
    newOutput(win, tabId, '<p>more</p>');
    assert.ok(
      !btn.hidden,
      'the button appears when an auto-scroll is suppressed',
    );
    assert.strictEqual(out.scrollTop, 100, 'the lock still holds the view');

    h.click(win, btn);
    assert.strictEqual(out.scrollTop, 800, 'the click jumps to the bottom');
    assert.ok(btn.hidden, 'and hides the button');
    // The content grows: only a view that follows it ends up lower.
    geo.sh = 1200;
    newOutput(win, tabId, '<p>even more</p>');
    assert.strictEqual(out.scrollTop, 1000, 'auto-scroll follows again');
    assert.ok(btn.hidden);

    // Scroll up again, then back down by hand: the button goes away.
    userScrollTo(win, out, 300);
    newOutput(win, tabId, '<p>later</p>');
    assert.ok(!btn.hidden);
    userScrollTo(win, out, 1000);
    assert.ok(btn.hidden, 'reaching the bottom hides it');
    win.close();
  });

  report('ui_antipattern_scroll_lock');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
