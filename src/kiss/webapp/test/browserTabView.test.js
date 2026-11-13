// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// E2E tests for the remote webapp's streamed browser tab view
// (browserTab.js): the address bar follows browserState unless it is
// being edited, Enter / Escape in it, the navigation error bar, the
// first frame replacing the placeholder, viewport reports on show /
// hide / resubscribe, and dispose: a tab hidden again before the
// layout tick reports nothing, and a hide already retracted the
// viewport so a later dispose does not repeat it, and touch: a tap is
// a click, a gesture a second finger joined, a cancelled touch or one
// the view was hidden under is not; a finger that goes down asks the
// daemon what a drag to the right would scroll, and only its answer
// frees that drag for the phone layout's swipe back.  jsdom lays
// nothing out, so the screen's size is stubbed.

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const {
  captureTimers,
  fireCapturedTimers,
  MEDIA,
} = require('./simplify2_harness');
const {IME_FORMS} = require('./ui_antipattern_harness');

function makeView() {
  const dom = new JSDOM('<!DOCTYPE html><body></body>', {
    runScripts: 'outside-only',
  });
  const win = dom.window;
  const timers = captureTimers(win);
  win.eval(fs.readFileSync(path.join(MEDIA, 'browserTab.js'), 'utf8'));
  const sent = [];
  const view = win.BrowserTabView.create('tab-1', {
    send: msg => sent.push(JSON.parse(JSON.stringify(msg))),
  });
  win.document.body.appendChild(view.el);
  const screen = view.el.querySelector('.browser-screen');
  Object.defineProperty(screen, 'clientWidth', {value: 640});
  Object.defineProperty(screen, 'clientHeight', {value: 480});
  // fireCapturedTimers(win, timers) runs the deferred layout tick now.
  return {win, view, sent, timers};
}

function keydown(win, el, key, init) {
  el.dispatchEvent(
    new win.KeyboardEvent('keydown', Object.assign({key, bubbles: true}, init)),
  );
}

/** A touch event as the handlers read it (jsdom has no TouchEvent). */
function touch(win, el, type, touches, changed) {
  const e = new win.Event(type, {bubbles: true, cancelable: true});
  e.touches = touches;
  e.changedTouches = changed || touches;
  el.dispatchEvent(e);
}

async function testAddressBarAndState() {
  const {win, view, sent} = makeView();
  const url = view.el.querySelector('.browser-url');
  const [back, fwd] = view.el.querySelectorAll('.browser-nav-btn');
  view.state({url: 'https://a.test/', canGoBack: false, canGoForward: true});
  assert.strictEqual(url.value, 'https://a.test/');
  assert.strictEqual(back.disabled, true);
  assert.strictEqual(fwd.disabled, false);

  // A state arriving while the user types does not clobber the field;
  // leaving the field without navigating restores the page's URL.
  url.focus();
  url.value = 'b.test';
  view.state({url: 'https://c.test/', canGoBack: true});
  assert.strictEqual(url.value, 'b.test');
  keydown(win, url, 'Escape');
  assert.strictEqual(url.value, 'https://c.test/');
  url.focus();
  url.value = ' d.test ';
  keydown(win, url, 'Enter');
  assert.deepStrictEqual(sent.pop(), {
    type: 'browserNavigate',
    tab_id: 'tab-1',
    action: 'go',
    url: 'd.test',
  });
  url.focus();
  url.value = 'typing';
  url.blur();
  assert.strictEqual(url.value, 'https://c.test/');

  // An IME's Enter commits the composition and its Escape cancels it:
  // neither is the field's shortcut (W1 F3: they navigated to the
  // half-composed text and reset the field).  Chromium flags such keys
  // with isComposing, or with keyCode 229 alone on some platforms.
  url.focus();
  url.value = '日本';
  for (const init of IME_FORMS) {
    for (const key of ['Enter', 'Escape']) {
      const e = new win.KeyboardEvent(
        'keydown',
        Object.assign({key, bubbles: true, cancelable: true}, init),
      );
      url.dispatchEvent(e);
      assert.strictEqual(e.defaultPrevented, false, key + ' composing');
      assert.strictEqual(url.value, '日本', key + ' composing keeps the text');
      assert.strictEqual(win.document.activeElement, url, key + ' composing');
    }
  }
  assert.strictEqual(sent.length, 0, 'a composing Enter navigates nowhere');
  // The composition committed, the next Enter navigates.
  keydown(win, url, 'Enter');
  assert.strictEqual(sent.pop().url, '日本');

  back.click();
  assert.strictEqual(sent.pop().action, 'back');
  fwd.click();
  assert.strictEqual(sent.length, 0, 'a disabled Forward sends nothing');
  view.state({url: 'https://c.test/', canGoForward: true});
  fwd.click();
  assert.strictEqual(sent.pop().action, 'forward');

  // The error bar shows a navigation error until the next state.
  const errorBar = view.el.querySelector('.browser-error');
  view.error('net::ERR_NAME_NOT_RESOLVED');
  assert.strictEqual(errorBar.style.display, '');
  assert.strictEqual(errorBar.textContent, 'net::ERR_NAME_NOT_RESOLVED');
  view.state({url: 'https://c.test/'});
  assert.strictEqual(errorBar.style.display, 'none');
  view.error('');
  assert.strictEqual(errorBar.style.display, 'none');

  // The first frame replaces the placeholder and records the page size.
  assert.ok(view.el.querySelector('.browser-placeholder'));
  view.frame({data: 'AAAA', width: 1200, height: 900});
  assert.strictEqual(view.el.querySelector('.browser-placeholder'), null);
  assert.strictEqual(view.pageWidth, 1200);
  assert.strictEqual(view.pageHeight, 900);
  view.frame({});
  view.frame({data: 'BBBB'});
  assert.strictEqual(view.pageWidth, 1200);

  view.setBadge('chromium', true);
  const badge = view.el.querySelector('.browser-badge');
  assert.strictEqual(badge.textContent, 'chromium');
  assert.ok(badge.title.includes('default browser'));
  view.setBadge('firefox', false);
  assert.ok(!badge.title.includes('default'));
  view.setBadge('firefox', false, 'Chrome is not installed.');
  assert.ok(badge.title.includes('Showing firefox instead.'));
}

async function testViewportAndDispose() {
  const {win, view, sent, timers} = makeView();
  view.setVisible(true);
  fireCapturedTimers(win, timers);
  assert.deepStrictEqual(sent, [
    {
      type: 'browserViewport',
      tab_id: 'tab-1',
      width: 640,
      height: 480,
      visible: true,
    },
  ]);
  view.setVisible(true);
  fireCapturedTimers(win, timers);
  assert.strictEqual(sent.length, 1, 'an unchanged viewport is not resent');
  view.resubscribe();
  assert.strictEqual(sent.length, 2, 'resubscribe reports again');
  view.setVisible(false);
  assert.strictEqual(sent[2].visible, false);
  view.setVisible(false);
  assert.strictEqual(sent.length, 3);
  view.resubscribe();
  assert.strictEqual(sent.length, 3, 'hidden: nothing to report');

  view.setVisible(true);
  fireCapturedTimers(win, timers);
  assert.strictEqual(sent.length, 4);
  view.dispose();
  view.dispose();
  assert.ok(view.disposed);
  assert.strictEqual(sent.length, 5);
  assert.strictEqual(sent[4].visible, false);
  view.frame({data: 'AAAA', width: 10, height: 10});
  assert.strictEqual(view.pageWidth, 0, 'no frame after dispose');
}

async function testShowHideBeforeTick() {
  const {win, view, sent, timers} = makeView();
  view.setVisible(true);
  view.setVisible(false);
  fireCapturedTimers(win, timers);
  assert.deepStrictEqual(sent, [], 'a tab hidden before layout reports nothing');
  view.dispose();
  assert.deepStrictEqual(sent, [], 'and dispose has nothing to retract');
}

async function testHideThenDispose() {
  const {win, view, sent, timers} = makeView();
  view.setVisible(true);
  fireCapturedTimers(win, timers);
  assert.strictEqual(sent.length, 1);
  assert.strictEqual(sent[0].visible, true);
  view.setVisible(false);
  assert.strictEqual(sent.length, 2);
  assert.strictEqual(sent[1].visible, false);
  view.dispose();
  assert.strictEqual(sent.length, 2, 'hide already retracted the viewport');
  view.setVisible(true);
  fireCapturedTimers(win, timers);
  assert.strictEqual(sent.length, 2, 'a disposed view never reports');
}

async function testTwoFingerTouchIsNoClick() {
  const {win, view, sent} = makeView();
  const screen = view.el.querySelector('.browser-screen');
  const one = {clientX: 50, clientY: 60};
  const two = {clientX: 150, clientY: 60};
  // What the remote pointer was sent (a finger's question apart).
  const pointer = () => sent.filter(m => m.event.kind === 'mouse');
  // A second finger joins a tap in progress (a pinch), both move, both lift.
  touch(win, screen, 'touchstart', [one]);
  touch(win, screen, 'touchstart', [one, two]);
  touch(win, screen, 'touchmove', [{clientX: 40, clientY: 60}, two]);
  touch(win, screen, 'touchend', [one], [two]);
  touch(win, screen, 'touchend', [], [one]);
  assert.deepStrictEqual(pointer(), [], 'a two-finger gesture is not a click');
  assert.strictEqual(sent.length, 1, 'only the first finger asked');
  // A cancelled touch is forgotten.
  touch(win, screen, 'touchstart', [one]);
  touch(win, screen, 'touchcancel', [], [one]);
  touch(win, screen, 'touchend', [], [one]);
  assert.deepStrictEqual(pointer(), [], 'a cancelled touch is not a click');
  // A one-finger tap is a click.
  touch(win, screen, 'touchstart', [one]);
  touch(win, screen, 'touchend', [], [one]);
  assert.deepStrictEqual(
    pointer().map(m => m.event.action),
    ['mousePressed', 'mouseReleased'],
  );
  // A one-finger drag scrolls, and lifting it is no click.  The remote
  // pointer goes to the finger before each wheel.
  sent.length = 0;
  touch(win, screen, 'touchstart', [one]);
  touch(win, screen, 'touchmove', [{clientX: 90, clientY: 60}]);
  touch(win, screen, 'touchmove', [{clientX: 100, clientY: 70}]);
  touch(win, screen, 'touchend', [], [{clientX: 100, clientY: 70}]);
  assert.deepStrictEqual(
    pointer().map(m => [
      m.event.action,
      m.event.x,
      m.event.deltaX,
      m.event.deltaY,
    ]),
    [
      ['mouseMoved', 90, undefined, undefined],
      ['mouseWheel', 90, -40, 0],
      ['mouseMoved', 100, undefined, undefined],
      ['mouseWheel', 100, -10, -10],
    ],
  );
  // A finger down when the view hides has no box left to be mapped
  // through: its lift is no click.
  sent.length = 0;
  touch(win, screen, 'touchstart', [one]);
  view.setVisible(false);
  touch(win, screen, 'touchend', [], [one]);
  assert.deepStrictEqual(pointer(), [], 'a touch the view was hidden under');
}

// A finger that goes down on the screen asks the daemon what is under
// it.  whenDragRightIsFree(callback) calls back when a drag to the
// right from there scrolls nothing on the page (the phone layout then
// turns back to the chat): at once when the daemon has said so, else
// when it says so within 600 ms.  Without its word the drag stays the
// page's.
async function testAFingerAsksWhatADragToTheRightScrolls() {
  const {win, view, sent, timers} = makeView();
  const screen = view.el.querySelector('.browser-screen');
  let free = 0;
  const count = () => free++;
  /** Put a finger down; the id its question carries. */
  const finger = () => {
    sent.length = 0;
    touch(win, screen, 'touchstart', [{clientX: 50, clientY: 60}]);
    assert.deepStrictEqual(
      sent.map(m => [m.type, m.tab_id, m.event.kind, m.event.x, m.event.y]),
      [['browserInput', 'tab-1', 'touch', 50, 60]],
    );
    return sent[0].event.id;
  };
  const answer = (id, scrollsLeft) => view.touched({id, scrollsLeft});

  // No finger has asked: nothing was said about the page.
  view.whenDragRightIsFree(count);
  assert.strictEqual(free, 0);

  // The daemon answered while the finger was down.
  let id = finger();
  answer(id, false);
  view.whenDragRightIsFree(count);
  assert.strictEqual(free, 1, 'nothing to scroll: free at once');
  const next = finger();
  assert.notStrictEqual(next, id, 'every finger asks under its own id');
  answer(next, true);
  view.whenDragRightIsFree(count);
  fireCapturedTimers(win, timers);
  assert.strictEqual(free, 1, 'the page scrolls: never free');

  // The answer comes after the finger lifted.
  id = finger();
  view.whenDragRightIsFree(count);
  assert.strictEqual(free, 1, 'not known yet');
  answer('another-finger', false);
  assert.strictEqual(free, 1, "another finger's answer");
  answer(id, false);
  assert.strictEqual(free, 2, 'the late answer frees the drag');
  answer(id, false);
  assert.strictEqual(free, 2, 'once');
  id = finger();
  view.whenDragRightIsFree(count);
  answer(id, true);
  fireCapturedTimers(win, timers);
  assert.strictEqual(free, 2, 'the late answer: the page scrolls');

  // No answer in time (an older daemon, a dead link): the drag stays
  // the page's, whatever is said later.
  id = finger();
  const armed = timers.length;
  view.whenDragRightIsFree(count);
  assert.strictEqual(timers[armed].ms, 600);
  fireCapturedTimers(win, timers, armed);
  answer(id, false);
  assert.strictEqual(free, 2, 'an answer after the wait turns nothing');

  // The next finger ends the wait for the last one, and its own wait
  // is not cut short by the last one's.
  const last = finger();
  const lastWait = timers.length;
  view.whenDragRightIsFree(count);
  const lastTimer = timers[lastWait];
  id = finger();
  view.whenDragRightIsFree(count);
  fireCapturedTimers(win, [lastTimer]);
  answer(last, false);
  assert.strictEqual(free, 2, 'the last finger is forgotten');
  answer(id, false);
  assert.strictEqual(free, 3, 'the next finger still waits for its answer');

  // The wait also ends when the view is hidden and when it is
  // disposed: that drag turns no screen any more.
  const hide = () => view.setVisible(false);
  for (const leave of [hide, view.dispose]) {
    id = finger();
    view.whenDragRightIsFree(count);
    leave();
    answer(id, false);
    fireCapturedTimers(win, timers);
    assert.strictEqual(free, 3);
  }
}

// A keyup goes to wherever the focus is then: one that leaves the
// screen (the address field, another tab, another application) never
// reaches it, and the remote page kept the key pressed.  Keys forwarded
// down are released when the screen loses focus, when the view is
// hidden and on dispose.  A keyup the screen gets is forwarded only for
// a key it forwarded down and has not released: not a second time
// after the blur, and not for Enter or Escape typed in the address
// field (they hand the focus to the screen before they go up) nor for
// the V of a paste whose Ctrl went up first.
async function testHeldKeysReleasedWithFocus() {
  const {win, view, sent} = makeView();
  const screen = view.el.querySelector('.browser-screen');
  const keys = () =>
    sent
      .filter(m => m.type === 'browserInput' && m.event.kind === 'key')
      .map(m => m.event.action + ':' + m.event.key);
  screen.focus();
  keydown(win, screen, 'Shift', {shiftKey: true});
  keydown(win, screen, 'a', {shiftKey: true});
  assert.deepStrictEqual(keys(), ['down:Shift', 'down:a']);
  // The user clicks into the address field with both keys still down.
  view.el.querySelector('.browser-url').focus();
  assert.deepStrictEqual(keys().slice(2), ['up:Shift', 'up:a']);
  assert.ok(
    sent
      .filter(m => m.event && m.event.action === 'up')
      .every(m => m.event.shift),
    'the releases carry the modifiers of the press',
  );
  const keyup = key =>
    screen.dispatchEvent(new win.KeyboardEvent('keyup', {key, bubbles: true}));
  // Back on the screen, the key's own keyup: it was released already.
  screen.focus();
  keyup('a');
  const url = view.el.querySelector('.browser-url');
  for (const key of ['Enter', 'Escape']) {
    url.focus();
    keydown(win, url, key);
    assert.strictEqual(win.document.activeElement, screen);
    keyup(key);
  }
  keydown(win, screen, 'Control', {ctrlKey: true});
  keydown(win, screen, 'v', {ctrlKey: true});
  keyup('Control');
  keyup('v');
  assert.deepStrictEqual(keys().slice(4), ['down:Control', 'up:Control']);
  // Hidden with a key down: released.  Disposed with one: released too.
  keydown(win, screen, 'b');
  view.setVisible(false);
  keydown(win, screen, 'c');
  view.dispose();
  assert.deepStrictEqual(keys().slice(6), ['down:b', 'up:b', 'down:c', 'up:c']);
}

(async () => {
  await testAddressBarAndState();
  await testViewportAndDispose();
  await testShowHideBeforeTick();
  await testHideThenDispose();
  await testTwoFingerTouchIsNoClick();
  await testAFingerAsksWhatADragToTheRightScrolls();
  await testHeldKeysReleasedWithFocus();
  console.log('browserTabView.test.js passed');
})().catch(err => {
  console.error(err);
  process.exit(1);
});
