// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Horizontal swipe detection for the phone layout of the remote webapp.
// main.js turns each swipe into a step between its four mobile screens
// (chats, chat, file, task info):
//
//   const dispose = MobileSwipe.install(document, direction => {...}, {
//     ignore: '#task-input, .browser-view',   // touches here never swipe
//     scrollPositions: () => [...],           // sideways scroll offsets
//   });                                       // the DOM cannot report
//
// `direction` is 'left' (the finger moved right to left) or 'right'.
// A touch is a swipe when one finger travels at least SWIPE_MIN_PX
// sideways, clearly more sideways than up or down, within
// SWIPE_MAX_MS, and nothing under it used the drag for itself:
//
//   * a second finger at any point (a pinch) cancels it;
//   * a touch that starts inside an `ignore` element is left to that
//     element (the composer's own swipe gestures, the streamed browser
//     screen that forwards drags to the remote page);
//   * an ancestor that scrolls sideways (overflow-x) and did scroll
//     during the touch, or any `scrollPositions()` entry that changed
//     (a Monaco editor scrolls sideways through transforms, not
//     scrollLeft), means the finger was scrolling, not swiping;
//   * a text selection made during the touch (long-press then drag)
//     is a selection, not a swipe.
//
// The listeners are capture-phase and passive: a target that stops
// propagation (Monaco's gesture handling) still lets them see the
// touch, and nothing here delays scrolling.
//
// A rendered .md / .html file is a sandboxed iframe (an opaque origin
// the page cannot script) that fills the file screen, and its touches
// never reach this document.  MobileSwipe.bootstrapHtml() is the same
// detector as a <script> for that document (built from these very
// functions, so the two cannot drift apart); it posts each swipe to
// the parent as {kissMobileSwipe: 'left' | 'right'}, and the parent
// honours only messages from the iframe it is showing.

(function (global) {
  'use strict';

  const SWIPE_MIN_PX = 60;
  const SWIPE_MAX_MS = 1500;
  // Sideways travel must exceed vertical travel by this factor.
  const SWIPE_AXIS_RATIO = 1.5;

  const SCRIPT_NONCE = readOwnNonce();

  function readOwnNonce() {
    if (typeof document === 'undefined') return '';
    const self = document.currentScript;
    if (!self) return '';
    return String(self.nonce || self.getAttribute('nonce') || '');
  }

  function escapeAttr(s) {
    return String(s)
      .replace(/&/g, '&amp;')
      .replace(/"/g, '&quot;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
  }

  /** Whether *el* can scroll sideways (overflow-x auto or scroll with
   *  content wider than its box). */
  function scrollsSideways(el) {
    if (!el || el.nodeType !== 1) return false;
    if (el.scrollWidth <= el.clientWidth + 1) return false;
    const overflowX = global.getComputedStyle(el).overflowX;
    return overflowX === 'auto' || overflowX === 'scroll';
  }

  /** The sideways-scrollable ancestors of *target* with their current
   *  scrollLeft, innermost first. */
  function sidewaysScrollers(target) {
    const out = [];
    for (let el = target; el; el = el.parentNode) {
      if (scrollsSideways(el)) out.push({el, left: el.scrollLeft});
    }
    return out;
  }

  function selectionCollapsed() {
    const sel = global.getSelection ? global.getSelection() : null;
    return !sel || sel.isCollapsed;
  }

  function sameNumbers(a, b) {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
    return true;
  }

  /**
   * Watch *root* for horizontal swipes and report each to *onSwipe*.
   *
   * @param {EventTarget} root Where the touch listeners go (document).
   * @param {function(string, TouchEvent): void} onSwipe Called with
   *   'left' or 'right' and the ending touch event.
   * @param {{ignore?: string, scrollPositions?: function(): number[]}}
   *   [options] `ignore`: a selector; a touch that starts inside a
   *   matching element never swipes.  `scrollPositions`: returns the
   *   sideways scroll offsets of scrollers the DOM cannot report; a
   *   change during the touch means it scrolled.
   * @returns {function(): void} Removes the listeners.
   */
  function install(root, onSwipe, options) {
    const ignore = (options && options.ignore) || '';
    const scrollPositions = (options && options.scrollPositions) || (() => []);
    let touch = null;

    function cancel() {
      touch = null;
    }

    function onStart(e) {
      if (e.touches.length !== 1) return cancel();
      const target = e.target;
      if (
        ignore &&
        target &&
        typeof target.closest === 'function' &&
        target.closest(ignore)
      ) {
        return cancel();
      }
      const t = e.touches[0];
      touch = {
        x: t.clientX,
        y: t.clientY,
        at: Date.now(),
        scrollers: sidewaysScrollers(target),
        positions: scrollPositions(),
        selectionWasCollapsed: selectionCollapsed(),
      };
    }

    function onMove(e) {
      if (e.touches.length !== 1) cancel();
    }

    function onEnd(e) {
      const start = touch;
      touch = null;
      if (!start || e.touches.length || e.changedTouches.length !== 1) return;
      const t = e.changedTouches[0];
      const dx = t.clientX - start.x;
      const dy = t.clientY - start.y;
      if (Math.abs(dx) < SWIPE_MIN_PX) return;
      if (Math.abs(dx) < Math.abs(dy) * SWIPE_AXIS_RATIO) return;
      if (Date.now() - start.at > SWIPE_MAX_MS) return;
      if (start.scrollers.some(s => s.el.scrollLeft !== s.left)) return;
      if (!sameNumbers(start.positions, scrollPositions())) return;
      if (start.selectionWasCollapsed && !selectionCollapsed()) return;
      onSwipe(dx < 0 ? 'left' : 'right', e);
    }

    const opts = {capture: true, passive: true};
    root.addEventListener('touchstart', onStart, opts);
    root.addEventListener('touchmove', onMove, opts);
    root.addEventListener('touchend', onEnd, opts);
    root.addEventListener('touchcancel', cancel, opts);
    return function dispose() {
      root.removeEventListener('touchstart', onStart, opts);
      root.removeEventListener('touchmove', onMove, opts);
      root.removeEventListener('touchend', onEnd, opts);
      root.removeEventListener('touchcancel', cancel, opts);
    };
  }

  /** Report a swipe made inside a preview iframe to its parent. */
  function postSwipeToParent(direction) {
    global.parent.postMessage({kissMobileSwipe: direction}, '*');
  }

  /**
   * The detector as a `<script>` for a sandboxed preview document: it
   * installs itself on that document and posts every swipe to the
   * parent window (see postSwipeToParent).
   *
   * @returns {string} HTML to append to the preview's body.
   */
  function bootstrapHtml() {
    const nonce = SCRIPT_NONCE
      ? ' nonce="' + escapeAttr(SCRIPT_NONCE) + '"'
      : '';
    return (
      '\n<script' +
      nonce +
      ' data-sorcar-swipe>' +
      BOOTSTRAP_SOURCE +
      '<' +
      '/script>\n'
    );
  }

  // The exact source shipped into the iframe: the functions above,
  // serialised, so the iframe's detector is this one.
  const BOOTSTRAP_SOURCE =
    '(function(global){"use strict";' +
    'var SWIPE_MIN_PX=' +
    SWIPE_MIN_PX +
    ';' +
    'var SWIPE_MAX_MS=' +
    SWIPE_MAX_MS +
    ';' +
    'var SWIPE_AXIS_RATIO=' +
    SWIPE_AXIS_RATIO +
    ';' +
    scrollsSideways.toString() +
    sidewaysScrollers.toString() +
    selectionCollapsed.toString() +
    sameNumbers.toString() +
    install.toString() +
    postSwipeToParent.toString() +
    'install(document,postSwipeToParent,{});' +
    '})(window);';

  global.MobileSwipe = {install, bootstrapHtml};
})(typeof window !== 'undefined' ? window : globalThis);
