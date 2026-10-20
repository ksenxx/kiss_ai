// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// Shared helpers of the remote-page panel-resize suites
// (remoteSidebarResize, remoteMetaResize, remoteDesktopWidths): a remote
// chat page with pointer capture stubbed, pointer-event dispatch, a drag
// and the rendered width of a docked panel.
'use strict';

const {makePage, fakeMatchMedia} = require('./simplify2_harness');

/**
 * Build a remote chat page for a resize test.
 *
 * `opts.desktopMatches` (default true) is what the desktop media query
 * reports; `opts.stored` maps localStorage keys (e.g. 'kiss-sidebar-w')
 * to the values persisted before main.js runs.  Returns {win, captured,
 * released}: the pointer ids setPointerCapture / releasePointerCapture
 * were called with.
 */
function makeResizePage(opts) {
  const {desktopMatches = true, stored = {}} = opts || {};
  const captured = [];
  const released = [];
  const {win} = makePage({
    bodyClass: 'remote-chat',
    beforeScripts: w => {
      w.Element.prototype.setPointerCapture = function (id) {
        captured.push(id);
      };
      w.Element.prototype.releasePointerCapture = function (id) {
        released.push(id);
      };
      for (const [key, value] of Object.entries(stored)) {
        w.localStorage.setItem(key, value);
      }
      fakeMatchMedia(w, desktopMatches === true);
    },
  });
  return {win, captured, released};
}

/** The inline value of a documentElement CSS variable such as '--sidebar-w'. */
function panelW(win, cssVar) {
  return win.document.documentElement.style.getPropertyValue(cssVar);
}

/** Dispatch a pointer-style MouseEvent of `type` on `el`; returns the event. */
function pointer(win, el, type, props) {
  const ev = new win.MouseEvent(type, {
    bubbles: true,
    cancelable: true,
    button: 0,
    ...props,
  });
  if (props && props.pointerId !== undefined) {
    Object.defineProperty(ev, 'pointerId', {value: props.pointerId});
  }
  el.dispatchEvent(ev);
  return ev;
}

/** Drag `resizer` from clientX x0 to x1 with pointer 1 and release it. */
function drag(win, resizer, x0, x1) {
  pointer(win, resizer, 'pointerdown', {clientX: x0, pointerId: 1});
  pointer(win, resizer, 'pointermove', {clientX: x1, pointerId: 1});
  pointer(win, resizer, 'pointerup', {clientX: x1, pointerId: 1});
}

module.exports = {makeResizePage, panelW, pointer, drag};
