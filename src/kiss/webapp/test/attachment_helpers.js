// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// Shared helpers for the composer-attachment suites (audit0902_fix_
// attachment_latch_per_tab, audit0911_attachment_added_mid_send,
// audit0911_remove_pending_attachment_unblocks_send): a chat page whose
// image decoder parks every call until the test settles it, plus the
// paste / Enter / settle primitives those suites drive it with.
'use strict';

const assert = require('assert');
const harness = require('./simplify2_harness');

/**
 * Build a chat page whose `createImageBitmap` parks every call until
 * the test settles it (JSDOM has no image decoder).
 *
 * @param {string} [sourceUrl] Name for the eval'd main.js, for a
 *   `*.coverage.js` gate that keys on it.
 * @returns {{win: object, posted: Array<object>, decodes: Array<{resolve: Function, reject: Function}>}}
 *   The page, the messages it posted and one {resolve, reject} pair per
 *   decode the page asked for, in call order.
 */
function makePage(sourceUrl) {
  const decodes = [];
  function parkDecoder(win) {
    win.createImageBitmap = function () {
      return new Promise((resolve, reject) => {
        decodes.push({resolve, reject});
      });
    };
  }
  const {win, posted} = harness.makePage({
    beforeScripts: parkDecoder,
    sourceUrl,
  });
  return {win, posted, decodes};
}

// main.js's ATTACH_PREPARE_TIMEOUT_MS: the attachment prepare deadline.
const ATTACH_DEADLINE_MS = 60 * 1000;

/**
 * Build a chat page whose decoder is wedged (every `createImageBitmap`
 * call parks for ever), so only the attachment prepare deadline can
 * settle an image slot.  That 60 s deadline is recorded instead of armed
 * (harness `recordTimers`), so a suite asserts "before the deadline" and
 * "after the deadline" without racing a real timer: `fireDeadline()` runs
 * the deadline main.js recorded for the oldest pending slot (asserting
 * one exists).  Await `settle()` afterwards for the rejection to land.
 *
 * @returns {{win: object, posted: Array<object>, decodeCount: Function,
 *   fireDeadline: Function}} The page, the messages it posted and a
 *   counter of decodes asked for.
 */
function makeHeldDeadlinePage() {
  let decodes = 0;
  let deadlines;
  const {win, posted} = harness.makePage({
    beforeScripts(w) {
      w.createImageBitmap = function () {
        decodes += 1;
        return new Promise(() => {});
      };
      deadlines = harness.recordTimers(w, ATTACH_DEADLINE_MS);
    },
  });
  function fireDeadline() {
    assert.ok(deadlines.length, 'main.js armed the attachment deadline');
    harness.fireTimers(deadlines.splice(0, 1));
  }
  return {win, posted, decodeCount: () => decodes, fireDeadline};
}

/**
 * Paste one file into the composer.
 *
 * @param {object} win The page window.
 * @param {File} file The file the clipboard carries.
 */
function paste(win, file) {
  const inp = win.document.getElementById('task-input');
  const ev = new win.Event('paste', {bubbles: true, cancelable: true});
  Object.defineProperty(ev, 'clipboardData', {
    value: {items: [{kind: 'file', getAsFile: () => file}]},
  });
  inp.dispatchEvent(ev);
}

/**
 * Type a prompt into the composer and press Enter.
 *
 * @param {object} win The page window.
 * @param {string} text The prompt.
 */
function typeAndEnter(win, text) {
  const inp = win.document.getElementById('task-input');
  inp.value = text;
  inp.dispatchEvent(
    new win.KeyboardEvent('keydown', {key: 'Enter', bubbles: true}),
  );
}

/**
 * The submit messages the page posted so far.
 *
 * @param {Array<object>} posted The page's posted messages.
 * @returns {Array<object>} The `submit` messages, in order.
 */
function submits(posted) {
  return posted.filter(m => m.type === 'submit');
}

/**
 * Let the page's FileReader + awaited chain run.  With `until`, first
 * wait (up to 3 s) for that outcome -- on a busy host the FileReader
 * alone can take longer than one macrotask turn -- then a short grace
 * period, so an "exactly once" check still sees a double.
 *
 * @param {Function} [until] Predicate that becomes true once the awaited
 *   side effect has happened.
 * @throws {Error} When `until` is still false at the deadline, so the
 *   failure names the side effect that did not happen instead of
 *   surfacing through whatever assertion the caller makes next.
 */
async function settle(until) {
  if (until) await harness.waitFor(until, 'settle: condition not met within 3 s');
  await harness.sleep(50);
}

/**
 * Reject the parked decode at index `first` and the retry the page
 * makes for it (encodeAsJpeg retries createImageBitmap once without
 * the orientation option, so a failed decode is two parked calls).
 *
 * @param {Array<{reject: Function}>} decodes The page's parked decodes.
 * @param {number} first Index of the first parked call of the pair.
 * @param {string} reason The failure message the chip should show.
 */
async function failDecode(decodes, first, reason) {
  decodes[first].reject(new Error(reason));
  await settle(() => decodes.length >= first + 2);
  assert.strictEqual(decodes.length, first + 2, 'the decoder was retried');
  decodes[first + 1].reject(new Error(reason));
  await settle();
}

/**
 * Click the x on the composer chip whose label contains `name`.
 *
 * @param {object} win The page window.
 * @param {string} name Text of the chip's label.
 * @throws {Error} When no chip carries `name`.
 */
function removeChip(win, name) {
  const chips = win.document.querySelectorAll('#file-chips .file-chip');
  for (const chip of chips) {
    if (chip.textContent.indexOf(name) >= 0) {
      chip.querySelector('.fc-rm').click();
      return;
    }
  }
  throw new Error('no chip found for ' + name);
}

/**
 * Park a send on a late attachment: PNG `shot.png` is still reading when
 * `prompt` is entered, so the send waits; a camera HEIC
 * `IMG_0002.HEIC` is pasted meanwhile and its decode parks in the page's
 * decoder (`decodes[0]`).  Asserts the send is still waiting, with the
 * HEIC's chip on screen, once the PNG has finished.
 *
 * @param {{win: object, posted: Array<object>, decodes: Array}} page A
 *   makePage page.
 * @param {string} prompt The prompt sent.
 * @returns {Promise<string>} The id of the tab the send belongs to.
 */
async function parkSendOnLateHeic(page, prompt) {
  const {win, posted, decodes} = page;
  win._testApi.endLaunch();
  const tabId = win._testApi.getActiveTabId();
  const png = new win.File([new Uint8Array([137, 80, 78, 71])], 'shot.png', {
    type: 'image/png',
  });
  paste(win, png);
  typeAndEnter(win, prompt);
  assert.strictEqual(submits(posted).length, 0, 'the send is waiting on the PNG');
  const heic = new win.File([new Uint8Array([1, 2, 3])], 'IMG_0002.HEIC', {
    type: 'image/heic',
  });
  paste(win, heic);
  await settle(() => decodes.length >= 1);
  assert.strictEqual(decodes.length, 1, 'the HEIC reached the decoder');
  assert.strictEqual(
    submits(posted).length,
    0,
    'the send must keep waiting while the HEIC converts (posting now ' +
      'silently omits it from the message)',
  );
  assert.match(
    win.document.getElementById('file-chips').textContent,
    /IMG_0002\.HEIC/,
    "the HEIC's chip stays visible while the send waits for it",
  );
  return tabId;
}

module.exports = {
  makePage,
  makeHeldDeadlinePage,
  paste,
  typeAndEnter,
  submits,
  settle,
  failDecode,
  removeChip,
  parkSendOnLateHeic,
};
