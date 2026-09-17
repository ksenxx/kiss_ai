// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) regression test for the removed-slot phantom error
// (audit candidate E / CAND-1, tmp/audit-media-cand-E.md).
//
// The bug: fillAttachmentSlot()'s catch block pushes to the owning tab's
// attachErrors UNCONDITIONALLY, even when ownerFiles.indexOf(slot) is -1,
// i.e. the user already removed the pending chip.  Interleaving:
//
//   1. the user pastes a camera HEIC whose decode hangs (the test's
//      createImageBitmap parks forever);
//   2. the user clicks the x on the pending chip — the chip disappears
//      and the slot leaves the tab's attachment list;
//   3. up to ATTACH_PREPARE_TIMEOUT_MS later withDeadline() rejects, the
//      catch runs with idx === -1, yet "IMG_0001.HEIC: it could not be
//      read within Ns" is pushed into the live attachErrors and rendered
//      by the finally — a phantom error chip for an attachment the user
//      deleted long ago.
//
// The fix confines the error push to idx >= 0 (a removed slot's failure
// is moot).  The preserved policy is asserted too: when the slot is NOT
// removed, the deadline failure must still produce the visible error chip
// (as concaudit_f4_attachment_prepare_deadline.test.js also pins).
//
// The production deadline is 60s; waiting a real minute per run is not
// acceptable, so main.js is loaded with that ONE constant lowered (and
// the substitution is asserted, so a rename fails loudly).  Everything
// else — DOM, events, the attachment pipeline, real timers — is the real
// code.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');
const SHORT_DEADLINE_MS = 300;

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
  win.HTMLElement.prototype.scrollTo = function () {};

  // A wedged decoder: every call parks for ever, so only the deadline
  // (or nothing) can settle a HEIC slot.
  let decodes = 0;
  win.createImageBitmap = function () {
    decodes += 1;
    return new Promise(() => {});
  };

  const posted = [];
  let state;
  win.acquireVsCodeApi = function () {
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
  const mainSrc = fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8');
  const needle = 'const ATTACH_PREPARE_TIMEOUT_MS = 60 * 1000;';
  assert.ok(
    mainSrc.includes(needle),
    'main.js no longer defines ATTACH_PREPARE_TIMEOUT_MS as expected',
  );
  win.eval(
    mainSrc.replace(
      needle,
      `const ATTACH_PREPARE_TIMEOUT_MS = ${SHORT_DEADLINE_MS};`,
    ) + '\n//# sourceURL=conc2026-attachment-removed-slot-main.js',
  );
  return {win, decodeCount: () => decodes};
}

function paste(win, file) {
  const inp = win.document.getElementById('task-input');
  const ev = new win.Event('paste', {bubbles: true, cancelable: true});
  Object.defineProperty(ev, 'clipboardData', {
    value: {items: [{kind: 'file', getAsFile: () => file}]},
  });
  inp.dispatchEvent(ev);
}

function delay(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function chipText(win) {
  return win.document.getElementById('file-chips').textContent;
}

function errorChips(win) {
  return Array.from(
    win.document.querySelectorAll('#file-chips .file-chip.error'),
  );
}

// Click the x on the chip whose label contains `name`.
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

async function main() {
  const {win, decodeCount} = makeWebview();
  win._testApi.endLaunch();

  // ---- Part 1: the bug. A pending HEIC whose decode is wedged is
  // removed by the user BEFORE the prepare deadline fires. The later
  // deadline failure of the removed slot must be moot: no error chip.
  const heic = new win.File([new Uint8Array([1, 2, 3])], 'IMG_0001.HEIC', {
    type: 'image/heic',
  });
  paste(win, heic);
  await delay(30);
  assert.strictEqual(decodeCount(), 1, 'the HEIC reached the decoder');
  assert.match(chipText(win), /IMG_0001\.HEIC/, 'the pending chip shows');

  removeChip(win, 'IMG_0001.HEIC');
  assert.strictEqual(
    chipText(win).indexOf('IMG_0001'),
    -1,
    'the chip is gone right after the user removed it',
  );

  // Well past the deadline: the removed slot's withDeadline rejection
  // lands. Post-fix nothing is rendered; pre-fix a phantom error chip
  // "IMG_0001.HEIC: it could not be read within Ns" appears.
  await delay(SHORT_DEADLINE_MS + 200);
  assert.strictEqual(
    errorChips(win).length,
    0,
    'no phantom error chip for an attachment the user already removed',
  );
  assert.doesNotMatch(
    chipText(win),
    /IMG_0001\.HEIC/,
    'the removed attachment must not reappear in the composer',
  );

  // ---- Part 2: the preserved behavior. The same deadline failure of a
  // slot the user did NOT remove must still surface the error chip.
  const heic2 = new win.File([new Uint8Array([4, 5, 6])], 'IMG_0002.HEIC', {
    type: 'image/heic',
  });
  paste(win, heic2);
  await delay(30);
  assert.strictEqual(decodeCount(), 2, 'the second HEIC reached the decoder');
  assert.match(chipText(win), /IMG_0002\.HEIC/, 'its pending chip shows');

  await delay(SHORT_DEADLINE_MS + 200);
  assert.strictEqual(
    errorChips(win).length,
    1,
    'a still-present attachment that hit the deadline shows one error chip',
  );
  assert.match(
    chipText(win),
    /IMG_0002\.HEIC: it could not be read within \d+s/,
    'the deadline failure is explained for the still-present slot',
  );

  console.log('conc2026_attachment_removed_slot: OK');
}

main().then(
  () => process.exit(0),
  err => {
    console.error(err && err.stack ? err.stack : err);
    process.exit(1);
  },
);
