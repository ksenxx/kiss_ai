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
// is moot).  The preserved policy (a slot the user did NOT remove still
// shows the deadline error chip) is pinned by
// concaudit_f4_attachment_prepare_deadline.test.js.
//
// The production deadline is 60s; waiting a real minute per run is not
// acceptable, so the page records main.js's 60 s timer instead of
// arming it and the test fires it by hand (a changed delay fails
// fireDeadline).  Everything else — DOM, events, the attachment
// pipeline — is the real code.

'use strict';

const assert = require('assert');
const {
  makeHeldDeadlinePage,
  paste,
  settle,
  removeChip,
} = require('./attachment_helpers');

function chipText(win) {
  return win.document.getElementById('file-chips').textContent;
}

function errorChips(win) {
  return Array.from(
    win.document.querySelectorAll('#file-chips .file-chip.error'),
  );
}

async function main() {
  const {win, decodeCount, fireDeadline} = makeHeldDeadlinePage();
  win._testApi.endLaunch();

  // A pending HEIC whose decode is wedged is removed by the user BEFORE
  // the prepare deadline fires. The later deadline failure of the
  // removed slot must be moot: no error chip.
  const heic = new win.File([new Uint8Array([1, 2, 3])], 'IMG_0001.HEIC', {
    type: 'image/heic',
  });
  paste(win, heic);
  await settle(() => decodeCount() === 1);
  assert.strictEqual(decodeCount(), 1, 'the HEIC reached the decoder');
  assert.match(chipText(win), /IMG_0001\.HEIC/, 'the pending chip shows');

  removeChip(win, 'IMG_0001.HEIC');
  assert.strictEqual(
    chipText(win).indexOf('IMG_0001'),
    -1,
    'the chip is gone right after the user removed it',
  );

  // The deadline fires and the removed slot's withDeadline rejection
  // lands.  Post-fix nothing is rendered; pre-fix a phantom error chip
  // "IMG_0001.HEIC: it could not be read within Ns" appears.
  fireDeadline();
  await settle();
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

  console.log('conc2026_attachment_removed_slot: OK');
}

main().then(
  () => process.exit(0),
  err => {
    console.error(err && err.stack ? err.stack : err);
    process.exit(1);
  },
);
