// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) regression test for review2-vscode.md #1: once the
// attachmentsReady() fixpoint began awaiting a late pending attachment,
// REMOVING that attachment's chip did not release the wait.  Interleaving:
//
//   1. attachment A (a PNG) is still reading when Enter is pressed: the
//      send parks on a pass that awaits A;
//   2. the user adds attachment B (a camera HEIC) whose decode is parked
//      in the test's decoder — a stand-in for a slow or hung codec;
//   3. A finishes -> the next fixpoint pass awaits B's promise;
//   4. the user clicks the visible x on B's pending chip: the slot leaves
//      the owning array, but its promise stays inside the already-created
//      Promise.all — the submit stayed blocked until the removed, hidden
//      conversion settled, potentially forever.
//
// The fix wakes the pass when an awaited slot is removed (each slot races
// its conversion promise against a removal signal) and re-reads the tab's
// live attachment list, so the send proceeds without the removed
// attachment even though its conversion promise NEVER settles here.  A
// failed conversion of a still-present attachment must keep blocking the
// submit.

'use strict';

const assert = require('assert');
const {
  makePage,
  paste,
  typeAndEnter,
  submits,
  settle,
  failDecode,
} = require('./attachment_helpers');

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
  const {win, posted, decodes} = makePage();
  win._testApi.endLaunch();
  const tabA = win._testApi.getActiveTabId();
  const chips = win.document.getElementById('file-chips');

  // 1. Attachment A: a PNG whose FileReader is still running when Enter
  //    arrives, so the send parks while A is pending.
  const png = new win.File([new Uint8Array([137, 80, 78, 71])], 'shot.png', {
    type: 'image/png',
  });
  paste(win, png);
  typeAndEnter(win, 'prompt minus the removed photo');
  assert.strictEqual(submits(posted).length, 0, 'the send is waiting on A');

  // 2. While the send waits, attachment B (a camera HEIC) is added; its
  //    decode parks in the test's decoder and will never settle.
  const heic = new win.File([new Uint8Array([1, 2, 3])], 'IMG_0002.HEIC', {
    type: 'image/heic',
  });
  paste(win, heic);

  // 3. A finishes during settle; the next fixpoint pass now awaits B.
  await settle(() => decodes.length >= 1);
  assert.strictEqual(decodes.length, 1, 'B reached the decoder');
  assert.strictEqual(
    submits(posted).length,
    0,
    'the send must keep waiting while B is present and converting',
  );
  assert.match(chips.textContent, /IMG_0002\.HEIC/, "B's chip is visible");

  // 4. The user removes B's pending chip.  The send must proceed with A
  //    only, even though B's conversion promise never settles: before the
  //    fix the wait stayed parked inside Promise.all on the removed slot.
  removeChip(win, 'IMG_0002.HEIC');
  await settle(() => submits(posted).some(m => m.tabId === tabA));
  const sent = submits(posted).filter(m => m.tabId === tabA);
  assert.strictEqual(
    sent.length,
    1,
    'removing the awaited pending attachment must release the send',
  );
  assert.strictEqual(sent[0].prompt, 'prompt minus the removed photo');
  // (JSON compare: the arrays come from the JSDOM realm, whose Array
  // prototype fails deepStrictEqual's cross-realm identity check.)
  assert.strictEqual(
    JSON.stringify(sent[0].attachments.map(a => a.name)),
    JSON.stringify(['shot.png']),
    'the submit ships without the removed attachment',
  );
  assert.strictEqual(
    chips.textContent.indexOf('shot.png'),
    -1,
    'the composer was reset after the released send',
  );

  // 5. Guard the preserved policy: a FAILED conversion of a still-present
  //    attachment must still block the submit.  A new send waits on a
  //    fresh HEIC whose decode is rejected while its chip stays in place.
  const heic2 = new win.File([new Uint8Array([4, 5, 6])], 'IMG_0003.HEIC', {
    type: 'image/heic',
  });
  paste(win, heic2);
  typeAndEnter(win, 'prompt with a broken photo');
  await settle(() => decodes.length >= 2);
  assert.strictEqual(decodes.length, 2, 'the new HEIC reached the decoder');
  assert.strictEqual(
    submits(posted).length,
    1,
    'the new send waits on the converting attachment',
  );
  await failDecode(decodes, 1, 'undecodable HEIC');
  assert.strictEqual(
    submits(posted).length,
    1,
    'a still-present attachment that failed must keep blocking the submit',
  );
  assert.match(
    chips.textContent,
    /undecodable HEIC/,
    'the failure is explained to the user',
  );

  console.log('audit0911_remove_pending_attachment_unblocks_send: OK');
}

main().then(
  () => process.exit(0),
  err => {
    console.error(err && err.stack ? err.stack : err);
    process.exit(1);
  },
);
