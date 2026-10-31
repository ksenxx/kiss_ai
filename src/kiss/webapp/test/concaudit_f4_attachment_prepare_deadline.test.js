// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) regression test: an attachment whose browser-side
// preparation never settles must not latch the tab's send for ever.
//
// The bug: fillAttachmentSlot() awaited prepareAttachment() with no
// deadline.  A decoder (createImageBitmap) that never calls back left
// the slot pending; Enter then set `tab.awaitingAttachments = true` and
// parked in attachmentsReady(), and every later Enter in that tab was
// discarded by the latch.  The only way out was to find and remove the
// stuck chip by hand.
//
// The fix races preparation against ATTACH_PREPARE_TIMEOUT_MS and routes
// a timeout through the existing failed-conversion path: the slot is
// dropped, an error chip explains why, and the waiting send is released
// (it does not ship, because a failed attachment blocks the submit --
// that policy is unchanged).  Removing the chip stays an immediate
// cancellation.
//
// The production deadline is 60s.  Waiting a real minute per run is not
// acceptable in the suite, and racing a shortened real timer against the
// host's scheduling is not deterministic, so makeHeldDeadlinePage
// records the 60 s timer instead of arming it (harness `recordTimers`)
// and the test fires it by hand with `fireDeadline()`, which fails when
// main.js armed no 60 s deadline.  Everything else -- DOM, events,
// FileReader, the attachment fixpoint, the latch -- is the real code.

'use strict';

const assert = require('assert');
const {
  makeHeldDeadlinePage,
  paste,
  typeAndEnter,
  submits,
  settle,
} = require('./attachment_helpers');

function chipText(win) {
  return win.document.getElementById('file-chips').textContent;
}

async function main() {
  const {win, posted, decodeCount, fireDeadline} = makeHeldDeadlinePage();
  win._testApi.endLaunch();

  // 1. A camera HEIC whose decode never settles; Enter parks the send.
  const heic = new win.File([new Uint8Array([1, 2, 3])], 'IMG_0001.HEIC', {
    type: 'image/heic',
  });
  paste(win, heic);
  await settle(() => decodeCount() === 1);
  assert.strictEqual(decodeCount(), 1, 'the HEIC reached the decoder');
  typeAndEnter(win, 'prompt with a photo that never decodes');
  assert.strictEqual(submits(posted).length, 0, 'the send is waiting');

  // 2. Before the deadline nothing has changed: the chip is still
  //    pending and a second Enter is swallowed by the tab's latch.
  await settle();
  typeAndEnter(win, 'prompt with a photo that never decodes');
  assert.strictEqual(submits(posted).length, 0, 'latched while pending');
  assert.match(chipText(win), /IMG_0001\.HEIC/, 'the pending chip shows');
  assert.doesNotMatch(chipText(win), /could not be read/, 'no error yet');

  // 3. Past the deadline the stuck preparation is reported like a failed
  //    conversion.  The parked send is released but does NOT ship: a
  //    failed attachment still blocks the submit (policy unchanged).
  fireDeadline();
  await settle(() => /could not be read/.test(chipText(win)));
  assert.match(
    chipText(win),
    /IMG_0001\.HEIC: it could not be read within \d+s/,
    'the timeout is explained through the visible error path',
  );
  assert.strictEqual(
    submits(posted).length,
    0,
    'a failed attachment still blocks the submit (policy unchanged)',
  );

  // 4. The tab is usable again without touching the chip: a plain Enter
  //    goes straight out.  Before the fix the slot stayed pending and
  //    the latch swallowed this Enter too (recoverable only by removing
  //    the chip by hand).
  typeAndEnter(win, 'plain prompt after the timeout');
  await settle(() => submits(posted).length === 1);
  const sent = submits(posted);
  assert.strictEqual(sent.length, 1, 'the tab sends again after the timeout');
  assert.strictEqual(sent[0].prompt, 'plain prompt after the timeout');
  assert.strictEqual(
    JSON.stringify(sent[0].attachments || []),
    '[]',
    'the timed-out attachment is not shipped',
  );
  assert.strictEqual(
    chipText(win).indexOf('IMG_0001'),
    -1,
    'the composer (including the error chip) was reset after the send',
  );

  console.log('concaudit_f4_attachment_prepare_deadline: OK');
}

main().then(
  () => process.exit(0),
  err => {
    console.error(err && err.stack ? err.stack : err);
    process.exit(1);
  },
);
