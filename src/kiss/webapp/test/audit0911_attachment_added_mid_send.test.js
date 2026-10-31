// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) regression test for review-vscode.md #2: an attachment
// added AFTER a send began waiting fell outside attachmentsReady()'s
// one-time promise snapshot.  Interleaving:
//
//   1. attachment A (a PNG) is still reading when Enter is pressed: the
//      send parks on a snapshot that contains only A's promise;
//   2. the user adds attachment B (a camera HEIC, parked in the decoder);
//   3. A finishes -> the snapshot resolves all-ok -> the submit is posted
//      WITHOUT B (it has no data yet) and resetComposerAfterSend() replaces
//      the attachment array: B later finishes into the abandoned array,
//      invisible and unsendable — silent data loss.
//
// The fix makes the wait a fixpoint: the send only proceeds once a pass
// over the tab's attachments finds no pending slot, so B either ships with
// the prompt or (on a failed conversion) blocks the submit exactly like a
// failed first attachment always has.

'use strict';

const assert = require('assert');
const {
  makePage,
  parkSendOnLateHeic,
  paste,
  typeAndEnter,
  submits,
  settle,
  failDecode,
} = require('./attachment_helpers');

async function main() {
  const page = makePage();
  const {win, posted, decodes} = page;

  // 1-3. A (shot.png) is still reading at Enter (paste and Enter happen
  //      in the same tick), B (IMG_0002.HEIC) is added while the send
  //      waits and its decode is parked; A finishes while B is still
  //      decoding.  The one-time snapshot resolved here and posted the
  //      submit without B, then reset the composer and discarded B.
  const tabA = await parkSendOnLateHeic(page, 'prompt with two photos');
  const chips = win.document.getElementById('file-chips');

  // 4. B's conversion fails: the submit must be dropped (sending without
  //    the photo is the silent loss to avoid) and A must be RETAINED in
  //    the composer, exactly like a failed first attachment.
  await failDecode(decodes, 0, 'undecodable HEIC');
  assert.strictEqual(
    submits(posted).length,
    0,
    'a submit whose late attachment failed must not be sent',
  );
  assert.match(
    chips.textContent,
    /undecodable HEIC/,
    'B shows why it was dropped',
  );
  assert.match(chips.textContent, /shot\.png/, 'A survived the failed wait');

  // 5. The user presses Enter again: the prompt ships with A attached.
  typeAndEnter(win, 'prompt with two photos');
  await settle(() => submits(posted).some(m => m.tabId === tabA));
  const sent = submits(posted).filter(m => m.tabId === tabA);
  assert.strictEqual(sent.length, 1, 'the retry submitted once');
  assert.strictEqual(sent[0].prompt, 'prompt with two photos');
  // (JSON compare: the arrays come from the JSDOM realm, whose Array
  // prototype fails deepStrictEqual's cross-realm identity check.)
  assert.strictEqual(
    JSON.stringify(sent[0].attachments.map(a => a.name)),
    JSON.stringify(['shot.png']),
    'the retry carried the surviving attachment',
  );

  // 6. Both-succeed path: a new prompt with a pending PNG at Enter and a
  //    second PNG added mid-wait must ship BOTH files in one submit.
  const png2 = new win.File([new Uint8Array([137, 80, 78, 71, 2])], 'a.png', {
    type: 'image/png',
  });
  paste(win, png2);
  typeAndEnter(win, 'both photos please');
  assert.strictEqual(
    submits(posted).length,
    1,
    'the new send waits on a.png',
  );
  const png3 = new win.File([new Uint8Array([137, 80, 78, 71, 3])], 'b.png', {
    type: 'image/png',
  });
  paste(win, png3);
  await settle(() => submits(posted).some(m => m.prompt === 'both photos please'));
  const both = submits(posted).filter(m => m.prompt === 'both photos please');
  assert.strictEqual(both.length, 1, 'the fixpoint send went out once');
  assert.strictEqual(
    JSON.stringify(both[0].attachments.map(a => a.name).sort()),
    JSON.stringify(['a.png', 'b.png']),
    'an attachment added while the send waited must ship with the prompt',
  );
  assert.strictEqual(
    chips.textContent.indexOf('a.png'),
    -1,
    'the composer was reset after the successful send',
  );

  console.log('audit0911_attachment_added_mid_send: OK');
}

main().then(
  () => process.exit(0),
  err => {
    console.error(err && err.stack ? err.stack : err);
    process.exit(1);
  },
);
