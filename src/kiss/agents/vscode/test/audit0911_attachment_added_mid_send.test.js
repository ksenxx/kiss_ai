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
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

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

  // The HEIC decoder the page asks the browser for: every call is parked
  // until the test settles it (JSDOM has no image decoder).
  const decodes = [];
  win.createImageBitmap = function () {
    return new Promise((resolve, reject) => {
      decodes.push({resolve, reject});
    });
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
  win.eval(
    fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8') +
      '\n//# sourceURL=audit0911-attach-main.js',
  );
  return {win, posted, decodes};
}

function paste(win, file) {
  const inp = win.document.getElementById('task-input');
  const ev = new win.Event('paste', {bubbles: true, cancelable: true});
  Object.defineProperty(ev, 'clipboardData', {
    value: {items: [{kind: 'file', getAsFile: () => file}]},
  });
  inp.dispatchEvent(ev);
}

function typeAndEnter(win, text) {
  const inp = win.document.getElementById('task-input');
  inp.value = text;
  inp.dispatchEvent(
    new win.KeyboardEvent('keydown', {key: 'Enter', bubbles: true}),
  );
}

function submits(posted) {
  return posted.filter(m => m.type === 'submit');
}

function settle() {
  // Several macrotask turns: FileReader completion + the awaited chain.
  return new Promise(resolve => setTimeout(resolve, 50));
}

// encodeAsJpeg retries createImageBitmap once without the orientation
// option, so a failed decode is two parked calls.
async function failDecode(decodes, first, reason) {
  decodes[first].reject(new Error(reason));
  await settle();
  assert.strictEqual(decodes.length, first + 2, 'the decoder was retried');
  decodes[first + 1].reject(new Error(reason));
  await settle();
}

async function main() {
  const {win, posted, decodes} = makeWebview();
  win._testApi.endLaunch();
  const tabA = win._testApi.getActiveTabId();

  // 1. Attachment A: a PNG whose FileReader is still running when Enter
  //    arrives (paste and Enter happen in the same tick, so the send parks
  //    while A is pending).
  const png = new win.File([new Uint8Array([137, 80, 78, 71])], 'shot.png', {
    type: 'image/png',
  });
  paste(win, png);
  typeAndEnter(win, 'prompt with two photos');
  assert.strictEqual(submits(posted).length, 0, 'the send is waiting on A');

  // 2. While the send waits, attachment B (a camera HEIC) is added; its
  //    decode is parked in the test's decoder.
  const heic = new win.File([new Uint8Array([1, 2, 3])], 'IMG_0002.HEIC', {
    type: 'image/heic',
  });
  paste(win, heic);

  // 3. A finishes (FileReader completes during settle) while B is still
  //    decoding.  The one-time snapshot resolved here and posted the
  //    submit without B, then reset the composer and discarded B.
  await settle();
  assert.strictEqual(decodes.length, 1, 'B reached the decoder');
  assert.strictEqual(
    submits(posted).length,
    0,
    'the submit was posted while attachment B was still converting: B ' +
      'was silently omitted from the message',
  );
  const chips = win.document.getElementById('file-chips');
  assert.match(
    chips.textContent,
    /IMG_0002\.HEIC/,
    "B's chip must still be visible while the send waits for it",
  );

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
  await settle();
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
  await settle();
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
