// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Audit 2026-10-12 (mainjs-3): a photo attached while the tab's task is
// running must not be silently lost.  sendMessage()'s append-to-running-
// task branch used to post `appendUserMessage({prompt, tabId})` with no
// attachments field, so the composer refused the send and kept the
// photo.  The daemon command now carries ``attachments`` (the same wire
// shape as ``submit``; server/commands.py _cmd_append_user_message
// decodes them for the running agent), so the photo goes out WITH the
// text and the composer is cleared like after any send -- never a lost
// photo, and no refusal toast either.

'use strict';

const assert = require('assert');
const {makePage, send} = require('./simplify2_harness.js');
const {paste, typeAndEnter, settle} = require('./attachment_helpers');

async function main() {
  const {win, posted} = makePage();
  const tabId = win._testApi.getActiveTabId();
  send(win, {type: 'status', running: true, tabId});
  const png = new win.File([new Uint8Array([137, 80, 78, 71])], 'shot.png', {
    type: 'image/png',
  });
  paste(win, png);
  await settle();
  const chips = win.document.getElementById('file-chips');
  assert.strictEqual(
    chips.querySelectorAll('.file-chip:not(.pending):not(.error)').length,
    1,
    'precondition: the photo is attached and ready',
  );
  typeAndEnter(win, 'and look at this photo');
  await settle();
  assert.ok(
    !posted.some(m => m && m.type === 'submit'),
    'no second task is submitted while one is running',
  );
  const append = posted.find(m => m && m.type === 'appendUserMessage');
  assert.ok(append, 'the message is appended to the running task');
  assert.strictEqual(append.prompt, 'and look at this photo');
  assert.strictEqual(append.tabId, tabId);
  assert.strictEqual(append.attachments.length, 1, 'the photo rides along');
  assert.strictEqual(append.attachments[0].name, 'shot.png');
  assert.strictEqual(append.attachments[0].mimeType, 'image/png');
  assert.strictEqual(
    append.attachments[0].data,
    Buffer.from([137, 80, 78, 71]).toString('base64'),
  );
  // Sent: the composer is cleared, text and chip, and nobody is told
  // that anything was dropped.
  assert.strictEqual(chips.querySelectorAll('.file-chip').length, 0);
  assert.strictEqual(win.document.getElementById('task-input').value, '');
  const toast = Array.from(
    win.document.querySelectorAll('.kiss-notification'),
  ).map(el => el.textContent);
  assert.ok(
    !toast.some(t => /attach/i.test(t)),
    'no toast about a dropped attachment; got ' + JSON.stringify(toast),
  );
  win.close();
  console.log(
    'audit1012_mainjs3_append_drops_attachments.test.js: all passed.',
  );
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
