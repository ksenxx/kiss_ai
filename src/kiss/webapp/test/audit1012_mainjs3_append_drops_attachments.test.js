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
const {makePage, send, waitFor} = require('./simplify2_harness.js');
const {paste, typeAndEnter, settle, removeChip} = require('./attachment_helpers');

async function main() {
  const {win, posted} = makePage();
  const tabId = win._testApi.getActiveTabId();
  send(win, {type: 'status', running: true, tabId});
  const png = new win.File([new Uint8Array([137, 80, 78, 71])], 'shot.png', {
    type: 'image/png',
  });
  const chips = win.document.getElementById('file-chips');
  const readyChips = () =>
    chips.querySelectorAll('.file-chip:not(.pending):not(.error)').length;
  paste(win, png);
  await settle(() => readyChips() === 1);
  assert.strictEqual(readyChips(), 1, 'precondition: the photo is attached and ready');
  typeAndEnter(win, 'and look at this photo');
  await settle(() => posted.some(m => m && m.type === 'appendUserMessage'));
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
}

// A prompt and its files travel in one frame, of which the daemon reads
// at most 64 MiB (it drops the connection beyond).  A message that
// cannot fit is refused in the composer: nothing is sent, the reason
// shows on a chip, and the text and the files stay to be edited.
async function oversized() {
  const {win, posted} = makePage();
  const tabId = win._testApi.getActiveTabId();
  const chips = win.document.getElementById('file-chips');
  const count = sel => chips.querySelectorAll(sel).length;
  const sent = () =>
    posted.filter(m => m && /^(submit|appendUserMessage)$/.test(m.type));
  const attach = async (...mib) => {
    mib.forEach((size, i) => {
      const bytes = new Uint8Array(size * 1024 * 1024);
      paste(win, new win.File([bytes], i + '.pdf', {type: 'application/pdf'}));
    });
    await waitFor(() => !count('.pending'), 'the files were not read', 60000);
    assert.strictEqual(count('.file-chip:not(.error)'), mib.length);
  };
  const refused = what => {
    typeAndEnter(win, 'read this');
    assert.strictEqual(sent().length, 0, 'BUG: ' + what + ' was sent');
    assert.strictEqual(count('.file-chip.error'), 1, what + ': the reason shows, once');
    assert.strictEqual(win.document.getElementById('task-input').value, 'read this');
  };
  // One 49 MiB file is 65 MiB of base64: over the frame as a new task...
  await attach(49);
  refused('an oversized new task');
  // ...and as a message for the running one.
  send(win, {type: 'status', running: true, tabId});
  refused('an oversized message for the running task');
  assert.strictEqual(count('.file-chip:not(.error)'), 1, 'the file stays attached');
  // Two files that each fit are refused together...
  removeChip(win, '0.pdf');
  await attach(25, 25);
  refused('two files that fit only apart');
  // ...and one of them alone goes out, taking the reason with it.
  removeChip(win, '1.pdf');
  typeAndEnter(win, 'read this');
  assert.strictEqual(sent().length, 1, 'a message that fits is sent');
  assert.strictEqual(sent()[0].attachments.length, 1);
  assert.strictEqual(count('.file-chip'), 0);
  win.close();
}

main()
  .then(oversized)
  .then(() => {
    console.log(
      'audit1012_mainjs3_append_drops_attachments.test.js: all passed.',
    );
  })
  .catch(err => {
    console.error(err);
    process.exit(1);
  });
