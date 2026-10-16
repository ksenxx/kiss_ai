// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Audit 2026-10-12 (mainjs-3): a share the PAGE refuses (empty chat,
// oversized page, content tab without an owner chat) only writes a
// transcript banner and flashes the share button red, while a share the
// DAEMON refuses goes through notifyShareFailed() -- banner plus a sticky
// error toast, because (its own words) "the button sits in the closed
// '...' menu".  The content-tab refusal is worse: #output is
// display:none under a content tab (setChatSurfaceVisible), so its banner
// is invisible and the user sees nothing at all.
//
// Expected: every share refusal raises the same error toast.

'use strict';

const assert = require('assert');
const {makePage, send} = require('./simplify2_harness');

function toasts(win) {
  return Array.from(win.document.querySelectorAll('.kiss-notification')).map(
    el => el.textContent,
  );
}

function main() {
  const {win, posted} = makePage();
  const tabId = win._testApi.getActiveTabId();
  win.document
    .getElementById('share-btn')
    .dispatchEvent(
      new win.MouseEvent('click', {bubbles: true, cancelable: true}),
    );
  const req = posted.find(m => m && m.type === 'shareChatTasks');
  assert.ok(req, 'the share button asks the daemon for the chat tasks');
  // The daemon knows no task of this chat and the screen is empty.
  send(win, {type: 'share_tasks', tabId, chatId: req.chatId, tasks: []});
  assert.ok(
    !posted.some(m => m && m.type === 'shareChat'),
    'an empty chat is not exported',
  );
  const banner = Array.from(win.document.querySelectorAll('#output .ev.err'))
    .map(el => el.textContent)
    .find(t => t.indexOf('Share failed') >= 0);
  assert.ok(banner, 'the refusal writes its transcript banner');
  const toast = toasts(win).find(t => t.indexOf('Share failed') >= 0);
  assert.ok(
    toast,
    'the page-side refusal raises the same error toast the daemon-side ' +
      'one does (notifyShareFailed); got toasts: ' +
      JSON.stringify(toasts(win)),
  );
  win.close();
  console.log('audit1012_mainjs3_share_refusal_toast.test.js: all passed.');
}

try {
  main();
} catch (err) {
  console.error(err);
  process.exit(1);
}
