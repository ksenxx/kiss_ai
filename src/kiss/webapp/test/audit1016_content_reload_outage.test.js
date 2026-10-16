// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// End-to-end (jsdom) regression test for media/main.js: a "Reload from
// disk" whose reply is swallowed by an outage must not stay armed.
//
// reloadContentTab sets `tab.contentReloadRequested` to the requesting
// chat's tab id and sends openFile; handleFileContent treats the next
// fileContent reply for that path from that chat as the reload and
// replaces the editor even when the tab is dirty.  When the connection
// drops before the reply lands, the request is dead (the daemonStatus
// handler already abandons in-flight saves for that reason), but the
// flag survived: the next plain open of the same file by the same chat
// (a file link the agent printed, the Explorer) was taken as the
// reload and the edits typed after the reconnect were lost without a
// question.
//
// Expected: after the outage the tab's edits win again, like on any
// other open of a dirty file (the editor is kept, only the line is
// revealed), and the Explorer/link open creates no second editor.

const assert = require('assert');
const {
  makePage,
  send,
  sleep,
  click,
  all,
  openDirtyContentTab,
} = require('./ui_antipattern_harness');

async function testReloadSwallowedByOutageKeepsEdits() {
  const ctx = makePage();
  const {win, posted, created} = ctx;
  send(win, {type: 'daemonStatus', connected: true});
  await openDirtyContentTab(ctx);
  const editor = created[0].editor;

  // Save -> the daemon refuses (changed on disk) -> the conflict toast.
  click(win, win.document.querySelector('.content-save-btn'));
  const saves = posted.filter(m => m.type === 'saveFile');
  assert.strictEqual(saves.length, 1, 'Save sends one saveFile');
  send(win, {
    type: 'fileSaved',
    token: saves[0].token,
    ok: false,
    conflict: true,
    error: 'notes.txt changed on disk',
  });
  const reload = all(win, '.kiss-notification-action').find(
    b => b.textContent.trim() === 'Reload from disk',
  );
  assert.ok(reload, "the conflict toast offers 'Reload from disk'");
  click(win, reload);
  assert.strictEqual(
    posted.filter(m => m.type === 'openFile').length,
    1,
    'the reload asks the daemon for the file',
  );

  // The reply never comes: the connection drops and comes back.
  send(win, {type: 'daemonStatus', connected: false, reconnecting: true});
  send(win, {type: 'daemonStatus', connected: true});
  // The user keeps editing the still-dirty tab after the reconnect.
  editor._type('v1 edited after the outage');

  // Later the same chat opens the same file (a printed file link): the
  // reply echoes the chat's tabId, exactly as the lost reload reply would.
  send(win, {
    type: 'fileContent',
    tabId: 'a1',
    path: '/shared/notes.txt',
    name: 'notes.txt',
    content: 'v2 on disk',
    version: 'ver-2',
  });
  await sleep(30);
  assert.strictEqual(
    created.length,
    1,
    'a dirty tab keeps its editor: the dead reload must not replace it',
  );
  assert.strictEqual(
    editor.getModel().getValue(),
    'v1 edited after the outage',
    'the edits typed after the outage survive',
  );
  assert.ok(
    win.document.querySelector('.chat-tab-dirty'),
    'the tab strip still shows the dirty marker',
  );
  win.close();
}

(async () => {
  await testReloadSwallowedByOutageKeepsEdits();
  console.log('  ok - reload swallowed by an outage keeps the edits');
})().catch(e => {
  console.error(e);
  process.exit(1);
});
