// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// End-to-end (jsdom) tests for two content-tab races in media/main.js.
//
// CAND-1 (handleFileContent, ev.error path): a fileContent ERROR reply
// clears `contentReloadRequested` on EVERY content tab whose
// contentPath matches, ignoring workspace scope — while the success
// path matches by scope. A failed open of the same path issued by
// ANOTHER workspace's conversation therefore wipes the flag of a dirty
// tab that just asked to reload from disk, and the tab's own later
// successful reload reply then hits the dirty-guard branch and never
// replaces the text: the user's explicit reload silently does nothing.
// Post-fix behavior asserted here:
//   - an error reply attributed to a DIFFERENT workspace scope must NOT
//     clear the flag (the tab's own success reply still reloads it);
//   - an error reply for the SAME scope still clears it (the next link
//     click must protect the edits again).
//
// CAND-2 (handleFileSaved conflict toast vs. tab close): the sticky
// 'file-save-conflict-<tabId>' notification is never dismissed when its
// content tab is closed; its 'Reload from disk' action then sends
// openFile for the CLOSED tab's path and the reply opens a brand-new
// content tab as a surprise side effect of a dead notification.
// Post-fix behavior asserted here:
//   - closing the conflicted content tab removes the toast from the DOM;
//   - after the tab is gone, the (dead) conflict actions must not open
//     surprise content tabs.

const assert = require('assert');
const {
  makePage: makeSharedPage,
  send,
  waitFor,
  click,
  stubMonaco,
} = require('./simplify2_harness');
const {
  toastButton,
  contentTabStrips,
  openDirtyContentTab,
  makeRunner,
} = require('./ui_antipattern_harness');

// The page harness of workspaceScopedTabs.test.js plus a fake Monaco
// rich enough for the EDITABLE content-tab path: renderCodeContent needs
// getModel/onDidChangeModelContent, saveContentTab needs
// getAlternativeVersionId/getValue, revealPendingContentLine needs
// getLineCount/setPosition/revealLineInCenter/getDomNode. Each created
// editor is recorded (with its initial value) and exposes _type(text) so
// a test can make the tab dirty like a real keystroke would.
function makePage() {
  let created;
  const {win, posted} = makeSharedPage({
    beforeScripts: win => {
      created = stubMonaco(win);
    },
  });
  return {win, posted, created};
}

function conflictToast(win) {
  return win.document.querySelector(
    '[data-notification-id^="file-save-conflict-"]',
  );
}

/**
 * Answer the in-page "unsaved changes" question (it replaced
 * window.confirm) with "Don't save".
 */
function discardEditsAndClose(win) {
  const toast = win.document.querySelector(
    '[data-notification-id^="close-dirty-"]',
  );
  assert.ok(toast, 'closing a dirty tab must ask in-page first');
  click(win, toastButton(toast, "Don't save"));
}

const FILE_PATH = '/shared/notes.txt';

// Drives a page into the state every test here starts from: two chat
// tabs in different workspaces (a1 active in /ws/a, b1 hidden in /ws/b),
// a content tab for FILE_PATH owned by a1 whose editor holds unsaved
// edits (ui_antipattern_harness.openDirtyContentTab), and — the only
// user path that arms contentReloadRequested — a save that hit a disk
// conflict, so the sticky conflict toast with the 'Overwrite' /
// 'Reload from disk' actions is on screen.
async function openDirtyConflictedTab(ctx) {
  const {win, posted, created} = ctx;
  await openDirtyContentTab(ctx);
  assert.strictEqual(created[0].value, 'v1 on disk');

  // The keystroke made the tab dirty (enables the Save button).
  const saveBtn = win.document.querySelector('.content-save-btn');
  assert.ok(saveBtn, 'the editable content tab must have a Save button');
  assert.strictEqual(saveBtn.disabled, false, 'dirty tab: Save enabled');
  click(win, saveBtn);
  const saves = posted.filter(m => m && m.type === 'saveFile');
  assert.strictEqual(saves.length, 1, 'Save must send exactly one saveFile');

  // The daemon refuses: the file changed on disk since it was opened.
  send(win, {
    type: 'fileSaved',
    token: saves[0].token,
    ok: false,
    conflict: true,
    error: 'notes.txt changed on disk',
  });
  const toast = conflictToast(win);
  assert.ok(toast, 'the save conflict must raise its sticky toast');
  return toast;
}

// CAND-1: a fileContent ERROR reply for the same path but a DIFFERENT
// workspace scope must not clear contentReloadRequested — the tab's own
// later successful reload reply must still replace its text.
async function testForeignScopeErrorMustNotSwallowReload() {
  const ctx = makePage();
  const {win, posted, created} = ctx;
  const toast = await openDirtyConflictedTab(ctx);

  // The user chooses to reload from disk, dropping the edits: this arms
  // contentReloadRequested and sends openFile for the tab's own scope.
  const reloadBtn = toastButton(toast, 'Reload from disk');
  assert.ok(reloadBtn, "the toast must offer 'Reload from disk'");
  click(win, reloadBtn);
  assert.strictEqual(
    posted.filter(m => m && m.type === 'openFile').length,
    1,
    'the reload must request the file from the daemon',
  );

  // RACE: before the reload reply lands, a request for the SAME path
  // issued by workspace /ws/b's conversation FAILS (deleted file,
  // permission). Its error reply must not touch the /ws/a tab's
  // pending reload.
  send(win, {
    type: 'fileContent',
    tabId: 'b1',
    path: FILE_PATH,
    error: 'Permission denied',
  });

  // The tab's own reload reply arrives: it MUST replace the editor
  // text with the disk content (a fresh editor holding 'v2 on disk').
  // Today the foreign error already cleared contentReloadRequested, so
  // this reply takes the dirty-guard branch and only reveals a line —
  // the explicit reload silently does nothing.
  send(win, {
    type: 'fileContent',
    tabId: 'a1',
    path: FILE_PATH,
    name: 'notes.txt',
    content: 'v2 on disk',
    version: 'ver-2',
  });
  await waitFor(
    () => created.length >= 2,
    "a foreign scope's failed open must not cancel this tab's reload: " +
      'the reload reply must re-render the editor from disk',
  );
  assert.strictEqual(
    created[created.length - 1].value,
    'v2 on disk',
    'the reloaded editor must hold the disk content',
  );
  win.close();
}

// CAND-1 guard: an error reply for the SAME scope still clears the
// flag, so a LATER plain open (a link click) keeps protecting the
// unsaved edits instead of overwriting them.
async function testSameScopeErrorStillClearsReloadFlag() {
  const ctx = makePage();
  const {win, created} = ctx;
  const toast = await openDirtyConflictedTab(ctx);

  const reloadBtn = toastButton(toast, 'Reload from disk');
  assert.ok(reloadBtn, "the toast must offer 'Reload from disk'");
  click(win, reloadBtn);

  // The tab's own reload FAILS (same scope: the reply is attributed to
  // the very conversation that owns the tab). The edits survive, and
  // the flag must be cleared...
  send(win, {
    type: 'fileContent',
    tabId: 'a1',
    path: FILE_PATH,
    error: 'notes.txt was deleted',
  });
  // ...so a later click on the file's link (a plain open, no reload
  // requested) must NOT replace the dirty editor: it only brings the
  // tab forward.
  send(win, {
    type: 'fileContent',
    tabId: 'a1',
    path: FILE_PATH,
    name: 'notes.txt',
    content: 'v3 on disk',
    version: 'ver-3',
  });
  // A (wrong) re-render of notes.txt would create its editor on the
  // same microtask chain as this sentinel open of another file, and
  // ahead of it: once the sentinel's editor exists, one editor between
  // the two proves nothing was re-rendered.
  send(win, {
    type: 'fileContent',
    tabId: 'a1',
    path: '/shared/sentinel.txt',
    name: 'sentinel.txt',
    content: 'sentinel',
    version: 'ver-s',
  });
  await waitFor(
    () => created.some(c => c.value === 'sentinel'),
    'the sentinel open must create its editor',
  );
  assert.strictEqual(
    created.length,
    2,
    'after its own reload failed, a plain re-open must not replace ' +
      "the tab's unsaved edits",
  );
  assert.strictEqual(
    created[0].editor.getModel().getValue(),
    'v1 edited',
    'the unsaved edits must survive the failed reload + re-open',
  );
  win.close();
}

// CAND-2: closing the content tab must remove its sticky conflict
// toast from the DOM (so its actions can never open a surprise content
// tab for the dead tab), and the close itself must not ask the daemon
// for the file again.
async function testClosingTabRemovesConflictToast() {
  const ctx = makePage();
  const {win, posted} = ctx;
  await openDirtyConflictedTab(ctx);

  // Close the (dirty) content tab, discarding the edits when asked.
  const strip = win.document.querySelector('.chat-tab.content-tab');
  assert.ok(strip, 'the content tab must have a strip in the tab bar');
  const closeBtn = strip.querySelector('.chat-tab-close');
  assert.ok(closeBtn, 'the content tab strip must have a close button');
  const opensBefore = posted.filter(m => m && m.type === 'openFile').length;
  click(win, closeBtn);
  discardEditsAndClose(win);
  assert.strictEqual(
    posted.filter(m => m && m.type === 'openFile').length,
    opensBefore,
    'closing the tab must not post an openFile for it',
  );
  assert.strictEqual(
    contentTabStrips(win).length,
    0,
    'the content tab must be gone after the confirmed close',
  );
  assert.strictEqual(
    conflictToast(win),
    null,
    "closing the tab must dismiss its 'file changed on disk' toast — " +
      'a sticky toast about a dead tab can never be resolved',
  );
  win.close();
}

(async () => {
  const {test, report} = makeRunner();
  await test(
    'a foreign-scope error does not swallow the reload',
    testForeignScopeErrorMustNotSwallowReload,
  );
  await test(
    'a same-scope error still clears the reload flag',
    testSameScopeErrorStillClearsReloadFlag,
  );
  await test(
    'closing the tab removes its conflict toast',
    testClosingTabRemovesConflictToast,
  );
  report('conc2026_content_tab_races');
})();
