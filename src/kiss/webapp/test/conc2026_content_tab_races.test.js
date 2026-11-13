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
  sleep,
  click,
  stubDesktop,
  stubMonaco,
} = require('./simplify2_harness');
const {
  toastButton,
  contentTabStrips,
  openDirtyContentTab,
  makeRunner,
} = require('./ui_antipattern_harness');

// The shared chat page with the harness's fake Monaco (stubMonaco), which
// serves the EDITABLE content-tab path.  Each created editor is recorded
// (with its initial value) and exposes _type(text) so a test can make
// the tab dirty like a real keystroke would.
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
  const reloads = posted.filter(m => m && m.type === 'openFile');
  assert.strictEqual(
    reloads.length,
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

  // The tab's own reload reply arrives (the daemon echoes the
  // request's token): it MUST replace the editor text with the disk
  // content (a fresh editor holding 'v2 on disk').
  // Before the fix the foreign error had already cleared
  // contentReloadRequested, so this reply took the dirty-guard branch and
  // only revealed a line: the explicit reload silently did nothing.
  send(win, {
    type: 'fileContent',
    tabId: 'a1',
    token: reloads[0].token,
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
  const {win, posted, created} = ctx;
  const toast = await openDirtyConflictedTab(ctx);

  const reloadBtn = toastButton(toast, 'Reload from disk');
  assert.ok(reloadBtn, "the toast must offer 'Reload from disk'");
  click(win, reloadBtn);
  const reload = posted.filter(m => m && m.type === 'openFile')[0];

  // The tab's own reload FAILS (the reply echoes the request's token).
  // The edits survive, and the flag must be cleared...
  send(win, {
    type: 'fileContent',
    tabId: 'a1',
    token: reload.token,
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

// A Save clicked while a Reload from disk awaits its reply is the
// later intent: it supersedes the reload.  The daemon answers both in
// order (the read, then the write); the read's reply must not replace
// the editor with the old text and, with it, drop the save's token --
// the buffer would then read clean on the old text while the disk
// holds the saved one, and the next save would carry the old version.
async function testSaveSupersedesPendingReload() {
  const ctx = makePage();
  const {win, posted, created} = ctx;
  await openDirtyContentTab(ctx);
  const refreshBtn = win.document.querySelector('.content-refresh-btn');
  assert.ok(refreshBtn, 'the save bar has a Reload from disk button');
  click(win, refreshBtn);
  const reload = posted.filter(m => m && m.type === 'openFile').pop();
  assert.ok(reload && /^reload:/.test(reload.token), 'setup: reload sent');
  click(win, win.document.querySelector('.content-save-btn'));
  const save = posted.filter(m => m && m.type === 'saveFile').pop();
  assert.ok(save, 'setup: the save is sent while the reload is pending');
  assert.strictEqual(save.version, 'ver-1');
  send(win, {
    type: 'fileContent',
    tabId: 'a1',
    token: reload.token,
    path: FILE_PATH,
    name: 'notes.txt',
    content: 'v1 on disk',
    version: 'ver-1',
  });
  send(win, {type: 'fileSaved', token: save.token, ok: true, version: 'ver-2'});
  await sleep(0);
  assert.strictEqual(
    created.length,
    1,
    'BUG: the superseded reload replaced the editor',
  );
  assert.strictEqual(created[0].editor.getModel().getValue(), 'v1 edited');
  const status = win.document.querySelector('.content-save-status');
  assert.strictEqual(status.textContent, 'Saved', 'the save was acknowledged');
  // The next save carries the version the acknowledged save produced.
  created[0].editor._type('v1 edited again');
  click(win, win.document.querySelector('.content-save-btn'));
  const next = posted.filter(m => m && m.type === 'saveFile').pop();
  assert.notStrictEqual(next.token, save.token, 'a second save is sent');
  assert.strictEqual(next.version, 'ver-2', 'with the saved version');
  win.close();
}

async function testReloadReplyLeavesTheUserWhereTheyAre() {
  // "Reload from disk" may answer late (a slow tunnel, a forwarded
  // machine, a big file).  Its reply repaints the file's tab where it
  // is: a user who went back to the chat meanwhile stays there.
  let created;
  const {win, posted} = makeSharedPage({
    bodyClass: 'remote-chat',
    beforeScripts: w => {
      created = stubMonaco(w);
    },
  });
  const api = win._testApi;
  const file = {type: 'fileContent', path: '/w/a.txt', name: 'a.txt'};
  send(win, {
    type: 'tabs_state',
    tabs: [{tabId: 'chat-B', title: 'chat B', chatId: 'cB'}],
  });
  api.switchToTab('chat-B');
  send(win, {...file, content: 'hello', version: 'v1', tabId: 'chat-B'});
  await sleep(0);
  const tab = api.openTabs().find(t => t.isContentTab);
  assert.strictEqual(
    api.getActiveTabId(),
    tab.id,
    'precondition: the file is on screen (stacked)',
  );
  click(win, win.document.querySelector('.content-refresh-btn'));
  const req = posted.filter(m => m.type === 'openFile').pop();
  assert.strictEqual(String(req.token).indexOf('reload:'), 0);
  // The user goes back to the chat before the reply arrives.
  api.switchToTab('chat-B');
  send(win, {
    ...file,
    content: 'hello again',
    version: 'v2',
    tabId: req.tabId,
    token: req.token,
  });
  await sleep(0);
  assert.strictEqual(api.getActiveTabId(), 'chat-B', 'not pulled to the file');
  assert.strictEqual(
    created[created.length - 1].value,
    'hello again',
    'the reload itself took: the tab holds the text from disk',
  );
  win.close();

  // Split layout: the user picked another file in the content pane.
  const split = makeSharedPage({
    bodyClass: 'remote-chat',
    beforeScripts: w => {
      stubDesktop(w, true);
      stubMonaco(w);
    },
  });
  const shown = () =>
    split.win.document
      .querySelector('#content-tab-list .chat-tab.active')
      .getAttribute('aria-label');
  send(split.win, {...file, content: 'hello', version: 'v1'});
  await sleep(0);
  click(split.win, split.win.document.querySelector('.content-refresh-btn'));
  const again = split.posted.filter(m => m.type === 'openFile').pop();
  send(split.win, {
    type: 'fileContent',
    path: '/w/b.txt',
    name: 'b.txt',
    content: 'other',
    version: 'v1',
  });
  await sleep(0);
  assert.strictEqual(shown(), 'b.txt', 'precondition: the other file is shown');
  send(split.win, {
    ...file,
    content: 'hello again',
    version: 'v2',
    tabId: again.tabId,
    token: again.token,
  });
  await sleep(0);
  assert.strictEqual(shown(), 'b.txt', 'the pane keeps the file picked since');
  split.win.close();
}

function saves(posted) {
  return posted.filter(m => m && m.type === 'saveFile');
}

/**
 * Click the content tab's close button.  Returns the "unsaved changes"
 * question, null when the tab closed without one.
 */
function closeContentTabStrip(win) {
  click(
    win,
    win.document.querySelector('.chat-tab.content-tab .chat-tab-close'),
  );
  return win.document.querySelector('[data-notification-id^="close-dirty-"]');
}

// W1 audit 2026-11-12: a save in flight means the disk is about to hold
// the text sent.  An undo back to the text the file was opened with read
// as "nothing to save" meanwhile (Monaco's version id is the saved one
// again): the tab closed without a question, the disk took the saved
// text and the text the user went back to was gone.
async function testUndoDuringASaveKeepsTheTabDirty() {
  const ctx = makePage();
  const {win, posted} = ctx;
  const {editor} = await openDirtyContentTab(ctx);
  click(win, win.document.querySelector('.content-save-btn'));
  const first = saves(posted)[0];
  assert.strictEqual(first.content, 'v1 edited');
  editor._undo();
  assert.strictEqual(editor.getModel().getValue(), 'v1 on disk');

  const question = closeContentTabStrip(win);
  assert.strictEqual(
    contentTabStrips(win).length,
    1,
    'BUG - the tab closed while its save was in flight',
  );
  assert.ok(question, 'the close asks first');
  assert.ok(win.document.querySelector('.chat-tab-dirty'), 'and shows the dot');
  // The close waits for the save in flight.  Its reply makes the text
  // the user went back to the unsaved one, so the close asks again.
  click(win, toastButton(question, 'Save and close'));
  assert.strictEqual(saves(posted).length, 1, 'one save at a time');
  send(win, {
    type: 'fileSaved',
    token: first.token,
    ok: true,
    version: 'ver-2',
  });
  assert.strictEqual(contentTabStrips(win).length, 1, 'not on disk yet');
  const again = win.document.querySelector(
    '[data-notification-id^="close-dirty-"]',
  );
  assert.ok(again, 'the close asks again');
  click(win, toastButton(again, 'Save and close'));
  const second = saves(posted)[1];
  assert.strictEqual(second.content, 'v1 on disk');
  assert.strictEqual(second.version, 'ver-2', 'on top of the first save');
  send(win, {
    type: 'fileSaved',
    token: second.token,
    ok: true,
    version: 'ver-3',
  });
  assert.strictEqual(contentTabStrips(win).length, 0, 'saved, then closed');
  win.close();
}

// The reply says what the disk holds.  A refused save wrote nothing: the
// text the user went back to is the disk's again and the tab closes
// without a question.  A save whose reply never came may have been
// written: no text is known to be the disk's, and the close asks.
async function testUndoneTextAfterARefusedOrALostSave() {
  const refused = makePage();
  const {editor} = await openDirtyContentTab(refused);
  click(refused.win, refused.win.document.querySelector('.content-save-btn'));
  editor._undo();
  send(refused.win, {
    type: 'fileSaved',
    token: saves(refused.posted)[0].token,
    ok: false,
    error: 'disk full',
  });
  assert.strictEqual(
    closeContentTabStrip(refused.win),
    null,
    'nothing unsaved',
  );
  assert.strictEqual(contentTabStrips(refused.win).length, 0);
  refused.win.close();

  const lost = makePage();
  send(lost.win, {type: 'daemonStatus', connected: true});
  const typed = await openDirtyContentTab(lost);
  click(lost.win, lost.win.document.querySelector('.content-save-btn'));
  typed.editor._undo();
  send(lost.win, {type: 'daemonStatus', connected: false});
  assert.ok(
    closeContentTabStrip(lost.win),
    'BUG - the tab closed although its save may have been written',
  );
  assert.strictEqual(contentTabStrips(lost.win).length, 1);
  lost.win.close();
}

(async () => {
  const {test, report} = makeRunner();
  await test(
    'an undo during a save keeps the tab dirty',
    testUndoDuringASaveKeepsTheTabDirty,
  );
  await test(
    'the undone text after a refused or a lost save',
    testUndoneTextAfterARefusedOrALostSave,
  );
  await test(
    'a late reload reply leaves the user where they are',
    testReloadReplyLeavesTheUserWhereTheyAre,
  );
  await test(
    'a foreign-scope error does not swallow the reload',
    testForeignScopeErrorMustNotSwallowReload,
  );
  await test(
    'a save sent while a reload is pending supersedes the reload',
    testSaveSupersedesPendingReload,
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
