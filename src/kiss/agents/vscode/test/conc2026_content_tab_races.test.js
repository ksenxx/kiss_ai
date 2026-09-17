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
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

// The webview harness of workspaceScopedTabs.test.js plus a fake Monaco
// rich enough for the EDITABLE content-tab path: renderCodeContent needs
// getModel/onDidChangeModelContent, saveContentTab needs
// getAlternativeVersionId/getValue, revealPendingContentLine needs
// getLineCount/setPosition/revealLineInCenter/getDomNode. Each created
// editor is recorded (with its initial value) and exposes _type(text) so
// a test can make the tab dirty like a real keystroke would.
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

  const created = [];
  win.monaco = {
    editor: {
      create: (holder, opts) => {
        let value = String(opts.value === undefined ? '' : opts.value);
        let version = 1;
        const listeners = [];
        const model = {
          getAlternativeVersionId: () => version,
          getValue: () => value,
          getLineCount: () => value.split('\n').length,
        };
        const editor = {
          getModel: () => model,
          onDidChangeModelContent: cb => listeners.push(cb),
          setPosition: () => {},
          revealLineInCenter: () => {},
          getDomNode: () => holder,
          dispose: () => {},
          layout: () => {},
          focus: () => {},
          _type: text => {
            value = String(text);
            version += 1;
            listeners.slice().forEach(cb => cb());
          },
        };
        created.push({holder, editor, value: opts.value, opts});
        return editor;
      },
    },
  };

  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
  return {win, posted, created};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function setWorkspace(win, dir) {
  send(win, {type: 'configData', config: {work_dir: dir}, apiKeys: {}});
}

function entry(tabId, workDir, chatId) {
  return {tabId, chatId: chatId || '', title: tabId, workDir: workDir || ''};
}

function clickEl(win, el) {
  el.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

async function waitFor(predicate, message, timeoutMs = 2000) {
  const start = Date.now();
  for (;;) {
    const value = predicate();
    if (value) return value;
    if (Date.now() - start >= timeoutMs) {
      throw new Error(message || 'waitFor timed out');
    }
    await sleep(10);
  }
}

function conflictToast(win) {
  return win.document.querySelector(
    '[data-notification-id^="file-save-conflict-"]',
  );
}

function toastActionButton(win, toast, label) {
  const buttons = Array.from(
    toast.querySelectorAll('.kiss-notification-action'),
  );
  return buttons.find(b => b.textContent.trim() === label) || null;
}

function contentTabStrips(win) {
  return win.document.querySelectorAll('.chat-tab.content-tab');
}

const FILE_PATH = '/shared/notes.txt';

// Drives a webview into the state every test here starts from: two chat
// tabs in different workspaces (a1 active in /ws/a, b1 hidden in /ws/b),
// a content tab for FILE_PATH owned by a1 whose editor holds unsaved
// edits, and — the only user path that arms contentReloadRequested — a
// save that hit a disk conflict, so the sticky conflict toast with the
// 'Overwrite' / 'Reload from disk' actions is on screen.
async function openDirtyConflictedTab(ctx) {
  const {win, posted, created} = ctx;
  setWorkspace(win, '/ws/a');
  send(win, {
    type: 'tabs_state',
    tabs: [entry('a1', '/ws/a', 'chat-1'), entry('b1', '/ws/b', 'chat-2')],
  });
  send(win, {
    type: 'fileContent',
    tabId: 'a1',
    path: FILE_PATH,
    name: 'notes.txt',
    content: 'v1 on disk',
    // The daemon's version stamp is what makes a file EDITABLE
    // (renderContentView keys the Save bar on it).
    version: 'ver-1',
  });
  await waitFor(
    () => created.length >= 1,
    'the content tab must create its editor',
  );
  assert.strictEqual(created[0].value, 'v1 on disk');

  // A keystroke makes the tab dirty (enables the Save button).
  created[0].editor._type('v1 edited');
  const saveBtn = win.document.querySelector('.content-save-btn');
  assert.ok(saveBtn, 'the editable content tab must have a Save button');
  assert.strictEqual(saveBtn.disabled, false, 'dirty tab: Save enabled');
  clickEl(win, saveBtn);
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
  const ctx = makeWebview();
  const {win, posted, created} = ctx;
  const toast = await openDirtyConflictedTab(ctx);

  // The user chooses to reload from disk, dropping the edits: this arms
  // contentReloadRequested and sends openFile for the tab's own scope.
  const reloadBtn = toastActionButton(win, toast, 'Reload from disk');
  assert.ok(reloadBtn, "the toast must offer 'Reload from disk'");
  clickEl(win, reloadBtn);
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
  console.log('  ok - foreign-scope error does not swallow the reload');
}

// CAND-1 guard: an error reply for the SAME scope still clears the
// flag, so a LATER plain open (a link click) keeps protecting the
// unsaved edits instead of overwriting them.
async function testSameScopeErrorStillClearsReloadFlag() {
  const ctx = makeWebview();
  const {win, created} = ctx;
  const toast = await openDirtyConflictedTab(ctx);

  const reloadBtn = toastActionButton(win, toast, 'Reload from disk');
  assert.ok(reloadBtn, "the toast must offer 'Reload from disk'");
  clickEl(win, reloadBtn);

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
  await sleep(80); // let any (wrong) async re-render surface
  assert.strictEqual(
    created.length,
    1,
    'after its own reload failed, a plain re-open must not replace ' +
      "the tab's unsaved edits",
  );
  assert.strictEqual(
    created[0].editor.getModel().getValue(),
    'v1 edited',
    'the unsaved edits must survive the failed reload + re-open',
  );
  win.close();
  console.log('  ok - same-scope error still clears the reload flag');
}

// CAND-2: closing the content tab must remove its sticky conflict
// toast from the DOM.
async function testClosingTabRemovesConflictToast() {
  const ctx = makeWebview();
  const {win} = ctx;
  await openDirtyConflictedTab(ctx);

  // Close the (dirty) content tab, confirming the unsaved-edits prompt.
  win.confirm = () => true;
  const strip = win.document.querySelector('.chat-tab.content-tab');
  assert.ok(strip, 'the content tab must have a strip in the tab bar');
  const closeBtn = strip.querySelector('.chat-tab-close');
  assert.ok(closeBtn, 'the content tab strip must have a close button');
  clickEl(win, closeBtn);
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
  console.log('  ok - closing the tab removes its conflict toast');
}

// CAND-2: once the tab is gone, the conflict actions are dead — they
// must not open surprise content tabs. (Post-fix the toast is removed
// and/or its actions are guarded; either way no openFile may be sent
// for the closed tab and no new content tab may appear.)
async function testDeadConflictActionsOpenNoSurpriseTabs() {
  const ctx = makeWebview();
  const {win, posted} = ctx;
  await openDirtyConflictedTab(ctx);

  win.confirm = () => true;
  const closeBtn = win.document.querySelector(
    '.chat-tab.content-tab .chat-tab-close',
  );
  clickEl(win, closeBtn);
  assert.strictEqual(contentTabStrips(win).length, 0);

  // If the toast (wrongly) survived the close, exercise its actions
  // the way a user facing the leftover toast would.
  const opensBefore = posted.filter(m => m && m.type === 'openFile').length;
  const leftover = conflictToast(win);
  if (leftover) {
    const overwriteBtn = toastActionButton(win, leftover, 'Overwrite');
    if (overwriteBtn) clickEl(win, overwriteBtn);
    const reloadBtn = toastActionButton(win, leftover, 'Reload from disk');
    if (reloadBtn) clickEl(win, reloadBtn);
  }
  // Every openFile a dead action managed to send gets the reply the
  // daemon would produce.
  const opens = posted.filter(m => m && m.type === 'openFile');
  for (let i = opensBefore; i < opens.length; i++) {
    send(win, {
      type: 'fileContent',
      tabId: opens[i].tabId,
      path: opens[i].path,
      name: 'notes.txt',
      content: 'v2 on disk',
      version: 'ver-2',
    });
  }
  assert.strictEqual(
    contentTabStrips(win).length,
    0,
    "a closed tab's conflict actions must not open a surprise " +
      'content tab',
  );
  win.close();
  console.log('  ok - dead conflict actions open no surprise tabs');
}

(async () => {
  const tests = [
    testForeignScopeErrorMustNotSwallowReload,
    testSameScopeErrorStillClearsReloadFlag,
    testClosingTabRemovesConflictToast,
    testDeadConflictActionsOpenNoSurpriseTabs,
  ];
  let failures = 0;
  for (const t of tests) {
    try {
      await t();
      console.log('PASS', t.name);
    } catch (err) {
      failures += 1;
      console.error('FAIL', t.name);
      console.error(err && err.stack ? err.stack : err);
    }
  }
  if (failures > 0) {
    console.error(failures + ' test(s) failed');
    process.exit(1);
  }
  console.log('All ' + tests.length + ' conc2026_content_tab_races tests passed');
})();
