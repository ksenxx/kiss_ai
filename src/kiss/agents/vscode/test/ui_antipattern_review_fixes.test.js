// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end (jsdom) regression tests for the review findings on the
// in-webview dialog work in media/main.js (tmp/review-R1.md, R2.md):
//
//   R1-1  Git prompts (Create Branch / Create Tag / Compare with) act on
//         the repository and tab they were opened from, not on whatever
//         is selected when the prompt is submitted.
//   R1-2  "Save and close" closes a tab that became clean while the
//         question was open.
//   R1-3  "Keep editing" (and Escape / X) withdraws a pending
//         save-and-close without cancelling the save.
//   R1-4  Dismissing a dialog by its X, replacing a dialog with the same
//         id, or chaining prompts (Create Tag step 2) hands focus back
//         to the original opener, never to <body>.
//   R2-1  A promptlet-add acknowledgement clears only the acknowledged
//         text, not a newer draft.
//   R2-2  Inline Delete / Keep confirms answer Escape without closing the
//         sheet around them, and a closing sheet resets them.
//   R2-3  A finished task never raises its editor tab after the user
//         has interacted since submitting.
//   R4-2  A prompt the daemon refuses comes back into the composer.
//
// Each test reproduces the reviewer's sequence and fails on the old code.
'use strict';

const assert = require('assert');
const h = require('./ui_antipattern_harness');

const {test, report} = h.makeRunner();

const FILE_ENTRIES = [
  {name: 'foo.py', path: h.WD + '/foo.py', isDir: false},
  {name: 'src', path: h.WD + '/src', isDir: true},
];

function typeInto(win, el, value) {
  el.value = value;
  el.dispatchEvent(new win.Event('input', {bubbles: true}));
}

function switchWorkDir(win, dir) {
  h.send(win, {type: 'configData', config: {work_dir: dir}});
}

function dirtyQuestion(win) {
  return win.document.querySelector('[data-notification-id^="close-dirty-"]');
}

function closeControl(win) {
  return win.document.querySelector('.chat-tab.content-tab .chat-tab-close');
}

/** Paste a tiny PNG into the composer, as the clipboard does. */
function pastePng(win, name) {
  const file = new win.File([new Uint8Array([137, 80, 78, 71])], name, {
    type: 'image/png',
  });
  const ev = new win.Event('paste', {bubbles: true, cancelable: true});
  Object.defineProperty(ev, 'clipboardData', {
    value: {items: [{kind: 'file', getAsFile: () => file}]},
  });
  h.byId(win, 'task-input').dispatchEvent(ev);
}

async function main() {
  // ---- R1-1 -----------------------------------------------------------

  await test('R1-1 Create Branch keeps the repository and tab it was opened from', () => {
    const {win, posted} = h.makeWebview();
    const row = h.openScm(win, posted);
    const openedOn = win._testApi.getActiveTabId();
    h.runMenuItem(win, row, 'Create Branch...');
    const t = h.toast(win, 'git-create-branch');
    // The prompt is not modal: the user switches repositories meanwhile.
    switchWorkDir(win, '/ws/other');
    h.toastInput(t).value = 'new-name';
    h.click(win, h.toastButton(t, 'Create branch'));
    const acts = h.ofType(posted, 'gitAction');
    assert.strictEqual(acts.length, 1);
    assert.strictEqual(acts[0].sha, h.SHA_A);
    assert.strictEqual(acts[0].workDir, h.WD, 'the repository of the commit');
    assert.strictEqual(acts[0].tabId, openedOn, 'the tab that asked');
    win.close();
  });

  await test('R1-1 Create Tag carries the opening repository through both steps', () => {
    const {win, posted} = h.makeWebview();
    const row = h.openScm(win, posted);
    h.runMenuItem(win, row, 'Create Tag...');
    const t1 = h.toast(win, 'git-create-tag');
    switchWorkDir(win, '/ws/other');
    h.toastInput(t1).value = 'v1';
    h.key(win, h.toastInput(t1), 'Enter');
    const t2 = h.toast(win, 'git-create-tag-message');
    assert.ok(t2, 'step 2 opens');
    h.toastInput(t2).value = 'release';
    h.click(win, h.toastButton(t2, 'Create tag'));
    const acts = h.ofType(posted, 'gitAction');
    assert.strictEqual(acts.length, 1);
    assert.strictEqual(acts[0].action, 'createTag');
    assert.strictEqual(acts[0].workDir, h.WD);
    win.close();
  });

  await test('R1-1 Compare with shows the diff in the repository it was opened from', () => {
    const {win, posted} = h.makeWebview();
    const row = h.openScm(win, posted);
    h.runMenuItem(win, row, 'Compare with...');
    const t = h.toast(win, 'git-compare-with');
    switchWorkDir(win, '/ws/other');
    h.toastInput(t).value = 'main';
    h.click(win, h.toastButton(t, 'Compare'));
    const shows = h.ofType(posted, 'gitShow');
    assert.strictEqual(shows.length, 1);
    assert.strictEqual(shows[0].base, 'main');
    assert.strictEqual(shows[0].workDir, h.WD);
    win.close();
  });

  // ---- R1-2 / R1-3 ---------------------------------------------------

  await test('R1-2 Save and close closes a tab saved by the Save bar while the question was open', async () => {
    const ctx = h.makeWebview();
    const {win, posted} = ctx;
    await h.openDirtyContentTab(ctx);
    h.click(win, closeControl(win));
    const t = dirtyQuestion(win);
    assert.ok(t, 'the question is open');
    // The user saves from the Save bar instead, leaving the question up.
    h.click(win, win.document.querySelector('.content-save-btn'));
    const saves = h.ofType(posted, 'saveFile');
    assert.strictEqual(saves.length, 1);
    h.send(win, {
      type: 'fileSaved',
      token: saves[0].token,
      ok: true,
      version: 'ver-2',
    });
    assert.strictEqual(h.contentTabStrips(win).length, 1, 'still open');
    h.click(win, h.toastButton(t, 'Save and close'));
    assert.strictEqual(h.ofType(posted, 'saveFile').length, 1, 'no new save');
    assert.strictEqual(h.contentTabStrips(win).length, 0, 'closed at once');
    assert.strictEqual(dirtyQuestion(win), null);
    win.close();
  });

  await test('R1-3 Keep editing withdraws a pending save-and-close; the save still lands', async () => {
    const ctx = h.makeWebview();
    const {win, posted, created} = ctx;
    await h.openDirtyContentTab(ctx);
    h.click(win, closeControl(win));
    h.click(win, h.toastButton(dirtyQuestion(win), 'Save and close'));
    const saves = h.ofType(posted, 'saveFile');
    assert.strictEqual(saves.length, 1, 'the save is in flight');
    // Second thoughts before the reply: close again, then Keep editing.
    h.click(win, closeControl(win));
    const again = dirtyQuestion(win);
    assert.ok(again, 'the question is asked again');
    h.click(win, h.toastButton(again, 'Keep editing'));
    h.send(win, {
      type: 'fileSaved',
      token: saves[0].token,
      ok: true,
      version: 'ver-2',
    });
    assert.strictEqual(h.contentTabStrips(win).length, 1, 'the tab stays');
    assert.ok(
      !win.document.querySelector('.chat-tab.content-tab .chat-tab-dirty'),
      'the save itself went through: the tab is clean',
    );
    assert.strictEqual(created[0].editor.getModel().getValue(), 'v1 edited');
    win.close();
  });

  await test('R1-3 Escape on the re-asked question also withdraws the pending close', async () => {
    const ctx = h.makeWebview();
    const {win, posted} = ctx;
    await h.openDirtyContentTab(ctx);
    h.click(win, closeControl(win));
    h.click(win, h.toastButton(dirtyQuestion(win), 'Save and close'));
    const saves = h.ofType(posted, 'saveFile');
    h.click(win, closeControl(win));
    h.key(win, win.document.activeElement, 'Escape');
    assert.strictEqual(dirtyQuestion(win), null, 'Escape closed the question');
    h.send(win, {
      type: 'fileSaved',
      token: saves[0].token,
      ok: true,
      version: 'v2',
    });
    assert.strictEqual(h.contentTabStrips(win).length, 1, 'the tab stays');
    win.close();
  });

  // ---- R1-4 -----------------------------------------------------------

  await test('R1-4 the X on a dialog cancels it and returns focus to the opener', async () => {
    const ctx = h.makeWebview();
    const {win, posted} = ctx;
    await h.openDirtyContentTab(ctx);
    // First arm a save-and-close, so the X has a cancel effect to prove.
    h.click(win, closeControl(win));
    h.click(win, h.toastButton(dirtyQuestion(win), 'Save and close'));
    const saves = h.ofType(posted, 'saveFile');
    const control = closeControl(win);
    control.focus();
    h.key(win, control, 'Enter');
    const t = dirtyQuestion(win);
    assert.ok(t, 'Enter on the close control asks');
    const x = t.querySelector('.kiss-notification-close');
    x.focus();
    h.click(win, x);
    assert.strictEqual(dirtyQuestion(win), null, 'X closed the question');
    assert.strictEqual(
      win.document.activeElement,
      control,
      'focus is back on the close control, not on body',
    );
    h.send(win, {
      type: 'fileSaved',
      token: saves[0].token,
      ok: true,
      version: 'v2',
    });
    assert.strictEqual(
      h.contentTabStrips(win).length,
      1,
      'X withdrew the pending close like Keep editing does',
    );
    win.close();
  });

  await test('R1-4 two overwrite questions are asked one per destination', () => {
    const {win, posted} = h.makeWebview();
    h.openExplorer(win, posted, FILE_ENTRIES);
    const inp = h.byId(win, 'task-input');
    // Two renames go out before either reply arrives.
    h.runMenuItem(win, h.explorerRow(win, h.WD + '/foo.py'), 'Rename...');
    let box = win.document.querySelector('.explorer-input');
    box.value = 'bar.py';
    h.key(win, box, 'Enter');
    h.runMenuItem(win, h.explorerRow(win, h.WD + '/src'), 'Rename...');
    box = win.document.querySelector('.explorer-input');
    box.value = 'lib';
    h.key(win, box, 'Enter');
    const actions = h.ofType(posted, 'fsAction');
    assert.strictEqual(actions.length, 2, 'two renames pending');
    inp.focus();
    assert.strictEqual(win.document.activeElement, inp, 'precondition');
    h.send(win, {
      type: 'fsResult',
      token: actions[0].token,
      error: 'exists',
      exists: true,
    });
    const first = h.toast(win, 'fs-overwrite:' + actions[0].dest);
    assert.ok(first, 'first question open');
    h.send(win, {
      type: 'fsResult',
      token: actions[1].token,
      error: 'exists',
      exists: true,
    });
    const second = h.toast(win, 'fs-overwrite:' + actions[1].dest);
    assert.ok(
      second
        .querySelector('.kiss-notification-message')
        .textContent.includes("'lib'"),
      'the second clash asks its own question',
    );
    assert.ok(
      first.isConnected &&
        first
          .querySelector('.kiss-notification-message')
          .textContent.includes("'bar.py'"),
      'the first question is still open, not replaced by the second',
    );
    h.key(win, win.document.activeElement, 'Escape');
    assert.strictEqual(h.toast(win, 'fs-overwrite:' + actions[1].dest), null);
    assert.ok(first.isConnected, 'Escape only dismissed the focused question');
    // Replacing on the first question re-sends that rename, and only it.
    h.click(win, h.toastButton(first, 'Replace'));
    const resent = h.ofType(posted, 'fsAction').slice(2);
    assert.strictEqual(resent.length, 1);
    assert.strictEqual(resent[0].dest, actions[0].dest);
    assert.strictEqual(resent[0].overwrite, true);
    assert.strictEqual(h.toast(win, 'fs-overwrite:' + actions[0].dest), null);
    win.close();
  });

  await test('R1-4 cancelling Create Tag step 2 returns focus to the commit row', () => {
    const {win, posted} = h.makeWebview();
    const row = h.openScm(win, posted);
    h.runMenuItem(win, row, 'Create Tag...');
    assert.ok(row.tabIndex >= 0, 'precondition: the row can hold focus');
    const t1 = h.toast(win, 'git-create-tag');
    h.toastInput(t1).value = 'v2';
    h.key(win, h.toastInput(t1), 'Enter');
    const t2 = h.toast(win, 'git-create-tag-message');
    assert.strictEqual(win.document.activeElement, h.toastInput(t2));
    h.click(win, h.toastButton(t2, 'Cancel'));
    assert.strictEqual(h.toast(win, 'git-create-tag-message'), null);
    assert.notStrictEqual(win.document.activeElement, win.document.body);
    assert.strictEqual(win.document.activeElement, row, 'back on the row');
    assert.strictEqual(h.ofType(posted, 'gitAction').length, 0);
    win.close();
  });

  await test('R1-4 a dialog whose opener is gone hands focus to the visible tab', async () => {
    // On the mobile remote a file tab replaces the chat and hides the
    // composer (body.content-tab-open), so the fallback is the active
    // tab-strip entry, never the hidden textarea.  (The desktop remote
    // splits the window instead and keeps the composer on screen.)
    const ctx = h.makeWebview({narrow: true});
    const {win, posted} = ctx;
    await h.openDirtyContentTab(ctx);
    const control = closeControl(win);
    control.focus();
    h.key(win, control, 'Enter');
    const t = dirtyQuestion(win);
    // A Save-bar save re-renders the tab strip: the opener is replaced.
    h.click(win, win.document.querySelector('.content-save-btn'));
    const saves = h.ofType(posted, 'saveFile');
    h.send(win, {
      type: 'fileSaved',
      token: saves[0].token,
      ok: true,
      version: 'v2',
    });
    assert.ok(!control.isConnected, 'precondition: the old control is gone');
    assert.ok(
      win.document.body.classList.contains('content-tab-open'),
      'precondition: the composer is hidden behind the file tab',
    );
    h.click(win, h.toastButton(t, 'Keep editing'));
    const activeTab = win.document.querySelector('#tab-list .chat-tab.active');
    assert.ok(
      activeTab,
      'precondition: the file tab is the active strip entry',
    );
    assert.strictEqual(
      win.document.activeElement,
      activeTab,
      'the visible active tab, not the hidden composer or body',
    );
    win.close();
  });

  // ---- R2-1 -----------------------------------------------------------

  await test('R2-1 a promptlet acknowledgement leaves a newer draft in the Add box', () => {
    const {win, posted} = h.makeWebview();
    win.__TRICKS__ = [];
    h.click(win, h.byId(win, 'tricks-btn'));
    const box = h.byId(win, 'tricks-add-input');
    typeInto(win, box, 'first');
    h.click(win, h.byId(win, 'tricks-add-btn'));
    assert.strictEqual(h.ofType(posted, 'addTrick').length, 1);
    // The user starts the next promptlet before the daemon answers.
    typeInto(win, box, 'second');
    h.send(win, {type: 'tricksData', tricks: ['first'], userCount: 1});
    assert.strictEqual(box.value, 'second', 'the newer draft survives');
    // Only text that still equals the acknowledged submission is cleared.
    h.click(win, h.byId(win, 'tricks-add-btn'));
    h.send(win, {
      type: 'tricksData',
      tricks: ['first', 'second'],
      userCount: 2,
    });
    assert.strictEqual(box.value, '', 'the acknowledged text is cleared');
    win.close();
  });

  // ---- R2-2 -----------------------------------------------------------

  await test('R2-2 Escape on the worktree Discard confirm keeps and refocuses Discard', () => {
    const {win, posted} = h.makeWebview();
    const tabId = win._testApi.getActiveTabId();
    h.send(win, {type: 'worktree_done', tabId, branch: 'kiss_wt-42'});
    const discard = Array.from(
      win.document.querySelectorAll('.wt-bar .wt-btns .wt-btn'),
    ).find(b => b.textContent === 'Discard');
    h.click(win, discard);
    const ask = win.document.querySelector('.wt-bar .wt-confirm');
    assert.ok(!ask.hidden, 'precondition: the question is shown');
    h.key(win, win.document.activeElement, 'Escape');
    assert.ok(ask.hidden, 'Escape folds the question away');
    assert.ok(!win.document.querySelector('.wt-bar .wt-btns').hidden);
    assert.strictEqual(
      win.document.activeElement,
      discard,
      'Discard refocused',
    );
    assert.strictEqual(h.ofType(posted, 'worktreeAction').length, 0);
    win.close();
  });

  await test('R2-2 Escape on a promptlet delete confirm cancels it without closing the sheet', () => {
    const {win, posted} = h.makeWebview();
    win.__TRICKS__ = ['mine', 'theirs'];
    win.__MY_TRICKS_COUNT__ = 1;
    h.click(win, h.byId(win, 'tricks-btn'));
    const panel = h.byId(win, 'tricks-panel');
    const row = win.document.querySelector('#tricks-list .tricks-item');
    const trash = row.querySelector('.sidebar-item-delete');
    h.click(win, trash);
    const confirmWrap = row.querySelector('.sidebar-item-confirm');
    assert.ok(
      confirmWrap.contains(win.document.activeElement),
      'focus moved into the question',
    );
    h.key(win, win.document.activeElement, 'Escape');
    assert.strictEqual(confirmWrap.style.display, 'none', 'question folded');
    assert.strictEqual(trash.style.display, '', 'the trash icon is back');
    assert.ok(panel.classList.contains('open'), 'the sheet stays open');
    assert.strictEqual(win.document.activeElement, trash, 'trash refocused');
    assert.strictEqual(h.ofType(posted, 'deleteTrick').length, 0);
    // The next Escape, with nothing inner open, closes the sheet.
    h.key(win, win.document.activeElement, 'Escape');
    assert.ok(!panel.classList.contains('open'), 'sheet closed');
    win.close();
  });

  await test('R2-2 closing Settings resets an open custom-model delete confirm', () => {
    const {win, posted} = h.makeWebview();
    h.click(win, h.byId(win, 'settings-btn'));
    h.send(win, {
      type: 'myModelsData',
      models: [{name: 'model-b', endpoint: '', api_key: '', headers: ''}],
    });
    const row = win.document.querySelector('.custom-model-row');
    const trash = row.querySelector('.custom-model-delete-btn');
    const confirmWrap = row.querySelector('.sidebar-item-confirm');
    h.click(win, trash);
    assert.notStrictEqual(confirmWrap.style.display, 'none', 'precondition');
    // Escape inside the question cancels it and leaves Settings open.
    h.key(win, win.document.activeElement, 'Escape');
    assert.strictEqual(confirmWrap.style.display, 'none');
    assert.ok(h.byId(win, 'settings-panel').classList.contains('open'));
    // Left open and the sheet closed by its X: the question is reset.
    h.click(win, trash);
    h.click(win, h.byId(win, 'settings-panel-close'));
    assert.ok(!h.byId(win, 'settings-panel').classList.contains('open'));
    assert.strictEqual(confirmWrap.style.display, 'none', 'question reset');
    assert.strictEqual(trash.style.display, '', 'trash icon visible again');
    assert.strictEqual(h.ofType(posted, 'deleteMyModel').length, 0);
    win.close();
  });

  // ---- R2-3 -----------------------------------------------------------

  await test('R2-3 editor-tab mode: no revealPanel once the user interacted since submitting', () => {
    const {win, posted} = h.makeWebview({bodyClass: 'editor-tab-mode'});
    const tabId = win._testApi.getActiveTabId();
    h.send(win, {type: 'status', running: true, tabId, startTs: Date.now()});
    win._testApi.endLaunch();
    win.document.body.dispatchEvent(
      new win.KeyboardEvent('keydown', {key: 'ArrowDown', bubbles: true}),
    );
    h.send(win, {type: 'task_done', tabId, startTs: 1000, endTs: 3000});
    assert.strictEqual(
      h.ofType(posted, 'revealPanel').length,
      0,
      'the finished chat does not replace the editor the user moved to',
    );
    win.close();
  });

  await test('R2-3 editor-tab mode: an undisturbed finish still reveals the panel', () => {
    const {win, posted} = h.makeWebview({bodyClass: 'editor-tab-mode'});
    const tabId = win._testApi.getActiveTabId();
    h.send(win, {type: 'status', running: true, tabId, startTs: Date.now()});
    h.send(win, {type: 'task_done', tabId, startTs: 1000, endTs: 3000});
    assert.strictEqual(h.ofType(posted, 'revealPanel').length, 1);
    win.close();
  });

  // ---- R4-2 -----------------------------------------------------------

  await test('R4-2 a refused prompt comes back into the empty composer, attachments too', async () => {
    const {win, posted} = h.makeWebview();
    const tabId = win._testApi.getActiveTabId();
    const inp = h.byId(win, 'task-input');
    pastePng(win, 'shot.png');
    for (let i = 0; i < 100 && h.byId(win, 'send-btn').disabled; i++) {
      await h.sleep(10);
    }
    typeInto(win, inp, 'do the thing');
    h.click(win, h.byId(win, 'send-btn'));
    for (let i = 0; i < 100 && !h.ofType(posted, 'submit').length; i++) {
      await h.sleep(10);
    }
    const submits = h.ofType(posted, 'submit');
    assert.strictEqual(submits.length, 1, 'the prompt went out');
    assert.strictEqual(submits[0].attachments.length, 1);
    assert.strictEqual(inp.value, '', 'the composer was cleared on send');
    assert.strictEqual(h.byId(win, 'file-chips').children.length, 0);
    h.send(win, {type: 'status', running: false, tabId});
    h.send(win, {
      type: 'error',
      code: 'prompt_refused',
      text: 'A task is already running in this chat.',
      tabId,
    });
    assert.strictEqual(inp.value, 'do the thing', 'the text is back');
    assert.strictEqual(
      h.byId(win, 'file-chips').children.length,
      1,
      'the attachment is back',
    );
    win.close();
  });

  await test('R4-2 a refusal leaves a newer draft alone', () => {
    const {win, posted} = h.makeWebview();
    const tabId = win._testApi.getActiveTabId();
    const inp = h.byId(win, 'task-input');
    typeInto(win, inp, 'first attempt');
    h.click(win, h.byId(win, 'send-btn'));
    assert.strictEqual(h.ofType(posted, 'submit').length, 1);
    typeInto(win, inp, 'newer draft');
    h.send(win, {type: 'status', running: false, tabId});
    h.send(win, {type: 'error', code: 'prompt_refused', text: 'busy', tabId});
    assert.strictEqual(inp.value, 'newer draft', 'untouched');
    // An ordinary error never restores anything.
    typeInto(win, inp, 'second attempt');
    h.click(win, h.byId(win, 'send-btn'));
    h.send(win, {type: 'error', text: 'some other failure', tabId});
    assert.strictEqual(inp.value, '', 'a plain error restores nothing');
    win.close();
  });

  await test('R4-2 a refusal addressed to a closed or unknown tab restores nothing', () => {
    const {win, posted} = h.makeWebview();
    const inp = h.byId(win, 'task-input');
    typeInto(win, inp, 'B prompt still awaiting acknowledgement');
    h.click(win, h.byId(win, 'send-btn'));
    assert.strictEqual(h.ofType(posted, 'submit').length, 1);
    assert.strictEqual(inp.value, '', 'the composer was cleared on send');
    // The daemon refuses a prompt of a tab this webview never had (a
    // closed tab, or another window's): the active tab is not the
    // addressee and must keep its empty composer.
    h.send(win, {type: 'status', running: false, tabId: 'tab-gone-elsewhere'});
    h.send(win, {
      type: 'error',
      code: 'prompt_refused',
      text: 'A refused after it was closed',
      tabId: 'tab-gone-elsewhere',
    });
    assert.strictEqual(inp.value, '', 'nothing restored for a foreign tab');
    win.close();
  });

  report('ui_antipattern_review_fixes');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
