// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end (jsdom) regression tests for the review findings on the
// in-page dialog work in media/main.js:
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
//   R4-2  A prompt the daemon refuses comes back into the composer.
//
// Each test reproduces the reviewer's sequence and fails on the old code.
/* global require, console, process */
'use strict';

const assert = require('assert');
const h = require('./ui_antipattern_harness');
const {typeInto, waitFor, persistedDrafts} = require('./simplify2_harness');
const {barButtons} = require('./worktree_bar_helpers');

const {test, report} = h.makeRunner();

const FILE_ENTRIES = [
  {name: 'foo.py', path: h.WD + '/foo.py', isDir: false},
  {name: 'src', path: h.WD + '/src', isDir: true},
];

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
    const {win, posted} = h.makePage();
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
    const {win, posted} = h.makePage();
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
    const {win, posted} = h.makePage();
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
    const ctx = h.makePage();
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
    const ctx = h.makePage();
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
    const ctx = h.makePage();
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
    const ctx = h.makePage();
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
    const {win, posted} = h.makePage();
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
    const first = h.toast(
      win,
      h.overwriteId('', actions[0].dest, actions[0].dest.split('/').pop()),
    );
    assert.ok(first, 'first question open');
    h.send(win, {
      type: 'fsResult',
      token: actions[1].token,
      error: 'exists',
      exists: true,
    });
    const second = h.toast(
      win,
      h.overwriteId('', actions[1].dest, actions[1].dest.split('/').pop()),
    );
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
    assert.strictEqual(
      h.toast(
        win,
        h.overwriteId('', actions[1].dest, actions[1].dest.split('/').pop()),
      ),
      null,
    );
    assert.ok(first.isConnected, 'Escape only dismissed the focused question');
    // Replacing on the first question re-sends that rename, and only it.
    h.click(win, h.toastButton(first, 'Replace'));
    const resent = h.ofType(posted, 'fsAction').slice(2);
    assert.strictEqual(resent.length, 1);
    assert.strictEqual(resent[0].dest, actions[0].dest);
    assert.strictEqual(resent[0].overwrite, true);
    assert.strictEqual(
      h.toast(
        win,
        h.overwriteId('', actions[0].dest, actions[0].dest.split('/').pop()),
      ),
      null,
    );
    win.close();
  });

  await test('two pastes into one folder ask one overwrite question each', () => {
    const {win, posted} = h.makePage();
    h.openExplorer(win, posted, [
      ...FILE_ENTRIES,
      {name: 'bar.py', path: h.WD + '/bar.py', isDir: false},
    ]);
    // Copy foo.py and bar.py into src: both requests carry the same
    // destination folder, so the question must be keyed by entry name.
    for (const name of ['foo.py', 'bar.py']) {
      h.runMenuItem(win, h.explorerRow(win, h.WD + '/' + name), 'Copy');
      h.runMenuItem(win, h.explorerRow(win, h.WD + '/src'), 'Paste');
    }
    const actions = h.ofType(posted, 'fsAction');
    assert.strictEqual(actions.length, 2, 'two copies pending');
    assert.strictEqual(actions[0].dest, actions[1].dest, 'same folder');
    for (const action of actions) {
      h.send(win, {
        type: 'fsResult',
        token: action.token,
        error: 'exists',
        exists: true,
      });
    }
    const first = h.toast(win, h.overwriteId('', actions[0].dest, 'foo.py'));
    const second = h.toast(win, h.overwriteId('', actions[0].dest, 'bar.py'));
    assert.ok(first && second, 'both questions are open at once');
    assert.ok(
      first
        .querySelector('.kiss-notification-message')
        .textContent.includes("'foo.py'") &&
        second
          .querySelector('.kiss-notification-message')
          .textContent.includes("'bar.py'"),
      'each question names its own entry',
    );
    h.click(win, h.toastButton(second, 'Replace'));
    const resent = h.ofType(posted, 'fsAction').slice(2);
    assert.strictEqual(resent.length, 1, 'only the answered copy is re-sent');
    assert.strictEqual(resent[0].path, h.WD + '/bar.py');
    assert.strictEqual(resent[0].overwrite, true);
    assert.ok(first.isConnected, 'the other question is still open');
    win.close();
  });

  await test('2026-11-04 W3 F3: the same clash on two machines asks one question each', () => {
    const {win, posted} = h.makePage();
    const entries = [
      {name: 'a.txt', path: h.WD + '/a.txt', isDir: false},
      // A legal file name that reads like "a.txt on gpu-box".
      {name: 'a.txt@gpu-box', path: h.WD + '/a.txt@gpu-box', isDir: false},
      {name: 'dest', path: h.WD + '/dest', isDir: true},
    ];
    h.openExplorer(win, posted, entries);
    const copyToDest = name => {
      h.runMenuItem(win, h.explorerRow(win, h.WD + '/' + name), 'Copy');
      h.runMenuItem(win, h.explorerRow(win, h.WD + '/dest'), 'Paste');
      return h
        .ofType(posted, 'fsAction')
        .filter(m => m.action === 'copy')
        .pop();
    };
    const local = copyToDest('a.txt');
    const localAt = copyToDest('a.txt@gpu-box');
    const owner = local.tabId;
    h.send(win, {type: 'tabMachine', tabId: owner, machine: 'gpu-box'});
    h.send(win, {
      type: 'machineWorkDirs',
      machine: 'gpu-box',
      workDir: h.WD,
      recentWorkDirs: [{path: h.WD, ts: 1}],
      error: '',
    });
    const list = h
      .ofType(posted, 'listDir')
      .filter(m => m.path === h.WD && m.machine === 'gpu-box')
      .pop();
    h.send(win, {
      type: 'dirListing',
      token: list.token,
      path: h.WD,
      root: h.WD,
      entries,
      machine: 'gpu-box',
    });
    const remote = copyToDest('a.txt');
    assert.strictEqual(remote.machine, 'gpu-box');
    for (const req of [local, localAt, remote]) {
      h.send(win, {
        type: 'fsResult',
        token: req.token,
        error: 'exists',
        exists: true,
        tabId: owner,
        machine: req.machine || '',
      });
    }
    const prompts = () =>
      Array.from(win.document.querySelectorAll('.kiss-notification')).filter(
        el => String(el.dataset.notificationId).startsWith('fs-overwrite:'),
      );
    assert.strictEqual(
      prompts().length,
      3,
      'both machines, and the @-named file, await their answer',
    );
    const localPrompt = h.toast(win, h.overwriteId('', local.dest, 'a.txt'));
    const remotePrompt = h.toast(
      win,
      h.overwriteId('gpu-box', remote.dest, 'a.txt'),
    );
    assert.ok(localPrompt && remotePrompt);
    assert.ok(
      h.toast(win, h.overwriteId('', localAt.dest, 'a.txt@gpu-box')),
      "the @-named file's own question",
    );
    const n = h.ofType(posted, 'fsAction').length;
    h.click(win, h.toastButton(localPrompt, 'Replace'));
    const retries = h.ofType(posted, 'fsAction').slice(n);
    assert.strictEqual(retries.length, 1);
    assert.ok(!retries[0].machine, 'Replace retries the local copy');
    assert.strictEqual(retries[0].overwrite, true);
    assert.ok(remotePrompt.isConnected, "gpu-box's question is still open");
    win.close();
  });

  await test('2026-11-04 W1 F2: an overwrite question for a file named with a newline is asked', () => {
    const {win, posted} = h.makePage();
    const name = 'odd\nname.py';
    h.openExplorer(win, posted, [
      ...FILE_ENTRIES,
      {name, path: h.WD + '/' + name, isDir: false},
    ]);
    const row = Array.from(win.document.querySelectorAll('.explorer-row')).find(
      r => r.dataset.explorerPath === h.WD + '/' + name,
    );
    h.runMenuItem(win, row, 'Copy');
    h.runMenuItem(win, h.explorerRow(win, h.WD + '/src'), 'Paste');
    const action = h.ofType(posted, 'fsAction').pop();
    assert.strictEqual(action.path, h.WD + '/' + name, 'the newline survives the request');
    const errors = [];
    win.addEventListener('error', e => {
      errors.push(String(e.error));
      e.preventDefault();
    });
    h.send(win, {type: 'fsResult', token: action.token, error: 'exists', exists: true});
    assert.deepStrictEqual(errors, [], 'no selector error');
    const question = Array.from(win.document.querySelectorAll('.kiss-notification')).find(
      el => el.dataset.notificationId === h.overwriteId('', action.dest, name),
    );
    assert.ok(question, 'the question is up');
    h.click(win, h.toastButton(question, 'Replace'));
    assert.ok(!question.isConnected, 'answered, the question goes');
    assert.strictEqual(h.ofType(posted, 'fsAction').pop().overwrite, true);
    win.close();
  });

  await test('R1-4 cancelling Create Tag step 2 returns focus to the commit row', () => {
    const {win, posted} = h.makePage();
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

  await test('R1-4 a dialog opened from a tab entry gets its opener back after the strip repaints', async () => {
    // On the mobile remote a file tab replaces the chat and hides the
    // composer (body.content-tab-open).  The question's opener is the
    // file's close control on the strip.  A repaint of the strip while
    // the question is open keeps that control (a tab keeps its node),
    // so the focus goes back to it, never to the hidden textarea.
    const ctx = h.makePage({narrow: true});
    const {win, posted} = ctx;
    await h.openDirtyContentTab(ctx);
    const control = closeControl(win);
    control.focus();
    h.key(win, control, 'Enter');
    const t = dirtyQuestion(win);
    // A Save-bar save repaints the tab strip: the dirty mark goes.
    h.click(win, win.document.querySelector('.content-save-btn'));
    const saves = h.ofType(posted, 'saveFile');
    h.send(win, {
      type: 'fileSaved',
      token: saves[0].token,
      ok: true,
      version: 'v2',
    });
    assert.strictEqual(
      win.document.querySelector('#tab-list .chat-tab-dirty'),
      null,
      'precondition: the strip was repainted',
    );
    assert.ok(
      win.document.body.classList.contains('content-tab-open'),
      'precondition: the composer is hidden behind the file tab',
    );
    h.click(win, h.toastButton(t, 'Keep editing'));
    assert.strictEqual(
      win.document.activeElement,
      control,
      'the control the question was opened from',
    );
    win.close();
  });

  await test('R1-4 a dialog whose opener left the strip hands focus to the visible tab', async () => {
    // Leaving the file for its chat takes the file's entry, the
    // question's opener, off the mobile strip.  Back on the file the
    // composer is hidden again: the answer hands the focus to the
    // active strip entry, never to the hidden textarea or the body.
    const ctx = h.makePage({narrow: true});
    const {win} = ctx;
    await h.openDirtyContentTab(ctx);
    const control = closeControl(win);
    control.focus();
    h.key(win, control, 'Enter');
    const fileId = win._testApi.getActiveTabId();
    win._testApi.switchToTab('a1');
    assert.ok(!control.isConnected, 'precondition: the opener is gone');
    win._testApi.switchToTab(fileId);
    const keep = h.toastButton(dirtyQuestion(win), 'Keep editing');
    keep.focus(); // as the press of a real click does
    h.click(win, keep);
    assert.strictEqual(
      win.document.activeElement,
      win.document.querySelector('#tab-list .chat-tab.active'),
      'the visible active tab',
    );
    win.close();
  });

  // ---- R2-1 -----------------------------------------------------------

  await test('R2-1 a promptlet acknowledgement leaves a newer draft in the Add box', () => {
    const {win, posted} = h.makePage();
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
    const {win, posted} = h.makePage();
    const tabId = win._testApi.getActiveTabId();
    h.send(win, {type: 'worktree_done', tabId, branch: 'kiss_wt-42'});
    const discard = barButtons(win).find(b => b.textContent === 'Discard');
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
    const {win, posted} = h.makePage();
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
    const {win, posted} = h.makePage();
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

  // ---- R4-2 -----------------------------------------------------------

  await test('R4-2 a refused prompt comes back into the empty composer, attachments too', async () => {
    const {win, posted} = h.makePage();
    const tabId = win._testApi.getActiveTabId();
    const inp = h.byId(win, 'task-input');
    pastePng(win, 'shot.png');
    await waitFor(() => !h.byId(win, 'send-btn').disabled, 'attachment read done');
    typeInto(win, inp, 'do the thing');
    h.click(win, h.byId(win, 'send-btn'));
    await waitFor(() => h.ofType(posted, 'submit').length, 'submit posted');
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
    const {win, posted} = h.makePage();
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
    const {win, posted} = h.makePage();
    const inp = h.byId(win, 'task-input');
    typeInto(win, inp, 'B prompt still awaiting acknowledgement');
    h.click(win, h.byId(win, 'send-btn'));
    assert.strictEqual(h.ofType(posted, 'submit').length, 1);
    assert.strictEqual(inp.value, '', 'the composer was cleared on send');
    // The daemon refuses a prompt of a tab this page never had (a
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

  /** The drafts the page would hand a reloaded page, by tab. */
  function drafts(win) {
    return persistedDrafts(win, win.acquireTransport().getState);
  }

  await test('2026-11-04 W5 F1: the run echo sent before the refusal does not retire the refused draft', async () => {
    for (const background of [false, true]) {
      const {win, posted} = h.makePage();
      const tabId = win._testApi.getActiveTabId();
      const inp = h.byId(win, 'task-input');
      pastePng(win, 'draft.png');
      await waitFor(() => !h.byId(win, 'send-btn').disabled, 'attachment read done');
      typeInto(win, inp, 'review the attached draft');
      h.click(win, h.byId(win, 'send-btn'));
      await waitFor(() => h.ofType(posted, 'submit').length, 'submit posted');
      assert.strictEqual(inp.value, '', 'the composer was cleared on send');
      if (background) win._testApi.createNewTab();
      // The daemon echoes every run before deciding on it: the refusal
      // (a merge in progress, say) follows the echo.
      h.send(win, {
        type: 'setTaskText',
        text: 'review the attached draft',
        tabId,
      });
      h.send(win, {type: 'status', running: false, tabId});
      h.send(win, {
        type: 'error',
        code: 'prompt_refused',
        text: 'merge in progress',
        tabId,
      });
      if (background) win._testApi.switchToTab(tabId);
      assert.strictEqual(
        inp.value,
        'review the attached draft',
        'the text is back',
      );
      assert.strictEqual(
        h.byId(win, 'file-chips').children.length,
        1,
        'the attachment is back',
      );
      win.close();
    }
    // An accepted run (its echo, then its `clear`) leaves no draft to
    // restore: a later refusal of another run restores that run only.
    const {win, posted} = h.makePage();
    const tabId = win._testApi.getActiveTabId();
    const inp = h.byId(win, 'task-input');
    typeInto(win, inp, 'accepted run');
    h.click(win, h.byId(win, 'send-btn'));
    assert.strictEqual(h.ofType(posted, 'submit').length, 1);
    h.send(win, {type: 'setTaskText', text: 'accepted run', tabId});
    assert.strictEqual(
      drafts(win)[tabId],
      undefined,
      'echoed: the daemon has it',
    );
    h.send(win, {type: 'clear', tabId});
    h.send(win, {type: 'status', running: false, tabId});
    typeInto(win, inp, 'refused run');
    h.click(win, h.byId(win, 'send-btn'));
    h.send(win, {type: 'setTaskText', text: 'refused run', tabId});
    h.send(win, {type: 'status', running: false, tabId});
    h.send(win, {type: 'error', code: 'prompt_refused', text: 'busy', tabId});
    assert.strictEqual(inp.value, 'refused run');
    win.close();
    // A run the daemon queued as a follow-up (its echo, then its
    // `prompt` echo, no `clear`) is accepted too: a later refused
    // message, with files of its own, comes back, not the follow-up.
    const queued = h.makePage();
    const qTab = queued.win._testApi.getActiveTabId();
    const qInp = h.byId(queued.win, 'task-input');
    typeInto(queued.win, qInp, 'accepted queued follow-up');
    h.click(queued.win, h.byId(queued.win, 'send-btn'));
    h.send(queued.win, {type: 'status', running: true, tabId: qTab});
    h.send(queued.win, {
      type: 'setTaskText',
      text: 'accepted queued follow-up',
      tabId: qTab,
    });
    h.send(queued.win, {
      type: 'prompt',
      steer: true,
      text: 'accepted queued follow-up',
      tabId: qTab,
    });
    pastePng(queued.win, 'refused.png');
    await waitFor(
      () => !h.byId(queued.win, 'send-btn').disabled,
      'attachment read done',
    );
    typeInto(queued.win, qInp, 'refused steering message');
    h.click(queued.win, h.byId(queued.win, 'send-btn'));
    await waitFor(
      () => h.ofType(queued.posted, 'appendUserMessage').length,
      'appendUserMessage posted',
    );
    assert.strictEqual(
      h.ofType(queued.posted, 'appendUserMessage')[0].prompt,
      'refused steering message',
    );
    h.send(queued.win, {
      type: 'error',
      code: 'prompt_refused',
      text: 'finished',
      tabId: qTab,
    });
    assert.strictEqual(
      qInp.value,
      'refused steering message',
      'the refused message is back',
    );
    assert.strictEqual(
      h.byId(queued.win, 'file-chips').children.length,
      1,
      'with its attachment',
    );
    assert.strictEqual(
      drafts(queued.win)[qTab],
      'refused steering message',
    );
    queued.win.close();
  });

  await test("2026-11-04 W4 F2: an earlier message's echo keeps the newer unacknowledged one", () => {
    const {win, posted} = h.makePage();
    const tabId = win._testApi.getActiveTabId();
    const inp = h.byId(win, 'task-input');
    h.send(win, {type: 'status', running: true, tabId});
    for (const text of ['FIRST steering message', 'SECOND steering message']) {
      typeInto(win, inp, text);
      h.click(win, h.byId(win, 'send-btn'));
    }
    assert.deepStrictEqual(
      h.ofType(posted, 'appendUserMessage').map(m => m.prompt),
      ['FIRST steering message', 'SECOND steering message'],
    );
    assert.strictEqual(drafts(win)[tabId], 'SECOND steering message');
    // The echo names no message: it acknowledges the one it carries.
    h.send(win, {
      type: 'prompt',
      steer: true,
      text: 'FIRST steering message',
      tabId,
    });
    assert.strictEqual(
      drafts(win)[tabId],
      'SECOND steering message',
      "FIRST's echo leaves SECOND's copy",
    );
    h.send(win, {
      type: 'prompt',
      steer: true,
      text: 'SECOND steering message',
      tabId,
    });
    assert.strictEqual(
      drafts(win)[tabId],
      undefined,
      "SECOND's echo retires it",
    );
    // Two messages with the same text: the daemon answers them in
    // order, so the first echo is the first message's.
    for (let i = 0; i < 2; i++) {
      typeInto(win, inp, 'Please continue');
      h.click(win, h.byId(win, 'send-btn'));
    }
    assert.strictEqual(h.ofType(posted, 'appendUserMessage').length, 4);
    h.send(win, {type: 'prompt', steer: true, text: 'Please continue', tabId});
    assert.strictEqual(
      drafts(win)[tabId],
      'Please continue',
      'the second copy outlives the first echo',
    );
    h.send(win, {type: 'prompt', steer: true, text: 'Please continue', tabId});
    assert.strictEqual(drafts(win)[tabId], undefined);
    win.close();
    // Two same-text runs submitted while idle.  The daemon echoes the
    // first run twice (its submit acknowledgment, then again once its
    // tab is registered) before its `clear`: the repeated echo is the
    // first run's still, so the second run's copy outlives it until
    // its own acknowledgment (queued as a follow-up: echo, `prompt`).
    const idle = h.makePage();
    const iTab = idle.win._testApi.getActiveTabId();
    const iInp = h.byId(idle.win, 'task-input');
    for (let i = 0; i < 2; i++) {
      typeInto(idle.win, iInp, 'Please continue');
      h.click(idle.win, h.byId(idle.win, 'send-btn'));
    }
    assert.strictEqual(h.ofType(idle.posted, 'submit').length, 2);
    const echo = {type: 'setTaskText', text: 'Please continue', tabId: iTab};
    for (let i = 0; i < 2; i++) {
      h.send(idle.win, echo);
      assert.strictEqual(
        drafts(idle.win)[iTab],
        'Please continue',
        "the first run's repeated echo leaves the second copy",
      );
    }
    h.send(idle.win, {type: 'clear', tabId: iTab});
    assert.strictEqual(
      drafts(idle.win)[iTab],
      'Please continue',
      'the first run started: the second copy stays',
    );
    h.send(idle.win, echo);
    h.send(idle.win, {...echo, type: 'prompt', steer: true});
    assert.strictEqual(
      drafts(idle.win)[iTab],
      undefined,
      "the second run's own acknowledgment retires it",
    );
    idle.win.close();
  });

  // ---- 2026-11-10 C09-F1: a prompt an outage left unacknowledged -------

  /** A page on an announced connection, as the transport always is. */
  function connectedPage() {
    const page = h.makePage();
    h.send(page.win, {type: 'daemonStatus', connected: true});
    page.tabId = page.win._testApi.getActiveTabId();
    page.inp = h.byId(page.win, 'task-input');
    return page;
  }
  function sendPrompt(page, text) {
    typeInto(page.win, page.inp, text);
    h.click(page.win, h.byId(page.win, 'send-btn'));
  }
  /** The connection drops and comes back; `owed` is still on its way. */
  function outage(page, owed) {
    h.send(page.win, {type: 'daemonStatus', connected: false, reconnecting: true});
    h.send(page.win, {type: 'daemonStatus', connected: true, unconfirmed: owed || []});
  }
  /**
   * The daemon's resync of a chat-bound tab showing `task` (id 7 unless
   * given), with the prompts it took from the tab when it names them.
   */
  function resync(page, task, events, taskId, promptsTaken) {
    const tabId = page.tabId;
    h.send(page.win, {type: 'tabs_state', tabs: [{tabId, title: task, chatId: 'c1'}]});
    h.send(page.win, {type: 'status', running: false, tabId});
    h.send(page.win, {
      type: 'task_events',
      tabId,
      task,
      task_id: taskId || 7,
      chat_id: 'c1',
      events: events || [],
      promptsTaken,
    });
  }
  const chatIds = win =>
    win._testApi
      .openTabs()
      .filter(t => !t.isContentTab && !t.isSubagentTab)
      .map(t => t.id);

  await test('C09-F1: a prompt that ran during the outage stops waiting for its echo', () => {
    const page = connectedPage();
    const {win, tabId, inp} = page;
    sendPrompt(page, 'first prompt');
    outage(page);
    assert.strictEqual(drafts(win)[tabId], 'first prompt', 'in doubt: kept');
    // Its task may be running: "+" leaves the chat open until the
    // daemon has reported on it.
    h.click(win, h.byId(win, 'new-chat-btn'));
    assert.ok(chatIds(win).includes(tabId), 'not retired while in doubt');
    win._testApi.switchToTab(tabId);
    resync(page, 'first prompt');
    assert.strictEqual(inp.value, '', 'it ran: nothing comes back');
    assert.strictEqual(drafts(win)[tabId], undefined, 'no draft for a reload');
    // The next refusal is the next prompt's.
    sendPrompt(page, 'second prompt');
    h.send(win, {type: 'error', code: 'prompt_refused', text: 'busy', tabId});
    assert.strictEqual(inp.value, 'second prompt');
    win.close();

    // A follow-up the running task took is in its transcript.
    const steer = connectedPage();
    h.send(steer.win, {type: 'status', running: true, tabId: steer.tabId});
    sendPrompt(steer, 'steer message');
    assert.strictEqual(h.ofType(steer.posted, 'appendUserMessage').length, 1);
    outage(steer);
    resync(steer, 'main task', [
      {type: 'prompt', steer: true, text: 'steer message'},
    ]);
    assert.strictEqual(steer.inp.value, '');
    assert.strictEqual(drafts(steer.win)[steer.tabId], undefined);
    steer.win.close();
  });

  await test('C09-F1: a prompt the outage lost comes back into the composer', () => {
    // The transcript shows another task.  The prompt was echoed, so it
    // is the daemon's until the daemon's word says otherwise.
    const page = connectedPage();
    sendPrompt(page, 'lost prompt');
    h.send(page.win, {type: 'setTaskText', text: 'lost prompt', tabId: page.tabId});
    outage(page);
    assert.strictEqual(drafts(page.win)[page.tabId], undefined);
    resync(page, 'older task', [{type: 'prompt', text: 'lost prompt'}]);
    assert.strictEqual(page.inp.value, 'lost prompt');
    page.win.close();

    // The daemon has no chat for the tab: no task was ever admitted.
    const fresh = connectedPage();
    sendPrompt(fresh, 'never arrived');
    outage(fresh);
    h.send(fresh.win, {
      type: 'tabs_state',
      tabs: [{tabId: fresh.tabId, title: 'new chat'}],
    });
    assert.strictEqual(fresh.inp.value, '', 'a stream may have the tab');
    // Nor has a machine's stream the tab (the bindings follow the
    // registry in the daemon's answer to `ready`).
    h.send(fresh.win, {type: 'tabMachines', machines: {}});
    assert.strictEqual(fresh.inp.value, 'never arrived');
    // Its state is known again: "+" no longer has to hold it open.
    h.click(fresh.win, h.byId(fresh.win, 'new-chat-btn'));
    assert.ok(!chatIds(fresh.win).includes(fresh.tabId), 'retired by "+"');
    fresh.win.close();

    // A newer prompt from the tab: the lost one is not offered back.
    const moved = connectedPage();
    sendPrompt(moved, 'old prompt');
    outage(moved);
    sendPrompt(moved, 'newer prompt');
    resync(moved, 'older task');
    assert.strictEqual(moved.inp.value, '');
    assert.strictEqual(drafts(moved.win)[moved.tabId], 'newer prompt');
    moved.win.close();
  });

  await test('W3 audit 2026-11-12: what the tab already showed is no answer to a lost prompt', () => {
    // The chat's previous task read the same.  Replayed after the
    // outage, that task says nothing of the new run; a new task does.
    for (const [replayedId, back] of [
      [7, 'continue'],
      [8, ''],
    ]) {
      const page = connectedPage();
      resync(page, 'continue');
      sendPrompt(page, 'continue');
      assert.strictEqual(h.ofType(page.posted, 'submit').length, 1);
      outage(page);
      resync(page, 'continue', [], replayedId);
      assert.strictEqual(page.inp.value, back, 'replayed task ' + replayedId);
      page.win.close();
    }
    // A follow-up reading like its task and like an earlier follow-up:
    // only one more message in that task's transcript is its own.
    const goOn = {type: 'prompt', steer: true, text: 'go on'};
    for (const [replayed, back] of [
      [[goOn], 'go on'],
      [[goOn, goOn], ''],
    ]) {
      const page = connectedPage();
      resync(page, 'go on', [goOn]);
      h.send(page.win, {type: 'status', running: true, tabId: page.tabId});
      sendPrompt(page, 'go on');
      assert.strictEqual(h.ofType(page.posted, 'appendUserMessage').length, 1);
      outage(page);
      resync(page, 'go on', replayed);
      assert.strictEqual(page.inp.value, back, replayed.length + ' replayed');
      page.win.close();
    }
    // The same text sent twice before the first is answered: what the
    // first left in the transcript, echoed in time or not, is no answer
    // to the second.  A first that did not run (refused, or lost in an
    // earlier outage: the user drops the draft handed back) leaves the
    // next message to the second.
    const sendTwice = (page, type) => {
      sendPrompt(page, 'go on');
      sendPrompt(page, 'go on');
      assert.strictEqual(h.ofType(page.posted, type).length, 2);
    };
    const messages = page => {
      h.send(page.win, {type: 'status', running: true, tabId: page.tabId});
      sendTwice(page, 'appendUserMessage');
    };
    const tell = (page, ev) => h.send(page.win, Object.assign({tabId: page.tabId}, ev));
    const firstOfTwo = {
      echoed: page => {
        messages(page);
        tell(page, goOn);
      },
      unanswered: messages,
      refused: page => {
        messages(page);
        tell(page, {type: 'error', code: 'prompt_refused', text: 'no'});
      },
      lost: page => {
        messages(page);
        outage(page, [{tabId: page.tabId, type: 'appendUserMessage'}]);
        resync(page, 'do it');
      },
      // Lost, and the second sent before the transcript settles it.
      unsettled: page => {
        h.send(page.win, {type: 'status', running: true, tabId: page.tabId});
        sendPrompt(page, 'go on');
        outage(page);
        sendPrompt(page, 'go on');
        assert.strictEqual(h.ofType(page.posted, 'appendUserMessage').length, 2);
      },
      // From an idle tab: the first starts task 8, which takes the
      // second as a message.
      'a run': page => {
        sendTwice(page, 'submit');
        tell(page, {type: 'setTaskText', text: 'go on'});
        tell(page, {type: 'clear'});
      },
      // Likewise, with no word from the daemon before the outage.  A new
      // task without the message is the first's as far as the page can
      // tell (the second may have started it, the first's being over):
      // the draft is offered again rather than dropped.
      'a blind run': page => sendTwice(page, 'submit'),
    };
    for (const [first, replayedMessages, back] of [
      ['echoed', 1, 'go on'],
      ['echoed', 2, ''],
      ['unanswered', 1, 'go on'],
      ['refused', 1, ''],
      ['lost', 1, ''],
      ['unsettled', 1, 'go on'],
      ['unsettled', 2, ''],
      ['a run', 0, 'go on'],
      ['a run', 1, ''],
      ['a blind run', 0, 'go on'],
      ['a blind run', 1, ''],
    ]) {
      const page = connectedPage();
      resync(page, 'do it');
      firstOfTwo[first](page);
      typeInto(page.win, page.inp, '');
      outage(page);
      const [task, id] = first.endsWith('run') ? ['go on', 8] : ['do it', 7];
      resync(page, task, Array(replayedMessages).fill(goOn), id);
      assert.strictEqual(page.inp.value, back, first + ', ' + replayedMessages + ' replayed');
      page.win.close();
    }
    // Sent before the tab's transcript is on the page: a tab bound to
    // its chat that shows no task yet (a reloaded page), or one whose
    // replay the page has asked for.  Nothing tells the prompt from what
    // that transcript held already: it comes back, whatever the replay
    // shows.
    const awaitingTranscript = {
      bound: page =>
        h.send(page.win, {
          type: 'tabs_state',
          tabs: [{tabId: page.tabId, title: 'do it', chatId: 'c1'}],
        }),
      'asked for': page => {
        h.send(page.win, {type: 'replay_deferred', tabId: page.tabId, chatId: 'c1', taskId: 7});
        assert.strictEqual(h.ofType(page.posted, 'resumeSession').length, 1);
      },
    };
    for (const [state, replayed] of [
      ['bound', [goOn]],
      ['bound', [goOn, goOn]],
      ['asked for', [goOn]],
    ]) {
      const page = connectedPage();
      awaitingTranscript[state](page);
      sendPrompt(page, 'go on');
      assert.strictEqual(h.ofType(page.posted, 'submit').length, 1);
      typeInto(page.win, page.inp, '');
      outage(page);
      resync(page, 'do it', replayed);
      assert.strictEqual(page.inp.value, 'go on', state + ', ' + replayed.length + ' replayed');
      page.win.close();
    }
    // A prompt offered again may have run all the same: this one went
    // out while a replay was asked for, and the late transcript shows it.
    // The same text sent after it still expects that message, which is
    // no answer to it.
    const twice = connectedPage();
    const running = () => h.send(twice.win, {type: 'status', running: true, tabId: twice.tabId});
    resync(twice, 'do it');
    running();
    awaitingTranscript['asked for'](twice);
    sendPrompt(twice, 'go on');
    resync(twice, 'do it');
    running();
    sendPrompt(twice, 'go on');
    assert.strictEqual(h.ofType(twice.posted, 'appendUserMessage').length, 2);
    outage(twice, [{tabId: twice.tabId, type: 'appendUserMessage'}]);
    resync(twice, 'do it', [goOn]);
    assert.strictEqual(twice.inp.value, 'go on', 'the first is offered again');
    typeInto(twice.win, twice.inp, '');
    outage(twice);
    resync(twice, 'do it', [goOn]);
    assert.strictEqual(twice.inp.value, 'go on', "the first's message is no answer to the second");
    twice.win.close();
    // A follow-up typed into a run whose id has not arrived yet, reading
    // like the run's own text: a follow-up never starts a task.
    const early = connectedPage();
    sendPrompt(early, 'build it');
    h.send(early.win, {type: 'setTaskText', text: 'build it', tabId: early.tabId});
    h.send(early.win, {type: 'clear', tabId: early.tabId});
    h.send(early.win, {type: 'status', running: true, tabId: early.tabId});
    sendPrompt(early, 'build it');
    assert.strictEqual(h.ofType(early.posted, 'appendUserMessage').length, 1);
    outage(early);
    resync(early, 'build it');
    assert.strictEqual(early.inp.value, 'build it');
    early.win.close();
  });

  // ---- 2026-11-12 R2-F1: the daemon names the prompt it answers --------

  /** The ids the page sent its prompts under, oldest first. */
  const promptIds = page =>
    page.posted
      .filter(m => m.type === 'submit' || m.type === 'appendUserMessage')
      .map(m => m.promptId);
  /** A frame of the daemon for the page's tab. */
  const tell = (page, ev) => h.send(page.win, Object.assign({tabId: page.tabId}, ev));
  const refuse = (page, ids) =>
    tell(page, {type: 'error', code: 'prompt_refused', text: 'no', promptIds: ids});

  await test('R2-F1: a tab on a machine, and a sub-agent tab, wait for the transcript of a lost prompt', () => {
    // The registry never holds a remote chat's id: the machine's
    // transcript says whether the prompt ran, as a local chat's does.
    // A tab whose stream is gone gets none.
    for (const [streamKept, replayed, back] of [
      [true, 'run on box', ''],
      [true, 'earlier task', 'run on box'],
      [false, null, 'run on box'],
    ]) {
      const page = connectedPage();
      const {win, tabId, inp} = page;
      h.send(win, {type: 'configData', config: {}, machine: ''});
      h.send(win, {
        type: 'machines',
        machines: [{name: 'box', host: 'box.example', reachable: true, kind: 'ssh'}],
        checkedAt: 1,
      });
      tell(page, {type: 'tabMachine', machine: 'box'});
      sendPrompt(page, 'run on box');
      assert.strictEqual(h.ofType(page.posted, 'submit')[0].machine, 'box');
      outage(page);
      h.send(win, {type: 'tabs_state', tabs: [{tabId, title: 'earlier'}]});
      h.send(win, {type: 'tabMachines', machines: streamKept ? {[tabId]: 'box'} : {}});
      if (replayed) {
        assert.strictEqual(inp.value, '', 'the stream has the tab');
        tell(page, {type: 'status', running: false});
        tell(page, {
          type: 'task_events',
          task: replayed,
          task_id: 9,
          chat_id: 'remote-chat',
          events: [],
        });
      }
      assert.strictEqual(inp.value, back, streamKept + ', ' + replayed);
      win.close();
    }
    // A sub-agent's tab is in no registry and on no machine either, yet
    // the daemon replays it with its chat: a message typed into it is
    // settled by that transcript, not by the bindings.
    for (const [taken, back] of [
      [true, ''],
      [false, 'look at the tests'],
    ]) {
      const page = connectedPage();
      const {win, tabId, inp} = page;
      tell(page, {type: 'status', running: true});
      h.send(win, {type: 'new_tab', task_id: 'child', parent_tab_id: tabId, taskId: ''});
      const subTab = tabId + '__sub_child';
      win._testApi.switchToTab(subTab);
      sendPrompt(page, 'look at the tests');
      const sent = h.ofType(page.posted, 'appendUserMessage');
      assert.deepStrictEqual(
        sent.map(m => m.tabId),
        [subTab],
      );
      outage(page);
      h.send(win, {type: 'tabs_state', tabs: [{tabId, title: 'parent', chatId: 'c1'}]});
      h.send(win, {type: 'tabMachines', machines: {}});
      assert.strictEqual(inp.value, '', 'its transcript is still to come');
      h.send(win, {
        type: 'task_events',
        tabId: subTab,
        task: 'child task',
        task_id: 'child',
        chat_id: 'c1',
        events: [],
        promptsTaken: taken ? [sent[0].promptId] : undefined,
      });
      assert.strictEqual(inp.value, back, 'taken: ' + taken);
      win.close();
    }
  });

  await test('R2-F1: the prompts the daemon took settle a lost prompt', () => {
    // Two identical runs sent blind.  The first started task 8 and the
    // second was lost, or both ran and the second started task 9: the
    // transcripts differ in the id alone, the prompts taken tell.
    for (const [taskId, secondRan, back] of [
      [8, false, 'go on'],
      [9, true, ''],
    ]) {
      const page = connectedPage();
      resync(page, 'do it');
      sendPrompt(page, 'go on');
      sendPrompt(page, 'go on');
      typeInto(page.win, page.inp, '');
      outage(page);
      const [first, second] = promptIds(page);
      assert.ok(first && second && first !== second, 'each prompt has its id');
      resync(page, 'go on', [], taskId, secondRan ? [first, second] : [first]);
      assert.strictEqual(page.inp.value, back, 'task ' + taskId);
      page.win.close();
    }
    // A follow-up lost while another surface typed the same text into
    // the task: the transcript holds one more message reading the same,
    // which is not this page's.  A daemon that still names a prompt it
    // took before the follow-up went out (the run's, at its `clear` or
    // in a transcript, or an earlier message's) has run ever since and
    // would name the follow-up had it taken it.  One that names none of
    // those (it started since, or names no prompt) leaves the
    // transcript to decide: a message recorded there under an id is
    // that prompt's (whatever the daemon still remembers), one
    // recorded under none may be this one.
    const goOn = {type: 'prompt', steer: true, text: 'go on'};
    const named = {
      clear: page => {
        tell(page, {type: 'setTaskText', text: 'do it', promptIds: ['earlier']});
        tell(page, {type: 'clear', promptIds: ['earlier']});
      },
      echo: page => tell(page, {type: 'prompt', steer: true, text: 'hm', promptIds: ['earlier']}),
      transcript: page => resync(page, 'do it', [], 7, ['earlier']),
      nowhere: () => {},
    };
    for (const [before, taken, back, recordedAs] of [
      ['clear', ['earlier', 'theirs'], 'go on'],
      ['echo', ['earlier', 'theirs'], 'go on'],
      ['transcript', ['earlier', 'theirs'], 'go on'],
      ['clear', ['earlier', 'theirs', 'MINE'], ''],
      ['clear', ['theirs'], ''],
      ['clear', undefined, ''],
      ['nowhere', ['earlier', 'theirs'], ''],
      ['clear', undefined, 'go on', 'theirs'],
      ['clear', undefined, '', 'MINE'],
      ['clear', ['earlier', 'theirs'], '', 'MINE'],
    ]) {
      const page = connectedPage();
      named[before](page);
      tell(page, {type: 'status', running: true});
      sendPrompt(page, 'go on');
      assert.strictEqual(h.ofType(page.posted, 'appendUserMessage').length, 1);
      outage(page);
      const id = name => (name === 'MINE' ? promptIds(page)[0] : name);
      const message = recordedAs ? Object.assign({promptIds: [id(recordedAs)]}, goOn) : goOn;
      resync(page, 'do it', [message], 7, taken && taken.map(id));
      assert.strictEqual(page.inp.value, back, [before, taken, recordedAs].join(', '));
      page.win.close();
    }
  });

  await test("R2-F1: an answer that names its prompt is that prompt's alone", () => {
    // A run echoed again after its `clear` (an agent script moved it to
    // another chat): the echo is not the same-text follow-up's.
    const page = connectedPage();
    sendPrompt(page, 'build it');
    const run = promptIds(page);
    tell(page, {type: 'setTaskText', text: 'build it', promptIds: run});
    tell(page, {type: 'clear', promptIds: run});
    tell(page, {type: 'status', running: true});
    sendPrompt(page, 'build it');
    tell(page, {type: 'setTaskText', text: 'build it', promptIds: run});
    tell(page, {type: 'clear', promptIds: run});
    assert.strictEqual(drafts(page.win)[page.tabId], 'build it', 'the follow-up still waits');
    refuse(page, [promptIds(page)[1]]);
    assert.strictEqual(page.inp.value, 'build it');
    page.win.close();

    // The messages a finished task left become one run.  Its answers
    // name them all: started, none of them waits; refused, they come
    // back as the prompt the daemon made of them.
    for (const [started, back] of [
      [true, ''],
      [false, 'first note\n\nsecond note'],
    ]) {
      const late = connectedPage();
      tell(late, {type: 'status', running: true});
      sendPrompt(late, 'first note');
      sendPrompt(late, 'second note');
      const ids = promptIds(late);
      tell(late, {type: 'setTaskText', text: 'first note\n\nsecond note', promptIds: ids});
      if (started) tell(late, {type: 'clear', promptIds: ids});
      else refuse(late, ids);
      assert.strictEqual(late.inp.value, back);
      assert.strictEqual(drafts(late.win)[late.tabId], back || undefined);
      late.win.close();
    }

    // A file opened instead of a task, and a refusal, name a prompt
    // that is not the oldest one waiting.
    const two = connectedPage();
    sendPrompt(two, 'older prompt');
    sendPrompt(two, 'notes.txt');
    sendPrompt(two, 'newer prompt');
    const [older, opened, newer] = promptIds(two);
    assert.strictEqual(h.ofType(two.posted, 'submit').length, 3);
    tell(two, {type: 'promptOpened', promptId: opened});
    refuse(two, [newer]);
    assert.strictEqual(two.inp.value, 'newer prompt');
    typeInto(two.win, two.inp, '');
    assert.strictEqual(drafts(two.win)[two.tabId], 'older prompt', 'the oldest still waits');
    refuse(two, [older]);
    assert.strictEqual(two.inp.value, 'older prompt');
    two.win.close();
  });

  await test('C09-F1: a prompt the transport still owes keeps waiting', () => {
    const page = connectedPage();
    const {win, tabId} = page;
    sendPrompt(page, 'sent before the drop');
    sendPrompt(page, 'held by the transport');
    outage(page, [
      null,
      {type: 'submit', tabId: 'other', prompt: 'held by the transport'},
      {type: 'submit', tabId, prompt: 'held by the transport'},
    ]);
    resync(page, 'sent before the drop');
    assert.strictEqual(page.inp.value, '');
    assert.strictEqual(drafts(win)[tabId], 'held by the transport');
    h.send(win, {type: 'setTaskText', text: 'held by the transport', tabId});
    h.send(win, {type: 'clear', tabId});
    assert.strictEqual(drafts(win)[tabId], undefined, 'answered as usual');
    win.close();
  });

  await test("C09-F1: a background tab's transcript is fetched at once", () => {
    const page = connectedPage();
    const {win, posted, tabId, inp} = page;
    h.send(win, {
      type: 'tabs_state',
      tabs: [
        {tabId, title: 'A', chatId: 'c1'},
        {tabId: 'second', title: 'B', chatId: 'c2'},
      ],
    });
    sendPrompt(page, 'left behind');
    win._testApi.switchToTab('second');
    outage(page);
    const before = h.ofType(posted, 'resumeSession').length;
    h.send(win, {type: 'replay_deferred', tabId: 'second', chatId: 'c2'});
    h.send(win, {type: 'replay_deferred', tabId, chatId: 'c1'});
    assert.deepStrictEqual(
      h
        .ofType(posted, 'resumeSession')
        .slice(before)
        .map(m => [m.tabId, m.scoped]),
      [
        ['second', true],
        [tabId, true],
      ],
      'the tab on screen, and the one with a prompt in doubt',
    );
    h.send(win, {
      type: 'task_events',
      tabId,
      task: 'older task',
      task_id: 3,
      chat_id: 'c1',
      events: [],
    });
    win._testApi.switchToTab(tabId);
    assert.strictEqual(inp.value, 'left behind');
    win.close();
  });

  report('ui_antipattern_review_fixes');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
