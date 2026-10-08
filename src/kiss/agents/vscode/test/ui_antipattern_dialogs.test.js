// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end (jsdom) tests for anti-pattern fixes A1/A2 in media/main.js:
// every native window.confirm / window.prompt (dirty-tab close, overwrite
// on move/rename, Find in Folder, Delete, Create Branch, Create Tag,
// Compare with) is replaced by an in-webview notification with
// verb-labelled buttons or a text input.  The harness makes window.prompt
// and window.confirm throw, so each test fails on the old code.
'use strict';

const assert = require('assert');
const h = require('./ui_antipattern_harness');

const {test, report} = h.makeRunner();

const FILE_ENTRIES = [
  {name: 'foo.py', path: h.WD + '/foo.py', isDir: false},
  {name: 'src', path: h.WD + '/src', isDir: true},
];

async function main() {
  await test('Delete asks in-webview with Delete / Keep, safe button focused', () => {
    const {win, posted} = h.makeWebview();
    h.openExplorer(win, posted, FILE_ENTRIES);
    h.runMenuItem(win, h.explorerRow(win, h.WD + '/foo.py'), 'Delete');
    const t = h.toast(win, 'fs-delete');
    assert.ok(t, 'the Delete question is a notification, not a native dialog');
    assert.strictEqual(
      t.querySelector('.kiss-notification-message').textContent,
      "Delete 'foo.py'? The server has no trash, so it cannot be restored.",
    );
    const del = h.toastButton(t, 'Delete');
    const keep = h.toastButton(t, 'Keep');
    assert.ok(del && keep, 'buttons are labelled with the verbs');
    assert.ok(
      del.classList.contains('kiss-notification-action-danger'),
      'the destructive button is marked danger',
    );
    assert.strictEqual(win.document.activeElement, keep, 'Keep has focus');
    assert.strictEqual(
      h.ofType(posted, 'fsAction').length,
      0,
      'nothing is deleted before the answer',
    );
    h.click(win, keep);
    assert.strictEqual(h.toast(win, 'fs-delete'), null, 'Keep closes it');
    assert.strictEqual(h.ofType(posted, 'fsAction').length, 0);

    h.runMenuItem(win, h.explorerRow(win, h.WD + '/foo.py'), 'Delete');
    h.click(win, h.toastButton(h.toast(win, 'fs-delete'), 'Delete'));
    const actions = h.ofType(posted, 'fsAction');
    assert.strictEqual(actions.length, 1, 'Delete sends the fsAction');
    assert.strictEqual(actions[0].action, 'delete');
    assert.strictEqual(actions[0].path, h.WD + '/foo.py');
    assert.strictEqual(h.toast(win, 'fs-delete'), null);
    win.close();
  });

  await test('Escape on the Delete question cancels it', () => {
    const {win, posted} = h.makeWebview();
    h.openExplorer(win, posted, FILE_ENTRIES);
    h.runMenuItem(win, h.explorerRow(win, h.WD + '/foo.py'), 'Delete');
    const t = h.toast(win, 'fs-delete');
    h.key(win, win.document.activeElement, 'Escape');
    assert.strictEqual(h.toast(win, 'fs-delete'), null, 'Escape closed it');
    assert.strictEqual(h.ofType(posted, 'fsAction').length, 0);
    assert.ok(t, 'the toast existed before Escape');
    win.close();
  });

  await test('Find in Folder is a text prompt: Enter searches, empty stays open', () => {
    const {win, posted} = h.makeWebview();
    h.openExplorer(win, posted, FILE_ENTRIES);
    h.runMenuItem(win, h.explorerRow(win, h.WD + '/src'), 'Find in Folder...');
    const t = h.toast(win, 'find-in-folder');
    assert.ok(t, 'the prompt is a notification');
    assert.strictEqual(
      t.querySelector('.kiss-notification-message').textContent,
      "Find in 'src'",
    );
    const input = h.toastInput(t);
    assert.ok(input, 'it carries a text input');
    assert.strictEqual(win.document.activeElement, input, 'input focused');
    assert.strictEqual(input.getAttribute('aria-label'), "Find in 'src'");
    assert.ok(h.toastButton(t, 'Find'), 'submit is labelled Find');
    // Enter with nothing typed: refused, the box stays.
    h.key(win, input, 'Enter');
    assert.ok(h.toast(win, 'find-in-folder'), 'empty query keeps the box');
    assert.strictEqual(h.ofType(posted, 'fsAction').length, 0);
    input.value = 'TODO';
    h.key(win, input, 'Enter');
    const actions = h.ofType(posted, 'fsAction');
    assert.strictEqual(actions.length, 1, 'Enter submits');
    assert.strictEqual(actions[0].action, 'findInFolder');
    assert.strictEqual(actions[0].query, 'TODO');
    assert.strictEqual(actions[0].path, h.WD + '/src');
    assert.strictEqual(h.toast(win, 'find-in-folder'), null, 'submit closes');
    win.close();
  });

  await test('Escape in the prompt input cancels without searching', () => {
    const {win, posted} = h.makeWebview();
    h.openExplorer(win, posted, FILE_ENTRIES);
    h.runMenuItem(win, h.explorerRow(win, h.WD + '/src'), 'Find in Folder...');
    const input = h.toastInput(h.toast(win, 'find-in-folder'));
    input.value = 'abc';
    h.key(win, input, 'Escape');
    assert.strictEqual(h.toast(win, 'find-in-folder'), null);
    assert.strictEqual(h.ofType(posted, 'fsAction').length, 0);
    win.close();
  });

  await test('overwrite on rename asks Replace / Keep existing and resends with overwrite', () => {
    const {win, posted} = h.makeWebview();
    h.openExplorer(win, posted, FILE_ENTRIES);
    const row = h.explorerRow(win, h.WD + '/foo.py');
    h.runMenuItem(win, row, 'Rename...');
    const box = win.document.querySelector('.explorer-input');
    assert.ok(box, 'the inline rename box opens');
    box.value = 'bar.py';
    h.key(win, box, 'Enter');
    let actions = h.ofType(posted, 'fsAction');
    assert.strictEqual(actions.length, 1);
    assert.strictEqual(actions[0].action, 'rename');
    assert.strictEqual(actions[0].overwrite, false);
    h.send(win, {
      type: 'fsResult',
      token: actions[0].token,
      error: 'bar.py exists',
      exists: true,
    });
    const t = h.toast(win, 'fs-overwrite:' + actions[0].dest + '|' + actions[0].dest.split('/').pop());
    assert.ok(t, 'the exists reply asks in-webview');
    assert.strictEqual(
      t.querySelector('.kiss-notification-message').textContent,
      "A file or folder named 'bar.py' already exists in the destination " +
        'folder. Replace it?',
    );
    const keep = h.toastButton(t, 'Keep existing');
    assert.strictEqual(win.document.activeElement, keep, 'safe button focused');
    h.click(win, h.toastButton(t, 'Replace'));
    actions = h.ofType(posted, 'fsAction');
    assert.strictEqual(actions.length, 2, 'Replace resends the rename');
    assert.strictEqual(actions[1].action, 'rename');
    assert.strictEqual(actions[1].overwrite, true);
    assert.strictEqual(actions[1].dest, h.WD + '/bar.py');
    win.close();
  });

  await test('Keep existing on the overwrite question sends nothing', () => {
    const {win, posted} = h.makeWebview();
    h.openExplorer(win, posted, FILE_ENTRIES);
    h.runMenuItem(win, h.explorerRow(win, h.WD + '/foo.py'), 'Rename...');
    const box = win.document.querySelector('.explorer-input');
    box.value = 'bar.py';
    h.key(win, box, 'Enter');
    const first = h.ofType(posted, 'fsAction')[0];
    h.send(win, {
      type: 'fsResult',
      token: first.token,
      error: 'x',
      exists: true,
    });
    const id = 'fs-overwrite:' + first.dest + '|' + first.dest.split('/').pop();
    h.click(win, h.toastButton(h.toast(win, id), 'Keep existing'));
    assert.strictEqual(h.toast(win, id), null);
    assert.strictEqual(h.ofType(posted, 'fsAction').length, 1);
    win.close();
  });

  await test('Create Branch prompts for a name; Create branch submits it', () => {
    const {win, posted} = h.makeWebview();
    const row = h.openScm(win, posted);
    assert.ok(row, 'the commit row renders');
    h.runMenuItem(win, row, 'Create Branch...');
    const t = h.toast(win, 'git-create-branch');
    assert.ok(t, 'the branch name prompt is a notification');
    assert.strictEqual(
      t.querySelector('.kiss-notification-message').textContent,
      'New branch from ' + h.SHA_A.slice(0, 7),
    );
    const input = h.toastInput(t);
    assert.strictEqual(input.placeholder, 'Branch name');
    // A blank name is refused and the typed whitespace kept.
    input.value = '   ';
    h.click(win, h.toastButton(t, 'Create branch'));
    assert.ok(h.toast(win, 'git-create-branch'), 'blank name keeps the box');
    assert.strictEqual(h.ofType(posted, 'gitAction').length, 0);
    input.value = ' feature/x ';
    h.click(win, h.toastButton(t, 'Create branch'));
    const acts = h.ofType(posted, 'gitAction');
    assert.strictEqual(acts.length, 1);
    assert.strictEqual(acts[0].action, 'createBranch');
    assert.strictEqual(acts[0].name, 'feature/x');
    assert.strictEqual(acts[0].sha, h.SHA_A);
    assert.strictEqual(h.toast(win, 'git-create-branch'), null);
    win.close();
  });

  await test('Create Tag: name, then optional message; empty message = lightweight tag', () => {
    const {win, posted} = h.makeWebview();
    const row = h.openScm(win, posted);
    h.runMenuItem(win, row, 'Create Tag...');
    const t1 = h.toast(win, 'git-create-tag');
    assert.ok(t1, 'step 1 asks for the tag name');
    const name = h.toastInput(t1);
    name.value = 'v9';
    h.key(win, name, 'Enter');
    assert.strictEqual(h.toast(win, 'git-create-tag'), null, 'step 1 closed');
    const t2 = h.toast(win, 'git-create-tag-message');
    assert.ok(t2, 'step 2 asks for the message');
    assert.strictEqual(
      t2.querySelector('.kiss-notification-message').textContent,
      "Message for tag 'v9' (leave empty for a lightweight tag)",
    );
    const msg = h.toastInput(t2);
    assert.strictEqual(
      win.document.activeElement,
      msg,
      'focus moves into the second prompt, not back to the opener',
    );
    assert.strictEqual(h.ofType(posted, 'gitAction').length, 0, 'not yet');
    h.click(win, h.toastButton(t2, 'Create tag'));
    const acts = h.ofType(posted, 'gitAction');
    assert.strictEqual(acts.length, 1);
    assert.strictEqual(acts[0].action, 'createTag');
    assert.strictEqual(acts[0].name, 'v9');
    assert.strictEqual(acts[0].message, '', 'lightweight tag');
    win.close();
  });

  await test('Create Tag: an annotation message is passed; Cancel creates no tag', () => {
    const {win, posted} = h.makeWebview();
    const row = h.openScm(win, posted);
    h.runMenuItem(win, row, 'Create Tag...');
    const name = h.toastInput(h.toast(win, 'git-create-tag'));
    name.value = 'v10';
    h.key(win, name, 'Enter');
    const t2 = h.toast(win, 'git-create-tag-message');
    h.toastInput(t2).value = ' release ten ';
    h.key(win, h.toastInput(t2), 'Enter');
    const acts = h.ofType(posted, 'gitAction');
    assert.strictEqual(acts.length, 1);
    assert.strictEqual(acts[0].message, 'release ten');

    h.runMenuItem(win, row, 'Create Tag...');
    const again = h.toastInput(h.toast(win, 'git-create-tag'));
    again.value = 'v11';
    h.key(win, again, 'Enter');
    h.click(
      win,
      h.toastButton(h.toast(win, 'git-create-tag-message'), 'Cancel'),
    );
    assert.strictEqual(h.toast(win, 'git-create-tag-message'), null);
    assert.strictEqual(h.ofType(posted, 'gitAction').length, 1, 'no tag');
    win.close();
  });

  await test('Compare with prompts with HEAD prefilled and sends gitShow', () => {
    const {win, posted} = h.makeWebview();
    const row = h.openScm(win, posted);
    h.runMenuItem(win, row, 'Compare with...');
    const t = h.toast(win, 'git-compare-with');
    assert.ok(t, 'the prompt is a notification');
    const input = h.toastInput(t);
    assert.strictEqual(input.value, 'HEAD', 'HEAD is prefilled');
    assert.strictEqual(win.document.activeElement, input);
    input.value = 'main~3';
    h.click(win, h.toastButton(t, 'Compare'));
    const shows = h.ofType(posted, 'gitShow');
    assert.strictEqual(shows.length, 1);
    assert.strictEqual(shows[0].sha, h.SHA_A);
    assert.strictEqual(shows[0].base, 'main~3');
    win.close();
  });

  await test("closing a dirty tab offers Save and close / Don't save / Keep editing", async () => {
    const ctx = h.makeWebview();
    const {win} = ctx;
    await h.openDirtyContentTab(ctx);
    const strip = win.document.querySelector('.chat-tab.content-tab');
    assert.ok(strip, 'the content tab is in the tab bar');
    h.click(win, strip.querySelector('.chat-tab-close'));
    assert.strictEqual(h.contentTabStrips(win).length, 1, 'still open');
    const t = win.document.querySelector(
      '[data-notification-id^="close-dirty-"]',
    );
    assert.ok(t, 'the question is a notification');
    assert.strictEqual(
      t.querySelector('.kiss-notification-message').textContent,
      "'notes.txt' has unsaved changes.",
    );
    const save = h.toastButton(t, 'Save and close');
    const dont = h.toastButton(t, "Don't save");
    const keep = h.toastButton(t, 'Keep editing');
    assert.ok(save && dont && keep, 'three verb-labelled choices');
    assert.ok(dont.classList.contains('kiss-notification-action-danger'));
    assert.strictEqual(
      win.document.activeElement,
      keep,
      'Keep editing focused',
    );
    h.click(win, keep);
    assert.strictEqual(h.contentTabStrips(win).length, 1, 'kept open');
    assert.strictEqual(
      win.document.querySelector('[data-notification-id^="close-dirty-"]'),
      null,
    );
    win.close();
  });

  await test("Don't save closes the dirty tab without saving", async () => {
    const ctx = h.makeWebview();
    const {win, posted} = ctx;
    await h.openDirtyContentTab(ctx);
    const strip = win.document.querySelector('.chat-tab.content-tab');
    h.click(win, strip.querySelector('.chat-tab-close'));
    const t = win.document.querySelector(
      '[data-notification-id^="close-dirty-"]',
    );
    h.click(win, h.toastButton(t, "Don't save"));
    assert.strictEqual(h.contentTabStrips(win).length, 0, 'closed');
    assert.strictEqual(h.ofType(posted, 'saveFile').length, 0, 'not saved');
    win.close();
  });

  await test('Save and close saves, then closes on the fileSaved reply', async () => {
    const ctx = h.makeWebview();
    const {win, posted} = ctx;
    await h.openDirtyContentTab(ctx);
    const strip = win.document.querySelector('.chat-tab.content-tab');
    h.click(win, strip.querySelector('.chat-tab-close'));
    const t = win.document.querySelector(
      '[data-notification-id^="close-dirty-"]',
    );
    h.click(win, h.toastButton(t, 'Save and close'));
    const saves = h.ofType(posted, 'saveFile');
    assert.strictEqual(saves.length, 1, 'one saveFile went out');
    assert.strictEqual(saves[0].content, 'v1 edited');
    assert.strictEqual(
      h.contentTabStrips(win).length,
      1,
      'open until the reply',
    );
    h.send(win, {
      type: 'fileSaved',
      token: saves[0].token,
      ok: true,
      version: 'ver-2',
    });
    assert.strictEqual(
      h.contentTabStrips(win).length,
      0,
      'closed after the save',
    );
    win.close();
  });

  await test('Save and close keeps the tab (and edits) when the save fails', async () => {
    const ctx = h.makeWebview();
    const {win, posted, created} = ctx;
    await h.openDirtyContentTab(ctx);
    const strip = win.document.querySelector('.chat-tab.content-tab');
    h.click(win, strip.querySelector('.chat-tab-close'));
    const t = win.document.querySelector(
      '[data-notification-id^="close-dirty-"]',
    );
    h.click(win, h.toastButton(t, 'Save and close'));
    const saves = h.ofType(posted, 'saveFile');
    h.send(win, {
      type: 'fileSaved',
      token: saves[0].token,
      ok: false,
      error: 'read-only',
    });
    assert.strictEqual(h.contentTabStrips(win).length, 1, 'tab stays');
    assert.strictEqual(created[0].editor.getModel().getValue(), 'v1 edited');
    assert.ok(
      win.document
        .querySelector('.content-save-status')
        .textContent.includes('read-only'),
      'the failure is shown in the Save bar',
    );
    // A later plain Save must not close the tab: the close request died
    // with the failed save.
    h.click(win, win.document.querySelector('.content-save-btn'));
    const saves2 = h.ofType(posted, 'saveFile');
    assert.strictEqual(saves2.length, 2);
    h.send(win, {
      type: 'fileSaved',
      token: saves2[1].token,
      ok: true,
      version: 'v3',
    });
    assert.strictEqual(
      h.contentTabStrips(win).length,
      1,
      'a plain save keeps the tab',
    );
    win.close();
  });

  await test('Escape on the unsaved-changes question keeps editing', async () => {
    const ctx = h.makeWebview();
    const {win} = ctx;
    await h.openDirtyContentTab(ctx);
    const strip = win.document.querySelector('.chat-tab.content-tab');
    h.click(win, strip.querySelector('.chat-tab-close'));
    const t = win.document.querySelector(
      '[data-notification-id^="close-dirty-"]',
    );
    assert.ok(t);
    h.key(win, win.document.activeElement, 'Escape');
    assert.strictEqual(
      win.document.querySelector('[data-notification-id^="close-dirty-"]'),
      null,
      'Escape dismissed the question',
    );
    assert.strictEqual(h.contentTabStrips(win).length, 1, 'the tab stays open');
    win.close();
  });

  report('ui_antipattern_dialogs');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
