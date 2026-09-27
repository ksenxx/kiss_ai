// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end (jsdom) tests for anti-patterns A10/A14 in media/main.js:
// destructive one-click actions get a second step.  The worktree and
// main-tree "Discard" buttons sit last in their bar and swap the row
// for an inline question (naming the branch / folder) with a Keep
// button; promptlet and custom-model deletes get the same inline
// Delete / Cancel pair frequent tasks already had; the server-reset
// confirm focuses Cancel so Enter never restarts the server.  Each test
// fails on the old code.
'use strict';

const assert = require('assert');
const h = require('./ui_antipattern_harness');

const {test, report} = h.makeRunner();

function barButtons(win) {
  return h.all(win, '.wt-bar .wt-btns .wt-btn');
}

function labels(btns) {
  return btns.map(b => b.textContent);
}

/** Posted messages are jsdom-realm objects: compare them as plain JSON. */
function plain(msgs) {
  return JSON.parse(JSON.stringify(msgs));
}

function barButton(win, text) {
  const btn = barButtons(win).find(b => b.textContent === text);
  assert.ok(btn, `the bar must have a "${text}" button`);
  return btn;
}

async function main() {
  await test('server-reset confirm focuses Cancel, not the destructive button', () => {
    const {win, posted} = h.makeWebview();
    const tabId = win._testApi.getActiveTabId();
    h.send(win, {type: 'status', running: true, tabId, startTs: Date.now()});
    h.click(win, h.byId(win, 'settings-btn'));
    h.click(win, h.byId(win, 'cfg-server-reset-btn'));
    assert.strictEqual(
      win.document.activeElement,
      h.byId(win, 'server-reset-confirm-cancel'),
      'the safe button holds focus',
    );
    assert.strictEqual(h.ofType(posted, 'serverReset').length, 0);
    win.close();
  });

  await test('worktree Discard is last and asks before posting, naming the branch', () => {
    const {win, posted} = h.makeWebview();
    const tabId = win._testApi.getActiveTabId();
    h.send(win, {type: 'worktree_done', tabId, branch: 'kiss_wt-42'});
    assert.deepStrictEqual(labels(barButtons(win)), [
      'Auto-commit and merge',
      'Do nothing',
      'Discard',
    ]);
    h.click(win, barButton(win, 'Discard'));
    assert.strictEqual(
      h.ofType(posted, 'worktreeAction').length,
      0,
      'the first click posts nothing',
    );
    const ask = win.document.querySelector('.wt-bar .wt-confirm');
    assert.ok(ask && !ask.hidden, 'the inline question is shown');
    assert.strictEqual(
      ask.querySelector('.wt-confirm-text').textContent,
      "Delete branch 'kiss_wt-42' and all of the task's changes?",
    );
    assert.ok(
      win.document.querySelector('.wt-bar .wt-btns').hidden,
      'the row of choices is swapped out meanwhile',
    );
    const keep = ask.querySelector('.wt-confirm-no');
    assert.strictEqual(keep.textContent, 'Keep');
    assert.strictEqual(win.document.activeElement, keep, 'Keep holds focus');
    h.click(win, keep);
    assert.ok(ask.hidden, 'Keep puts the row back');
    assert.ok(!win.document.querySelector('.wt-bar .wt-btns').hidden);
    assert.strictEqual(h.ofType(posted, 'worktreeAction').length, 0);

    h.click(win, barButton(win, 'Discard'));
    h.click(win, ask.querySelector('.wt-confirm-yes'));
    assert.deepStrictEqual(plain(h.ofType(posted, 'worktreeAction')), [
      {type: 'worktreeAction', action: 'discard', tabId},
    ]);
    assert.ok(
      barButtons(win).every(b => b.disabled),
      'the bar is disarmed while the discard is in flight',
    );
    win.close();
  });

  await test('main-tree Discard is last and asks before posting', () => {
    const {win, posted} = h.makeWebview();
    const tabId = win._testApi.getActiveTabId();
    h.send(win, {type: 'main_tree_done', tabId, workDir: '/ws/repo'});
    assert.deepStrictEqual(labels(barButtons(win)), [
      'Auto commit',
      'Do nothing',
      'Discard',
    ]);
    h.click(win, barButton(win, 'Discard'));
    assert.strictEqual(h.ofType(posted, 'mainTreeAction').length, 0);
    const ask = win.document.querySelector('.wt-bar .wt-confirm');
    assert.strictEqual(
      ask.querySelector('.wt-confirm-text').textContent,
      "Throw away the task's uncommitted changes in /ws/repo?",
    );
    h.click(win, ask.querySelector('.wt-confirm-yes'));
    assert.deepStrictEqual(plain(h.ofType(posted, 'mainTreeAction')), [
      {type: 'mainTreeAction', action: 'discard', tabId, workDir: '/ws/repo'},
    ]);
    win.close();
  });

  await test('promptlet delete needs the inline Delete confirm', () => {
    const {win, posted} = h.makeWebview();
    win.__TRICKS__ = ['mine', 'theirs'];
    win.__MY_TRICKS_COUNT__ = 1;
    h.click(win, h.byId(win, 'tricks-btn'));
    const row = win.document.querySelector('#tricks-list .tricks-item');
    const trash = row.querySelector('.sidebar-item-delete');
    assert.ok(trash, 'the user-owned row has a delete button');
    h.click(win, trash);
    assert.strictEqual(h.ofType(posted, 'deleteTrick').length, 0);
    assert.strictEqual(trash.style.display, 'none', 'the icon gives way');
    const confirmWrap = row.querySelector('.sidebar-item-confirm');
    assert.notStrictEqual(confirmWrap.style.display, 'none');
    h.click(win, confirmWrap.querySelector('.sidebar-confirm-no'));
    assert.strictEqual(h.ofType(posted, 'deleteTrick').length, 0);
    assert.strictEqual(trash.style.display, '', 'Cancel restores the icon');
    h.click(win, trash);
    h.click(win, confirmWrap.querySelector('.sidebar-confirm-yes'));
    assert.deepStrictEqual(plain(h.ofType(posted, 'deleteTrick')), [
      {type: 'deleteTrick', text: 'mine'},
    ]);
    win.close();
  });

  await test('custom-model delete needs the inline Delete confirm', () => {
    const {win, posted} = h.makeWebview();
    h.click(win, h.byId(win, 'settings-btn'));
    h.send(win, {
      type: 'myModelsData',
      models: [{name: 'model-b', endpoint: '', api_key: '', headers: ''}],
    });
    const row = win.document.querySelector('.custom-model-row');
    const trash = row.querySelector('.custom-model-delete-btn');
    assert.strictEqual(trash.getAttribute('aria-label'), 'Delete model-b');
    h.click(win, trash);
    assert.strictEqual(h.ofType(posted, 'deleteMyModel').length, 0);
    const confirmWrap = row.querySelector('.sidebar-item-confirm');
    assert.notStrictEqual(confirmWrap.style.display, 'none');
    h.click(win, confirmWrap.querySelector('.sidebar-confirm-yes'));
    assert.deepStrictEqual(plain(h.ofType(posted, 'deleteMyModel')), [
      {type: 'deleteMyModel', name: 'model-b'},
    ]);
    win.close();
  });

  report('ui_antipattern_destructive_confirm');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
