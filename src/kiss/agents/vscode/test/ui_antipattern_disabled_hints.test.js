// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end (jsdom) tests for anti-patterns A4/A9 in media/main.js:
// a disabled control says why (promptlet Add, working-directory Open,
// Git Commit while a commit runs), a Max budget the form will ignore is
// flagged under the field, Update / Update Models progress is visible
// inside the settings sheet, and a failed share is a sticky error toast
// with the reason.  Each test fails on the old code.
'use strict';

const assert = require('assert');
const h = require('./ui_antipattern_harness');

const {test, report} = h.makeRunner();

function typeInto(win, el, value) {
  el.value = value;
  el.dispatchEvent(new win.Event('input', {bubbles: true}));
}

async function main() {
  await test('the disabled promptlet Add button explains itself', () => {
    const {win} = h.makeWebview();
    h.click(win, h.byId(win, 'tricks-btn'));
    const btn = h.byId(win, 'tricks-add-btn');
    assert.ok(btn.disabled);
    assert.strictEqual(btn.title, 'Type a name and prompt to add a promptlet');
    typeInto(win, h.byId(win, 'tricks-add-input'), 'x');
    assert.ok(!btn.disabled);
    assert.ok(!btn.hasAttribute('title'), 'no stale reason once enabled');
    win.close();
  });

  await test('the disabled working-directory Open button explains itself', () => {
    const {win} = h.makeWebview();
    h.click(win, h.byId(win, 'workdir-btn'));
    const btn = h.byId(win, 'workdir-open-btn');
    assert.ok(btn.disabled);
    assert.strictEqual(btn.title, 'Pick or type a folder first');
    typeInto(win, h.byId(win, 'workdir-input'), '/tmp/x');
    assert.ok(!btn.disabled);
    assert.ok(!btn.hasAttribute('title'));
    win.close();
  });

  await test('Git Commit reads "Committing…" with a reason while disabled', () => {
    const {win, posted} = h.makeWebview();
    win._testApi.endLaunch();
    const btn = h.byId(win, 'autocommit-btn');
    const label = btn.querySelector('.more-item-label');
    h.click(win, btn);
    assert.strictEqual(h.ofType(posted, 'autocommitAction').length, 1);
    assert.ok(btn.disabled);
    assert.strictEqual(label.textContent, 'Committing…');
    assert.strictEqual(btn.title, 'A commit is in progress');
    assert.strictEqual(btn.dataset.tooltip, 'A commit is in progress');
    h.send(win, {
      type: 'autocommit_done',
      tabId: win._testApi.getActiveTabId(),
      success: true,
      manual: true,
    });
    assert.ok(!btn.disabled);
    assert.strictEqual(label.textContent, 'Git Commit');
    assert.ok(!btn.hasAttribute('title'));
    assert.strictEqual(btn.dataset.tooltip, 'git commit');
    win.close();
  });

  await test('an ignored Max budget is flagged under the field, not dropped silently', () => {
    const {win, posted} = h.makeWebview();
    h.click(win, h.byId(win, 'settings-btn'));
    h.send(win, {type: 'configData', config: {max_budget: 250}, apiKeys: {}});
    const box = h.byId(win, 'cfg-max-budget');
    assert.strictEqual(box.value, '250');
    // Clearing the box (what a type=number input shows for any
    // non-numeric text too) is the value the form cannot use.
    typeInto(win, box, '');
    const note = h.byId(win, 'cfg-max-budget-note');
    assert.ok(note && !note.hidden, 'a note appears under the field');
    assert.strictEqual(note.textContent, 'Empty: the saved budget is kept.');
    assert.strictEqual(
      box.getAttribute('aria-describedby'),
      'cfg-max-budget-note',
    );
    assert.ok(!win.document.querySelector('.kiss-notification'), 'no toast');
    typeInto(win, box, '300');
    assert.ok(note.hidden, 'a parseable value clears the note');
    h.click(win, h.byId(win, 'settings-panel-close'));
    const saved = h.ofType(posted, 'saveConfig').pop();
    assert.strictEqual(saved.config.max_budget, 300);
    win.close();
  });

  await test('Update / Update Models progress shows inside the settings sheet', () => {
    const {win, posted} = h.makeWebview();
    h.click(win, h.byId(win, 'settings-btn'));
    h.click(win, h.byId(win, 'cfg-update-models-btn'));
    assert.strictEqual(h.ofType(posted, 'updateModels').length, 1);
    const status = h.byId(win, 'settings-update-status');
    assert.ok(status && !status.hidden, 'a status line appears in the sheet');
    assert.strictEqual(status.textContent, 'Updating the model catalog…');
    assert.strictEqual(status.getAttribute('role'), 'status');
    h.send(win, {
      type: 'notice',
      text: 'Model catalog update complete (output: /x/update_models.log).',
    });
    assert.strictEqual(
      status.textContent,
      'Model catalog update complete (output: /x/update_models.log).',
      "the daemon's answer lands in the sheet",
    );
    h.click(win, h.byId(win, 'cfg-update-btn'));
    assert.strictEqual(status.textContent, 'Updating…');
    h.send(win, {
      type: 'error',
      text: 'Failed to start the update: no installer',
    });
    assert.strictEqual(
      status.textContent,
      'Failed to start the update: no installer',
    );
    assert.ok(status.classList.contains('config-field-note-error'));
    h.click(win, h.byId(win, 'settings-panel-close'));
    assert.ok(status.hidden, 'closing the sheet clears the line');
    win.close();
  });

  await test('a failed share is a sticky error toast with the reason', () => {
    const {win} = h.makeWebview();
    h.send(win, {type: 'share_done', ok: false, error: 'disk full'});
    const t = h.toast(win, 'share-failed');
    assert.ok(t, 'a toast, not only a 2 s button flash');
    assert.strictEqual(
      t.querySelector('.kiss-notification-message').textContent,
      'Share failed: disk full',
    );
    assert.strictEqual(t.dataset.notificationSticky, 'true');
    assert.ok(
      h.byId(win, 'share-btn').classList.contains('share-err'),
      'the flash is kept',
    );
    win.close();
  });

  report('ui_antipattern_disabled_hints');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
