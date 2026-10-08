// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end (jsdom) tests for anti-patterns A3/A8 in media/main.js:
// user text is never thrown away silently.  The promptlet Add box keeps
// its text until the daemon confirms and shows a rejection inside the
// sheet; closing the settings sheet mid custom-model edit asks before
// discarding typed changes.  Each test fails on the old code.
'use strict';

const assert = require('assert');
const h = require('./ui_antipattern_harness');

const {test, report} = h.makeRunner();

function typeInto(win, el, value) {
  el.value = value;
  el.dispatchEvent(new win.Event('input', {bubbles: true}));
}

async function main() {
  await test('promptlet Add keeps the text until tricksData confirms; a rejection shows in the sheet', () => {
    const {win, posted} = h.makeWebview();
    win.__TRICKS__ = [];
    h.click(win, h.byId(win, 'tricks-btn'));
    const box = h.byId(win, 'tricks-add-input');
    typeInto(win, box, 'new promptlet');
    h.click(win, h.byId(win, 'tricks-add-btn'));
    assert.strictEqual(h.ofType(posted, 'addTrick').length, 1, 'posted once');
    assert.strictEqual(
      box.value,
      'new promptlet',
      'the box is not cleared yet',
    );

    // The daemon rejects it (error, then the unchanged list for the sender).
    h.send(win, {type: 'error', text: 'Duplicate promptlet'});
    h.send(win, {type: 'tricksData', tricks: [], userCount: 0});
    const err = h.byId(win, 'tricks-add-error');
    assert.ok(err && !err.hidden, 'the rejection is shown under the box');
    assert.strictEqual(err.textContent, 'Duplicate promptlet');
    assert.strictEqual(err.getAttribute('role'), 'alert');
    assert.strictEqual(box.value, 'new promptlet', 'the text survives');

    typeInto(win, box, 'new promptlet 2');
    assert.ok(err.hidden, 'editing the text clears the message');
    h.click(win, h.byId(win, 'tricks-add-btn'));
    h.send(win, {
      type: 'tricksData',
      tricks: ['new promptlet 2'],
      userCount: 1,
    });
    assert.strictEqual(box.value, '', 'the confirming list clears the box');
    assert.ok(h.byId(win, 'tricks-add-btn').disabled);
    win.close();
  });

  await test("a foreign tricksData does not clear a pending add it doesn't list", () => {
    const {win} = h.makeWebview();
    win.__TRICKS__ = [];
    h.click(win, h.byId(win, 'tricks-btn'));
    const box = h.byId(win, 'tricks-add-input');
    typeInto(win, box, 'pending');
    h.click(win, h.byId(win, 'tricks-add-btn'));
    h.send(win, {type: 'tricksData', tricks: ['someone else'], userCount: 1});
    assert.strictEqual(box.value, 'pending');
    win.close();
  });

  await test('closing settings mid custom-model edit asks: Discard changes / Keep editing', () => {
    const {win} = h.makeWebview();
    h.click(win, h.byId(win, 'settings-btn'));
    h.send(win, {type: 'configData', config: {}, apiKeys: {}});
    h.send(win, {
      type: 'myModelsData',
      models: [
        {name: 'model-b', endpoint: 'http://b/v1', api_key: '', headers: ''},
      ],
    });
    h.click(win, win.document.querySelector('.custom-model-edit-btn'));
    typeInto(win, h.byId(win, 'cfg-custom-endpoint'), 'http://b-new/v1');
    h.click(win, h.byId(win, 'settings-panel-close'));
    const panel = h.byId(win, 'settings-panel');
    assert.ok(panel.classList.contains('open'), 'the sheet stays open');
    const t = h.toast(win, 'settings-discard-model-edit');
    assert.ok(t, 'the question is asked');
    assert.strictEqual(
      t.querySelector('.kiss-notification-message').textContent,
      'The custom model "model-b" has unsaved changes.',
    );
    const discard = h.toastButton(t, 'Discard changes');
    const keep = h.toastButton(t, 'Keep editing');
    assert.ok(discard && keep, 'verb-labelled buttons');
    assert.strictEqual(
      win.document.activeElement,
      keep,
      'the safe button has focus',
    );
    h.click(win, keep);
    assert.ok(panel.classList.contains('open'));
    assert.strictEqual(
      h.byId(win, 'custom-model-save-btn').style.display,
      '',
      'the edit is still in progress',
    );
    assert.strictEqual(
      h.byId(win, 'cfg-custom-endpoint').value,
      'http://b-new/v1',
    );

    h.click(win, h.byId(win, 'settings-panel-close'));
    h.click(
      win,
      h.toastButton(
        h.toast(win, 'settings-discard-model-edit'),
        'Discard changes',
      ),
    );
    assert.ok(!panel.classList.contains('open'), 'Discard closes the sheet');
    assert.strictEqual(
      h.byId(win, 'custom-model-save-btn').style.display,
      'none',
    );
    win.close();
  });

  await test('closing settings with an untouched edit needs no question', () => {
    const {win} = h.makeWebview();
    h.click(win, h.byId(win, 'settings-btn'));
    h.send(win, {type: 'configData', config: {}, apiKeys: {}});
    h.send(win, {
      type: 'myModelsData',
      models: [{name: 'model-b', endpoint: '', api_key: '', headers: ''}],
    });
    h.click(win, win.document.querySelector('.custom-model-edit-btn'));
    h.click(win, h.byId(win, 'settings-panel-close'));
    assert.ok(!h.byId(win, 'settings-panel').classList.contains('open'));
    assert.ok(!h.toast(win, 'settings-discard-model-edit'));
    win.close();
  });

  report('ui_antipattern_drafts');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
