// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end (jsdom) tests for anti-patterns A12/A13 in media/main.js:
// the Settings and Promptlets sheets close on Escape like every other
// popup and hand focus back to the control that opened them; an inner
// popup (promptlet editor, server-reset confirm) consumes Escape first;
// the promptlet toggle exposes its state through aria-expanded.  Each
// test fails on the old code.  (The frequent-tasks sheet has no opener
// in chat.html, so it cannot be exercised end to end.)
'use strict';

const assert = require('assert');
const h = require('./ui_antipattern_harness');

const {test, report} = h.makeRunner();

function isOpen(win, id) {
  return h.byId(win, id).classList.contains('open');
}

function escape(win) {
  h.key(win, win.document.activeElement || win.document.body, 'Escape');
}

async function main() {
  await test('Escape closes the settings sheet and focuses the "..." button', () => {
    const {win} = h.makeWebview();
    const settingsBtn = h.byId(win, 'settings-btn');
    settingsBtn.focus();
    h.click(win, settingsBtn);
    assert.ok(isOpen(win, 'settings-panel'), 'precondition: sheet open');
    escape(win);
    assert.ok(!isOpen(win, 'settings-panel'), 'Escape closes the sheet');
    assert.ok(!isOpen(win, 'settings-overlay'), 'and its backdrop');
    // #settings-btn lives in the (now closed) "..." menu and cannot
    // take focus there: its trigger gets it.
    assert.strictEqual(win.document.activeElement, h.byId(win, 'more-btn'));
    win.close();
  });

  await test('Escape closes the promptlets sheet and returns focus to its button', () => {
    const {win} = h.makeWebview();
    win.__TRICKS__ = ['do the thing'];
    const tricksBtn = h.byId(win, 'tricks-btn');
    tricksBtn.focus();
    h.click(win, tricksBtn);
    assert.ok(isOpen(win, 'tricks-panel'));
    assert.strictEqual(tricksBtn.getAttribute('aria-expanded'), 'true');
    h.byId(win, 'tricks-search').focus();
    escape(win);
    assert.ok(!isOpen(win, 'tricks-panel'), 'Escape closes the sheet');
    assert.strictEqual(tricksBtn.getAttribute('aria-expanded'), 'false');
    assert.strictEqual(win.document.activeElement, tricksBtn);
    win.close();
  });

  await test('an open promptlet editor consumes Escape; the sheet stays', () => {
    const {win} = h.makeWebview();
    win.__TRICKS__ = ['mine'];
    win.__MY_TRICKS_COUNT__ = 1;
    h.click(win, h.byId(win, 'tricks-btn'));
    h.click(win, win.document.querySelector('#tricks-list .sidebar-item-edit'));
    const editor = win.document.querySelector('#tricks-list textarea');
    assert.ok(editor, 'precondition: the in-place editor is open');
    editor.focus();
    escape(win);
    assert.ok(
      !win.document.querySelector('#tricks-list textarea'),
      'the editor closed',
    );
    assert.ok(isOpen(win, 'tricks-panel'), 'the sheet under it stays open');
    escape(win);
    assert.ok(!isOpen(win, 'tricks-panel'), 'a second Escape closes the sheet');
    win.close();
  });

  await test('the server-reset confirm consumes Escape; settings stays open', () => {
    const {win, posted} = h.makeWebview();
    const tabId = win._testApi.getActiveTabId();
    h.send(win, {type: 'status', running: true, tabId, startTs: Date.now()});
    h.click(win, h.byId(win, 'settings-btn'));
    h.click(win, h.byId(win, 'cfg-server-reset-btn'));
    const modal = 'server-reset-confirm-modal';
    assert.ok(isOpen(win, modal), 'precondition: confirm open');
    escape(win);
    assert.ok(!isOpen(win, modal), 'Escape closed the confirm');
    assert.ok(isOpen(win, 'settings-panel'), 'the sheet under it stays open');
    assert.strictEqual(h.ofType(posted, 'serverReset').length, 0);
    win.close();
  });

  report('ui_antipattern_sheets_escape');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
