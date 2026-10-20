// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) tests for the manual Git Commit flow of the remote
// webapp page (media/main.js), started from the "..." menu's Git Commit
// button:
//  - a click posts an `autocommitAction` for the active chat tab with
//    its work dir, and the button disables while the commit is in
//    flight;
//  - an open settings sheet is closed so the transcript's
//    autocommit_done line stays visible;
//  - a second click while one is in flight is dropped (the daemon
//    silently discards duplicates, so re-sending would just look dead);
//  - the terminal `autocommit_done` re-arms the flow: the next click
//    commits again.

'use strict';

const assert = require('assert');
const {makePage, send, click, ofType} = require('./simplify2_harness');

const {win, posted} = makePage({bodyClass: 'remote-chat'});
const ROOT = win._testApi.getActiveTabId();
assert.ok(ROOT, 'the page opens with an active chat tab');

const autocommitBtn = win.document.getElementById('autocommit-btn');
assert.ok(autocommitBtn, 'the "..." menu has the Git Commit button');
const settingsPanel = win.document.getElementById('settings-panel');

// --- a click commits the active chat tab -----------------------------------
// Open the settings sheet first: the commit must close it so the
// transcript's progress lines are not hidden behind the sheet.
click(win, win.document.getElementById('settings-btn'));
assert.ok(
  settingsPanel.classList.contains('open'),
  'precondition: settings sheet open',
);

click(win, autocommitBtn);
let actions = ofType(posted, 'autocommitAction');
assert.strictEqual(actions.length, 1, 'one autocommitAction posted');
assert.strictEqual(actions[0].tabId, ROOT, 'commits the active chat tab');
assert.strictEqual(
  typeof actions[0].workDir,
  'string',
  'carries the work dir field',
);
assert.ok(
  !settingsPanel.classList.contains('open'),
  'the settings sheet closes so progress lines stay visible',
);
assert.strictEqual(
  autocommitBtn.disabled,
  true,
  'the button disables while the commit is in flight',
);

// --- a duplicate while in flight is dropped -------------------------------
click(win, autocommitBtn);
actions = ofType(posted, 'autocommitAction');
assert.strictEqual(
  actions.length,
  1,
  'a second click while in flight is a no-op',
);

// --- autocommit_done re-arms the flow --------------------------------------
send(win, {
  type: 'autocommit_done',
  tabId: ROOT,
  success: true,
  manual: true,
  message: 'Committed.',
});
assert.strictEqual(
  autocommitBtn.disabled,
  false,
  'autocommit_done re-arms the button',
);

click(win, autocommitBtn);
actions = ofType(posted, 'autocommitAction');
assert.strictEqual(actions.length, 2, 'a re-armed click commits again');
assert.strictEqual(actions[1].tabId, ROOT);
assert.strictEqual(autocommitBtn.disabled, true, 'and disables in flight');

// The last commit is left in flight on purpose, which arms main.js's
// 120 s autocommit re-arm timer: close the window so that timer does
// not keep node alive for two minutes after the assertions are done.
win.close();
console.log('editorTitleGitCommit: all tests passed');
