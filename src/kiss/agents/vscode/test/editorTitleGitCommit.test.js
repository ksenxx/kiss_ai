// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) tests for the host's `gitCommit` message — the
// editor-title git-commit button of editor-tabs mode (media/main.js):
//  - the message runs the same manual-commit flow as the settings
//    drawer's Git Commit button: an `autocommitAction` for the root
//    chat tab with its work dir, and the drawer button disables while
//    the commit is in flight;
//  - an open settings drawer is closed so the transcript's
//    autocommit_progress / autocommit_done lines stay visible;
//  - a second `gitCommit` while one is in flight is dropped (the
//    daemon silently discards duplicates, so re-sending would just
//    look dead);
//  - the terminal `autocommit_done` re-arms the flow: the next
//    `gitCommit` commits again;
//  - the settings drawer's own button still works through the shared
//    trigger after the refactor.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

function makeWebview(bodyAttrs) {
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace('{{BODY_CLASS_ATTR}}', bodyAttrs);
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

  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(
    fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8') +
      '\n//# sourceURL=git-commit-main.js',
  );

  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function byType(posted, type) {
  return posted.filter(m => m.type === type);
}

const ROOT = 'root-tab-0001';

const {win, posted} = makeWebview(
  ' class="editor-tab-mode"' +
    ` data-kiss-tab-id="${ROOT}"` +
    ' data-kiss-tab-title="My chat"',
);

const autocommitBtn = win.document.getElementById('autocommit-btn');
assert.ok(autocommitBtn, 'the settings drawer has the Git Commit button');

// --- gitCommit message commits the root chat tab -------------------------
// Open the settings drawer first: the commit must close it so the
// transcript's progress lines are not hidden behind the sheet.
send(win, {type: 'openSettings'});
const settingsPanel = win.document.getElementById('settings-panel');
assert.ok(
  settingsPanel.classList.contains('open'),
  'precondition: settings drawer open',
);

send(win, {type: 'gitCommit'});
let actions = byType(posted, 'autocommitAction');
assert.strictEqual(actions.length, 1, 'one autocommitAction posted');
assert.strictEqual(actions[0].tabId, ROOT, 'commits the root chat tab');
assert.strictEqual(
  typeof actions[0].workDir,
  'string',
  'carries the work dir field',
);
assert.ok(
  !settingsPanel.classList.contains('open'),
  'the settings drawer closes so progress lines stay visible',
);
assert.strictEqual(
  autocommitBtn.disabled,
  true,
  'the drawer button disables while the commit is in flight',
);

// --- a duplicate while in flight is dropped -------------------------------
send(win, {type: 'gitCommit'});
actions = byType(posted, 'autocommitAction');
assert.strictEqual(
  actions.length,
  1,
  'a second gitCommit while in flight is a no-op',
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

send(win, {type: 'gitCommit'});
actions = byType(posted, 'autocommitAction');
assert.strictEqual(actions.length, 2, 'a re-armed gitCommit commits again');
assert.strictEqual(actions[1].tabId, ROOT);

// --- the settings drawer's own button shares the trigger -------------------
send(win, {type: 'autocommit_done', tabId: ROOT, success: true, manual: true});
send(win, {type: 'openSettings'});
autocommitBtn.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
actions = byType(posted, 'autocommitAction');
assert.strictEqual(actions.length, 3, 'the drawer button still commits');
assert.strictEqual(actions[2].tabId, ROOT);
assert.ok(
  !settingsPanel.classList.contains('open'),
  'the drawer button still closes the drawer',
);
assert.strictEqual(autocommitBtn.disabled, true, 'and disables in flight');

console.log('editorTitleGitCommit: all tests passed');
