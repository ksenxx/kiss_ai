// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) tests for the "Auto commit" / "Use worktree"
// settings toggles: they must be INITIALIZED from the server's
// ``configData`` (keys ``auto_commit_mode`` / ``is_worktree``) instead
// of the hardcoded ``checked`` state shipped in chat.html, and their
// state must be PERSISTED back through ``saveConfig`` when the
// settings panel closes.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

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

  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));

  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function lastMsg(posted, type) {
  for (let i = posted.length - 1; i >= 0; i -= 1) {
    if (posted[i] && posted[i].type === type) return posted[i];
  }
  return null;
}

function typeAndSend(win, text) {
  const inp = win.document.getElementById('task-input');
  const sendBtn = win.document.getElementById('send-btn');
  inp.value = text;
  inp.dispatchEvent(new win.Event('input', {bubbles: true}));
  sendBtn.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

function testTogglesInitializedFalseFromConfigData() {
  const {win, posted} = makeWebview();
  send(win, {
    type: 'configData',
    config: {auto_commit_mode: false, is_worktree: false},
    apiKeys: {},
  });

  const ac = win.document.getElementById('cfg-auto-commit');
  const wt = win.document.getElementById('cfg-use-worktree');
  assert.strictEqual(
    ac.checked,
    false,
    'configData {auto_commit_mode:false} must uncheck #cfg-auto-commit ' +
      '(was left at the hardcoded checked state from chat.html)',
  );
  assert.strictEqual(
    wt.checked,
    false,
    'configData {is_worktree:false} must uncheck #cfg-use-worktree ' +
      '(was left at the hardcoded checked state from chat.html)',
  );

  // The initialized state must flow into the actual submit command.
  typeAndSend(win, 'do something');
  const run = lastMsg(posted, 'submit');
  assert.ok(run, 'clicking send must post a submit command');
  assert.strictEqual(
    run.autoCommit,
    false,
    'submit must carry autoCommit:false after configData turned the toggle off',
  );
  assert.strictEqual(
    run.useWorktree,
    false,
    'submit must carry useWorktree:false after configData turned the toggle off',
  );
  win.close();
  console.log('  ok - configData false values initialize toggles and submit flags');
}

function testTogglesInitializedTrueFromConfigData() {
  const {win} = makeWebview();
  // Start from the opposite state so a no-op would be caught.
  win.document.getElementById('cfg-auto-commit').checked = false;
  win.document.getElementById('cfg-use-worktree').checked = false;

  send(win, {
    type: 'configData',
    config: {auto_commit_mode: true, is_worktree: true},
    apiKeys: {},
  });

  assert.strictEqual(
    win.document.getElementById('cfg-auto-commit').checked,
    true,
    'configData {auto_commit_mode:true} must check #cfg-auto-commit',
  );
  assert.strictEqual(
    win.document.getElementById('cfg-use-worktree').checked,
    true,
    'configData {is_worktree:true} must check #cfg-use-worktree',
  );
  win.close();
  console.log('  ok - configData true values re-check the toggles');
}

function testMissingKeysDefaultToChecked() {
  const {win} = makeWebview();
  win.document.getElementById('cfg-auto-commit').checked = false;
  win.document.getElementById('cfg-use-worktree').checked = false;

  // Older servers / partial configs omit the keys: default is true,
  // matching vscode_config.DEFAULTS.
  send(win, {type: 'configData', config: {}, apiKeys: {}});

  assert.strictEqual(
    win.document.getElementById('cfg-auto-commit').checked,
    true,
    'missing auto_commit_mode must default #cfg-auto-commit to checked',
  );
  assert.strictEqual(
    win.document.getElementById('cfg-use-worktree').checked,
    true,
    'missing is_worktree must default #cfg-use-worktree to checked',
  );
  win.close();
  console.log('  ok - missing config keys default the toggles to checked');
}

function testToggleStatePersistedOnSettingsClose() {
  const {win, posted} = makeWebview();
  // populateConfigForm sets configFormPopulated = true, arming the
  // settings-close flush.
  send(win, {
    type: 'configData',
    config: {auto_commit_mode: true, is_worktree: true},
    apiKeys: {},
  });

  // The user turns both toggles off, then closes the settings panel.
  const ac = win.document.getElementById('cfg-auto-commit');
  const wt = win.document.getElementById('cfg-use-worktree');
  ac.checked = false;
  ac.dispatchEvent(new win.Event('change', {bubbles: true}));
  wt.checked = false;
  wt.dispatchEvent(new win.Event('change', {bubbles: true}));

  win.document
    .getElementById('settings-panel-close')
    .dispatchEvent(new win.MouseEvent('click', {bubbles: true}));

  const save = lastMsg(posted, 'saveConfig');
  assert.ok(save, 'closing the settings panel must post saveConfig');
  assert.strictEqual(
    save.config.auto_commit_mode,
    false,
    'saveConfig must persist auto_commit_mode from #cfg-auto-commit',
  );
  assert.strictEqual(
    save.config.is_worktree,
    false,
    'saveConfig must persist is_worktree from #cfg-use-worktree',
  );

  // Round-trip: the server echoes the saved config back; a fresh
  // populate must land on the persisted (unchecked) state.
  ac.checked = true;
  wt.checked = true;
  send(win, {
    type: 'configData',
    config: {auto_commit_mode: false, is_worktree: false},
    apiKeys: {},
  });
  assert.strictEqual(ac.checked, false, 'echoed configData must re-apply');
  assert.strictEqual(wt.checked, false, 'echoed configData must re-apply');
  win.close();
  console.log('  ok - settings close persists toggle state via saveConfig');
}

function main() {
  testTogglesInitializedFalseFromConfigData();
  testTogglesInitializedTrueFromConfigData();
  testMissingKeysDefaultToChecked();
  testToggleStatePersistedOnSettingsClose();
  console.log('configToggleInit.test.js: all tests passed');
}

main();
