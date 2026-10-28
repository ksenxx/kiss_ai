// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Audit 2026-10-12 (mainjs-3): the Git Commit button of the "..."
// menu runs triggerManualGitCommit(), which calls closeSettingsPanel()
// unconditionally -- even while the settings sheet is closed.
// closeSettingsPanel() flushes the form through
// saveSettingsIfPopulated(), and the form used to count as populated
// after EVERY configData reply (the daemon re-pushes one every 2 s),
// not only the one the sheet asked for.  So every Git Commit click posted a full
// `saveConfig` (API keys included) nobody asked for, and a stale form
// can overwrite a change another window saved since this window's last
// poll.
//
// Expected: a Git Commit click with the sheet closed posts
// autocommitAction and nothing else config-related.

'use strict';

const assert = require('assert');
const {makePage: makeChatPage, send} = require('./simplify2_harness');

function makePage() {
  return makeChatPage({bodyClass: 'remote-chat'});
}

function main() {
  const {win, posted} = makePage();
  // The routine configData poll lands while the settings sheet is
  // closed (it was never opened in this session).
  send(win, {
    type: 'configData',
    config: {max_budget: 42, use_web_browser: true},
    apiKeys: {OPENAI_API_KEY: 'sk-test'},
  });
  assert.ok(
    !win.document.getElementById('settings-panel').classList.contains('open'),
    'precondition: the settings sheet is closed',
  );
  posted.length = 0;
  const btn = win.document.getElementById('autocommit-btn');
  assert.ok(btn && !btn.disabled, 'precondition: Git Commit is armed');
  btn.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  const commit = posted.find(m => m && m.type === 'autocommitAction');
  assert.ok(commit, 'the Git Commit click posts autocommitAction');
  const save = posted.find(m => m && m.type === 'saveConfig');
  assert.strictEqual(
    save,
    undefined,
    'a Git Commit with the settings sheet closed must not post saveConfig ' +
      '(got ' +
      JSON.stringify(save) +
      ')',
  );
  win.close();
  console.log(
    'audit1012_mainjs3_gitcommit_spurious_saveconfig.test.js: all passed.',
  );
}

try {
  main();
} catch (err) {
  console.error(err);
  process.exit(1);
}
