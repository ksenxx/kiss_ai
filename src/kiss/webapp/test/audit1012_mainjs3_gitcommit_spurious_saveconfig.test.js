// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Audit 2026-10-12 (mainjs-3): the Git Commit button of the "..."
// menu (triggerManualGitCommit) closes the settings sheet so the
// commit's transcript line is not hidden behind it, and closing the
// sheet flushes the config form as a full `saveConfig` (API keys
// included).  A Git Commit click with the sheet closed used to post
// such a saveConfig nobody asked for, and a stale form could overwrite
// a change another window had saved.
//
// Expected: a Git Commit click with the sheet closed posts
// autocommitAction and nothing else config-related.  Two things hold
// that rule: triggerManualGitCommit closes the sheet only when it is
// open, and saveSettingsIfPopulated flushes the form only after the
// user edited a field, which every close forgets.  Closing a closed
// sheet has no other effect to observe, so this test pins the rule,
// not the first guard alone.

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
