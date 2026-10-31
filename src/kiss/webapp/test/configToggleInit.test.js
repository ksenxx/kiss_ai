// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) tests for the "Use web tools" and "Use persistent
// memory" settings toggles: they must be INITIALIZED from the server's
// ``configData`` (keys ``use_web_browser`` / ``use_memory``) instead of
// the hardcoded ``checked`` state shipped in chat.html, and their state
// must be PERSISTED back through ``saveConfig`` when the settings panel
// closes.  The "Classify tasks before running" and "Use Jev" boxes were
// removed from the Settings UI: the form must neither show nor save
// ``classify_tasks`` / ``classify_with_decisions`` (the daemon keeps
// its stored values).  The memory directory (config key ``memory_dir``)
// has no settings field either: the panel never shows it and
// ``saveConfig`` never carries it.

'use strict';

const assert = require('assert');
const {makePage, send, click, byId, lastMsg} = require('./simplify2_harness');
const {
  openSettings,
  closeSettings,
  typeInto,
} = require('./settings_helpers');

function typeAndSend(win, text) {
  typeInto(win, 'task-input', text);
  click(win, byId(win, 'send-btn'));
}

function testTogglesInitializedFalseFromConfigData() {
  const {win, posted} = makePage();
  send(win, {
    type: 'configData',
    config: {use_web_browser: false, use_memory: false},
    apiKeys: {},
  });

  const web = win.document.getElementById('cfg-use-web-tools');
  const mem = win.document.getElementById('cfg-use-memory');
  assert.strictEqual(
    web.checked,
    false,
    'configData {use_web_browser:false} must uncheck #cfg-use-web-tools ' +
      '(was left at the hardcoded checked state from chat.html)',
  );
  assert.strictEqual(
    mem.checked,
    false,
    'configData {use_memory:false} must uncheck #cfg-use-memory',
  );

  // The initialized state must flow into the actual submit command.
  typeAndSend(win, 'do something');
  const run = lastMsg(posted, 'submit');
  assert.ok(run, 'clicking send must post a submit command');
  // The worktree and commit decisions belong to the agent's own
  // create_worktree / commit_and_merge tools: no per-run flags.
  assert.ok(!('useWorktree' in run), 'submit must not carry useWorktree');
  assert.ok(!('autoCommit' in run), 'submit must not carry autoCommit');
  assert.strictEqual(
    run.useWebTools,
    false,
    'submit must carry useWebTools:false after configData turned the toggle off',
  );
  win.close();
  console.log(
    '  ok - configData false values initialize toggles and submit flags',
  );
}

function testTogglesInitializedTrueFromConfigData() {
  const {win} = makePage();
  // Start from the opposite state so a no-op would be caught.
  win.document.getElementById('cfg-use-web-tools').checked = false;
  win.document.getElementById('cfg-use-memory').checked = false;

  send(win, {
    type: 'configData',
    config: {use_web_browser: true, use_memory: true},
    apiKeys: {},
  });

  assert.strictEqual(
    win.document.getElementById('cfg-use-web-tools').checked,
    true,
    'configData {use_web_browser:true} must check #cfg-use-web-tools',
  );
  assert.strictEqual(
    win.document.getElementById('cfg-use-memory').checked,
    true,
    'configData {use_memory:true} must check #cfg-use-memory',
  );
  win.close();
  console.log('  ok - configData true values re-check the toggles');
}

function testMissingKeysDefaultToChecked() {
  const {win} = makePage();
  win.document.getElementById('cfg-use-web-tools').checked = false;
  win.document.getElementById('cfg-use-memory').checked = false;

  // Older servers / partial configs omit the keys: default is true,
  // matching user_config.DEFAULTS.
  send(win, {type: 'configData', config: {}, apiKeys: {}});

  assert.strictEqual(
    win.document.getElementById('cfg-use-web-tools').checked,
    true,
    'missing use_web_browser must default #cfg-use-web-tools to checked',
  );
  assert.strictEqual(
    win.document.getElementById('cfg-use-memory').checked,
    true,
    'missing use_memory must default #cfg-use-memory to checked',
  );
  win.close();
  console.log('  ok - missing config keys default the toggles');
}

function testSubmitBeforeConfigDataOmitsWebTools() {
  const {win, posted} = makePage();

  // Submit BEFORE any configData reply: the checkbox still holds
  // chat.html's hardcoded checked state, which is NOT the user's
  // persisted setting.  The submit must omit useWebTools so the daemon
  // falls back to the persisted use_web_browser config instead of an
  // invented per-run override.
  typeAndSend(win, 'run before config arrives');
  const early = lastMsg(posted, 'submit');
  assert.ok(early, 'clicking send must post a submit command');
  assert.strictEqual(
    'useWebTools' in early,
    false,
    'a submit before configData must omit useWebTools (an invented ' +
      'override would defeat the daemon config fallback)',
  );

  // A user toggle IS a known state, even without configData.
  const web = win.document.getElementById('cfg-use-web-tools');
  web.checked = false;
  web.dispatchEvent(new win.Event('change', {bubbles: true}));
  typeAndSend(win, 'run after an explicit toggle');
  const afterToggle = lastMsg(posted, 'submit');
  assert.strictEqual(
    afterToggle.useWebTools,
    false,
    'a submit after the user toggled the checkbox must carry its state',
  );
  win.close();
  console.log('  ok - submit omits useWebTools until the state is known');
}

function testToggleStatePersistedOnSettingsClose() {
  const {win, posted} = makePage();
  // The user opens the sheet; the stored config (the reply to the
  // sheet's getConfig) lands in the form before they touch it.
  openSettings(win);
  send(win, {
    type: 'configData',
    config: {
      use_web_browser: true,
      use_memory: true,
      memory_dir: '~/notes/memories',
      classify_tasks: true,
      classify_with_decisions: true,
    },
    apiKeys: {},
  });

  // The user turns the toggles off, then closes the settings panel.
  const web = win.document.getElementById('cfg-use-web-tools');
  const mem = win.document.getElementById('cfg-use-memory');
  web.checked = false;
  web.dispatchEvent(new win.Event('change', {bubbles: true}));
  mem.checked = false;
  mem.dispatchEvent(new win.Event('change', {bubbles: true}));

  closeSettings(win);

  const save = lastMsg(posted, 'saveConfig');
  assert.ok(save, 'closing the settings panel must post saveConfig');
  // (copied: the posted object belongs to the jsdom realm, whose
  // Object.prototype deepStrictEqual would reject)
  assert.deepStrictEqual(
    {...save.config},
    {use_web_browser: false, use_memory: false},
    'saveConfig carries exactly the two edited toggles: the daemon merges ' +
      'the payload, so the stored memory_dir / classify_* values (no ' +
      'Settings field) and the retired auto_commit_mode / is_worktree stay ' +
      'out of it',
  );

  // Round-trip: the server echoes the saved config back; a fresh
  // populate must land on the persisted (unchecked) state.
  web.checked = true;
  mem.checked = true;
  send(win, {
    type: 'configData',
    config: {use_web_browser: false, use_memory: false, memory_dir: '~/notes/memories'},
    apiKeys: {},
  });
  assert.strictEqual(web.checked, false, 'echoed configData must re-apply');
  assert.strictEqual(mem.checked, false, 'echoed configData must re-apply');
  win.close();
  console.log('  ok - settings close persists toggle state via saveConfig');
}

function testMemoryToggleEditSurvivesLateReplyAndFlushesAlone() {
  const {win, posted} = makePage();
  // Open the panel: no configData has arrived yet, the user unticks
  // "Use persistent memory" right away.
  openSettings(win);
  const mem = win.document.getElementById('cfg-use-memory');
  mem.checked = false;
  mem.dispatchEvent(new win.Event('change', {bubbles: true}));

  // The sheet-open getConfig reply lands late, with the stored (true)
  // value: an edited box must NOT be repainted.
  send(win, {
    type: 'configData',
    config: {use_memory: true, use_web_browser: true},
    apiKeys: {},
  });
  assert.strictEqual(
    mem.checked,
    false,
    "a late configData must not overwrite the user's unticked #cfg-use-memory",
  );
  // An untouched sibling still follows the reply.
  assert.strictEqual(
    win.document.getElementById('cfg-use-web-tools').checked,
    true,
    'the untouched #cfg-use-web-tools must still follow configData',
  );

  closeSettings(win);
  const save = lastMsg(posted, 'saveConfig');
  assert.ok(save, 'closing the settings panel must post saveConfig');
  assert.strictEqual(
    save.config.use_memory,
    false,
    "saveConfig must carry the user's use_memory:false",
  );
  // The daemon merges the payload: the untouched sibling is left out,
  // so its fetched value cannot paint over another window's save.
  assert.ok(
    !('use_web_browser' in save.config),
    'the untouched use_web_browser box is not re-sent',
  );
  win.close();
  console.log('  ok - an edited memory box survives a late reply and saves');
}

/**
 * No configData ever arrived.  The user
 * makes the given edits (`budget` edits the max-budget box, `memory`
 * unticks the memory checkbox) and closes the panel: the partial
 * payload must carry exactly those fields -- sending the untouched
 * checkboxes too would flush chat.html's hardcoded checked state over
 * whatever values are stored on the server, and a checkbox edit alone
 * must be enough to trigger the save.
 */
function testPartialSaveWhenPanelClosedBeforeConfigData({budget, memory}) {
  const {win, posted} = makePage();
  openSettings(win);
  const expected = {};
  if (budget) {
    const box = win.document.getElementById('cfg-max-budget');
    box.value = '77';
    box.dispatchEvent(new win.Event('input', {bubbles: true}));
    expected.max_budget = 77;
  }
  if (memory) {
    const mem = win.document.getElementById('cfg-use-memory');
    mem.checked = false;
    mem.dispatchEvent(new win.Event('change', {bubbles: true}));
    expected.use_memory = false;
  }

  closeSettings(win);

  const label = JSON.stringify({budget, memory});
  const save = lastMsg(posted, 'saveConfig');
  assert.ok(save, `closing the panel after an edit must post saveConfig ${label}`);
  // (copied: the posted object belongs to the jsdom realm, whose
  // Object.prototype deepStrictEqual would reject)
  assert.deepStrictEqual(
    {...save.config},
    expected,
    'a panel closed before configData must save ONLY the edited keys ' +
      '(the daemon merges the payload; an invented value would clobber ' +
      `the stored toggle) and memory_dir is never part of the payload ${label}`,
  );
  win.close();
  console.log(`  ok - a panel closed before configData saves only edits ${label}`);
}

function main() {
  testTogglesInitializedFalseFromConfigData();
  testTogglesInitializedTrueFromConfigData();
  testMissingKeysDefaultToChecked();
  testSubmitBeforeConfigDataOmitsWebTools();
  testToggleStatePersistedOnSettingsClose();
  testMemoryToggleEditSurvivesLateReplyAndFlushesAlone();
  testPartialSaveWhenPanelClosedBeforeConfigData({budget: true, memory: false});
  testPartialSaveWhenPanelClosedBeforeConfigData({budget: false, memory: true});
  testPartialSaveWhenPanelClosedBeforeConfigData({budget: true, memory: true});
  console.log('configToggleInit.test.js: all tests passed');
}

main();
