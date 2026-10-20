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
const {makePage, send} = require('./simplify2_harness');
const {lastMsg} = require('./settings_helpers');

function typeAndSend(win, text) {
  const inp = win.document.getElementById('task-input');
  const sendBtn = win.document.getElementById('send-btn');
  inp.value = text;
  inp.dispatchEvent(new win.Event('input', {bubbles: true}));
  sendBtn.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

function testClassifyTogglesAreGone() {
  const {win} = makePage();
  assert.strictEqual(
    win.document.getElementById('cfg-classify-tasks'),
    null,
    'the Settings UI must not show "Classify tasks before running"',
  );
  assert.strictEqual(
    win.document.getElementById('cfg-classify-with-decisions'),
    null,
    'the Settings UI must not show "Use Jev (decisions model)"',
  );
  assert.strictEqual(
    win.document.getElementById('cfg-memory-dir'),
    null,
    'the memory directory is not a settings field: memory_dir stays a ' +
      'config.json key without a box in the panel',
  );
  win.close();
  console.log('  ok - the classify toggles and memory dir are not in the Settings UI');
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
  // populateConfigForm sets configFormPopulated = true, arming the
  // settings-close flush.
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

  win.document
    .getElementById('settings-panel-close')
    .dispatchEvent(new win.MouseEvent('click', {bubbles: true}));

  const save = lastMsg(posted, 'saveConfig');
  assert.ok(save, 'closing the settings panel must post saveConfig');
  assert.ok(
    !('auto_commit_mode' in save.config) && !('is_worktree' in save.config),
    'saveConfig must not carry the retired auto_commit_mode / is_worktree',
  );
  assert.ok(
    !('classify_tasks' in save.config) &&
      !('classify_with_decisions' in save.config),
    'saveConfig must not carry classify_tasks / classify_with_decisions ' +
      '(no longer in the Settings UI; the daemon keeps its stored values)',
  );
  assert.strictEqual(
    save.config.use_web_browser,
    false,
    'saveConfig must persist use_web_browser from #cfg-use-web-tools',
  );
  assert.strictEqual(
    save.config.use_memory,
    false,
    'saveConfig must persist use_memory from #cfg-use-memory',
  );
  assert.strictEqual(
    'memory_dir' in save.config,
    false,
    'a full-form save must not carry memory_dir: the daemon merges the ' +
      'payload, and a blank invented here would wipe the stored directory',
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

function testMemoryToggleEditSurvivesPollAndFlushesAlone() {
  const {win, posted} = makePage();
  // Open the panel: no configData has arrived yet (configFormPopulated
  // stays false), the user unticks "Use persistent memory" right away.
  win.document
    .getElementById('settings-btn')
    .dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  const mem = win.document.getElementById('cfg-use-memory');
  mem.checked = false;
  mem.dispatchEvent(new win.Event('change', {bubbles: true}));

  // The 2-second configData poll now lands with the stored (true)
  // value: an edited box must NOT be repainted.
  send(win, {
    type: 'configData',
    config: {use_memory: true, use_web_browser: true},
    apiKeys: {},
  });
  assert.strictEqual(
    mem.checked,
    false,
    "a configData poll must not overwrite the user's unticked #cfg-use-memory",
  );
  // An untouched sibling still follows the poll.
  assert.strictEqual(
    win.document.getElementById('cfg-use-web-tools').checked,
    true,
    'the untouched #cfg-use-web-tools must still follow configData',
  );

  win.document
    .getElementById('settings-panel-close')
    .dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  const save = lastMsg(posted, 'saveConfig');
  assert.ok(save, 'closing the settings panel must post saveConfig');
  assert.strictEqual(
    save.config.use_memory,
    false,
    "saveConfig must carry the user's use_memory:false",
  );
  assert.strictEqual(
    save.config.use_web_browser,
    true,
    'the populated form saves use_web_browser alongside',
  );
  win.close();
  console.log('  ok - an edited memory box survives the poll and saves');
}

function testMemoryToggleFlushedAloneWithoutConfigData() {
  const {win, posted} = makePage();
  // Panel opened and closed before ANY configData reply: only the
  // edited field may be sent (the daemon merges), never a form full of
  // chat.html's hardcoded defaults.
  win.document
    .getElementById('settings-btn')
    .dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  const mem = win.document.getElementById('cfg-use-memory');
  mem.checked = false;
  mem.dispatchEvent(new win.Event('change', {bubbles: true}));
  win.document
    .getElementById('settings-panel-close')
    .dispatchEvent(new win.MouseEvent('click', {bubbles: true}));

  const save = lastMsg(posted, 'saveConfig');
  assert.ok(save, 'closing the panel after an edit must post saveConfig');
  assert.deepStrictEqual(
    Object.keys(save.config),
    ['use_memory'],
    'a panel closed before configData must save ONLY the edited key',
  );
  assert.strictEqual(save.config.use_memory, false);
  win.close();
  console.log('  ok - the memory edit alone is flushed before configData');
}

function testPartialSaveWhenPanelClosedBeforeConfigData() {
  const {win, posted} = makePage();

  // No configData ever arrived (configFormPopulated is false).  The
  // user edits ONLY the budget and closes the panel: the partial
  // payload must carry just that field -- sending the untouched
  // checkboxes too would flush chat.html's hardcoded checked state over
  // whatever values are stored on the server.
  const budget = win.document.getElementById('cfg-max-budget');
  budget.value = '77';
  budget.dispatchEvent(new win.Event('input', {bubbles: true}));

  win.document
    .getElementById('settings-panel-close')
    .dispatchEvent(new win.MouseEvent('click', {bubbles: true}));

  const save = lastMsg(posted, 'saveConfig');
  assert.ok(save, 'closing the settings panel must post saveConfig');
  assert.strictEqual(
    save.config.max_budget,
    77,
    'the edited max_budget must be saved even before any configData',
  );
  assert.deepStrictEqual(
    Object.keys(save.config),
    ['max_budget'],
    'an untouched checkbox must be omitted from a partial save (the ' +
      'daemon merges the payload; an invented value would clobber the ' +
      'stored toggle) and memory_dir is never part of the payload',
  );
  win.close();
  console.log('  ok - a panel closed before configData saves only edits');
}

function main() {
  testClassifyTogglesAreGone();
  testTogglesInitializedFalseFromConfigData();
  testTogglesInitializedTrueFromConfigData();
  testMissingKeysDefaultToChecked();
  testSubmitBeforeConfigDataOmitsWebTools();
  testToggleStatePersistedOnSettingsClose();
  testMemoryToggleEditSurvivesPollAndFlushesAlone();
  testMemoryToggleFlushedAloneWithoutConfigData();
  testPartialSaveWhenPanelClosedBeforeConfigData();
  console.log('configToggleInit.test.js: all tests passed');
}

main();
