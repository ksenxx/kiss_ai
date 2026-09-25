// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) tests for the "Use persistent memory" settings
// toggle (#cfg-use-memory, config key ``use_memory``): it must be
// INITIALIZED from the server's ``configData`` instead of the hardcoded
// state shipped in chat.html, PERSISTED back through ``saveConfig``
// when the settings panel closes, and a toggle the user has flipped
// must never be repainted by a late ``configData`` poll.  The memory
// directory (config key ``memory_dir``) has no settings field: the
// panel never shows it and ``saveConfig`` never carries it.

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

function testMemoryInitializedFromConfigData() {
  const {win} = makeWebview();
  send(win, {
    type: 'configData',
    config: {use_memory: false, memory_dir: '/data/agent-memories'},
    apiKeys: {},
  });

  assert.strictEqual(
    win.document.getElementById('cfg-use-memory').checked,
    false,
    'configData {use_memory:false} must uncheck #cfg-use-memory ' +
      '(was left at the hardcoded checked state from chat.html)',
  );
  assert.strictEqual(
    win.document.getElementById('cfg-memory-dir'),
    null,
    'the memory directory is not a settings field any more: memory_dir ' +
      'stays a config.json key without a box in the panel',
  );
  win.close();
  console.log('  ok - configData initializes the memory toggle');
}

function testMemoryInitializedTrueFromConfigData() {
  const {win} = makeWebview();
  // Start from the opposite state so a no-op would be caught.
  win.document.getElementById('cfg-use-memory').checked = false;

  send(win, {
    type: 'configData',
    config: {use_memory: true, memory_dir: ''},
    apiKeys: {},
  });

  assert.strictEqual(
    win.document.getElementById('cfg-use-memory').checked,
    true,
    'configData {use_memory:true} must check #cfg-use-memory',
  );
  win.close();
  console.log('  ok - configData true value re-applies to the toggle');
}

function testMissingKeysDefaultToOn() {
  const {win} = makeWebview();
  win.document.getElementById('cfg-use-memory').checked = false;

  // Older servers / partial configs omit the keys: default is true,
  // matching vscode_config.DEFAULTS (memory is on by default).
  send(win, {type: 'configData', config: {}, apiKeys: {}});

  assert.strictEqual(
    win.document.getElementById('cfg-use-memory').checked,
    true,
    'missing use_memory must default #cfg-use-memory to checked',
  );
  win.close();
  console.log('  ok - missing config keys default the memory toggle to on');
}

function testMemoryStatePersistedOnSettingsClose() {
  const {win, posted} = makeWebview();
  // populateConfigForm sets configFormPopulated = true, arming the
  // settings-close flush.
  send(win, {
    type: 'configData',
    config: {use_memory: true, memory_dir: '~/notes/memories'},
    apiKeys: {},
  });

  // The user turns memory off, then closes the settings panel.
  const toggle = win.document.getElementById('cfg-use-memory');
  toggle.checked = false;
  toggle.dispatchEvent(new win.Event('change', {bubbles: true}));

  win.document
    .getElementById('settings-panel-close')
    .dispatchEvent(new win.MouseEvent('click', {bubbles: true}));

  const save = lastMsg(posted, 'saveConfig');
  assert.ok(save, 'closing the settings panel must post saveConfig');
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
  // populate must land on the persisted state.  The panel close
  // cleared settingsEditedFields, so the repaint is allowed again.
  toggle.checked = true;
  send(win, {
    type: 'configData',
    config: {use_memory: false, memory_dir: '~/notes/memories'},
    apiKeys: {},
  });
  assert.strictEqual(toggle.checked, false, 'echoed configData must re-apply');
  win.close();
  console.log('  ok - settings close persists memory state via saveConfig');
}

function testEditedToggleNotRepaintedByPoll() {
  const {win} = makeWebview();
  send(win, {
    type: 'configData',
    config: {use_memory: true},
    apiKeys: {},
  });

  // The user flipped the toggle; the 2-second configData poll re-pushes
  // the stored value and must NOT clobber the edit (settingsEditedFields
  // guard).
  const toggle = win.document.getElementById('cfg-use-memory');
  toggle.checked = false;
  toggle.dispatchEvent(new win.Event('change', {bubbles: true}));

  send(win, {
    type: 'configData',
    config: {use_memory: true},
    apiKeys: {},
  });

  assert.strictEqual(
    toggle.checked,
    false,
    'a configData poll must not repaint #cfg-use-memory mid-edit',
  );
  win.close();
  console.log('  ok - configData poll never clobbers an in-progress edit');
}

function testPartialSaveWhenPanelClosedBeforeConfigData() {
  const {win, posted} = makeWebview();

  // No configData ever arrived (configFormPopulated is false).  The
  // user edits ONLY the budget and closes the panel: the partial
  // payload must carry just that field — sending the untouched
  // checkbox too would flush chat.html's hardcoded checked state over
  // whatever use_memory value is stored on the server.
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
  assert.strictEqual(
    'use_memory' in save.config,
    false,
    'an untouched #cfg-use-memory must be omitted from a partial save ' +
      '(the daemon merges the payload; an invented value would clobber ' +
      'the stored toggle)',
  );
  assert.strictEqual(
    'memory_dir' in save.config,
    false,
    'and memory_dir is never part of the payload',
  );
  win.close();
  console.log('  ok - a panel closed before configData saves only edits');
}

function main() {
  testMemoryInitializedFromConfigData();
  testMemoryInitializedTrueFromConfigData();
  testMissingKeysDefaultToOn();
  testMemoryStatePersistedOnSettingsClose();
  testEditedToggleNotRepaintedByPoll();
  testPartialSaveWhenPanelClosedBeforeConfigData();
  console.log('configMemoryToggle.test.js: all tests passed');
}

main();
