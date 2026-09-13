// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) tests for the "Use persistent memory" settings
// toggle (#cfg-use-memory, config key ``use_memory``) and the "Memory
// directory" field (#cfg-memory-dir, config key ``memory_dir``): they
// must be INITIALIZED from the server's ``configData`` instead of the
// hardcoded state shipped in chat.html, PERSISTED back through
// ``saveConfig`` when the settings panel closes, and a field the user
// is editing must never be repainted by a late ``configData`` poll.

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
    win.document.getElementById('cfg-memory-dir').value,
    '/data/agent-memories',
    'configData memory_dir must populate #cfg-memory-dir',
  );
  win.close();
  console.log('  ok - configData initializes the memory toggle and directory');
}

function testMemoryInitializedTrueFromConfigData() {
  const {win} = makeWebview();
  // Start from the opposite state so a no-op would be caught.
  win.document.getElementById('cfg-use-memory').checked = false;
  win.document.getElementById('cfg-memory-dir').value = '/stale';

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
  assert.strictEqual(
    win.document.getElementById('cfg-memory-dir').value,
    '',
    'configData {memory_dir:""} must clear #cfg-memory-dir ' +
      '(empty means the ~/.kiss/memories default)',
  );
  win.close();
  console.log('  ok - configData true/empty values re-apply to the fields');
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
  assert.strictEqual(
    win.document.getElementById('cfg-memory-dir').value,
    '',
    'missing memory_dir must leave #cfg-memory-dir empty',
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
    config: {use_memory: true, memory_dir: ''},
    apiKeys: {},
  });

  // The user turns memory off and points it at a custom directory,
  // then closes the settings panel.
  const toggle = win.document.getElementById('cfg-use-memory');
  toggle.checked = false;
  toggle.dispatchEvent(new win.Event('change', {bubbles: true}));
  const dir = win.document.getElementById('cfg-memory-dir');
  dir.value = '  ~/notes/memories  ';
  dir.dispatchEvent(new win.Event('input', {bubbles: true}));

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
    save.config.memory_dir,
    '~/notes/memories',
    'saveConfig must persist the trimmed memory_dir from #cfg-memory-dir',
  );

  // Round-trip: the server echoes the saved config back; a fresh
  // populate must land on the persisted state.  The panel close
  // cleared settingsEditedFields, so the repaint is allowed again.
  toggle.checked = true;
  dir.value = '';
  send(win, {
    type: 'configData',
    config: {use_memory: false, memory_dir: '~/notes/memories'},
    apiKeys: {},
  });
  assert.strictEqual(toggle.checked, false, 'echoed configData must re-apply');
  assert.strictEqual(
    dir.value,
    '~/notes/memories',
    'echoed configData must re-apply the directory',
  );
  win.close();
  console.log('  ok - settings close persists memory state via saveConfig');
}

function testEditedDirNotRepaintedByPoll() {
  const {win} = makeWebview();
  send(win, {
    type: 'configData',
    config: {use_memory: true, memory_dir: '/old'},
    apiKeys: {},
  });

  // The user is typing a new directory; the 2-second configData poll
  // re-pushes the stored value and must NOT clobber the in-progress
  // edit (settingsEditedFields guard).
  const dir = win.document.getElementById('cfg-memory-dir');
  dir.value = '/half-typed/new-pl';
  dir.dispatchEvent(new win.Event('input', {bubbles: true}));

  send(win, {
    type: 'configData',
    config: {use_memory: true, memory_dir: '/old'},
    apiKeys: {},
  });

  assert.strictEqual(
    dir.value,
    '/half-typed/new-pl',
    'a configData poll must not repaint #cfg-memory-dir mid-edit',
  );
  win.close();
  console.log('  ok - configData poll never clobbers an in-progress edit');
}

function testPartialSaveWhenPanelClosedBeforeConfigData() {
  const {win, posted} = makeWebview();

  // No configData ever arrived (configFormPopulated is false).  The
  // user edits ONLY the memory directory and closes the panel: the
  // partial payload must carry just that field — sending the untouched
  // checkbox too would flush chat.html's hardcoded checked state over
  // whatever use_memory value is stored on the server.
  const dir = win.document.getElementById('cfg-memory-dir');
  dir.value = '/only/this/field';
  dir.dispatchEvent(new win.Event('input', {bubbles: true}));

  win.document
    .getElementById('settings-panel-close')
    .dispatchEvent(new win.MouseEvent('click', {bubbles: true}));

  const save = lastMsg(posted, 'saveConfig');
  assert.ok(save, 'closing the settings panel must post saveConfig');
  assert.strictEqual(
    save.config.memory_dir,
    '/only/this/field',
    'the edited memory_dir must be saved even before any configData',
  );
  assert.strictEqual(
    'use_memory' in save.config,
    false,
    'an untouched #cfg-use-memory must be omitted from a partial save ' +
      '(the daemon merges the payload; an invented value would clobber ' +
      'the stored toggle)',
  );
  win.close();
  console.log('  ok - a panel closed before configData saves only edits');
}

function main() {
  testMemoryInitializedFromConfigData();
  testMemoryInitializedTrueFromConfigData();
  testMissingKeysDefaultToOn();
  testMemoryStatePersistedOnSettingsClose();
  testEditedDirNotRepaintedByPoll();
  testPartialSaveWhenPanelClosedBeforeConfigData();
  console.log('configMemoryToggle.test.js: all tests passed');
}

main();
