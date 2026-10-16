// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) tests for deleting an API key from the settings
// panel.
//
// The daemon's ``save_api_key`` treats an EMPTY value as a delete: the
// ``export`` line is removed from the canonical key store.  For that
// to ever happen the client must actually serialize the empty string --
// and it must do so ONLY for a field the user explicitly cleared.  The
// ``saveConfig`` payload is merged by the daemon, so an untouched empty
// box (a key that was never set, or a form that never got its
// ``configData``) must stay omitted: serializing those would wipe keys
// saved by another window.
//
// These tests pin both sides of that contract by driving the real
// chat.html + api.js + main.js.

'use strict';

const assert = require('assert');
const {makePage, send} = require('./simplify2_harness');

function lastMsg(posted, type) {
  for (let i = posted.length - 1; i >= 0; i -= 1) {
    if (posted[i] && posted[i].type === type) return posted[i];
  }
  return null;
}

function openSettings(win) {
  const gear = win.document.querySelector('#settings-btn');
  assert.ok(gear, 'the tab bar must render the settings gear');
  gear.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

function closeSettings(win) {
  win.document
    .getElementById('settings-panel-close')
    .dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

function typeInto(win, id, value) {
  const node = win.document.getElementById(id);
  assert.ok(node, `#${id} must exist`);
  node.value = value;
  node.dispatchEvent(new win.Event('input', {bubbles: true}));
}

const STORED = {
  max_budget: 100,
  custom_endpoint: '',
  custom_api_key: '',
  custom_headers: '',
  remote_password: '',
};

// A field the user deliberately cleared must reach the daemon as "",
// or the key can never be deleted from the settings UI at all.
function testClearedKeyIsSentAsEmptyString() {
  const {win, posted} = makePage();
  openSettings(win);
  send(win, {
    type: 'configData',
    config: STORED,
    apiKeys: {ANTHROPIC_API_KEY: 'sk-existing', OPENAI_API_KEY: 'sk-keep'},
  });
  typeInto(win, 'cfg-key-ANTHROPIC_API_KEY', '');
  closeSettings(win);

  const msg = lastMsg(posted, 'saveConfig');
  assert.ok(msg, 'closing the panel must save the form');
  assert.strictEqual(
    msg.apiKeys.ANTHROPIC_API_KEY,
    '',
    'a deliberately cleared key must be serialized as "" so the ' +
      'daemon deletes it from the shell RC',
  );
  assert.strictEqual(
    msg.apiKeys.OPENAI_API_KEY,
    'sk-keep',
    'an untouched populated key is re-sent unchanged',
  );
  win.close();
  console.log('  ok - a cleared key field is sent as an empty string');
}

// The daemon merges the payload, so boxes that were empty all along
// (keys that were never configured) must be omitted, not sent as "":
// sending them would no-op-delete on this machine but could wipe a key
// another window saved between our populate and our close.
function testUntouchedEmptyKeysStayOmitted() {
  const {win, posted} = makePage();
  openSettings(win);
  send(win, {
    type: 'configData',
    config: STORED,
    apiKeys: {OPENAI_API_KEY: 'sk-keep'},
  });
  closeSettings(win);

  const msg = lastMsg(posted, 'saveConfig');
  assert.ok(msg, 'closing a populated panel still saves the form');
  assert.ok(
    !('ANTHROPIC_API_KEY' in msg.apiKeys),
    'a key box that was empty all along must be omitted from the payload',
  );
  assert.ok(
    !('GEMINI_API_KEY' in msg.apiKeys),
    'no unedited empty box may be serialized',
  );
  assert.strictEqual(msg.apiKeys.OPENAI_API_KEY, 'sk-keep');
  win.close();
  console.log('  ok - untouched empty key boxes are omitted');
}

// The panel-never-populated path (configData never arrived) sends only
// the edited fields; a cleared box the user explicitly touched is an
// edit like any other and must still carry the deletion.
function testClearedKeyIsSentEvenWithoutConfigData() {
  const {win, posted} = makePage();
  openSettings(win);
  typeInto(win, 'cfg-key-GEMINI_API_KEY', 'typed-then-cleared');
  typeInto(win, 'cfg-key-GEMINI_API_KEY', '');
  closeSettings(win);

  const msg = lastMsg(posted, 'saveConfig');
  assert.ok(msg, 'a form with edits must be saved even without configData');
  assert.strictEqual(
    msg.apiKeys.GEMINI_API_KEY,
    '',
    'an edited-then-cleared key must be serialized as "" on the ' +
      'partial-save path too',
  );
  assert.ok(
    !('ANTHROPIC_API_KEY' in msg.apiKeys),
    'unedited fields stay out of a partial payload',
  );
  win.close();
  console.log('  ok - the partial-save path also carries the deletion');
}

function main() {
  testClearedKeyIsSentAsEmptyString();
  testUntouchedEmptyKeysStayOmitted();
  testClearedKeyIsSentEvenWithoutConfigData();
  console.log('settingsApiKeyDelete.test.js: all tests passed');
}

main();
