// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) regression test for review-vscode.md #5: the settings
// panel's first-party API-key inventory was declared VERBATIM in both the
// load path (populateConfigForm) and the save path (collectConfigForm), so
// a key added to one list silently drifted out of the other: it would
// display but never save, or save but never display.  The fix drives both
// loops (and the secret-input listeners) from one shared constant; this
// test pins the behavior the constant guarantees — every key the daemon
// can send is round-tripped: displayed on configData, editable, and
// serialized back into saveConfig.

'use strict';

const assert = require('assert');
const {makePage, send} = require('./simplify2_harness');

// The full first-party inventory the daemon's key store supports (see
// chat.html's cfg-key-* inputs).
const KEYS = [
  'GEMINI_API_KEY',
  'OPENAI_API_KEY',
  'ANTHROPIC_API_KEY',
  'ANTHROPIC_WORKSPACE_ID',
  'TOGETHER_API_KEY',
  'OPENROUTER_API_KEY',
  'ZAI_API_KEY',
  'MOONSHOT_API_KEY',
  'AWS_BEARER_TOKEN_BEDROCK',
  'AWS_ACCESS_KEY_ID',
  'AWS_SECRET_ACCESS_KEY',
  'AWS_SESSION_TOKEN',
  'AWS_REGION',
];

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

function main() {
  const {win, posted} = makePage();
  openSettings(win);

  // Load path: every stored key populates its box.
  const stored = {};
  for (const k of KEYS) stored[k] = 'sk-' + k.toLowerCase();
  send(win, {
    type: 'configData',
    config: {max_budget: 100},
    apiKeys: stored,
  });
  for (const k of KEYS) {
    const box = win.document.getElementById('cfg-key-' + k);
    assert.ok(box, `settings panel must have a box for ${k}`);
    assert.strictEqual(
      box.value,
      stored[k],
      `configData must populate ${k} (load-path inventory is missing it)`,
    );
    // Every credential is masked with a reveal toggle; the AWS region
    // is plain text (an identifier, not a secret) and gets no toggle.
    const secret = k !== 'AWS_REGION';
    assert.strictEqual(box.type, secret ? 'password' : 'text', `${k} input type`);
    assert.strictEqual(
      Boolean(win.document.getElementById('cfg-key-' + k + '-toggle')),
      secret,
      `${k} reveal toggle presence`,
    );
  }

  // Save path: every edited key is serialized back.
  for (const k of KEYS) typeInto(win, 'cfg-key-' + k, 'new-' + k);
  closeSettings(win);
  const msg = lastMsg(posted, 'saveConfig');
  assert.ok(msg, 'closing the panel must save the form');
  for (const k of KEYS) {
    assert.strictEqual(
      msg.apiKeys && msg.apiKeys[k],
      'new-' + k,
      `saveConfig must serialize ${k} (save-path inventory is missing it)`,
    );
  }

  console.log('audit0911_api_key_inventory_parity: OK');
}

try {
  main();
  process.exit(0);
} catch (err) {
  console.error(err && err.stack ? err.stack : err);
  process.exit(1);
}
