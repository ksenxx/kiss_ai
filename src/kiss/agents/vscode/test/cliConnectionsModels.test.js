'use strict';
const assert = require('assert');
const {makeWebview, send} = require('./ui_antipattern_harness');
const models = [
  {
    name: 'cc/opus',
    vendor: 'Anthropic',
    provider: 'Claude Code CLI',
    access_type: 'cli',
    uses: 0,
    cost_label: 'Subscription',
  },
  {
    name: 'cc/fable',
    vendor: 'Anthropic',
    provider: 'Claude Code CLI',
    access_type: 'cli',
    available: false,
    unavailable_reason: 'Enable usage credits in Settings.',
    uses: 0,
    cost_label: 'Usage credits may apply',
  },
  {
    name: 'gpt-6.1-sol-max',
    vendor: 'OpenAI',
    provider: 'OpenAI',
    access_type: 'api',
    uses: 0,
    inp: 2,
    out: 10,
  },
  {
    name: 'gpt-5.6-sol',
    vendor: 'OpenAI',
    provider: 'OpenAI',
    access_type: 'api',
    uses: 0,
    inp: 4,
    out: 20,
  },
  {name: 'haiku-api', vendor: 'Anthropic', uses: 4, inp: 1, out: 5},
  {name: 'local-long-custom-model', vendor: 'Custom', uses: 0, inp: 0, out: 0},
  {name: 'autorouter', vendor: 'Router', uses: 0, cost_label: 'Task-dependent'},
];
const {win, posted} = makeWebview();
const doc = win.document;
send(win, {type: 'models', models, selected: 'cc/opus'});
doc.getElementById('model-btn').click();
const headings = [...doc.querySelectorAll('.model-group-hdr')].map(
  el => el.textContent,
);
for (const name of [
  'Frequently Used',
  'CLI Models',
  'API Models',
  'Custom Models',
  'Routers',
  'OpenAI · Sol',
])
  assert.ok(headings.includes(name), name);
assert.strictEqual(
  doc.getElementById('model-btn').getAttribute('aria-expanded'),
  'true',
);
const rows = [...doc.querySelectorAll('.model-item')];
assert.ok(
  rows.findIndex(el => el.textContent.includes('gpt-6.1')) <
    rows.findIndex(el => el.textContent.includes('gpt-5.6')),
);
assert.ok(
  rows
    .find(el => el.textContent.includes('cc/fable'))
    .textContent.includes('Enable usage credits'),
);
const search = doc.getElementById('model-search');
function query(value) {
  search.value = value;
  search.dispatchEvent(new win.Event('input', {bubbles: true}));
}
query('CLI');
assert.strictEqual(doc.querySelectorAll('.model-item').length, 2);
search.dispatchEvent(
  new win.KeyboardEvent('keydown', {key: 'ArrowDown', bubbles: true}),
);
assert.ok(search.getAttribute('aria-activedescendant').includes('cc%2Fopus'));
search.dispatchEvent(
  new win.KeyboardEvent('keydown', {key: 'Enter', bubbles: true}),
);
assert.strictEqual(
  posted.filter(m => m.type === 'selectModel').at(-1).model,
  'cc/opus',
);
doc.getElementById('model-btn').click();
query('unknown-no-model');
assert.strictEqual(
  doc.querySelector('.model-picker-empty').textContent,
  'No models match your search.',
);
query('Sol');
assert.strictEqual(doc.querySelectorAll('.model-item').length, 2);
// Refresh/reconnect does not replace a valid tab selection.
send(win, {type: 'models', models, selected: 'gpt-5.6-sol'});
assert.ok(doc.getElementById('model-name').textContent.includes('cc/opus'));
doc.getElementById('settings-btn').click();
assert.ok(posted.some(m => m.type === 'getCLIConnections'));
send(win, {
  type: 'cliConnections',
  machine: 'server',
  platform: 'linux',
  connections: [
    {
      provider: 'claude',
      status: 'connected',
      version: '2.1.300',
      billing_mode: 'existing',
      message: 'Uses existing billing.',
    },
  ],
});
const mode = doc.getElementById('cfg-claude-cli-billing-mode');
assert.strictEqual(mode.value, 'existing');
mode.value = 'subscription';
mode.dispatchEvent(new win.Event('change', {bubbles: true}));
assert.strictEqual(
  posted.filter(m => m.type === 'saveConfig').at(-1).config
    .claude_cli_billing_mode,
  'subscription',
);
send(win, {
  type: 'cliConnections',
  machine: 'server',
  platform: 'linux',
  connections: [
    {
      provider: 'claude',
      status: 'signed_out',
      billing_mode: 'existing',
      message: 'Sign in.',
    },
  ],
});
assert.strictEqual(
  mode.value,
  'subscription',
  'late refresh must not replace edited billing mode',
);
assert.ok(
  doc.getElementById('cli-claude-status').textContent.includes('Signed out'),
);
send(win, {
  type: 'models',
  models: models.filter(m => m.name !== 'cc/opus'),
  selected: 'gpt-5.6-sol',
});
assert.ok(
  doc.getElementById('model-name').textContent.includes('gpt-5.6-sol'),
  'a removed custom or catalog selection must release the pin',
);
win.close();
console.log(
  'CLI connection settings, picker grouping/search/keyboard/reconnect passed',
);
