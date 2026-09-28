// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// The daemon lists SEAs registered as models (`autorouter`, `bestrouter`)
// in the model picker under the `Router` vendor. They are not models, so
// each row carries a `cost_label` string instead of per-1M prices, and the
// picker shows that label where every other row shows "$in / $out".
// Picking one posts a plain `selectModel` like any other row.
//
// Driven through the real webview: media/chat.html plus media/main.js in
// jsdom, fed the `models` event the daemon emits.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

const REAL_MODEL = 'claude-opus-5';
const LABEL = 'routes by its own protocol';

const MODEL_LIST = [
  {
    name: 'autorouter',
    inp: 0,
    out: 0,
    uses: 0,
    vendor: 'Router',
    cost_label: LABEL,
  },
  {
    name: 'bestrouter',
    inp: 0,
    out: 0,
    uses: 0,
    vendor: 'Router',
    cost_label: LABEL,
  },
  {name: REAL_MODEL, inp: 5, out: 25, uses: 3, vendor: 'anthropic'},
];

function makeWebview() {
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, REAL_MODEL);
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
  win.requestAnimationFrame = function (cb) {
    cb();
    return 0;
  };

  const posted = [];
  win.acquireVsCodeApi = function () {
    let state;
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
  win.eval(
    fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8') +
      '\n//# sourceURL=autorouter-main.js',
  );
  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

/** Open the picker and return its rows as {name, cost, el}. */
function openPicker(win) {
  win.document
    .getElementById('model-btn')
    .dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  return Array.from(
    win.document.querySelectorAll('#model-list .model-item'),
  ).map(el => ({
    name: el
      .querySelector('.model-item-name')
      .textContent.replace(/\u200e/g, ''),
    cost: el.querySelector('.model-cost').textContent,
    el,
  }));
}

let passed = 0;
const failures = [];

function test(name, fn) {
  try {
    fn();
    passed++;
    console.log(`  \u2713 ${name}`);
  } catch (e) {
    failures.push({name, error: e});
    console.log(`  \u2717 ${name}`);
    console.log(`      ${e.stack || e.message}`);
  }
}

test('the router rows show their cost_label, priced rows show prices', () => {
  const {win} = makeWebview();
  send(win, {type: 'models', models: MODEL_LIST, selected: REAL_MODEL});

  const rows = openPicker(win);
  for (const name of ['autorouter', 'bestrouter']) {
    const router = rows.find(r => r.name === name);
    assert.ok(router, `the picker must list ${name}`);
    assert.strictEqual(router.cost, LABEL);
  }
  const real = rows.find(r => r.name === REAL_MODEL);
  assert.ok(real, 'the picker must list the real model');
  assert.strictEqual(real.cost, '$5.00 / $25.00');
  win.close();
});

test('the router rows sit in their own vendor group', () => {
  const {win} = makeWebview();
  send(win, {type: 'models', models: MODEL_LIST, selected: REAL_MODEL});

  openPicker(win);
  const groups = Array.from(
    win.document.querySelectorAll('#model-list .model-group-hdr'),
  ).map(el => el.textContent);
  assert.ok(
    groups.indexOf('Router') >= 0,
    `expected a Router group; got ${JSON.stringify(groups)}`,
  );
  win.close();
});

test('picking autorouter posts selectModel like any other row', () => {
  const {win, posted} = makeWebview();
  send(win, {type: 'models', models: MODEL_LIST, selected: REAL_MODEL});

  const rows = openPicker(win);
  rows
    .find(r => r.name === 'autorouter')
    .el.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));

  const picks = posted.filter(m => m && m.type === 'selectModel');
  assert.ok(picks.length > 0, 'no selectModel message was posted');
  assert.strictEqual(picks[picks.length - 1].model, 'autorouter');
  assert.strictEqual(
    win.document
      .getElementById('model-name')
      .textContent.replace(/\u200e/g, ''),
    'autorouter',
  );
  win.close();
});

console.log(`\n${passed} passed, ${failures.length} failed`);
process.exit(failures.length > 0 ? 1 : 0);
