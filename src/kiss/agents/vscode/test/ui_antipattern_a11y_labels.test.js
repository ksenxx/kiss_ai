// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// UI anti-pattern fixes A5 / A2 / A10 / A15 in chat.html and main.css:
// every icon-only control has an accessible name, the destructive
// restart-server button names its outcome instead of "OK", the clear
// buttons are 24x24 hit targets, the model pill and model search keep a
// visible focus ring, and disabled menu items stay legible.

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
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
  return {win, posted};
}

let passed = 0;
const failures = [];

function test(name, fn) {
  try {
    fn();
    passed += 1;
    console.log(`  ok - ${name}`);
  } catch (err) {
    failures.push({name, err});
    console.log(`  not ok - ${name}`);
  }
}

const {win} = makeWebview();
const doc = win.document;

function label(id) {
  const el = doc.getElementById(id);
  assert.ok(el, `#${id} must exist`);
  return (el.getAttribute('aria-label') || '').trim();
}

test('icon-only buttons have an accessible name', () => {
  const expected = {
    'input-clear-btn': 'Clear input',
    'sidebar-close': 'Close sidebar',
    'model-search-clear': 'Clear search',
    'history-search-clear': 'Clear search',
  };
  for (const id of Object.keys(expected)) {
    assert.strictEqual(label(id), expected[id], `#${id} aria-label`);
  }
  for (const id of [
    'frequent-panel-close',
    'tricks-panel-close',
    'settings-panel-close',
  ]) {
    assert.ok(/^Close /.test(label(id)), `#${id} must be named "Close ..."`);
  }
});

test('#menu-btn has both a tooltip and an accessible name that say what it does', () => {
  const btn = doc.getElementById('menu-btn');
  const tip = btn.getAttribute('data-tooltip') || '';
  assert.ok(tip.length > 0, 'data-tooltip must be set');
  assert.strictEqual(btn.getAttribute('aria-label'), tip);
  assert.ok(/history/i.test(tip), `must mention history, got: ${tip}`);
  assert.notStrictEqual(
    tip,
    doc.getElementById('more-btn').getAttribute('aria-label'),
    'must not collide with the "More actions" button',
  );
});

test('every data-tooltip-only button carries a matching aria-label', () => {
  for (const id of [
    'tricks-btn',
    'workdir-btn',
    'voice-btn',
    'share-btn',
    'upload-btn',
    'settings-btn',
    'model-btn',
    'send-btn',
    'stop-btn',
  ]) {
    const el = doc.getElementById(id);
    assert.ok(el, `#${id} must exist`);
    assert.strictEqual(
      el.getAttribute('aria-label'),
      el.getAttribute('data-tooltip'),
      `#${id} aria-label must equal its data-tooltip`,
    );
  }
  assert.strictEqual(label('autocommit-btn'), 'Git commit');
});

test('search inputs are named by aria-label, not only by placeholder', () => {
  assert.strictEqual(label('model-search'), 'Search models');
  assert.strictEqual(label('history-search'), 'Search history');
});

test('restart-server confirm names the outcome and keeps Cancel first', () => {
  const ok = doc.getElementById('server-reset-confirm-ok');
  const cancel = doc.getElementById('server-reset-confirm-cancel');
  assert.strictEqual(ok.textContent.trim(), 'Restart and abort task');
  assert.strictEqual(cancel.textContent.trim(), 'Cancel');
  assert.strictEqual(
    cancel.compareDocumentPosition(ok) & win.Node.DOCUMENT_POSITION_FOLLOWING,
    win.Node.DOCUMENT_POSITION_FOLLOWING,
    'Cancel must come before the destructive button',
  );
});

const css = fs.readFileSync(path.join(MEDIA, 'main.css'), 'utf8');

function rule(selector) {
  const re = new RegExp(
    selector.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + '\\s*\\{([^}]*)\\}',
  );
  const m = css.match(re);
  assert.ok(m, `rule for ${selector} must exist`);
  return m[1];
}

test('clear buttons are at least 24x24 px hit targets', () => {
  for (const sel of ['#input-clear-btn', '.search-clear-btn']) {
    const body = rule(sel);
    const w = /width:\s*(\d+)px/.exec(body);
    const h = /height:\s*(\d+)px/.exec(body);
    assert.ok(w && Number(w[1]) >= 24, `${sel} width >= 24, got ${body}`);
    assert.ok(h && Number(h[1]) >= 24, `${sel} height >= 24, got ${body}`);
  }
});

test('model pill and model search show a focus ring instead of outline:none', () => {
  assert.ok(!/outline:\s*none/.test(rule('#model-btn')), '#model-btn');
  assert.ok(!/outline:\s*none/.test(rule('#model-search')), '#model-search');
  assert.ok(
    /outline:\s*1px solid var\(--accent\)/.test(
      rule('#model-btn:focus-visible'),
    ),
  );
  assert.ok(
    /outline:\s*1px solid var\(--accent\)/.test(
      rule('#model-search:focus-visible'),
    ),
  );
});

test('disabled menu items keep their label readable', () => {
  const m = /opacity:\s*([0-9.]+)/.exec(rule('.more-menu-item:disabled'));
  assert.ok(m && Number(m[1]) >= 0.5, `opacity >= 0.5, got ${m && m[1]}`);
});

console.log(`\n${passed} passed, ${failures.length} failed`);
for (const f of failures) {
  console.error(`\n${f.name}\n${f.err && f.err.stack}`);
}
if (failures.length) process.exit(1);
