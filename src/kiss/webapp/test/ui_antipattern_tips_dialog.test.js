// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// UI anti-pattern fix A12 / A7 for the Tips dialog (media/tips.js): the
// aria-modal panel closes on Escape and on a backdrop click, moves focus
// into itself when it opens, hands focus back to the opener when it
// closes, keeps Tab inside the panel, and offers a persisted "Don't show
// tips automatically" choice that is relayed to the host through the
// `kiss-tips-opt-out` window event.

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

function loadChatDom(cfg, storage) {
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
  html = html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');
  const dom = new JSDOM(html, {
    runScripts: 'outside-only',
    pretendToBeVisual: true,
    url: 'https://localhost/',
  });
  const win = dom.window;
  win.localStorage.clear();
  if (storage) {
    for (const k of Object.keys(storage))
      win.localStorage.setItem(k, storage[k]);
  }
  win.eval(`window.__TIPS__ = ${JSON.stringify(cfg)};`);
  win.eval(fs.readFileSync(path.join(MEDIA, 'tips.js'), 'utf8'));
  return win;
}

function panel(win) {
  return win.document.body.querySelector('kiss-tips-panel');
}

function part(win, selector) {
  const host = panel(win);
  assert.ok(host, 'tips panel must be mounted');
  return host.shadowRoot.querySelector(selector);
}

function key(win, target, name, init) {
  target.dispatchEvent(
    new win.KeyboardEvent(
      'keydown',
      Object.assign(
        {key: name, bubbles: true, cancelable: true, composed: true},
        init,
      ),
    ),
  );
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

const TIPS = {tips: ['# One\n\nfirst', '# Two\n\nsecond'], show: false};

test('opening moves focus into the dialog and closing returns it to the opener', () => {
  const win = loadChatDom(TIPS);
  const opener = win.document.getElementById('tips-btn');
  assert.ok(opener, '#tips-btn must exist in chat.html');
  opener.focus();
  assert.strictEqual(win.document.activeElement, opener);
  opener.click();
  const host = panel(win);
  assert.ok(host, 'clicking #tips-btn opens the dialog');
  assert.strictEqual(
    win.document.activeElement,
    host,
    'focus is inside the dialog host',
  );
  assert.strictEqual(
    host.shadowRoot.activeElement,
    part(win, '.tips-close'),
    'the close button receives focus',
  );
  part(win, '.tips-close').click();
  assert.strictEqual(panel(win), null, 'dialog is removed');
  assert.strictEqual(
    win.document.activeElement,
    opener,
    'focus returns to #tips-btn',
  );
});

test('Escape closes the dialog', () => {
  const win = loadChatDom(TIPS);
  const opener = win.document.getElementById('tips-btn');
  opener.focus();
  opener.click();
  assert.ok(panel(win));
  key(win, part(win, '.tips-close'), 'Escape');
  assert.strictEqual(panel(win), null, 'Escape removes the dialog');
  assert.strictEqual(win.document.activeElement, opener);
  // The listener is gone with the dialog: a second Escape is harmless.
  key(win, win.document.body, 'Escape');
  assert.strictEqual(panel(win), null);
});

test('clicking the dimmed backdrop closes; clicking inside the panel does not', () => {
  const win = loadChatDom(TIPS);
  win.document.getElementById('tips-btn').click();
  part(win, '.tips-panel').dispatchEvent(
    new win.MouseEvent('click', {bubbles: true, composed: true}),
  );
  assert.ok(panel(win), 'a click inside the panel keeps it open');
  part(win, '.tips-overlay').dispatchEvent(
    new win.MouseEvent('click', {bubbles: true}),
  );
  assert.strictEqual(panel(win), null, 'a backdrop click closes it');
});

test('Tab wraps inside the dialog in both directions', () => {
  const win = loadChatDom(TIPS);
  win.document.getElementById('tips-btn').click();
  const host = panel(win);
  const close = part(win, '.tips-close');
  const next = part(win, '.tips-next');
  // On the first tip Previous is disabled, so the tab ring is
  // close -> opt-out checkbox -> Next.
  assert.ok(part(win, '.tips-prev').disabled);
  next.focus();
  key(win, next, 'Tab');
  assert.strictEqual(
    host.shadowRoot.activeElement,
    close,
    'Tab from the last item wraps to the first',
  );
  key(win, close, 'Tab', {shiftKey: true});
  assert.strictEqual(
    host.shadowRoot.activeElement,
    next,
    'Shift+Tab from the first item wraps to the last',
  );
});

test('opt-out checkbox persists locally and posts a tipsOptOut message for the host', () => {
  const win = loadChatDom(TIPS);
  const relayed = [];
  // Objects cross the jsdom realm boundary, so compare by value.
  win.addEventListener('kiss-tips-opt-out', e =>
    relayed.push(JSON.parse(JSON.stringify(e.detail))),
  );
  win.document.getElementById('tips-btn').click();
  const box = part(win, '.tips-optout-input');
  assert.ok(box, 'the footer has the opt-out checkbox');
  assert.strictEqual(box.checked, false);
  assert.ok(/don't show tips/i.test(part(win, '.tips-optout').textContent));
  box.checked = true;
  box.dispatchEvent(new win.Event('change', {bubbles: true}));
  assert.strictEqual(win.localStorage.getItem('kissTipsOptOut'), '1');
  assert.deepStrictEqual(relayed, [{type: 'tipsOptOut', optOut: true}]);
  box.checked = false;
  box.dispatchEvent(new win.Event('change', {bubbles: true}));
  assert.strictEqual(win.localStorage.getItem('kissTipsOptOut'), null);
  assert.deepStrictEqual(relayed[1], {type: 'tipsOptOut', optOut: false});
});

test('auto-show on first run is skipped once the user opted out', () => {
  const shown = loadChatDom({tips: TIPS.tips, show: true});
  assert.ok(panel(shown), 'without an opt-out the first-run dialog opens');
  const optedOut = loadChatDom(
    {tips: TIPS.tips, show: true},
    {kissTipsOptOut: '1'},
  );
  assert.strictEqual(panel(optedOut), null, 'with the opt-out it stays closed');
  // The user can still open tips on purpose, and sees the choice as checked.
  optedOut.document.getElementById('tips-btn').click();
  assert.ok(panel(optedOut));
  assert.strictEqual(part(optedOut, '.tips-optout-input').checked, true);
});

console.log(`\n${passed} passed, ${failures.length} failed`);
for (const f of failures) {
  console.error(`\n${f.name}\n${f.err && f.err.stack}`);
}
if (failures.length) process.exit(1);
