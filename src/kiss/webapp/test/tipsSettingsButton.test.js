// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// The Tips button in the settings panel of chat.html: placement,
// icon/label, and the tips window it opens through tips.js (navigation,
// single panel, empty state, Markdown fallback).

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');
const {newDom, MEDIA} = require('./simplify2_harness');
const {makeRunner} = require('./ui_antipattern_harness');
const {test, report} = makeRunner();

// The real chat.html markup with its placeholders blanked and its script
// tags removed; tips.js (and marked) are then evaluated in the page the
// way web_server._build_html serves them (main.js is never loaded).
function loadChatDom(cfg, {withConfig = true, withMarked = true} = {}) {
  const win = newDom({runScripts: 'outside-only', bodyClass: 'remote-chat'});
  const run = file =>
    win.eval(fs.readFileSync(path.join(MEDIA, file), 'utf-8'));
  if (withMarked) run('marked.min.js');
  if (withConfig) win.eval(`window.__TIPS__ = ${JSON.stringify(cfg)};`);
  run('panelCopy.js');
  run('tips.js');
  return win;
}

function panel(win) {
  return win.document.body.querySelector('kiss-tips-panel');
}

function panelParts(win) {
  const host = panel(win);
  assert.ok(host, 'kiss-tips-panel must be mounted on document.body');
  const root = host.shadowRoot;
  assert.ok(root, 'kiss-tips-panel must use shadow DOM');
  return {
    host,
    root,
    body: root.querySelector('.tips-body'),
    counter: root.querySelector('.tips-counter'),
    prev: root.querySelector('.tips-prev'),
    next: root.querySelector('.tips-next'),
    close: root.querySelector('.tips-close'),
  };
}

const THREE_TIPS = [
  '## First tip\n\n- alpha\n- beta',
  'Second tip with **bold** text.',
  '### Third tip\n\n`code` here.',
];

test('Tips button sits in the settings row', () => {
  const win = loadChatDom({tips: ['Hello.'], show: false});
  const doc = win.document;
  const tips = doc.getElementById('tips-btn');
  assert.ok(tips, '#tips-btn must exist in the chat HTML');
  assert.ok(
    tips.closest('#settings-panel'),
    'Tips button must live inside the settings panel',
  );
  assert.ok(
    tips.parentElement.classList.contains('config-update-row'),
    'Tips button must live in the .config-update-row button row',
  );
});

test('Tips button carries a lightbulb icon, "Tips" label, tooltip, and aria-label', () => {
  const win = loadChatDom({tips: ['Hello.'], show: false});
  const btn = win.document.getElementById('tips-btn');
  assert.ok(btn, '#tips-btn must exist');
  assert.strictEqual(btn.tagName, 'BUTTON');
  assert.ok(btn.querySelector('svg'), 'bulb button must contain an SVG');
  const tooltip = btn.getAttribute('data-tooltip') || '';
  assert.ok(/tip/i.test(tooltip), `tooltip must mention tips: "${tooltip}"`);
  const label = btn.getAttribute('aria-label') || '';
  assert.ok(/tip/i.test(label), `aria-label must mention tips: "${label}"`);
  const span = btn.querySelector('span');
  assert.ok(span, 'Tips button must carry a visible text label');
  assert.strictEqual(span.textContent.trim(), 'Tips');
  assert.ok(
    btn.classList.contains('config-update-btn'),
    'Tips button must share the settings-row button styling',
  );
});

test('rendered HTML contains exactly one #tips-btn', () => {
  const win = loadChatDom({tips: ['Hello.'], show: false});
  assert.strictEqual(
    win.document.querySelectorAll('#tips-btn').length,
    1,
    'the bulb button id must be unique',
  );
});

test('clicking the bulb shows the tips window with the tips', () => {
  const win = loadChatDom({tips: THREE_TIPS, show: false});
  assert.strictEqual(panel(win), null, 'no panel before the click');
  win.document.getElementById('tips-btn').click();
  const {body, counter, prev, next} = panelParts(win);
  assert.strictEqual(counter.textContent, '1 / 3');
  assert.ok(
    body.innerHTML.includes('<h2') && body.innerHTML.includes('<li>'),
    `first tip markdown must render to HTML, got: ${body.innerHTML}`,
  );
  assert.strictEqual(prev.disabled, true, 'Previous disabled on first tip');
  assert.strictEqual(next.disabled, false, 'Next enabled on first tip');
});

test('clicking the bulb twice does not stack a second panel', () => {
  const win = loadChatDom({tips: THREE_TIPS, show: false});
  const btn = win.document.getElementById('tips-btn');
  btn.click();
  btn.click();
  assert.strictEqual(
    win.document.querySelectorAll('kiss-tips-panel').length,
    1,
    'only one tips panel instance may exist at a time',
  );
});

test('after Close the bulb reopens the tips window from the first tip', () => {
  const win = loadChatDom({tips: THREE_TIPS, show: false});
  const btn = win.document.getElementById('tips-btn');
  btn.click();
  const first = panelParts(win);
  first.next.click();
  assert.strictEqual(first.counter.textContent, '2 / 3');
  first.close.click();
  assert.strictEqual(panel(win), null);
  btn.click();
  const second = panelParts(win);
  assert.strictEqual(second.counter.textContent, '1 / 3');
});

test('bulb also dismisses-and-reopens after a fresh-install auto-show', () => {
  const win = loadChatDom({tips: THREE_TIPS, show: true});
  const btn = win.document.getElementById('tips-btn');
  btn.click();
  assert.strictEqual(
    win.document.querySelectorAll('kiss-tips-panel').length,
    1,
  );
  panelParts(win).close.click();
  btn.click();
  assert.strictEqual(panelParts(win).counter.textContent, '1 / 3');
});

test('bulb opens an empty, closeable panel when there are no tips', () => {
  const win = loadChatDom({tips: [], show: false});
  win.document.getElementById('tips-btn').click();
  const {counter, prev, next, close} = panelParts(win);
  assert.strictEqual(counter.textContent, '');
  assert.strictEqual(prev.disabled, true);
  assert.strictEqual(next.disabled, true);
  close.click();
  assert.strictEqual(panel(win), null);
});

test('bulb click does not throw when window.__TIPS__ is missing', () => {
  const win = loadChatDom(null, {withConfig: false});
  win.document.getElementById('tips-btn').click();
  const {counter} = panelParts(win);
  assert.strictEqual(counter.textContent, '');
});

test('tips.js tolerates a DOM without #tips-btn', () => {
  const dom = new JSDOM('<!DOCTYPE html><html><body></body></html>', {
    runScripts: 'outside-only',
  });
  dom.window.eval(
    `window.__TIPS__ = ${JSON.stringify({tips: THREE_TIPS, show: false})};`,
  );
  dom.window.eval(fs.readFileSync(path.join(MEDIA, 'tips.js'), 'utf-8'));
  assert.strictEqual(
    dom.window.document.querySelector('kiss-tips-panel'),
    null,
  );
});

report('tipsSettingsButton');
