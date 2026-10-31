// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// A rebranded install (brand.json productName, injected as
// {{PRODUCT_NAME}} by web_server._build_html and as window.__BRAND__)
// shows its own name everywhere a user can read it: the Browser /
// Terminal tooltips of the "..." menu (chat.html) and the streamed
// browser tab's badge title (browserTab.js) used to say "KISS Sorcar".

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');
const {makeRunner} = require('./ui_antipattern_harness');
const runner = makeRunner();

const MEDIA = path.join(__dirname, '..', 'media');
const BRAND = 'Seamless Loop';
const LABEL_ATTRS = [
  'title',
  'aria-label',
  'data-tooltip',
  'placeholder',
  'alt',
];

/**
 * Every string of the parsed page *doc* a user can read: its title, the
 * text outside scripts and styles, and the labelling attributes.
 */
function readableStrings(doc) {
  const body = doc.body.cloneNode(true);
  for (const el of body.querySelectorAll('script, style')) el.remove();
  const strings = [doc.title, body.textContent];
  for (const el of doc.querySelectorAll('*')) {
    for (const name of LABEL_ATTRS) {
      if (el.hasAttribute(name)) strings.push(el.getAttribute(name));
    }
  }
  return strings;
}

runner.test(
  'the rendered chat page names the brand, never the stock product',
  () => {
    const html = fs
      .readFileSync(path.join(MEDIA, 'chat.html'), 'utf8')
      .replace(/\{\{([A-Z_]+)\}\}/g, (m, name) =>
        name === 'PRODUCT_NAME' ? BRAND : '',
      );
    const {window: win} = new JSDOM(html);
    const browserTip = win.document
      .getElementById('browser-btn')
      .getAttribute('data-tooltip');
    const terminalTip = win.document
      .getElementById('terminal-btn')
      .getAttribute('data-tooltip');
    assert.strictEqual(
      browserTip,
      'Open the browser of the machine running Seamless Loop as a tab',
    );
    assert.strictEqual(
      terminalTip,
      'Open a terminal on the machine running Seamless Loop as a tab',
    );
    const readable = readableStrings(win.document);
    assert.ok(
      readable.some(s => s.includes('Welcome to ' + BRAND)),
      'the welcome heading names the brand',
    );
    assert.deepStrictEqual(
      readable.filter(s => s.includes('KISS Sorcar')),
      [],
      'no user-visible string of the rendered page names the stock product',
    );
    win.close();
  },
);

runner.test('the browser badge title names the brand', () => {
  const {window: win} = new JSDOM('<!DOCTYPE html><body></body>', {
    runScripts: 'outside-only',
  });
  win.__BRAND__ = {productName: BRAND};
  win.eval(fs.readFileSync(path.join(MEDIA, 'browserTab.js'), 'utf8'));
  const view = win.BrowserTabView.create('tab-1', {send: () => {}});
  const badge = view.el.querySelector('.browser-badge');
  view.setBadge('chromium', true);
  assert.strictEqual(
    badge.title,
    'chromium (default browser of the machine running Seamless Loop)',
  );
  view.setBadge('firefox', false);
  assert.strictEqual(
    badge.title,
    'firefox (browser on the machine running Seamless Loop)',
  );
  view.setBadge('firefox', false, 'Chrome is not installed.');
  assert.strictEqual(
    badge.title,
    'Chrome is not installed. Showing firefox instead.',
  );
  win.close();
});

runner.test('without a brand the badge falls back to the stock name', () => {
  const {window: win} = new JSDOM('<!DOCTYPE html><body></body>', {
    runScripts: 'outside-only',
  });
  win.eval(fs.readFileSync(path.join(MEDIA, 'browserTab.js'), 'utf8'));
  const view = win.BrowserTabView.create('tab-1', {send: () => {}});
  view.setBadge('chromium', true);
  assert.strictEqual(
    view.el.querySelector('.browser-badge').title,
    'chromium (default browser of the machine running KISS Sorcar)',
  );
  win.close();
});

runner.report('brandName_sweep1026');
