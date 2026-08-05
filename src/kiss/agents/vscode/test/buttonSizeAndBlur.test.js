// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

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

  const style = win.document.createElement('style');
  style.textContent = fs.readFileSync(path.join(MEDIA, 'main.css'), 'utf8');
  win.document.head.appendChild(style);

  win.__TIPS__ = {tips: [], show: false};
  win.eval(fs.readFileSync(path.join(MEDIA, 'tips.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));

  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(
fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
  return {win, posted};
}

function cssRule(win, selector) {
  for (const sheet of win.document.styleSheets) {
    for (const rule of sheet.cssRules) {
      if (!rule.selectorText) continue;
      const parts = rule.selectorText.split(',').map(s => s.trim());
      if (parts.includes(selector)) return rule.style;
    }
  }
  return null;
}

function px(value) {
  return parseFloat(String(value));
}

function assertAtLeast(actual, min, what) {
  assert.ok(
    px(actual) >= min,
    what + ' must be at least ' + min + 'px, got "' + actual + '"',
  );
}

function testInputFooterButtonsAreBigger() {
  const {win} = makeWebview();
  const doc = win.document;

  const send = win.getComputedStyle(doc.getElementById('send-btn'));
  assertAtLeast(send.width, 32, '#send-btn width');
  assertAtLeast(send.height, 32, '#send-btn height');
  const sendSvg = cssRule(win, '#send-btn svg');
  assert.ok(sendSvg, 'main.css must size #send-btn svg');
  assertAtLeast(sendSvg.width, 15, '#send-btn svg width');

  const stop = win.getComputedStyle(doc.getElementById('stop-btn'));
  assertAtLeast(stop.width, 32, '#stop-btn width');
  assertAtLeast(stop.height, 32, '#stop-btn height');
  const stopSvg = cssRule(win, '#stop-btn svg');
  assert.ok(stopSvg, 'main.css must size #stop-btn svg');
  assertAtLeast(stopSvg.width, 15, '#stop-btn svg width');

  for (const id of [
    'menu-btn',
    'model-btn',
    'upload-btn',
    'tricks-btn',
    'voice-btn',
  ]) {
    const svg = doc.querySelector('#' + id + ' svg');
    assert.ok(svg, '#' + id + ' must contain an svg icon');
    assertAtLeast(svg.getAttribute('width'), 16, '#' + id + ' svg width');
    assertAtLeast(svg.getAttribute('height'), 16, '#' + id + ' svg height');
  }
  const uploadSvg = cssRule(win, '#upload-btn svg');
  if (uploadSvg && uploadSvg.width) {
    assertAtLeast(uploadSvg.width, 16, '#upload-btn svg CSS width');
  }
  win.close();
  console.log('  ok - input-footer buttons are bigger');
}

function testTabBarPlusAndSettingsAreBigger() {
  const {win} = makeWebview();
  const doc = win.document;

  const addBtn = doc.querySelector('.chat-tab-add');
  assert.ok(addBtn, 'tab bar must render the "+" button');
  const add = cssRule(win, '.chat-tab-add');
  assert.ok(add, 'main.css must style .chat-tab-add');
  assertAtLeast(add.minWidth, 28, '.chat-tab-add min-width');
  assertAtLeast(add.fontSize, 20, '.chat-tab-add font-size');

  const settingsBtn = doc.querySelector('.chat-tab-settings');
  assert.ok(settingsBtn, 'tab bar must render the settings button');
  const settings = cssRule(win, '.chat-tab-settings');
  assert.ok(settings, 'main.css must style .chat-tab-settings');
  assertAtLeast(settings.minWidth, 28, '.chat-tab-settings min-width');
  const gear = settingsBtn.querySelector('svg');
  assert.ok(gear, 'settings button must contain the gear svg');
  assertAtLeast(gear.getAttribute('width'), 18, 'settings gear svg width');
  assertAtLeast(gear.getAttribute('height'), 18, 'settings gear svg height');
  win.close();
  console.log('  ok - tab-bar "+" and settings buttons are bigger');
}

function testCloseButtonsAreBigger() {
  const {win} = makeWebview();
  const doc = win.document;

  assert.ok(
    doc.querySelector('.chat-tab-close'),
    'tab bar must render a tab-header X button',
  );
  const tabClose = cssRule(win, '.chat-tab-close');
  assert.ok(tabClose, 'main.css must style .chat-tab-close');
  assertAtLeast(tabClose.width, 20, '.chat-tab-close width');
  assertAtLeast(tabClose.height, 20, '.chat-tab-close height');
  assertAtLeast(tabClose.fontSize, 16, '.chat-tab-close font-size');

  const inputClear = cssRule(win, '#input-clear-btn');
  assert.ok(inputClear, 'main.css must style #input-clear-btn');
  assertAtLeast(inputClear.width, 16, '#input-clear-btn width');
  assertAtLeast(inputClear.height, 16, '#input-clear-btn height');
  assertAtLeast(inputClear.fontSize, 16, '#input-clear-btn font-size');

  const searchClear = cssRule(win, '.search-clear-btn');
  assert.ok(searchClear, 'main.css must style .search-clear-btn');
  assertAtLeast(searchClear.width, 18, '.search-clear-btn width');
  assertAtLeast(searchClear.height, 18, '.search-clear-btn height');
  assertAtLeast(searchClear.fontSize, 16, '.search-clear-btn font-size');

  win.__kissShowTipsPanel(['# Tip 1\nHello']);
  const panel = doc.querySelector('kiss-tips-panel');
  assert.ok(panel && panel.shadowRoot, 'tips panel must mount a shadow root');
  const shadowCss = panel.shadowRoot.querySelector('style').textContent;
  const m = shadowCss.match(/\.tips-close\s*\{[^}]*font-size:\s*([0-9.]+)px/);
  assert.ok(m, 'tips.js must give .tips-close an explicit px font-size');
  assertAtLeast(m[1], 26, '.tips-close font-size');
  win.close();
  console.log('  ok - all X buttons are bigger');
}

function clickAndAssertBlurred(win, el, what) {
  if (el.tabIndex < 0) el.tabIndex = 0;
  el.focus();
  assert.strictEqual(
    win.document.activeElement,
    el,
    what + ' must be focusable in this harness (test setup)',
  );
  el.dispatchEvent(
    new win.MouseEvent('click', {bubbles: true, cancelable: true}),
  );
  assert.notStrictEqual(
    win.document.activeElement,
    el,
    what + ' must not keep focus after being clicked',
  );
}

function testButtonsLoseFocusAfterClick() {
  const {win} = makeWebview();
  const doc = win.document;

  for (const id of [
    'send-btn',
    'stop-btn',
    'menu-btn',
    'model-btn',
    'upload-btn',
    'tricks-btn',
    'voice-btn',
    'input-clear-btn',
    'model-search-clear',
    'history-search-clear',
  ]) {
    const el = doc.getElementById(id);
    assert.ok(el, '#' + id + ' must exist');
    el.disabled = false;
    clickAndAssertBlurred(win, el, '#' + id);
  }

  for (const sel of [
    '.chat-tab-add',
    '.chat-tab-settings',
    '.chat-tab-close',
  ]) {
    const el = doc.querySelector(sel);
    assert.ok(el, sel + ' must exist in the tab bar');
    clickAndAssertBlurred(win, el, sel);
  }
  win.close();
  console.log('  ok - buttons lose focus after click');
}

function testTipsCloseLosesFocusAndClosesPanel() {
  const {win} = makeWebview();
  const doc = win.document;

  win.__kissShowTipsPanel(['# Tip 1\nHello']);
  const panel = doc.querySelector('kiss-tips-panel');
  assert.ok(panel && panel.shadowRoot, 'tips panel must mount');
  const close = panel.shadowRoot.querySelector('.tips-close');
  assert.ok(close, 'tips panel must have a .tips-close button');
  close.focus();
  close.dispatchEvent(
    new win.MouseEvent('click', {bubbles: true, cancelable: true}),
  );
  assert.strictEqual(
    doc.querySelector('kiss-tips-panel'),
    null,
    'clicking the tips X must remove the panel',
  );
  assert.notStrictEqual(
    doc.activeElement,
    close,
    'the tips X must not keep focus after being clicked',
  );
  win.close();
  console.log('  ok - tips X closes the panel without keeping focus');
}

function main() {
  console.log('buttonSizeAndBlur.test.js');
  testInputFooterButtonsAreBigger();
  testTabBarPlusAndSettingsAreBigger();
  testCloseButtonsAreBigger();
  testButtonsLoseFocusAfterClick();
  testTipsCloseLosesFocusAndClosesPanel();
  console.log('all ok');
}

main();
