// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const {JSDOM} = require('jsdom');

function makeDomWebview() {
  const mediaDir = path.join(__dirname, '..', 'media');
  let html = fs.readFileSync(path.join(mediaDir, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
  html = html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');

  const dom = new JSDOM(html, {
    runScripts: 'outside-only',
    pretendToBeVisual: true,
    url: 'https://localhost/',
  });
  const win = dom.window;
  win.Element.prototype.scrollIntoView = function () {};
  win.Element.prototype.scrollTo = function () {};
  win.HTMLElement.prototype.scrollTo = function () {};

  const posted = [];
  win.acquireVsCodeApi = function () {
    return {
      postMessage: msg => posted.push(msg),
      getState: () => undefined,
      setState: () => {},
    };
  };
  vm.runInContext(
    fs.readFileSync(path.join(mediaDir, 'panelCopy.js'), 'utf8'),
    dom.getInternalVMContext(),
  );
  vm.runInContext(
    fs.readFileSync(path.join(mediaDir, 'api.js'), 'utf8'),
    dom.getInternalVMContext(),
  );
  vm.runInContext(
    fs.readFileSync(path.join(mediaDir, 'main.js'), 'utf8'),
    dom.getInternalVMContext(),
  );
  return {win, posted, close: () => win.close()};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function click(el) {
  el.dispatchEvent(
    new el.ownerDocument.defaultView.MouseEvent('click', {bubbles: true}),
  );
}

async function waitFor(predicate, message) {
  for (let i = 0; i < 100; i++) {
    const value = predicate();
    if (value) return value;
    await new Promise(r => setTimeout(r, 20));
  }
  throw new Error(message || 'waitFor timed out');
}

async function waitForFalsy(predicate, message) {
  for (let i = 0; i < 100; i++) {
    if (!predicate()) return;
    await new Promise(r => setTimeout(r, 20));
  }
  throw new Error(message || 'waitForFalsy timed out');
}

async function runTests() {
  const wv = makeDomWebview();
  try {
    send(wv.win, {
      type: 'update_available',
      available: true,
      latest: '9999.0.0',
      current: '2026.6.29',
    });

    const toast = await waitFor(
      () => wv.win.document.querySelector('.kiss-notification'),
      'update_available with available:true must render a permanent webview notification',
    );

    assert.strictEqual(
      toast.getAttribute('data-notification-sticky'),
      'true',
      'update notification must be marked sticky on the DOM element',
    );

    const actionButton = toast.querySelector('.kiss-notification-action');
    assert.ok(
      actionButton,
      'update notification must expose an action button users can click',
    );
    const svg = actionButton.querySelector('svg');
    assert.ok(
      svg,
      'update notification action button must contain an inline <svg> icon',
    );
    assert.strictEqual(
      svg.namespaceURI,
      'http://www.w3.org/2000/svg',
      'action-button SVG must be in the SVG namespace',
    );
    const ariaLabel = (
      actionButton.getAttribute('aria-label') ||
      actionButton.textContent ||
      ''
    ).toLowerCase();
    assert.ok(
      ariaLabel.includes('update'),
      'action button must advertise itself as an update action',
    );

    click(actionButton);
    await waitFor(
      () => wv.posted.some(m => m.type === 'runUpdate'),
      'clicking the update notification button must post {type: "runUpdate"}',
    );

    send(wv.win, {
      type: 'update_available',
      available: false,
      latest: '',
      current: '',
    });
    await waitForFalsy(
      () => wv.win.document.querySelector('.kiss-notification'),
      'available:false broadcast must dismiss the permanent update notification',
    );

    send(wv.win, {
      type: 'update_available',
      available: true,
      latest: '9999.0.0',
      current: '2026.6.29',
    });
    await waitFor(
      () => wv.win.document.querySelector('.kiss-notification'),
      'second broadcast must re-create the permanent notification',
    );
    send(wv.win, {
      type: 'update_available',
      available: true,
      latest: '9999.0.0',
      current: '2026.6.29',
    });
    const toasts = wv.win.document.querySelectorAll('.kiss-notification');
    assert.strictEqual(
      toasts.length,
      1,
      'repeated update_available broadcasts must not stack duplicate notifications',
    );
  } finally {
    wv.close();
  }
}

runTests().then(
  () => {
    console.log('\nAll tests passed');
    process.exit(0);
  },
  err => {
    console.error('FAIL:', err && err.stack ? err.stack : err);
    process.exit(1);
  },
);
