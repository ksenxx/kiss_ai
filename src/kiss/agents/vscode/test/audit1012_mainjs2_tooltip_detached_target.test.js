// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// Audit 2026-10-12 (mainjs-2): the [data-tooltip] hover timer in
// media/main.js (showTooltipFor, armed 400 ms after mouseover) keeps a
// reference to the hovered element and shows its tooltip even when the
// element has left the document meanwhile.  History rows carry their
// preview as data-tooltip and the list is rebuilt on every `history`
// reply (one per status / tasks_updated), so hovering a row while a
// task runs replaces it under the pointer: the timer then measures a
// detached node (a 0x0 rect at the origin) and parks the tooltip in the
// top-left corner of the window, where it stays -- the removed element
// never fires the mouseout that would hide it.

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
  win.acquireVsCodeApi = function () {
    let state;
    return {
      postMessage: () => {},
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };
  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
  return win;
}

function wait(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function testRemovedTargetShowsNoTooltip() {
  const win = makeWebview();
  const row = win.document.createElement('div');
  row.className = 'sidebar-item';
  row.dataset.tooltip = 'the task preview';
  win.document.getElementById('history-list').appendChild(row);
  row.dispatchEvent(new win.MouseEvent('mouseover', {bubbles: true}));
  // The history re-render replaces the row before the 400 ms timer.
  row.remove();
  await wait(500);
  const tip = win.document.getElementById('custom-tooltip');
  assert.ok(
    !tip.classList.contains('visible'),
    'no tooltip for an element that left the document (got "' +
      tip.textContent + '" at ' + tip.style.left + ',' + tip.style.top + ')',
  );
  win.close();
  console.log('  ok - a removed hover target shows no tooltip');
}

async function testConnectedTargetStillShowsItsTooltip() {
  const win = makeWebview();
  const btn = win.document.getElementById('new-chat-btn');
  btn.dispatchEvent(new win.MouseEvent('mouseover', {bubbles: true}));
  await wait(500);
  const tip = win.document.getElementById('custom-tooltip');
  assert.ok(tip.classList.contains('visible'), 'tooltip shown');
  assert.strictEqual(tip.textContent, btn.dataset.tooltip);
  btn.dispatchEvent(new win.MouseEvent('mouseout', {bubbles: true}));
  assert.ok(!tip.classList.contains('visible'), 'hidden on mouseout');
  win.close();
  console.log('  ok - a live hover target still shows its tooltip');
}

async function runTests() {
  await testRemovedTargetShowsNoTooltip();
  await testConnectedTargetStillShowsItsTooltip();
}

runTests().then(
  () => {
    console.log('\n2 passed, 0 failed');
    process.exit(0);
  },
  err => {
    console.error('FAIL:', err && err.stack ? err.stack : err);
    process.exit(1);
  },
);
