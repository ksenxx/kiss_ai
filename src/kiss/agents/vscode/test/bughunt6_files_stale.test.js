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
fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));

  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function setInput(win, text) {
  const inp = win.document.getElementById('task-input');
  inp.value = text;
  inp.setSelectionRange(text.length, text.length);
  return inp;
}

function pickerDisplay(win) {
  return win.document.getElementById('autocomplete').style.display;
}

const FILES = [
  {type: 'file', text: 'src/util.py'},
  {type: 'file', text: 'src/utils/helpers.py'},
];

function testLateFilesEventWithoutAtContextStaysHidden() {
  const {win} = makeWebview();
  setInput(win, 'fix the bug');

  send(win, {type: 'files', files: FILES});

  assert.notStrictEqual(
    pickerDisplay(win),
    'block',
    'a late files event must NOT open the picker when no @-mention ' +
      'is being typed (the phantom picker swallows the next Enter)',
  );
}

function testStalePrefixFilesEventIgnored() {
  const {win} = makeWebview();
  setInput(win, '@util');

  send(win, {type: 'files', prefix: 'u', files: FILES});

  assert.notStrictEqual(
    pickerDisplay(win),
    'block',
    'a files reply for a stale prefix must not open the picker',
  );
}

function testMatchingPrefixFilesEventRenders() {
  const {win} = makeWebview();
  setInput(win, '@util');

  send(win, {type: 'files', prefix: 'util', files: FILES});

  assert.strictEqual(
    pickerDisplay(win),
    'block',
    'a files reply matching the typed @-prefix must open the picker',
  );
  const text = win.document.getElementById('autocomplete').textContent;
  assert.ok(text.includes('util.py'), 'picker must list the files');
}

function testPrefixlessFilesEventStillRendersWithAtContext() {
  const {win} = makeWebview();
  setInput(win, '@ut');

  send(win, {type: 'files', files: FILES});

  assert.strictEqual(
    pickerDisplay(win),
    'block',
    'a prefix-less files reply must render while an @-mention is active',
  );
}

const tests = [
  testLateFilesEventWithoutAtContextStaysHidden,
  testStalePrefixFilesEventIgnored,
  testMatchingPrefixFilesEventRenders,
  testPrefixlessFilesEventStillRendersWithAtContext,
];

let failed = 0;
for (const t of tests) {
  try {
    t();
    console.log('PASS', t.name);
  } catch (err) {
    failed += 1;
    console.error('FAIL', t.name);
    console.error(err && err.stack ? err.stack : err);
  }
}
if (failed) {
  console.error(failed + ' test(s) failed');
  process.exit(1);
}
console.log('All tests passed');
