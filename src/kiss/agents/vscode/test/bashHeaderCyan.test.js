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

  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));

  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(
fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function findBashHeader(win) {
  const headers = win.document.querySelectorAll('#output .tc-h');
  for (const h of headers) {
    const txt = (h.textContent || '').replace(/^[^A-Za-z]+/, '').trim();
    if (txt === 'Bash') return h;
  }
  return null;
}

function testBashHeaderHasBashMarkerClass() {
  const {win} = makeWebview();
  send(win, {
    type: 'tool_call',
    name: 'Bash',
    command: 'ls -la',
    description: 'list files',
  });
  const hdr = findBashHeader(win);
  assert.ok(hdr, 'Bash tool_call must render a .tc-h header with text "Bash"');
  assert.ok(
    hdr.classList.contains('tc-h-bash'),
    'BUG: Bash tool_call header must carry a "tc-h-bash" CSS hook ' +
      'so the stylesheet can paint it cyan — got classes: ' +
      JSON.stringify(Array.from(hdr.classList)),
  );
  win.close();
  console.log('  ok - Bash tool_call header has tc-h-bash class');
}

function testNonBashHeaderDoesNotHaveBashMarker() {
  const {win} = makeWebview();
  send(win, {
    type: 'tool_call',
    name: 'Read',
    path: '/tmp/x.txt',
  });
  const headers = Array.from(
    win.document.querySelectorAll('#output .tc-h'),
  );
  const read = headers.find(
    h => (h.textContent || '').replace(/^[^A-Za-z]+/, '').trim() === 'Read',
  );
  assert.ok(read, 'Read tool_call must render a .tc-h header');
  assert.ok(
    !read.classList.contains('tc-h-bash'),
    'non-Bash tool_call headers must NOT carry the tc-h-bash CSS hook',
  );
  win.close();
  console.log('  ok - non-Bash tool_call header has no tc-h-bash class');
}

function testCssDeclaresCyanRuleForBashHeader() {
  const css = fs.readFileSync(path.join(MEDIA, 'main.css'), 'utf8');
  const blockRe = /([^{}]*tc-h-bash[^{}]*)\{([^}]*)\}/g;
  let match;
  let foundCyanRule = false;
  while ((match = blockRe.exec(css)) !== null) {
    const body = match[2];
    if (/color\s*:\s*var\(\s*--cyan\s*\)/.test(body)) {
      foundCyanRule = true;
      break;
    }
  }
  assert.ok(
    foundCyanRule,
    'BUG: main.css must declare a "tc-h-bash" selector with ' +
      '"color: var(--cyan)" so the Bash tool_call header renders ' +
      'in the cyan theme color',
  );
  console.log('  ok - main.css colours .tc-h-bash with var(--cyan)');
}

function testBashHeaderComputedStyleIsCyan() {
  const {win} = makeWebview();
  const css = fs.readFileSync(path.join(MEDIA, 'main.css'), 'utf8');
  const styleEl = win.document.createElement('style');
  styleEl.textContent = css;
  win.document.head.appendChild(styleEl);

  send(win, {type: 'tool_call', name: 'Bash', command: 'echo hi'});
  const hdr = findBashHeader(win);
  assert.ok(hdr, 'Bash tool_call header must exist');
  const computed = win.getComputedStyle(hdr);
  const color = computed.getPropertyValue('color').trim();
  assert.ok(
    hdr.classList.contains('tc-h-bash'),
    'Bash header lost its tc-h-bash hook when main.css was injected',
  );
  assert.ok(
    !/255,?\s*125|^orange$/i.test(color),
    'Bash header computed color must not be an orange literal: ' +
      JSON.stringify(color),
  );
  win.close();
  console.log('  ok - Bash header computed style picks up the cyan rule');
}

function runTests() {
  testBashHeaderHasBashMarkerClass();
  testNonBashHeaderDoesNotHaveBashMarker();
  testCssDeclaresCyanRuleForBashHeader();
  testBashHeaderComputedStyleIsCyan();
}

try {
  runTests();
  console.log('\n4 passed, 0 failed');
  process.exit(0);
} catch (err) {
  console.error('FAIL:', err && err.stack ? err.stack : err);
  process.exit(1);
}
