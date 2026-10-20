// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {makePage, send} = require('./simplify2_harness');

const MEDIA = path.join(__dirname, '..', 'media');

// Every transcript panel header shares ONE quiet style: the Bash tool
// call is no longer singled out in the accent blue, and no collapsible
// header is bold.  main.css is injected into the JSDOM page so the
// assertions are on the cascaded values main.js's panels actually get.

function injectMainCss(win) {
  const styleEl = win.document.createElement('style');
  styleEl.textContent = fs.readFileSync(path.join(MEDIA, 'main.css'), 'utf8');
  win.document.head.appendChild(styleEl);
}

/** The .tc-h header whose tool name (its .tc-h-name span) is *name*. */
function headerNamed(win, name) {
  const headers = win.document.querySelectorAll('#output .tc-h');
  for (const h of headers) {
    const nameEl = h.querySelector('.tc-h-name');
    const txt = ((nameEl || h).textContent || '').replace(/^[^A-Za-z]+/, '').trim();
    if (txt === name) return h;
  }
  return null;
}

function testBashHeaderMatchesTheOtherToolHeaders() {
  const {win} = makePage();
  injectMainCss(win);
  send(win, {type: 'status', running: true});
  send(win, {type: 'tool_call', name: 'Bash', command: 'ls -la', description: 'list files'});
  send(win, {type: 'tool_call', name: 'Read', path: '/tmp/x.txt'});
  const bash = headerNamed(win, 'Bash');
  const read = headerNamed(win, 'Read');
  assert.ok(bash && read, 'both tool calls render a .tc-h header');
  assert.ok(bash.classList.contains('tc-h-bash'), 'the Bash header keeps its tool hook');
  assert.ok(!read.classList.contains('tc-h-bash'), 'only the Bash header has it');
  const b = win.getComputedStyle(bash);
  const r = win.getComputedStyle(read);
  for (const prop of ['color', 'background-color', 'font-weight']) {
    assert.strictEqual(
      b.getPropertyValue(prop),
      r.getPropertyValue(prop),
      `the Bash header's ${prop} equals every other tool header's`,
    );
  }
  assert.strictEqual(b.getPropertyValue('font-weight'), '400', 'headers are not bold');
  assert.ok(
    !/var\(\s*--accent/.test(b.getPropertyValue('color')),
    'the Bash header is not painted in the accent: ' + b.getPropertyValue('color'),
  );
  win.close();
  console.log('  ok - the Bash header matches the other tool headers');
}

// Every collapsible header the transcript renders, by its markup
// (`.tr .rl` is the result label inside a tool-result panel).
const HEADER_MARKUP = [
  '<div class="tc-h">Read</div>',
  '<div class="tr"><span class="rl">Result</span></div>',
  '<div class="system-prompt-h">System prompt</div>',
  '<div class="prompt-h">Prompt</div>',
  '<div class="llm-panel-hdr">Thoughts</div>',
  '<span class="ask-answer-label">Answer</span>',
  '<div class="tc-h tc-h-bash">Bash</div>',
];

function testNoCollapsibleHeaderIsBold() {
  const {win} = makePage();
  injectMainCss(win);
  const output = win.document.getElementById('output');
  for (const markup of HEADER_MARKUP) {
    output.insertAdjacentHTML('beforeend', markup);
    const el = output.lastElementChild.matches('.tr')
      ? output.lastElementChild.firstElementChild
      : output.lastElementChild;
    const style = win.getComputedStyle(el);
    assert.strictEqual(
      style.getPropertyValue('font-weight'),
      '400',
      `${markup} is not bold`,
    );
    assert.ok(
      !/var\(\s*--accent/.test(style.getPropertyValue('color')),
      `${markup} is not painted in the accent: ${style.getPropertyValue('color')}`,
    );
  }
  win.close();
  console.log('  ok - no collapsible panel header is bold');
}

function runTests() {
  testBashHeaderMatchesTheOtherToolHeaders();
  testNoCollapsibleHeaderIsBold();
}

try {
  runTests();
  console.log('\n2 passed, 0 failed');
  process.exit(0);
} catch (err) {
  console.error('FAIL:', err && err.stack ? err.stack : err);
  process.exit(1);
}
