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
  const {win} = makeWebview();
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

function testNoCollapsibleHeaderIsBold() {
  const css = fs.readFileSync(path.join(MEDIA, 'main.css'), 'utf8');
  const headerSelectors = [
    '.tc-h',
    '.tr .rl',
    '.system-prompt-h, .prompt-h',
    '.llm-panel-hdr',
    '.ask-answer-label',
  ];
  for (const sel of headerSelectors) {
    const re = new RegExp(
      '\\n' + sel.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + '\\s*\\{([^}]*)\\}',
    );
    const m = re.exec(css);
    assert.ok(m, `main.css declares ${sel}`);
    assert.ok(
      /font-weight:\s*400/.test(m[1]),
      `${sel} is font-weight 400, got: ${m[1].trim()}`,
    );
  }
  assert.ok(
    !/tc-h-bash[^{]*\{[^}]*--accent/.test(css),
    'no rule paints .tc-h-bash in the accent',
  );
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
