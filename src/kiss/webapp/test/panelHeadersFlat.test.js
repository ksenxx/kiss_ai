// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// Event panel headers are flat and uniform: no chevron in front of the
// title, no background and no border of their own (the panel keeps its
// single outer border), and ONE quiet style for every header (the Bash
// tool call is not singled out in the accent blue, no collapsible header
// is bold).  The panels are rendered by the real media/main.js with
// media/main.css injected, so the assertions are on the cascaded values
// the page actually paints.  Where the panels sit (the user's on the
// right, the rest on the left) is transcriptPanelAlignment's.

const assert = require('assert');
const harness = require('./simplify2_harness');

const send = harness.send;

/**
 * One harness page with main.css applied and a transcript rendered: a
 * system prompt, a prompt, a thought and two tool calls.  Every test
 * reads (or folds and unfolds) the same headers, so they share it.
 */
function renderTranscript() {
  const {win, posted} = harness.makePage();
  harness.loadMainCss(win);
  send(win, {
    type: 'task_events',
    events: [],
    task: 'Flatten the headers',
    tabId: harness.readyTabId(posted),
    chat_id: 'chat-flat-headers',
  });
  send(win, {type: 'status', running: true});
  send(win, {type: 'system_prompt', text: 'You are a test.'});
  send(win, {type: 'prompt', text: 'Flatten the headers'});
  send(win, {type: 'thinking_start'});
  send(win, {type: 'thinking_delta', text: 'first thought'});
  send(win, {type: 'thinking_end'});
  send(win, {type: 'tool_call', name: 'Bash', command: 'ls', description: 'list'});
  send(win, {type: 'tool_call', name: 'Read', path: '/tmp/x.txt'});
  return win;
}

function testNoChevronInAnyHeader(win) {
  const out = win.document.getElementById('output');
  assert.strictEqual(out.querySelectorAll('.collapse-chv').length, 0, 'no chevron span');
  // The thinking text having no header of its own is
  // ui_antipattern_keyboard's.
  const headers = out.querySelectorAll('.collapse-header');
  assert.ok(headers.length >= 4, 'the transcript rendered its headers: ' + headers.length);
  for (const h of headers) {
    assert.ok(
      !/^[\s\u25B8\u25BE\u25B6\u25BC]/.test(h.textContent),
      'the header title starts with its text, not a glyph: ' + JSON.stringify(h.textContent),
    );
  }
  // Folding works without the chevron: the header is the control.  A
  // tool panel starts folded; a click unfolds it, the next folds it.
  const bash = out.querySelector('.tc.collapsible');
  const hdr = bash.querySelector('.collapse-header');
  assert.ok(bash.classList.contains('collapsed'), 'a tool panel starts folded');
  assert.strictEqual(hdr.getAttribute('aria-expanded'), 'false');
  assert.ok(
    !/Bash/.test(hdr.querySelector('.collapse-preview').textContent),
    'the collapsed preview summarizes the body, not the header itself',
  );
  hdr.click();
  assert.ok(!bash.classList.contains('collapsed'), 'a header click unfolds it');
  assert.strictEqual(hdr.getAttribute('aria-expanded'), 'true');
  hdr.click();
  assert.ok(bash.classList.contains('collapsed'), 'a second click folds it');
  console.log('  ok - no event panel header starts with a chevron');
}

function testToolCallHeaderIsFlat(win) {
  const out = win.document.getElementById('output');
  const flat = ['.tc-h', '.system-prompt-h', '.prompt-h'];
  for (const sel of flat) {
    const h = out.querySelector(sel);
    assert.ok(h, sel + ' rendered');
    const cs = win.getComputedStyle(h);
    assert.strictEqual(cs.getPropertyValue('background-color'), 'rgba(0, 0, 0, 0)', sel + ' paints no background');
    for (const side of ['top', 'right', 'bottom', 'left']) {
      assert.strictEqual(cs.getPropertyValue('border-' + side + '-style'), 'none', sel + ' has no ' + side + ' border');
    }
  }
  // JSDOM drops a `border: 1px solid var(--border)` shorthand from the
  // computed style (var() in a shorthand), so the panel's outer border
  // is read from the parsed stylesheet rule's CSSOM style instead.
  let tcRule = null;
  for (const sheet of win.document.styleSheets) {
    for (const rule of sheet.cssRules) {
      // .tc shares its frame rule with the other panel frames: the
      // selector list names them all.
      if (
        rule.selectorText &&
        rule.selectorText.split(',').some(x => x.trim() === '.tc')
      )
        tcRule = rule;
    }
  }
  assert.ok(tcRule, 'main.css declares .tc');
  assert.strictEqual(
    tcRule.style.border,
    '1px solid var(--border)',
    'the panel keeps its outer border in the --border token',
  );
  console.log('  ok - the tool call header has no background and no border of its own');
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

function testBashHeaderMatchesTheOtherToolHeaders(win) {
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

function testNoCollapsibleHeaderIsBold(win) {
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
  console.log('  ok - no collapsible panel header is bold');
}

function runTests() {
  const win = renderTranscript();
  testToolCallHeaderIsFlat(win);
  testBashHeaderMatchesTheOtherToolHeaders(win);
  testNoCollapsibleHeaderIsBold(win);
  // Last: its clicks leave the Bash header in jsdom's :hover colour.
  testNoChevronInAnyHeader(win);
  win.close();
}

try {
  runTests();
  console.log('\npanelHeadersFlat.test.js: all tests passed');
  process.exit(0);
} catch (err) {
  console.error('FAIL:', err && err.stack ? err.stack : err);
  process.exit(1);
}
