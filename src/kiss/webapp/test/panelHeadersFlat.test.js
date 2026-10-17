// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// Event panel headers are flat: no chevron in front of the title, no
// background and no border of their own (the panel keeps its single
// outer border).  The panels are rendered by the real media/main.js
// with media/main.css injected, so the assertions are on the cascaded
// values the page actually paints.  Where the panels sit (the user's
// on the right, the rest on the left) is transcriptPanelAlignment's.

const assert = require('assert');
const harness = require('./simplify2_harness');

/** The harness page with main.css applied; `win.__posted` is its ledger. */
function makePage() {
  const {win, posted} = harness.makePage();
  harness.loadMainCss(win);
  win.__posted = posted;
  return win;
}

const send = harness.send;

function renderTranscript(win) {
  const ready = win.__posted.find(m => m.type === 'ready');
  assert.ok(ready && ready.tabId, 'the page posts ready with a tabId');
  send(win, {
    type: 'task_events',
    events: [],
    task: 'Flatten the headers',
    tabId: ready.tabId,
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
}

function testNoChevronInAnyHeader() {
  const win = makePage();
  renderTranscript(win);
  const out = win.document.getElementById('output');
  assert.strictEqual(out.querySelectorAll('.collapse-chv').length, 0, 'no chevron span');
  // The thinking tokens are plain text inside the Thoughts panel: no
  // header of their own to flatten.
  assert.strictEqual(out.querySelectorAll('.think .lbl').length, 0, 'no thinking header');
  assert.ok(out.querySelector('.llm-panel > .think'), 'the thinking text block sits in the Thoughts panel');
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
  win.close();
  console.log('  ok - no event panel header starts with a chevron');
}

function testToolCallHeaderIsFlat() {
  const win = makePage();
  renderTranscript(win);
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
  // is read from the parsed stylesheet rule instead.
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
  assert.ok(/border:\s*1px solid/.test(tcRule.cssText), 'the panel keeps its outer border: ' + tcRule.cssText);
  win.close();
  console.log('  ok - the tool call header has no background and no border of its own');
}

function runTests() {
  testNoChevronInAnyHeader();
  testToolCallHeaderIsFlat();
}

try {
  runTests();
  console.log('\n2 passed, 0 failed');
  process.exit(0);
} catch (err) {
  console.error('FAIL:', err && err.stack ? err.stack : err);
  process.exit(1);
}
