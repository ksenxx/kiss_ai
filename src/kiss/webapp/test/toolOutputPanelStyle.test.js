// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end tests for two styling requirements of the chat page:
//
// 1. Panels showing tool call output (.bash-panel) paint the shared
//    neutral panel tint (the translucent --panel-tint) standalone;
//    nested inside a .tc they are a section of the open call's card,
//    which paints that tint once behind all its sections, so the
//    nested panel has no fill of its own (two layers of the
//    translucent tint would render darker).  (The thinking tokens
//    being plain italic text inside the Thoughts panel is pinned by
//    transcriptPanelAlignment.test.js.)
// 2. The model picker button (#model-btn) width caps at
//    clamp(72px, 21vw, 220px).

'use strict';

const assert = require('assert');
const {
  makePage: makeChatPage,
  send,
  loadRemoteCss,
  readyTabId,
} = require('./simplify2_harness');

// The design tokens are inlined, so the two tints compare as the
// concrete colour expressions main.css defines.
function makePage() {
  return makeChatPage({
    bodyClass: 'remote-chat',
    beforeScripts: w => loadRemoteCss(w, true),
  });
}

// Replays a transcript containing a Bash tool call (whose streaming
// output panel nests inside the .tc card) and a bare tool_result with
// no preceding tool call (a standalone output panel).
function replayPanels(win, posted) {
  send(win, {
    type: 'task_events',
    tabId: readyTabId(posted),
    chat_id: 'chat-style',
    task: 'style parity task',
    events: [
      {type: 'tool_result', content: 'orphan output', is_error: false},
      {type: 'tool_call', name: 'Bash', command: 'ls', ts: 2000},
      {type: 'tool_result', content: 'file-a\nfile-b', is_error: false},
    ],
  });
}

function testToolOutputPaintsThePanelTint() {
  const {win, posted} = makePage();
  replayPanels(win, posted);
  const d = win.document;

  const standalone = d.querySelector('#output > .bash-panel');
  assert.ok(standalone, 'the bare tool_result must render a standalone panel');
  const tint = win.getComputedStyle(standalone).background;
  assert.ok(
    tint.includes('transparent'),
    `a standalone tool output panel paints the translucent panel tint: ${tint}`,
  );

  const nested = d.querySelector('.tc > .bash-panel');
  assert.ok(nested, 'the Bash tool call must nest an output panel in its .tc');
  // The open .tc paints the tint behind its sections (its ::before, a
  // pseudo-element jsdom cannot compute, so that rule is read from the
  // parsed stylesheet); the nested panel adds no second layer.
  assert.strictEqual(
    win.getComputedStyle(nested).backgroundColor,
    'rgba(0, 0, 0, 0)',
    'a nested tool output panel has no fill of its own',
  );
  let card = null;
  for (const sheet of d.styleSheets) {
    for (const rule of sheet.cssRules) {
      const selectors = (rule.selectorText || '').split(',');
      if (selectors.some(s => s.trim() === '.tc:not(.collapsed)::before'))
        card = rule;
    }
  }
  assert.ok(card, 'main.css declares the open tool call card');
  assert.strictEqual(
    card.style.background,
    tint,
    'the open tool call card paints the standalone tint',
  );
  win.close();
}

function testModelPillWidthCaps() {
  const page = makePage();
  assert.strictEqual(
    page.win.getComputedStyle(page.win.document.getElementById('model-btn'))
      .maxWidth,
    'clamp(72px, 21vw, 220px)',
    'the pill cap must be clamp(72px, 21vw, 220px)',
  );
  page.win.close();
}

function runTests() {
  const tests = [testToolOutputPaintsThePanelTint, testModelPillWidthCaps];
  for (const t of tests) {
    t();
    console.log('PASS', t.name);
  }
}

try {
  runTests();
  console.log('\nAll tests passed');
} catch (err) {
  console.error('FAIL:', err && err.message ? err.message : err);
  process.exit(1);
}
