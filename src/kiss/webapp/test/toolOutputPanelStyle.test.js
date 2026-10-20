// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end tests for two styling requirements of the chat page:
//
// 1. Panels showing tool call output (.bash-panel) paint the shared
//    neutral panel tint: the translucent --panel-tint standalone, and
//    the same tint pre-mixed over --bg (--panel-tint-solid) when nested
//    inside a .tc card (whose opaque --surface would otherwise make the
//    translucent tint render lighter).  The model's thinking tokens are
//    plain text inside the Thoughts panel, with no box of their own.
// 2. The model picker pill (#model-btn) width caps at
//    clamp(72px, 21vw, 220px).

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {
  makePage: makeChatPage,
  send,
  loadMainCss,
  MEDIA,
} = require('./simplify2_harness');

const CSS = fs.readFileSync(path.join(MEDIA, 'main.css'), 'utf8');

function makePage() {
  return makeChatPage({
    bodyClass: 'remote-chat',
    sourceUrl: 'toolstyle-main.js',
    beforeScripts(win) {
      loadMainCss(win);
      const remoteStyle = win.document.createElement('style');
      remoteStyle.textContent = fs.readFileSync(
        path.join(MEDIA, 'remote-codex.css'),
        'utf8',
      );
      win.document.head.appendChild(remoteStyle);
    },
  });
}

function readyTabId(posted) {
  const ready = posted.find(m => m.type === 'ready');
  assert.ok(ready, 'main.js must post a ready message');
  return ready.tabId;
}

// Replays a transcript containing a thought, a Bash tool call
// (whose streaming output panel nests inside the .tc card), and a bare
// tool_result with no preceding tool call (a standalone output panel).
function replayPanels(win, posted) {
  send(win, {
    type: 'task_events',
    tabId: readyTabId(posted),
    chat_id: 'chat-style',
    task: 'style parity task',
    events: [
      {type: 'thinking_start', ts: 1000},
      {type: 'thinking_delta', text: 'pondering'},
      {type: 'thinking_end'},
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

  const think = d.querySelector('.llm-panel > .think');
  assert.ok(think, 'the thought is a text block inside the Thoughts panel');
  assert.strictEqual(think.textContent, 'pondering');
  const thinkStyle = win.getComputedStyle(think);
  assert.strictEqual(
    thinkStyle.background,
    '',
    'the thinking text paints no box of its own',
  );
  assert.strictEqual(
    thinkStyle.borderStyle,
    'none',
    'the thinking text draws no border',
  );
  assert.strictEqual(
    thinkStyle.fontStyle,
    'italic',
    'the thinking text is italic',
  );
  assert.ok(
    !think.querySelector('.lbl'),
    'the thinking text has no "Thinking" header',
  );

  const standalone = d.querySelector('#output > .bash-panel');
  assert.ok(standalone, 'the bare tool_result must render a standalone panel');
  assert.strictEqual(
    win.getComputedStyle(standalone).background,
    'var(--panel-tint)',
    'a standalone tool output panel paints the neutral panel tint',
  );

  const nested = d.querySelector('.tc > .bash-panel');
  assert.ok(nested, 'the Bash tool call must nest an output panel in its .tc');
  // The .tc card paints the opaque --surface behind its children, so the
  // nested panel paints the same tint pre-mixed over --bg (what a
  // standalone panel sits on) instead of relying on transparency.  jsdom
  // leaves var() unresolved, so compare the token definitions in the
  // :root palette of main.css.
  assert.strictEqual(
    win.getComputedStyle(nested).background,
    'var(--panel-tint-solid)',
    'a nested tool output panel must paint the pre-mixed panel tint',
  );
  const root = CSS.match(/:root\s*\{([^}]*)\}/)[1];
  const tint = /--panel-tint:\s*([^;]+);/.exec(root)[1].trim();
  const solid = /--panel-tint-solid:\s*([^;]+);/.exec(root)[1].trim();
  assert.strictEqual(
    solid,
    tint.replace('transparent', 'var(--bg)'),
    '--panel-tint-solid must be --panel-tint mixed over --bg',
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
  process.exit(0);
} catch (err) {
  console.error('FAIL:', err && err.message ? err.message : err);
  process.exit(1);
}
