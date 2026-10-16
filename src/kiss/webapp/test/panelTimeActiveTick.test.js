// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const {makePage, send, waitFor} = require('./simplify2_harness');

function panelTimeMs(panel) {
  const footer = panel.querySelector(':scope > .panel-time');
  return footer ? parsePanelTimeMs(footer.textContent) : NaN;
}

function parsePanelTimeMs(text) {
  const t = String(text || '').trim();
  let m = t.match(/^(\d+)m\s+(\d+(?:\.\d+)?)s$/);
  if (m) return (parseInt(m[1], 10) * 60 + parseFloat(m[2])) * 1000;
  m = t.match(/^(\d+(?:\.\d+)?)s$/);
  if (m) return parseFloat(m[1]) * 1000;
  m = t.match(/^(\d+)ms$/);
  if (m) return parseInt(m[1], 10);
  return NaN;
}

async function runTests() {
  const wv = makePage();
  const win = wv.win;
  const ready = wv.posted.find(m => m.type === 'ready');
  assert.ok(ready && ready.tabId, 'page must post ready with a tabId');
  const TAB = ready.tabId;

  send(win, {type: 'clear', chat_id: 'chat-active-tick', tabId: TAB});
  send(win, {
    type: 'status',
    running: true,
    tabId: TAB,
    startTs: Date.now(),
  });

  send(win, {type: 'thinking_start', tabId: TAB});

  const output = win.document.getElementById('output');
  const immediatePanel = output.querySelector('.llm-panel');
  assert.ok(
    immediatePanel,
    'BUG: thinking_start with no thought tokens must immediately show ' +
      'the Thoughts panel so the user can see that the model is thinking.',
  );
  const thinkingContent = immediatePanel.querySelector('.think');
  assert.ok(
    thinkingContent,
    'expected the tokenless Thoughts panel to contain an empty ' +
      'thinking text block',
  );
  assert.strictEqual(
    thinkingContent.textContent,
    '',
    'test setup must reproduce the tokenless-thinking state before ' +
      'checking the live time footer',
  );
  let immediateFooters = immediatePanel.querySelectorAll(':scope > .panel-time');
  assert.strictEqual(
    immediateFooters.length,
    1,
    'BUG: a tokenless active .llm-panel must show a .panel-time ' +
      'footer immediately when thinking_start arrives.',
  );

  // The footer ticks once a second; wait for the first tick rather than
  // a fixed interval so a loaded host cannot make this flake.
  await waitFor(
    () => panelTimeMs(immediatePanel) >= 1000,
    'BUG: the live footer of a tokenless active panel never reached ' +
      '1000ms within 5s.  The page is not ticking the active footer.',
    5000,
  );

  const llmPanels = output.querySelectorAll('.llm-panel');
  assert.strictEqual(
    llmPanels.length,
    1,
    'expected exactly one open .llm-panel, got ' + llmPanels.length,
  );
  const panel = llmPanels[0];
  let footers = panel.querySelectorAll(':scope > .panel-time');
  assert.strictEqual(
    footers.length,
    1,
    'BUG: a tokenless active .llm-panel must have a .panel-time ' +
      'footer that updates live (got ' + footers.length +
      ' footers).  The page is only rendering the footer when ' +
      'the panel closes.',
  );

  const firstMs = parsePanelTimeMs(footers[0].textContent);
  assert.ok(
    !Number.isNaN(firstMs),
    'panel-time text must parse as a duration, got: ' +
      JSON.stringify(footers[0].textContent),
  );
  await waitFor(
    () => panelTimeMs(panel) - firstMs >= 800,
    'BUG: the live .panel-time footer must keep advancing each second; ' +
      'it did not grow by >= 800ms from ' + firstMs + 'ms within 5s.',
    5000,
  );
  footers = panel.querySelectorAll(':scope > .panel-time');
  assert.strictEqual(
    footers.length,
    1,
    'expected the live .panel-time footer to remain a single ' +
      'direct child after the second tick (got ' + footers.length + ')',
  );
  const secondMs = parsePanelTimeMs(footers[0].textContent);
  assert.ok(
    !Number.isNaN(secondMs),
    'panel-time text must still parse as a duration after the ' +
      'second tick, got: ' + JSON.stringify(footers[0].textContent),
  );
  assert.ok(
    secondMs - firstMs >= 800,
    'BUG: the live .panel-time footer must keep advancing each ' +
      'second; after ~1.2s more it must grow by >= 800ms (grew from ' +
      firstMs + 'ms to ' + secondMs + 'ms).',
  );

  const last = panel.lastElementChild;
  assert.ok(
    last && last.classList.contains('panel-time'),
    'BUG: live .panel-time must be the LAST child of its panel; last ' +
      'child is <' + (last ? last.tagName.toLowerCase() : 'null') + '>',
  );

  assert.strictEqual(
    thinkingContent.textContent,
    '',
    'the Thoughts panel must still have zero thought tokens while its ' +
      'live time footer is ticking',
  );

  win.close();
}

runTests().then(
  () => {
    console.log('\nAll tests passed');
    process.exit(0);
  },
  err => {
    console.error('FAIL:', err && err.message ? err.message : err);
    process.exit(1);
  },
);
