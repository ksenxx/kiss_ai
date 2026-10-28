// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const {makePage, send, parsePanelTimeMs} = require('./simplify2_harness');

// The page measures a panel's time with Date.now(), so it runs on a
// clock the test advances by hand: the labels are exact and no
// wall-clock time is waited out.
const clock = {now: 1_700_000_000_000};

function seedClock(win) {
  win.Date.now = () => clock.now;
}

async function runTests() {
  const wv = makePage({beforeScripts: seedClock});
  const win = wv.win;
  const ready = wv.posted.find(m => m.type === 'ready');
  assert.ok(ready && ready.tabId, 'page must post ready with a tabId');
  const TAB = ready.tabId;

  send(win, {type: 'clear', chat_id: 'chat-time', tabId: TAB});
  send(win, {
    type: 'status',
    running: true,
    tabId: TAB,
    startTs: clock.now,
  });

  send(win, {type: 'thinking_start', tabId: TAB});
  send(win, {type: 'thinking_delta', text: 'Plan: list files.', tabId: TAB});
  clock.now += 60;
  send(win, {type: 'thinking_end', tabId: TAB});

  send(win, {
    type: 'tool_call',
    name: 'Bash',
    command: 'ls -la',
    description: 'list files',
    tabId: TAB,
  });
  clock.now += 80;
  send(win, {
    type: 'tool_result',
    content: 'README.md\nmain.js\n',
    is_error: false,
    tabId: TAB,
  });

  send(win, {type: 'thinking_start', tabId: TAB});
  send(win, {type: 'thinking_delta', text: 'Done.', tabId: TAB});
  clock.now += 40;
  send(win, {type: 'thinking_end', tabId: TAB});

  send(win, {
    type: 'result',
    success: true,
    summary: 'finished',
    tabId: TAB,
  });
  send(win, {type: 'status', running: false, tabId: TAB});

  const output = win.document.getElementById('output');
  const llmPanels = output.querySelectorAll('.llm-panel');
  assert.ok(
    llmPanels.length >= 2,
    'expected at least two .llm-panel elements (one per thinking step), ' +
      'got ' + llmPanels.length,
  );
  const tcPanels = output.querySelectorAll('.tc');
  assert.strictEqual(
    tcPanels.length,
    1,
    'expected exactly one .tc tool-call panel, got ' + tcPanels.length,
  );

  for (const p of llmPanels) {
    const footers = p.querySelectorAll(':scope > .panel-time');
    assert.strictEqual(
      footers.length,
      1,
      'BUG: each .llm-panel must have exactly one .panel-time footer ' +
        'as a direct child (got ' + footers.length + '). The page is ' +
        'not stamping per-panel time.',
    );
    const ms = parsePanelTimeMs(footers[0].textContent);
    assert.ok(
      !Number.isNaN(ms),
      'panel-time text must parse as a duration, got: ' +
        JSON.stringify(footers[0].textContent),
    );
    assert.ok(
      ms >= 0,
      'panel-time duration must be non-negative (got ' + ms + 'ms)',
    );
  }

  const tc = tcPanels[0];
  const tcFooters = tc.querySelectorAll(':scope > .panel-time');
  assert.strictEqual(
    tcFooters.length,
    1,
    'BUG: each .tc tool-call panel must have exactly one .panel-time ' +
      'footer as a direct child (got ' + tcFooters.length + '). The ' +
      'page is not stamping per-panel time on tool calls.',
  );
  const tcMs = parsePanelTimeMs(tcFooters[0].textContent);
  assert.ok(
    !Number.isNaN(tcMs),
    'tool-call panel-time text must parse as a duration, got: ' +
      JSON.stringify(tcFooters[0].textContent),
  );
  assert.strictEqual(
    tcMs,
    80,
    'tool-call panel-time must be the 80ms between tool_call and ' +
      'tool_result (got ' + tcMs + 'ms). The page is stamping time but ' +
      'not measuring it correctly.',
  );

  // The first Thoughts panel is sealed when the tool call that follows
  // it starts, 60 ms after it opened.  (The trailing one is only
  // refreshed by the 1 s active-panel tick, which is never due here.)
  const firstMs = parsePanelTimeMs(
    llmPanels[0].querySelector(':scope > .panel-time').textContent,
  );
  assert.strictEqual(
    firstMs,
    60,
    'first .llm-panel time must be the 60ms between its thinking_start ' +
      'and the next panel (got ' + firstMs + 'ms)',
  );

  // The footer re-anchors itself below streamed text in a microtask.
  await new Promise(resolve => setImmediate(resolve));
  for (const p of [llmPanels[0], llmPanels[1], tc]) {
    const last = p.lastElementChild;
    assert.ok(
      last && last.classList.contains('panel-time'),
      'BUG: .panel-time must be the LAST child of its panel (so it ' +
        'renders at the bottom); last child is <' +
        (last ? last.tagName.toLowerCase() : 'null') + '>',
    );
  }

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
