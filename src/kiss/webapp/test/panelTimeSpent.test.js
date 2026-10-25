// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const {
  makePage,
  send,
  sleep,
  parsePanelTimeMs,
} = require('./simplify2_harness');

async function runTests() {
  const wv = makePage();
  const win = wv.win;
  const ready = wv.posted.find(m => m.type === 'ready');
  assert.ok(ready && ready.tabId, 'page must post ready with a tabId');
  const TAB = ready.tabId;

  send(win, {type: 'clear', chat_id: 'chat-time', tabId: TAB});
  send(win, {
    type: 'status',
    running: true,
    tabId: TAB,
    startTs: Date.now(),
  });

  send(win, {type: 'thinking_start', tabId: TAB});
  send(win, {type: 'thinking_delta', text: 'Plan: list files.', tabId: TAB});
  await sleep(60);
  send(win, {type: 'thinking_end', tabId: TAB});

  send(win, {
    type: 'tool_call',
    name: 'Bash',
    command: 'ls -la',
    description: 'list files',
    tabId: TAB,
  });
  await sleep(80);
  send(win, {
    type: 'tool_result',
    content: 'README.md\nmain.js\n',
    is_error: false,
    tabId: TAB,
  });

  send(win, {type: 'thinking_start', tabId: TAB});
  send(win, {type: 'thinking_delta', text: 'Done.', tabId: TAB});
  await sleep(40);
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
  assert.ok(
    tcMs >= 50,
    'tool-call panel-time must reflect the ~80ms gap between tool_call ' +
      'and tool_result (got ' + tcMs + 'ms). The page is stamping ' +
      'time but not measuring it correctly.',
  );

  const firstThoughts = llmPanels[0];
  const firstMs = parsePanelTimeMs(
    firstThoughts.querySelector(':scope > .panel-time').textContent,
  );
  assert.ok(
    firstMs >= 30,
    'first .llm-panel time must reflect the ~60ms wall-clock gap (got ' +
      firstMs + 'ms)',
  );

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
