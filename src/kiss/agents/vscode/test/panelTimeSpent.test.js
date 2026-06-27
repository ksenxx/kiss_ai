// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end test for the "time spent in each panel" feature.
//
// REQUIREMENT: The chat webview must show, at the bottom of each panel,
// how long that panel took.  Concretely, after a live agent run produces
// a thinking step, a tool call, and a result, the DOM under #output must
// contain:
//
//   * an ``.llm-panel`` (the Thoughts panel) whose direct child includes
//     a ``.panel-time`` footer showing a real elapsed duration; and
//   * a ``.tc`` (Tool call panel) whose direct child includes a
//     ``.panel-time`` footer showing a real elapsed duration.
//
// This test drives the production ``media/main.js`` inside jsdom, sends
// a realistic event stream with deliberate gaps between events, and
// asserts that the ``.panel-time`` footers exist with sane values.
//
// Before the fix, neither ``.llm-panel`` nor ``.tc`` panels carry a
// ``.panel-time`` footer and this test fails — reproducing the issue.
//
// Run directly with ``node``:
//
//     node src/kiss/agents/vscode/test/panelTimeSpent.test.js

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

/**
 * Build a jsdom window running the production chat webview.
 *
 * @returns {{win: object, posted: object[]}} the jsdom window and a
 *   list capturing every ``vscode.postMessage`` payload.
 */
function makeWebview() {
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
  // Strip the inline <script>...</script> tags injected by chat.html —
  // they reference template placeholders that don't exist in tests.
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
    return {
      postMessage: msg => posted.push(msg),
      getState: () => undefined,
      setState: () => {},
    };
  };

  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));

  return {win, posted};
}

/** Dispatch a backend→webview event exactly like the extension does. */
function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

/** Sleep for ``ms`` real milliseconds. */
function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/** Parse the textContent of a ``.panel-time`` footer into milliseconds. */
function parsePanelTimeMs(text) {
  const t = String(text || '').trim();
  // Accepts "Ns", "N.Ns", "Nms", "Nm Ns", "Nm N.Ns"
  let m = t.match(/^(\d+)m\s+(\d+(?:\.\d+)?)s$/);
  if (m) return (parseInt(m[1], 10) * 60 + parseFloat(m[2])) * 1000;
  m = t.match(/^(\d+(?:\.\d+)?)s$/);
  if (m) return parseFloat(m[1]) * 1000;
  m = t.match(/^(\d+)ms$/);
  if (m) return parseInt(m[1], 10);
  return NaN;
}

async function runTests() {
  const wv = makeWebview();
  const win = wv.win;
  const ready = wv.posted.find(m => m.type === 'ready');
  assert.ok(ready && ready.tabId, 'webview must post ready with a tabId');
  const TAB = ready.tabId;

  // Hand the webview a backend chat id and mark the tab "running" —
  // the same handshake the daemon performs after a fresh ``submit``.
  send(win, {type: 'clear', chat_id: 'chat-time', tabId: TAB});
  send(win, {
    type: 'status',
    running: true,
    tabId: TAB,
    startTs: Date.now(),
  });

  // First step: a thinking block.  Wait a bit so the panel accumulates
  // real wall-clock time before the next step closes it.
  send(win, {type: 'thinking_start', tabId: TAB});
  send(win, {type: 'thinking_delta', text: 'Plan: list files.', tabId: TAB});
  await sleep(60);
  send(win, {type: 'thinking_end', tabId: TAB});

  // Tool call: a Bash command.  Wait, then send the tool result.
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

  // Second thinking step closes the tool-call window and opens a new
  // Thoughts panel that we then close with the final ``result``.
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

  // ---- Assertions: every Thoughts and Tool-call panel has a footer.
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
        'as a direct child (got ' + footers.length + '). The webview is ' +
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
      'webview is not stamping per-panel time on tool calls.',
  );
  const tcMs = parsePanelTimeMs(tcFooters[0].textContent);
  assert.ok(
    !Number.isNaN(tcMs),
    'tool-call panel-time text must parse as a duration, got: ' +
      JSON.stringify(tcFooters[0].textContent),
  );
  // The tool-call panel had ~80ms of wall-clock time; require at least a
  // measurable non-trivial duration to verify timing actually advances.
  assert.ok(
    tcMs >= 50,
    'tool-call panel-time must reflect the ~80ms gap between tool_call ' +
      'and tool_result (got ' + tcMs + 'ms). The webview is stamping ' +
      'time but not measuring it correctly.',
  );

  // The first thinking panel saw ~60ms before the tool call closed it.
  const firstThoughts = llmPanels[0];
  const firstMs = parsePanelTimeMs(
    firstThoughts.querySelector(':scope > .panel-time').textContent,
  );
  assert.ok(
    firstMs >= 30,
    'first .llm-panel time must reflect the ~60ms wall-clock gap (got ' +
      firstMs + 'ms)',
  );

  // The .panel-time footer must be the LAST child of each panel so it
  // renders visually at the bottom.
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
