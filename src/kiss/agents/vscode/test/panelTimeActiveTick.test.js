// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end test for the "live panel time" feature.
//
// REQUIREMENT: While a panel (a Thoughts ``.llm-panel`` or a Tool-call
// ``.tc`` panel) is still active — i.e. it has been opened but not yet
// closed by ``thinking_end`` / ``tool_result`` / ``result`` — the
// webview must update its ``.panel-time`` footer every second so the
// user sees the elapsed time tick up live.
//
// Before the fix, ``.panel-time`` is only rendered when the panel
// closes, so an active panel shows no footer (and a freshly opened
// panel has no visible elapsed time at all).  This test reproduces
// that bug by opening a Thoughts panel and never closing it, waiting
// real wall-clock time, and asserting that the footer exists and that
// its value grows by ~1 second between ticks.
//
// Run directly with ``node``:
//
//     node src/kiss/agents/vscode/test/panelTimeActiveTick.test.js

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
  // Strip inline <script>...</script> tags that reference template
  // placeholders that don't exist in tests.
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

  // Hand the webview a backend chat id and mark the tab "running".
  send(win, {type: 'clear', chat_id: 'chat-active-tick', tabId: TAB});
  send(win, {
    type: 'status',
    running: true,
    tabId: TAB,
    startTs: Date.now(),
  });

  // Open a Thoughts panel but DO NOT close it — the panel must remain
  // active so we can observe the live-tick footer growing in real time.
  send(win, {type: 'thinking_start', tabId: TAB});
  send(win, {type: 'thinking_delta', text: 'Planning…', tabId: TAB});

  const output = win.document.getElementById('output');

  // After ~2.2s the live ticker should have rendered at least once.
  await sleep(2200);

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
    'BUG: an active .llm-panel must have a .panel-time footer that ' +
      'updates live (got ' + footers.length + ' footers).  The webview ' +
      'is only rendering the footer when the panel closes.',
  );

  const firstMs = parsePanelTimeMs(footers[0].textContent);
  assert.ok(
    !Number.isNaN(firstMs),
    'panel-time text must parse as a duration, got: ' +
      JSON.stringify(footers[0].textContent),
  );
  assert.ok(
    firstMs >= 1000,
    'BUG: after waiting ~2.2s with an active panel, the live footer ' +
      'must have ticked at least once and show >= 1000ms (got ' +
      firstMs + 'ms).  The webview is not ticking the active footer.',
  );

  // After another ~1.2s the footer value must have strictly grown by
  // at least ~800ms, proving the tick keeps firing each second.
  await sleep(1200);
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

  // The .panel-time footer must stay anchored as the LAST child even
  // while it is ticking live, so it always renders at the bottom.
  const last = panel.lastElementChild;
  assert.ok(
    last && last.classList.contains('panel-time'),
    'BUG: live .panel-time must be the LAST child of its panel; last ' +
      'child is <' + (last ? last.tagName.toLowerCase() : 'null') + '>',
  );

  // Closing the panel (via ``result``) must freeze the footer at the
  // closing time and stop the live tick from mutating it further.
  send(win, {type: 'thinking_end', tabId: TAB});
  send(win, {
    type: 'result',
    success: true,
    summary: 'finished',
    tabId: TAB,
  });
  send(win, {type: 'status', running: false, tabId: TAB});
  const closedMs = parsePanelTimeMs(
    panel.querySelector(':scope > .panel-time').textContent,
  );
  await sleep(1300);
  const frozenMs = parsePanelTimeMs(
    panel.querySelector(':scope > .panel-time').textContent,
  );
  assert.ok(
    Math.abs(frozenMs - closedMs) < 200,
    'BUG: after the result event the .panel-time footer must freeze ' +
      'at the closing value; it changed from ' + closedMs + 'ms to ' +
      frozenMs + 'ms during ~1.3s of idle time.',
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
