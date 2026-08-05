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
    return {
      postMessage: msg => posted.push(msg),
      getState: () => undefined,
      setState: () => {},
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

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
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
  const wv = makeWebview();
  const win = wv.win;
  const ready = wv.posted.find(m => m.type === 'ready');
  assert.ok(ready && ready.tabId, 'webview must post ready with a tabId');
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
  const thinkingContent = immediatePanel.querySelector('.ev.think .cnt');
  assert.ok(
    thinkingContent,
    'expected the tokenless Thoughts panel to contain an empty ' +
      'thinking content element',
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
    'BUG: a tokenless active .llm-panel must have a .panel-time ' +
      'footer that updates live (got ' + footers.length +
      ' footers).  The webview is only rendering the footer when ' +
      'the panel closes.',
  );

  const firstMs = parsePanelTimeMs(footers[0].textContent);
  assert.ok(
    !Number.isNaN(firstMs),
    'panel-time text must parse as a duration, got: ' +
      JSON.stringify(footers[0].textContent),
  );
  assert.ok(
    firstMs >= 1000,
    'BUG: after waiting ~2.2s with a tokenless active panel, the ' +
      'live footer must have ticked at least once and show >= 1000ms (got ' +
      firstMs + 'ms).  The webview is not ticking the active footer.',
  );

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
