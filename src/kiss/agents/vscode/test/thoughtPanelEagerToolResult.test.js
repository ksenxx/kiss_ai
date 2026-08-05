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
  win.requestAnimationFrame = function (cb) {
    cb();
    return 0;
  };
  win.cancelAnimationFrame = function () {};

  const posted = [];
  win.acquireVsCodeApi = function () {
    let state;
    return {
      postMessage: msg => posted.push(msg),
      getState: () => state,
      setState: s => {
        state = s;
      },
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

function llmPanels(root) {
  return root.querySelectorAll('.llm-panel');
}

function panelFooter(panel) {
  const footers = panel.querySelectorAll(':scope > .panel-time');
  return footers.length === 1 ? footers[0] : null;
}

function stepsText(win) {
  return win.document.getElementById('status-steps').textContent;
}

async function testEagerPanelAfterToolResult() {
  const wv = makeWebview();
  const win = wv.win;
  const ready = wv.posted.find(m => m.type === 'ready');
  assert.ok(ready && ready.tabId, 'webview must post ready with a tabId');
  const TAB = ready.tabId;
  const output = win.document.getElementById('output');

  send(win, {type: 'clear', chat_id: 'chat-eager', tabId: TAB});
  send(win, {type: 'status', running: true, tabId: TAB, startTs: Date.now()});

  send(win, {type: 'thinking_start', tabId: TAB});
  send(win, {type: 'thinking_delta', text: 'Plan: run ls.', tabId: TAB});
  send(win, {type: 'thinking_end', tabId: TAB});
  assert.strictEqual(
    llmPanels(output).length,
    1,
    'first thinking step must render exactly one Thoughts panel',
  );
  assert.strictEqual(stepsText(win), 'Steps: 1');

  send(win, {
    type: 'tool_call',
    name: 'Bash',
    command: 'ls',
    description: 'list',
    tabId: TAB,
  });
  send(win, {
    type: 'tool_result',
    content: 'README.md\n',
    is_error: false,
    tabId: TAB,
  });

  let panels = llmPanels(output);
  assert.strictEqual(
    panels.length,
    2,
    'BUG: a new Thoughts panel must be appended EAGERLY right after ' +
      'tool_result (before the model streams anything); got ' +
      panels.length + ' .llm-panel elements',
  );
  const eager = panels[1];
  assert.ok(
    eager.querySelector('.llm-panel-hdr'),
    'eager panel must carry the Thoughts header',
  );
  assert.strictEqual(
    output.lastElementChild,
    eager,
    'the eager Thoughts panel must be appended after the tool result',
  );
  let footer = panelFooter(eager);
  assert.ok(
    footer,
    'BUG: the eager Thoughts panel must have exactly one .panel-time ' +
      'footer as a direct child from the moment it is created',
  );
  const ms0 = parsePanelTimeMs(footer.textContent);
  assert.ok(
    !Number.isNaN(ms0) && ms0 >= 0,
    'eager panel footer must show a parseable duration, got ' +
      JSON.stringify(footer.textContent),
  );
  assert.strictEqual(
    stepsText(win),
    'Steps: 1',
    'step count must not advance until model output lands in the panel',
  );

  await sleep(1150);
  footer = panelFooter(eager);
  const msTicked = parsePanelTimeMs(footer.textContent);
  assert.ok(
    msTicked >= 1000,
    'BUG: the eager panel footer must keep ticking while waiting for ' +
      'the model (expected >= 1000ms after 1.15s, got ' + msTicked + 'ms)',
  );

  send(win, {type: 'thinking_start', tabId: TAB});
  send(win, {type: 'thinking_delta', text: 'EAGER-TOKEN-XYZ', tabId: TAB});
  panels = llmPanels(output);
  assert.strictEqual(
    panels.length,
    2,
    'BUG: streamed thinking must reuse the eager Thoughts panel, not ' +
      'create a new one (got ' + panels.length + ' panels)',
  );
  assert.ok(
    eager.textContent.includes('EAGER-TOKEN-XYZ'),
    'BUG: streamed thinking tokens must be appended INSIDE the eager ' +
      'Thoughts panel',
  );
  assert.strictEqual(
    stepsText(win),
    'Steps: 2',
    'the step count must advance exactly once when model output ' +
      'arrives in the eager panel',
  );
  send(win, {type: 'thinking_end', tabId: TAB});
  send(win, {type: 'thinking_start', tabId: TAB});
  send(win, {type: 'thinking_delta', text: 'more', tabId: TAB});
  send(win, {type: 'thinking_end', tabId: TAB});
  assert.strictEqual(llmPanels(output).length, 2);
  assert.strictEqual(stepsText(win), 'Steps: 2');

  send(win, {type: 'tool_call', name: 'finish', tabId: TAB});
  send(win, {type: 'tool_result', content: 'done', tabId: TAB});
  assert.strictEqual(
    llmPanels(output).length,
    2,
    'BUG: no eager Thoughts panel may be created after the finish ' +
      "tool's result",
  );
  send(win, {type: 'result', success: true, summary: 'ok', tabId: TAB});
  assert.strictEqual(llmPanels(output).length, 2);

  footer = panelFooter(eager);
  const msFinal = parsePanelTimeMs(footer.textContent);
  assert.ok(
    msFinal >= 1000,
    'final footer must cover the time spent waiting for the model ' +
      '(got ' + msFinal + 'ms)',
  );
  assert.strictEqual(
    eager.lastElementChild,
    footer,
    '.panel-time must be the LAST child of the eager panel',
  );
  win.close();
}

async function testProvisionalPanelRemovedOnNextToolCall() {
  const wv = makeWebview();
  const win = wv.win;
  const TAB = wv.posted.find(m => m.type === 'ready').tabId;
  const output = win.document.getElementById('output');

  send(win, {type: 'clear', chat_id: 'chat-par', tabId: TAB});
  send(win, {type: 'status', running: true, tabId: TAB, startTs: Date.now()});

  send(win, {type: 'thinking_start', tabId: TAB});
  send(win, {type: 'thinking_delta', text: 'Two tools.', tabId: TAB});
  send(win, {type: 'thinking_end', tabId: TAB});

  send(win, {type: 'tool_call', name: 'Bash', command: 'ls', tabId: TAB});
  send(win, {type: 'tool_result', content: 'a\n', tabId: TAB});
  assert.strictEqual(
    llmPanels(output).length,
    2,
    'first tool_result must open an eager Thoughts panel',
  );

  send(win, {type: 'tool_call', name: 'Read', file_path: '/x', tabId: TAB});
  assert.strictEqual(
    llmPanels(output).length,
    1,
    'BUG: the still-empty provisional Thoughts panel must be removed ' +
      'when the turn continues with another tool call',
  );
  assert.strictEqual(
    stepsText(win),
    'Steps: 1',
    'a removed provisional panel must not change the step count',
  );
  send(win, {type: 'tool_result', content: 'content', tabId: TAB});
  assert.strictEqual(
    llmPanels(output).length,
    2,
    'the second tool_result must open a fresh eager Thoughts panel',
  );

  send(win, {type: 'thinking_start', tabId: TAB});
  send(win, {type: 'thinking_delta', text: 'SECOND-TURN-TOK', tabId: TAB});
  send(win, {type: 'thinking_end', tabId: TAB});
  const panels = llmPanels(output);
  assert.strictEqual(panels.length, 2);
  assert.ok(panels[1].textContent.includes('SECOND-TURN-TOK'));
  assert.strictEqual(stepsText(win), 'Steps: 2');

  for (const p of panels) {
    assert.ok(
      p.querySelector('.think, .txt'),
      'BUG: transcript contains an empty Thoughts panel',
    );
  }

  send(win, {type: 'tool_call', name: 'finish', tabId: TAB});
  send(win, {type: 'tool_result', content: 'done', tabId: TAB});
  send(win, {type: 'result', success: true, summary: 'ok', tabId: TAB});
  assert.strictEqual(llmPanels(output).length, 2);
  win.close();
}

async function testTextDeltaFillsEagerPanel() {
  const wv = makeWebview();
  const win = wv.win;
  const TAB = wv.posted.find(m => m.type === 'ready').tabId;
  const output = win.document.getElementById('output');

  send(win, {type: 'clear', chat_id: 'chat-text', tabId: TAB});
  send(win, {type: 'status', running: true, tabId: TAB, startTs: Date.now()});

  send(win, {type: 'text_delta', text: 'first turn text', tabId: TAB});
  send(win, {type: 'text_end', tabId: TAB});
  assert.strictEqual(llmPanels(output).length, 1);
  assert.strictEqual(stepsText(win), 'Steps: 1');

  send(win, {type: 'tool_call', name: 'Bash', command: 'ls', tabId: TAB});
  send(win, {type: 'tool_result', content: 'b\n', tabId: TAB});
  assert.strictEqual(
    llmPanels(output).length,
    2,
    'tool_result must open an eager Thoughts panel',
  );

  send(win, {type: 'text_delta', text: 'PLAIN-TEXT-TOK', tabId: TAB});
  send(win, {type: 'text_end', tabId: TAB});
  const panels = llmPanels(output);
  assert.strictEqual(
    panels.length,
    2,
    'BUG: text_delta must reuse the eager Thoughts panel',
  );
  assert.ok(
    panels[1].textContent.includes('PLAIN-TEXT-TOK'),
    'text tokens must land inside the eager panel',
  );
  assert.strictEqual(stepsText(win), 'Steps: 2');

  send(win, {type: 'tool_call', name: 'finish', tabId: TAB});
  send(win, {type: 'tool_result', content: 'done', tabId: TAB});
  send(win, {type: 'result', success: true, summary: 'ok', tabId: TAB});
  win.close();
}

async function testEagerPanelBackgroundTab() {
  const wv = makeWebview();
  const win = wv.win;
  const api = win._testApi;
  const tab1 = api.getActiveTabId();

  api.createNewTab();
  const tab2 = api.getActiveTabId();
  assert.ok(tab2 && tab2 !== tab1, 'a fresh second tab must be active');

  send(win, {type: 'clear', chat_id: 'chat-bg', tabId: tab1});
  send(win, {type: 'status', running: true, tabId: tab1, startTs: Date.now()});
  send(win, {type: 'thinking_start', tabId: tab1});
  send(win, {type: 'thinking_delta', text: 'bg plan', tabId: tab1});
  send(win, {type: 'thinking_end', tabId: tab1});
  send(win, {type: 'tool_call', name: 'Bash', command: 'ls', tabId: tab1});
  send(win, {type: 'tool_result', content: 'x\n', tabId: tab1});
  send(win, {type: 'tool_call', name: 'Read', file_path: '/y', tabId: tab1});
  send(win, {type: 'tool_result', content: 'y-content', tabId: tab1});
  send(win, {type: 'thinking_start', tabId: tab1});
  send(win, {type: 'thinking_delta', text: 'BG-EAGER-TOK', tabId: tab1});
  send(win, {type: 'thinking_end', tabId: tab1});
  send(win, {type: 'tool_call', name: 'finish', tabId: tab1});
  send(win, {type: 'tool_result', content: 'done', tabId: tab1});
  send(win, {type: 'result', success: true, summary: 'ok', tabId: tab1});

  const activeOut = win.document.getElementById('output');
  assert.ok(
    !activeOut.textContent.includes('BG-EAGER-TOK'),
    'background-tab events must not render in the active tab',
  );

  const tabEl = win.document.querySelector('.chat-tab[data-tab-id="' + tab1 + '"]');
  assert.ok(tabEl, 'tab1 element must exist in the tab bar');
  tabEl.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  const output = win.document.getElementById('output');
  const panels = llmPanels(output);
  assert.strictEqual(
    panels.length,
    2,
    'BUG: background tab must show exactly two Thoughts panels ' +
      '(provisional empty panel removed), got ' + panels.length,
  );
  assert.ok(
    panels[1].textContent.includes('BG-EAGER-TOK'),
    'BUG: bg-streamed thinking must land inside the eager bg panel',
  );
  const footer = panelFooter(panels[1]);
  assert.ok(
    footer,
    'BUG: the eager bg Thoughts panel must carry a .panel-time footer',
  );
  assert.ok(
    !Number.isNaN(parsePanelTimeMs(footer.textContent)),
    'bg panel footer must show a parseable duration',
  );
  for (const p of panels) {
    assert.ok(
      p.querySelector('.think, .txt'),
      'BUG: background transcript contains an empty Thoughts panel',
    );
  }
  win.close();
}

async function testEagerPanelSurvivesTabSwitch() {
  const wv = makeWebview();
  const win = wv.win;
  const api = win._testApi;
  const tab1 = api.getActiveTabId();
  const output = win.document.getElementById('output');

  send(win, {type: 'clear', chat_id: 'chat-switch', tabId: tab1});
  send(win, {type: 'status', running: true, tabId: tab1, startTs: Date.now()});
  send(win, {type: 'thinking_start', tabId: tab1});
  send(win, {type: 'thinking_delta', text: 'plan', tabId: tab1});
  send(win, {type: 'thinking_end', tabId: tab1});
  send(win, {type: 'tool_call', name: 'Bash', command: 'ls', tabId: tab1});
  send(win, {type: 'tool_result', content: 'z\n', tabId: tab1});
  assert.strictEqual(llmPanels(output).length, 2);

  api.createNewTab();
  const backEl = win.document.querySelector('.chat-tab[data-tab-id="' + tab1 + '"]');
  backEl.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.strictEqual(
    llmPanels(win.document.getElementById('output')).length,
    2,
    'the provisional eager panel must survive a tab switch',
  );

  send(win, {type: 'thinking_start', tabId: tab1});
  send(win, {type: 'thinking_delta', text: 'AFTER-SWITCH-TOK', tabId: tab1});
  send(win, {type: 'thinking_end', tabId: tab1});
  const panels = llmPanels(win.document.getElementById('output'));
  assert.strictEqual(panels.length, 2);
  assert.ok(
    panels[1].textContent.includes('AFTER-SWITCH-TOK'),
    'thinking after a tab round-trip must fill the surviving eager panel',
  );
  assert.strictEqual(stepsText(win), 'Steps: 2');
  win.close();
}

async function testEagerPanelFooterTicksAfterTabRestore() {
  const wv = makeWebview();
  const win = wv.win;
  const api = win._testApi;
  const tab1 = api.getActiveTabId();

  send(win, {type: 'clear', chat_id: 'chat-tick-restore', tabId: tab1});
  send(win, {type: 'status', running: true, tabId: tab1, startTs: Date.now()});
  send(win, {type: 'thinking_start', tabId: tab1});
  send(win, {type: 'thinking_delta', text: 'plan', tabId: tab1});
  send(win, {type: 'thinking_end', tabId: tab1});
  send(win, {type: 'tool_call', name: 'Bash', command: 'ls', tabId: tab1});
  send(win, {type: 'tool_result', content: 'w\n', tabId: tab1});

  api.createNewTab();
  await sleep(1150);

  const backEl = win.document.querySelector(
    '.chat-tab[data-tab-id="' + tab1 + '"]',
  );
  backEl.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  const output = win.document.getElementById('output');
  const eager = llmPanels(output)[1];
  assert.ok(eager, 'eager panel must survive the tab round-trip');

  await sleep(1250);
  const footer = panelFooter(eager);
  assert.ok(footer, 'restored eager panel must still have a footer');
  const ms = parsePanelTimeMs(footer.textContent);
  assert.ok(
    ms >= 2000,
    'BUG: the eager panel footer must RESUME live ticking after the ' +
      'tab is restored (expected >= 2000ms of accumulated wait, got ' +
      ms + 'ms — the ticker pruned the detached panel and never ' +
      're-registered it)',
  );

  send(win, {type: 'thinking_start', tabId: tab1});
  send(win, {type: 'thinking_delta', text: 'RESUMED-TOK', tabId: tab1});
  send(win, {type: 'thinking_end', tabId: tab1});
  assert.ok(eager.textContent.includes('RESUMED-TOK'));
  send(win, {type: 'tool_call', name: 'finish', tabId: tab1});
  send(win, {type: 'tool_result', content: 'done', tabId: tab1});
  send(win, {type: 'result', success: true, summary: 'ok', tabId: tab1});
  win.close();
}

async function testProvisionalPanelDiscardedOnTaskStop() {
  const wv = makeWebview();
  const win = wv.win;
  const TAB = wv.posted.find(m => m.type === 'ready').tabId;
  const output = win.document.getElementById('output');

  send(win, {type: 'clear', chat_id: 'chat-stop', tabId: TAB});
  send(win, {type: 'status', running: true, tabId: TAB, startTs: Date.now()});
  send(win, {type: 'thinking_start', tabId: TAB});
  send(win, {type: 'thinking_delta', text: 'plan', tabId: TAB});
  send(win, {type: 'thinking_end', tabId: TAB});
  send(win, {type: 'tool_call', name: 'Bash', command: 'ls', tabId: TAB});
  send(win, {type: 'tool_result', content: 'q\n', tabId: TAB});
  assert.strictEqual(llmPanels(output).length, 2);

  send(win, {type: 'task_stopped', tabId: TAB});
  assert.strictEqual(
    llmPanels(output).length,
    1,
    'BUG: stopping the task must discard the still-empty provisional ' +
      'Thoughts panel instead of leaving it ticking forever',
  );

  send(win, {type: 'status', running: true, tabId: TAB, startTs: Date.now()});
  send(win, {type: 'thinking_start', tabId: TAB});
  send(win, {type: 'thinking_delta', text: 'RESUME-TOK', tabId: TAB});
  send(win, {type: 'thinking_end', tabId: TAB});
  const panels = llmPanels(output);
  assert.strictEqual(
    panels.length,
    2,
    'a stream resumed after task_stopped must create a fresh Thoughts panel',
  );
  assert.ok(panels[1].textContent.includes('RESUME-TOK'));
  win.close();
}

async function testTaskEndFreezesFilledPanelAndCleansBgTab() {
  const wv = makeWebview();
  const win = wv.win;
  const api = win._testApi;
  const tab1 = api.getActiveTabId();
  const output = win.document.getElementById('output');

  send(win, {type: 'clear', chat_id: 'chat-end', tabId: tab1});
  send(win, {type: 'status', running: true, tabId: tab1, startTs: Date.now()});
  send(win, {type: 'thinking_start', tabId: tab1});
  send(win, {type: 'thinking_delta', text: 'still thinking', tabId: tab1});
  await sleep(60);
  send(win, {type: 'task_error', tabId: tab1});
  const panel = llmPanels(output)[0];
  assert.ok(panel, 'the filled panel must remain after task_error');
  const footer = panelFooter(panel);
  assert.ok(footer, 'the filled panel must keep its .panel-time footer');
  const frozenMs = parsePanelTimeMs(footer.textContent);
  await sleep(1150);
  const laterMs = parsePanelTimeMs(panelFooter(panel).textContent);
  assert.ok(
    Math.abs(laterMs - frozenMs) < 200,
    'BUG: task_error must freeze the open panel footer (changed from ' +
      frozenMs + 'ms to ' + laterMs + 'ms)',
  );

  api.createNewTab();
  send(win, {type: 'clear', chat_id: 'chat-bg-stop', tabId: tab1});
  send(win, {type: 'status', running: true, tabId: tab1, startTs: Date.now()});
  send(win, {type: 'thinking_start', tabId: tab1});
  send(win, {type: 'thinking_delta', text: 'bg', tabId: tab1});
  send(win, {type: 'thinking_end', tabId: tab1});
  send(win, {type: 'tool_call', name: 'Bash', command: 'ls', tabId: tab1});
  send(win, {type: 'tool_result', content: 'r\n', tabId: tab1});
  send(win, {type: 'task_stopped', tabId: tab1});
  const backEl = win.document.querySelector(
    '.chat-tab[data-tab-id="' + tab1 + '"]',
  );
  backEl.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  const bgPanels = llmPanels(win.document.getElementById('output'));
  assert.strictEqual(
    bgPanels.length,
    1,
    'BUG: a background tab whose task stopped must not keep its empty ' +
      'provisional Thoughts panel (got ' + bgPanels.length + ' panels)',
  );
  assert.ok(bgPanels[0].textContent.includes('bg'));
  win.close();
}

async function runTests() {
  await testEagerPanelAfterToolResult();
  console.log('ok - testEagerPanelAfterToolResult');
  await testProvisionalPanelRemovedOnNextToolCall();
  console.log('ok - testProvisionalPanelRemovedOnNextToolCall');
  await testTextDeltaFillsEagerPanel();
  console.log('ok - testTextDeltaFillsEagerPanel');
  await testEagerPanelBackgroundTab();
  console.log('ok - testEagerPanelBackgroundTab');
  await testEagerPanelSurvivesTabSwitch();
  console.log('ok - testEagerPanelSurvivesTabSwitch');
  await testEagerPanelFooterTicksAfterTabRestore();
  console.log('ok - testEagerPanelFooterTicksAfterTabRestore');
  await testProvisionalPanelDiscardedOnTaskStop();
  console.log('ok - testProvisionalPanelDiscardedOnTaskStop');
  await testTaskEndFreezesFilledPanelAndCleansBgTab();
  console.log('ok - testTaskEndFreezesFilledPanelAndCleansBgTab');
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
