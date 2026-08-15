// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) tests for the rule that decides when the webview
// opens a Thoughts panel.
//
// Everything the agent says or thinks belongs inside a "Thoughts" panel;
// loose thinking text dumped straight into the transcript is the symptom
// this file guards against. Two transitions arm the next panel:
//
//   * a `tool_call` -- the agent stopped talking to run something, so
//     whatever it says next starts a new block. This has to hold even
//     when the tool's return value never reaches the transcript, which
//     is why the arming lives on the CALL and not on the result.
//   * a `result` -- RelentlessAgent starts a fresh sub-session after a
//     summarizer finishes, and the first thinking of that sub-session
//     would otherwise land outside a panel, because by then the step
//     count is no longer zero and nothing else armed the flag.
//
// The rule has to hold for all three transcripts a tab can have: the one
// on screen, the detached fragment of a tab that ran while hidden, and a
// replay of a stored task. It also has to survive a `clear` on a hidden
// tab, which starts a new task in a tab whose streaming state is still
// half way through the previous one.

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
  let state;
  win.acquireVsCodeApi = function () {
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
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
  win._testApi.endLaunch();
  win._testApi.hideWelcome();
  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function clickTab(win, tabId) {
  const el = win.document.querySelector(
    `.chat-tab[data-tab-id=${JSON.stringify(tabId)}]`,
  );
  assert.ok(el, `tab ${tabId} must exist in the tab bar`);
  el.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

const TS = 1767225600000;

// thinking_end flushes the buffered deltas synchronously, so a thought
// written this way is on screen by the time the helper returns.
function think(win, tabId, text) {
  send(win, {type: 'thinking_start', tabId, ts: TS});
  send(win, {type: 'thinking_delta', tabId, text, ts: TS});
  send(win, {type: 'thinking_end', tabId, ts: TS});
}

// The events for one thought, as they are stored in a task's transcript.
function thoughtEvents(text) {
  return [
    {type: 'thinking_start', ts: TS},
    {type: 'thinking_delta', text, ts: TS},
    {type: 'thinking_end', ts: TS},
  ];
}

// Where a thought ended up: the Thoughts panel holding it, or null when
// it was written loose into the transcript.
function panelOfThought(win, text) {
  const out = win.document.getElementById('output');
  const thoughts = Array.from(out.querySelectorAll('.ev.think .cnt'));
  const hit = thoughts.find(cnt => cnt.textContent.includes(text));
  assert.ok(hit, `the transcript must contain the thought ${JSON.stringify(text)}`);
  return hit.closest('.llm-panel');
}

function panelCount(win) {
  return win.document.getElementById('output').querySelectorAll('.llm-panel')
    .length;
}

// --- a tool_call arms the next panel -------------------------------------
//
// The call is followed by NO tool_result: that is a tool whose return
// value the printer never broadcast, and the case where arming on the
// result alone would leave the next thought homeless.

function armingRunEvents() {
  return [
    ...thoughtEvents('first look'),
    {type: 'tool_call', name: 'Read', path: 'src/one.py', ts: TS},
    {type: 'tool_result', content: 'one', ts: TS},
    ...thoughtEvents('read it'),
    {type: 'tool_call', name: 'screenshot', ts: TS},
    ...thoughtEvents('after the screenshot'),
  ];
}

function assertArmingTranscript(win, where) {
  const first = panelOfThought(win, 'first look');
  const second = panelOfThought(win, 'read it');
  const third = panelOfThought(win, 'after the screenshot');
  assert.ok(first, `the first thought must open a Thoughts panel (${where})`);
  assert.ok(
    third,
    'a thought that follows a tool_call whose result never reached the ' +
      `transcript must still be inside a Thoughts panel (${where})`,
  );
  assert.notStrictEqual(
    third,
    second,
    'the thought after the tool_call must start a NEW panel, not continue ' +
      `the one the call ended (${where})`,
  );
  assert.strictEqual(
    panelCount(win),
    3,
    `each of the three thoughts must have its own panel (${where})`,
  );
}

function testToolCallArmsPanelOnTheVisibleTab() {
  const {win} = makeWebview();
  const tab = win._testApi.getActiveTabId();
  send(win, {type: 'status', running: true, tabId: tab, startTs: TS});
  for (const ev of armingRunEvents()) send(win, {...ev, tabId: tab});
  assertArmingTranscript(win, 'visible tab');
  assert.strictEqual(
    win.document.getElementById('status-steps').textContent,
    'Steps: 3',
    'each opened panel counts one step',
  );
  win.close();
  console.log('  ok - a tool_call arms the next panel on the visible tab');
}

function testToolCallArmsPanelOnABackgroundTab() {
  const {win} = makeWebview();
  const tabA = win._testApi.getActiveTabId();
  win._testApi.createNewTab();
  const tabB = win._testApi.getActiveTabId();
  clickTab(win, tabA);

  send(win, {type: 'status', running: true, tabId: tabB, startTs: TS});
  for (const ev of armingRunEvents()) send(win, {...ev, tabId: tabB});
  assert.strictEqual(
    win._testApi.getActiveTabId(),
    tabA,
    'a background run must not steal the screen',
  );
  clickTab(win, tabB);
  assertArmingTranscript(win, 'background tab');
  win.close();
  console.log('  ok - a tool_call arms the next panel on a background tab');
}

function testToolCallArmsPanelDuringReplay() {
  const {win} = makeWebview();
  const tab = win._testApi.getActiveTabId();
  send(win, {type: 'task_events', tabId: tab, events: armingRunEvents()});
  assertArmingTranscript(win, 'replay');
  win.close();
  console.log('  ok - a tool_call arms the next panel during replay');
}

// --- a result arms the next panel ----------------------------------------
//
// What RelentlessAgent does: a sub-session finishes, its summary is
// rendered as a result, and the next sub-session starts thinking.

function subSessionEvents() {
  return [
    ...thoughtEvents('working on it'),
    {type: 'tool_call', name: 'finish', ts: TS},
    {type: 'tool_result', content: 'done', ts: TS},
    {type: 'result', summary: 'Session one done.', success: true, ts: TS},
    ...thoughtEvents('starting session two'),
  ];
}

function assertSubSessionTranscript(win, where) {
  const before = panelOfThought(win, 'working on it');
  const after = panelOfThought(win, 'starting session two');
  assert.ok(
    after,
    'the first thought of the sub-session that follows a result must be ' +
      `inside a Thoughts panel (${where})`,
  );
  assert.notStrictEqual(
    after,
    before,
    `the new sub-session must get a panel of its own (${where})`,
  );
}

function testResultArmsPanelOnTheVisibleTab() {
  const {win} = makeWebview();
  const tab = win._testApi.getActiveTabId();
  send(win, {type: 'status', running: true, tabId: tab, startTs: TS});
  for (const ev of subSessionEvents()) send(win, {...ev, tabId: tab});
  assertSubSessionTranscript(win, 'visible tab');
  win.close();
  console.log('  ok - a result arms the next panel on the visible tab');
}

function testResultArmsPanelOnABackgroundTab() {
  const {win} = makeWebview();
  const tabA = win._testApi.getActiveTabId();
  win._testApi.createNewTab();
  const tabB = win._testApi.getActiveTabId();
  clickTab(win, tabA);

  send(win, {type: 'status', running: true, tabId: tabB, startTs: TS});
  for (const ev of subSessionEvents()) send(win, {...ev, tabId: tabB});
  clickTab(win, tabB);
  assertSubSessionTranscript(win, 'background tab');
  win.close();
  console.log('  ok - a result arms the next panel on a background tab');
}

function testResultArmsPanelDuringReplay() {
  const {win} = makeWebview();
  const tab = win._testApi.getActiveTabId();
  send(win, {type: 'task_events', tabId: tab, events: subSessionEvents()});
  assertSubSessionTranscript(win, 'replay');
  win.close();
  console.log('  ok - a result arms the next panel during replay');
}

// --- the step count the daemon reports wins ------------------------------

function testResultStepCountReplacesTheCountedOne() {
  const {win} = makeWebview();
  const tab = win._testApi.getActiveTabId();
  send(win, {type: 'status', running: true, tabId: tab, startTs: TS});
  think(win, tab, 'one thought only');
  send(win, {
    type: 'result',
    tabId: tab,
    summary: 'Done.',
    success: true,
    step_count: 12,
    ts: TS,
  });
  think(win, tab, 'and one more');
  assert.ok(
    panelOfThought(win, 'and one more'),
    'the thought after the result still gets a panel even though the ' +
      'daemon just pushed the step count well past zero',
  );
  win.close();
  console.log("  ok - the daemon's step count does not suppress the panel");
}

// --- clear resets a hidden tab's streaming state -------------------------

function testClearResetsABackgroundTabsStreamState() {
  const {win} = makeWebview();
  const tabA = win._testApi.getActiveTabId();
  win._testApi.createNewTab();
  const tabB = win._testApi.getActiveTabId();
  clickTab(win, tabA);

  // A first task runs to its end in the hidden tab, leaving the tab with
  // an open panel and a non-zero step count.
  send(win, {type: 'status', running: true, tabId: tabB, startTs: TS});
  think(win, tabB, 'the previous task');
  send(win, {type: 'status', running: false, tabId: tabB});

  // A second task starts in that same hidden tab.
  send(win, {type: 'clear', tabId: tabB});
  send(win, {type: 'status', running: true, tabId: tabB, startTs: TS});
  think(win, tabB, 'the next task');

  clickTab(win, tabB);
  const out = win.document.getElementById('output');
  assert.ok(
    !out.textContent.includes('the previous task'),
    'clear must wipe the hidden tab’s transcript',
  );
  assert.ok(
    panelOfThought(win, 'the next task'),
    'the first thought after a clear on a hidden tab must open a Thoughts ' +
      'panel: the tab’s streaming state belongs to the task that ended',
  );
  assert.strictEqual(
    panelCount(win),
    1,
    'the new transcript holds exactly the new task’s panel',
  );
  win.close();
  console.log("  ok - clear resets a background tab's streaming state");
}

function main() {
  testToolCallArmsPanelOnTheVisibleTab();
  testToolCallArmsPanelOnABackgroundTab();
  testToolCallArmsPanelDuringReplay();
  testResultArmsPanelOnTheVisibleTab();
  testResultArmsPanelOnABackgroundTab();
  testResultArmsPanelDuringReplay();
  testResultStepCountReplacesTheCountedOne();
  testClearResetsABackgroundTabsStreamState();
  console.log('thoughtsPanelArming.test.js: all tests passed');
}

main();
