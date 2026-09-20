// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// The composer keeps its autocomplete help while a task is running.
//
// While an agent works, the textbox drafts the message that will be queued
// for it, and that draft used to lose ghost text and the completions picker:
// requestGhost() refused to ask the daemon for a completion and
// renderCompletions() dropped any reply while the tab was marked running.
// The @-mention file picker was never gated, so it is covered here too to
// pin down the whole running-state contract of the composer in one place.

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
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function input(win) {
  return win.document.getElementById('task-input');
}

function typeChar(win, ch) {
  const inp = input(win);
  inp.value += ch;
  inp.setSelectionRange(inp.value.length, inp.value.length);
  inp.dispatchEvent(new win.Event('input', {bubbles: true}));
  return inp;
}

function typeText(win, text) {
  for (const ch of text) typeChar(win, ch);
  return input(win);
}

function pressKey(win, key) {
  const ev = new win.KeyboardEvent('keydown', {
    key,
    bubbles: true,
    cancelable: true,
  });
  input(win).dispatchEvent(ev);
  return ev;
}

function picker(win) {
  return win.document.getElementById('autocomplete');
}

function pickerVisible(win) {
  return picker(win).style.display === 'block';
}

function ghostText(win) {
  const el = win.document
    .getElementById('ghost-overlay')
    .querySelector('.ghost-text');
  return el ? el.textContent : '';
}

// Mark the visible tab's task as running the way the daemon does: a
// `status` event addressed to the tab. The Stop button proves it landed.
function startRunning(win) {
  const tabId = win._testApi.getActiveTabId();
  send(win, {type: 'status', running: true, tabId});
  assert.strictEqual(
    win.document.getElementById('stop-btn').style.display,
    'flex',
    'precondition: the tab must be in the running state',
  );
  return tabId;
}

// Wait past requestGhost()'s 300 ms debounce.
function afterDebounce(win) {
  return new Promise(resolve => win.setTimeout(resolve, 450));
}

const COMPLETIONS = [
  {type: 'task', text: 'fix the parser bug now'},
  {type: 'trick', text: 'fix the parser then commit'},
];

async function testTypingWhileRunningRequestsCompletion() {
  const {win, posted} = makeWebview();
  const tabId = startRunning(win);
  posted.length = 0;
  typeText(win, 'fix');
  await afterDebounce(win);
  const cmd = posted.find(p => p && p.type === 'complete');
  assert.ok(cmd, 'typing during a run must dispatch a ``complete`` command');
  assert.strictEqual(cmd.query, 'fix');
  assert.strictEqual(
    cmd.tabId,
    tabId,
    'the request must be stamped with the running tab so the reply routes back',
  );
}

function testCompletionsPickerOpensWhileRunning() {
  const {win} = makeWebview();
  const tabId = startRunning(win);
  const inp = typeText(win, 'fix');
  send(win, {
    type: 'completions',
    completions: COMPLETIONS,
    query: 'fix',
    tabId,
  });
  assert.strictEqual(
    pickerVisible(win),
    true,
    'a completions reply must open the picker while the task runs',
  );
  const items = picker(win).querySelectorAll('.ac-item');
  assert.strictEqual(items.length, COMPLETIONS.length);

  const ev = pressKey(win, 'Tab');
  assert.strictEqual(ev.defaultPrevented, true, 'Tab must be consumed');
  assert.strictEqual(inp.value, 'fix the parser bug now ');
  assert.strictEqual(pickerVisible(win), false, 'accepting closes the picker');
}

function testGhostRendersAndAcceptsWhileRunning() {
  const {win} = makeWebview();
  const tabId = startRunning(win);
  const inp = typeText(win, 'fix');
  send(win, {type: 'ghost', suggestion: ' the bug', query: 'fix', tabId});
  assert.strictEqual(ghostText(win), ' the bug');
  pressKey(win, 'Tab');
  assert.strictEqual(inp.value, 'fix the bug ');
  assert.strictEqual(ghostText(win), '', 'accepting clears the ghost');
}

function testAtMentionPickerWhileRunning() {
  const {win, posted} = makeWebview();
  const tabId = startRunning(win);
  posted.length = 0;
  const inp = typeText(win, 'look at @sr');
  const reqs = posted.filter(p => p && p.type === 'getFiles');
  assert.ok(reqs.length >= 1, 'typing ``@`` during a run must ask for files');
  const last = reqs[reqs.length - 1];
  assert.strictEqual(last.prefix, 'sr');
  assert.strictEqual(last.tabId, tabId);

  send(win, {
    type: 'files',
    files: [{type: 'file', text: 'src/main.py'}],
    prefix: 'sr',
    tabId,
  });
  assert.strictEqual(
    pickerVisible(win),
    true,
    'a files reply must open the @-mention picker while the task runs',
  );
  pressKey(win, 'Tab');
  assert.strictEqual(inp.value, 'look at ./src/main.py ');
  assert.strictEqual(pickerVisible(win), false);
}

async function testCompletionSurvivesRunEndingMidDraft() {
  // The gates used to make the composer's help depend on run state; now
  // the same draft gets the same help whether the run ends or not.
  const {win, posted} = makeWebview();
  const tabId = startRunning(win);
  posted.length = 0;
  typeText(win, 'fix');
  send(win, {type: 'status', running: false, tabId});
  await afterDebounce(win);
  assert.ok(
    posted.find(p => p && p.type === 'complete'),
    'the completion scheduled during the run must still be sent',
  );
  send(win, {
    type: 'completions',
    completions: COMPLETIONS,
    query: 'fix',
    tabId,
  });
  assert.strictEqual(pickerVisible(win), true);
}

function testEmptyCompletionsStillHidePickerWhileRunning() {
  const {win} = makeWebview();
  const tabId = startRunning(win);
  typeText(win, 'fix');
  send(win, {
    type: 'completions',
    completions: COMPLETIONS,
    query: 'fix',
    tabId,
  });
  assert.strictEqual(pickerVisible(win), true);
  send(win, {type: 'completions', completions: [], query: 'fix', tabId});
  assert.strictEqual(
    pickerVisible(win),
    false,
    'an empty reply must still close the picker during a run',
  );
}

function testProgrammaticTabSwitchDropsPickerAndGhost() {
  // A background task finishing pulls its tab forward WITHOUT blurring
  // the textbox, so the blur handler never gets to drop the picker and
  // ghost text computed for the tab that just left the screen. Tab must
  // not paste tab A's suggestion into tab B's empty composer.
  const {win} = makeWebview();
  const api = win._testApi;
  const tabA = api.getActiveTabId();
  api.createNewTab();
  const tabB = api.getActiveTabId();
  assert.ok(tabB && tabB !== tabA, 'a fresh second tab must be active');
  send(win, {type: 'status', running: true, tabId: tabB});
  api.endLaunch();
  // Back on A, still running, with a live picker and ghost for its draft.
  send(win, {type: 'task_done', tabId: tabA, startTs: 1, endTs: 2});
  assert.strictEqual(api.getActiveTabId(), tabA);
  send(win, {type: 'status', running: true, tabId: tabA});
  typeText(win, 'fix');
  send(win, {
    type: 'completions',
    completions: [{type: 'task', text: 'fix FROM FIRST TAB'}],
    query: 'fix',
    tabId: tabA,
  });
  send(win, {
    type: 'ghost',
    suggestion: ' FROM FIRST TAB',
    query: 'fix',
    tabId: tabA,
  });
  assert.strictEqual(pickerVisible(win), true);
  assert.strictEqual(ghostText(win), ' FROM FIRST TAB');

  // B's task ends: the webview switches to B on its own.
  send(win, {type: 'task_done', tabId: tabB, startTs: 1, endTs: 2});
  assert.strictEqual(
    api.getActiveTabId(),
    tabB,
    'precondition: B is on screen',
  );
  assert.strictEqual(input(win).value, '', "B's composer starts empty");
  assert.strictEqual(
    pickerVisible(win),
    false,
    "tab A's completions picker must not survive the switch",
  );
  assert.strictEqual(ghostText(win), '', "tab A's ghost must not survive");
  pressKey(win, 'Tab');
  assert.strictEqual(
    input(win).value,
    '',
    "Tab on B must not paste A's suggestion into B's composer",
  );
}

const tests = [
  testTypingWhileRunningRequestsCompletion,
  testCompletionsPickerOpensWhileRunning,
  testGhostRendersAndAcceptsWhileRunning,
  testAtMentionPickerWhileRunning,
  testCompletionSurvivesRunEndingMidDraft,
  testEmptyCompletionsStillHidePickerWhileRunning,
  testProgrammaticTabSwitchDropsPickerAndGhost,
];

async function main() {
  let failed = 0;
  for (const t of tests) {
    try {
      await t();
      console.log('PASS', t.name);
    } catch (err) {
      failed += 1;
      console.error('FAIL', t.name);
      console.error(err && err.stack ? err.stack : err);
    }
  }
  if (failed) {
    console.error(failed + ' test(s) failed');
    process.exit(1);
  }
  console.log('All ' + tests.length + ' tests passed');
  // A running tab keeps the elapsed-time ticker alive; do not let it
  // pin the process open once every assertion has run.
  process.exit(0);
}

main();
