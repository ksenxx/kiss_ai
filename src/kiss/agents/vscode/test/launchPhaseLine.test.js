// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// The `launch_phase` event: between the prompt echo and the agent's first
// output the daemon reports what a launch is doing ("Classifying task…",
// "Preparing worktree…").  The webview shows ONE dim status line at the
// end of the transcript, replaces its text with each later phase, and
// removes it on an empty text or when the task ends.  A phase addressed to
// a background tab lands in that tab's parked transcript and shows up when
// the user switches to it; the visible tab never sees it.

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
  win._testApi.hideWelcome();
  win._testApi.endLaunch();
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

function phaseLines(win) {
  return Array.from(win.document.querySelectorAll('#output .launch-phase'));
}

function phaseText(win) {
  const lines = phaseLines(win);
  assert.strictEqual(lines.length, 1, 'exactly one launch-phase line is shown');
  return lines[0].textContent;
}

function testPhasesReplaceOneLineAndClear() {
  const {win} = makeWebview();
  const tabId = win._testApi.getActiveTabId();
  const O = win.document.getElementById('output');
  send(win, {type: 'status', running: true, tabId, startTs: Date.now()});
  send(win, {type: 'prompt', text: 'fix the bug', tabId, taskId: '', early: true});

  send(win, {type: 'launch_phase', text: 'Classifying task…', tabId});
  assert.strictEqual(phaseText(win), 'Classifying task…');
  assert.ok(
    phaseLines(win)[0].querySelector('.status-spinner'),
    'the line carries the shared spinner ring',
  );
  assert.strictEqual(O.lastElementChild, phaseLines(win)[0], 'the line is last');

  send(win, {type: 'launch_phase', text: 'Preparing worktree…', tabId});
  assert.strictEqual(phaseText(win), 'Preparing worktree…');
  assert.strictEqual(O.lastElementChild, phaseLines(win)[0], 'still the last child');

  send(win, {type: 'launch_phase', text: '', tabId});
  assert.strictEqual(phaseLines(win).length, 0, 'an empty text removes the line');

  // A phase that arrives after the transcript has grown still sits last.
  send(win, {type: 'launch_phase', text: 'Classifying task…', tabId});
  send(win, {type: 'notice', text: 'heads up', tabId});
  send(win, {type: 'launch_phase', text: 'Preparing worktree…', tabId});
  assert.strictEqual(O.lastElementChild, phaseLines(win)[0], 'moved after the notice');

  win.close();
  console.log('  ok - phases replace one line, keep it last, and clear on empty text');
}

function testTaskEndRemovesAnUnclearedLine() {
  const {win} = makeWebview();
  const tabId = win._testApi.getActiveTabId();
  send(win, {type: 'status', running: true, tabId, startTs: Date.now()});
  send(win, {type: 'launch_phase', text: 'Classifying task…', tabId});
  assert.strictEqual(phaseText(win), 'Classifying task…');
  // The launch failed before the agent started (no clearing phase).
  send(win, {type: 'status', running: false, tabId});
  assert.strictEqual(phaseLines(win).length, 0, 'the task end removes the line');
  win.close();
  console.log('  ok - a task ending removes an uncleared line');
}

function testBackgroundTabPhaseStaysOffScreenUntilSwitched() {
  const {win} = makeWebview();
  const api = win._testApi;
  const runningTab = api.getActiveTabId();
  api.createNewTab();
  const otherTab = api.getActiveTabId();
  assert.notStrictEqual(otherTab, runningTab, 'a second tab is on screen');

  send(win, {type: 'launch_phase', text: 'Preparing worktree…', tabId: runningTab});
  assert.strictEqual(phaseLines(win).length, 0, 'the visible tab shows no foreign phase');

  clickTab(win, runningTab);
  assert.strictEqual(api.getActiveTabId(), runningTab);
  assert.strictEqual(
    phaseText(win),
    'Preparing worktree…',
    'the phase travelled with the background transcript',
  );

  // The clearing phase for a tab that is now in the background removes
  // its parked line: switching back shows nothing.
  clickTab(win, otherTab);
  send(win, {type: 'launch_phase', text: '', tabId: runningTab});
  clickTab(win, runningTab);
  assert.strictEqual(phaseLines(win).length, 0, 'cleared while in the background');

  // A task ending in the background removes its parked line too.
  clickTab(win, otherTab);
  send(win, {type: 'launch_phase', text: 'Classifying task…', tabId: runningTab});
  send(win, {type: 'status', running: false, tabId: runningTab});
  clickTab(win, runningTab);
  assert.strictEqual(phaseLines(win).length, 0, 'removed by the background task end');

  win.close();
  console.log('  ok - a background tab keeps its phase off screen until switched to');
}

function testPhaseForUnknownTabIsIgnored() {
  const {win} = makeWebview();
  const tabId = win._testApi.getActiveTabId();
  send(win, {type: 'launch_phase', text: 'Classifying task…', tabId: 'no-such-tab'});
  assert.strictEqual(phaseLines(win).length, 0, 'an unknown tab draws nothing');
  // A closed tab's task ending must not clear the visible tab's line.
  send(win, {type: 'status', running: true, tabId, startTs: Date.now()});
  send(win, {type: 'launch_phase', text: 'Classifying task…', tabId});
  send(win, {type: 'status', running: false, tabId: 'no-such-tab'});
  assert.strictEqual(phaseText(win), 'Classifying task…', 'a foreign task end changes nothing');
  win.close();
  console.log('  ok - events for an unknown tab neither draw nor clear anything');
}

function runTests() {
  testPhasesReplaceOneLineAndClear();
  testTaskEndRemovesAnUnclearedLine();
  testBackgroundTabPhaseStaysOffScreenUntilSwitched();
  testPhaseForUnknownTabIsIgnored();
}

try {
  runTests();
  console.log('\n4 passed, 0 failed');
  process.exit(0);
} catch (err) {
  console.error('FAIL:', err && err.message ? err.message : err);
  process.exit(1);
}
