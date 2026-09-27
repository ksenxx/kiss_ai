// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// The daemon broadcasts the followup_suggestion ("Suggested next") event
// only after the whole task lifecycle -- auto-commit and worktree merge --
// has finished (server/task_runner.py), which in production arrived 18 s
// after task_done.  When the task wrote a reports/**.html file, task_done
// opens that report in a content tab that takes focus, so by the time the
// bar arrives the task's own chat tab is in the background.  The webview
// used to keep the bar only when the task tab was the active tab, so the
// live bar was silently lost.
//
// These end-to-end jsdom tests replay that exact sequence with a fake
// clock advanced by 18 s and check that the bar is parked in the task
// tab's own transcript and shown when the user returns to it, without
// leaking into any other tab and without being shown twice.

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');
const LATE_MS = 18000;

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
  // A controllable wall clock so the 18 s gap is real webview time
  // without the test having to wait for it.
  const clock = {now: 1790457672532};
  win.Date.now = function () {
    return clock.now;
  };
  win.acquireVsCodeApi = function () {
    let state;
    return {
      postMessage: function () {},
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };
  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
  return {win, clock};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function tabEl(win, tabId) {
  return win.document.querySelector(
    `.chat-tab[data-tab-id=${JSON.stringify(tabId)}]`,
  );
}

function clickTab(win, tabId) {
  const el = tabEl(win, tabId);
  assert.ok(el, `tab ${tabId} must exist in the tab bar`);
  el.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

function outputBars(win) {
  return Array.from(win.document.querySelectorAll('#output .followup-bar'));
}

function activeContentTab(win) {
  return win.document.querySelector('#tab-list .chat-tab.content-tab.active');
}

// A task in tab *tabId* writes an HTML report and finishes; task_done
// opens the report as a content tab that takes focus.
function runTaskWithReport(win, clock, tabId, taskId) {
  send(win, {type: 'clear', chat_id: 'chat-' + taskId, tabId});
  send(win, {type: 'task_events', tabId, task_id: taskId});
  const file = 'reports/' + taskId + '/index.html';
  const body = '<!DOCTYPE html><html><body><h1>' + taskId + '</h1></body></html>';
  send(win, {type: 'tool_call', name: 'Write', path: file, content: body, tabId, taskId});
  send(win, {
    type: 'tool_result',
    content: 'Successfully wrote ' + body.length + ' characters to ' + file,
    is_error: false,
    tool_name: 'Write',
    path: file,
    tabId,
    taskId,
  });
  send(win, {type: 'task_done', success: true, tabId, taskId, ts: clock.now});
}

function lateFollowup(win, clock, tabId, text) {
  clock.now += LATE_MS;
  send(win, {type: 'followup_suggestion', tabId, text, ts: clock.now});
}

let passed = 0;
const failures = [];
function test(name, fn) {
  try {
    fn();
    passed++;
    console.log(`  \u2713 ${name}`);
  } catch (e) {
    failures.push({name, error: e});
    console.log(`  \u2717 ${name}`);
    console.log(`      ${e.stack || e.message}`);
  }
}

test('a bar arriving 18 s after task_done survives the report tab taking focus', () => {
  const {win, clock} = makeWebview();
  const api = win._testApi;
  const taskTab = api.getActiveTabId();

  runTaskWithReport(win, clock, taskTab, 'task-late-QX01');
  assert.ok(
    activeContentTab(win),
    'task_done must have opened the report tab and made it active',
  );
  assert.notStrictEqual(api.getActiveTabId(), taskTab, 'the task tab is now in the background');

  lateFollowup(win, clock, taskTab, 'late_next_step_QX01');
  assert.strictEqual(
    outputBars(win).length,
    0,
    'the bar must not be painted on the report surface',
  );

  clickTab(win, taskTab);
  const bars = outputBars(win);
  assert.strictEqual(
    bars.length,
    1,
    'BUG: the Suggested next bar that arrived while the report tab had ' +
      'focus was dropped instead of appended to the task tab transcript',
  );
  assert.ok(bars[0].textContent.includes('late_next_step_QX01'));
  // The bar follows the task's panels (the welcome screen div that
  // #output always keeps at its end is not a transcript entry).
  const panels = win.document.querySelectorAll('#output .collapsible');
  assert.ok(panels.length > 0, 'the transcript panels were restored too');
  assert.ok(
    panels[panels.length - 1].compareDocumentPosition(bars[0]) &
      win.Node.DOCUMENT_POSITION_FOLLOWING,
    'the bar must come after the last transcript panel',
  );

  // The bar keeps its click behaviour after being parked and restored.
  bars[0].dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.strictEqual(
    win.document.getElementById('task-input').value,
    'late_next_step_QX01',
    'clicking the restored bar must copy the suggestion into the input',
  );
  win.close();
});

test('the bar is shown once when the task tab is still active', () => {
  const {win, clock} = makeWebview();
  const api = win._testApi;
  const taskTab = api.getActiveTabId();
  send(win, {type: 'clear', chat_id: 'chat-QX02', tabId: taskTab});
  send(win, {type: 'task_events', tabId: taskTab, task_id: 'task-QX02'});
  send(win, {type: 'task_done', success: true, tabId: taskTab, ts: clock.now});

  lateFollowup(win, clock, taskTab, 'active_next_step_QX02');
  assert.strictEqual(outputBars(win).length, 1, 'exactly one bar on the active tab');
  assert.ok(outputBars(win)[0].textContent.includes('active_next_step_QX02'));
  win.close();
});

test('a late bar never leaks into an unrelated chat tab', () => {
  const {win, clock} = makeWebview();
  const api = win._testApi;
  const taskTab = api.getActiveTabId();
  runTaskWithReport(win, clock, taskTab, 'task-QX03');

  api.createNewTab();
  const otherTab = api.getActiveTabId();
  assert.notStrictEqual(otherTab, taskTab);
  send(win, {type: 'clear', chat_id: 'chat-other-QX03', tabId: otherTab});

  lateFollowup(win, clock, taskTab, 'owner_only_QX03');
  assert.strictEqual(
    outputBars(win).length,
    0,
    'the other chat tab must not show the bar of a task it never ran',
  );

  clickTab(win, taskTab);
  assert.strictEqual(outputBars(win).length, 1, 'the owner tab shows its bar');
  assert.ok(outputBars(win)[0].textContent.includes('owner_only_QX03'));

  clickTab(win, otherTab);
  assert.strictEqual(
    outputBars(win).length,
    0,
    'switching back must not carry the bar into the other tab',
  );
  win.close();
});

test('a late bar for a closed tab is dropped without error', () => {
  const {win, clock} = makeWebview();
  const api = win._testApi;
  const taskTab = api.getActiveTabId();
  api.createNewTab();
  const survivor = api.getActiveTabId();
  send(win, {type: 'clear', chat_id: 'chat-QX04', tabId: survivor});

  clock.now += LATE_MS;
  send(win, {
    type: 'followup_suggestion',
    tabId: 'tab-that-was-closed-QX04',
    text: 'orphan_QX04',
    ts: clock.now,
  });
  assert.strictEqual(outputBars(win).length, 0, 'no tab owns the bar');
  clickTab(win, taskTab);
  assert.strictEqual(outputBars(win).length, 0, 'no tab owns the bar');
  win.close();
});

console.log(`\n${passed} passed, ${failures.length} failed`);
if (failures.length) process.exit(1);
