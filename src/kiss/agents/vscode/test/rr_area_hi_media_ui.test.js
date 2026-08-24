// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Webview regressions for the area-I redundancy fixes:
// - I-R1: every tab-title write clips through clipTabTitle (30 chars +
//   ellipsis), for the active tab (setTaskText) and background tabs
//   (task_events) alike.
// - I-R4: the "Suggested next" followup bar renders and inserts its
//   text on click, both live (followup_suggestion event) and replayed
//   (task_events history).
// - I-RC2: replacing the input history cache resets a live ArrowUp/
//   ArrowDown cycling position instead of resuming at a stale index
//   into the new list.

/* global require, process, console, __dirname, global, setTimeout, clearTimeout */

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

function allTabTitles(win) {
  return Array.from(win.document.querySelectorAll('[role="tab"]')).map(el =>
    el.getAttribute('aria-label'),
  );
}

function key(win, el, k) {
  el.dispatchEvent(
    new win.KeyboardEvent('keydown', {key: k, bubbles: true, cancelable: true}),
  );
}

const LONG = 'x'.repeat(45);
const CLIPPED = 'x'.repeat(30) + '\u2026';

function testActiveTabTitleClipped() {
  const {win, posted} = makeWebview();
  const tabId = posted.find(m => m.type === 'ready').tabId;
  send(win, {type: 'setTaskText', tabId, text: LONG});
  assert.ok(
    allTabTitles(win).includes(CLIPPED),
    `active tab title not clipped: ${JSON.stringify(allTabTitles(win))}`,
  );
  console.log('  ok - setTaskText clips the active tab title');
}

function testBackgroundTabTitleClipped() {
  const {win, posted} = makeWebview();
  const tab1 = posted.find(m => m.type === 'ready').tabId;
  win._testApi.createNewTab(); // becomes the active tab
  send(win, {type: 'task_events', tabId: tab1, task: LONG, events: []});
  assert.ok(
    allTabTitles(win).includes(CLIPPED),
    `background tab title not clipped: ${JSON.stringify(allTabTitles(win))}`,
  );
  console.log('  ok - task_events clips a background tab title');
}

function testFollowupBarLive() {
  const {win, posted} = makeWebview();
  const tabId = posted.find(m => m.type === 'ready').tabId;
  win._testApi.hideWelcome();
  send(win, {type: 'followup_suggestion', tabId, text: 'Do X next'});
  const bar = win.document.querySelector('#output .followup-bar');
  assert.ok(bar, 'live followup bar not rendered');
  assert.strictEqual(bar.querySelector('.fu-text').textContent, 'Do X next');
  assert.strictEqual(
    bar.querySelector('.fu-label').textContent,
    'Suggested next',
  );
  bar.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.strictEqual(
    win.document.getElementById('task-input').value,
    'Do X next',
    'clicking the live followup bar must insert its text',
  );
  console.log('  ok - live followup bar renders and inserts on click');
}

function testFollowupBarReplay() {
  const {win, posted} = makeWebview();
  const tabId = posted.find(m => m.type === 'ready').tabId;
  win._testApi.hideWelcome();
  send(win, {
    type: 'task_events',
    tabId,
    chat_id: 'chat-r',
    task_id: '7',
    task: 'Old task',
    events: [
      {type: 'task_start', task: 'Old task'},
      {type: 'system_output', text: 'did things\n'},
      {type: 'followup_suggestion', text: 'Replay me'},
    ],
  });
  const bar = win.document.querySelector('.followup-bar');
  assert.ok(bar, 'replayed followup bar not rendered');
  assert.strictEqual(bar.querySelector('.fu-text').textContent, 'Replay me');
  bar.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.strictEqual(
    win.document.getElementById('task-input').value,
    'Replay me',
    'clicking the replayed followup bar must insert its text',
  );
  console.log('  ok - replayed followup bar renders and inserts on click');
}

function testInputHistoryReplaceResetsCycling() {
  const {win} = makeWebview();
  const inp = win.document.getElementById('task-input');

  send(win, {type: 'inputHistory', tasks: ['old-0', 'old-1', 'old-2']});
  key(win, inp, 'ArrowUp');
  assert.strictEqual(inp.value, 'old-0', 'first ArrowUp recalls newest');
  key(win, inp, 'ArrowUp');
  assert.strictEqual(inp.value, 'old-1', 'second ArrowUp goes further back');

  // The daemon pushes a fresh history while the user is mid-cycle.
  send(win, {type: 'inputHistory', tasks: ['new-0']});

  // A stale index would now step through the NEW list from position 1.
  key(win, inp, 'ArrowDown');
  assert.strictEqual(
    inp.value,
    'old-1',
    'ArrowDown after a history replace must not jump into the new list',
  );

  // Cycling restarts cleanly from the top of the new list.
  inp.value = '';
  key(win, inp, 'ArrowUp');
  assert.strictEqual(
    inp.value,
    'new-0',
    'ArrowUp after a history replace starts at the new newest entry',
  );
  console.log('  ok - inputHistory replace resets the cycling position');
}

function main() {
  testActiveTabTitleClipped();
  testBackgroundTabTitleClipped();
  testFollowupBarLive();
  testFollowupBarReplay();
  testInputHistoryReplaceResetsCycling();
  console.log('rr_area_hi_media_ui: all tests passed');
  process.exit(0);
}

try {
  main();
} catch (err) {
  console.error(err);
  process.exit(1);
}
