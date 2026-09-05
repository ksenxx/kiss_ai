// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// Clicking a "Suggested next" bar must copy the suggested prompt into
// the chat input box — on EVERY surface the bar is rendered on:
//
//  * the live stream (followup_suggestion event of the running task);
//  * the active tab's transcript replay (task_events) after a webview
//    reload;
//  * a background tab's transcript replay after a reload, once the user
//    switches to that tab;
//  * an earlier task of the chat spliced in by overscrolling to the top
//    (adjacent_task_events -> replayDetachedTranscript).
//
// The reported bug: after a chat webview reload, bars of earlier tasks
// (the adjacent-transcript surface) were rendered without a click
// handler, so clicking them silently did nothing.  The welcome screen's
// "Suggested prompt" chips share the same copy-to-input behavior and
// are covered here too.

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
      postMessage: (msg) => posted.push(msg),
      getState: () => state,
      setState: (s) => {
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

function click(win, el) {
  el.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

function inputBox(win) {
  return win.document.getElementById('task-input');
}

/** Click *bar* and assert the input box now holds *text*. */
function assertClickCopies(win, bar, text, surface) {
  assert.ok(bar, surface + ': the "Suggested next" bar must be rendered');
  assert.strictEqual(
    bar.querySelector('.fu-text').textContent,
    text,
    surface + ': the bar must show the suggested prompt',
  );
  inputBox(win).value = '';
  click(win, bar);
  assert.strictEqual(
    inputBox(win).value,
    text,
    surface + ': clicking the bar must copy the prompt into the input box',
  );
}

function testLiveFollowupClickCopies() {
  const {win, posted} = makeWebview();
  const tabId = posted.find((m) => m.type === 'ready').tabId;
  win._testApi.hideWelcome();
  send(win, {type: 'followup_suggestion', text: 'live next step', tabId});
  const bar = win.document.querySelector('#output .followup-bar');
  assertClickCopies(win, bar, 'live next step', 'live stream');
  win.close();
  console.log('PASS live followup bar click copies to input');
}

function testActiveTabReplayFollowupClickCopies() {
  const {win, posted} = makeWebview();
  const tabId = posted.find((m) => m.type === 'ready').tabId;
  win._testApi.hideWelcome();
  // A webview reload replays the newest task through task_events.
  send(win, {
    type: 'task_events',
    tabId,
    chat_id: 'chat-1',
    task_id: '42',
    task: 'My reloaded task',
    events: [
      {type: 'task_start', task: 'My reloaded task'},
      {type: 'system_output', text: 'did things\n'},
      {type: 'followup_suggestion', text: 'replayed next step'},
      {type: 'task_done'},
    ],
  });
  const bar = win.document.querySelector('#output .followup-bar');
  assertClickCopies(win, bar, 'replayed next step', 'active-tab replay');
  win.close();
  console.log('PASS active-tab replay followup bar click copies to input');
}

function testAdjacentTaskFollowupClickCopies() {
  const {win, posted} = makeWebview();
  const tabId = posted.find((m) => m.type === 'ready').tabId;
  win._testApi.hideWelcome();
  // The reload scenario of the bug report: the newest task is replayed,
  // then the user scrolls up and an EARLIER task of the chat is spliced
  // in as a detached adjacent transcript.
  send(win, {
    type: 'task_events',
    tabId,
    chat_id: 'chat-1',
    task_id: '42',
    task: 'Newest task',
    events: [
      {type: 'task_start', task: 'Newest task'},
      {type: 'system_output', text: 'newest output\n'},
    ],
  });
  send(win, {
    type: 'adjacent_task_events',
    tabId,
    direction: 'prev',
    task: 'Older task',
    task_id: '41',
    events: [
      {type: 'task_start', task: 'Older task'},
      {type: 'system_output', text: 'older output\n'},
      {type: 'followup_suggestion', text: 'older suggested step'},
      {type: 'task_done'},
    ],
  });
  const cont = win.document.querySelector(
    '#output .adjacent-task[data-task-id="41"]',
  );
  assert.ok(cont, 'the earlier task must be spliced into the chat');
  const bar = cont.querySelector('.followup-bar');
  assertClickCopies(
    win,
    bar,
    'older suggested step',
    'adjacent transcript (earlier task after reload)',
  );
  win.close();
  console.log('PASS adjacent-task followup bar click copies to input');
}

function testBackgroundTabReplayFollowupClickCopies() {
  const {win, posted} = makeWebview();
  const firstTabId = posted.find((m) => m.type === 'ready').tabId;
  win._testApi.hideWelcome();
  win._testApi.endLaunch();
  // Open a second tab; the first becomes a background tab whose
  // transcript then arrives (as after a reload of a multi-tab webview).
  win._testApi.createNewTab();
  assert.notStrictEqual(win._testApi.getActiveTabId(), firstTabId);
  send(win, {
    type: 'task_events',
    tabId: firstTabId,
    chat_id: 'chat-1',
    task_id: '42',
    task: 'Background task',
    events: [
      {type: 'task_start', task: 'Background task'},
      {type: 'system_output', text: 'bg output\n'},
      {type: 'followup_suggestion', text: 'bg next step'},
      {type: 'task_done'},
    ],
  });
  // Switch back to the first tab via its tab strip, like a user would.
  const strip = win.document.querySelector(
    `.chat-tab[data-tab-id="${firstTabId}"]`,
  );
  assert.ok(strip, 'the background tab must have a strip in the tab bar');
  click(win, strip);
  assert.strictEqual(win._testApi.getActiveTabId(), firstTabId);
  const bar = win.document.querySelector('#output .followup-bar');
  assertClickCopies(win, bar, 'bg next step', 'background-tab replay');
  win.close();
  console.log('PASS background-tab replay followup bar click copies to input');
}

function testWelcomeSuggestionChipClickCopies() {
  const {win} = makeWebview();
  send(win, {
    type: 'welcome_suggestions',
    suggestions: [{text: 'try this prompt'}],
  });
  const chip = win.document.querySelector('#suggestions .suggestion-chip');
  assert.ok(chip, 'the welcome suggestion chip must be rendered');
  click(win, chip);
  assert.strictEqual(
    inputBox(win).value,
    'try this prompt',
    'clicking a welcome suggestion chip must copy the prompt into the ' +
      'input box',
  );
  win.close();
  console.log('PASS welcome suggestion chip click copies to input');
}

testLiveFollowupClickCopies();
testActiveTabReplayFollowupClickCopies();
testAdjacentTaskFollowupClickCopies();
testBackgroundTabReplayFollowupClickCopies();
testWelcomeSuggestionChipClickCopies();
console.log('All tests passed');
