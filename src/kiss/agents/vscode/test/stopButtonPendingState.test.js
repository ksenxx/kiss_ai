// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end webview tests for the Stop button's pending state.
//
// Post-mortem reports/stop_button_delay_2026-08-05.html: the button only
// toggled its own visibility, so a stop the agent had not reached yet (it
// was inside a model request that produced nothing for 178 seconds) looked
// exactly like a stop the daemon never received -- and clicking again was
// the only sensible reaction. A stop the daemon could not route to any
// running task was discarded without a word too.
//
// The button must therefore say "Stopping" from the moment it is clicked,
// go back to normal when the daemon reports it found nothing to stop, and
// never carry that state into the next task.

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

function startRunningTask(win, posted) {
  const tabId = posted.find(m => m.type === 'ready').tabId;
  win._testApi.hideWelcome();
  send(win, {type: 'status', running: true, tabId});
  return tabId;
}

function stopBtn(win) {
  return win.document.getElementById('stop-btn');
}

function click(win, el) {
  el.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

function clickTab(win, tabId) {
  const el = Array.from(
    win.document.querySelectorAll('#tab-list .chat-tab'),
  ).find(e => e.dataset.tabId === tabId);
  assert.ok(el, 'tab ' + tabId + ' must be in the tab bar');
  click(win, el);
  assert.strictEqual(
    win.document.querySelector('#tab-list .chat-tab.active').dataset.tabId,
    tabId,
    'clicking tab ' + tabId + ' must activate it',
  );
}

function testClickShowsPendingState() {
  const {win, posted} = makeWebview();
  const tabId = startRunningTask(win, posted);
  const btn = stopBtn(win);
  assert.ok(!btn.classList.contains('stopping'), 'button starts idle');

  click(win, btn);

  assert.ok(
    posted.some(m => m.type === 'stop' && m.tabId === tabId),
    'the click must still send the stop command',
  );
  assert.ok(
    btn.classList.contains('stopping'),
    'a clicked Stop must show that the stop is pending, not sit there ' +
      'looking identical to a button that did nothing',
  );
  assert.match(btn.getAttribute('data-tooltip'), /Stopping/);
  win.close();
  console.log('PASS clicking Stop shows a pending state');
}

function testUnroutableStopClearsPendingState() {
  const {win, posted} = makeWebview();
  const tabId = startRunningTask(win, posted);
  const btn = stopBtn(win);
  click(win, btn);
  assert.ok(btn.classList.contains('stopping'));

  send(win, {type: 'stop_ack', accepted: false, tabId});

  assert.ok(
    !btn.classList.contains('stopping'),
    'a stop the daemon could not route must not leave the button pending',
  );
  assert.strictEqual(
    win.document.getElementById('status-text').textContent,
    'No running task to stop',
  );
  assert.strictEqual(
    btn.style.display,
    'none',
    'no running task owns the tab, so it must stop looking like it is ' +
      'running instead of contradicting the message',
  );
  win.close();
  console.log('PASS an unroutable stop is reported instead of discarded');
}

function testAcceptedStopKeepsPendingState() {
  const {win, posted} = makeWebview();
  const tabId = startRunningTask(win, posted);
  const btn = stopBtn(win);
  click(win, btn);

  send(win, {type: 'stop_ack', accepted: true, tabId});

  assert.ok(
    btn.classList.contains('stopping'),
    'an accepted stop stays pending until the task actually ends',
  );
  win.close();
  console.log('PASS an accepted stop stays pending until the task ends');
}

function testPendingStateEndsWithTheTask() {
  const {win, posted} = makeWebview();
  const tabId = startRunningTask(win, posted);
  const btn = stopBtn(win);
  click(win, btn);
  assert.ok(btn.classList.contains('stopping'));

  send(win, {type: 'status', running: false, tabId});
  send(win, {type: 'status', running: true, tabId});

  assert.ok(
    !btn.classList.contains('stopping'),
    'the pending state must not survive into the next task',
  );
  assert.strictEqual(btn.getAttribute('data-tooltip'), 'Stop agent');
  win.close();
  console.log('PASS the pending state does not survive the task');
}

function testBackgroundTabPendingStateEndsWithItsTask() {
  const {win, posted} = makeWebview();
  const tabId = startRunningTask(win, posted);
  const btn = stopBtn(win);
  click(win, btn);
  assert.ok(btn.classList.contains('stopping'));

  // The tab is no longer the one on screen when its task finishes.
  win._testApi.endLaunch();
  win._testApi.createNewTab();
  send(win, {type: 'status', running: false, tabId});
  send(win, {type: 'status', running: true, tabId});
  clickTab(win, tabId);

  assert.ok(
    !btn.classList.contains('stopping'),
    'a background tab that finished while stopping must not open its ' +
      'next task with the Stop button already pulsing',
  );
  win.close();
  console.log('PASS a background tab drops its pending stop when it ends');
}

testClickShowsPendingState();
testUnroutableStopClearsPendingState();
testAcceptedStopKeepsPendingState();
testPendingStateEndsWithTheTask();
testBackgroundTabPendingStateEndsWithItsTask();
