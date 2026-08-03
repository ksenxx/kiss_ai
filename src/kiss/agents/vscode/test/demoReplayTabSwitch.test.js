// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Demo replay types a Result panel into the output surface one word at a
// time, so it lives across many awaits. #output is a SINGLETON: on a tab
// switch main.js moves its children into the outgoing tab's detached
// fragment and hangs the incoming tab's children off the same element. So a
// replay that appends to whatever #output holds -- or even to the #output
// element it captured earlier -- writes one session's result into whichever
// conversation the user switched to.
//
// This test switches tabs at the two moments that matter: while the replay is
// awaiting its events, and while the Result panel is still being typed.

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
      postMessage: msg => {
        posted.push(msg);
        if (win._onPosted) win._onPosted(msg);
      },
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };

  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'demo.js'), 'utf8'));

  return {win, posted};
}

function dispatch(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function sleep(ms) {
  return new Promise(resolve => {
    setTimeout(resolve, ms);
  });
}

const MARKER = 'DEMO_RESULT_LEAK_QK95';

const REPLAY_EVENTS = [
  {type: 'system_output', text: 'demo replay running'},
  {
    type: 'result',
    summary: MARKER + ' ' + new Array(40).join('word '),
    total_tokens: 10,
    cost: '$0.01',
  },
];

function startHistoryReplay(win, onResumeSession) {
  dispatch(win, {type: 'configData', config: {demo_mode: true}, apiKeys: {}});
  dispatch(win, {
    type: 'history',
    offset: 0,
    generation: 0,
    sessions: [
      {
        id: 'chat-QK95',
        preview: 'Do the demo task',
        title: 'Do the demo task',
        has_events: true,
        ts: Date.now() / 1000,
      },
    ],
  });
  const prev = win._onPosted;
  win._onPosted = msg => {
    if (prev) prev(msg);
    if (msg.type !== 'resumeSession') return;
    onResumeSession(msg);
  };
  const row = win.document.querySelector('#history-list > div');
  assert.ok(row, 'a history row must render');
  row.click();
}

async function waitForReplayEnd(win) {
  const t0 = Date.now();
  while (Date.now() - t0 < 20000) {
    await sleep(20);
    if (!win._demoApi.active && Date.now() - t0 > 300) return;
  }
  throw new Error('demo replay did not finish within 20s');
}

function switchToTab(win, tabId) {
  const el = win.document.querySelector(
    '.chat-tab[data-tab-id="' + tabId + '"]',
  );
  assert.ok(el, 'tab element must exist for ' + tabId);
  el.click();
}

function visibleText(win) {
  return win.document.getElementById('output').textContent;
}

// A tab opened in the instant between the events arriving and the replay
// resuming after its await. The replay used to resolve its output surface
// only after that await, so this is the moment that steered the whole Result
// panel into the wrong conversation.
async function testSwitchDuringEventFetch() {
  const {win} = makeWebview();
  const api = win._demoApi;
  // Clicking a history row in demo mode opens a NEW tab for the replayed
  // session, so the conversation the replay owns is the one named on the
  // resumeSession request -- not whichever tab was on screen before.
  let replayTab = null;
  let otherTab = null;

  startHistoryReplay(win, msg => {
    replayTab = msg.tabId;
    setTimeout(() => {
      dispatch(win, {
        type: 'task_events',
        tabId: msg.tabId,
        events: REPLAY_EVENTS,
        task: 'Do the demo task',
        chat_id: 'chat-QK95',
        extra: '',
      });
      // The events have been handed to the replay but it has not resumed
      // yet: the user opens a second conversation in that gap.
      api.createNewTab();
      otherTab = api.getActiveTabId();
    }, 10);
  });

  await waitForReplayEnd(win);

  assert.ok(otherTab && otherTab !== replayTab, 'a second tab must exist');
  assert.strictEqual(
    api.getActiveTabId(),
    otherTab,
    'the second tab must still be the one on screen',
  );
  assert.ok(
    !visibleText(win).includes(MARKER),
    'a demo Result panel replayed for another tab must never appear in the ' +
      'conversation the user switched to',
  );

  switchToTab(win, replayTab);
  assert.ok(
    visibleText(win).includes(MARKER),
    'it must have been written into the tab the replay belongs to instead',
  );

  win.close();
  console.log('  ok - tab switch during the replay event fetch leaks nothing');
}

// A tab opened while the Result panel is still being typed word by word.
async function testSwitchWhileTypingResult() {
  const {win} = makeWebview();
  const api = win._demoApi;
  let replayTab = null;
  let otherTab = null;

  startHistoryReplay(win, msg => {
    replayTab = msg.tabId;
    setTimeout(() => {
      dispatch(win, {
        type: 'task_events',
        tabId: msg.tabId,
        events: REPLAY_EVENTS,
        task: 'Do the demo task',
        chat_id: 'chat-QK95',
        extra: '',
      });
      // Switch tabs mid-typing: the panel exists but is still growing.
      setTimeout(() => {
        api.createNewTab();
        otherTab = api.getActiveTabId();
      }, 25);
    }, 10);
  });

  await waitForReplayEnd(win);

  assert.ok(otherTab && otherTab !== replayTab, 'a second tab must exist');
  assert.ok(
    !visibleText(win).includes(MARKER),
    'the tail of a Result panel must not spill into the tab the user ' +
      'switched to while it was being typed',
  );

  switchToTab(win, replayTab);
  assert.ok(
    visibleText(win).includes(MARKER),
    'the whole panel must stay in the tab the replay belongs to',
  );

  win.close();
  console.log('  ok - tab switch while typing the Result panel leaks nothing');
}

async function main() {
  await testSwitchDuringEventFetch();
  await testSwitchWhileTypingResult();
  console.log('demoReplayTabSwitch.test.js: all assertions passed.');
}

main().catch(err => {
  console.error(err && err.stack ? err.stack : err);
  process.exit(1);
});
