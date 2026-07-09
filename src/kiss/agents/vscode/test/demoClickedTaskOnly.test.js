// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end regression tests for demo mode replaying the WHOLE chat
// chain instead of just the clicked task (real chat.html + main.js +
// demo.js in jsdom, a recording acquireVsCodeApi stub standing in for
// the extension host / daemon).  Reproduces the report "when I click
// a task in the task history in demo mode, it replays the entire
// chain of tasks of that chat from oldest to newest — it must demo
// ONLY the task that was clicked":
//
//   * Clicking a history row of a multi-task chat replayed EVERY
//     top-level task of that chat, oldest first, instead of only the
//     clicked task.
//
//   * Clicking a sub-agent row must (still) replay just that
//     sub-agent task.
//
// Run directly with ``node``:
//
//     node src/kiss/agents/vscode/test/demoClickedTaskOnly.test.js

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

/**
 * Build a jsdom window running the production chat webview: the real
 * ``chat.html`` body, ``panelCopy.js``, ``main.js`` AND ``demo.js``
 * evaluated in the window, plus a recording ``acquireVsCodeApi`` stub.
 * ``win._onPosted`` (settable per test) observes every posted message
 * so tests can answer like the extension host / daemon would.
 */
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
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'demo.js'), 'utf8'));

  return {win, posted};
}

/**
 * Install a recording Audio implementation whose clips play and fire
 * ``onended`` after 15ms.  Returns the created players.
 */
function installAudio(win) {
  const players = [];
  win.Audio = function Audio(src) {
    this.src = src;
    players.push(this);
    this.play = () => {
      setTimeout(() => {
        if (typeof this.onended === 'function') this.onended();
      }, 15);
      return Promise.resolve();
    };
  };
  return players;
}

/** Install a no-op Web Speech API so fallbacks never hang. */
function installSpeech(win) {
  const spoken = [];
  win.SpeechSynthesisUtterance = function (text) {
    this.text = text;
  };
  win.speechSynthesis = {
    getVoices: () => [],
    speak: u => {
      spoken.push(u);
      setTimeout(() => {
        if (typeof u.onend === 'function') u.onend();
      }, 5);
    },
    cancel: () => {},
    resume: () => {},
  };
  return spoken;
}

/** Deliver a daemon/extension-host message to the webview. */
function dispatch(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

/** Answer every posted 'demoSpeak' with a synthesized clip after 10ms. */
function autoAnswerDemoSpeak(win) {
  const prev = win._onPosted;
  win._onPosted = msg => {
    if (prev) prev(msg);
    if (msg.type !== 'demoSpeak') return;
    setTimeout(() => {
      dispatch(win, {
        type: 'demoSpeakAudio',
        reqId: msg.reqId,
        audioB64: 'QUJD',
        audioMime: 'audio/mpeg',
        tabId: msg.tabId,
      });
    }, 10);
  };
}

function sleep(ms) {
  return new Promise(resolve => {
    setTimeout(resolve, ms);
  });
}

// Server sends history newest-first.  chat-A has THREE top-level
// tasks (a1 oldest, a2, a3 newest); chat-B has a top-level task with
// a sub-agent row.
const SESSIONS = [
  {
    id: 'chat-A',
    task_id: 'a3',
    preview: 'Task A3 newest',
    title: 'Task A3 newest',
    has_events: true,
    ts: 6000,
  },
  {
    id: 'chat-B',
    task_id: 'b-sub',
    parent_task_id: 'b1',
    preview: 'sub-agent task of chat B',
    title: 'sub-agent task of chat B',
    has_events: true,
    ts: 5000,
  },
  {
    id: 'chat-B',
    task_id: 'b1',
    preview: 'Chat B top-level task',
    title: 'Chat B top-level task',
    has_events: true,
    ts: 4000,
  },
  {
    id: 'chat-A',
    task_id: 'a2',
    preview: 'Task A2 middle',
    title: 'Task A2 middle',
    has_events: true,
    ts: 3000,
  },
  {
    id: 'chat-A',
    task_id: 'a1',
    preview: 'Task A1 oldest',
    title: 'Task A1 oldest',
    has_events: true,
    ts: 1000,
  },
];

/** Small replayed event stream ending in a result. */
function eventsFor(label) {
  return [
    {type: 'thinking_start'},
    {type: 'thinking_delta', text: 'working on ' + label},
    {type: 'result', summary: label + ' done.', total_tokens: 1, cost: '$0'},
  ];
}

/**
 * Enable demo mode, deliver SESSIONS, auto-answer resumeSession with
 * per-task events, click the history row whose title is *rowTitle*.
 * Returns {resumes, done} — the recorded resumeSession posts and a
 * promise resolving when the replay ends.
 */
function startDemoFlow(win, rowTitle) {
  dispatch(win, {type: 'updateSetting', key: 'demo_mode', value: true});
  dispatch(win, {type: 'history', offset: 0, generation: 0, sessions: SESSIONS});

  const resumes = [];
  const prev = win._onPosted;
  win._onPosted = msg => {
    if (prev) prev(msg);
    if (msg.type !== 'resumeSession') return;
    resumes.push(msg);
    setTimeout(() => {
      dispatch(win, {
        type: 'task_events',
        tabId: msg.tabId,
        events: eventsFor(String(msg.taskId || msg.id)),
        task: 'task ' + String(msg.taskId || msg.id),
        chat_id: msg.id,
        extra: '',
      });
    }, 10);
  };

  const rows = Array.from(win.document.querySelectorAll('#history-list > div'));
  const row = rows.find(r => r.textContent.indexOf(rowTitle) !== -1);
  assert.ok(row, 'history row "' + rowTitle + '" rendered');
  row.click();

  const done = (async () => {
    const t0 = Date.now();
    while (Date.now() - t0 < 30000) {
      await sleep(50);
      if (!win._demoApi.active && Date.now() - t0 > 500) return;
    }
    throw new Error('demo replay did not finish within 30s');
  })();
  return {resumes, done};
}

async function testClickingMiddleTaskReplaysOnlyThatTask() {
  const {win} = makeWebview();
  installAudio(win);
  installSpeech(win);
  autoAnswerDemoSpeak(win);

  const {resumes, done} = startDemoFlow(win, 'Task A2 middle');
  await done;

  assert.deepStrictEqual(
    resumes.map(m => [String(m.id), String(m.taskId)]),
    [['chat-A', 'a2']],
    'demo must replay ONLY the clicked task — not the whole chain ' +
      'of tasks of the chat from oldest to newest (got ' +
      JSON.stringify(resumes.map(m => String(m.taskId))) +
      ')',
  );
  win.close();
  console.log('PASS: clicking a middle task replays only that task');
}

async function testClickingNewestTaskReplaysOnlyThatTask() {
  const {win} = makeWebview();
  installAudio(win);
  installSpeech(win);
  autoAnswerDemoSpeak(win);

  const {resumes, done} = startDemoFlow(win, 'Task A3 newest');
  await done;

  assert.deepStrictEqual(
    resumes.map(m => [String(m.id), String(m.taskId)]),
    [['chat-A', 'a3']],
    'demo must replay ONLY the clicked (newest) task, never its ' +
      'older siblings (got ' +
      JSON.stringify(resumes.map(m => String(m.taskId))) +
      ')',
  );
  win.close();
  console.log('PASS: clicking the newest task replays only that task');
}

async function testClickingSubagentRowReplaysOnlyThatSubagentTask() {
  const {win} = makeWebview();
  installAudio(win);
  installSpeech(win);
  autoAnswerDemoSpeak(win);

  const {resumes, done} = startDemoFlow(win, 'sub-agent task of chat B');
  await done;

  assert.deepStrictEqual(
    resumes.map(m => [String(m.id), String(m.taskId)]),
    [['chat-B', 'b-sub']],
    'clicking a sub-agent row must replay just that sub-agent task ' +
      '(got ' +
      JSON.stringify(resumes.map(m => String(m.taskId))) +
      ')',
  );
  win.close();
  console.log('PASS: clicking a sub-agent row replays only that task');
}

(async () => {
  await testClickingMiddleTaskReplaysOnlyThatTask();
  await testClickingNewestTaskReplaysOnlyThatTask();
  await testClickingSubagentRowReplaysOnlyThatSubagentTask();
  console.log('demoClickedTaskOnly.test.js: all tests passed');
  // Demo/webview timers can keep the node event loop alive; match the
  // other jsdom e2e demo tests and exit explicitly once assertions pass.
  process.exit(0);
})().catch(err => {
  console.error(err);
  process.exit(1);
});
