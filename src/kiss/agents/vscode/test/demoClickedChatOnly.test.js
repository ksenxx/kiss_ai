// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end regression tests for demo mode replaying the WRONG tasks
// (real chat.html + main.js + demo.js in jsdom, a recording
// acquireVsCodeApi stub standing in for the extension host / daemon).
// Reproduces the field report "demo mode keeps opening and replaying
// tabs of random tasks; I cannot hear any voice":
//
//   * Clicking a history row in demo mode replayed EVERY session in
//     the history (other chats, other workspaces, sub-agent rows)
//     instead of ONLY the CLICKED task — the demo-mode spec is "when
//     I click a task in the task history, it will replay only that
//     task" (see demoClickedTaskOnly.test.js for the follow-up spec
//     tightening from "the clicked chat's tasks" to "only the
//     clicked task").
//
//   * The replayed task issued ``resumeSession`` WITHOUT a
//     ``taskId``, so clicking an older task of a multi-task chat
//     replayed the latest task's events instead of its own events.
//
//   * Clicking another history row while a demo replay was already
//     active opened ANOTHER chat tab on every click ("keeps opening
//     tabs").
//
// Run directly with ``node``:
//
//     node src/kiss/agents/vscode/test/demoClickedChatOnly.test.js

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

function sleep(ms) {
  return new Promise(resolve => {
    setTimeout(resolve, ms);
  });
}

// A huge recorded prompt (like the multi-KB bug-hunt prompts that fill
// a real history) — the replay must handle such rows without choking.
const HUGE_PROMPT =
  'You are a bug-hunting agent working in a giant repository. ' +
  'Audit every file and reproduce every inconsistency you find. '.repeat(
    120,
  );

// Server sends history newest-first.  chat-A (the clicked chat) has
// two tasks; chat-B is a DIFFERENT chat (the "random tasks") with a
// top-level task and a sub-agent row.
const SESSIONS = [
  {
    id: 'chat-A',
    task_id: 'a2',
    preview: 'Task A2 follow-up',
    title: 'Task A2 follow-up',
    has_events: true,
    ts: 4000,
  },
  {
    id: 'chat-B',
    task_id: 'b-sub',
    parent_task_id: 'b1',
    preview: 'sub-agent task of chat B',
    title: 'sub-agent task of chat B',
    has_events: true,
    ts: 3000,
  },
  {
    id: 'chat-B',
    task_id: 'b1',
    preview: 'Completely unrelated bug-hunt task',
    title: 'Completely unrelated bug-hunt task',
    has_events: true,
    ts: 2000,
  },
  {
    id: 'chat-A',
    task_id: 'a1',
    preview: HUGE_PROMPT,
    title: 'Task A1 original',
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

async function testReplaysOnlyClickedTask() {
  const {win} = makeWebview();
  installAudio(win);
  installSpeech(win);

  const {resumes, done} = startDemoFlow(win, 'Task A1 original');
  await done;

  const chats = resumes.map(m => String(m.id));
  assert.deepStrictEqual(
    chats,
    ['chat-A'],
    'demo must replay ONLY the clicked task — never other chats, ' +
      'sub-agent rows, or the clicked chat\'s other tasks (got ' +
      JSON.stringify(chats) +
      ')',
  );
  assert.deepStrictEqual(
    resumes.map(m => String(m.taskId)),
    ['a1'],
    'the replayed task must load its OWN events via its taskId',
  );
  console.log('PASS: demo replays only the clicked task');
}

async function testNoPromptNarrationRequested() {
  const {win} = makeWebview();
  const players = installAudio(win);
  const spoken = installSpeech(win);

  const {done} = startDemoFlow(win, 'Task A1 original');
  await done;

  // The prompt display + "User said ..." narration step was removed
  // and demo mode never synthesizes speech: a replay with no ``talk``
  // tool calls must play NO audio at all — even for a huge recorded
  // prompt.
  assert.strictEqual(
    players.length,
    0,
    'demo replay must not narrate the recorded prompt (got ' +
      players.length +
      ' audio players)',
  );
  assert.strictEqual(
    spoken.length,
    0,
    'the robotic Web Speech voice must never narrate the prompt',
  );
  console.log('PASS: demo replay requests no prompt narration');
}

async function testClickWhileActiveOpensNoNewTab() {
  const {win} = makeWebview();
  installAudio(win);
  installSpeech(win);

  const {done} = startDemoFlow(win, 'Task A1 original');
  await sleep(300); // replay is now active, showing the first panel
  assert.ok(win._demoApi.active, 'demo replay is running');
  const tabsBefore = win.document.querySelectorAll(
    '#tab-list .chat-tab',
  ).length;

  // Re-open the sidebar and click other history rows mid-replay.
  const rows = Array.from(
    win.document.querySelectorAll('#history-list > div'),
  );
  const other = rows.find(
    r => r.textContent.indexOf('Completely unrelated bug-hunt task') !== -1,
  );
  assert.ok(other, 'other history row rendered');
  other.click();
  other.click();
  await sleep(100);

  const tabsAfter = win.document.querySelectorAll(
    '#tab-list .chat-tab',
  ).length;
  assert.strictEqual(
    tabsAfter,
    tabsBefore,
    'clicking history rows while a demo replay is active must NOT ' +
      'keep opening new chat tabs',
  );
  await done;
  console.log('PASS: no new tabs open while a demo replay is active');
}

(async () => {
  await testReplaysOnlyClickedTask();
  await testNoPromptNarrationRequested();
  await testClickWhileActiveOpensNoNewTab();
  console.log('demoClickedChatOnly.test.js: all tests passed');
  // Demo/webview timers can keep the node event loop alive; match the
  // other jsdom e2e demo tests and exit explicitly once assertions pass.
  process.exit(0);
})().catch(err => {
  console.error(err);
  process.exit(1);
});
