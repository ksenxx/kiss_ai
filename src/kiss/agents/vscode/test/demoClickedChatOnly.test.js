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
//     instead of the tasks of the CLICKED chat session — the original
//     demo-mode spec is "when I click a task in the task history, it
//     will replay the tasks in the chat session ... for each task
//     starting from the first task".
//
//   * Each replayed task issued ``resumeSession`` WITHOUT a
//     ``taskId``, so a multi-task chat replayed the latest task's
//     events over and over instead of each task's own events.
//
//   * Clicking another history row while a demo replay was already
//     active opened ANOTHER chat tab on every click ("keeps opening
//     tabs").
//
//   * Narrating a huge recorded prompt verbatim made the GPT-audio
//     synthesis slow/failing (silence) — narration must be capped to
//     a short lead-in so the demo voice stays fast and reliable.
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

/** Answer every posted 'demoSpeak' with a synthesized clip after 10ms. */
function autoAnswerDemoSpeak(win) {
  autoAnswerDemoSpeakWith(win, () => 'QUJD');
}

/**
 * Answer posted 'demoSpeak' requests with audio selected by *choose*.
 * Returning '' simulates GPT-audio synthesis failure/timeout; in the
 * VS Code webview that degrades to SILENCE, reproducing the reported
 * "I cannot hear any voice" behavior for huge uncapped prompts.
 */
function autoAnswerDemoSpeakWith(win, choose) {
  const prev = win._onPosted;
  win._onPosted = msg => {
    if (prev) prev(msg);
    if (msg.type !== 'demoSpeak') return;
    setTimeout(() => {
      dispatch(win, {
        type: 'demoSpeakAudio',
        reqId: msg.reqId,
        audioB64: choose(msg),
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

// A huge recorded prompt (like the multi-KB bug-hunt prompts that fill
// a real history) — narrating this verbatim is exactly what made the
// demo voice time out into silence.
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

async function testReplaysOnlyClickedChatTasks() {
  const {win} = makeWebview();
  installAudio(win);
  installSpeech(win);
  autoAnswerDemoSpeak(win);

  const {resumes, done} = startDemoFlow(win, 'Task A1 original');
  await done;

  const chats = resumes.map(m => String(m.id));
  assert.deepStrictEqual(
    chats,
    ['chat-A', 'chat-A'],
    'demo must replay ONLY the clicked chat session, oldest task ' +
      'first — never other chats or sub-agent rows (got ' +
      JSON.stringify(chats) +
      ')',
  );
  assert.deepStrictEqual(
    resumes.map(m => String(m.taskId)),
    ['a1', 'a2'],
    "each replayed task must load its OWN events via its taskId, " +
      'oldest first',
  );
  console.log('PASS: demo replays only the clicked chat session tasks');
}

async function testNarrationIsCappedAndAudible() {
  const {win, posted} = makeWebview();
  const players = installAudio(win);
  installSpeech(win);
  // Simulate the real failure mode: huge demoSpeak scripts fail (or
  // time out) and the VS Code webview intentionally degrades to
  // silence.  After the fix, the narration is capped before synthesis
  // and therefore gets an Audio clip.
  autoAnswerDemoSpeakWith(win, msg => (msg.text.length <= 320 ? 'QUJD' : ''));

  const {done} = startDemoFlow(win, 'Task A1 original');
  await done;

  const speaks = posted.filter(m => m.type === 'demoSpeak');
  assert.ok(speaks.length > 0, 'demo narration requested synthesis');
  for (const msg of speaks) {
    assert.ok(
      msg.text.length <= 320,
      'narration must be capped to a short lead-in so GPT-audio ' +
        'synthesis stays fast and audible (got ' +
        msg.text.length +
        ' chars)',
    );
  }
  assert.strictEqual(
    players.length,
    speaks.length,
    'every demo narration request should be short enough to get and ' +
      'play a natural Audio clip instead of becoming silent',
  );
  console.log('PASS: demo narration of huge prompts is capped and audible');
}

async function testClickWhileActiveOpensNoNewTab() {
  const {win} = makeWebview();
  installAudio(win);
  installSpeech(win);
  autoAnswerDemoSpeak(win);

  const {done} = startDemoFlow(win, 'Task A1 original');
  await sleep(300); // replay is now active, in the first narration
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
  await testReplaysOnlyClickedChatTasks();
  await testNarrationIsCappedAndAudible();
  await testClickWhileActiveOpensNoNewTab();
  console.log('demoClickedChatOnly.test.js: all tests passed');
  // Demo/webview timers can keep the node event loop alive; match the
  // other jsdom e2e demo tests and exit explicitly once assertions pass.
  process.exit(0);
})().catch(err => {
  console.error(err);
  process.exit(1);
});
