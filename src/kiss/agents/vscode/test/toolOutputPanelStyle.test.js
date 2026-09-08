// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end tests for two styling requirements of the chat webview:
//
// 1. Panels showing tool call output (.bash-panel) use the same
//    background as the thinking panel (.think): the identical
//    translucent cyan tint standalone, and the same tint mixed over
//    --bg when nested inside a .tc card (whose opaque --surface would
//    otherwise make the translucent tint render lighter).
// 2. The model picker pill (#model-btn) width caps, raised by 70%:
//    min(510px, 85vw) in the extension webview and
//    clamp(122px, 36vw, 374px) in the remote web app.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

function makeWebview(opts) {
  const remote = !!(opts && opts.remote);
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
  html = html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');
  if (remote) html = html.replace('<body', '<body class="remote-chat"');

  const dom = new JSDOM(html, {
    runScripts: 'dangerously',
    pretendToBeVisual: true,
    url: 'https://localhost/',
  });
  const win = dom.window;
  win.Element.prototype.scrollIntoView = function () {};
  win.Element.prototype.scrollTo = function () {};
  win.HTMLElement.prototype.scrollTo = function () {};

  const style = win.document.createElement('style');
  style.textContent = fs.readFileSync(path.join(MEDIA, 'main.css'), 'utf8');
  win.document.head.appendChild(style);
  if (remote) {
    const remoteStyle = win.document.createElement('style');
    remoteStyle.textContent = fs.readFileSync(
      path.join(MEDIA, 'remote-codex.css'),
      'utf8',
    );
    win.document.head.appendChild(remoteStyle);
  }

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
  win.eval(
    fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8') +
      '\n//# sourceURL=toolstyle-main.js',
  );
  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function readyTabId(posted) {
  const ready = posted.find(m => m.type === 'ready');
  assert.ok(ready, 'main.js must post a ready message');
  return ready.tabId;
}

// Replays a transcript containing a thinking panel, a Bash tool call
// (whose streaming output panel nests inside the .tc card), and a bare
// tool_result with no preceding tool call (a standalone output panel).
function replayPanels(win, posted) {
  send(win, {
    type: 'task_events',
    tabId: readyTabId(posted),
    chat_id: 'chat-style',
    task: 'style parity task',
    events: [
      {type: 'thinking_start', ts: 1000},
      {type: 'thinking_delta', text: 'pondering'},
      {type: 'thinking_end'},
      {type: 'tool_result', content: 'orphan output', is_error: false},
      {type: 'tool_call', name: 'Bash', command: 'ls', ts: 2000},
      {type: 'tool_result', content: 'file-a\nfile-b', is_error: false},
    ],
  });
}

function testToolOutputMatchesThinkingBackground() {
  const {win, posted} = makeWebview();
  replayPanels(win, posted);
  const d = win.document;

  const think = d.querySelector('.think');
  assert.ok(think, 'the transcript must contain a thinking panel');
  const thinkBg = win.getComputedStyle(think).background;
  assert.ok(
    thinkBg && thinkBg !== 'none',
    'the thinking panel must declare a background',
  );

  const standalone = d.querySelector('#output > .bash-panel');
  assert.ok(standalone, 'the bare tool_result must render a standalone panel');
  assert.strictEqual(
    win.getComputedStyle(standalone).background,
    thinkBg,
    'a standalone tool output panel must share the thinking background',
  );

  const nested = d.querySelector('.tc > .bash-panel');
  assert.ok(nested, 'the Bash tool call must nest an output panel in its .tc');
  // The .tc card paints the opaque --surface behind its children, so the
  // nested panel mixes the same tint over --bg (what the thinking panel
  // sits on) instead of relying on transparency.
  assert.strictEqual(
    win.getComputedStyle(nested).background,
    thinkBg.replace('transparent', 'var(--bg)'),
    'a nested tool output panel must mix the thinking tint over --bg',
  );
  win.close();
}

function testModelPillWidthCaps() {
  const webview = makeWebview();
  assert.strictEqual(
    webview.win.getComputedStyle(
      webview.win.document.getElementById('model-btn'),
    ).maxWidth,
    'min(510px, 85vw)',
    'extension webview: the pill cap must be 70% above min(300px, 50vw)',
  );
  webview.win.close();

  const remote = makeWebview({remote: true});
  assert.strictEqual(
    remote.win.getComputedStyle(remote.win.document.getElementById('model-btn'))
      .maxWidth,
    'clamp(122px, 36vw, 374px)',
    'remote web app: the pill cap must be 70% above clamp(72px, 21vw, 220px)',
  );
  remote.win.close();
}

function runTests() {
  const tests = [
    testToolOutputMatchesThinkingBackground,
    testModelPillWidthCaps,
  ];
  for (const t of tests) {
    t();
    console.log('PASS', t.name);
  }
}

try {
  runTests();
  console.log('\nAll tests passed');
  process.exit(0);
} catch (err) {
  console.error('FAIL:', err && err.message ? err.message : err);
  process.exit(1);
}
