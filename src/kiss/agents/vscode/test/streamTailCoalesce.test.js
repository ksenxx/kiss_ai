// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end tests for the coalesced stream-tail sweep: a streamed
// delta chunk (thinking_delta / text_delta / system_output) whose DOM
// write is buffered into a pending animation-frame flush must SKIP the
// synchronous per-event tail (older-panel collapse, chevron sweep,
// visible-task re-derivation, auto-scroll pass) and replay it in ONE
// coalesced sweep per frame — with the exact same end-of-frame state a
// per-event tail produced.  Chunks that open a thoughts panel, chunks
// rendered without a pending flush, and environments whose
// requestAnimationFrame runs callbacks synchronously must keep the
// original per-event tail.
//
// Unreachable-branch note (no mocks are used, per testing policy):
// inside the deferTail condition, `thinkRaf && !thinkCnt` and
// `bashRaf && !bashPanel` cannot occur — thinking_end cancels thinkRaf
// before clearing thinkCnt, and tool_call zeroes bashRaf when it
// clears bashPanel — so the `!!tState.thinkCnt` / `!!tState.bashPanel`
// guards are pure defence and their false sides are untestable
// end-to-end.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

function makeWebview(opts) {
  opts = opts || {};
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  if (opts.remote) {
    html = html.replace('{{BODY_CLASS_ATTR}}', ' class="remote-chat"');
  }
  html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
  html = html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');

  const dom = new JSDOM(html, {
    runScripts: 'dangerously',
    pretendToBeVisual: true,
    url: 'https://localhost/',
  });
  const win = dom.window;

  // jsdom has no layout engine: stub scrollIntoView (used by the tab
  // bar) so main.js can initialize.
  win.Element.prototype.scrollIntoView = function () {};

  // Class-keyed fake geometry lets a test give scroll geometry to
  // panels that are created *during* event handling (per-element
  // fakeGeometry own properties still take precedence).
  const geoByClass = {};
  win._geoByClass = geoByClass;
  Object.defineProperty(win.Element.prototype, 'scrollHeight', {
    configurable: true,
    get() {
      for (const k in geoByClass)
        if (this.classList && this.classList.contains(k))
          return geoByClass[k].sh;
      return 0;
    },
  });
  Object.defineProperty(win.Element.prototype, 'clientHeight', {
    configurable: true,
    get() {
      for (const k in geoByClass)
        if (this.classList && this.classList.contains(k))
          return geoByClass[k].ch;
      return 0;
    },
  });

  if (opts.syncRaf) {
    // Environments (and many existing suites) where rAF runs the
    // callback synchronously and returns 0: the flush handle stays 0,
    // so every chunk must keep the original per-event tail.
    win.requestAnimationFrame = function (cb) {
      cb();
      return 0;
    };
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
  win.eval(
    fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8') +
      '\n//# sourceURL=streamtail-api.js',
  );
  win.eval(
    fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8') +
      '\n//# sourceURL=streamtail-main.js',
  );
  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function fakeGeometry(el, geo) {
  Object.defineProperty(el, 'scrollHeight', {
    get: () => geo.sh,
    configurable: true,
  });
  Object.defineProperty(el, 'clientHeight', {
    get: () => geo.ch,
    configurable: true,
  });
}

function bottom(geo) {
  return Math.max(0, geo.sh - geo.ch);
}

function nextFrames(win, n = 3) {
  return new Promise(resolve => {
    let left = n;
    function step() {
      if (--left <= 0) return resolve();
      win.requestAnimationFrame(step);
    }
    win.requestAnimationFrame(step);
  });
}

function startRunningTask(win, posted) {
  const ready = posted.find(m => m.type === 'ready');
  assert.ok(ready && ready.tabId, 'webview must post ready with a tabId');
  win._testApi.hideWelcome();
  send(win, {
    type: 'status',
    running: true,
    tabId: ready.tabId,
    startTs: Date.now() - 2000,
  });
  return ready.tabId;
}

const label = remote => (remote ? 'remote webapp' : 'extension webview');

// --------------------------------------------------------------------
// system_output chunks: the older-panel collapse a chunk used to run
// synchronously (via streamEnd) is deferred to the per-frame sweep,
// and the end-of-frame state is identical: text flushed, older panel
// collapsed, chat scrolled to the bottom.
// --------------------------------------------------------------------

async function testBashChunkDefersTailSweepReplaysIt(remote) {
  const {win, posted} = makeWebview({remote});
  const O = win.document.getElementById('output');
  const geo = {sh: 3000, ch: 500};
  fakeGeometry(O, geo);
  startRunningTask(win, posted);

  // A system_output WITHOUT a bash panel is not a buffered chunk: it
  // renders synchronously and keeps the synchronous tail.
  send(win, {type: 'system_output', text: 'plain sys line\n'});
  assert.ok(
    O.querySelector('.ev.sys'),
    'BUG (' +
      label(remote) +
      '): non-buffered system_output must render synchronously',
  );

  send(win, {type: 'tool_call', name: 'Bash', command: 'make one'});
  const panel1 = O.querySelector('.ev.tc');
  assert.ok(panel1, 'first tool_call panel missing');
  send(win, {type: 'system_output', text: 'one\n'});
  send(win, {type: 'tool_call', name: 'Bash', command: 'make two'});
  const panels = O.querySelectorAll('.ev.tc');
  assert.strictEqual(panels.length, 2, 'expected two tool panels');
  const panel2 = panels[1];
  assert.ok(
    panel1.classList.contains('collapsed'),
    'BUG (' +
      label(remote) +
      '): a tool_call (full synchronous tail) must collapse the ' +
      'older panel immediately',
  );

  // Re-expand the older panel, then stream a buffered chunk into the
  // new panel: the chunk must NOT collapse it synchronously (the tail
  // is deferred)...
  panel1.classList.remove('collapsed');
  geo.sh += 400;
  send(win, {type: 'system_output', text: 'two\n'});
  assert.ok(
    !panel1.classList.contains('collapsed'),
    'BUG (' +
      label(remote) +
      '): a buffered system_output chunk ran the per-event tail ' +
      'synchronously instead of deferring it to the per-frame sweep',
  );
  const bash2 = panel2.querySelector('.bash-panel-content');
  assert.strictEqual(
    bash2.textContent,
    '',
    'chunk text must still be buffered before the flush frame',
  );

  // ...but the per-frame sweep must replay the collapse, the flush
  // must apply the text, and the chat must sit at its bottom.
  await nextFrames(win);
  assert.ok(
    panel1.classList.contains('collapsed'),
    'BUG (' +
      label(remote) +
      '): the per-frame sweep did not replay the older-panel collapse',
  );
  assert.strictEqual(
    bash2.textContent,
    'two\n',
    'BUG (' + label(remote) + '): the flush frame did not apply chunk text',
  );
  assert.strictEqual(
    O.scrollTop,
    bottom(geo),
    'BUG (' +
      label(remote) +
      '): the chat is not at its bottom after the flush frame',
  );
  win.close();
  console.log(
    '  ok - buffered bash chunk defers the tail, sweep replays it (' +
      label(remote) +
      ')',
  );
}

// --------------------------------------------------------------------
// text_delta chunks: the first chunk after a tool_result adopts the
// provisional thoughts panel and counts a step — it must keep the
// FULL synchronous tail.  Later chunks with a pending flush defer,
// and the end-of-frame text and scroll state are unchanged.
// --------------------------------------------------------------------

async function testTextDeltaStreamParity() {
  const {win, posted} = makeWebview({});
  const O = win.document.getElementById('output');
  const geo = {sh: 3000, ch: 500};
  fakeGeometry(O, geo);
  startRunningTask(win, posted);
  const statusSteps = win.document.getElementById('status-steps');

  send(win, {type: 'tool_call', name: 'Bash', command: 'true'});
  send(win, {type: 'tool_result', content: 'ok', is_error: false});
  send(win, {type: 'text_delta', text: 'hello '});
  assert.strictEqual(
    statusSteps.textContent,
    'Steps: 1',
    'BUG: a text_delta that adopts the thoughts panel counts a step ' +
      'and must keep the synchronous tail',
  );
  geo.sh += 300;
  send(win, {type: 'text_delta', text: 'world'});
  const txt = O.querySelector('.llm-panel .txt');
  assert.ok(txt, 'streamed text element missing');
  assert.strictEqual(
    txt.textContent,
    '',
    'chunk text must still be buffered before the flush frame',
  );
  await nextFrames(win);
  assert.strictEqual(
    txt.textContent,
    'hello world',
    'BUG: deferred text_delta chunks lost text across the flush frame',
  );
  assert.strictEqual(
    O.scrollTop,
    bottom(geo),
    'BUG: the chat is not at its bottom after deferred text chunks',
  );
  win.close();
  console.log('  ok - text_delta: panel-opening chunk sync, rest deferred');
}

// --------------------------------------------------------------------
// thinking_delta chunks defer like text chunks; the flushed thinking
// text and the bottom-pinned chat are unchanged.
// --------------------------------------------------------------------

async function testThinkingDeltaStreamParity() {
  const {win, posted} = makeWebview({});
  const O = win.document.getElementById('output');
  const geo = {sh: 3000, ch: 500};
  fakeGeometry(O, geo);
  startRunningTask(win, posted);

  send(win, {type: 'thinking_start'});
  const cnt = O.querySelector('.think .cnt');
  assert.ok(cnt, 'thinking panel missing');
  geo.sh += 300;
  send(win, {type: 'thinking_delta', text: 'pondering '});
  send(win, {type: 'thinking_delta', text: 'deeply'});
  assert.strictEqual(
    cnt.textContent,
    '',
    'thinking chunks must still be buffered before the flush frame',
  );
  await nextFrames(win);
  assert.strictEqual(
    cnt.textContent,
    'pondering deeply',
    'BUG: deferred thinking_delta chunks lost text across the flush frame',
  );
  assert.strictEqual(
    O.scrollTop,
    bottom(geo),
    'BUG: the chat is not at its bottom after deferred thinking chunks',
  );
  win.close();
  console.log('  ok - thinking_delta chunks defer and flush with parity');
}

// --------------------------------------------------------------------
// Synchronous-rAF environments: the flush runs inline and leaves no
// pending handle, so every chunk must keep the original per-event
// synchronous tail.
// --------------------------------------------------------------------

function testSyncRafKeepsPerEventTail() {
  const {win, posted} = makeWebview({syncRaf: true});
  const O = win.document.getElementById('output');
  fakeGeometry(O, {sh: 3000, ch: 500});
  startRunningTask(win, posted);

  send(win, {type: 'tool_call', name: 'Bash', command: 'make one'});
  send(win, {type: 'tool_call', name: 'Bash', command: 'make two'});
  const panels = O.querySelectorAll('.ev.tc');
  const panel1 = panels[0];
  const panel2 = panels[1];
  panel1.classList.remove('collapsed');
  send(win, {type: 'system_output', text: 'two\n'});
  assert.strictEqual(
    panel2.querySelector('.bash-panel-content').textContent,
    'two\n',
    'BUG: with synchronous rAF the chunk must flush inline',
  );
  assert.ok(
    panel1.classList.contains('collapsed'),
    'BUG: with synchronous rAF the chunk must keep the per-event ' +
      'synchronous tail (older panel collapsed immediately)',
  );
  win.close();
  console.log('  ok - synchronous rAF keeps the per-event tail');
}

// --------------------------------------------------------------------
// Tab switches: a pending sweep belongs to the transcript that left
// the screen — it must neither collapse the new tab's panels nor
// carry its collapse debt over to the new tab's own sweep.
// --------------------------------------------------------------------

async function testTabSwitchDropsPendingSweep() {
  const {win, posted} = makeWebview({});
  const O = win.document.getElementById('output');
  fakeGeometry(O, {sh: 3000, ch: 500});
  startRunningTask(win, posted);
  win._testApi.endLaunch();

  // Tab 1: leave a sweep pending WITH collapse debt.
  send(win, {type: 'tool_call', name: 'Bash', command: 'make one'});
  send(win, {type: 'tool_call', name: 'Bash', command: 'make two'});
  const t1panel1 = O.querySelectorAll('.ev.tc')[0];
  t1panel1.classList.remove('collapsed');
  send(win, {type: 'system_output', text: 'two\n'});
  assert.ok(!t1panel1.classList.contains('collapsed'), 'tail must be pending');

  // Switch tabs before the frame fires, and make tab 2 run a stream
  // of its own whose chunks carry NO collapse debt.
  win._testApi.createNewTab();
  const tab2 = win._testApi.getActiveTabId();
  send(win, {type: 'status', running: true, tabId: tab2, startTs: Date.now()});
  send(win, {type: 'tool_call', name: 'Bash', command: 'true', tabId: tab2});
  send(win, {
    type: 'tool_result',
    content: 'ok',
    is_error: false,
    tabId: tab2,
  });
  send(win, {type: 'text_delta', text: 'hi ', tabId: tab2});
  const t2panel = O.querySelector('.ev.tc');
  t2panel.classList.remove('collapsed');
  // This chunk re-arms the sweep for tab 2: the debt tab 1 left behind
  // must have been dropped with tab 1's transcript.
  send(win, {type: 'text_delta', text: 'there', tabId: tab2});

  await nextFrames(win);
  assert.ok(
    !t2panel.classList.contains('collapsed'),
    'BUG: the collapse debt of a transcript that left the screen was ' +
      "applied to the NEW tab's transcript",
  );
  assert.ok(
    !t1panel1.classList.contains('collapsed'),
    "BUG: a pending sweep collapsed a hidden tab's panel",
  );
  assert.strictEqual(
    O.querySelector('.llm-panel .txt').textContent,
    'hi there',
    'BUG: tab 2 lost streamed text across the tab switch',
  );

  // A sweep still pending when its own tab leaves the screen must do
  // nothing at all when it fires.
  send(win, {
    type: 'tool_call',
    name: 'Bash',
    command: 'make three',
    tabId: tab2,
  });
  const t2panels = O.querySelectorAll('.ev.tc');
  t2panels[0].classList.remove('collapsed');
  send(win, {type: 'system_output', text: 'three\n', tabId: tab2});
  win._testApi.createNewTab();
  await nextFrames(win);
  assert.ok(
    !t2panels[0].classList.contains('collapsed'),
    'BUG: a sweep fired for a transcript that had left the screen',
  );
  win.close();
  console.log('  ok - tab switches drop the pending sweep and its debt');
}

// --------------------------------------------------------------------
// Retargeting: when the FIRST event after a tab switch is already a
// buffered chunk (the restored tab still holds a pending flush), the
// sweep pending for the outgoing tab is cancelled and re-armed for
// this tab — its collapse debt must not leak, and the re-armed sweep
// runs behind this tab's flush.
// --------------------------------------------------------------------

async function testChunkAfterSwitchRetargetsSweep() {
  const {win, posted} = makeWebview({});
  const O = win.document.getElementById('output');
  fakeGeometry(O, {sh: 3000, ch: 500});
  const tab1 = startRunningTask(win, posted);
  win._testApi.endLaunch();

  // Tab 1 gets a bash panel so a later chunk can buffer into it.
  send(win, {type: 'tool_call', name: 'Bash', command: 'make one'});

  // Tab 2 runs its own stream with an open thoughts panel mid-text.
  win._testApi.createNewTab();
  const tab2 = win._testApi.getActiveTabId();
  send(win, {type: 'status', running: true, tabId: tab2, startTs: Date.now()});
  send(win, {type: 'tool_call', name: 'Bash', command: 'true', tabId: tab2});
  send(win, {
    type: 'tool_result',
    content: 'ok',
    is_error: false,
    tabId: tab2,
  });
  send(win, {type: 'text_delta', text: 'hi ', tabId: tab2});

  // Back on tab 1, leave a sweep pending WITH collapse debt...
  win.document.querySelector('[data-tab-id="' + tab1 + '"]').click();
  send(win, {type: 'system_output', text: 'one\n', tabId: tab1});

  // ...switch to tab 2 and make its FIRST event a buffered chunk: the
  // pending sweep is retargeted, dropping tab 1's debt.
  win.document.querySelector('[data-tab-id="' + tab2 + '"]').click();
  const t2panel = O.querySelector('.ev.tc');
  t2panel.classList.remove('collapsed');
  send(win, {type: 'text_delta', text: 'there', tabId: tab2});

  await nextFrames(win);
  assert.ok(
    !t2panel.classList.contains('collapsed'),
    'BUG: the outgoing tab\u2019s collapse debt leaked into the sweep ' +
      'retargeted to the new tab',
  );
  assert.strictEqual(
    O.querySelector('.llm-panel .txt').textContent,
    'hi there',
    'BUG: the retargeted sweep ran before the flush applied the text',
  );
  win.close();
  console.log('  ok - a chunk right after a tab switch retargets the sweep');
}

// --------------------------------------------------------------------
// Task end while a sweep is pending: the deferred tail must run
// BEFORE the running state flips off — swept after it, the chevron
// pass would hide the finished task's panels (chv-hidden) and the
// collapse debt would be dropped.  A non-chunk event must likewise
// settle the pending sweep first, keeping the old tail-per-event
// ordering.
// --------------------------------------------------------------------

async function testTaskEndFlushesPendingSweepWhileRunning() {
  const {win, posted} = makeWebview({});
  const O = win.document.getElementById('output');
  fakeGeometry(O, {sh: 3000, ch: 500});
  const tabId = startRunningTask(win, posted);

  send(win, {type: 'tool_call', name: 'Bash', command: 'make one'});
  send(win, {type: 'tool_call', name: 'Bash', command: 'make two'});
  const panels = O.querySelectorAll('.ev.tc');
  panels[0].classList.remove('collapsed');
  send(win, {type: 'system_output', text: 'late output\n'});
  assert.ok(!panels[0].classList.contains('collapsed'), 'tail must defer');

  // The task ends before the sweep's frame fires.
  send(win, {type: 'status', running: false, tabId: tabId});
  assert.ok(
    panels[0].classList.contains('collapsed'),
    'BUG: the pending sweep was not flushed before the running state ' +
      'flipped off, dropping the collapse debt',
  );
  assert.strictEqual(
    O.querySelectorAll('.chv-hidden').length,
    0,
    'BUG: sweeping after setRunningState(false) hid the finished ' +
      "task's panels (chv-hidden), which the synchronous tail never did",
  );
  await nextFrames(win);
  assert.strictEqual(
    O.querySelectorAll('.chv-hidden').length,
    0,
    'BUG: a straggler sweep hid panels after the task ended',
  );
  assert.ok(
    panels[1].querySelector('.bash-panel-content').textContent.includes(
      'late output',
    ),
    'buffered chunk text must still flush after the task ends',
  );
  win.close();
  console.log('  ok - task end flushes the pending sweep while running');
}

// --------------------------------------------------------------------
// Sweep autoscroll parity: the old per-event tail scrolled EVERY
// scrollable subpanel of the latest event panel, not only the panels
// enclosing the streamed text — a completed sibling .think subpanel
// must still be pinned to its end by the per-frame sweep.
// --------------------------------------------------------------------

async function testSweepScrollsSiblingSubpanels() {
  const {win, posted} = makeWebview({});
  const O = win.document.getElementById('output');
  fakeGeometry(O, {sh: 3000, ch: 500});
  startRunningTask(win, posted);

  send(win, {type: 'thinking_start'});
  send(win, {type: 'thinking_delta', text: 'mull'});
  send(win, {type: 'thinking_end'});
  send(win, {type: 'text_delta', text: 'answer '});
  const think = O.querySelector('.think');
  assert.ok(think, 'think subpanel missing');
  win._geoByClass.think = {sh: 800, ch: 100};
  think.scrollTop = 10; // reader left it mid-way; no user lock involved
  send(win, {type: 'text_delta', text: 'text'});
  await nextFrames(win);
  assert.strictEqual(
    think.scrollTop,
    700,
    'BUG: the per-frame sweep did not scroll a sibling .think ' +
      'subpanel of the latest event panel to its end',
  );
  win.close();
  console.log('  ok - sweep scrolls sibling subpanels of the latest panel');
}

// --------------------------------------------------------------------
// Transcript replacement (replay) cancels the pending sweep: the
// deferred tail belongs to the outgoing transcript and must neither
// run against the replayed one nor crash.
// --------------------------------------------------------------------

async function testReplayCancelsPendingSweep() {
  const {win, posted} = makeWebview({});
  const O = win.document.getElementById('output');
  fakeGeometry(O, {sh: 3000, ch: 500});
  const tabId = startRunningTask(win, posted);

  send(win, {type: 'tool_call', name: 'Bash', command: 'make one'});
  send(win, {type: 'system_output', text: 'one\n'});

  // Replace the transcript while the sweep (and the flush) are pending.
  send(win, {
    type: 'task_events',
    tabId: tabId,
    task: 'replayed task',
    events: [{type: 'tool_call', name: 'Bash', command: 'replayed'}],
  });
  const replayed = O.querySelectorAll('.ev.tc');
  assert.strictEqual(replayed.length, 1, 'replay must replace the transcript');
  await nextFrames(win);
  assert.strictEqual(
    O.querySelectorAll('.chv-hidden').length,
    0,
    'BUG: a sweep deferred for the outgoing transcript ran against ' +
      'the replayed one',
  );
  assert.ok(
    replayed[0].textContent.includes('replayed'),
    'replayed transcript must stay intact across the dropped sweep',
  );
  win.close();
  console.log('  ok - replay cancels the pending sweep');
}

async function main() {
  await testBashChunkDefersTailSweepReplaysIt(false);
  await testBashChunkDefersTailSweepReplaysIt(true);
  await testTextDeltaStreamParity();
  await testThinkingDeltaStreamParity();
  testSyncRafKeepsPerEventTail();
  await testTabSwitchDropsPendingSweep();
  await testChunkAfterSwitchRetargetsSweep();
  await testTaskEndFlushesPendingSweepWhileRunning();
  await testSweepScrollsSiblingSubpanels();
  await testReplayCancelsPendingSweep();
  console.log('streamTailCoalesce: all tests passed');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
