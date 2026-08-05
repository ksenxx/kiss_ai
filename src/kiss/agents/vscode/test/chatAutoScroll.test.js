// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end auto-scroll tests: the chat webview scrolls to the end of
// the latest event panel, in both the extension webview and the remote
// webapp (same main.js, remote-chat body class).  Every scrollable
// subpanel of an event panel (thinking, bash output, thoughts/llm
// panel, tool bodies) must also scroll to its own end as streamed text
// appears inside it.  While a task is running, a user scrolling the
// chat up by at least 1/8th of its visible height DISABLES the outer
// auto-scroll; it RESUMES once the user scrolls back to the bottom.
// Background-tab events must never touch the visible chat's scroll.

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
      '\n//# sourceURL=autoscroll-api.js',
  );
  win.eval(
    fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8') +
      '\n//# sourceURL=autoscroll-main.js',
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

function userScroll(win, el, top) {
  el.scrollTop = top;
  el.dispatchEvent(new win.Event('scroll'));
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
// Outer chat: streamed events pin the chat to the end of the latest
// event panel while the user stays near the bottom.  Scrolling up by
// at least 1/8th of the visible height disables auto-scroll; it
// resumes once the user scrolls back to the bottom.
// --------------------------------------------------------------------

async function testOuterChatFollowsLocksAndResumes(remote) {
  const {win, posted} = makeWebview({remote});
  const O = win.document.getElementById('output');
  const geo = {sh: 3000, ch: 500};
  fakeGeometry(O, geo);
  startRunningTask(win, posted);

  // Streaming events must pin the chat to the bottom.
  send(win, {type: 'tool_call', name: 'Bash', command: 'make -j'});
  geo.sh += 400;
  send(win, {type: 'system_output', text: 'a'.repeat(200) + '\n'});
  await nextFrames(win);
  assert.strictEqual(
    O.scrollTop,
    bottom(geo),
    'BUG (' +
      label(remote) +
      '): streamed bash output did not auto-scroll the chat to the end',
  );

  // A small user scroll up (less than 1/8th of the visible height)
  // must NOT disable auto-scroll.
  userScroll(win, O, bottom(geo) - 40);
  send(win, {type: 'thinking_start'});
  geo.sh += 300;
  send(win, {type: 'thinking_delta', text: 'b'.repeat(200)});
  await nextFrames(win);
  assert.strictEqual(
    O.scrollTop,
    bottom(geo),
    'BUG (' +
      label(remote) +
      '): a user scroll-up smaller than 1/8th of the viewport ' +
      'disabled auto-scroll',
  );

  // Scrolling up by at least 1/8th of the visible height must DISABLE
  // auto-scroll while the task is running.
  const lockedTop = bottom(geo) - 120;
  userScroll(win, O, lockedTop);
  geo.sh += 400;
  send(win, {type: 'text_delta', text: 'streamed text '});
  await nextFrames(win);
  assert.strictEqual(
    O.scrollTop,
    lockedTop,
    'BUG (' +
      label(remote) +
      '): streaming auto-scrolled the chat although the user scrolled ' +
      'up at least 1/8th of the viewport',
  );

  // Non-streamed events must not move the locked chat either.
  send(win, {type: 'tool_result', content: 'done', is_error: false});
  geo.sh += 200;
  send(win, {type: 'system_output', text: 'plain sys line\n'});
  await nextFrames(win);
  assert.strictEqual(
    O.scrollTop,
    lockedTop,
    'BUG (' +
      label(remote) +
      '): an event auto-scrolled the chat while the user scroll lock ' +
      'was engaged',
  );

  // Scrolling down, but not all the way to the bottom, keeps
  // auto-scroll disabled.
  const nearBottom = bottom(geo) - 30;
  userScroll(win, O, nearBottom);
  geo.sh += 200;
  send(win, {type: 'system_output', text: 'another sys line\n'});
  await nextFrames(win);
  assert.strictEqual(
    O.scrollTop,
    nearBottom,
    'BUG (' +
      label(remote) +
      '): auto-scroll resumed before the user reached the bottom',
  );

  // Scrolling to the bottom must RESUME auto-scroll.
  userScroll(win, O, bottom(geo));
  geo.sh += 200;
  send(win, {type: 'result', summary: 'all done', success: true});
  assert.strictEqual(
    O.scrollTop,
    bottom(geo),
    'BUG (' +
      label(remote) +
      '): auto-scroll did not resume after the user scrolled to the ' +
      'bottom',
  );
  win.close();
  console.log(
    '  ok - outer chat follows, locks on user scroll-up, resumes at ' +
      'the bottom (' +
      label(remote) +
      ')',
  );
}

// --------------------------------------------------------------------
// Think subpanel: streamed thinking must scroll the think panel AND
// its enclosing thoughts (llm) panel AND the outer chat to their ends.
// --------------------------------------------------------------------

async function testThinkPanelAutoScrolls(remote) {
  const {win, posted} = makeWebview({remote});
  const O = win.document.getElementById('output');
  const geoO = {sh: 3000, ch: 500};
  fakeGeometry(O, geoO);
  startRunningTask(win, posted);

  send(win, {type: 'thinking_start'});
  const lp = O.querySelector('.llm-panel');
  assert.ok(lp, 'thinking_start must create a thoughts (llm) panel');
  const think = lp.querySelector('.ev.think');
  assert.ok(think, 'thinking_start must create a think panel');
  const geoT = {sh: 1000, ch: 200};
  const geoL = {sh: 1400, ch: 350};
  fakeGeometry(think, geoT);
  fakeGeometry(lp, geoL);

  send(win, {type: 'thinking_delta', text: 'a'.repeat(80)});
  geoT.sh += 200;
  geoL.sh += 200;
  geoO.sh += 200;
  send(win, {type: 'thinking_delta', text: 'b'.repeat(80)});
  await nextFrames(win);
  assert.strictEqual(
    think.scrollTop,
    bottom(geoT),
    'BUG (' +
      label(remote) +
      '): streamed thinking did not scroll the think panel to its end',
  );
  assert.strictEqual(
    lp.scrollTop,
    bottom(geoL),
    'BUG (' +
      label(remote) +
      '): streamed thinking did not scroll the thoughts panel to its end',
  );
  assert.strictEqual(
    O.scrollTop,
    bottom(geoO),
    'BUG (' +
      label(remote) +
      '): streamed thinking did not scroll the chat to its end',
  );

  // Even after the user scrolls the subpanel up, more streamed
  // thinking must bring it back to its end.
  userScroll(win, think, 50);
  geoT.sh += 200;
  send(win, {type: 'thinking_delta', text: 'c'.repeat(80)});
  await nextFrames(win);
  assert.strictEqual(
    think.scrollTop,
    bottom(geoT),
    'BUG (' +
      label(remote) +
      '): the think panel did not follow its end after a user scroll',
  );

  // A thinking_end arriving before the rAF flush must still flush the
  // pending text and scroll the panel to its end.
  userScroll(win, think, 30);
  geoT.sh += 200;
  send(win, {type: 'thinking_delta', text: 'd'.repeat(80)});
  send(win, {type: 'thinking_end'});
  assert.strictEqual(
    think.scrollTop,
    bottom(geoT),
    'BUG (' +
      label(remote) +
      '): thinking_end did not scroll the flushed think panel to its end',
  );
  win.close();
  console.log(
    '  ok - think subpanel auto-scrolls to its end (' + label(remote) + ')',
  );
}

// --------------------------------------------------------------------
// Bash subpanel: streamed bash output must scroll the bash panel
// content and the outer chat to their ends.
// --------------------------------------------------------------------

async function testBashPanelAutoScrolls(remote) {
  const {win, posted} = makeWebview({remote});
  const O = win.document.getElementById('output');
  const geoO = {sh: 3000, ch: 500};
  fakeGeometry(O, geoO);
  startRunningTask(win, posted);

  send(win, {type: 'tool_call', name: 'Bash', command: 'make -j'});
  const bp = O.querySelector('.bash-panel-content');
  assert.ok(bp, 'a Bash tool_call must create a bash output panel');
  const geoB = {sh: 1000, ch: 200};
  fakeGeometry(bp, geoB);

  send(win, {type: 'system_output', text: 'a'.repeat(120) + '\n'});
  geoB.sh += 300;
  geoO.sh += 300;
  send(win, {type: 'system_output', text: 'b'.repeat(120) + '\n'});
  await nextFrames(win);
  assert.strictEqual(
    bp.scrollTop,
    bottom(geoB),
    'BUG (' +
      label(remote) +
      '): streamed bash output did not scroll the bash panel to its end',
  );
  assert.strictEqual(
    O.scrollTop,
    bottom(geoO),
    'BUG (' +
      label(remote) +
      '): streamed bash output did not scroll the chat to its end',
  );

  // After the user scrolls the bash panel up, more output must bring
  // it back to its end.
  userScroll(win, bp, 40);
  geoB.sh += 300;
  send(win, {type: 'system_output', text: 'c'.repeat(120) + '\n'});
  await nextFrames(win);
  assert.strictEqual(
    bp.scrollTop,
    bottom(geoB),
    'BUG (' +
      label(remote) +
      '): the bash panel did not follow its end after a user scroll',
  );

  // The tool_result output panel of the next tool call must be
  // scrolled to its end the moment it appears, even though a
  // provisional thoughts panel is appended right after it.
  send(win, {type: 'tool_call', name: 'Read', path: '/tmp/x'});
  const tc = O.lastElementChild;
  const geoR = {sh: 900, ch: 200};
  win._geoByClass['tr-content'] = geoR;
  win._geoByClass['bash-panel-content'] = geoR;
  send(win, {type: 'tool_result', content: 'z'.repeat(500), is_error: false});
  const rp = tc.querySelectorAll('.bash-panel-content')[0];
  assert.ok(rp, 'a tool_result must create an output panel');
  assert.strictEqual(
    rp.scrollTop,
    bottom(geoR),
    'BUG (' +
      label(remote) +
      '): the tool result subpanel was not scrolled to its end',
  );
  delete win._geoByClass['tr-content'];
  delete win._geoByClass['bash-panel-content'];
  win.close();
  console.log(
    '  ok - bash subpanel auto-scrolls to its end (' + label(remote) + ')',
  );
}

// --------------------------------------------------------------------
// Thoughts (llm) subpanel: streamed assistant text must scroll the
// thoughts panel and the outer chat to their ends.
// --------------------------------------------------------------------

async function testThoughtsPanelAutoScrolls(remote) {
  const {win, posted} = makeWebview({remote});
  const O = win.document.getElementById('output');
  const geoO = {sh: 3000, ch: 500};
  fakeGeometry(O, geoO);
  startRunningTask(win, posted);

  send(win, {type: 'text_delta', text: 'hello '});
  const lp = O.querySelector('.llm-panel');
  assert.ok(lp, 'streaming text must create a thoughts (llm) panel');
  const geoL = {sh: 1000, ch: 300};
  fakeGeometry(lp, geoL);

  send(win, {type: 'text_delta', text: 'world '});
  geoL.sh += 200;
  geoO.sh += 200;
  send(win, {type: 'text_delta', text: 'more text '});
  await nextFrames(win);
  assert.strictEqual(
    lp.scrollTop,
    bottom(geoL),
    'BUG (' +
      label(remote) +
      '): streamed text did not scroll the thoughts panel to its end',
  );
  assert.strictEqual(
    O.scrollTop,
    bottom(geoO),
    'BUG (' +
      label(remote) +
      '): streamed text did not scroll the chat to its end',
  );

  userScroll(win, lp, 60);
  geoL.sh += 200;
  send(win, {type: 'text_delta', text: 'even more '});
  await nextFrames(win);
  assert.strictEqual(
    lp.scrollTop,
    bottom(geoL),
    'BUG (' +
      label(remote) +
      '): the thoughts panel did not follow its end after a user scroll',
  );

  // Finalizing the text (markdown re-render) must keep the thoughts
  // subpanel at its end; the outer chat, which the user scrolled up by
  // at least 1/8th of its height, must stay where the user left it.
  userScroll(win, lp, 70);
  userScroll(win, O, 90);
  send(win, {type: 'text_end'});
  assert.strictEqual(
    lp.scrollTop,
    bottom(geoL),
    'BUG (' +
      label(remote) +
      '): text_end did not scroll the thoughts panel to its end',
  );
  assert.strictEqual(
    O.scrollTop,
    90,
    'BUG (' +
      label(remote) +
      '): text_end auto-scrolled the chat although the user had ' +
      'scrolled up',
  );
  win.close();
  console.log(
    '  ok - thoughts subpanel auto-scrolls to its end (' + label(remote) + ')',
  );
}

// --------------------------------------------------------------------
// Replaying a task (opening an existing chat) must land the chat at
// the end of the latest event panel.
// --------------------------------------------------------------------

async function testReplayLandsAtEnd(remote) {
  const {win, posted} = makeWebview({remote});
  const O = win.document.getElementById('output');
  const geoO = {sh: 4000, ch: 500};
  fakeGeometry(O, geoO);
  const ready = posted.find(m => m.type === 'ready');
  win._testApi.hideWelcome();
  userScroll(win, O, 33);
  send(win, {
    type: 'task_events',
    tabId: ready.tabId,
    task: 'replayed task',
    task_id: 7,
    events: [
      {type: 'prompt', text: 'do the thing'},
      {type: 'tool_call', name: 'Bash', command: 'ls'},
      {type: 'system_output', text: 'file-a\nfile-b\n'},
      {type: 'tool_result', content: 'ok', is_error: false},
      {type: 'thinking_start'},
      {type: 'thinking_delta', text: 'hmm'},
      {type: 'thinking_end'},
      {type: 'text_delta', text: 'answer'},
      {type: 'text_end'},
      {type: 'result', summary: 'done', success: true},
    ],
  });
  assert.strictEqual(
    O.scrollTop,
    bottom(geoO),
    'BUG (' +
      label(remote) +
      '): replaying task events did not land the chat at the end',
  );
  win.close();
  console.log(
    '  ok - task replay lands at the end of the latest panel (' +
      label(remote) +
      ')',
  );
}

// --------------------------------------------------------------------
// Background-tab events stream into a detached fragment and must never
// move the visible chat's scroll position.
// --------------------------------------------------------------------

async function testBackgroundTabDoesNotScrollActiveChat(remote) {
  const {win, posted} = makeWebview({remote});
  const O = win.document.getElementById('output');
  const geoO = {sh: 3000, ch: 500};
  fakeGeometry(O, geoO);
  const tabId = startRunningTask(win, posted);
  send(win, {type: 'system_output', text: 'visible line\n'});
  const bgTabId = tabId + '-bg';
  send(win, {
    type: 'task_events',
    tabId: bgTabId,
    task: 'background task',
    task_id: 9,
    events: [],
  });
  // Stay within 1/8th of the bottom so the visible chat's auto-scroll
  // remains ENABLED: background-tab streaming must still never move
  // the visible chat.
  const pos = bottom(geoO) - 30;
  userScroll(win, O, pos);

  // Streamed events for the background tab (thinking, bash output,
  // text) must not scroll the visible chat.
  send(win, {type: 'status', running: true, tabId: bgTabId, startTs: 1});
  send(win, {type: 'tool_call', name: 'Bash', command: 'ls', tabId: bgTabId});
  send(win, {type: 'system_output', text: 'bg out\n', tabId: bgTabId});
  send(win, {type: 'thinking_start', tabId: bgTabId});
  send(win, {type: 'thinking_delta', text: 'bg think', tabId: bgTabId});
  send(win, {type: 'text_delta', text: 'bg text', tabId: bgTabId});
  await nextFrames(win);
  assert.strictEqual(
    O.scrollTop,
    pos,
    'BUG (' +
      label(remote) +
      '): background-tab streaming moved the visible chat scroll',
  );
  win.close();
  console.log(
    '  ok - background-tab streaming leaves the visible chat alone (' +
      label(remote) +
      ')',
  );
}

// --------------------------------------------------------------------
// Static bounded subpanels (prompt body, tool-call body, error tool
// result) must be scrolled to their end the moment their text appears.
// --------------------------------------------------------------------

async function testStaticSubpanelsAutoScroll(remote) {
  const {win, posted} = makeWebview({remote});
  const O = win.document.getElementById('output');
  const geoO = {sh: 3000, ch: 500};
  fakeGeometry(O, geoO);
  startRunningTask(win, posted);

  const geoP = {sh: 800, ch: 400};
  win._geoByClass['prompt-body'] = geoP;
  send(win, {type: 'prompt', text: 'long prompt\n'.repeat(50)});
  const pb = O.querySelector('.prompt-body');
  assert.ok(pb, 'a prompt event must create a prompt body');
  assert.strictEqual(
    pb.scrollTop,
    bottom(geoP),
    'BUG (' + label(remote) + '): the prompt body was not scrolled to its end',
  );
  delete win._geoByClass['prompt-body'];

  const geoTcb = {sh: 700, ch: 200};
  win._geoByClass['tc-b'] = geoTcb;
  send(win, {type: 'tool_call', name: 'Write', content: 'x\n'.repeat(100)});
  const tcb = O.lastElementChild.querySelector('.tc-b');
  assert.ok(tcb, 'a tool_call must create a tool body');
  assert.strictEqual(
    tcb.scrollTop,
    bottom(geoTcb),
    'BUG (' +
      label(remote) +
      '): the tool-call body was not scrolled to its end',
  );
  delete win._geoByClass['tc-b'];

  const geoTr = {sh: 600, ch: 150};
  win._geoByClass['tr'] = geoTr;
  const tcEl = O.lastElementChild;
  send(win, {type: 'tool_result', content: 'boom', is_error: true});
  const tr = tcEl.querySelector('.tr.err');
  assert.ok(tr, 'an error tool_result must create an error panel');
  assert.strictEqual(
    tr.scrollTop,
    bottom(geoTr),
    'BUG (' +
      label(remote) +
      '): the error tool-result panel was not scrolled to its end',
  );
  delete win._geoByClass['tr'];
  win.close();
  console.log(
    '  ok - static subpanels auto-scroll to their end (' + label(remote) + ')',
  );
}

// --------------------------------------------------------------------
// A tool_call arriving before the bash rAF flush fires must still
// scroll the PREVIOUS tool's bash subpanel when its pending output is
// flushed synchronously.
// --------------------------------------------------------------------

async function testBashFlushOnNextToolCallScrolls(remote) {
  const {win, posted} = makeWebview({remote});
  const O = win.document.getElementById('output');
  const geoO = {sh: 3000, ch: 500};
  fakeGeometry(O, geoO);
  startRunningTask(win, posted);

  send(win, {type: 'tool_call', name: 'Bash', command: 'make -j'});
  const bp = O.querySelector('.bash-panel-content');
  const geoB = {sh: 1000, ch: 200};
  fakeGeometry(bp, geoB);
  userScroll(win, bp, 20);
  // No frame elapses between the output and the next tool_call: the
  // pending buffer is flushed synchronously into the OLD bash panel.
  send(win, {type: 'system_output', text: 'tail line\n'});
  geoB.sh += 300;
  send(win, {type: 'tool_call', name: 'Read', path: '/tmp/y'});
  assert.strictEqual(
    bp.scrollTop,
    bottom(geoB),
    'BUG (' +
      label(remote) +
      "): the previous tool's bash subpanel was not scrolled when its " +
      'pending output was flushed by the next tool_call',
  );
  assert.ok(bp.textContent.includes('tail line'), 'flush must land the text');
  win.close();
  console.log(
    '  ok - pending bash output flushed by a tool_call auto-scrolls (' +
      label(remote) +
      ')',
  );
}

// --------------------------------------------------------------------
// Directly-appended latest event panels — error/notice/warning
// banners, worktree/autocommit action results, and follow-up
// suggestion bars — scroll the chat to their end while auto-scroll is
// enabled, and leave the chat alone while the user scroll lock is
// engaged.
// --------------------------------------------------------------------

async function testBannersActionResultsAndFollowupsRespectLock(remote) {
  const {win, posted} = makeWebview({remote});
  const O = win.document.getElementById('output');
  const geoO = {sh: 3000, ch: 500};
  fakeGeometry(O, geoO);
  startRunningTask(win, posted);

  // While the user is scrolled up (lock engaged), banners must NOT
  // move the chat.
  userScroll(win, O, 30);
  geoO.sh += 100;
  send(win, {type: 'warning', message: 'careful'});
  assert.strictEqual(
    O.scrollTop,
    30,
    'BUG (' +
      label(remote) +
      '): a warning banner auto-scrolled the locked chat',
  );

  geoO.sh += 100;
  send(win, {type: 'error', text: 'kaboom'});
  assert.strictEqual(
    O.scrollTop,
    30,
    'BUG (' +
      label(remote) +
      '): an error banner auto-scrolled the locked chat',
  );

  // Back at the bottom, auto-scroll resumes for banners.
  userScroll(win, O, bottom(geoO));
  geoO.sh += 100;
  send(win, {type: 'notice', text: 'heads up'});
  assert.strictEqual(
    O.scrollTop,
    bottom(geoO),
    'BUG (' +
      label(remote) +
      '): a notice banner did not auto-scroll after the user returned ' +
      'to the bottom',
  );

  // A small scroll up (<1/8th) keeps auto-scroll enabled for action
  // result panels.
  userScroll(win, O, bottom(geoO) - 20);
  geoO.sh += 100;
  send(win, {type: 'worktree_result', success: true, message: 'Merged ok.'});
  assert.ok(
    O.querySelector('.wt-result-ok'),
    'a worktree_result must append an action result panel',
  );
  assert.strictEqual(
    O.scrollTop,
    bottom(geoO),
    'BUG (' + label(remote) + '): a worktree action result did not auto-scroll',
  );

  // Locked again: a follow-up suggestion bar must not move the chat…
  userScroll(win, O, 30);
  geoO.sh += 100;
  send(win, {type: 'followup_suggestion', text: 'try this next'});
  assert.ok(
    O.querySelector('.followup-bar'),
    'a followup_suggestion must append a follow-up bar',
  );
  assert.strictEqual(
    O.scrollTop,
    30,
    'BUG (' +
      label(remote) +
      '): a follow-up suggestion bar auto-scrolled the locked chat',
  );

  // …until the user scrolls back to the bottom.
  userScroll(win, O, bottom(geoO));
  geoO.sh += 100;
  send(win, {type: 'followup_suggestion', text: 'or try this'});
  assert.strictEqual(
    O.scrollTop,
    bottom(geoO),
    'BUG (' +
      label(remote) +
      '): a follow-up suggestion bar did not auto-scroll after the ' +
      'user returned to the bottom',
  );
  win.close();
  console.log(
    '  ok - banners, action results and follow-ups respect the user ' +
      'scroll lock (' +
      label(remote) +
      ')',
  );
}

// --------------------------------------------------------------------
// Switching to a tab whose events streamed into a hidden fragment
// must land the restored chat at the end of its latest event panel.
// --------------------------------------------------------------------

async function testTabRestoreLandsAtEnd(remote) {
  const {win, posted} = makeWebview({remote});
  const O = win.document.getElementById('output');
  const geoO = {sh: 3000, ch: 500};
  fakeGeometry(O, geoO);
  const parentId = startRunningTask(win, posted);
  send(win, {type: 'system_output', text: 'parent line\n'});

  // Open a sub-agent tab (background) and stream events into it while
  // it is hidden.
  send(win, {
    type: 'openSubagentTab',
    tab_id: 'sub-tab-1',
    parent_tab_id: parentId,
    description: 'background worker',
    task_id: 5,
  });
  send(win, {
    type: 'tool_call',
    name: 'Bash',
    command: 'ls',
    tabId: 'sub-tab-1',
  });
  send(win, {type: 'system_output', text: 'bg tail\n', tabId: 'sub-tab-1'});
  send(win, {
    type: 'tool_result',
    content: 'done',
    is_error: false,
    tabId: 'sub-tab-1',
  });

  userScroll(win, O, 40);
  geoO.sh += 500;
  const tabEl = win.document.querySelector(
    '.chat-tab[data-tab-id="sub-tab-1"]',
  );
  assert.ok(tabEl, 'the sub-agent tab must appear in the tab bar');
  tabEl.click();
  assert.ok(
    O.textContent.includes('bg tail'),
    'switching must restore the background output',
  );
  assert.strictEqual(
    O.scrollTop,
    bottom(geoO),
    'BUG (' +
      label(remote) +
      '): restoring a tab with hidden streamed events did not land the ' +
      'chat at the end',
  );
  win.close();
  console.log(
    '  ok - tab restore lands at the end of the latest panel (' +
      label(remote) +
      ')',
  );
}

// --------------------------------------------------------------------
// The scroll-lock threshold is exactly 1/8th of the chat's visible
// height, and the lock only releases at the very bottom.
// --------------------------------------------------------------------

async function testScrollLockThresholdAndResume(remote) {
  const {win, posted} = makeWebview({remote});
  const O = win.document.getElementById('output');
  const geo = {sh: 2000, ch: 400}; // 1/8th of the viewport = 50px
  fakeGeometry(O, geo);
  startRunningTask(win, posted);

  send(win, {type: 'tool_call', name: 'Bash', command: 'tail -f log'});
  send(win, {type: 'system_output', text: 'line 1\n'});
  await nextFrames(win);
  assert.strictEqual(
    O.scrollTop,
    bottom(geo),
    'streaming must pin the chat to the bottom first (' + label(remote) + ')',
  );

  // 49px above the bottom (just below 1/8th): auto-scroll stays on.
  userScroll(win, O, bottom(geo) - 49);
  geo.sh += 100;
  send(win, {type: 'system_output', text: 'line 2\n'});
  await nextFrames(win);
  assert.strictEqual(
    O.scrollTop,
    bottom(geo),
    'BUG (' +
      label(remote) +
      '): a scroll-up just below the 1/8th threshold disabled ' +
      'auto-scroll',
  );

  // Exactly 1/8th above the bottom: auto-scroll must be disabled.
  const lockedTop = bottom(geo) - 50;
  userScroll(win, O, lockedTop);
  geo.sh += 100;
  send(win, {type: 'system_output', text: 'line 3\n'});
  await nextFrames(win);
  assert.strictEqual(
    O.scrollTop,
    lockedTop,
    'BUG (' +
      label(remote) +
      '): scrolling up by exactly 1/8th of the viewport did not ' +
      'disable auto-scroll',
  );

  // Coming within 2px of the bottom is still not "at the bottom":
  // the lock stays engaged.
  const almost = bottom(geo) - 2;
  userScroll(win, O, almost);
  geo.sh += 100;
  send(win, {type: 'system_output', text: 'line 4\n'});
  await nextFrames(win);
  assert.strictEqual(
    O.scrollTop,
    almost,
    'BUG (' +
      label(remote) +
      '): auto-scroll resumed before the user reached the bottom',
  );

  // At the bottom, the lock releases and auto-scroll resumes.
  userScroll(win, O, bottom(geo));
  geo.sh += 100;
  send(win, {type: 'system_output', text: 'line 5\n'});
  await nextFrames(win);
  assert.strictEqual(
    O.scrollTop,
    bottom(geo),
    'BUG (' +
      label(remote) +
      '): auto-scroll did not resume once the user scrolled to the ' +
      'bottom',
  );
  win.close();
  console.log(
    '  ok - lock engages at exactly 1/8th of the viewport and ' +
      'releases at the bottom (' +
      label(remote) +
      ')',
  );
}

// --------------------------------------------------------------------
// Sending a message does NOT release the scroll lock: auto-scroll only
// resumes once the user actually scrolls back to the bottom.
// --------------------------------------------------------------------

async function testSendMessageKeepsLock(remote) {
  const {win, posted} = makeWebview({remote});
  const O = win.document.getElementById('output');
  const geo = {sh: 3000, ch: 500};
  fakeGeometry(O, geo);
  startRunningTask(win, posted);

  send(win, {type: 'text_delta', text: 'hello '});
  await nextFrames(win);
  userScroll(win, O, 30);
  geo.sh += 100;
  send(win, {type: 'text_delta', text: 'world '});
  await nextFrames(win);
  assert.strictEqual(
    O.scrollTop,
    30,
    'precondition (' + label(remote) + '): the user scroll lock must be engaged',
  );

  // The user submits a follow-up message while the task is running and
  // while deliberately reading history: the lock must stay engaged.
  const inp = win.document.getElementById('task-input');
  inp.value = 'follow this up';
  win.document.getElementById('send-btn').click();
  assert.ok(
    posted.some(m => m.type === 'appendUserMessage'),
    'sending while running must post appendUserMessage',
  );
  geo.sh += 100;
  send(win, {type: 'text_delta', text: 'more '});
  await nextFrames(win);
  assert.strictEqual(
    O.scrollTop,
    30,
    'BUG (' +
      label(remote) +
      '): sending a message released the scroll lock before the user ' +
      'scrolled to the bottom',
  );

  // Only scrolling to the bottom resumes auto-scroll.
  userScroll(win, O, bottom(geo));
  geo.sh += 100;
  send(win, {type: 'text_delta', text: 'and more '});
  await nextFrames(win);
  assert.strictEqual(
    O.scrollTop,
    bottom(geo),
    'BUG (' +
      label(remote) +
      '): auto-scroll did not resume after the user scrolled to the ' +
      'bottom',
  );
  win.close();
  console.log(
    '  ok - sending a message keeps the scroll lock until the user ' +
      'returns to the bottom (' +
      label(remote) +
      ')',
  );
}

// --------------------------------------------------------------------
// Scrolling up while NO task is running must not engage the lock: a
// task started afterwards streams with auto-scroll enabled.
// --------------------------------------------------------------------

async function testIdleScrollDoesNotLock(remote) {
  const {win, posted} = makeWebview({remote});
  const O = win.document.getElementById('output');
  const geo = {sh: 3000, ch: 500};
  fakeGeometry(O, geo);
  const ready = posted.find(m => m.type === 'ready');
  win._testApi.hideWelcome();

  // The user browses history while the chat is idle.
  userScroll(win, O, 30);

  // A task then starts (e.g. synchronized from another window): the
  // idle scroll must not have engaged the lock, so streaming follows.
  send(win, {
    type: 'status',
    running: true,
    tabId: ready.tabId,
    startTs: Date.now(),
  });
  geo.sh += 100;
  send(win, {type: 'text_delta', text: 'streaming '});
  await nextFrames(win);
  assert.strictEqual(
    O.scrollTop,
    bottom(geo),
    'BUG (' +
      label(remote) +
      '): a scroll-up while no task was running engaged a stale lock ' +
      'that suppressed auto-scroll of the next task',
  );
  win.close();
  console.log(
    '  ok - an idle scroll-up does not engage the scroll lock (' +
      label(remote) +
      ')',
  );
}

// --------------------------------------------------------------------
// A `clear` event (a new task starting in this tab — both backends
// broadcast it, including for externally started viewer tasks) resets
// the scroll lock along with the output.
// --------------------------------------------------------------------

async function testClearEventResetsLock(remote) {
  const {win, posted} = makeWebview({remote});
  const O = win.document.getElementById('output');
  const geo = {sh: 3000, ch: 500};
  fakeGeometry(O, geo);
  const tabId = startRunningTask(win, posted);

  send(win, {type: 'text_delta', text: 'old task text '});
  await nextFrames(win);
  userScroll(win, O, 30);
  geo.sh += 100;
  send(win, {type: 'text_delta', text: 'still old '});
  await nextFrames(win);
  assert.strictEqual(
    O.scrollTop,
    30,
    'precondition (' + label(remote) + '): the lock must be engaged',
  );

  // A new task starts in this tab: the backend broadcasts `clear`.
  send(win, {type: 'clear', tabId: tabId});
  geo.sh = 2000;
  send(win, {type: 'text_delta', text: 'new task streaming '});
  await nextFrames(win);
  assert.strictEqual(
    O.scrollTop,
    bottom(geo),
    'BUG (' +
      label(remote) +
      "): a new task's clear event did not reset the scroll lock",
  );
  win.close();
  console.log(
    '  ok - a clear event (new task) resets the scroll lock (' +
      label(remote) +
      ')',
  );
}

async function main() {
  for (const remote of [false, true]) {
    await testOuterChatFollowsLocksAndResumes(remote);
    await testThinkPanelAutoScrolls(remote);
    await testBashPanelAutoScrolls(remote);
    await testThoughtsPanelAutoScrolls(remote);
    await testReplayLandsAtEnd(remote);
    await testBackgroundTabDoesNotScrollActiveChat(remote);
    await testStaticSubpanelsAutoScroll(remote);
    await testBashFlushOnNextToolCallScrolls(remote);
    await testBannersActionResultsAndFollowupsRespectLock(remote);
    await testTabRestoreLandsAtEnd(remote);
    await testScrollLockThresholdAndResume(remote);
    await testSendMessageKeepsLock(remote);
    await testIdleScrollDoesNotLock(remote);
    await testClearEventResetsLock(remote);
  }
  console.log('chatAutoScroll.test.js: all tests passed');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
