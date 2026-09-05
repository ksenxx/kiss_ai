// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end auto-scroll tests: the chat webview scrolls to the end of
// the latest event panel, in both the extension webview and the remote
// webapp (same main.js, remote-chat body class).  Every scrollable
// subpanel of an event panel (thinking, bash output, thoughts/llm
// panel, tool bodies) must also scroll to its own end as streamed text
// appears inside it.  ANY user scroll away from the bottom of the chat
// DISABLES the outer auto-scroll — the chat must never scroll to the
// end unless the user is at the end — and auto-scroll RESUMES once the
// user scrolls back to the bottom.  The SAME user scroll lock applies
// to every scrollable subpanel independently: a user scroll away from
// a subpanel's bottom disables only that subpanel's auto-scroll, and
// it resumes once the user scrolls that subpanel back to its bottom.
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
// event panel while the user stays at the bottom.  Any scroll away
// from the bottom disables auto-scroll; it resumes once the user
// scrolls back to the bottom.
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

  // Even a SMALL user scroll up must DISABLE auto-scroll: the chat
  // must never scroll to the end unless the user is at the end.
  const smallTop = bottom(geo) - 40;
  userScroll(win, O, smallTop);
  send(win, {type: 'thinking_start'});
  geo.sh += 300;
  send(win, {type: 'thinking_delta', text: 'b'.repeat(200)});
  await nextFrames(win);
  assert.strictEqual(
    O.scrollTop,
    smallTop,
    'BUG (' +
      label(remote) +
      '): a small user scroll-up did not disable auto-scroll',
  );

  // A larger scroll up keeps auto-scroll disabled too.
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
      '): streaming auto-scrolled the chat although the user had ' +
      'scrolled up',
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

  // A user scroll away from the think panel's bottom engages the
  // panel's OWN lock: more streamed thinking must leave the think
  // panel where the user put it, while the unlocked thoughts panel
  // and the chat keep following their ends.
  userScroll(win, think, 50);
  geoT.sh += 200;
  geoL.sh += 200;
  geoO.sh += 200;
  send(win, {type: 'thinking_delta', text: 'c'.repeat(80)});
  await nextFrames(win);
  assert.strictEqual(
    think.scrollTop,
    50,
    'BUG (' +
      label(remote) +
      '): streamed thinking auto-scrolled the think panel although ' +
      'the user had scrolled it up',
  );
  assert.strictEqual(
    lp.scrollTop,
    bottom(geoL),
    'BUG (' +
      label(remote) +
      "): the locked think panel disabled its parent thoughts panel's " +
      'auto-scroll',
  );
  assert.strictEqual(
    O.scrollTop,
    bottom(geoO),
    'BUG (' +
      label(remote) +
      "): the locked think panel disabled the chat's auto-scroll",
  );

  // Scrolling the think panel back to its bottom releases its lock:
  // the panel follows its end again.
  userScroll(win, think, bottom(geoT));
  geoT.sh += 200;
  send(win, {type: 'thinking_delta', text: 'd'.repeat(80)});
  await nextFrames(win);
  assert.strictEqual(
    think.scrollTop,
    bottom(geoT),
    'BUG (' +
      label(remote) +
      '): the think panel did not resume following after the user ' +
      'scrolled it back to its bottom',
  );

  // The REVERSE nesting: a locked thoughts (llm) panel must stay put
  // while the unlocked think panel inside it keeps following its end.
  userScroll(win, lp, 40);
  geoT.sh += 200;
  geoL.sh += 200;
  geoO.sh += 200;
  send(win, {type: 'thinking_delta', text: 'x'.repeat(80)});
  await nextFrames(win);
  assert.strictEqual(
    think.scrollTop,
    bottom(geoT),
    'BUG (' +
      label(remote) +
      "): a locked thoughts panel disabled the inner think panel's " +
      'auto-scroll',
  );
  assert.strictEqual(
    lp.scrollTop,
    40,
    'BUG (' +
      label(remote) +
      '): streamed thinking auto-scrolled a thoughts panel the user ' +
      'had scrolled up',
  );
  assert.strictEqual(
    O.scrollTop,
    bottom(geoO),
    'BUG (' +
      label(remote) +
      "): the locked thoughts panel disabled the chat's auto-scroll",
  );
  userScroll(win, lp, bottom(geoL));

  // A thinking_end arriving before the rAF flush must still flush the
  // pending text and scroll the (unlocked) panel to its end.
  geoT.sh += 200;
  send(win, {type: 'thinking_delta', text: 'e'.repeat(80)});
  send(win, {type: 'thinking_end'});
  assert.strictEqual(
    think.scrollTop,
    bottom(geoT),
    'BUG (' +
      label(remote) +
      '): thinking_end did not scroll the flushed think panel to its end',
  );

  // A LOCKED think panel is left alone even by the thinking_end
  // flush; the pending text must still land in the panel.
  send(win, {type: 'thinking_start'});
  const think2 = O.querySelectorAll('.ev.think')[1];
  assert.ok(think2, 'a second thinking_start must create a think panel');
  const geoT2 = {sh: 1000, ch: 200};
  fakeGeometry(think2, geoT2);
  send(win, {type: 'thinking_delta', text: 'f'.repeat(80)});
  await nextFrames(win);
  userScroll(win, think2, 30);
  geoT2.sh += 200;
  send(win, {type: 'thinking_delta', text: 'g'.repeat(80)});
  send(win, {type: 'thinking_end'});
  await nextFrames(win);
  assert.strictEqual(
    think2.scrollTop,
    30,
    'BUG (' +
      label(remote) +
      '): the thinking_end flush auto-scrolled a think panel the user ' +
      'had scrolled up',
  );
  assert.ok(
    think2.textContent.includes('g'.repeat(80)),
    'the pending thinking text must still flush into the locked panel',
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

  // After the user scrolls the bash panel up, its own lock engages:
  // more output must leave the bash panel where the user put it while
  // the chat keeps following.
  userScroll(win, bp, 40);
  geoB.sh += 300;
  geoO.sh += 300;
  send(win, {type: 'system_output', text: 'c'.repeat(120) + '\n'});
  await nextFrames(win);
  assert.strictEqual(
    bp.scrollTop,
    40,
    'BUG (' +
      label(remote) +
      '): streamed bash output auto-scrolled the bash panel although ' +
      'the user had scrolled it up',
  );
  assert.strictEqual(
    O.scrollTop,
    bottom(geoO),
    'BUG (' +
      label(remote) +
      "): the locked bash panel disabled the chat's auto-scroll",
  );

  // Scrolling the bash panel back to its bottom releases its lock.
  userScroll(win, bp, bottom(geoB));
  geoB.sh += 300;
  send(win, {type: 'system_output', text: 'd'.repeat(120) + '\n'});
  await nextFrames(win);
  assert.strictEqual(
    bp.scrollTop,
    bottom(geoB),
    'BUG (' +
      label(remote) +
      '): the bash panel did not resume following after the user ' +
      'scrolled it back to its bottom',
  );

  // The reverse independence: a locked CHAT must not stop an unlocked
  // bash subpanel from following its own end.
  userScroll(win, O, 50);
  geoB.sh += 300;
  send(win, {type: 'system_output', text: 'e'.repeat(120) + '\n'});
  await nextFrames(win);
  assert.strictEqual(
    bp.scrollTop,
    bottom(geoB),
    'BUG (' +
      label(remote) +
      "): the locked chat disabled the bash subpanel's auto-scroll",
  );
  assert.strictEqual(
    O.scrollTop,
    50,
    'BUG (' +
      label(remote) +
      '): streamed bash output auto-scrolled the chat although the ' +
      'user had scrolled it up',
  );
  userScroll(win, O, bottom(geoO));

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

  // A user scroll away from the thoughts panel's bottom engages its
  // own lock: streamed text must leave the panel where the user put
  // it while the chat keeps following.
  userScroll(win, lp, 60);
  geoL.sh += 200;
  geoO.sh += 200;
  send(win, {type: 'text_delta', text: 'even more '});
  await nextFrames(win);
  assert.strictEqual(
    lp.scrollTop,
    60,
    'BUG (' +
      label(remote) +
      '): streamed text auto-scrolled the thoughts panel although the ' +
      'user had scrolled it up',
  );
  assert.strictEqual(
    O.scrollTop,
    bottom(geoO),
    'BUG (' +
      label(remote) +
      "): the locked thoughts panel disabled the chat's auto-scroll",
  );

  // Scrolling the thoughts panel back to its bottom releases its lock.
  userScroll(win, lp, bottom(geoL));
  geoL.sh += 200;
  send(win, {type: 'text_delta', text: 'and on '});
  await nextFrames(win);
  assert.strictEqual(
    lp.scrollTop,
    bottom(geoL),
    'BUG (' +
      label(remote) +
      '): the thoughts panel did not resume following after the user ' +
      'scrolled it back to its bottom',
  );

  // Finalizing the text (text_end re-triggers the latest-panel scroll
  // pass; the markdown branch itself is skipped here, the harness does
  // not load marked) while BOTH the thoughts subpanel and the chat are
  // user-scrolled must leave both where the user left them.
  userScroll(win, lp, 70);
  userScroll(win, O, 90);
  send(win, {type: 'text_end'});
  await nextFrames(win);
  assert.strictEqual(
    lp.scrollTop,
    70,
    'BUG (' +
      label(remote) +
      '): text_end auto-scrolled the thoughts panel although the user ' +
      'had scrolled it up',
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
  // Stay AT the bottom so the visible chat's auto-scroll remains
  // ENABLED, then grow the chat: background-tab streaming must still
  // never move the visible chat, because its events target a detached
  // fragment, not the on-screen transcript.
  const pos = bottom(geoO);
  userScroll(win, O, pos);
  geoO.sh += 400;

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

  // A user scroll away from the tool body's bottom engages its lock:
  // the tool_result event below re-scrolls the latest panel's
  // subpanels and must leave the locked tool body where the user put
  // it while scrolling the new error panel to its end.
  userScroll(win, tcb, 10);

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
  assert.strictEqual(
    tcb.scrollTop,
    10,
    'BUG (' +
      label(remote) +
      '): a tool_result auto-scrolled a tool body the user had ' +
      'scrolled up',
  );
  delete win._geoByClass['tc-b'];
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

  // The same flush must leave a bash subpanel the user scrolled up
  // alone (its own lock is engaged); the text must still land.
  send(win, {type: 'tool_call', name: 'Bash', command: 'make more'});
  const _bps = O.querySelectorAll('.bash-panel-content');
  const bp2 = _bps[_bps.length - 1];
  assert.ok(
    bp2 && bp2 !== bp,
    'the second Bash tool_call must create a bash panel',
  );
  const geoB2 = {sh: 1000, ch: 200};
  fakeGeometry(bp2, geoB2);
  userScroll(win, bp2, 20);
  send(win, {type: 'system_output', text: 'locked tail\n'});
  geoB2.sh += 300;
  send(win, {type: 'tool_call', name: 'Read', path: '/tmp/z'});
  assert.strictEqual(
    bp2.scrollTop,
    20,
    'BUG (' +
      label(remote) +
      '): the flush on the next tool_call auto-scrolled a bash ' +
      'subpanel the user had scrolled up',
  );
  assert.ok(
    bp2.textContent.includes('locked tail'),
    'the flush must still land the text in the locked panel',
  );
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

  // At the bottom, auto-scroll is enabled for action result panels.
  userScroll(win, O, bottom(geoO));
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
// The scroll lock engages on ANY scroll away from the bottom (a 1px
// tolerance absorbs fractional scroll positions), and it only releases
// at the very bottom.
// --------------------------------------------------------------------

async function testScrollLockThresholdAndResume(remote) {
  const {win, posted} = makeWebview({remote});
  const O = win.document.getElementById('output');
  const geo = {sh: 2000, ch: 400};
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

  // 1px above the bottom is within the fractional-scroll tolerance:
  // auto-scroll stays on.
  userScroll(win, O, bottom(geo) - 1);
  geo.sh += 100;
  send(win, {type: 'system_output', text: 'line 2\n'});
  await nextFrames(win);
  assert.strictEqual(
    O.scrollTop,
    bottom(geo),
    'BUG (' +
      label(remote) +
      '): a 1px scroll offset (fractional-scroll tolerance) disabled ' +
      'auto-scroll',
  );

  // 2px above the bottom: auto-scroll must be disabled.
  const lockedTop = bottom(geo) - 2;
  userScroll(win, O, lockedTop);
  geo.sh += 100;
  send(win, {type: 'system_output', text: 'line 3\n'});
  await nextFrames(win);
  assert.strictEqual(
    O.scrollTop,
    lockedTop,
    'BUG (' +
      label(remote) +
      '): scrolling up away from the bottom did not disable auto-scroll',
  );

  // Scrolling further up keeps the lock engaged.
  const higher = bottom(geo) - 300;
  userScroll(win, O, higher);
  geo.sh += 100;
  send(win, {type: 'system_output', text: 'line 4\n'});
  await nextFrames(win);
  assert.strictEqual(
    O.scrollTop,
    higher,
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
    '  ok - lock engages on any scroll away from the bottom and ' +
      'releases at the bottom (' +
      label(remote) +
      ')',
  );
}

// --------------------------------------------------------------------
// Subpanel scroll lock threshold: a subpanel's lock engages on ANY
// user scroll away from its bottom beyond the 1px fractional-scroll
// tolerance, and releases only at the subpanel's bottom.
// --------------------------------------------------------------------

async function testSubpanelLockThresholdAndResume(remote) {
  const {win, posted} = makeWebview({remote});
  const O = win.document.getElementById('output');
  const geoO = {sh: 3000, ch: 500};
  fakeGeometry(O, geoO);
  startRunningTask(win, posted);

  send(win, {type: 'tool_call', name: 'Bash', command: 'tail -f log'});
  const bp = O.querySelector('.bash-panel-content');
  const geoB = {sh: 1000, ch: 200};
  fakeGeometry(bp, geoB);
  send(win, {type: 'system_output', text: 'line 1\n'});
  await nextFrames(win);
  assert.strictEqual(
    bp.scrollTop,
    bottom(geoB),
    'streaming must pin the bash panel to its bottom first (' +
      label(remote) +
      ')',
  );

  // 1px above the subpanel's bottom is within the fractional-scroll
  // tolerance: its auto-scroll stays on.
  userScroll(win, bp, bottom(geoB) - 1);
  geoB.sh += 100;
  send(win, {type: 'system_output', text: 'line 2\n'});
  await nextFrames(win);
  assert.strictEqual(
    bp.scrollTop,
    bottom(geoB),
    'BUG (' +
      label(remote) +
      "): a 1px offset (fractional-scroll tolerance) disabled the " +
      "bash panel's auto-scroll",
  );

  // 2px above the bottom: the subpanel's auto-scroll must be disabled.
  const lockedTop = bottom(geoB) - 2;
  userScroll(win, bp, lockedTop);
  geoB.sh += 100;
  send(win, {type: 'system_output', text: 'line 3\n'});
  await nextFrames(win);
  assert.strictEqual(
    bp.scrollTop,
    lockedTop,
    'BUG (' +
      label(remote) +
      "): scrolling a subpanel away from its bottom did not disable " +
      'its auto-scroll',
  );

  // Scrolling down, but not to the bottom, keeps the lock engaged.
  const nearBottom = bottom(geoB) - 30;
  userScroll(win, bp, nearBottom);
  geoB.sh += 100;
  send(win, {type: 'system_output', text: 'line 4\n'});
  await nextFrames(win);
  assert.strictEqual(
    bp.scrollTop,
    nearBottom,
    'BUG (' +
      label(remote) +
      "): a subpanel's auto-scroll resumed before the user reached its " +
      'bottom',
  );

  // At the subpanel's bottom, the lock releases and it follows again.
  userScroll(win, bp, bottom(geoB));
  geoB.sh += 100;
  send(win, {type: 'system_output', text: 'line 5\n'});
  await nextFrames(win);
  assert.strictEqual(
    bp.scrollTop,
    bottom(geoB),
    'BUG (' +
      label(remote) +
      "): a subpanel's auto-scroll did not resume once the user " +
      'scrolled it to its bottom',
  );
  win.close();
  console.log(
    '  ok - subpanel lock engages away from its bottom and releases ' +
      'at its bottom (' +
      label(remote) +
      ')',
  );
}

// --------------------------------------------------------------------
// A subpanel content shrink (e.g. its text being re-rendered shorter)
// can land a locked subpanel at its bottom WITHOUT any scroll event:
// the stale lock must release and the subpanel must follow again.
// --------------------------------------------------------------------

async function testSubpanelStaleLockReleasedByShrink(remote) {
  const {win, posted} = makeWebview({remote});
  const O = win.document.getElementById('output');
  const geoO = {sh: 3000, ch: 500};
  fakeGeometry(O, geoO);
  startRunningTask(win, posted);

  send(win, {type: 'tool_call', name: 'Bash', command: 'make -j'});
  const bp = O.querySelector('.bash-panel-content');
  const geoB = {sh: 1000, ch: 200};
  fakeGeometry(bp, geoB);
  send(win, {type: 'system_output', text: 'line 1\n'});
  await nextFrames(win);

  // The user scrolls the subpanel to its top: its lock engages.
  userScroll(win, bp, 0);
  geoB.sh += 100;
  send(win, {type: 'system_output', text: 'line 2\n'});
  await nextFrames(win);
  assert.strictEqual(
    bp.scrollTop,
    0,
    'precondition (' + label(remote) + "): the subpanel's lock must be " +
      'engaged',
  );

  // The subpanel shrinks until it is not scrollable at all; scrollTop
  // is already 0, so NO scroll event fires — the cached lock is stale.
  // The next event observes the subpanel at its bottom and must
  // release the stale lock…
  geoB.sh = 150;
  send(win, {type: 'system_output', text: 'small\n'});
  await nextFrames(win);

  // …so the subpanel follows its end again when it grows back.
  geoB.sh = 900;
  send(win, {type: 'system_output', text: 'growing again\n'});
  await nextFrames(win);
  assert.strictEqual(
    bp.scrollTop,
    bottom(geoB),
    'BUG (' +
      label(remote) +
      '): a stale subpanel lock left behind by a content shrink kept ' +
      'suppressing its auto-scroll',
  );
  win.close();
  console.log(
    '  ok - a subpanel shrink to its bottom releases a stale subpanel ' +
      'lock (' +
      label(remote) +
      ')',
  );
}

// --------------------------------------------------------------------
// A subpanel lock lives on the subpanel element: it survives a switch
// to another tab and back (the restore pass scrolls every OTHER
// subpanel of the latest panel to its end, but must leave the locked
// one where the user put it), and it never leaks into the chat lock.
// --------------------------------------------------------------------

async function testSubpanelLockSurvivesTabSwitch(remote) {
  const {win, posted} = makeWebview({remote});
  const O = win.document.getElementById('output');
  const geoO = {sh: 3000, ch: 500};
  fakeGeometry(O, geoO);
  const parentId = startRunningTask(win, posted);

  send(win, {type: 'tool_call', name: 'Bash', command: 'make -j'});
  const bp = O.querySelector('.bash-panel-content');
  const geoB = {sh: 1000, ch: 200};
  fakeGeometry(bp, geoB);
  send(win, {type: 'system_output', text: 'parent output\n'});
  await nextFrames(win);

  // The user scrolls the bash subpanel up: its lock engages.
  userScroll(win, bp, 25);

  // Open a sub-agent tab and switch to it, then back to the parent.
  send(win, {
    type: 'openSubagentTab',
    tab_id: 'sub-tab-lock',
    parent_tab_id: parentId,
    description: 'background worker',
    task_id: 6,
  });
  const subTabEl = win.document.querySelector(
    '.chat-tab[data-tab-id="sub-tab-lock"]',
  );
  assert.ok(subTabEl, 'the sub-agent tab must appear in the tab bar');
  subTabEl.click();
  // Real browsers destroy the scroller of a detached element, silently
  // resetting its scroll offset to 0 (no scroll event fires). jsdom
  // keeps the property, so the reset is simulated here.
  bp.scrollTop = 0;
  const parentTabEl = win.document.querySelector(
    '.chat-tab[data-tab-id="' + parentId + '"]',
  );
  assert.ok(parentTabEl, 'the parent tab must stay in the tab bar');
  parentTabEl.click();

  // The restore pass lands the chat at the end of the latest panel
  // but must put the locked bash subpanel back where the user was
  // reading (not leave it at the detach-reset top, not yank it to its
  // end).
  assert.strictEqual(
    bp.scrollTop,
    25,
    'BUG (' +
      label(remote) +
      '): switching tabs and back lost a locked subpanel\'s reading ' +
      'position',
  );
  assert.strictEqual(
    O.scrollTop,
    bottom(geoO),
    'BUG (' +
      label(remote) +
      '): the restored chat did not land at the end of the latest panel',
  );

  // The lock (and the reading position) still holds against a second
  // switch cycle…
  geoB.sh += 100;
  subTabEl.click();
  bp.scrollTop = 0; // simulated detach reset
  parentTabEl.click();
  assert.strictEqual(
    bp.scrollTop,
    25,
    'BUG (' +
      label(remote) +
      '): a subpanel lock did not survive a second tab switch and back',
  );

  // …until the user scrolls the subpanel back to its bottom: the next
  // restore pass follows the subpanel's end again.
  userScroll(win, bp, bottom(geoB));
  geoB.sh += 100;
  subTabEl.click();
  bp.scrollTop = 0; // simulated detach reset
  parentTabEl.click();
  assert.strictEqual(
    bp.scrollTop,
    bottom(geoB),
    'BUG (' +
      label(remote) +
      "): a subpanel's auto-scroll did not resume after the user " +
      'returned it to its bottom',
  );
  win.close();
  console.log(
    '  ok - a subpanel lock survives a tab switch and back (' +
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
    'precondition (' +
      label(remote) +
      '): the user scroll lock must be engaged',
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
// Scrolling up while NO task is running engages the lock too: a task
// started afterwards (without this tab's own `clear`, e.g. a viewer
// task synchronized from another window) must not yank the reading
// user to the end; auto-scroll starts once the user is at the bottom.
// --------------------------------------------------------------------

async function testIdleScrollAlsoLocks(remote) {
  const {win, posted} = makeWebview({remote});
  const O = win.document.getElementById('output');
  const geo = {sh: 3000, ch: 500};
  fakeGeometry(O, geo);
  const ready = posted.find(m => m.type === 'ready');
  win._testApi.hideWelcome();

  // The user browses history while the chat is idle.
  userScroll(win, O, 30);

  // A task then starts (e.g. synchronized from another window): the
  // chat must not scroll to the end while the user is reading above.
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
    30,
    'BUG (' +
      label(remote) +
      '): a task started while the user was scrolled up yanked the ' +
      'chat to the end',
  );

  // Scrolling to the bottom enables auto-scroll for the new task.
  userScroll(win, O, bottom(geo));
  geo.sh += 100;
  send(win, {type: 'text_delta', text: 'more '});
  await nextFrames(win);
  assert.strictEqual(
    O.scrollTop,
    bottom(geo),
    'BUG (' +
      label(remote) +
      '): auto-scroll did not start after the user scrolled to the ' +
      'bottom',
  );
  win.close();
  console.log(
    '  ok - an idle scroll-up keeps the chat where the user is when a ' +
      'task starts (' +
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

// --------------------------------------------------------------------
// Reading a PREVIOUS task spliced into the same chat: event panels
// generated by the running task must never scroll the chat to the end
// while the user is reading above; auto-scroll resumes only once the
// user scrolls to the end of the chat.
// --------------------------------------------------------------------

async function testReadingPreviousTaskNeverYanked(remote) {
  const {win, posted} = makeWebview({remote});
  const O = win.document.getElementById('output');
  const geo = {sh: 2000, ch: 400};
  fakeGeometry(O, geo);
  const tabId = startRunningTask(win, posted);

  // The running task streams; the chat follows its end.
  send(win, {type: 'setTaskText', text: 'current task', tabId: tabId});
  send(win, {type: 'tool_call', name: 'Bash', command: 'make -j'});
  send(win, {type: 'system_output', text: 'building…\n'});
  await nextFrames(win);
  assert.strictEqual(
    O.scrollTop,
    bottom(geo),
    'precondition (' + label(remote) + '): streaming must follow the end',
  );

  // The user scrolls to the top and the previous task of the same chat
  // is spliced in above the running task's transcript.
  userScroll(win, O, 0);
  send(win, {
    type: 'adjacent_task_events',
    tabId: tabId,
    direction: 'prev',
    task: 'previous task',
    task_id: 3,
    events: [
      {type: 'prompt', text: 'earlier request'},
      {type: 'text_delta', text: 'earlier answer'},
      {type: 'text_end'},
      {type: 'result', summary: 'earlier done', success: true},
    ],
  });
  assert.ok(
    O.querySelector('.adjacent-task'),
    'the previous task must be spliced into the chat',
  );
  geo.sh += 800;

  // The user reads the previous task while the running task keeps
  // generating event panels of every kind: the chat must not move.
  userScroll(win, O, 120);
  geo.sh += 100;
  send(win, {type: 'tool_call', name: 'Read', path: '/tmp/f'});
  geo.sh += 100;
  send(win, {type: 'tool_result', content: 'file body', is_error: false});
  geo.sh += 100;
  send(win, {type: 'thinking_start'});
  send(win, {type: 'thinking_delta', text: 'pondering'});
  send(win, {type: 'thinking_end'});
  geo.sh += 100;
  send(win, {type: 'text_delta', text: 'answer '});
  send(win, {type: 'text_end'});
  geo.sh += 100;
  send(win, {type: 'warning', message: 'heads up'});
  geo.sh += 100;
  send(win, {type: 'system_output', text: 'tail\n'});
  await nextFrames(win);
  assert.strictEqual(
    O.scrollTop,
    120,
    'BUG (' +
      label(remote) +
      "): the running task's event panels scrolled the chat to the " +
      'end while the user was reading a previous task',
  );

  // Only scrolling to the end of the chat resumes auto-scroll.
  userScroll(win, O, bottom(geo));
  geo.sh += 100;
  send(win, {type: 'system_output', text: 'more tail\n'});
  await nextFrames(win);
  assert.strictEqual(
    O.scrollTop,
    bottom(geo),
    'BUG (' +
      label(remote) +
      '): auto-scroll did not resume after the user scrolled to the ' +
      'end of the chat',
  );
  win.close();
  console.log(
    '  ok - reading a previous task is never yanked to the end (' +
      label(remote) +
      ')',
  );
}

// --------------------------------------------------------------------
// A content shrink (e.g. older panels collapsing) can land the chat at
// its bottom WITHOUT the browser firing any scroll event (the scroll
// position itself does not change).  A lock cached before the shrink
// is stale: the next event must release it and resume auto-scroll.
// --------------------------------------------------------------------

async function testCollapseShrinkReleasesStaleLock(remote) {
  const {win, posted} = makeWebview({remote});
  const O = win.document.getElementById('output');
  const geo = {sh: 3000, ch: 500};
  fakeGeometry(O, geo);
  startRunningTask(win, posted);

  send(win, {type: 'text_delta', text: 'tall content '});
  await nextFrames(win);

  // The user scrolls to the very top: the lock engages.
  userScroll(win, O, 0);
  geo.sh += 100;
  send(win, {type: 'text_delta', text: 'more '});
  await nextFrames(win);
  assert.strictEqual(
    O.scrollTop,
    0,
    'precondition (' + label(remote) + '): the lock must be engaged',
  );

  // Panels collapse: the chat shrinks until it is not scrollable at
  // all.  scrollTop is already 0, so the browser fires NO scroll event
  // — the cached lock is now stale.
  geo.sh = 400;

  // The next event panel must release the stale lock (the chat is at
  // its bottom) instead of suppressing auto-scroll forever.
  send(win, {type: 'tool_call', name: 'Bash', command: 'true'});
  geo.sh = 1200;
  send(win, {type: 'system_output', text: 'growing again\n'});
  await nextFrames(win);
  assert.strictEqual(
    O.scrollTop,
    bottom(geo),
    'BUG (' +
      label(remote) +
      '): a stale lock left behind by a content shrink (no scroll ' +
      'event) kept suppressing auto-scroll',
  );
  win.close();
  console.log(
    '  ok - a content shrink to the bottom releases a stale lock (' +
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
    await testSubpanelLockThresholdAndResume(remote);
    await testSubpanelStaleLockReleasedByShrink(remote);
    await testSubpanelLockSurvivesTabSwitch(remote);
    await testSendMessageKeepsLock(remote);
    await testIdleScrollAlsoLocks(remote);
    await testClearEventResetsLock(remote);
    await testReadingPreviousTaskNeverYanked(remote);
    await testCollapseShrinkReleasesStaleLock(remote);
  }
  console.log('chatAutoScroll.test.js: all tests passed');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
