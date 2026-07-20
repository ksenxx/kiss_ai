// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end tests for the per-panel event timestamp badge.
//
// REQUIREMENT: every TOP-LEVEL event panel in the chat webview
// (tool_call, Result, Prompt, System Prompt, Thoughts) must show a
// human-readable timestamp of the event — its FULL DATE plus a
// SECONDS-precision time of day
// ("Mar 5, 2021 2:07:33 PM") — from the event's ``ts`` field (ms
// since epoch) at the LEFT of the panel's
// BOTTOM footer bar — the same ``div.panel-time`` bar whose RIGHT
// side shows the right-aligned "time spent" label
// (``span.panel-elapsed``) — in both the VS Code extension webview
// and the remote web app (which boot the very same media/chat.html +
// panelCopy.js + main.js).  The badge must NOT render in the panel's
// title row.
//
// SUB-panels nested INSIDE a tool-call event panel (the inline bash
// output panel, a successful tool_result output panel, and the
// FAILED tool_result panel) must show NO timestamp badge at all —
// the owning tool-call panel's own footer badge already carries the
// event time.  A tool_result panel that lands at TOP level (no
// owning tool_call panel in the stream) still stamps its own badge.
//
// Additional invariants covered:
//   * the footer bar is the panel's LAST child (bottom), the badge is
//     the bar's FIRST child (left), and the "time spent" label always
//     follows the badge in the same bar;
//   * neither the badge text nor the elapsed label leaks into the
//     Copy button's clipboard payload or the collapsed header
//     preview;
//   * events without a ``ts`` render no badge (old persisted rows);
//   * replayed ``task_events`` show the ORIGINAL event date + time at
//     the bottom of the panel, with the bar re-anchored below content
//     appended after it;
//   * the badge is idempotent (one per panel) and survives the
//     early-prompt in-place replacement.
//
// Before the fix, the ``.panel-ts`` badge was rendered as a direct
// child of the panel absolutely anchored in its TITLE row (not inside
// the ``.panel-time`` bottom bar), so these tests fail — reproducing
// the issue.  Runs the REAL production webview in jsdom (no mocks of
// project code):
//
//     node src/kiss/agents/vscode/test/panelEventTimestamp.test.js

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

let passed = 0;
const failures = [];

async function test(name, fn) {
  try {
    await fn();
    passed++;
    console.log(`  \u2713 ${name}`);
  } catch (e) {
    failures.push({name, error: e});
    console.log(`  \u2717 ${name}`);
    console.log(`      ${e.message}`);
  }
}

/**
 * Build a jsdom window running the production chat webview: the real
 * ``chat.html`` body (placeholders blanked), ``panelCopy.js`` and
 * ``main.js`` evaluated in the window, and a recording
 * ``acquireVsCodeApi`` stub (the only host API the webview has).
 *
 * ``opts.remote`` boots the window the way the REMOTE WEB APP does:
 * the ``remote-chat`` body class (``BODY_CLASS_ATTR`` substitution in
 * web_server.py), the remote-only markup (#frequent-tasks-btn), and a
 * WebSocket-backed ``acquireVsCodeApi`` shim (``_WS_SHIM_JS`` in
 * web_server.py) whose postMessage forwards to the server socket.
 */
function makeWebview(opts) {
  const remote = !!(opts && opts.remote);
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  if (remote) {
    // web_server.py substitutes BODY_CLASS_ATTR with the remote-chat
    // class; the extension host leaves it empty.
    html = html.replace('{{BODY_CLASS_ATTR}}', ' class="remote-chat"');
  }
  html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
  html = html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');

  const dom = new JSDOM(html, {
    runScripts: 'dangerously',
    pretendToBeVisual: true,
    url: remote ? 'https://remote.example/' : 'https://localhost/',
  });
  const win = dom.window;

  win.Element.prototype.scrollIntoView = function () {};
  win.Element.prototype.scrollTo = function () {};
  win.HTMLElement.prototype.scrollTo = function () {};
  // Synchronous rAF so streamed deltas flush immediately.
  win.requestAnimationFrame = function (cb) {
    cb();
    return 0;
  };
  win.cancelAnimationFrame = function () {};

  const posted = [];
  let clipboardText = null;
  Object.defineProperty(win.navigator, 'clipboard', {
    configurable: true,
    value: {
      writeText: text => {
        clipboardText = String(text);
        return Promise.resolve();
      },
    },
  });

  if (remote) {
    // The remote web app provides extra markup around the shared
    // chat.html body (e.g. the frequent-tasks toggle) before the
    // shared scripts run.
    const btn = win.document.createElement('button');
    btn.id = 'frequent-tasks-btn';
    win.document.body.appendChild(btn);
  }
  // Both hosts expose ``acquireVsCodeApi``: the real API in the
  // extension webview, the WebSocket-backed shim (``_WS_SHIM_JS``) in
  // the remote web app.  Recreate that surface with a recorder.
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

  // The sourceURL pragmas name these eval instances in V8 coverage
  // output so panelEventTimestamp.coverage.js can locate them and
  // enforce 100% line coverage of the panelts-coverage regions.
  win.eval(
    fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8') +
      '\n//# sourceURL=panelts-panelCopy.js',
  );
  win.eval(
    fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8') +
      '\n//# sourceURL=panelts-main.js',
  );

  return {win, posted, getClipboard: () => clipboardText};
}

/** Dispatch a backend→webview event exactly like the extension does. */
function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

/** The webview's own tab id (posted in its 'ready' handshake). */
function tabIdOf(wv) {
  const ready = wv.posted.find(m => m.type === 'ready');
  assert.ok(ready && ready.tabId, 'webview must post ready with a tabId');
  return ready.tabId;
}

/** Flush the microtask queue (for the clipboard promise chain). */
function flushMicrotasks() {
  return new Promise(resolve => setImmediate(resolve));
}

/**
 * The label the badge must show for ``ts``.  Mirrors the documented
 * contract: EVERY event carries its full date ("Mon D, YYYY") and a
 * seconds-precision time of day, in the user's locale.
 */
function expectedLabel(ts) {
  const d = new Date(ts);
  const day = d.toLocaleDateString([], {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  });
  const time = d.toLocaleTimeString([], {
    hour: 'numeric',
    minute: '2-digit',
    second: '2-digit',
  });
  return day + ' ' + time;
}

/**
 * Assert that ``panel`` carries exactly one bottom footer bar
 * (direct-child ``.panel-time``, anchored as the panel's LAST child)
 * whose FIRST child is a ``.panel-ts`` badge showing ``ts``'s full
 * date + seconds-precision time label — i.e. the timestamp renders
 * at the bottom-LEFT of the panel,
 * in the same bar as the "time spent" label, NOT in the title row.
 */
function assertBadge(panel, ts, what) {
  const bars = panel.querySelectorAll(':scope > .panel-time');
  assert.strictEqual(
    bars.length,
    1,
    `${what}: expected exactly one .panel-time footer bar`,
  );
  const bar = bars[0];
  assert.strictEqual(
    panel.lastElementChild,
    bar,
    `${what}: the footer bar must be the panel's LAST child (bottom)`,
  );
  const badges = bar.querySelectorAll(':scope > .panel-ts');
  assert.strictEqual(
    badges.length,
    1,
    `${what}: expected one .panel-ts inside the footer bar`,
  );
  const badge = badges[0];
  assert.strictEqual(
    bar.firstElementChild,
    badge,
    `${what}: the badge must be the FIRST child of the bar (left side)`,
  );
  assert.strictEqual(
    badge.textContent,
    expectedLabel(ts),
    `${what}: badge text`,
  );
  // The badge must NOT render in the title row (as a direct panel
  // child) — that is the pre-fix layout.
  assert.strictEqual(
    panel.querySelectorAll(':scope > .panel-ts').length,
    0,
    `${what}: no .panel-ts badge may remain a direct child of the panel`,
  );
  // When the panel also shows a "time spent" label, it must live in
  // the SAME bar, to the RIGHT of (i.e. after) the badge.
  const elapsed = panel.querySelectorAll('.panel-elapsed');
  for (let i = 0; i < elapsed.length; i++) {
    if (elapsed[i].closest('.panel-time') !== bar) continue;
    assert.strictEqual(
      elapsed[i].parentElement,
      bar,
      `${what}: the elapsed label must be a direct child of the bar`,
    );
    assert.ok(
      badge.compareDocumentPosition(elapsed[i]) &
        panel.ownerDocument.defaultView.Node.DOCUMENT_POSITION_FOLLOWING,
      `${what}: the elapsed label must follow the badge in the bar`,
    );
  }
}

/**
 * Assert that ``panel`` — a SUB-panel nested inside a tool-call event
 * panel — carries NO ``.panel-ts`` timestamp badge anywhere in its
 * subtree: the owning tool-call panel's footer badge already shows
 * the event time.
 */
function assertNoBadge(panel, what) {
  assert.strictEqual(
    panel.querySelectorAll('.panel-ts').length,
    0,
    `${what}: a sub-panel of a tool-call panel must show NO timestamp badge`,
  );
}

/** Drive one full agent turn and assert every panel shows its badge. */
async function runFullTranscript(wv, contextName) {
  const win = wv.win;
  const TAB = tabIdOf(wv);
  const output = win.document.getElementById('output');
  const now = Date.now();

  send(win, {type: 'clear', chat_id: 'chat-ts', tabId: TAB});
  send(win, {type: 'status', running: true, tabId: TAB, startTs: now});

  // System prompt + prompt.
  send(win, {
    type: 'system_prompt',
    text: 'You are a test agent.',
    tabId: TAB,
    ts: now,
  });
  send(win, {type: 'prompt', text: 'do the thing', tabId: TAB, ts: now});
  // Footer-bar re-anchoring (MutationObserver) is delivered on the
  // microtask queue — flush before asserting bar positions.
  await flushMicrotasks();
  const sysPanel = output.querySelector('.ev.system-prompt');
  const promptPanel = output.querySelector('.ev.prompt');
  assert.ok(sysPanel, `${contextName}: system prompt panel exists`);
  assert.ok(promptPanel, `${contextName}: prompt panel exists`);
  assertBadge(sysPanel, now, `${contextName}: system prompt`);
  assertBadge(promptPanel, now, `${contextName}: prompt`);

  // Thoughts panel (thinking stream).
  send(win, {type: 'thinking_start', tabId: TAB, ts: now});
  send(win, {type: 'thinking_delta', text: 'Plan: run ls.', tabId: TAB});
  send(win, {type: 'thinking_end', tabId: TAB});
  await flushMicrotasks();
  const thoughts = output.querySelector('.llm-panel');
  assert.ok(thoughts, `${contextName}: Thoughts panel exists`);
  assertBadge(thoughts, now, `${contextName}: Thoughts`);

  // Bash tool call with streamed output.
  send(win, {
    type: 'tool_call',
    name: 'Bash',
    command: 'ls',
    description: 'list',
    tabId: TAB,
    ts: now,
  });
  await flushMicrotasks();
  const tc = output.querySelector('.ev.tc');
  assert.ok(tc, `${contextName}: tool_call panel exists`);
  assertBadge(tc, now, `${contextName}: tool_call`);
  const bp = tc.querySelector('.bash-panel');
  assert.ok(bp, `${contextName}: inline bash panel exists`);
  assertNoBadge(bp, `${contextName}: inline bash panel`);
  send(win, {type: 'system_output', text: 'README.md\n', tabId: TAB});
  send(win, {
    type: 'tool_result',
    content: 'README.md\n',
    is_error: false,
    tabId: TAB,
    ts: now,
  });

  // Non-bash tool call whose successful result renders an output panel.
  send(win, {
    type: 'tool_call',
    name: 'Read',
    path: '/tmp/x',
    tabId: TAB,
    ts: now,
  });
  send(win, {
    type: 'tool_result',
    content: 'file body',
    is_error: false,
    tabId: TAB,
    ts: now,
  });
  await flushMicrotasks();
  const tcs = output.querySelectorAll('.ev.tc');
  const readTc = tcs[tcs.length - 1];
  const op = readTc.querySelector('.bash-panel');
  assert.ok(op, `${contextName}: tool_result output panel exists`);
  assertNoBadge(op, `${contextName}: tool_result output`);
  // The owning tool-call panel keeps exactly ONE badge — its own.
  assert.strictEqual(
    readTc.querySelectorAll('.panel-ts').length,
    1,
    `${contextName}: the tool-call panel keeps only its own badge`,
  );

  // Failing tool call → FAILED panel.
  send(win, {
    type: 'tool_call',
    name: 'Edit',
    path: '/tmp/y',
    tabId: TAB,
    ts: now,
  });
  send(win, {
    type: 'tool_result',
    content: 'boom',
    is_error: true,
    tabId: TAB,
    ts: now,
  });
  await flushMicrotasks();
  const trErr = output.querySelector('.ev.tr.err');
  assert.ok(trErr, `${contextName}: FAILED tool_result panel exists`);
  assertNoBadge(trErr, `${contextName}: FAILED tool_result`);

  // Result panel.
  send(win, {
    type: 'result',
    summary: '# All done',
    total_tokens: 12,
    cost: '$0.01',
    tabId: TAB,
    ts: now,
  });
  await flushMicrotasks();
  const rc = output.querySelector('.ev.rc');
  assert.ok(rc, `${contextName}: Result panel exists`);
  assertBadge(rc, now, `${contextName}: Result`);

  // Live panels that were stamped by the ticker must show the "time
  // spent" label in the SAME bar, to the RIGHT of the timestamp: the
  // bar's last child is the .panel-elapsed span with a duration.
  const tcBar = tc.querySelector(':scope > .panel-time');
  const tcElapsed = tcBar.querySelector(':scope > .panel-elapsed');
  assert.ok(
    tcElapsed,
    `${contextName}: closed tool_call bar must carry a time-spent label`,
  );
  assert.ok(
    /^(\d+ms|\d+(\.\d+)?s|\d+m \d+(\.\d+)?s)$/.test(tcElapsed.textContent),
    `${contextName}: elapsed label must be a duration, got ` +
      JSON.stringify(tcElapsed.textContent),
  );
  assert.strictEqual(
    tcBar.lastElementChild,
    tcElapsed,
    `${contextName}: the time-spent label is the bar's RIGHTMOST item`,
  );
  assert.strictEqual(
    tcBar.firstElementChild.className,
    'panel-ts',
    `${contextName}: the timestamp is the bar's LEFTMOST item`,
  );
}

async function run() {
  // -------------------------------------------------------------------
  // 1. Extension webview: every event panel shows its badge.
  // -------------------------------------------------------------------
  await test('extension webview: every event panel shows the date + seconds timestamp at the LEFT of its bottom time-spent bar', async () => {
    const wv = makeWebview();
    await runFullTranscript(wv, 'extension');
    wv.win.close();
  });

  // -------------------------------------------------------------------
  // 2. Remote web app: same wiring, same badges.
  // -------------------------------------------------------------------
  await test('remote web app: every event panel shows the date + seconds timestamp at the LEFT of its bottom time-spent bar', async () => {
    const wv = makeWebview({remote: true});
    assert.ok(
      wv.win.document.body.classList.contains('remote-chat'),
      'remote harness must boot with the production remote-chat class',
    );
    await runFullTranscript(wv, 'remote');
    wv.win.close();
  });

  // -------------------------------------------------------------------
  // 3. Events without ``ts`` render no badge (old persisted rows).
  // -------------------------------------------------------------------
  await test('events without ts render no badge', () => {
    const wv = makeWebview();
    const win = wv.win;
    const TAB = tabIdOf(wv);
    const output = win.document.getElementById('output');
    send(win, {type: 'clear', chat_id: 'chat-nots', tabId: TAB});
    send(win, {
      type: 'tool_call',
      name: 'Bash',
      command: 'ls',
      tabId: TAB,
    });
    const tc = output.querySelector('.ev.tc');
    assert.ok(tc, 'tool_call panel exists');
    assert.strictEqual(
      tc.querySelectorAll('.panel-ts').length,
      0,
      'no badge without ts',
    );
    win.close();
  });

  // -------------------------------------------------------------------
  // 3b. A tool_result with NO owning tool_call panel in the stream
  //     lands at top level and still stamps its own badge (both the
  //     FAILED and the successful-output shapes).
  // -------------------------------------------------------------------
  await test('top-level tool_result panels (no owning tool_call) keep their badge', async () => {
    const wv = makeWebview();
    const win = wv.win;
    const TAB = tabIdOf(wv);
    const output = win.document.getElementById('output');
    const now = Date.now();
    send(win, {type: 'clear', chat_id: 'chat-toplevel-tr', tabId: TAB});
    send(win, {
      type: 'tool_result',
      content: 'orphan failure',
      is_error: true,
      tabId: TAB,
      ts: now,
    });
    send(win, {
      type: 'tool_result',
      content: 'orphan output',
      is_error: false,
      tabId: TAB,
      ts: now,
    });
    await flushMicrotasks();
    const trErr = output.querySelector('.ev.tr.err');
    assert.ok(trErr, 'top-level FAILED panel exists');
    assertBadge(trErr, now, 'top-level FAILED tool_result');
    const op = output.querySelector('.bash-panel');
    assert.ok(op, 'top-level output panel exists');
    assertBadge(op, now, 'top-level tool_result output');
    win.close();
  });

  // -------------------------------------------------------------------
  // 4. Replayed task_events show the ORIGINAL (old) event time with a
  //    date part.
  // -------------------------------------------------------------------
  await test('replayed task_events show the original event date + time', async () => {
    const wv = makeWebview();
    const win = wv.win;
    const TAB = tabIdOf(wv);
    const output = win.document.getElementById('output');
    // An event from a clearly different day/year than "now".
    const oldTs = new Date(2021, 2, 5, 14, 7).getTime();
    send(win, {
      type: 'task_events',
      tabId: TAB,
      chat_id: 'chat-replay',
      task: 'replayed task',
      events: [
        {type: 'prompt', text: 'old prompt', ts: oldTs},
        {type: 'thinking_start', ts: oldTs},
        {type: 'thinking_delta', text: 'old thought'},
        {type: 'thinking_end'},
        {
          type: 'tool_call',
          name: 'Bash',
          command: 'ls',
          ts: oldTs,
        },
        {type: 'tool_result', content: 'ok', is_error: false, ts: oldTs},
        {
          type: 'result',
          summary: 'done long ago',
          total_tokens: 1,
          cost: '$0',
          ts: oldTs,
        },
      ],
    });
    // Replayed panels get content appended AFTER their footer bar was
    // created — the bar's MutationObserver re-anchors it as the last
    // child on the microtask queue.
    await flushMicrotasks();
    const promptPanel = output.querySelector('.ev.prompt');
    assert.ok(promptPanel, 'replayed prompt panel exists');
    assertBadge(promptPanel, oldTs, 'replayed prompt');
    const thoughts = output.querySelector('.llm-panel');
    assert.ok(thoughts, 'replayed Thoughts panel exists');
    assertBadge(thoughts, oldTs, 'replayed Thoughts');
    const tc = output.querySelector('.ev.tc');
    assert.ok(tc, 'replayed tool_call panel exists');
    assertBadge(tc, oldTs, 'replayed tool_call');
    const rc = output.querySelector('.ev.rc');
    assert.ok(rc, 'replayed result panel exists');
    assertBadge(rc, oldTs, 'replayed result');
    // Replayed panels never enter the live ticker: the bar shows the
    // ORIGINAL event time only, no wall-clock "time spent" label.
    assert.strictEqual(
      thoughts.querySelectorAll('.panel-elapsed').length,
      0,
      'replayed panels show no elapsed label',
    );
    // The label must include the date (old year), not just the time.
    const label = promptPanel.querySelector(
      ':scope > .panel-time > .panel-ts',
    ).textContent;
    assert.ok(
      label.includes('2021'),
      `other-year label must include the year, got ${JSON.stringify(label)}`,
    );
    win.close();
  });

  // -------------------------------------------------------------------
  // 4b. Legacy persisted rows (pre-``ts`` schema) carry only the
  //     ``_timestamp`` DB column (seconds float) injected by the
  //     persistence loaders — the webview must backfill ``ts`` from it
  //     so old tasks show real event times too.  ``ts`` wins when both
  //     are present; junk ``_timestamp`` values render no badge.
  // -------------------------------------------------------------------
  await test('legacy replayed events with only _timestamp show the badge', async () => {
    const wv = makeWebview();
    const win = wv.win;
    const TAB = tabIdOf(wv);
    const output = win.document.getElementById('output');
    const legacySec = new Date(2020, 6, 4, 9, 30).getTime() / 1000;
    const explicitTs = new Date(2022, 1, 2, 8, 15).getTime();
    send(win, {
      type: 'task_events',
      tabId: TAB,
      chat_id: 'chat-legacy',
      task: 'legacy task',
      events: [
        // Pre-``ts`` row: only the loader-injected ``_timestamp``.
        {
          type: 'prompt',
          text: 'legacy prompt',
          _timestamp: legacySec,
        },
        // Both fields: the explicit ``ts`` stamp must win.
        {
          type: 'tool_call',
          name: 'Bash',
          command: 'ls',
          ts: explicitTs,
          _timestamp: legacySec,
        },
        // Junk ``_timestamp`` (TEXT column garbage): no badge.
        {
          type: 'tool_call',
          name: 'Read',
          path: '/tmp/z',
          _timestamp: 'garbage-not-a-number',
        },
      ],
    });
    await flushMicrotasks();
    const promptPanel = output.querySelector('.ev.prompt');
    assert.ok(promptPanel, 'legacy prompt panel exists');
    assertBadge(promptPanel, legacySec * 1000, 'legacy prompt');
    const tcs = output.querySelectorAll('.ev.tc');
    assert.strictEqual(tcs.length, 2, 'both tool_call panels exist');
    assertBadge(tcs[0], explicitTs, 'ts-wins tool_call');
    assert.strictEqual(
      tcs[1].querySelectorAll('.panel-ts').length,
      0,
      'junk _timestamp renders no badge',
    );
    win.close();
  });

  // -------------------------------------------------------------------
  // 5. The badge never leaks into the clipboard payload.
  // -------------------------------------------------------------------
  await test('copy button payload excludes the timestamp badge', async () => {
    const wv = makeWebview();
    const win = wv.win;
    const TAB = tabIdOf(wv);
    const output = win.document.getElementById('output');
    const now = Date.now();
    send(win, {type: 'clear', chat_id: 'chat-copy', tabId: TAB});
    send(win, {
      type: 'tool_call',
      name: 'Bash',
      command: 'echo hi',
      description: 'greet',
      tabId: TAB,
      ts: now,
    });
    await flushMicrotasks();
    const tc = output.querySelector('.ev.tc');
    assertBadge(tc, now, 'copied tool_call');
    const bar = tc.querySelector(':scope > .panel-time');
    const label = bar.querySelector(':scope > .panel-ts').textContent;
    const btn = tc.querySelector(':scope > .panel-copy-btn');
    btn.dispatchEvent(
      new win.MouseEvent('click', {bubbles: true, cancelable: true}),
    );
    await flushMicrotasks();
    const copied = wv.getClipboard();
    assert.ok(copied && copied.includes('echo hi'), 'payload has the command');
    assert.ok(
      !copied.includes(label),
      `clipboard payload must not contain the badge text ${JSON.stringify(label)}: ${JSON.stringify(copied)}`,
    );
    // Nor may the bar's "time spent" label leak into the payload.
    const elapsed = bar.querySelector(':scope > .panel-elapsed');
    assert.ok(elapsed, 'live tool_call bar has an elapsed label');
    assert.ok(
      !copied.includes(elapsed.textContent),
      `clipboard payload must not contain the elapsed text ${JSON.stringify(elapsed.textContent)}: ${JSON.stringify(copied)}`,
    );
    win.close();
  });

  // -------------------------------------------------------------------
  // 6. The badge never leaks into the collapsed header preview.
  // -------------------------------------------------------------------
  await test('collapsed header preview excludes the timestamp badge', () => {
    const wv = makeWebview();
    const win = wv.win;
    const TAB = tabIdOf(wv);
    const output = win.document.getElementById('output');
    const now = Date.now();
    send(win, {type: 'clear', chat_id: 'chat-prev', tabId: TAB});
    send(win, {
      type: 'tool_call',
      name: 'Edit',
      description: 'tweak file',
      tabId: TAB,
      ts: now,
    });
    const tc = output.querySelector('.ev.tc');
    const label = tc.querySelector(
      ':scope > .panel-time > .panel-ts',
    ).textContent;
    // Collapse via a header click, exactly like the user does.
    tc.querySelector('.tc-h').dispatchEvent(
      new win.MouseEvent('click', {bubbles: true, cancelable: true}),
    );
    assert.ok(tc.classList.contains('collapsed'), 'panel collapsed');
    const prev = tc.querySelector('.collapse-preview');
    assert.ok(
      prev.textContent.includes('tweak file'),
      'preview has the description',
    );
    assert.ok(
      !prev.textContent.includes(label),
      `preview must not contain the badge text ${JSON.stringify(label)}`,
    );
    win.close();
  });

  // -------------------------------------------------------------------
  // 7. Early prompt replaced in place keeps exactly one badge.
  // -------------------------------------------------------------------
  await test('early prompt replacement re-stamps exactly one badge', async () => {
    const wv = makeWebview();
    const win = wv.win;
    const TAB = tabIdOf(wv);
    const output = win.document.getElementById('output');
    const earlyTs = Date.now() - 5000;
    const realTs = Date.now();
    send(win, {type: 'clear', chat_id: 'chat-early', tabId: TAB});
    send(win, {
      type: 'prompt',
      text: 'optimistic echo',
      early: true,
      tabId: TAB,
      ts: earlyTs,
    });
    send(win, {
      type: 'prompt',
      text: 'optimistic echo',
      tabId: TAB,
      ts: realTs,
    });
    await flushMicrotasks();
    const prompts = output.querySelectorAll('.ev.prompt');
    assert.strictEqual(prompts.length, 1, 'early panel replaced in place');
    assertBadge(prompts[0], realTs, 'replaced prompt');
    win.close();
  });

  // -------------------------------------------------------------------
  // 8. formatEventTs: date + seconds format edge cases (shared helper).
  // -------------------------------------------------------------------
  await test('formatEventTs always formats full date + seconds-precision time', () => {
    const wv = makeWebview();
    const {PanelCopy} = wv.win;
    assert.strictEqual(typeof PanelCopy.formatEventTs, 'function');
    const sameDay = new Date(2026, 5, 15, 9, 5, 42).getTime();
    const sameYear = new Date(2026, 0, 2, 23, 59, 7).getTime();
    const otherYear = new Date(2024, 11, 31, 0, 30, 59).getTime();
    assert.strictEqual(
      PanelCopy.formatEventTs(sameDay),
      expectedLabel(sameDay),
    );
    assert.strictEqual(
      PanelCopy.formatEventTs(sameYear),
      expectedLabel(sameYear),
    );
    assert.strictEqual(
      PanelCopy.formatEventTs(otherYear),
      expectedLabel(otherYear),
    );
    // EVERY label carries the full date (year included) — even for a
    // recent same-day event — and the seconds of the time of day.
    assert.ok(/2026/.test(PanelCopy.formatEventTs(sameDay)));
    assert.ok(/:05:42/.test(PanelCopy.formatEventTs(sameDay)));
    assert.ok(/2026/.test(PanelCopy.formatEventTs(sameYear)));
    assert.ok(/:59:07/.test(PanelCopy.formatEventTs(sameYear)));
    assert.ok(/2024/.test(PanelCopy.formatEventTs(otherYear)));
    assert.ok(/:30:59/.test(PanelCopy.formatEventTs(otherYear)));
    // Invalid / missing / non-positive inputs render nothing.
    assert.strictEqual(PanelCopy.formatEventTs(undefined), '');
    assert.strictEqual(PanelCopy.formatEventTs(null), '');
    assert.strictEqual(PanelCopy.formatEventTs('junk'), '');
    assert.strictEqual(PanelCopy.formatEventTs(0), '');
    assert.strictEqual(PanelCopy.formatEventTs(-5), '');
    // Finite values beyond the ECMAScript Date range (±8.64e15 ms)
    // produce an Invalid Date — must render nothing, never
    // "Invalid Date Invalid Date".
    assert.strictEqual(PanelCopy.formatEventTs(8.64e15 + 1), '');
    assert.strictEqual(PanelCopy.formatEventTs(1e308), '');
    // Default-now path (no second argument).
    const nowTs = Date.now();
    assert.strictEqual(PanelCopy.formatEventTs(nowTs), expectedLabel(nowTs));
    wv.win.close();
  });

  // -------------------------------------------------------------------
  // 9. addPanelTimestamp: direct API behavior (idempotency, no-ts,
  //    badge lands in the footer bar, hover tooltip).
  // -------------------------------------------------------------------
  await test('addPanelTimestamp stamps the footer bar, is idempotent and skips missing ts', () => {
    const wv = makeWebview();
    const win = wv.win;
    const {PanelCopy} = win;
    assert.strictEqual(typeof PanelCopy.addPanelTimestamp, 'function');
    const doc = win.document;
    const panel = doc.createElement('div');
    doc.body.appendChild(panel);
    const ts = new Date(2026, 0, 1, 12, 0).getTime();
    // No footer bar yet: the bar is created as the panel's last child
    // and the badge is its first child.
    const badge = PanelCopy.addPanelTimestamp(panel, ts);
    assert.ok(badge, 'badge created');
    const bar = panel.querySelector(':scope > .panel-time');
    assert.ok(bar, 'footer bar created');
    assert.strictEqual(panel.lastElementChild, bar, 'bar is the last child');
    assert.strictEqual(bar.firstElementChild, badge, 'badge is bar-first');
    assert.strictEqual(badge.title, new Date(ts).toLocaleString());
    // Second call: same badge, no duplicate.
    const again = PanelCopy.addPanelTimestamp(panel, ts + 1000);
    assert.strictEqual(again, badge);
    assert.strictEqual(panel.querySelectorAll('.panel-ts').length, 1);
    assert.strictEqual(panel.querySelectorAll('.panel-time').length, 1);
    // Missing / invalid ts: no badge, null return.
    const bare = doc.createElement('div');
    assert.strictEqual(PanelCopy.addPanelTimestamp(bare, undefined), null);
    assert.strictEqual(PanelCopy.addPanelTimestamp(bare, 'junk'), null);
    assert.strictEqual(bare.querySelectorAll('.panel-ts').length, 0);
    assert.strictEqual(PanelCopy.addPanelTimestamp(null, ts), null);
    // With an existing bar already holding an elapsed label, the badge
    // is inserted BEFORE it (timestamp left, time spent right).
    const withElapsed = doc.createElement('div');
    doc.body.appendChild(withElapsed);
    const bar2 = PanelCopy.ensurePanelFoot(withElapsed);
    const el2 = doc.createElement('span');
    el2.className = 'panel-elapsed';
    el2.textContent = '3.4s';
    bar2.appendChild(el2);
    const b2 = PanelCopy.addPanelTimestamp(withElapsed, ts);
    assert.strictEqual(b2.parentElement, bar2, 'badge joins the same bar');
    assert.strictEqual(
      b2.nextElementSibling,
      el2,
      'badge must sit to the LEFT of the elapsed label',
    );
    win.close();
  });

  // -------------------------------------------------------------------
  // 9b. ensurePanelFoot: creates one bar, reuses it, and keeps it
  //     anchored as the panel's LAST child when content is appended
  //     after it (MutationObserver re-anchoring).
  // -------------------------------------------------------------------
  await test('ensurePanelFoot reuses one bar and re-anchors it below appended content', async () => {
    const wv = makeWebview();
    const win = wv.win;
    const {PanelCopy} = win;
    assert.strictEqual(typeof PanelCopy.ensurePanelFoot, 'function');
    const doc = win.document;
    const panel = doc.createElement('div');
    doc.body.appendChild(panel);
    const bar = PanelCopy.ensurePanelFoot(panel);
    assert.ok(bar.classList.contains('panel-time'), 'bar has .panel-time');
    assert.strictEqual(panel.lastElementChild, bar, 'bar appended last');
    // Second call returns the SAME bar (no duplicates, one observer).
    assert.strictEqual(PanelCopy.ensurePanelFoot(panel), bar);
    assert.strictEqual(panel.querySelectorAll('.panel-time').length, 1);
    // Content appended after the bar: the observer re-anchors the bar
    // as the last child on the microtask queue.
    const late = doc.createElement('div');
    late.textContent = 'late content';
    panel.appendChild(late);
    assert.strictEqual(panel.lastElementChild, late, 'content lands after');
    await flushMicrotasks();
    assert.strictEqual(
      panel.lastElementChild,
      bar,
      'bar must be re-anchored as the LAST child (bottom of the panel)',
    );
    // A bar inside a NESTED panel is never mistaken for the parent's.
    const nested = doc.createElement('div');
    panel.insertBefore(nested, bar);
    const nestedBar = PanelCopy.ensurePanelFoot(nested);
    assert.notStrictEqual(nestedBar, bar, 'nested panel gets its own bar');
    assert.strictEqual(PanelCopy.ensurePanelFoot(panel), bar);
    win.close();
  });

  // -------------------------------------------------------------------
  // 10. getRawText skips the whole footer bar (shared SKIP_CLASSES
  //     contract): neither timestamp nor elapsed leaks.
  // -------------------------------------------------------------------
  await test('getRawText skips the .panel-time bar and the .panel-ts badge', () => {
    const wv = makeWebview();
    const {PanelCopy} = wv.win;
    const doc = wv.win.document;
    const panel = doc.createElement('div');
    panel.innerHTML =
      '<div>real content</div>' +
      '<div class="panel-time"><span class="panel-ts">9:05 AM</span>' +
      '<span class="panel-elapsed">3.4s</span></div>';
    assert.strictEqual(PanelCopy.getRawText(panel), 'real content');
    // A stray badge outside the bar is skipped too.
    const stray = doc.createElement('div');
    stray.innerHTML =
      '<span class="panel-ts">9:05 AM</span><div>real content</div>';
    assert.strictEqual(PanelCopy.getRawText(stray), 'real content');
    wv.win.close();
  });

  // -------------------------------------------------------------------
  // 11. The stylesheet renders the footer as one flex bar: timestamp
  //     left (in flow, not absolute) and time-spent pushed right.
  // -------------------------------------------------------------------
  await test('main.css renders .panel-time as a flex bar with .panel-ts left and .panel-elapsed right', () => {
    const css = fs.readFileSync(path.join(MEDIA, 'main.css'), 'utf8');
    const barRule = css.match(/\.panel-time\s*\{([^}]*)\}/);
    assert.ok(barRule, 'main.css must style .panel-time');
    assert.ok(
      /display:\s*flex/.test(barRule[1]),
      'the footer bar must be a flex row',
    );
    const elapsedRule = css.match(/\.panel-elapsed\s*\{([^}]*)\}/);
    assert.ok(elapsedRule, 'main.css must style .panel-elapsed');
    assert.ok(
      /margin-left:\s*auto/.test(elapsedRule[1]),
      'the time-spent label must be pushed to the RIGHT of the bar',
    );
    const tsRule = css.match(/\.panel-ts\s*\{([^}]*)\}/);
    assert.ok(tsRule, 'main.css must style .panel-ts');
    assert.ok(
      !/position:\s*absolute/.test(tsRule[1]),
      'the badge must be IN-FLOW in the bar, not absolute in the title row',
    );
    // Collapsing a panel hides the whole bar (no stale .panel-ts
    // exemption remains in the hide rules).
    assert.ok(
      /\.tc\.collapsed\s*>\s*:not\(\.tc-h,\s*\.panel-copy-btn\)/.test(css),
      'collapsed .tc panels hide everything but header and copy button',
    );
    assert.ok(
      /\.llm-panel\.collapsed\s*>\s*:not\(\.llm-panel-hdr,\s*\.panel-copy-btn\)/.test(
        css,
      ),
      'collapsed .llm-panel panels hide everything but header and copy button',
    );
    assert.ok(
      !/:not\([^)]*\.panel-ts/.test(css),
      'no stale .panel-ts exemptions in any :not() hide rule',
    );
    // The remote web app restyles the SAME bar (shared media files).
    const remoteCss = fs.readFileSync(
      path.join(MEDIA, 'remote-codex.css'),
      'utf8',
    );
    assert.ok(
      /body\.remote-chat\s+\.panel-time\s*\{/.test(remoteCss),
      'remote-codex.css keeps its body.remote-chat .panel-time override',
    );
  });

  // ---------------------------------------------------------------------
  // Summary
  // ---------------------------------------------------------------------
  console.log('');
  if (failures.length === 0) {
    console.log(`All ${passed} panelEventTimestamp tests passed.`);
    process.exit(0);
  } else {
    console.log(`${failures.length} test(s) failed (${passed} passed):`);
    for (const f of failures) {
      console.log(`  - ${f.name}: ${f.error.stack || f.error.message}`);
    }
    process.exit(1);
  }
}

run().catch(e => {
  console.error(e);
  process.exit(1);
});
