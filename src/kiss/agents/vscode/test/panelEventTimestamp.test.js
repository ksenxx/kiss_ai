// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end tests for the per-panel event timestamp badge.
//
// REQUIREMENT: the title row of every event panel in the chat webview
// (tool_call, tool_result error, tool output, Result, Prompt, System
// Prompt, Thoughts) must show a human-readable COMPACT timestamp of
// the event (its ``ts`` field, ms since epoch) to the LEFT of the
// panel's Copy button — in both the VS Code extension webview and the
// remote web app (which boot the very same media/chat.html +
// panelCopy.js + main.js).
//
// Additional invariants covered:
//   * the badge text never leaks into the Copy button's clipboard
//     payload nor into the collapsed header preview;
//   * events without a ``ts`` render no badge (old persisted rows);
//   * replayed ``task_events`` show the ORIGINAL event time (with a
//     date part when the event is from another day / year);
//   * the badge is idempotent (one per panel) and survives the
//     early-prompt in-place replacement.
//
// Before the fix, no ``.panel-ts`` badge exists anywhere, so these
// tests fail.  Runs the REAL production webview in jsdom (no mocks of
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
 * no ``acquireVsCodeApi`` (main.js falls back to its WebSocket
 * transport shim) plus the remote-only markup (#frequent-tasks-btn).
 */
function makeWebview(opts) {
  const remote = !!(opts && opts.remote);
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
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
    // chat.html body (e.g. the frequent-tasks toggle) and injects a
    // WebSocket-backed ``acquireVsCodeApi`` shim (``_WS_SHIM_JS`` in
    // web_server.py) before the shared scripts run.  Recreate that
    // host surface: a shim whose postMessage forwards to the server
    // socket (recorded here) instead of the extension host.
    const btn = win.document.createElement('button');
    btn.id = 'frequent-tasks-btn';
    win.document.body.appendChild(btn);
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
  } else {
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
  }

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
 * The compact label the badge must show for ``ts``.  Mirrors the
 * documented contract: time-of-day for a same-day event, "Mon D" +
 * time for another day of the current year, "Mon D, YYYY" + time
 * otherwise (all in the user's locale).
 */
function expectedLabel(ts, now) {
  const d = new Date(ts);
  const n = now instanceof Date ? now : new Date();
  const time = d.toLocaleTimeString([], {hour: 'numeric', minute: '2-digit'});
  if (
    d.getFullYear() === n.getFullYear() &&
    d.getMonth() === n.getMonth() &&
    d.getDate() === n.getDate()
  )
    return time;
  const day = d.toLocaleDateString([], {month: 'short', day: 'numeric'});
  if (d.getFullYear() === n.getFullYear()) return day + ' ' + time;
  return (
    d.toLocaleDateString([], {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
    }) +
    ' ' +
    time
  );
}

/**
 * Assert that ``panel`` carries exactly one direct-child ``.panel-ts``
 * badge showing ``ts``'s compact label, positioned immediately to the
 * LEFT of the panel's Copy button.
 */
function assertBadge(panel, ts, what) {
  const badges = panel.querySelectorAll(':scope > .panel-ts');
  assert.strictEqual(badges.length, 1, `${what}: expected one .panel-ts`);
  const badge = badges[0];
  assert.strictEqual(
    badge.textContent,
    expectedLabel(ts),
    `${what}: badge text`,
  );
  const btn = panel.querySelector(':scope > .panel-copy-btn');
  assert.ok(btn, `${what}: panel must have a copy button`);
  assert.strictEqual(
    badge.nextElementSibling,
    btn,
    `${what}: badge must sit immediately to the left of the copy button`,
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
  const tc = output.querySelector('.ev.tc');
  assert.ok(tc, `${contextName}: tool_call panel exists`);
  assertBadge(tc, now, `${contextName}: tool_call`);
  const bp = tc.querySelector('.bash-panel');
  assert.ok(bp, `${contextName}: inline bash panel exists`);
  assertBadge(bp, now, `${contextName}: inline bash panel`);
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
  const tcs = output.querySelectorAll('.ev.tc');
  const readTc = tcs[tcs.length - 1];
  const op = readTc.querySelector('.bash-panel');
  assert.ok(op, `${contextName}: tool_result output panel exists`);
  assertBadge(op, now, `${contextName}: tool_result output`);

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
  const trErr = output.querySelector('.ev.tr.err');
  assert.ok(trErr, `${contextName}: FAILED tool_result panel exists`);
  assertBadge(trErr, now, `${contextName}: FAILED tool_result`);

  // Result panel.
  send(win, {
    type: 'result',
    summary: '# All done',
    total_tokens: 12,
    cost: '$0.01',
    tabId: TAB,
    ts: now,
  });
  const rc = output.querySelector('.ev.rc');
  assert.ok(rc, `${contextName}: Result panel exists`);
  assertBadge(rc, now, `${contextName}: Result`);
}

async function run() {
  // -------------------------------------------------------------------
  // 1. Extension webview: every event panel shows its badge.
  // -------------------------------------------------------------------
  await test('extension webview: every event panel title shows the compact timestamp left of the copy button', async () => {
    const wv = makeWebview();
    await runFullTranscript(wv, 'extension');
    wv.win.close();
  });

  // -------------------------------------------------------------------
  // 2. Remote web app: same wiring, same badges.
  // -------------------------------------------------------------------
  await test('remote web app: every event panel title shows the compact timestamp left of the copy button', async () => {
    const wv = makeWebview({remote: true});
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
      tc.querySelectorAll(':scope > .panel-ts').length,
      0,
      'no badge without ts',
    );
    win.close();
  });

  // -------------------------------------------------------------------
  // 4. Replayed task_events show the ORIGINAL (old) event time with a
  //    date part.
  // -------------------------------------------------------------------
  await test('replayed task_events show the original event date + time', () => {
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
    // The label must include the date (old year), not just the time.
    const label = promptPanel.querySelector(':scope > .panel-ts').textContent;
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
  await test('legacy replayed events with only _timestamp show the badge', () => {
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
    const promptPanel = output.querySelector('.ev.prompt');
    assert.ok(promptPanel, 'legacy prompt panel exists');
    assertBadge(promptPanel, legacySec * 1000, 'legacy prompt');
    const tcs = output.querySelectorAll('.ev.tc');
    assert.strictEqual(tcs.length, 2, 'both tool_call panels exist');
    assertBadge(tcs[0], explicitTs, 'ts-wins tool_call');
    assert.strictEqual(
      tcs[1].querySelectorAll(':scope > .panel-ts').length,
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
    const tc = output.querySelector('.ev.tc');
    assertBadge(tc, now, 'copied tool_call');
    const label = tc.querySelector(':scope > .panel-ts').textContent;
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
    const label = tc.querySelector(':scope > .panel-ts').textContent;
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
  await test('early prompt replacement re-stamps exactly one badge', () => {
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
    const prompts = output.querySelectorAll('.ev.prompt');
    assert.strictEqual(prompts.length, 1, 'early panel replaced in place');
    assertBadge(prompts[0], realTs, 'replaced prompt');
    win.close();
  });

  // -------------------------------------------------------------------
  // 8. formatEventTs: compact-format edge cases (shared helper).
  // -------------------------------------------------------------------
  await test('formatEventTs formats same-day / same-year / other-year / invalid', () => {
    const wv = makeWebview();
    const {PanelCopy} = wv.win;
    assert.strictEqual(typeof PanelCopy.formatEventTs, 'function');
    const now = new Date(2026, 5, 15, 10, 0); // Jun 15 2026 10:00
    const sameDay = new Date(2026, 5, 15, 9, 5).getTime();
    const sameYear = new Date(2026, 0, 2, 23, 59).getTime();
    const otherYear = new Date(2024, 11, 31, 0, 30).getTime();
    assert.strictEqual(
      PanelCopy.formatEventTs(sameDay, now),
      expectedLabel(sameDay, now),
    );
    assert.strictEqual(
      PanelCopy.formatEventTs(sameYear, now),
      expectedLabel(sameYear, now),
    );
    assert.strictEqual(
      PanelCopy.formatEventTs(otherYear, now),
      expectedLabel(otherYear, now),
    );
    // Same-day labels are pure time-of-day (no date part).
    assert.ok(!/2026/.test(PanelCopy.formatEventTs(sameDay, now)));
    // Same-year labels carry a date but no year.
    assert.ok(!/2026/.test(PanelCopy.formatEventTs(sameYear, now)));
    // Other-year labels carry the year.
    assert.ok(/2024/.test(PanelCopy.formatEventTs(otherYear, now)));
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
  //    badge-only panels, hover tooltip).
  // -------------------------------------------------------------------
  await test('addPanelTimestamp is idempotent and skips missing ts', () => {
    const wv = makeWebview();
    const win = wv.win;
    const {PanelCopy} = win;
    assert.strictEqual(typeof PanelCopy.addPanelTimestamp, 'function');
    const doc = win.document;
    const panel = doc.createElement('div');
    doc.body.appendChild(panel);
    const ts = new Date(2026, 0, 1, 12, 0).getTime();
    // No copy button yet: badge is appended and panel becomes copyable.
    const badge = PanelCopy.addPanelTimestamp(panel, ts);
    assert.ok(badge, 'badge created');
    assert.ok(panel.classList.contains('copyable'));
    assert.strictEqual(badge.title, new Date(ts).toLocaleString());
    // Second call: same badge, no duplicate.
    const again = PanelCopy.addPanelTimestamp(panel, ts + 1000);
    assert.strictEqual(again, badge);
    assert.strictEqual(panel.querySelectorAll('.panel-ts').length, 1);
    // Missing / invalid ts: no badge, null return.
    const bare = doc.createElement('div');
    assert.strictEqual(PanelCopy.addPanelTimestamp(bare, undefined), null);
    assert.strictEqual(PanelCopy.addPanelTimestamp(bare, 'junk'), null);
    assert.strictEqual(bare.querySelectorAll('.panel-ts').length, 0);
    assert.strictEqual(PanelCopy.addPanelTimestamp(null, ts), null);
    // With an existing copy button the badge lands right before it.
    const withBtn = doc.createElement('div');
    doc.body.appendChild(withBtn);
    PanelCopy.addCopyButton(withBtn);
    const b2 = PanelCopy.addPanelTimestamp(withBtn, ts);
    assert.strictEqual(
      b2.nextElementSibling,
      withBtn.querySelector(':scope > .panel-copy-btn'),
    );
    win.close();
  });

  // -------------------------------------------------------------------
  // 10. getRawText skips the badge (shared SKIP_CLASSES contract).
  // -------------------------------------------------------------------
  await test('getRawText skips the .panel-ts badge', () => {
    const wv = makeWebview();
    const {PanelCopy} = wv.win;
    const doc = wv.win.document;
    const panel = doc.createElement('div');
    panel.innerHTML =
      '<span class="panel-ts">9:05 AM</span><div>real content</div>';
    assert.strictEqual(PanelCopy.getRawText(panel), 'real content');
    wv.win.close();
  });

  // -------------------------------------------------------------------
  // 11. The stylesheet keeps the badge visible on collapsed panels and
  //     anchors it left of the copy button.
  // -------------------------------------------------------------------
  await test('main.css anchors .panel-ts left of the copy button and keeps it visible when collapsed', () => {
    const css = fs.readFileSync(path.join(MEDIA, 'main.css'), 'utf8');
    const tsRule = css.match(/\.panel-ts\s*\{([^}]*)\}/);
    assert.ok(tsRule, 'main.css must style .panel-ts');
    const body = tsRule[1];
    assert.ok(/position:\s*absolute/.test(body), '.panel-ts is absolute');
    const right = body.match(/right:\s*(\d+)px/);
    assert.ok(right, '.panel-ts sets a right offset');
    const btnRule = css.match(/\.panel-copy-btn\s*\{([^}]*)\}/);
    const btnRight = btnRule[1].match(/right:\s*(\d+)px/);
    assert.ok(
      Number(right[1]) > Number(btnRight[1]),
      'badge right offset must clear the copy button (sit to its left)',
    );
    // Collapsed panels keep the badge visible (exempt from the hide rules).
    assert.ok(
      /\.tc\.collapsed\s*>\s*:not\(\.tc-h,\s*\.panel-copy-btn,\s*\.panel-ts\)/.test(
        css,
      ),
      'collapsed .tc panels must exempt .panel-ts from hiding',
    );
    assert.ok(
      /\.llm-panel\.collapsed\s*>\s*:not\(\.llm-panel-hdr,\s*\.panel-copy-btn,\s*\.panel-ts\)/.test(
        css,
      ),
      'collapsed .llm-panel panels must exempt .panel-ts from hiding',
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
