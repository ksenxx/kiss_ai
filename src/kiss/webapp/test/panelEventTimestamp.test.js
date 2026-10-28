// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const {
  makePage: makeChatPage,
  send,
  stubRaf,
  loadRemoteCss,
  readyTabId,
  stubClipboard,
} = require('./simplify2_harness');

const {test, report} = require('./ui_antipattern_harness').makeRunner();

function makePage() {
  let clip;
  const {win, posted} = makeChatPage({
    bodyClass: 'remote-chat',
    sourceUrl: {
      'panelCopy.js': 'panelts-panelCopy.js',
      'main.js': 'panelts-main.js',
    },
    beforeScripts(win) {
      stubRaf(win);
      clip = stubClipboard(win);
    },
  });
  return {win, posted, getClipboard: () => clip.last()};
}

function flushMicrotasks() {
  return new Promise(resolve => setImmediate(resolve));
}

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
    panel.ownerDocument.defaultView.PanelCopy.formatEventTs(ts),
    `${what}: badge text`,
  );
  assert.strictEqual(
    panel.querySelectorAll(':scope > .panel-ts').length,
    0,
    `${what}: no .panel-ts badge may remain a direct child of the panel`,
  );
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

function assertNoBadge(panel, what) {
  assert.strictEqual(
    panel.querySelectorAll('.panel-ts').length,
    0,
    `${what}: a sub-panel of a tool-call panel must show NO timestamp badge`,
  );
}

async function runFullTranscript(wv, contextName) {
  const win = wv.win;
  const TAB = readyTabId(wv.posted);
  const output = win.document.getElementById('output');
  const now = Date.now();

  send(win, {type: 'clear', chat_id: 'chat-ts', tabId: TAB});
  send(win, {type: 'status', running: true, tabId: TAB, startTs: now});

  send(win, {
    type: 'system_prompt',
    text: 'You are a test agent.',
    tabId: TAB,
    ts: now,
  });
  send(win, {type: 'prompt', text: 'do the thing', tabId: TAB, ts: now});
  await flushMicrotasks();
  const sysPanel = output.querySelector('.ev.system-prompt');
  const promptPanel = output.querySelector('.ev.prompt');
  assert.ok(sysPanel, `${contextName}: system prompt panel exists`);
  assert.ok(promptPanel, `${contextName}: prompt panel exists`);
  assertBadge(sysPanel, now, `${contextName}: system prompt`);
  assertBadge(promptPanel, now, `${contextName}: prompt`);

  send(win, {type: 'thinking_start', tabId: TAB, ts: now});
  send(win, {type: 'thinking_delta', text: 'Plan: run ls.', tabId: TAB});
  send(win, {type: 'thinking_end', tabId: TAB});
  await flushMicrotasks();
  const thoughts = output.querySelector('.llm-panel');
  assert.ok(thoughts, `${contextName}: Thoughts panel exists`);
  assertBadge(thoughts, now, `${contextName}: Thoughts`);

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
  assert.strictEqual(
    readTc.querySelectorAll('.panel-ts').length,
    1,
    `${contextName}: the tool-call panel keeps only its own badge`,
  );

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
  await test('remote web app: every event panel shows the date + seconds timestamp at the LEFT of its bottom time-spent bar', async () => {
    const wv = makePage();
    assert.ok(
      wv.win.document.body.classList.contains('remote-chat'),
      'the harness must boot with the production remote-chat class',
    );
    await runFullTranscript(wv, 'remote');
    wv.win.close();
  });

  await test('events without ts render no badge', () => {
    const wv = makePage();
    const win = wv.win;
    const TAB = readyTabId(wv.posted);
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

  await test('top-level tool_result panels (no owning tool_call) keep their badge', async () => {
    const wv = makePage();
    const win = wv.win;
    const TAB = readyTabId(wv.posted);
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

  await test('replayed task_events show the original event date + time', async () => {
    const wv = makePage();
    const win = wv.win;
    const TAB = readyTabId(wv.posted);
    const output = win.document.getElementById('output');
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
        {
          type: 'tool_result',
          content: 'ok',
          is_error: false,
          ts: oldTs + 2500,
        },
        {
          type: 'result',
          summary: 'done long ago',
          total_tokens: 1,
          cost: '$0',
          ts: oldTs,
        },
      ],
    });
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
    // Replayed panels derive their durations from the persisted event
    // timestamps: the Thoughts panel was closed by the tool_call at the
    // same instant it opened (0ms), the tool_call by its tool_result
    // 2.5s later. Panels no event closes (prompt, result) carry none.
    const thoughtsElapsed = thoughts.querySelectorAll('.panel-elapsed');
    assert.strictEqual(
      thoughtsElapsed.length,
      1,
      'replayed Thoughts panel shows a timestamp-derived elapsed label',
    );
    assert.strictEqual(
      thoughtsElapsed[0].textContent,
      '0ms',
      'Thoughts duration is the gap to the tool_call that closed it',
    );
    const tcReplayElapsed = tc.querySelectorAll('.panel-elapsed');
    assert.strictEqual(
      tcReplayElapsed.length,
      1,
      'replayed tool_call panel shows a timestamp-derived elapsed label',
    );
    assert.strictEqual(
      tcReplayElapsed[0].textContent,
      '2.5s',
      'tool_call duration is the gap to its tool_result',
    );
    assert.strictEqual(
      promptPanel.querySelectorAll('.panel-elapsed').length,
      0,
      'a replayed prompt panel has no duration',
    );
    assert.strictEqual(
      rc.querySelectorAll('.panel-elapsed').length,
      0,
      'a replayed result panel has no duration',
    );
    assert.strictEqual(
      output.querySelectorAll('[data-start-ms]').length,
      0,
      'a finished replay must not put any panel on the live tick',
    );
    const label = promptPanel.querySelector(
      ':scope > .panel-time > .panel-ts',
    ).textContent;
    assert.ok(
      label.includes('2021'),
      `other-year label must include the year, got ${JSON.stringify(label)}`,
    );
    win.close();
  });

  await test('legacy replayed events with only _timestamp show the badge', async () => {
    const wv = makePage();
    const win = wv.win;
    const TAB = readyTabId(wv.posted);
    const output = win.document.getElementById('output');
    const legacySec = new Date(2020, 6, 4, 9, 30).getTime() / 1000;
    const explicitTs = new Date(2022, 1, 2, 8, 15).getTime();
    send(win, {
      type: 'task_events',
      tabId: TAB,
      chat_id: 'chat-legacy',
      task: 'legacy task',
      events: [
        {
          type: 'prompt',
          text: 'legacy prompt',
          _timestamp: legacySec,
        },
        {
          type: 'tool_call',
          name: 'Bash',
          command: 'ls',
          ts: explicitTs,
          _timestamp: legacySec,
        },
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

  await test('copy button payload excludes the timestamp badge', async () => {
    const wv = makePage();
    const win = wv.win;
    const TAB = readyTabId(wv.posted);
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
    const elapsed = bar.querySelector(':scope > .panel-elapsed');
    assert.ok(elapsed, 'live tool_call bar has an elapsed label');
    assert.ok(
      !copied.includes(elapsed.textContent),
      `clipboard payload must not contain the elapsed text ${JSON.stringify(elapsed.textContent)}: ${JSON.stringify(copied)}`,
    );
    win.close();
  });

  await test('collapsed header preview excludes the timestamp badge', () => {
    const wv = makePage();
    const win = wv.win;
    const TAB = readyTabId(wv.posted);
    const output = win.document.getElementById('output');
    const now = Date.now();
    send(win, {type: 'clear', chat_id: 'chat-prev', tabId: TAB});
    send(win, {type: 'status', running: true, tabId: TAB, startTs: now});
    send(win, {
      type: 'tool_call',
      name: 'Edit',
      description: 'tweak file',
      tabId: TAB,
      ts: now,
    });
    const tc = output.querySelector('.ev.tc');
    assert.ok(
      tc.classList.contains('collapsed'),
      'the live panel starts folded',
    );
    const label = tc.querySelector(
      ':scope > .panel-time > .panel-ts',
    ).textContent;
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

  await test('early prompt replacement re-stamps exactly one badge', async () => {
    const wv = makePage();
    const win = wv.win;
    const TAB = readyTabId(wv.posted);
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
    // The in-place replacement is pinned by promptPanelEarlyReplace.
    assertBadge(output.querySelector('.ev.prompt'), realTs, 'replaced prompt');
    win.close();
  });

  await test('formatEventTs always formats full date + seconds-precision time', () => {
    const wv = makePage();
    const {PanelCopy} = wv.win;
    assert.strictEqual(typeof PanelCopy.formatEventTs, 'function');
    const sameDay = new Date(2026, 5, 15, 9, 5, 42).getTime();
    const sameYear = new Date(2026, 0, 2, 23, 59, 7).getTime();
    const otherYear = new Date(2024, 11, 31, 0, 30, 59).getTime();
    assert.ok(/2026/.test(PanelCopy.formatEventTs(sameDay)));
    assert.ok(/:05:42/.test(PanelCopy.formatEventTs(sameDay)));
    assert.ok(/2026/.test(PanelCopy.formatEventTs(sameYear)));
    assert.ok(/:59:07/.test(PanelCopy.formatEventTs(sameYear)));
    assert.ok(/2024/.test(PanelCopy.formatEventTs(otherYear)));
    assert.ok(/:30:59/.test(PanelCopy.formatEventTs(otherYear)));
    assert.strictEqual(PanelCopy.formatEventTs(undefined), '');
    assert.strictEqual(PanelCopy.formatEventTs(null), '');
    assert.strictEqual(PanelCopy.formatEventTs('junk'), '');
    assert.strictEqual(PanelCopy.formatEventTs(0), '');
    assert.strictEqual(PanelCopy.formatEventTs(-5), '');
    assert.strictEqual(PanelCopy.formatEventTs(8.64e15 + 1), '');
    assert.strictEqual(PanelCopy.formatEventTs(1e308), '');
    wv.win.close();
  });

  await test('addPanelTimestamp stamps the footer bar, is idempotent and skips missing ts', () => {
    const wv = makePage();
    const win = wv.win;
    const {PanelCopy} = win;
    assert.strictEqual(typeof PanelCopy.addPanelTimestamp, 'function');
    const doc = win.document;
    const panel = doc.createElement('div');
    doc.body.appendChild(panel);
    const ts = new Date(2026, 0, 1, 12, 0).getTime();
    const badge = PanelCopy.addPanelTimestamp(panel, ts);
    assert.ok(badge, 'badge created');
    const bar = panel.querySelector(':scope > .panel-time');
    assert.ok(bar, 'footer bar created');
    assert.strictEqual(panel.lastElementChild, bar, 'bar is the last child');
    assert.strictEqual(bar.firstElementChild, badge, 'badge is bar-first');
    assert.strictEqual(badge.title, new Date(ts).toLocaleString());
    const again = PanelCopy.addPanelTimestamp(panel, ts + 1000);
    assert.strictEqual(again, badge);
    assert.strictEqual(panel.querySelectorAll('.panel-ts').length, 1);
    assert.strictEqual(panel.querySelectorAll('.panel-time').length, 1);
    const bare = doc.createElement('div');
    assert.strictEqual(PanelCopy.addPanelTimestamp(bare, undefined), null);
    assert.strictEqual(PanelCopy.addPanelTimestamp(bare, 'junk'), null);
    assert.strictEqual(bare.querySelectorAll('.panel-ts').length, 0);
    assert.strictEqual(PanelCopy.addPanelTimestamp(null, ts), null);
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

  await test('ensurePanelFoot reuses one bar and re-anchors it below appended content', async () => {
    const wv = makePage();
    const win = wv.win;
    const {PanelCopy} = win;
    assert.strictEqual(typeof PanelCopy.ensurePanelFoot, 'function');
    const doc = win.document;
    const panel = doc.createElement('div');
    doc.body.appendChild(panel);
    const bar = PanelCopy.ensurePanelFoot(panel);
    assert.ok(bar.classList.contains('panel-time'), 'bar has .panel-time');
    assert.strictEqual(panel.lastElementChild, bar, 'bar appended last');
    assert.strictEqual(PanelCopy.ensurePanelFoot(panel), bar);
    assert.strictEqual(panel.querySelectorAll('.panel-time').length, 1);
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
    const nested = doc.createElement('div');
    panel.insertBefore(nested, bar);
    const nestedBar = PanelCopy.ensurePanelFoot(nested);
    assert.notStrictEqual(nestedBar, bar, 'nested panel gets its own bar');
    assert.strictEqual(PanelCopy.ensurePanelFoot(panel), bar);
    win.close();
  });

  await test('getRawText skips the .panel-time bar and the .panel-ts badge', () => {
    const wv = makePage();
    const {PanelCopy} = wv.win;
    const doc = wv.win.document;
    const panel = doc.createElement('div');
    panel.innerHTML =
      '<div>real content</div>' +
      '<div class="panel-time"><span class="panel-ts">9:05 AM</span>' +
      '<span class="panel-elapsed">3.4s</span></div>';
    assert.strictEqual(PanelCopy.getRawText(panel), 'real content');
    const stray = doc.createElement('div');
    stray.innerHTML =
      '<span class="panel-ts">9:05 AM</span><div>real content</div>';
    assert.strictEqual(PanelCopy.getRawText(stray), 'real content');
    wv.win.close();
  });

  await test('the remote page paints .panel-time as a flex bar (badge in flow, elapsed pushed right) that folds with its panel', () => {
    // The remote webapp's stylesheets (main.css plus remote-codex.css)
    // are applied to the real transcript and the cascaded values read.
    const wv = makePage();
    const win = wv.win;
    loadRemoteCss(win, true);
    const TAB = readyTabId(wv.posted);
    const output = win.document.getElementById('output');
    const now = Date.now();
    const style = el => win.getComputedStyle(el);
    send(win, {type: 'clear', chat_id: 'chat-css', tabId: TAB});
    send(win, {type: 'status', running: true, tabId: TAB, startTs: now});
    send(win, {type: 'thinking_start', tabId: TAB, ts: now});
    send(win, {type: 'thinking_delta', text: 'a thought', tabId: TAB});
    send(win, {type: 'thinking_end', tabId: TAB});
    send(win, {
      type: 'tool_call',
      name: 'Bash',
      command: 'ls',
      description: 'list',
      tabId: TAB,
      ts: now,
    });
    send(win, {
      type: 'tool_result',
      content: 'README.md',
      is_error: false,
      tabId: TAB,
      ts: now,
    });
    const tc = output.querySelector('.ev.tc');
    const bar = tc.querySelector(':scope > .panel-time');
    assert.ok(tc.classList.contains('collapsed'), 'the tool panel starts folded');
    assert.strictEqual(
      style(bar).display,
      'none',
      'a folded .tc panel hides its footer bar (and so its badge)',
    );
    assert.notStrictEqual(style(tc.querySelector('.tc-h')).display, 'none', 'the header stays');
    assert.notStrictEqual(
      style(tc.querySelector('.panel-copy-btn')).display,
      'none',
      'the copy button stays',
    );
    assert.strictEqual(style(tc.querySelector('.bash-panel')).display, 'none', 'the body folds');
    tc.querySelector('.tc-h').click();
    assert.ok(!tc.classList.contains('collapsed'), 'a header click unfolds the panel');
    assert.strictEqual(style(bar).display, 'flex', 'the footer bar is a flex row');
    assert.strictEqual(
      style(bar.querySelector('.panel-ts')).position,
      'static',
      'the badge is IN-FLOW in the bar, not absolute in the title row',
    );
    assert.strictEqual(
      style(bar.querySelector('.panel-elapsed')).marginLeft,
      'auto',
      'the time-spent label is pushed to the RIGHT of the bar',
    );
    const thoughts = output.querySelector('.llm-panel');
    const thoughtsBar = thoughts.querySelector(':scope > .panel-time');
    assert.strictEqual(style(thoughtsBar).display, 'flex');
    thoughts.classList.add('collapsed');
    assert.strictEqual(
      style(thoughtsBar).display,
      'none',
      'a folded Thoughts panel hides its footer bar too',
    );
    assert.notStrictEqual(style(thoughts.querySelector('.llm-panel-hdr')).display, 'none');
    const thoughtsCopy = thoughts.querySelector(':scope > .panel-copy-btn');
    assert.ok(thoughtsCopy, 'a Thoughts panel has its copy button');
    assert.notStrictEqual(
      style(thoughtsCopy).display,
      'none',
      'a folded Thoughts panel keeps its copy button visible',
    );
    win.close();
  });

  report('panelEventTimestamp');
  process.exit(0);
}

run().catch(e => {
  console.error(e);
  process.exit(1);
});
