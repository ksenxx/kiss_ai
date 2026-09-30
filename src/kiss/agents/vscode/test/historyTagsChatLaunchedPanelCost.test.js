// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Three additions to the chat webview, tested against the real
// chat.html + main.js + panelCopy.js in JSDOM:
//
//  * every task panel in the task-history list shows its classification
//    tags ("work · coding") right after its "... ago" age label;
//  * every collapsible chat panel carries a "last launched ... ago" line
//    under its header, fed by the daemon's `chat_last_launched` stamp
//    (falling back to the newest row loaded) and kept current by the
//    same 30 s sweep as the task labels;
//  * a thoughts panel shows the USD cost of the model call it holds next
//    to its time label, from the `llm_call` event, live and on replay.

'use strict';

/* global require, __dirname, console, process, setImmediate */

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');
const HOUR_MS = 3600 * 1000;

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
  win.cancelAnimationFrame = function () {};
  const style = win.document.createElement('style');
  style.textContent = fs.readFileSync(path.join(MEDIA, 'main.css'), 'utf8');
  win.document.head.appendChild(style);
  // Every setTimeout the webview arms, so a test can fire a long timer
  // (the 30 s "launched ... ago" sweep) without waiting for it.
  const timers = [];
  const realSetTimeout = win.setTimeout;
  win.setTimeout = function (cb, delay) {
    timers.push({cb, delay});
    return realSetTimeout.apply(win, arguments);
  };
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
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
  return {win, posted, timers};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function tabIdOf(wv) {
  const ready = wv.posted.find(m => m.type === 'ready');
  assert.ok(ready && ready.tabId, 'webview must post ready with a tabId');
  return ready.tabId;
}

function flush() {
  return new Promise(resolve => setImmediate(resolve));
}

function session(overrides) {
  return Object.assign(
    {
      id: 'chat-1',
      task_id: 'task-1',
      title: 'refactor the parser',
      preview: 'refactor the parser',
      has_events: true,
      tokens: 1,
      cost: 0.1,
      steps: 1,
      timestamp: 1700000000,
      model: 'test-model',
    },
    overrides || {},
  );
}

function loadHistory(win, sessions) {
  send(win, {type: 'history', offset: 0, generation: 0, sessions});
}

function rows(win) {
  return Array.from(
    win.document.querySelectorAll('#history-list .sidebar-item'),
  );
}

function groups(win) {
  return Array.from(
    win.document.querySelectorAll('#history-list > .history-chat-group'),
  );
}

function testTagsAfterLaunched() {
  const {win} = makeWebview();
  const now = Date.now();
  loadHistory(win, [
    session({
      id: 'chat-a',
      task_id: 'task-a',
      startTs: now - 2 * HOUR_MS,
      tags: 'work,coding, testing ,',
    }),
    session({
      id: 'chat-b',
      task_id: 'task-b',
      startTs: now - HOUR_MS,
      tags: '',
    }),
    session({id: 'chat-c', task_id: 'task-c', startTs: now - HOUR_MS}),
    session({
      id: 'chat-d',
      task_id: 'task-d',
      startTs: now - HOUR_MS,
      tags: 42,
    }),
  ]);
  const list = rows(win);
  assert.strictEqual(list.length, 4, 'one row per session');
  const tagged = list[0].querySelector('.sidebar-item-tags');
  assert.ok(tagged, 'a tagged row renders the tags span');
  assert.strictEqual(tagged.textContent, 'work · coding · testing');
  assert.strictEqual(tagged.title, 'Tags: work, coding, testing');
  const launched = list[0].querySelector('.sidebar-item-launched');
  assert.strictEqual(
    launched.nextElementSibling,
    tagged,
    'the tags sit immediately after the "... ago" age label',
  );
  assert.strictEqual(
    list[0].querySelector('.sidebar-item-collapse').nextElementSibling,
    launched,
    'the age label follows the last button of the strip',
  );
  assert.strictEqual(launched.textContent, '2 hours ago');
  assert.ok(
    launched.title.indexOf('Launched ') === 0,
    'the tooltip still spells out the launch instant',
  );
  assert.strictEqual(
    tagged.parentElement,
    list[0].querySelector('.sidebar-item-actions'),
    'the tags live in the action strip like the age label',
  );
  for (let i = 1; i < 4; i++) {
    assert.strictEqual(
      list[i].querySelector('.sidebar-item-tags'),
      null,
      `row ${i}: no tags span without usable tags`,
    );
    assert.ok(list[i].querySelector('.sidebar-item-launched'));
  }
  win.close();
  console.log('  ok - task rows show their tags after the age label');
}

function testChatLastLaunchedLine() {
  const {win} = makeWebview();
  const now = Date.now();
  const stamped = now - 3 * HOUR_MS;
  loadHistory(win, [
    // Newest row first, as the daemon sends them; both rows of chat-a
    // carry the chat's stamp.
    session({
      id: 'chat-a',
      task_id: 'task-a2',
      startTs: now - 5 * HOUR_MS,
      timestamp: Math.floor((now - 5 * HOUR_MS) / 1000),
      chat_last_launched: stamped,
    }),
    session({
      id: 'chat-a',
      task_id: 'task-a1',
      startTs: now - 30 * HOUR_MS,
      timestamp: Math.floor((now - 30 * HOUR_MS) / 1000),
      chat_last_launched: stamped,
    }),
    // An older daemon: no stamp, so the newest row loaded decides.
    session({
      id: 'chat-b',
      task_id: 'task-b2',
      startTs: now - 26 * HOUR_MS,
      timestamp: Math.floor((now - 26 * HOUR_MS) / 1000),
    }),
    session({
      id: 'chat-b',
      task_id: 'task-b1',
      startTs: now - 50 * HOUR_MS,
      timestamp: Math.floor((now - 50 * HOUR_MS) / 1000),
    }),
    // Neither a stamp nor a usable row instant: the line stays empty.
    session({id: 'chat-c', task_id: 'task-c', startTs: 0, timestamp: null}),
    // Epoch zero is a real launch instant (imported databases carry it).
    session({id: 'chat-d', task_id: 'task-d', startTs: 0, timestamp: 0}),
  ]);
  const gs = groups(win);
  assert.strictEqual(gs.length, 4, 'one chat panel per chat');
  const byChat = {};
  gs.forEach(g => {
    byChat[g.dataset.chatId] = g;
  });
  const lineA = byChat['chat-a'].querySelector(
    ':scope > .history-chat-launched',
  );
  assert.ok(lineA, 'the chat panel has a launched line');
  assert.strictEqual(
    lineA.previousElementSibling,
    byChat['chat-a'].querySelector(':scope > .history-chat-header'),
    'the line sits directly under the collapsible header',
  );
  assert.strictEqual(
    lineA.nextElementSibling,
    byChat['chat-a'].querySelector(':scope > .history-chat-body'),
    'the line sits above the task rows',
  );
  const labelA = lineA.querySelector('.sidebar-item-launched');
  assert.strictEqual(labelA.textContent, 'last launched 3 hours ago');
  assert.ok(
    labelA.title.indexOf('Last launched ') === 0,
    'tooltip shows the absolute time',
  );
  assert.strictEqual(
    byChat['chat-b'].querySelector(
      '.history-chat-launched .sidebar-item-launched',
    ).textContent,
    'last launched 1 day ago',
    'without a stamp the newest row loaded decides',
  );
  assert.strictEqual(
    byChat['chat-c'].querySelector('.history-chat-launched').children.length,
    0,
    'no usable instant renders no label',
  );
  const labelD = byChat['chat-d'].querySelector(
    '.history-chat-launched .sidebar-item-launched',
  );
  assert.ok(labelD, 'an epoch-zero launch instant renders a label');
  assert.strictEqual(labelD.dataset.launchTs, '0');
  assert.ok(
    /^last launched \d+ years ago$/.test(labelD.textContent),
    labelD.textContent,
  );

  // The chat panel is folded: the line stays visible (it is not part
  // of the body the header folds).
  const header = byChat['chat-a'].querySelector('.history-chat-header');
  if (!byChat['chat-a'].classList.contains('collapsed')) header.click();
  assert.ok(byChat['chat-a'].classList.contains('collapsed'), 'chat folded');
  assert.strictEqual(
    win.getComputedStyle(byChat['chat-a'].querySelector('.history-chat-body'))
      .display,
    'none',
    'the folded body is hidden',
  );
  assert.notStrictEqual(
    win.getComputedStyle(lineA).display,
    'none',
    'the launched line stays visible when the chat is folded',
  );
  assert.strictEqual(
    win.getComputedStyle(
      byChat['chat-c'].querySelector('.history-chat-launched'),
    ).display,
    'none',
    'an empty line takes no room',
  );

  // A refresh with a newer stamp moves the line forward; a stale
  // page does not move it back.
  loadHistory(win, [
    session({
      id: 'chat-a',
      task_id: 'task-a3',
      startTs: now - HOUR_MS,
      timestamp: Math.floor((now - HOUR_MS) / 1000),
      chat_last_launched: now - HOUR_MS,
    }),
  ]);
  const lineA2 = groups(win)
    .find(g => g.dataset.chatId === 'chat-a')
    .querySelector('.history-chat-launched .sidebar-item-launched');
  assert.strictEqual(lineA2.textContent, 'last launched 1 hour ago');
  assert.strictEqual(lineA2.dataset.launchPrefix, 'last launched');
  win.close();
  console.log('  ok - chat panels show a "last launched ... ago" line');
}

function testRefreshSweepKeepsPrefix() {
  const {win, timers} = makeWebview();
  const now = Date.now();
  loadHistory(win, [
    session({
      id: 'chat-a',
      task_id: 'task-a',
      startTs: now - 2 * HOUR_MS,
      tags: 'work',
      chat_last_launched: now - 2 * HOUR_MS,
    }),
  ]);
  const labels = Array.from(
    win.document.querySelectorAll('#history-list .sidebar-item-launched'),
  );
  assert.strictEqual(labels.length, 2, 'the chat line and the task row');
  // Age both labels by five hours, then fire the 30 s sweep timer.
  labels.forEach(el => {
    el.dataset.launchTs = String(now - 7 * HOUR_MS);
  });
  const sweeps = timers.filter(t => t.delay === 30000);
  assert.ok(sweeps.length >= 1, 'the sweep timer is armed');
  sweeps[sweeps.length - 1].cb();
  const after = Array.from(
    win.document.querySelectorAll('#history-list .sidebar-item-launched'),
  ).map(el => el.textContent);
  assert.deepStrictEqual(
    after.sort(),
    ['7 hours ago', 'last launched 7 hours ago'],
    "the sweep keeps each label's own leading words (none for a task row)",
  );
  win.close();
  console.log("  ok - the 30 s sweep keeps the chat line's prefix");
}

async function testThoughtsPanelCostLive() {
  const wv = makeWebview();
  const win = wv.win;
  const TAB = tabIdOf(wv);
  const output = win.document.getElementById('output');
  const now = Date.now();
  send(win, {type: 'clear', chat_id: 'chat-cost', tabId: TAB});
  send(win, {type: 'status', running: true, tabId: TAB, startTs: now});
  send(win, {type: 'prompt', text: 'do the thing', tabId: TAB, ts: now});
  send(win, {type: 'thinking_start', tabId: TAB, ts: now});
  send(win, {type: 'thinking_delta', text: 'Plan: run ls.', tabId: TAB});
  send(win, {type: 'thinking_end', tabId: TAB});
  await flush();
  const thoughts = output.querySelector('.llm-panel');
  assert.ok(thoughts, 'Thoughts panel exists');
  assert.strictEqual(
    thoughts.querySelector('.panel-cost'),
    null,
    'no cost yet',
  );
  send(win, {
    type: 'llm_call',
    model: 'test-model',
    cost: 0.01234,
    input_tokens: 1000,
    output_tokens: 20,
    duration_ms: 1500,
    step: 1,
    tabId: TAB,
    ts: now + 1500,
  });
  send(win, {
    type: 'usage_info',
    text: 'Steps: 1/10',
    total_tokens: 1020,
    cost: '$0.0123',
    total_steps: 1,
    tabId: TAB,
  });
  await flush();
  const cost = thoughts.querySelector(':scope > .panel-time > .panel-cost');
  assert.ok(cost, 'the cost sits in the panel footer');
  assert.strictEqual(cost.textContent, '$0.0123');
  assert.strictEqual(cost.title, 'Cost of this model call (test-model)');
  const ts = thoughts.querySelector(':scope > .panel-time > .panel-ts');
  assert.ok(ts, 'the time label is there');
  assert.strictEqual(ts.nextElementSibling, cost, 'cost follows the time text');

  // The tool call seals the panel; the next call's cost lands in the
  // next thoughts panel, not this one.
  send(win, {
    type: 'tool_call',
    name: 'Bash',
    command: 'ls',
    tabId: TAB,
    ts: now + 2000,
  });
  send(win, {
    type: 'tool_result',
    content: 'ok',
    is_error: false,
    tabId: TAB,
    ts: now + 2500,
  });
  send(win, {type: 'text_delta', text: 'Done.', tabId: TAB, ts: now + 3000});
  send(win, {
    type: 'llm_call',
    model: 'test-model',
    cost: 0.5,
    tabId: TAB,
    ts: now + 3100,
  });
  await flush();
  const panels = output.querySelectorAll('.llm-panel');
  assert.strictEqual(panels.length, 2, 'two thoughts panels');
  assert.strictEqual(
    panels[0].querySelector('.panel-cost').textContent,
    '$0.0123',
  );
  assert.strictEqual(
    panels[1].querySelector('.panel-cost').textContent,
    '$0.5000',
  );

  // A cost with no open thoughts panel (a tool-only response after the
  // tool_call sealed the panel) is not rendered anywhere.
  send(win, {
    type: 'tool_call',
    name: 'Bash',
    command: 'pwd',
    tabId: TAB,
    ts: now + 4000,
  });
  send(win, {
    type: 'llm_call',
    model: 'test-model',
    cost: 0.25,
    tabId: TAB,
    ts: now + 4100,
  });
  await flush();
  assert.strictEqual(output.querySelectorAll('.panel-cost').length, 2);

  // A text-only reply gets a retry call whose words land in the same
  // panel: the panel shows the SUM of its calls, never the last one.
  send(win, {
    type: 'tool_result',
    content: '/',
    is_error: false,
    tabId: TAB,
    ts: now + 5000,
  });
  send(win, {
    type: 'text_delta',
    text: 'I am done.',
    tabId: TAB,
    ts: now + 5100,
  });
  send(win, {type: 'text_end', tabId: TAB});
  send(win, {
    type: 'llm_call',
    model: 'call-one',
    cost: 0.25,
    step: 3,
    tabId: TAB,
  });
  await flush();
  const third = output.querySelectorAll('.llm-panel')[2];
  assert.strictEqual(third.querySelector('.panel-cost').textContent, '$0.2500');
  send(win, {
    type: 'text_delta',
    text: 'Calling finish now.',
    tabId: TAB,
    ts: now + 6000,
  });
  send(win, {
    type: 'llm_call',
    model: 'call-two',
    cost: 0.01,
    step: 4,
    tabId: TAB,
  });
  send(win, {
    type: 'llm_call',
    model: 'call-two',
    cost: 'garbage',
    step: 4,
    tabId: TAB,
  });
  await flush();
  const summed = third.querySelector('.panel-cost');
  assert.strictEqual(summed.textContent, '$0.2600', 'both calls are summed');
  assert.strictEqual(
    summed.title,
    'Cost of the 2 model calls in this panel (call-two)',
  );
  assert.strictEqual(third.querySelectorAll('.panel-cost').length, 1);
  win.close();
  console.log('  ok - live llm_call cost is shown under the thoughts panel');
}

async function testThoughtsPanelCostReplay() {
  const wv = makeWebview();
  const win = wv.win;
  const TAB = tabIdOf(wv);
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
      {type: 'llm_call', model: 'm', cost: 0.00004, ts: oldTs + 100},
      {type: 'tool_call', name: 'Bash', command: 'ls', ts: oldTs + 200},
      {type: 'tool_result', content: 'ok', is_error: false, ts: oldTs + 2500},
      {type: 'text_delta', text: 'final words', ts: oldTs + 3000},
      {type: 'llm_call', model: 'm', cost: 0, ts: oldTs + 3100},
      {type: 'result', text: 'done', ts: oldTs + 3200},
    ],
  });
  await flush();
  const costs = Array.from(
    output.querySelectorAll('.llm-panel .panel-cost'),
  ).map(el => el.textContent);
  assert.deepStrictEqual(costs, ['<$0.0001', '$0'], 'replayed costs render');
  win.close();
  console.log('  ok - replayed llm_call costs render under their panels');
}

function testFormatCallCost() {
  const {win} = makeWebview();
  const fmt = win.PanelCopy.formatCallCost;
  assert.strictEqual(fmt(0.5), '$0.5000');
  assert.strictEqual(fmt(0), '$0');
  assert.strictEqual(fmt(0.00001), '<$0.0001');
  assert.strictEqual(fmt(-1), '');
  assert.strictEqual(fmt('abc'), '');
  assert.strictEqual(fmt(undefined), '');
  const doc = win.document;
  const panel = doc.createElement('div');
  assert.strictEqual(win.PanelCopy.setPanelCost(null, 1), null);
  assert.strictEqual(win.PanelCopy.setPanelCost(panel, NaN), null);
  // No time label yet: the cost leads the footer; a later timestamp
  // still goes first (addPanelTimestamp inserts at the front).
  const span = win.PanelCopy.setPanelCost(panel, 0.2);
  assert.strictEqual(span.parentElement.className, 'panel-time');
  assert.strictEqual(span.title, 'Cost of this model call');
  win.PanelCopy.addPanelTimestamp(panel, Date.now());
  assert.strictEqual(panel.querySelector('.panel-time').children[1], span);
  // Idempotent: a second cost replaces the text in place.
  const again = win.PanelCopy.setPanelCost(panel, 0.3, 'm');
  assert.strictEqual(again, span);
  assert.strictEqual(span.textContent, '$0.3000');
  assert.strictEqual(panel.querySelectorAll('.panel-cost').length, 1);
  // A footer whose time label is the last child: the cost is appended.
  const other = doc.createElement('div');
  win.PanelCopy.addPanelTimestamp(other, Date.now());
  const s2 = win.PanelCopy.setPanelCost(other, 1);
  assert.strictEqual(s2.previousElementSibling.className, 'panel-ts');
  assert.strictEqual(s2.nextElementSibling, null);
  win.close();
  console.log('  ok - formatCallCost / setPanelCost edge cases');
}

(async () => {
  testTagsAfterLaunched();
  testChatLastLaunchedLine();
  testRefreshSweepKeepsPrefix();
  await testThoughtsPanelCostLive();
  await testThoughtsPanelCostReplay();
  testFormatCallCost();
  console.log('All historyTagsChatLaunchedPanelCost tests passed.');
})().catch(e => {
  console.error(e);
  process.exit(1);
});
