// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) parity test for the streaming state machine.
//
// The invariant: a tab renders the SAME transcript whether or not it was
// on screen while its task ran, and the transcript it comes back to
// after a reload says the same thing again. The webview used to carry
// three hand-maintained copies of the "tool_call resets the panel /
// tool_result arms the next one / the first thinking_start|text_delta
// after that opens a thoughts panel and counts a step" logic -- one for
// the visible tab, one for background tabs, one for replay -- plus a
// fourth copy that only counted steps. They had already drifted:
//
//   * the background copy intercepted `usage_info` before the renderer
//     and re-implemented only its numeric branch, so the "Context: …
//     Budget: … Steps: …" text form was silently ignored;
//   * the background copy never collapsed older panels, so a tab that ran
//     forty steps while hidden came back fully expanded;
//   * the replay step counter seeded its first step with a different rule
//     from the live one.
//
// This test feeds ONE recorded run to a visible tab, to a hidden tab and
// to a replay, and demands the three transcripts agree.
//
// The live stream and the replayed transcript are NOT the same list of
// events, and the difference is the daemon's, not this test's: only
// display events are recorded and persisted (`usage_info` among them,
// so a reload can repopulate the tokens/cost header of a task that has
// no `result` event yet). Rather than restate that rule here -- or read
// it out of the daemon's source, which would fail for a harmless
// refactor and pass for a real change of behaviour --
// `persistedTranscript()` below runs the run through the daemon's OWN
// recorder and replays whatever comes back.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {spawnSync} = require('child_process');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');
const REPO_ROOT = path.join(__dirname, '..', '..', '..', '..', '..');

// `JsonPrinter._filter_and_coalesce` is the function `stop_recording()`
// and `peek_recording()` run every event list through on its way to the
// database, so its answer IS what a reload, a history click or a
// background tab's `task_events` gets back.
const RECORDER_PROBE = [
  'import json, sys',
  'from kiss.server.json_printer import JsonPrinter',
  'json.dump(JsonPrinter._filter_and_coalesce(json.load(sys.stdin)), sys.stdout)',
].join('\n');

/**
 * Record a run the way the daemon does and hand back what survived.
 *
 * @param {Array<object>} events The events the agent broadcast.
 * @returns {Array<object>} The transcript the daemon would store.
 */
function recordAsDaemon(events) {
  const res = spawnSync('uv', ['run', 'python', '-c', RECORDER_PROBE], {
    cwd: REPO_ROOT,
    input: JSON.stringify(events),
    encoding: 'utf8',
  });
  assert.ok(
    !res.error && res.status === 0,
    'the daemon\u2019s recorder must be runnable from the repo root: ' +
      `${res.error || ''} ${res.stderr || ''}`,
  );
  return JSON.parse(res.stdout);
}

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

  const posted = [];
  let state;
  win.acquireVsCodeApi = function () {
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
  win._testApi.endLaunch();
  win._testApi.hideWelcome();
  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function clickTab(win, tabId) {
  const el = win.document.querySelector(
    `.chat-tab[data-tab-id=${JSON.stringify(tabId)}]`,
  );
  assert.ok(el, `tab ${tabId} must exist in the tab bar`);
  el.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

const TS = 1767225600000;

// A realistic run: think, speak, three tool calls, the per-step
// `usage_info`, and the result. The two carry the daemon's own numbers
// exactly as `json_printer` broadcasts them -- `usage_info` with
// `total_tokens` / `cost` / `total_steps` beside its text line, and
// `result` with the `step_count` `_broadcast_result` always attaches.
// Three tool calls do not mean three steps: the agent was seven steps in
// by the time it finished, and only the daemon knows that.
function recordedRun() {
  return [
    {type: 'thinking_start', ts: TS},
    {type: 'thinking_delta', text: 'let me look', ts: TS},
    {type: 'thinking_end', ts: TS},
    {type: 'text_delta', text: 'Reading the file.', ts: TS},
    {type: 'text_end', text: 'Reading the file.', ts: TS},
    {type: 'tool_call', name: 'Read', path: 'src/one.py', ts: TS},
    {type: 'tool_result', content: 'one', ts: TS},
    {type: 'thinking_start', ts: TS},
    {type: 'thinking_delta', text: 'now the second', ts: TS},
    {type: 'thinking_end', ts: TS},
    {type: 'tool_call', name: 'Read', path: 'src/two.py', ts: TS},
    {type: 'tool_result', content: 'two', ts: TS},
    {type: 'text_delta', text: 'Both read.', ts: TS},
    {type: 'text_end', text: 'Both read.', ts: TS},
    {type: 'tool_call', name: 'Write', path: 'src/three.py', ts: TS},
    {type: 'tool_result', content: 'written', ts: TS},
    {
      type: 'usage_info',
      text: 'Steps: 7/100, Context: 12,345/500,000 tokens, Budget: $0.42/$10.00',
      total_tokens: 12345,
      cost: '$0.42',
      total_steps: 7,
      ts: TS,
    },
    {
      type: 'result',
      text: 'summary: Done.\nsuccess: true\n',
      summary: 'Done.',
      success: true,
      total_tokens: 12345,
      cost: '$0.42',
      step_count: 7,
      ts: TS,
    },
  ];
}

// What is left of that run once the daemon has recorded it: what a
// reload, a history click or a background tab's `task_events` gets back.
function persistedTranscript() {
  const events = recordAsDaemon(recordedRun());
  assert.ok(
    events.some(ev => ev.type === 'result'),
    'the daemon must keep the result event, or there is nothing to replay',
  );
  assert.ok(
    events.some(ev => ev.type === 'usage_info'),
    'the daemon keeps usage_info when storing a run: it is the only ' +
      'carrier of the tokens/cost header for a task that stopped, ' +
      'errored, or is still running (no result event yet)',
  );
  return events;
}

function runInto(win, tabId, events) {
  send(win, {type: 'status', running: true, tabId, startTs: TS});
  for (const ev of events || recordedRun()) send(win, {...ev, tabId});
}

// What the user actually sees: the shape of the transcript, which panels
// are collapsed, and the three header numbers.
function snapshot(win) {
  const out = win.document.getElementById('output');
  const panels = Array.from(out.children).map(el => ({
    cls: Array.from(el.classList).sort().join(' '),
    tool: el.querySelector('.tc-h')
      ? el.querySelector('.tc-h').textContent
      : '',
  }));
  const doc = win.document;
  return {
    panels,
    llmPanels: out.querySelectorAll('.llm-panel').length,
    toolPanels: out.querySelectorAll('.ev.tc').length,
    steps: doc.getElementById('status-steps').textContent,
    tokens: doc.getElementById('status-tokens').textContent,
    budget: doc.getElementById('status-budget').textContent,
  };
}

function testHiddenTabRendersLikeAVisibleOne() {
  const {win} = makeWebview();
  const tabA = win._testApi.getActiveTabId();
  win._testApi.createNewTab();
  const tabB = win._testApi.getActiveTabId();
  clickTab(win, tabA);
  assert.strictEqual(win._testApi.getActiveTabId(), tabA);

  // The same run, once on screen and once hidden.
  runInto(win, tabA);
  const visible = snapshot(win);

  runInto(win, tabB);
  assert.strictEqual(
    win._testApi.getActiveTabId(),
    tabA,
    'a background run must not steal the screen',
  );
  clickTab(win, tabB);
  const hidden = snapshot(win);

  assert.ok(visible.toolPanels >= 3, 'the run must render its tool calls');
  assert.ok(visible.llmPanels >= 3, 'the run must render its thoughts panels');

  assert.strictEqual(
    hidden.tokens,
    visible.tokens,
    'a usage_info must move the token counter of a hidden tab exactly ' +
      'as it moves a visible one',
  );
  assert.strictEqual(
    hidden.budget,
    visible.budget,
    'the same holds for the cost counter',
  );
  assert.strictEqual(
    hidden.steps,
    visible.steps,
    'the same holds for the step counter',
  );
  assert.deepStrictEqual(
    hidden.panels.map(p => p.cls + '|' + p.tool),
    visible.panels.map(p => p.cls + '|' + p.tool),
    'a tab that ran while hidden must come back with the same panels, ' +
      'in the same order, collapsed the same way, as one that ran on ' +
      'screen',
  );
  win.close();
  console.log('  ok - a hidden tab renders like a visible one');
}

// main.js reads the usage report two ways: the numeric fields the daemon
// sends today, and the "Steps: … Context: … Budget: …" text beside them,
// for a payload that carries no numbers. Both transcripts must read both
// forms the same way -- the background copy used to intercept usage_info
// and re-implement only the numeric one.
//
// The run stops before the result on purpose: the result carries the
// same three numbers a second time, and would hide a usage_info that
// never landed.
function testBothUsageFormsAreReadTheSameWayByBothTranscripts() {
  const withoutResult = recordedRun().filter(ev => ev.type !== 'result');
  const forms = {
    numeric: withoutResult.map(ev =>
      ev.type === 'usage_info' ? {...ev, text: ''} : ev,
    ),
    text: withoutResult.map(ev =>
      ev.type === 'usage_info'
        ? {type: 'usage_info', text: ev.text, ts: ev.ts}
        : ev,
    ),
  };

  for (const [form, events] of Object.entries(forms)) {
    const {win} = makeWebview();
    const tabA = win._testApi.getActiveTabId();
    win._testApi.createNewTab();
    const tabB = win._testApi.getActiveTabId();
    clickTab(win, tabA);

    runInto(win, tabA, events);
    const visible = snapshot(win);
    runInto(win, tabB, events);
    clickTab(win, tabB);
    const hidden = snapshot(win);

    assert.strictEqual(
      visible.steps,
      'Steps: 7',
      `the ${form} form of the usage report carries the daemon's step ` +
        'count and the header must show it',
    );
    assert.strictEqual(visible.tokens, 'Tokens: 12,345', form);
    assert.strictEqual(visible.budget, 'Cost: $0.42', form);
    assert.strictEqual(hidden.steps, visible.steps, form);
    assert.strictEqual(hidden.tokens, visible.tokens, form);
    assert.strictEqual(hidden.budget, visible.budget, form);
    win.close();
  }
  console.log('  ok - both transcripts read both usage forms the same way');
}

// The counters shown for a replayed transcript must match the counters
// the live stream produced for the very same run -- the stored
// usage_info and the result's step_count both survive recording, and
// either alone must be enough to rebuild the header.
function testReplayAgreesWithTheLiveStream() {
  const {win} = makeWebview();
  const tabA = win._testApi.getActiveTabId();
  runInto(win, tabA);
  const live = snapshot(win);

  assert.strictEqual(
    live.steps,
    'Steps: 7',
    'the live stream shows the daemon\u2019s count, not the number of ' +
      'panels it happened to draw',
  );

  const livePanels = Array.from(win.document.getElementById('output').children);
  assert.ok(livePanels.length > 0, 'the live stream must have drawn panels');

  send(win, {type: 'status', running: false, tabId: tabA});
  send(win, {
    type: 'task_events',
    tabId: tabA,
    events: persistedTranscript(),
  });
  const replayed = snapshot(win);

  // Without this the test would be vacuous: an unhandled task_events
  // would leave the live transcript on screen and every assertion
  // below would be comparing the live stream with itself.
  assert.ok(
    livePanels.every(el => !win.document.contains(el)),
    'the replay must rebuild the transcript rather than leave the live ' +
      'one standing',
  );

  assert.strictEqual(
    replayed.steps,
    live.steps,
    'replaying a transcript must show the same step count the live ' +
      'stream showed',
  );
  assert.strictEqual(
    replayed.tokens,
    live.tokens,
    'and the same token count, which survives on the result event',
  );
  assert.strictEqual(replayed.budget, live.budget, 'and the same cost');
  assert.strictEqual(
    replayed.toolPanels,
    live.toolPanels,
    'replaying a transcript must render the same tool panels',
  );
  assert.strictEqual(
    replayed.llmPanels,
    live.llmPanels,
    'replaying a transcript must render the same thoughts panels',
  );
  win.close();
  console.log('  ok - a replayed transcript agrees with the live stream');
}

// A run_parallel fan-out: the daemon's live usage monitor adds every
// sub-agent's steps to the parent's, so its count runs far ahead of the
// number of panels this transcript has drawn. Counting the next panel
// must carry on from the daemon's number, not from the renderer's own
// estimate -- otherwise the header jumps back from 43 to 2.
function testTheStepCounterNeverRunsBackwards() {
  const {win} = makeWebview();
  const tabA = win._testApi.getActiveTabId();
  win._testApi.createNewTab();
  const tabB = win._testApi.getActiveTabId();
  clickTab(win, tabA);

  const fanOut = [
    {type: 'thinking_start', ts: TS},
    {type: 'thinking_delta', text: 'fan out', ts: TS},
    {type: 'thinking_end', ts: TS},
    {type: 'tool_call', name: 'run_parallel', ts: TS},
    {
      type: 'usage_info',
      text: 'Tokens: 900,000, Budget: $4.0000 (live, incl. parallel sub-agents), ',
      total_tokens: 900000,
      cost: '$4.0000',
      total_steps: 43,
      ts: TS,
    },
    {type: 'tool_result', content: 'all four done', ts: TS},
    {type: 'thinking_start', ts: TS},
    {type: 'thinking_delta', text: 'now I write it up', ts: TS},
    {type: 'thinking_end', ts: TS},
  ];

  send(win, {type: 'status', running: true, tabId: tabA, startTs: TS});
  for (const ev of fanOut) {
    send(win, {...ev, tabId: tabA});
    if (ev.type === 'usage_info') {
      assert.strictEqual(
        snapshot(win).steps,
        'Steps: 43',
        'a usage_info puts the daemon\u2019s count up as it arrives',
      );
    }
  }
  const visible = snapshot(win);
  assert.strictEqual(
    visible.steps,
    'Steps: 44',
    'the thought after a run_parallel is step 44 of 43 already spent, ' +
      'not step 2 of the panels this tab drew',
  );

  runInto(win, tabB, fanOut);
  clickTab(win, tabB);
  assert.strictEqual(
    snapshot(win).steps,
    visible.steps,
    'a tab that fanned out while hidden must come back with the same ' +
      'step count',
  );
  win.close();
  console.log('  ok - the step counter never runs backwards');
}

// The remote web app's reconnect path: the daemon replays every open
// tab's transcript as a `task_events` envelope, and only one of those
// tabs is on screen. A hidden tab's replay borrows the visible status
// row while it renders and gives it back afterwards -- but the numbers
// the replayed usage_info / result painted belong to the hidden tab and
// must come back with it. This used to keep only the step count, so a
// reloaded web app showed "· Steps: 22" with no tokens and no cost
// until the task's next live step.
function testBackgroundReplayKeepsTokensAndCost() {
  const {win} = makeWebview();
  const tabA = win._testApi.getActiveTabId();
  win._testApi.createNewTab();
  const tabB = win._testApi.getActiveTabId();
  clickTab(win, tabA);

  const before = snapshot(win);
  send(win, {
    type: 'task_events',
    tabId: tabB,
    task: 'benchmark research',
    events: persistedTranscript(),
  });
  const after = snapshot(win);
  assert.strictEqual(
    after.tokens,
    before.tokens,
    'a hidden tab\u2019s replay must not leave its numbers on the ' +
      'visible tab\u2019s status row',
  );
  assert.strictEqual(after.budget, before.budget, 'nor its cost');
  assert.strictEqual(after.steps, before.steps, 'nor its step count');

  clickTab(win, tabB);
  const replayed = snapshot(win);
  assert.strictEqual(
    replayed.tokens,
    'Tokens: 12,345',
    'switching to a tab whose transcript was replayed while hidden ' +
      'must show the tokens the replay carried',
  );
  assert.strictEqual(
    replayed.budget,
    'Cost: $0.42',
    'and the cost -- not a bare "· Steps: N" header',
  );
  assert.strictEqual(replayed.steps, 'Steps: 7', 'and the step count');
  win.close();
  console.log('  ok - a background replay keeps its tokens and cost');
}

// A hidden tab's replay borrows the visible status row while it
// renders. A replayed transcript that carries no usage event (a run
// recorded before usage_info was persisted, or a task that has not
// spent anything yet) must NOT walk off with the numbers the visible
// tab left on that row -- the hidden tab keeps its own previous
// numbers, exactly as staticTaskPanelVisibleTask.test.js pins for a
// same-task replay.
function testMetricFreeReplayDoesNotStealTheVisibleTabsNumbers() {
  const {win} = makeWebview();
  const tabA = win._testApi.getActiveTabId();
  win._testApi.createNewTab();
  const tabB = win._testApi.getActiveTabId();
  clickTab(win, tabA);

  // A runs live with numbers of its own, distinct from B's.
  const aRun = recordedRun().map(ev => {
    if (ev.type === 'usage_info')
      return {...ev, text: '', total_tokens: 999, cost: '$9.99', total_steps: 9};
    if (ev.type === 'result')
      return {...ev, total_tokens: 999, cost: '$9.99', step_count: 9};
    return ev;
  });
  runInto(win, tabA, aRun);
  const visibleBefore = snapshot(win);
  assert.strictEqual(visibleBefore.tokens, 'Tokens: 999');

  // B replays a transcript WITH numbers, then a metric-free one.
  send(win, {type: 'task_events', tabId: tabB, events: persistedTranscript()});
  send(win, {
    type: 'task_events',
    tabId: tabB,
    events: [{type: 'prompt', text: 'brand new task', ts: TS}],
  });

  const visibleAfter = snapshot(win);
  assert.strictEqual(
    visibleAfter.tokens,
    visibleBefore.tokens,
    'the hidden replays must leave the visible tab\u2019s numbers alone',
  );
  assert.strictEqual(visibleAfter.budget, visibleBefore.budget);
  assert.strictEqual(visibleAfter.steps, visibleBefore.steps);

  clickTab(win, tabB);
  const hidden = snapshot(win);
  assert.strictEqual(
    hidden.tokens,
    'Tokens: 12,345',
    'a metric-free replay must not inherit the visible tab\u2019s ' +
      'token count -- the tab keeps its own',
  );
  assert.strictEqual(hidden.budget, 'Cost: $0.42', 'nor its cost');
  assert.strictEqual(hidden.steps, 'Steps: 7', 'nor its step count');
  win.close();
  console.log(
    '  ok - a metric-free replay does not steal the visible numbers',
  );
}

// The replay's collapse pass can take the screen away mid-flight: a
// finished run_parallel panel closes its sub-agent tabs, and if the tab
// on screen is one of them, the switch that follows repaints the shared
// status row. The numbers the replay painted must already be on the
// parent tab by then -- captured after rendering, before collapsing.
function testHiddenReplayThatClosesTheVisibleTabKeepsItsNumbers() {
  const {win, posted} = makeWebview();
  const tabA = win._testApi.getActiveTabId();

  send(win, {type: 'status', running: true, tabId: tabA, startTs: TS});
  send(win, {
    type: 'tool_call',
    name: 'run_parallel',
    tabId: tabA,
    extras: {tasks: JSON.stringify(['sub 1'])},
  });
  const before = posted.length;
  send(win, {
    type: 'new_tab',
    task_id: 'sub-task-1',
    parent_tab_id: tabA,
    taskId: '',
  });
  const resume = posted
    .slice(before)
    .find(m => m.type === 'resumeSession' && m.taskId === 'sub-task-1');
  assert.ok(resume, 'new_tab must make the webview post resumeSession');
  send(win, {
    type: 'openSubagentTab',
    tab_id: resume.tabId,
    parent_tab_id: tabA,
    description: 'sub 1',
    task_id: 'sub-task-1',
    taskIndex: 0,
  });
  clickTab(win, resume.tabId);
  assert.strictEqual(win._testApi.getActiveTabId(), resume.tabId);

  // The parent's finished transcript replays while the sub-agent tab
  // is on screen; its collapse pass closes that tab.
  const parentRun = recordAsDaemon([
    {type: 'thinking_start', ts: TS},
    {type: 'thinking_delta', text: 'fanning out', ts: TS},
    {type: 'thinking_end', ts: TS},
    {
      type: 'tool_call',
      name: 'run_parallel',
      extras: {tasks: JSON.stringify(['sub 1'])},
      ts: TS,
    },
    {type: 'tool_result', content: 'sub done', ts: TS},
    {
      type: 'usage_info',
      text: 'Steps: 7/100, Context: 12,345/500,000 tokens, Budget: $0.42/$10.00',
      total_tokens: 12345,
      cost: '$0.42',
      total_steps: 7,
      ts: TS,
    },
    {
      type: 'result',
      text: 'summary: Done.\nsuccess: true\n',
      summary: 'Done.',
      success: true,
      total_tokens: 12345,
      cost: '$0.42',
      step_count: 7,
      ts: TS,
    },
  ]);
  send(win, {type: 'task_events', tabId: tabA, events: parentRun});

  if (win._testApi.getActiveTabId() !== tabA) clickTab(win, tabA);
  const shown = snapshot(win);
  assert.strictEqual(
    shown.tokens,
    'Tokens: 12,345',
    'a hidden replay that closed the tab on screen must still hand ' +
      'its tokens to the parent tab',
  );
  assert.strictEqual(shown.budget, 'Cost: $0.42', 'and its cost');
  assert.strictEqual(shown.steps, 'Steps: 7', 'and its step count');
  win.close();
  console.log(
    '  ok - a hidden replay that closes the visible tab keeps its numbers',
  );
}

function main() {
  testHiddenTabRendersLikeAVisibleOne();
  testBothUsageFormsAreReadTheSameWayByBothTranscripts();
  testReplayAgreesWithTheLiveStream();
  testTheStepCounterNeverRunsBackwards();
  testBackgroundReplayKeepsTokensAndCost();
  testMetricFreeReplayDoesNotStealTheVisibleTabsNumbers();
  testHiddenReplayThatClosesTheVisibleTabKeepsItsNumbers();
  console.log('bgTabStreamParity.test.js: all tests passed');
}

main();
