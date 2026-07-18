// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end tests: the chat webview's handling of the agent's no-op
// ``summary`` tool.  When a ``summary`` tool_call event renders, the
// webview must (a) move the previous (up to) 6 top-level event panels
// into the summary panel as sub-panels, (b) collapse the summary
// panel, and (c) keep the ``description`` argument FULLY visible
// while the panel is collapsed (no ellipsis truncation).
//
// The tests exercise the real ``media/main.js`` against the real
// ``media/chat.html`` in jsdom (same harness as
// ``bashHeaderCyan.test.js``), covering the live streaming path, the
// history replay path (``task_events``), boundary conditions
// (.prompt / .system-prompt / .adjacent-task / fewer than 6 panels),
// chained summaries, the collapse toggle, the tool_result routing,
// and the CSS visibility contract.
//
// Run directly with ``node``:
//
//     node src/kiss/agents/vscode/test/summaryToolCollapse.test.js

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

const DESC =
  'The agent explored the repository layout and read the main entry ' +
  'points. It located the tool-registration code in the agent module. ' +
  'It then ran the existing test suite to establish a green baseline. ' +
  'Next it drafted the new feature behind a flag. It wired the flag ' +
  'into the CLI. Finally it re-ran the impacted tests and they passed.';

/**
 * Build a jsdom window running the real chat webview (chat.html +
 * panelCopy.js + main.js).
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
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
  return {win, posted};
}

/** Dispatch a backend→webview event exactly like the extension does. */
function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

/** Inject the real production stylesheet so computed styles resolve. */
function injectCss(win) {
  const css = fs.readFileSync(path.join(MEDIA, 'main.css'), 'utf8');
  const styleEl = win.document.createElement('style');
  styleEl.textContent = css;
  win.document.head.appendChild(styleEl);
}

/** Emit *n* distinct tool_call panels (Read path-arg panels). */
function sendToolPanels(win, n, offset) {
  const start = offset || 0;
  for (let i = start; i < start + n; i++) {
    send(win, {type: 'tool_call', name: 'Read', path: '/tmp/f' + i + '.txt'});
    send(win, {type: 'tool_result', name: 'Read', content: 'data ' + i});
  }
}

function output(win) {
  return win.document.getElementById('output');
}

function summaryPanels(win) {
  return Array.from(output(win).querySelectorAll('.tc.tc-summary'));
}

/** Direct #output children (the top-level event panels, sans #welcome). */
function topLevel(win) {
  return Array.from(output(win).children).filter(el => el.id !== 'welcome');
}

// ── tests ──────────────────────────────────────────────────────────

function testSummaryPanelCreatedAndCollapsed() {
  const {win} = makeWebview();
  send(win, {type: 'prompt', text: 'do the thing'});
  sendToolPanels(win, 2);
  send(win, {type: 'tool_call', name: 'summary', description: DESC});
  const panels = summaryPanels(win);
  assert.strictEqual(panels.length, 1, 'one summary panel must render');
  const p = panels[0];
  assert.ok(
    p.classList.contains('collapsed'),
    'summary panel must auto-collapse when it renders',
  );
  const hdr = p.querySelector('.tc-h');
  assert.ok(hdr, 'summary panel must have a .tc-h header');
  assert.ok(
    (hdr.textContent || '').includes('summary'),
    'header must name the summary tool',
  );
  win.close();
  console.log('  ok - summary tool_call renders a collapsed .tc-summary');
}

function testDescriptionIsDirectChildWithFullText() {
  const {win} = makeWebview();
  send(win, {type: 'prompt', text: 'go'});
  sendToolPanels(win, 1);
  send(win, {type: 'tool_call', name: 'summary', description: DESC});
  const p = summaryPanels(win)[0];
  const desc = p.querySelector(':scope > .tc-summary-desc');
  assert.ok(
    desc,
    'description must render in a .tc-summary-desc DIRECT child of the ' +
      'panel (so the collapsed-state CSS can keep it visible)',
  );
  assert.strictEqual(
    desc.textContent,
    DESC,
    'the FULL description text must be present, untruncated',
  );
  win.close();
  console.log('  ok - full description rendered in .tc-summary-desc');
}

function testNestsExactlySixMostRecentPanelsInOrder() {
  const {win} = makeWebview();
  send(win, {type: 'prompt', text: 'go'});
  // 7 tool panels + 1 Thoughts (llm) panel = 8 top-level event panels.
  sendToolPanels(win, 7);
  send(win, {type: 'thinking_start'});
  send(win, {type: 'thinking_delta', text: 'pondering...'});
  send(win, {type: 'thinking_end'});
  const before = topLevel(win);
  const expectNested = before.slice(-6);
  send(win, {type: 'tool_call', name: 'summary', description: DESC});
  const p = summaryPanels(win)[0];
  const sub = p.querySelector(':scope > .summary-sub');
  assert.ok(sub, 'summary panel must have a .summary-sub container');
  const nested = Array.from(sub.children);
  assert.strictEqual(
    nested.length,
    6,
    'exactly the last 6 event panels must nest — got ' + nested.length,
  );
  for (let i = 0; i < 6; i++) {
    assert.strictEqual(
      nested[i],
      expectNested[i],
      'nested panel ' + i + ' must be the original panel, in order',
    );
  }
  assert.ok(
    nested.some(el => el.classList.contains('llm-panel')),
    'Thoughts (llm-panel) panels count as event panels and must nest',
  );
  // The older panels (prompt + first 2 tool panels) stay top-level.
  const after = topLevel(win);
  assert.strictEqual(
    after.length,
    4,
    'top level must be: prompt + 2 old panels + summary panel',
  );
  assert.ok(after[0].classList.contains('prompt'), 'prompt stays first');
  assert.strictEqual(after[3], p, 'summary panel is the last child');
  win.close();
  console.log('  ok - exactly the last 6 panels nest, original order kept');
}

function testFewerThanSixStopsAtPromptBoundary() {
  const {win} = makeWebview();
  send(win, {type: 'prompt', text: 'go'});
  sendToolPanels(win, 3);
  send(win, {type: 'tool_call', name: 'summary', description: DESC});
  const p = summaryPanels(win)[0];
  const nested = Array.from(p.querySelector(':scope > .summary-sub').children);
  assert.strictEqual(nested.length, 3, 'only the 3 available panels nest');
  const after = topLevel(win);
  assert.strictEqual(after.length, 2, 'prompt + summary panel remain');
  assert.ok(
    after[0].classList.contains('prompt'),
    'the user prompt panel must NEVER be swallowed by a summary',
  );
  win.close();
  console.log('  ok - nesting stops at the .prompt boundary');
}

function testAdjacentTaskAndSystemPromptAreBoundaries() {
  const {win} = makeWebview();
  const O = output(win);
  const adj = win.document.createElement('div');
  adj.className = 'adjacent-task';
  O.appendChild(adj);
  const sys = win.document.createElement('div');
  sys.className = 'ev system-prompt';
  O.appendChild(sys);
  sendToolPanels(win, 2);
  send(win, {type: 'tool_call', name: 'summary', description: DESC});
  const p = summaryPanels(win)[0];
  const nested = Array.from(p.querySelector(':scope > .summary-sub').children);
  assert.strictEqual(
    nested.length,
    2,
    'nesting must stop before .system-prompt / .adjacent-task blocks',
  );
  assert.ok(sys.parentElement === O && adj.parentElement === O);
  win.close();
  console.log('  ok - .adjacent-task and .system-prompt are boundaries');
}

function testWelcomeBlockNeverAdopted() {
  // No prompt event at all: the adopt walk reaches past the tool
  // panels to the #welcome block, which is NOT an event panel and
  // must act as a hard boundary.
  const {win} = makeWebview();
  sendToolPanels(win, 2);
  send(win, {type: 'tool_call', name: 'summary', description: DESC});
  const p = summaryPanels(win)[0];
  const nested = Array.from(p.querySelector(':scope > .summary-sub').children);
  assert.strictEqual(nested.length, 2, 'only the 2 tool panels nest');
  const welcome = win.document.getElementById('welcome');
  assert.strictEqual(
    welcome.parentElement,
    output(win),
    'the #welcome block must never be swallowed by a summary panel',
  );
  win.close();
  console.log('  ok - non-panel elements (#welcome) are a hard boundary');
}

function testNoPrecedingPanels() {
  const {win} = makeWebview();
  send(win, {type: 'prompt', text: 'go'});
  send(win, {type: 'tool_call', name: 'summary', description: DESC});
  const p = summaryPanels(win)[0];
  const sub = p.querySelector(':scope > .summary-sub');
  const count = sub ? sub.children.length : 0;
  assert.strictEqual(count, 0, 'nothing to nest right after the prompt');
  assert.ok(p.classList.contains('collapsed'), 'still collapses');
  win.close();
  console.log('  ok - summary right after prompt nests nothing');
}

function testEmptyDescription() {
  const {win} = makeWebview();
  sendToolPanels(win, 1);
  send(win, {type: 'tool_call', name: 'summary'});
  const p = summaryPanels(win)[0];
  const desc = p.querySelector(':scope > .tc-summary-desc');
  assert.ok(desc, 'desc element renders even without a description');
  assert.strictEqual(desc.textContent, '');
  win.close();
  console.log('  ok - missing description handled');
}

function testDescriptionFullyVisibleWhileCollapsedViaCss() {
  const {win} = makeWebview();
  injectCss(win);
  send(win, {type: 'prompt', text: 'go'});
  sendToolPanels(win, 6);
  send(win, {type: 'tool_call', name: 'summary', description: DESC});
  const p = summaryPanels(win)[0];
  assert.ok(p.classList.contains('collapsed'));
  const desc = p.querySelector(':scope > .tc-summary-desc');
  const st = win.getComputedStyle(desc);
  assert.notStrictEqual(
    st.getPropertyValue('display').trim(),
    'none',
    'description must stay VISIBLE while the summary panel is collapsed',
  );
  // Fully visible = no single-line ellipsis truncation.
  assert.notStrictEqual(
    st.getPropertyValue('white-space').trim(),
    'nowrap',
    'description must wrap (not be clipped to one line)',
  );
  assert.notStrictEqual(
    st.getPropertyValue('text-overflow').trim(),
    'ellipsis',
    'description must not be ellipsised',
  );
  // The nested panels, by contrast, must be hidden while collapsed.
  const sub = p.querySelector(':scope > .summary-sub');
  assert.strictEqual(
    win.getComputedStyle(sub).getPropertyValue('display').trim(),
    'none',
    'nested sub-panels must hide while the summary panel is collapsed',
  );
  win.close();
  console.log('  ok - collapsed summary keeps description visible (CSS)');
}

function testCollapsePreviewSuppressedForSummary() {
  const {win} = makeWebview();
  sendToolPanels(win, 6);
  send(win, {type: 'tool_call', name: 'summary', description: DESC});
  const p = summaryPanels(win)[0];
  const prev = p.querySelector('.tc-h .collapse-preview');
  assert.ok(prev, 'addCollapse still installs the preview span');
  assert.strictEqual(
    (prev.textContent || '').trim(),
    '',
    'the ellipsised header preview must stay empty for summary panels ' +
      '(the description is already fully visible below the header)',
  );
  win.close();
  console.log('  ok - header collapse-preview suppressed for summary');
}

function testHeaderClickTogglesAndKeepsChildren() {
  const {win} = makeWebview();
  sendToolPanels(win, 6);
  send(win, {type: 'tool_call', name: 'summary', description: DESC});
  const p = summaryPanels(win)[0];
  const hdr = p.querySelector('.tc-h');
  hdr.dispatchEvent(
    new win.MouseEvent('click', {bubbles: true, cancelable: true}),
  );
  assert.ok(
    !p.classList.contains('collapsed'),
    'clicking the header must expand the summary panel',
  );
  let nested = p.querySelector(':scope > .summary-sub').children;
  assert.strictEqual(nested.length, 6, 'children survive the expand');
  hdr.dispatchEvent(
    new win.MouseEvent('click', {bubbles: true, cancelable: true}),
  );
  assert.ok(
    p.classList.contains('collapsed'),
    'clicking again re-collapses the panel',
  );
  nested = p.querySelector(':scope > .summary-sub').children;
  assert.strictEqual(nested.length, 6, 'children survive the re-collapse');
  const prev = p.querySelector('.tc-h .collapse-preview');
  assert.strictEqual(
    (prev.textContent || '').trim(),
    '',
    're-collapsing must not fill the truncated header preview',
  );
  win.close();
  console.log('  ok - header click toggles collapse, children preserved');
}

function testChainedSummariesNestEarlierSummary() {
  const {win} = makeWebview();
  send(win, {type: 'prompt', text: 'go'});
  sendToolPanels(win, 8);
  send(win, {type: 'tool_call', name: 'summary', description: 'first. ' + DESC});
  send(win, {type: 'tool_result', name: 'summary', content: 'Summary recorded.'});
  sendToolPanels(win, 2, 100);
  send(win, {type: 'tool_call', name: 'summary', description: 'second. ' + DESC});
  const panels = summaryPanels(win);
  assert.strictEqual(panels.length, 2, 'both summary panels exist');
  const second = panels.filter(
    x => x.querySelector(':scope > .tc-summary-desc').textContent.startsWith('second'),
  )[0];
  const nested = Array.from(
    second.querySelector(':scope > .summary-sub').children,
  );
  const first = panels.filter(x => x !== second)[0];
  assert.ok(
    nested.indexOf(first) !== -1,
    'a later summary must nest the earlier summary panel (hierarchy)',
  );
  assert.ok(
    first.querySelector(':scope > .summary-sub').children.length === 6,
    'the earlier summary keeps its own 6 nested panels',
  );
  win.close();
  console.log('  ok - chained summaries nest hierarchically');
}

function testToolResultLandsInsideCollapsedSummaryPanel() {
  const {win} = makeWebview();
  sendToolPanels(win, 2);
  send(win, {type: 'tool_call', name: 'summary', description: DESC});
  send(win, {type: 'tool_result', name: 'summary', content: 'Summary recorded.'});
  const p = summaryPanels(win)[0];
  assert.ok(
    p.querySelector('.bash-panel'),
    'the summary tool_result output must land inside the summary panel',
  );
  assert.ok(
    p.classList.contains('collapsed'),
    'the tool_result must not un-collapse the summary panel',
  );
  win.close();
  console.log('  ok - summary tool_result targets the collapsed panel');
}

function testNonSummaryToolCallUnaffected() {
  const {win} = makeWebview();
  sendToolPanels(win, 7);
  send(win, {
    type: 'tool_call',
    name: 'Bash',
    command: 'ls',
    description: 'list',
  });
  const O = output(win);
  const last = O.lastElementChild;
  assert.ok(!last.classList.contains('tc-summary'));
  assert.ok(
    !last.querySelector('.summary-sub'),
    'non-summary tools must never adopt preceding panels',
  );
  assert.ok(
    !last.classList.contains('collapsed'),
    'non-summary tools are not auto-collapsed on render',
  );
  assert.strictEqual(summaryPanels(win).length, 0);
  win.close();
  console.log('  ok - non-summary tool_call rendering unchanged');
}

function testReplayPathNestsAndCollapses() {
  const {win} = makeWebview();
  const events = [{type: 'prompt', text: 'replayed task'}];
  for (let i = 0; i < 7; i++) {
    events.push({type: 'tool_call', name: 'Read', path: '/tmp/r' + i});
    events.push({type: 'tool_result', name: 'Read', content: 'x' + i});
  }
  events.push({type: 'tool_call', name: 'summary', description: DESC});
  events.push({
    type: 'tool_result',
    name: 'summary',
    content: 'Summary recorded.',
  });
  events.push({
    type: 'result',
    text: 'done',
    summary: 'done',
    success: true,
  });
  send(win, {
    type: 'task_events',
    task: 'replayed task',
    task_id: 42,
    events: events,
  });
  const panels = summaryPanels(win);
  assert.strictEqual(panels.length, 1, 'replay renders the summary panel');
  const p = panels[0];
  assert.ok(p.classList.contains('collapsed'), 'replayed panel collapses');
  const nested = p.querySelector(':scope > .summary-sub').children;
  assert.strictEqual(
    nested.length,
    6,
    'replay nests the last 6 panels exactly like the live stream',
  );
  assert.strictEqual(
    p.querySelector(':scope > .tc-summary-desc').textContent,
    DESC,
    'replayed description is fully preserved',
  );
  win.close();
  console.log('  ok - history replay (task_events) path behaves the same');
}

/**
 * True when *el* and every ancestor up to and including #output
 * computes display !== 'none'.  (#app above #output stays hidden in
 * this harness until the extension's init handshake — unrelated to
 * panel rendering.)
 */
function isDisplayed(win, el) {
  for (let n = el; n && n.nodeType === 1; n = n.parentElement) {
    if (win.getComputedStyle(n).getPropertyValue('display').trim() === 'none')
      return false;
    if (n.id === 'output') break;
  }
  return true;
}

/** Replay a completed task containing a summary; return its window. */
function replayCompletedSummaryTask(win) {
  const events = [{type: 'prompt', text: 'replayed task'}];
  for (let i = 0; i < 7; i++) {
    events.push({type: 'tool_call', name: 'Read', path: '/tmp/r' + i});
    events.push({type: 'tool_result', name: 'Read', content: 'x' + i});
  }
  events.push({type: 'tool_call', name: 'summary', description: DESC});
  events.push({
    type: 'tool_result',
    name: 'summary',
    content: 'Summary recorded.',
  });
  events.push({type: 'result', text: 'done', summary: 'done', success: true});
  send(win, {
    type: 'task_events',
    task: 'replayed task',
    task_id: 42,
    events: events,
  });
}

function testReplayedSummaryStaysVisibleDespiteChevronCollapse() {
  // Regression (review finding): the task-level "Collapse Chats"
  // machinery (applyChevronState → .chv-hidden { display:none
  // !important }) runs after a completed-task replay and used to hide
  // the ENTIRE summary panel, defeating the always-visible digest.
  const {win} = makeWebview();
  injectCss(win);
  replayCompletedSummaryTask(win);
  const p = summaryPanels(win)[0];
  assert.ok(p, 'summary panel replays');
  assert.ok(
    !p.classList.contains('chv-hidden'),
    'the summary panel must be exempt from task-level chv-hidden',
  );
  assert.ok(
    p.classList.contains('collapsed'),
    'the replayed summary panel stays in its collapsed digest state',
  );
  assert.ok(
    isDisplayed(win, p),
    'the summary panel (and all its ancestors) must remain displayed ' +
      'after a completed-task replay',
  );
  const desc = p.querySelector(':scope > .tc-summary-desc');
  assert.ok(
    isDisplayed(win, desc),
    'the description must be fully visible after replay',
  );
  // Ordinary top-level panels are still tucked away by the task-level
  // collapse — the digest is the only thing that stays readable.
  const plainTc = topLevel(win)
    .flatMap(el =>
      el.classList && el.classList.contains('adjacent-task')
        ? Array.from(el.querySelectorAll(':scope > .tc'))
        : [el],
    )
    .filter(
      el =>
        el.classList &&
        el.classList.contains('tc') &&
        !el.classList.contains('tc-summary'),
    );
  assert.ok(
    plainTc.every(el => el.classList.contains('chv-hidden')),
    'non-summary panels keep the pre-existing chv-hidden behavior',
  );
  win.close();
  console.log('  ok - replayed summary stays visible (no chv-hidden)');
}

function testAdoptedPanelsRevealAfterManualExpandPostReplay() {
  // The adopted sub-panels must not carry chv-hidden either: after the
  // user manually expands the replayed summary, its nested panels must
  // actually appear.
  const {win} = makeWebview();
  injectCss(win);
  replayCompletedSummaryTask(win);
  const p = summaryPanels(win)[0];
  const hdr = p.querySelector('.tc-h');
  hdr.dispatchEvent(
    new win.MouseEvent('click', {bubbles: true, cancelable: true}),
  );
  assert.ok(!p.classList.contains('collapsed'), 'header click expands');
  const nested = Array.from(p.querySelector(':scope > .summary-sub').children);
  assert.strictEqual(nested.length, 6);
  for (const el of nested) {
    assert.ok(
      !el.classList.contains('chv-hidden'),
      'adopted panels must not be chv-hidden inside the summary',
    );
    assert.ok(
      isDisplayed(win, el),
      'every adopted panel must be visible once the summary is expanded',
    );
  }
  win.close();
  console.log('  ok - adopted panels reveal after manual expand post-replay');
}

function testAdoptedPanelKeepsOwnCollapsePreview() {
  // Regression (review finding): the old descendant selector
  // ``.tc-summary .tc-h .collapse-preview`` also hid the header
  // previews of panels ADOPTED into the summary.
  const {win} = makeWebview();
  injectCss(win);
  sendToolPanels(win, 6);
  send(win, {type: 'tool_call', name: 'summary', description: DESC});
  const p = summaryPanels(win)[0];
  const hdr = p.querySelector(':scope > .tc-h');
  hdr.dispatchEvent(
    new win.MouseEvent('click', {bubbles: true, cancelable: true}),
  );
  const nestedPanel = p.querySelector(':scope > .summary-sub > .tc');
  const nestedHdr = nestedPanel.querySelector(':scope > .tc-h');
  nestedHdr.dispatchEvent(
    new win.MouseEvent('click', {bubbles: true, cancelable: true}),
  );
  assert.ok(nestedPanel.classList.contains('collapsed'));
  const nestedPrev = nestedHdr.querySelector('.collapse-preview');
  assert.ok(
    (nestedPrev.textContent || '').length > 0,
    'a collapsed adopted panel fills its own header preview',
  );
  assert.notStrictEqual(
    win.getComputedStyle(nestedPrev).getPropertyValue('display').trim(),
    'none',
    'the adopted panel preview must stay visible (only the summary ' +
      "panel's OWN preview is suppressed)",
  );
  const ownPrev = hdr.querySelector('.collapse-preview');
  assert.strictEqual(
    win.getComputedStyle(ownPrev).getPropertyValue('display').trim(),
    'none',
    "the summary panel's own header preview stays suppressed",
  );
  win.close();
  console.log('  ok - adopted panel keeps its own collapse preview');
}

function testRcResultPanelIsBoundary() {
  const {win} = makeWebview();
  send(win, {type: 'prompt', text: 'go'});
  sendToolPanels(win, 2);
  send(win, {type: 'result', text: 'partial', summary: 'partial'});
  sendToolPanels(win, 2, 50);
  send(win, {type: 'tool_call', name: 'summary', description: DESC});
  const p = summaryPanels(win)[0];
  const nested = Array.from(p.querySelector(':scope > .summary-sub').children);
  assert.strictEqual(
    nested.length,
    2,
    'adoption must stop at the .rc result panel — got ' + nested.length,
  );
  const rc = output(win).querySelector('.rc');
  assert.ok(rc, 'the result panel exists');
  assert.ok(
    !p.contains(rc),
    'a result (.rc) panel must never be swallowed by a summary',
  );
  win.close();
  console.log('  ok - .rc result panels are a boundary');
}

function testAdjacentTaskAloneIsBoundary() {
  // Unlike testAdjacentTaskAndSystemPromptAreBoundaries, ONLY an
  // .adjacent-task block precedes the tool panels, proving it is a
  // boundary in its own right.
  const {win} = makeWebview();
  const O = output(win);
  const adj = win.document.createElement('div');
  adj.className = 'adjacent-task';
  O.appendChild(adj);
  sendToolPanels(win, 2);
  send(win, {type: 'tool_call', name: 'summary', description: DESC});
  const p = summaryPanels(win)[0];
  const nested = Array.from(p.querySelector(':scope > .summary-sub').children);
  assert.strictEqual(nested.length, 2, 'adoption stops at .adjacent-task');
  assert.strictEqual(
    adj.parentElement,
    O,
    'the previous-task history block must never be adopted',
  );
  win.close();
  console.log('  ok - .adjacent-task alone is a boundary');
}

function runTests() {
  testSummaryPanelCreatedAndCollapsed();
  testDescriptionIsDirectChildWithFullText();
  testNestsExactlySixMostRecentPanelsInOrder();
  testFewerThanSixStopsAtPromptBoundary();
  testAdjacentTaskAndSystemPromptAreBoundaries();
  testWelcomeBlockNeverAdopted();
  testNoPrecedingPanels();
  testEmptyDescription();
  testDescriptionFullyVisibleWhileCollapsedViaCss();
  testCollapsePreviewSuppressedForSummary();
  testHeaderClickTogglesAndKeepsChildren();
  testChainedSummariesNestEarlierSummary();
  testToolResultLandsInsideCollapsedSummaryPanel();
  testNonSummaryToolCallUnaffected();
  testReplayPathNestsAndCollapses();
  testReplayedSummaryStaysVisibleDespiteChevronCollapse();
  testAdoptedPanelsRevealAfterManualExpandPostReplay();
  testAdoptedPanelKeepsOwnCollapsePreview();
  testRcResultPanelIsBoundary();
  testAdjacentTaskAloneIsBoundary();
}

try {
  runTests();
  console.log('\n20 passed, 0 failed');
  process.exit(0);
} catch (err) {
  console.error('FAIL:', err && err.stack ? err.stack : err);
  process.exit(1);
}
