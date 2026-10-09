// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// The "System Prompt" and "Prompt" panels of a running task start
// collapsed: their header is all the transcript shows until the user
// unfolds one.  The first `summary` tool call of the run then folds
// them under the summary digest together with the step panels it
// recounts, so the transcript above the digest holds nothing but the
// task panel.  A later summary stops at the earlier one, as before.

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

function makeWebview() {
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
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
  // Markdown rendering, so a prompt can show a picture.
  win.eval(fs.readFileSync(path.join(MEDIA, 'marked.min.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
  const styleEl = win.document.createElement('style');
  styleEl.textContent = fs.readFileSync(path.join(MEDIA, 'main.css'), 'utf8');
  win.document.head.appendChild(styleEl);
  win.__posted = posted;
  return win;
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function readyTabId(win) {
  const ready = win.__posted.find(m => m.type === 'ready');
  assert.ok(ready && ready.tabId, 'the webview posts ready with a tabId');
  return ready.tabId;
}

function output(win) {
  return win.document.getElementById('output');
}

/** Start a running task on the ready tab; sends no prompt events. */
function startTask(win, task) {
  send(win, {
    type: 'task_events',
    events: [],
    task,
    tabId: readyTabId(win),
    chat_id: 'chat-' + task.replace(/\W+/g, '-'),
  });
  send(win, {type: 'status', running: true});
}

function sendStep(win, i) {
  send(win, {type: 'thinking_start'});
  send(win, {type: 'thinking_delta', text: 'thought ' + i});
  send(win, {type: 'thinking_end'});
  send(win, {
    type: 'tool_call',
    name: 'Bash',
    command: 'echo ' + i,
    description: 'step ' + i,
  });
  send(win, {type: 'tool_result', name: 'Bash', output: String(i), success: true});
}

function topLevel(win) {
  return Array.from(output(win).children).filter(el => el.id !== 'welcome');
}

function header(panel, cls) {
  return panel.querySelector(':scope > .' + cls + '-h');
}

function testPromptPanelsStartCollapsed() {
  const win = makeWebview();
  startTask(win, 'Fold the prompts');
  send(win, {type: 'system_prompt', text: 'You are a careful assistant.'});
  send(win, {type: 'prompt', text: 'Fold the prompts'});
  const sys = output(win).querySelector('.ev.system-prompt');
  const pr = output(win).querySelector('.ev.prompt');
  assert.ok(sys && pr, 'both panels render');
  for (const [panel, cls] of [[sys, 'system-prompt'], [pr, 'prompt']]) {
    assert.ok(
      panel.classList.contains('collapsed'),
      'the ' + cls + ' panel starts collapsed while the task runs',
    );
    const hdr = header(panel, cls);
    assert.strictEqual(hdr.getAttribute('aria-expanded'), 'false');
    const body = panel.querySelector(':scope > .' + cls + '-body');
    assert.strictEqual(
      win.getComputedStyle(body).display,
      'none',
      'the collapsed ' + cls + ' body is hidden by the stylesheet',
    );
    const preview = hdr.querySelector('.collapse-preview');
    assert.ok(
      preview && preview.textContent.length > 0,
      'the folded header carries a one-line preview of the text',
    );
  }
  win.close();
  console.log('  ok - System Prompt and Prompt panels start collapsed');
}

function testEarlyPromptPanelsStartCollapsedAndStayCollapsedWhenReplaced() {
  const win = makeWebview();
  startTask(win, 'Early prompts');
  send(win, {type: 'system_prompt', text: 'early system', early: true});
  send(win, {type: 'prompt', text: 'early prompt', early: true});
  let sys = output(win).querySelector('.ev.system-prompt');
  let pr = output(win).querySelector('.ev.prompt');
  assert.ok(sys.classList.contains('collapsed'), 'early system prompt folded');
  assert.ok(pr.classList.contains('collapsed'), 'early prompt folded');
  send(win, {type: 'system_prompt', text: 'authoritative system'});
  send(win, {type: 'prompt', text: 'authoritative prompt'});
  sys = output(win).querySelector('.ev.system-prompt');
  pr = output(win).querySelector('.ev.prompt');
  assert.strictEqual(output(win).querySelectorAll('.ev.prompt').length, 1);
  assert.ok(sys.classList.contains('collapsed'), 'replaced system prompt folded');
  assert.ok(pr.classList.contains('collapsed'), 'replaced prompt folded');
  assert.strictEqual(header(pr, 'prompt').getAttribute('aria-expanded'), 'false');
  assert.ok(pr.textContent.includes('authoritative prompt'));
  win.close();
  console.log('  ok - early placeholders and their replacements are collapsed');
}

function testUserUnfoldSurvivesEarlyReplace() {
  const win = makeWebview();
  startTask(win, 'Unfold early');
  send(win, {type: 'prompt', text: 'early prompt', early: true});
  const pr = output(win).querySelector('.ev.prompt');
  header(pr, 'prompt').dispatchEvent(
    new win.MouseEvent('click', {bubbles: true, cancelable: true}),
  );
  assert.ok(!pr.classList.contains('collapsed'), 'the click unfolds it');
  send(win, {type: 'prompt', text: 'authoritative prompt'});
  assert.strictEqual(output(win).querySelector('.ev.prompt'), pr);
  assert.ok(
    !pr.classList.contains('collapsed'),
    'the authoritative text lands in the panel the user unfolded, still open',
  );
  assert.strictEqual(header(pr, 'prompt').getAttribute('aria-expanded'), 'true');
  win.close();
  console.log('  ok - a prompt the user unfolded stays open when its text arrives');
}

function testUserCanUnfoldAndRefold() {
  const win = makeWebview();
  startTask(win, 'Toggle');
  send(win, {type: 'system_prompt', text: 'system text'});
  const sys = output(win).querySelector('.ev.system-prompt');
  const hdr = header(sys, 'system-prompt');
  hdr.dispatchEvent(new win.MouseEvent('click', {bubbles: true, cancelable: true}));
  assert.ok(!sys.classList.contains('collapsed'), 'a click opens the panel');
  assert.ok(sys.classList.contains('user-pinned'), 'and pins it open');
  assert.strictEqual(hdr.getAttribute('aria-expanded'), 'true');
  const body = sys.querySelector(':scope > .system-prompt-body');
  assert.notStrictEqual(win.getComputedStyle(body).display, 'none');
  for (let i = 1; i <= 4; i++) sendStep(win, i);
  assert.ok(
    !sys.classList.contains('collapsed'),
    'the streaming sweep leaves a pinned panel open',
  );
  hdr.dispatchEvent(new win.MouseEvent('click', {bubbles: true, cancelable: true}));
  assert.ok(sys.classList.contains('collapsed'), 'a second click folds it again');
  win.close();
  console.log('  ok - the user can unfold and refold a prompt panel');
}

function testFirstSummaryFoldsPromptsUnderIt() {
  const win = makeWebview();
  startTask(win, 'Summarize');
  send(win, {type: 'system_prompt', text: 'system text'});
  send(win, {type: 'prompt', text: 'Summarize'});
  for (let i = 1; i <= 3; i++) sendStep(win, i);
  const sys = output(win).querySelector('.ev.system-prompt');
  const pr = output(win).querySelector('.ev.prompt');
  // The panels the summary recounts: everything after the task panel
  // except the empty provisional Thoughts panel the next tool call
  // discards.
  const expectNested = topLevel(win)
    .slice(1)
    .filter(el => !el.classList.contains('panel-active'));
  send(win, {type: 'tool_call', name: 'summary', description: 'three steps'});
  const summary = output(win).querySelector('.tc-summary');
  assert.ok(summary, 'the summary panel renders');
  assert.ok(summary.classList.contains('collapsed'), 'the summary folds');
  const sub = summary.querySelector(':scope > .summary-sub');
  const nested = Array.from(sub.children);
  assert.strictEqual(nested[0], sys, 'the System Prompt is the first panel in the digest');
  assert.strictEqual(nested[1], pr, 'the Prompt follows it');
  assert.strictEqual(nested.length, 8, '2 prompts + 3 Thoughts + 3 tool panels');
  assert.deepStrictEqual(
    nested,
    expectNested,
    'every step panel follows the prompts, in order',
  );
  const top = topLevel(win);
  assert.deepStrictEqual(
    top.map(el => el.className.split(' ').slice(0, 2).join(' ')),
    ['ev task-panel', 'ev tc'],
    'above the digest only the task panel remains',
  );
  assert.ok(sys.classList.contains('collapsed'), 'the adopted System Prompt stays folded');
  assert.ok(pr.classList.contains('collapsed'), 'the adopted Prompt stays folded');
  win.close();
  console.log('  ok - the first summary folds the prompts under itself');
}

function testFirstSummaryFoldsPromptsEvenWhenUserUnfoldedThem() {
  const win = makeWebview();
  startTask(win, 'Pinned prompt');
  send(win, {type: 'prompt', text: 'Pinned prompt'});
  const pr = output(win).querySelector('.ev.prompt');
  header(pr, 'prompt').dispatchEvent(
    new win.MouseEvent('click', {bubbles: true, cancelable: true}),
  );
  assert.ok(pr.classList.contains('user-pinned'));
  sendStep(win, 1);
  send(win, {type: 'tool_call', name: 'summary', description: 'one step'});
  const summary = output(win).querySelector('.tc-summary');
  assert.strictEqual(
    pr.parentElement,
    summary.querySelector(':scope > .summary-sub'),
    'the summary digest recounts the prompt too, pinned or not',
  );
  win.close();
  console.log('  ok - a pinned prompt is folded under the first summary as well');
}

function testPromptWithImageIsFoldedUnderSummary() {
  const win = makeWebview();
  startTask(win, 'Picture prompt');
  send(win, {
    type: 'prompt',
    text: 'Look at this ![shot](data:image/png;base64,iVBORw0KGgo=)',
  });
  const pr = output(win).querySelector('.ev.prompt');
  assert.ok(pr.querySelector('img'), 'the prompt shows its image');
  assert.ok(pr.classList.contains('collapsed'), 'and still starts folded');
  sendStep(win, 1);
  send(win, {type: 'tool_call', name: 'summary', description: 'one step'});
  const summary = output(win).querySelector('.tc-summary');
  assert.strictEqual(
    pr.parentElement,
    summary.querySelector(':scope > .summary-sub'),
    'a prompt is adopted even when it shows a picture',
  );
  assert.strictEqual(
    summary.nextElementSibling,
    null,
    'nothing is moved after the summary',
  );
  win.close();
  console.log('  ok - a prompt showing an image is folded under the summary too');
}

function testSecondSummaryStopsAtTheFirst() {
  const win = makeWebview();
  startTask(win, 'Two summaries');
  send(win, {type: 'system_prompt', text: 'system text'});
  send(win, {type: 'prompt', text: 'Two summaries'});
  sendStep(win, 1);
  send(win, {type: 'tool_call', name: 'summary', description: 'first'});
  send(win, {type: 'tool_result', name: 'summary', content: 'ok'});
  sendStep(win, 2);
  sendStep(win, 3);
  send(win, {type: 'tool_call', name: 'summary', description: 'second'});
  const summaries = Array.from(output(win).querySelectorAll('.tc-summary'));
  assert.strictEqual(summaries.length, 2);
  const first = summaries[0];
  const second = summaries[1];
  assert.strictEqual(first.parentElement, output(win), 'the first summary stays top-level');
  assert.ok(
    first.querySelector(':scope > .summary-sub > .ev.prompt') &&
      first.querySelector(':scope > .summary-sub > .ev.system-prompt'),
    'the prompts sit under the first summary',
  );
  assert.strictEqual(
    second.querySelectorAll(':scope > .summary-sub > .ev.prompt, :scope > .summary-sub > .ev.system-prompt').length,
    0,
    'the second summary adopts no prompt',
  );
  assert.strictEqual(
    second.querySelectorAll(':scope > .summary-sub > .tc').length,
    2,
    'the second summary adopts only the steps after the first',
  );
  win.close();
  console.log('  ok - the second summary stops at the first');
}

function testSteerMessageStaysOutsideWhilePromptEchoFolds() {
  const win = makeWebview();
  startTask(win, 'Steered');
  send(win, {type: 'prompt', text: 'Steered'});
  sendStep(win, 1);
  send(win, {type: 'prompt', text: 'also do this', steer: true});
  sendStep(win, 2);
  send(win, {type: 'tool_call', name: 'summary', description: 'two steps'});
  const summary = output(win).querySelector('.tc-summary');
  const msg = output(win).querySelector('.ev.user-msg');
  assert.strictEqual(msg.parentElement, output(win), 'the steer message stays top-level');
  assert.strictEqual(msg.previousElementSibling, summary, 'right after the summary');
  assert.ok(
    summary.querySelector(':scope > .summary-sub > .ev.prompt'),
    'while the Prompt panel folds under the summary',
  );
  win.close();
  console.log('  ok - a steer message is preserved; the prompt is folded');
}

function testReplayOfRunningTaskFoldsPromptsUnderSummary() {
  const win = makeWebview();
  const events = [
    {type: 'system_prompt', text: 'system text'},
    {type: 'prompt', text: 'replayed'},
    {type: 'tool_call', name: 'Read', path: '/tmp/a'},
    {type: 'tool_result', name: 'Read', content: 'a'},
    {type: 'tool_call', name: 'summary', description: 'so far'},
    {type: 'tool_result', name: 'summary', content: 'ok'},
    {type: 'tool_call', name: 'Read', path: '/tmp/b'},
    {type: 'tool_result', name: 'Read', content: 'b'},
  ];
  send(win, {
    type: 'task_events',
    events,
    task: 'replayed',
    tabId: readyTabId(win),
    chat_id: 'chat-replay',
  });
  send(win, {type: 'status', running: true});
  const summary = output(win).querySelector('.tc-summary');
  assert.ok(summary, 'the replayed summary renders');
  const sys = output(win).querySelector('.ev.system-prompt');
  const pr = output(win).querySelector('.ev.prompt');
  const sub = summary.querySelector(':scope > .summary-sub');
  assert.strictEqual(sys.parentElement, sub, 'replayed System Prompt under the summary');
  assert.strictEqual(pr.parentElement, sub, 'replayed Prompt under the summary');
  assert.ok(sys.classList.contains('collapsed') && pr.classList.contains('collapsed'));
  win.close();
  console.log('  ok - a replayed running task folds the prompts under its summary');
}

function testFinishedTaskFoldsEverythingIntoTrajectory() {
  const win = makeWebview();
  startTask(win, 'Finished');
  send(win, {type: 'system_prompt', text: 'system text'});
  send(win, {type: 'prompt', text: 'Finished'});
  sendStep(win, 1);
  send(win, {type: 'tool_call', name: 'summary', description: 'one step'});
  send(win, {type: 'tool_result', name: 'summary', content: 'ok'});
  send(win, {type: 'result', text: 'done', success: true});
  send(win, {type: 'task_done'});
  send(win, {type: 'status', running: false});
  const traj = output(win).querySelector('.ev.trajectory');
  assert.ok(traj, 'the finished task folds into a Trajectory panel');
  const summary = traj.querySelector(':scope > .trajectory-sub > .tc-summary');
  assert.ok(summary, 'the summary sits in the trajectory');
  assert.ok(
    summary.querySelector(':scope > .summary-sub > .ev.system-prompt') &&
      summary.querySelector(':scope > .summary-sub > .ev.prompt'),
    'with the prompts still under it',
  );
  win.close();
  console.log('  ok - a finished task keeps the prompts under its summary');
}

function runTests() {
  testPromptPanelsStartCollapsed();
  testEarlyPromptPanelsStartCollapsedAndStayCollapsedWhenReplaced();
  testUserUnfoldSurvivesEarlyReplace();
  testUserCanUnfoldAndRefold();
  testFirstSummaryFoldsPromptsUnderIt();
  testFirstSummaryFoldsPromptsEvenWhenUserUnfoldedThem();
  testPromptWithImageIsFoldedUnderSummary();
  testSecondSummaryStopsAtTheFirst();
  testSteerMessageStaysOutsideWhilePromptEchoFolds();
  testReplayOfRunningTaskFoldsPromptsUnderSummary();
  testFinishedTaskFoldsEverythingIntoTrajectory();
}

try {
  runTests();
  console.log('\n11 passed, 0 failed');
  process.exit(0);
} catch (err) {
  console.error('FAIL:', err && err.stack ? err.stack : err);
  process.exit(1);
}
