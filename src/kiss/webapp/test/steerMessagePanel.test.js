// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// A message the user types into a RUNNING task comes back from the
// daemon as a `prompt` event flagged `steer`.  The task and the
// steering messages are both the user's words, so the page shows
// the message in a panel like the task panel (accent tint, plain
// text), headed "Message", and no automatic pass ever folds it.  A
// prompt event without the flag is still the neutral "Prompt" panel.

const assert = require('assert');
const harness = require('./simplify2_harness');
const {readyTabId, output} = harness;

/** A harness page with main.css applied: {win, posted}. */
function makePage() {
  return harness.makePage({beforeScripts: harness.loadMainCss});
}

/** The window of a fresh page on which *task* has been started. */
function startedPage(task) {
  const {win, posted} = makePage();
  harness.startTask(win, posted, task);
  return win;
}

const send = harness.send;

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
  send(win, {
    type: 'tool_result',
    name: 'Bash',
    output: String(i),
    success: true,
  });
}

function testSteerPromptRendersLikeTaskPanel() {
  const win = startedPage('Fix the login bug');
  sendStep(win, 1);
  send(win, {
    type: 'prompt',
    text: 'Also check the logout path\nand the tests',
    steer: true,
    ts: 1767225600000,
  });
  const msg = output(win).querySelector('.ev.user-msg');
  assert.ok(msg, 'the steering message renders a .ev.user-msg panel');
  assert.ok(msg.classList.contains('collapsible'), 'it is a collapsible panel');
  const hdr = msg.querySelector('.task-panel-h');
  assert.ok(
    hdr && hdr.classList.contains('collapse-header'),
    'its header is the task panel header',
  );
  assert.strictEqual(hdr.firstChild.textContent, 'Message', 'headed "Message"');
  const body = msg.querySelector('.task-panel-text');
  assert.strictEqual(
    body.textContent,
    'Also check the logout path\nand the tests',
    'plain text body',
  );
  assert.strictEqual(
    msg.dataset.rawText,
    'Also check the logout path\nand the tests',
    'copyable raw text',
  );
  assert.strictEqual(
    output(win).querySelectorAll('.ev.prompt').length,
    1,
    'the task prompt is still the one Prompt panel; the steer echo is not a second one',
  );
  // Same accent look as the task panel.  jsdom leaves a var() in the
  // `background` shorthand unresolved (backgroundColor reads transparent
  // for every panel), so the declared backgrounds are compared.
  const task = output(win).querySelector('.ev.task-panel');
  assert.ok(task, 'the task panel is on screen');
  const tint = win.getComputedStyle(task).background;
  assert.strictEqual(
    win.getComputedStyle(msg).background,
    tint,
    'same background as the task panel',
  );
  assert.notStrictEqual(
    win.getComputedStyle(output(win).querySelector('.ev.prompt')).background,
    tint,
    'the neutral Prompt panel does not carry the accent tint',
  );
  assert.strictEqual(
    win.getComputedStyle(body).whiteSpace,
    win.getComputedStyle(task.querySelector('.task-panel-text')).whiteSpace,
    'same text layout as the task panel',
  );
  assert.ok(
    msg.querySelector('.panel-ts'),
    'the message carries its event timestamp',
  );
  win.close();
  console.log(
    '  ok - a steer prompt renders as a "Message" panel like the task panel',
  );
}

function testPlainPromptStaysPromptPanel() {
  const win = startedPage('Plain prompt');
  const before = output(win).querySelectorAll('.ev.prompt').length;
  send(win, {type: 'prompt', text: 'authoritative prompt text'});
  assert.strictEqual(
    output(win).querySelectorAll('.ev.user-msg').length,
    0,
    'no Message panel',
  );
  assert.strictEqual(
    output(win).querySelectorAll('.ev.prompt').length,
    before + 1,
    'the plain prompt renders a Prompt panel',
  );
  win.close();
  console.log('  ok - a prompt without the steer flag is still a Prompt panel');
}

function testMessagePanelNeverAutoFolded() {
  const win = startedPage('Never fold the message');
  sendStep(win, 1);
  send(win, {type: 'prompt', text: 'steer one', steer: true});
  for (let i = 2; i <= 5; i++) sendStep(win, i);
  let msg = output(win).querySelector('.ev.user-msg');
  assert.ok(
    !msg.classList.contains('collapsed'),
    'the streaming sweep leaves the message open',
  );
  assert.ok(
    output(win).querySelector('.tc-bash').classList.contains('collapsed'),
    'while the older tool panels fold',
  );
  // A summary tool call adopts the tool panels but leaves the message
  // open (that it stays top-level is pinned by
  // promptPanelsCollapsedAndSummarized.test.js).
  send(win, {type: 'tool_call', name: 'summary', description: 'five steps'});
  msg = output(win).querySelector('.ev.user-msg');
  assert.ok(!msg.classList.contains('collapsed'));
  win.close();

  // A replayed, finished task keeps it open too.
  const {win: win2, posted: posted2} = makePage();
  const events = [{type: 'prompt', text: 'replayed'}];
  for (let i = 1; i <= 2; i++) {
    events.push({
      type: 'tool_call',
      name: 'Bash',
      command: 'x',
      description: 'd' + i,
    });
    events.push({
      type: 'tool_result',
      name: 'Bash',
      output: 'y',
      success: true,
    });
    events.push({type: 'prompt', text: 'steer ' + i, steer: true});
  }
  events.push({type: 'result', text: 'done', success: true});
  send(win2, {
    type: 'task_events',
    events,
    task: 'replayed',
    tabId: readyTabId(posted2),
    chat_id: 'c-replay',
  });
  send(win2, {type: 'status', running: false});
  send(win2, {type: 'usage_info', text: 'Steps: 2/10'});
  const msgs = Array.from(output(win2).querySelectorAll('.ev.user-msg'));
  assert.strictEqual(msgs.length, 2, 'both replayed messages render');
  for (const m of msgs) {
    assert.ok(!m.classList.contains('collapsed'), 'a replayed message is open');
    assert.ok(
      !m.classList.contains('chv-hidden'),
      'and not hidden by the digest',
    );
    assert.strictEqual(
      m.querySelector('.task-panel-h').firstChild.textContent,
      'Message',
    );
  }
  assert.ok(
    Array.from(output(win2).querySelectorAll('.tc-bash')).every(p =>
      p.classList.contains('collapsed'),
    ),
    'the replay folds the tool panels',
  );
  win2.close();
  console.log('  ok - no automatic pass folds or hides a Message panel');
}

function testMessagePanelFoldsByClick() {
  const win = startedPage('Fold by hand');
  send(win, {type: 'prompt', text: 'fold me please', steer: true});
  const msg = output(win).querySelector('.ev.user-msg');
  const hdr = msg.querySelector('.collapse-header');
  hdr.click();
  assert.ok(msg.classList.contains('collapsed'), 'the header click folds it');
  assert.strictEqual(
    win.getComputedStyle(msg.querySelector('.task-panel-text')).display,
    'none',
    'the text hides',
  );
  assert.strictEqual(
    hdr.querySelector('.collapse-preview').textContent,
    'fold me please',
    'the header previews the message',
  );
  hdr.click();
  assert.ok(!msg.classList.contains('collapsed'), 'a second click unfolds it');
  win.close();
  console.log('  ok - only a header click folds a Message panel');
}

function main() {
  testSteerPromptRendersLikeTaskPanel();
  testPlainPromptStaysPromptPanel();
  testMessagePanelNeverAutoFolded();
  testMessagePanelFoldsByClick();
  console.log('steerMessagePanel: all tests passed');
}

main();
