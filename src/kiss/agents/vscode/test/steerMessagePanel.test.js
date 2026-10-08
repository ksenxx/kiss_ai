// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// A message the user types into a RUNNING task comes back from the
// daemon as a `prompt` event flagged `steer`.  The task and the
// steering messages are both the user's words, so the webview shows
// the message in a panel like the task panel (accent tint, plain
// text), headed "Message", and no automatic pass ever folds it.  A
// prompt event without the flag is still the neutral "Prompt" panel.

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

function startTask(win, task) {
  send(win, {
    type: 'task_events',
    events: [],
    task,
    tabId: readyTabId(win),
    chat_id: 'chat-' + task.replace(/\W+/g, '-'),
  });
  send(win, {type: 'status', running: true});
  send(win, {type: 'prompt', text: task});
}

function sendStep(win, i) {
  send(win, {type: 'thinking_start'});
  send(win, {type: 'thinking_delta', text: 'thought ' + i});
  send(win, {type: 'thinking_end'});
  send(win, {type: 'tool_call', name: 'Bash', command: 'echo ' + i, description: 'step ' + i});
  send(win, {type: 'tool_result', name: 'Bash', output: String(i), success: true});
}

/** The stylesheet rule whose selector list names *selector*. */
function ruleFor(win, selector) {
  const sheets = win.document.styleSheets;
  for (let s = 0; s < sheets.length; s++) {
    const rules = sheets[s].cssRules;
    for (let r = 0; r < rules.length; r++) {
      const sel = rules[r].selectorText || '';
      if (sel.split(',').map(x => x.trim()).includes(selector)) return rules[r];
    }
  }
  return null;
}

function testSteerPromptRendersLikeTaskPanel() {
  const win = makeWebview();
  startTask(win, 'Fix the login bug');
  sendStep(win, 1);
  send(win, {type: 'prompt', text: 'Also check the logout path\nand the tests', steer: true, ts: 1767225600000});
  const msg = output(win).querySelector('.ev.user-msg');
  assert.ok(msg, 'the steering message renders a .ev.user-msg panel');
  assert.ok(msg.classList.contains('collapsible'), 'it is a collapsible panel');
  const hdr = msg.querySelector('.task-panel-h');
  assert.ok(hdr && hdr.classList.contains('collapse-header'), 'its header is the task panel header');
  assert.strictEqual(hdr.firstChild.textContent, 'Message', 'headed "Message"');
  const body = msg.querySelector('.task-panel-text');
  assert.strictEqual(body.textContent, 'Also check the logout path\nand the tests', 'plain text body');
  assert.strictEqual(msg.dataset.rawText, 'Also check the logout path\nand the tests', 'copyable raw text');
  assert.strictEqual(
    output(win).querySelectorAll('.ev.prompt').length,
    1,
    'the task prompt is still the one Prompt panel; the steer echo is not a second one',
  );
  // Same accent look as the task panel: one stylesheet rule paints both.
  const task = output(win).querySelector('.ev.task-panel');
  assert.ok(task, 'the task panel is on screen');
  const rule = ruleFor(win, '.ev.user-msg');
  assert.ok(rule && rule.selectorText.includes('.ev.task-panel'), 'the task panel rule also names .ev.user-msg: ' + (rule && rule.selectorText));
  assert.strictEqual(
    win.getComputedStyle(msg).backgroundColor,
    win.getComputedStyle(task).backgroundColor,
    'same background as the task panel',
  );
  assert.strictEqual(
    win.getComputedStyle(body).whiteSpace,
    win.getComputedStyle(task.querySelector('.task-panel-text')).whiteSpace,
    'same text layout as the task panel',
  );
  assert.ok(msg.querySelector('.panel-ts'), 'the message carries its event timestamp');
  win.close();
  console.log('  ok - a steer prompt renders as a "Message" panel like the task panel');
}

function testPlainPromptStaysPromptPanel() {
  const win = makeWebview();
  startTask(win, 'Plain prompt');
  send(win, {type: 'prompt', text: 'authoritative prompt text'});
  assert.strictEqual(output(win).querySelectorAll('.ev.user-msg').length, 0, 'no Message panel');
  const prompts = output(win).querySelectorAll('.ev.prompt');
  assert.ok(prompts.length >= 1, 'the plain prompt renders a Prompt panel');
  win.close();
  console.log('  ok - a prompt without the steer flag is still a Prompt panel');
}

function testMessagePanelNeverAutoFolded() {
  const win = makeWebview();
  startTask(win, 'Never fold the message');
  sendStep(win, 1);
  send(win, {type: 'prompt', text: 'steer one', steer: true});
  for (let i = 2; i <= 5; i++) sendStep(win, i);
  let msg = output(win).querySelector('.ev.user-msg');
  assert.ok(!msg.classList.contains('collapsed'), 'the streaming sweep leaves the message open');
  assert.ok(
    output(win).querySelector('.tc-bash').classList.contains('collapsed'),
    'while the older tool panels fold',
  );
  // A summary tool call adopts the tool panels but leaves the message.
  send(win, {type: 'tool_call', name: 'summary', description: 'five steps'});
  msg = output(win).querySelector('.ev.user-msg');
  assert.ok(!msg.closest('.summary-sub'), 'a summary does not adopt the message');
  assert.strictEqual(msg.parentElement, output(win), 'it stays top-level');
  assert.ok(!msg.classList.contains('collapsed'));
  win.close();

  // A replayed, finished task keeps it open too.
  const win2 = makeWebview();
  const events = [{type: 'prompt', text: 'replayed'}];
  for (let i = 1; i <= 2; i++) {
    events.push({type: 'tool_call', name: 'Bash', command: 'x', description: 'd' + i});
    events.push({type: 'tool_result', name: 'Bash', output: 'y', success: true});
    events.push({type: 'prompt', text: 'steer ' + i, steer: true});
  }
  events.push({type: 'result', text: 'done', success: true});
  send(win2, {type: 'task_events', events, task: 'replayed', tabId: readyTabId(win2), chat_id: 'c-replay'});
  send(win2, {type: 'status', running: false});
  send(win2, {type: 'usage_info', text: 'Steps: 2/10'});
  const msgs = Array.from(output(win2).querySelectorAll('.ev.user-msg'));
  assert.strictEqual(msgs.length, 2, 'both replayed messages render');
  for (const m of msgs) {
    assert.ok(!m.classList.contains('collapsed'), 'a replayed message is open');
    assert.ok(!m.classList.contains('chv-hidden'), 'and not hidden by the digest');
    assert.strictEqual(m.querySelector('.task-panel-h').firstChild.textContent, 'Message');
  }
  assert.ok(
    Array.from(output(win2).querySelectorAll('.tc-bash')).every(p => p.classList.contains('collapsed')),
    'the replay folds the tool panels',
  );
  win2.close();
  console.log('  ok - no automatic pass folds or hides a Message panel');
}

function testMessagePanelFoldsByClick() {
  const win = makeWebview();
  startTask(win, 'Fold by hand');
  send(win, {type: 'prompt', text: 'fold me please', steer: true});
  const msg = output(win).querySelector('.ev.user-msg');
  const hdr = msg.querySelector('.collapse-header');
  hdr.click();
  assert.ok(msg.classList.contains('collapsed'), 'the header click folds it');
  assert.strictEqual(win.getComputedStyle(msg.querySelector('.task-panel-text')).display, 'none', 'the text hides');
  assert.strictEqual(hdr.querySelector('.collapse-preview').textContent, 'fold me please', 'the header previews the message');
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
