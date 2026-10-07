// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// Chat webview event panels (user requirements, 2026-10-07):
//   - a Thoughts panel is never collapsed or hidden by any automatic
//     pass (streaming sweep, replay, finished-task digest, summary
//     adoption); only a click on its header folds it;
//   - a collapsed Bash panel's header reads "Bash <description>", with
//     no "description:" label;
//   - a collapsed Read or Write panel's header reads "<Tool> <path>",
//     with no "path:" label;
//   - no collapsible panel header is rendered in uppercase.
// The transcript is rendered by the real media/main.js with
// media/main.css injected; share.js's collapsePreview is checked on the
// serialized DOM the way a shared page toggles it.

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

function topLevel(win) {
  return Array.from(output(win).children).filter(
    el => el.classList && el.classList.contains('collapsible'),
  );
}

function thoughtsPanels(win) {
  return Array.from(output(win).querySelectorAll('.llm-panel'));
}

/** One agent step: a thought, a tool call and its result. */
function sendStep(win, i) {
  send(win, {type: 'thinking_start'});
  send(win, {type: 'thinking_delta', text: 'thought number ' + i});
  send(win, {type: 'thinking_end'});
  send(win, {type: 'text_delta', text: 'words of step ' + i});
  send(win, {type: 'text_end'});
  send(win, {
    type: 'tool_call',
    name: 'Bash',
    command: 'echo ' + i,
    description: 'echo step ' + i,
  });
  send(win, {type: 'tool_result', name: 'Bash', output: String(i), success: true});
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

function assertThoughtsVisible(win, where) {
  const panels = thoughtsPanels(win);
  assert.ok(panels.length > 0, where + ': the transcript has Thoughts panels');
  for (const p of panels) {
    assert.ok(!p.classList.contains('collapsed'), where + ': a Thoughts panel is not collapsed');
    assert.ok(!p.classList.contains('chv-hidden'), where + ': a Thoughts panel is not hidden');
    assert.ok(!p.closest('.summary-sub'), where + ': a Thoughts panel is not adopted by a summary');
    assert.strictEqual(
      win.getComputedStyle(p).display !== 'none',
      true,
      where + ': a Thoughts panel is displayed',
    );
  }
  return panels.length;
}

function testThoughtsNeverCollapsedWhileStreaming() {
  const win = makeWebview();
  startTask(win, 'Keep the thoughts open');
  for (let i = 1; i <= 5; i++) sendStep(win, i);
  const n = assertThoughtsVisible(win, 'streaming');
  assert.ok(n >= 5, 'one Thoughts panel per step: ' + n);
  // The sweep still folds the older tool panels, so it did run.
  const bash = Array.from(output(win).querySelectorAll('.tc-bash'));
  assert.ok(bash.length >= 5);
  assert.ok(
    bash[0].classList.contains('collapsed'),
    'the streaming sweep folds older Bash panels',
  );
  win.close();
  console.log('  ok - Thoughts panels stay open while the task streams');
}

function testSummaryFoldsThoughtsToo() {
  // The one automatic fold a Thoughts panel takes part in: a summary
  // tool call adopts the Thoughts panels of the steps it recounts
  // along with their tool panels (a steering Message, an answer or a
  // Question still stay out in front of it).
  const win = makeWebview();
  startTask(win, 'Summarize around the thoughts');
  for (let i = 1; i <= 3; i++) sendStep(win, i);
  send(win, {type: 'prompt', text: 'steer', steer: true});
  send(win, {type: 'tool_call', name: 'summary', description: 'three steps'});
  const summary = output(win).querySelector('.tc-summary');
  assert.ok(summary.classList.contains('collapsed'), 'the summary panel folds');
  const adopted = Array.from(summary.querySelector('.summary-sub').children);
  const adoptedThoughts = adopted.filter(el => el.classList.contains('llm-panel'));
  assert.strictEqual(adoptedThoughts.length, 3, 'the summary adopts the three Thoughts panels');
  assert.strictEqual(
    adopted.filter(el => el.classList.contains('tc-bash')).length,
    3,
    'the summary adopts the tool panels: ' + adopted.map(e => e.className).join(','),
  );
  const top = topLevel(win);
  assert.strictEqual(top[top.length - 1], summary, 'the summary is the newest panel');
  assert.strictEqual(
    top.filter(p => p.classList.contains('llm-panel')).length,
    0,
    'no Thoughts panel of the recounted steps stays top-level',
  );
  assert.ok(
    top.some(p => p.classList.contains('user-msg')),
    'the steering Message panel stays top-level, in front of the summary',
  );
  win.close();
  console.log('  ok - a summary tool call folds the Thoughts panels of its steps');
}

function testThoughtsNeverCollapsedOnReplayOrDigest() {
  const win = makeWebview();
  const events = [{type: 'prompt', text: 'replayed'}];
  for (let i = 1; i <= 3; i++) {
    events.push({type: 'thinking_start'});
    events.push({type: 'thinking_delta', text: 'replayed thought ' + i});
    events.push({type: 'thinking_end'});
    events.push({
      type: 'tool_call',
      name: 'Read',
      path: '/tmp/file' + i + '.txt',
    });
    events.push({type: 'tool_result', name: 'Read', output: 'x', success: true});
  }
  events.push({type: 'result', text: 'done', success: true});
  send(win, {
    type: 'task_events',
    events,
    task: 'replayed',
    tabId: readyTabId(win),
    chat_id: 'chat-replayed',
  });
  send(win, {type: 'status', running: false});
  send(win, {type: 'usage_info', text: 'Steps: 3/10'});
  assertThoughtsVisible(win, 'replay');
  const reads = Array.from(output(win).querySelectorAll('.tc-path'));
  assert.ok(reads.length === 3 && reads.every(p => p.classList.contains('collapsed')),
    'the replay folds the Read panels');
  win.close();
  console.log('  ok - a replayed finished task keeps its Thoughts panels open');
}

function testThoughtsFoldOnlyByClick() {
  const win = makeWebview();
  startTask(win, 'Fold by hand');
  sendStep(win, 1);
  const p = thoughtsPanels(win)[0];
  const hdr = p.querySelector('.collapse-header');
  hdr.click();
  assert.ok(p.classList.contains('collapsed'), 'the header click folds the Thoughts panel');
  assert.ok(/thought number 1|words of step 1/.test(hdr.querySelector('.collapse-preview').textContent));
  hdr.click();
  assert.ok(!p.classList.contains('collapsed'), 'a second click unfolds it');
  win.close();
  console.log('  ok - only a header click folds a Thoughts panel');
}

/**
 * What a header reads as: its title and its preview span, which the
 * stylesheet sets apart with a margin, joined by one space.
 */
function headerText(hdr) {
  return Array.from(hdr.childNodes)
    .map(n => n.textContent.replace(/\s+/g, ' ').trim())
    .filter(Boolean)
    .join(' ');
}

function collapsedHeaderText(panel) {
  const hdr = panel.querySelector('.collapse-header');
  // The streaming sweep may have folded an older panel already.
  if (!panel.classList.contains('collapsed')) hdr.click();
  assert.ok(panel.classList.contains('collapsed'));
  return headerText(hdr);
}

function testBashCollapsedHeaderIsDescription() {
  const win = makeWebview();
  startTask(win, 'Bash header');
  send(win, {
    type: 'tool_call',
    name: 'Bash',
    command: 'ls -la /tmp && echo done',
    description: 'List the temp dir',
  });
  const panel = output(win).querySelector('.tc-bash');
  const expanded = headerText(panel.querySelector('.tc-h'));
  assert.strictEqual(expanded, 'Bash', 'expanded: the header is the tool name');
  assert.ok(
    /description:\s*List the temp dir/.test(panel.querySelector('.tc-b').textContent),
    'expanded: the body still labels the description',
  );
  const txt = collapsedHeaderText(panel);
  assert.strictEqual(txt, 'Bash List the temp dir', 'collapsed: "Bash <description>"');
  assert.ok(!/description/.test(txt), 'no "description:" label in the header');
  assert.ok(!/ls -la/.test(txt), 'the command is not in the header');
  win.close();
  console.log('  ok - a collapsed Bash panel reads "Bash <description>"');
}

function testBashWithoutDescriptionPreviewsBody() {
  const win = makeWebview();
  startTask(win, 'Bash no description');
  send(win, {type: 'tool_call', name: 'Bash', command: 'uptime'});
  const panel = output(win).querySelector('.tc-bash');
  const txt = collapsedHeaderText(panel);
  assert.strictEqual(txt, 'Bash uptime', 'without a description the body is the preview');
  win.close();
  console.log('  ok - a Bash panel without description previews its command');
}

function testReadWriteCollapsedHeaderIsPath() {
  const win = makeWebview();
  startTask(win, 'Read Write headers');
  send(win, {type: 'tool_call', name: 'Read', path: '/home/u/proj/src/app.py'});
  send(win, {
    type: 'tool_call',
    name: 'Write',
    path: '/home/u/proj/src/new.py',
    content: 'print("hello")\n',
    lang: 'python',
  });
  send(win, {
    type: 'tool_call',
    name: 'Edit',
    path: '/home/u/proj/src/old.py',
    old_string: 'a',
    new_string: 'b',
  });
  const panels = Array.from(output(win).querySelectorAll('.tc'));
  assert.strictEqual(panels.length, 3);
  const [read, write, edit] = panels;
  assert.strictEqual(collapsedHeaderText(read), 'Read /home/u/proj/src/app.py');
  assert.strictEqual(collapsedHeaderText(write), 'Write /home/u/proj/src/new.py');
  for (const p of [read, write]) {
    const txt = p.querySelector('.collapse-header').textContent;
    assert.ok(!/path:/i.test(txt), 'no "path:" label in the header: ' + txt);
    assert.ok(
      /path:/.test(p.querySelector('.tc-b').textContent),
      'the expanded body still labels the path',
    );
    assert.ok(p.querySelector('.tc-arg-path .tp'), 'the path stays a clickable .tp');
  }
  // Other tools keep the whole-body preview.
  const editTxt = collapsedHeaderText(edit);
  assert.ok(/^Edit path: \/home\/u\/proj\/src\/old\.py/.test(editTxt), editTxt);
  win.close();
  console.log('  ok - collapsed Read/Write panels read "<Tool> <path>"');
}

function testSharePagePreviewMatches() {
  const win = makeWebview();
  startTask(win, 'Share preview');
  send(win, {type: 'tool_call', name: 'Bash', command: 'ls', description: 'list files'});
  send(win, {type: 'tool_call', name: 'Write', path: '/tmp/w.txt', content: 'x'});
  send(win, {type: 'tool_call', name: 'Edit', path: '/tmp/e.txt', old_string: 'a', new_string: 'b'});
  // A shared page is the serialized transcript, re-driven by share.js.
  const html = '<!doctype html><html><body><div id="output">' +
    output(win).innerHTML + '</div></body></html>';
  win.close();
  const dom = new JSDOM(html, {runScripts: 'dangerously', url: 'https://localhost/'});
  const sw = dom.window;
  sw.eval(fs.readFileSync(path.join(MEDIA, 'share.js'), 'utf8'));
  const panels = Array.from(sw.document.querySelectorAll('.tc'));
  assert.strictEqual(panels.length, 3);
  const expect = ['Bash list files', 'Write /tmp/w.txt', 'Edit path: /tmp/e.txt - a + b'];
  panels.forEach((p, i) => {
    const hdr = p.querySelector('.collapse-header');
    // Open a panel the live sweep had folded, then fold it again so the
    // preview is share.js's own, not the serialized one.
    if (p.classList.contains('collapsed')) hdr.click();
    hdr.querySelector('.collapse-preview').textContent = 'stale';
    hdr.click();
    assert.ok(p.classList.contains('collapsed'), 'share.js folds the panel');
    const txt = headerText(hdr);
    assert.strictEqual(txt, expect[i], 'shared page header ' + i);
  });
  sw.close();
  console.log('  ok - a shared page folds panels to the same headers');
}

function testHeadersAreMixedCase() {
  const win = makeWebview();
  startTask(win, 'Mixed case headers');
  send(win, {type: 'system_prompt', text: 'You are a test.'});
  send(win, {type: 'ask_answer', question: 'q?', answer: 'a.'});
  sendStep(win, 1);
  send(win, {type: 'tool_call', name: 'Bash', command: 'false', description: 'fail'});
  send(win, {type: 'tool_result', name: 'Bash', content: 'boom', is_error: true});
  send(win, {type: 'tool_call', name: 'ask_user_question', extras: {question: 'why?'}});
  const sels = [
    '.tc-h',
    '.llm-panel-hdr',
    '.system-prompt-h',
    '.prompt-h',
    '.task-panel-h',
    '.ask-answer-label',
    '.tr .rl',
  ];
  for (const sel of sels) {
    const els = output(win).querySelectorAll(sel);
    assert.ok(els.length > 0, sel + ' rendered');
    for (const el of els) {
      const tt = win.getComputedStyle(el).getPropertyValue('text-transform').trim();
      assert.ok(tt === '' || tt === 'none', sel + ' is not transformed: ' + JSON.stringify(tt));
    }
  }
  const names = Array.from(output(win).querySelectorAll('.tc-h')).map(h =>
    h.firstChild.textContent.trim(),
  );
  assert.deepStrictEqual(names, ['Bash', 'Bash', 'Question'], 'tool names keep their own case');
  assert.strictEqual(output(win).querySelector('.llm-panel-hdr').firstChild.textContent, 'Thoughts');
  win.close();
  console.log('  ok - collapsible panel headers are mixed case');
}

function main() {
  testThoughtsNeverCollapsedWhileStreaming();
  testSummaryFoldsThoughtsToo();
  testThoughtsNeverCollapsedOnReplayOrDigest();
  testThoughtsFoldOnlyByClick();
  testBashCollapsedHeaderIsDescription();
  testBashWithoutDescriptionPreviewsBody();
  testReadWriteCollapsedHeaderIsPath();
  testSharePagePreviewMatches();
  testHeadersAreMixedCase();
  console.log('panelHeadersBriefMixedCase: all tests passed');
}

main();
