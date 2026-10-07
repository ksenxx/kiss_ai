// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// The panels holding the user's words -- the task panel, a steering
// Message, the Prompt and the System Prompt panels -- take four fifths
// of the chat's width, so they read apart from the full-width
// transcript panels (tool calls, Thoughts, results).  The thinking
// tokens inside a Thoughts panel are plain text, not a boxed
// "Thinking" subpanel with a header of its own.

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

function renderTranscript(win) {
  send(win, {
    type: 'task_events',
    events: [],
    task: 'Widen nothing',
    tabId: readyTabId(win),
    chat_id: 'chat-width',
  });
  send(win, {type: 'status', running: true});
  send(win, {type: 'system_prompt', text: 'You are a test.'});
  send(win, {type: 'prompt', text: 'Widen nothing'});
  send(win, {type: 'thinking_start'});
  send(win, {type: 'thinking_delta', text: 'weighing the widths'});
  send(win, {type: 'thinking_end'});
  send(win, {type: 'text_delta', text: 'On it.'});
  send(win, {type: 'text_end'});
  send(win, {type: 'tool_call', name: 'Bash', command: 'ls', description: 'list'});
  send(win, {type: 'tool_result', name: 'Bash', output: 'a', success: true});
  send(win, {type: 'prompt', text: 'also check the margins', steer: true});
}

function testUserPanelsTakeFourFifths() {
  const win = makeWebview();
  renderTranscript(win);
  const out = output(win);
  const userPanels = {
    'task panel': out.querySelector(':scope > .ev.task-panel'),
    'steering Message': out.querySelector(':scope > .ev.user-msg'),
    Prompt: out.querySelector(':scope > .prompt'),
    'System Prompt': out.querySelector(':scope > .system-prompt'),
  };
  for (const [name, panel] of Object.entries(userPanels)) {
    assert.ok(panel, 'the transcript rendered the ' + name);
    const cs = win.getComputedStyle(panel);
    assert.strictEqual(cs.width, '80%', 'the ' + name + ' is 4/5 of the chat wide');
    assert.strictEqual(cs.boxSizing, 'border-box', 'the ' + name + ' width includes its border');
  }
  for (const sel of ['.ev.tc', '.llm-panel']) {
    const panel = out.querySelector(':scope > ' + sel);
    assert.ok(panel, sel + ' rendered');
    assert.notStrictEqual(
      win.getComputedStyle(panel).width,
      '80%',
      sel + ' keeps the full width of the transcript',
    );
  }
  win.close();
  console.log('  ok - task, Message, Prompt and System Prompt panels are 80% wide');
}

function testThinkingIsPlainTextInsideThoughts() {
  const win = makeWebview();
  renderTranscript(win);
  const panel = output(win).querySelector(':scope > .llm-panel');
  const think = panel.querySelector(':scope > .think');
  assert.ok(think, 'the thinking text is a direct child of the Thoughts panel');
  assert.strictEqual(think.textContent, 'weighing the widths');
  assert.strictEqual(think.children.length, 0, 'no header or content box inside it');
  assert.strictEqual(output(win).querySelectorAll('.think .lbl, .think .cnt').length, 0);
  const txt = panel.querySelector(':scope > .txt');
  assert.ok(txt && txt.textContent.includes('On it.'), 'the words follow in the same panel');
  assert.ok(
    think.compareDocumentPosition(txt) & win.Node.DOCUMENT_POSITION_FOLLOWING,
    'the thinking text comes before the words',
  );
  const cs = win.getComputedStyle(think);
  assert.strictEqual(cs.fontStyle, 'italic', 'the thinking text is italic');
  assert.strictEqual(cs.borderStyle, 'none', 'the thinking text draws no box');
  win.close();
  console.log('  ok - thinking tokens are plain text inside the Thoughts panel');
}

testUserPanelsTakeFourFifths();
testThinkingIsPlainTextInsideThoughts();
console.log('userPanelsFourFifthsWidth: all tests passed');
