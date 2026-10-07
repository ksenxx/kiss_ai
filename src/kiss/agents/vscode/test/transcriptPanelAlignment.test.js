// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// The panels holding the user's own words -- the task panel and a
// steering Message -- are right-justified at four fifths of the chat's
// width; every other transcript panel (Prompt, System Prompt, tool
// calls, Thoughts, results, status lines) is left-justified at seven
// eighths, so the thread reads as a conversation.  A panel nested
// inside another (a tool error under its tool call) fills its parent.
// The thinking tokens inside a Thoughts panel are plain text, not a
// boxed "Thinking" subpanel with a header of its own.

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

function testUserPanelsRightAgentPanelsLeft() {
  const win = makeWebview();
  renderTranscript(win);
  const out = output(win);
  const userPanels = {
    'task panel': out.querySelector(':scope > .ev.task-panel'),
    'steering Message': out.querySelector(':scope > .ev.user-msg'),
  };
  for (const [name, panel] of Object.entries(userPanels)) {
    assert.ok(panel, 'the transcript rendered the ' + name);
    const cs = win.getComputedStyle(panel);
    assert.strictEqual(cs.width, '80%', 'the ' + name + ' is 4/5 of the chat wide');
    assert.strictEqual(cs.marginLeft, 'auto', 'the ' + name + ' is pushed to the right edge');
    assert.notStrictEqual(cs.marginRight, 'auto', 'the ' + name + ' touches the right edge');
    assert.strictEqual(cs.boxSizing, 'border-box', 'the ' + name + ' width includes its border');
  }
  const agentPanels = {
    Prompt: out.querySelector(':scope > .ev.prompt'),
    'System Prompt': out.querySelector(':scope > .ev.system-prompt'),
    'tool call': out.querySelector(':scope > .ev.tc'),
    Thoughts: out.querySelector(':scope > .llm-panel'),
  };
  for (const [name, panel] of Object.entries(agentPanels)) {
    assert.ok(panel, 'the transcript rendered the ' + name);
    const cs = win.getComputedStyle(panel);
    assert.strictEqual(cs.width, '87.5%', 'the ' + name + ' is 7/8 of the chat wide');
    assert.notStrictEqual(cs.marginLeft, 'auto', 'the ' + name + ' stays on the left edge');
    assert.strictEqual(cs.boxSizing, 'border-box', 'the ' + name + ' width includes its border');
  }
  win.close();
  console.log('  ok - user panels are right-justified at 80%, the rest left at 87.5%');
}

function testNestedPanelFillsItsParent() {
  const win = makeWebview();
  renderTranscript(win);
  send(win, {type: 'tool_call', name: 'Bash', command: 'false', description: 'fail'});
  send(win, {type: 'tool_result', name: 'Bash', content: 'boom', is_error: true});
  const out = output(win);
  const nested = out.querySelector(':scope > .ev.tc .ev.tr');
  assert.ok(nested, 'the tool error renders inside its tool call panel');
  assert.strictEqual(
    win.getComputedStyle(nested).width,
    'auto',
    'a panel nested in another fills its parent instead of 7/8 of it',
  );
  win.close();
  console.log('  ok - a nested panel fills its parent');
}

function testSummaryAdoptedPanelsFillTheSummary() {
  const win = makeWebview();
  renderTranscript(win);
  send(win, {type: 'tool_call', name: 'summary', description: 'Progress so far'});
  const out = output(win);
  const sub = out.querySelector(':scope > .ev.tc-summary > .summary-sub');
  assert.ok(sub, 'the summary adopted its neighbours into .summary-sub');
  const adopted = {
    Thoughts: sub.querySelector(':scope > .llm-panel'),
    'tool call': sub.querySelector(':scope > .ev.tc'),
  };
  for (const [name, panel] of Object.entries(adopted)) {
    assert.ok(panel, 'the summary adopted the ' + name);
    assert.strictEqual(
      win.getComputedStyle(panel).width,
      'auto',
      'the adopted ' + name + ' fills the summary instead of 7/8 of it',
    );
  }
  win.close();
  console.log('  ok - panels a summary adopts fill the summary');
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

testUserPanelsRightAgentPanelsLeft();
testNestedPanelFillsItsParent();
testSummaryAdoptedPanelsFillTheSummary();
testThinkingIsPlainTextInsideThoughts();
console.log('transcriptPanelAlignment: all tests passed');
