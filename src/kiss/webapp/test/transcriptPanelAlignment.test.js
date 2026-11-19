// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// The panels holding the user's own words -- the task panel and a
// steering Message -- are bubbles: as wide as their text (at most 85%
// of the chat column) against the column's right edge.  Every other
// transcript panel (Prompt, System Prompt, tool calls, Thoughts,
// results, status lines) runs the column's full width from its left
// edge, so the thread reads as a conversation.  A panel nested inside
// another (a tool error under its tool call) fills its parent.  The
// thinking tokens inside a Thoughts panel are plain text, not a boxed
// "Thinking" subpanel with a header of its own.

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const harness = require('./simplify2_harness');
const {installTokenisedCss, lastCssRule} = require('./designTokens');
const {readyTabId, output} = harness;

const MAIN_CSS = fs.readFileSync(
  path.join(__dirname, '..', 'media', 'main.css'),
  'utf8',
);

/**
 * The harness page with main.css applied; `win.__posted` is its ledger.
 * The design tokens are inlined: jsdom drops the longhands of a
 * shorthand that holds a var() (the bubble's `margin`), so the spacing
 * and radius tokens are read as their concrete values.
 */
function makePage() {
  const {win, posted} = harness.makePage();
  installTokenisedCss(win, 'main.css');
  win.__posted = posted;
  return win;
}

const send = harness.send;

function renderTranscript(win) {
  send(win, {
    type: 'task_events',
    events: [],
    task: 'Widen nothing',
    tabId: readyTabId(win.__posted),
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
  send(win, {
    type: 'tool_call',
    name: 'Bash',
    command: 'ls',
    description: 'list',
  });
  send(win, {type: 'tool_result', name: 'Bash', output: 'a', success: true});
  send(win, {type: 'prompt', text: 'also check the margins', steer: true});
}

function testUserPanelsRightAgentPanelsLeft() {
  const win = makePage();
  renderTranscript(win);
  const out = output(win);
  const userPanels = {
    'task panel': out.querySelector(':scope > .ev.task-panel'),
    'steering Message': out.querySelector(':scope > .ev.user-msg'),
  };
  for (const [name, panel] of Object.entries(userPanels)) {
    assert.ok(panel, 'the transcript rendered the ' + name);
    const cs = win.getComputedStyle(panel);
    assert.strictEqual(
      cs.width,
      'fit-content',
      'the ' + name + ' is as wide as its text',
    );
    assert.strictEqual(
      cs.maxWidth,
      '85%',
      'the ' + name + ' leaves the left of the chat column clear',
    );
    assert.strictEqual(
      cs.minWidth,
      '120px',
      'the ' + name + ' keeps its label and copy button apart',
    );
    assert.strictEqual(
      cs.marginLeft,
      'auto',
      'the ' + name + ' is pushed to the right edge',
    );
    assert.strictEqual(
      cs.marginRight,
      '0px',
      'the ' + name + ' touches the right edge',
    );
    assert.strictEqual(
      cs.boxSizing,
      'border-box',
      'the ' + name + ' width includes its border',
    );
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
    assert.strictEqual(
      cs.width,
      'auto',
      'the ' + name + ' runs the full width of the chat column',
    );
    assert.strictEqual(cs.maxWidth, 'none', 'the ' + name + ' has no cap');
    assert.strictEqual(
      cs.marginLeft,
      '0px',
      'the ' + name + ' starts at the left edge',
    );
    assert.strictEqual(
      cs.marginRight,
      '0px',
      'the ' + name + ' ends at the right edge',
    );
    assert.strictEqual(
      cs.boxSizing,
      'border-box',
      'the ' + name + ' width includes its border',
    );
  }
  win.close();
  console.log(
    '  ok - user panels are right-justified bubbles, the rest full width',
  );
}

function testNestedPanelFillsItsParent() {
  const win = makePage();
  renderTranscript(win);
  send(win, {
    type: 'tool_call',
    name: 'Bash',
    command: 'false',
    description: 'fail',
  });
  send(win, {
    type: 'tool_result',
    name: 'Bash',
    content: 'boom',
    is_error: true,
  });
  const out = output(win);
  const nested = out.querySelector(':scope > .ev.tc .ev.tr');
  assert.ok(nested, 'the tool error renders inside its tool call panel');
  assert.strictEqual(
    win.getComputedStyle(nested).width,
    'auto',
    'a panel nested in another fills its parent',
  );
  win.close();
  console.log('  ok - a nested panel fills its parent');
}

function testSummaryAdoptedPanelsFillTheSummary() {
  const win = makePage();
  renderTranscript(win);
  send(win, {
    type: 'tool_call',
    name: 'summary',
    description: 'Progress so far',
  });
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
      'the adopted ' + name + ' fills the summary',
    );
  }
  win.close();
  console.log('  ok - panels a summary adopts fill the summary');
}

function testThinkingIsPlainTextInsideThoughts() {
  const win = makePage();
  renderTranscript(win);
  const panel = output(win).querySelector(':scope > .llm-panel');
  const think = panel.querySelector(':scope > .think');
  assert.ok(think, 'the thinking text is a direct child of the Thoughts panel');
  assert.strictEqual(think.textContent, 'weighing the widths');
  assert.strictEqual(
    think.children.length,
    0,
    'no header or content box inside it',
  );
  assert.strictEqual(
    output(win).querySelectorAll('.think .lbl, .think .cnt').length,
    0,
  );
  const txt = panel.querySelector(':scope > .txt');
  assert.ok(
    txt && txt.textContent.includes('On it.'),
    'the words follow in the same panel',
  );
  assert.ok(
    think.compareDocumentPosition(txt) & win.Node.DOCUMENT_POSITION_FOLLOWING,
    'the thinking text comes before the words',
  );
  const cs = win.getComputedStyle(think);
  assert.strictEqual(cs.fontStyle, 'italic', 'the thinking text is italic');
  assert.strictEqual(cs.borderStyle, 'none', 'the thinking text draws no box');
  // The Thoughts panel is the one scroller: the thinking text has no
  // height cap and no scrollbar of its own.
  assert.ok(
    ['', 'none'].includes(cs.maxHeight),
    `.think is no scroller of its own (max-height ${cs.maxHeight})`,
  );
  assert.ok(
    !['auto', 'scroll'].includes(cs.overflowY),
    `.think is no scroller of its own (overflow-y ${cs.overflowY})`,
  );
  win.close();
  console.log(
    '  ok - thinking tokens are plain text inside the Thoughts panel',
  );
}

// Only the user's words sit in a bubble: a neutral wash
// (--bubble-user-tint) with one 10px corner all round and no outline.
// The agent's side is plain on the canvas: a tool call is a line with
// no fill and no frame, and the result is prose under a hairline.
// JSDOM hands back the unresolved var() names of the colours, which is
// what we compare.
function testOnlyTheUserPanelsAreBubbles() {
  const win = makePage();
  renderTranscript(win);
  send(win, {
    type: 'result',
    summary: 'All done.',
    success: true,
    total_tokens: 10,
    cost: '0.01',
  });
  const out = output(win);
  const userPanels = {
    'task panel': out.querySelector(':scope > .ev.task-panel'),
    'steering Message': out.querySelector(':scope > .ev.user-msg'),
  };
  for (const [name, panel] of Object.entries(userPanels)) {
    assert.ok(panel, 'the transcript rendered the ' + name);
    const cs = win.getComputedStyle(panel);
    assert.strictEqual(
      cs.background,
      'var(--bubble-user-tint)',
      'the ' + name + ' wears the user bubble tint',
    );
    assert.strictEqual(
      cs.borderRadius,
      '10px',
      'the ' + name + ' has one 10px corner all round',
    );
    assert.strictEqual(
      cs.borderTopStyle,
      'none',
      'the ' + name + ' draws no outline',
    );
  }
  const taskText = userPanels['task panel'].querySelector('.task-panel-text');
  assert.strictEqual(
    win.getComputedStyle(taskText).color,
    'var(--fg)',
    'the task text is the transcript foreground, not an inverted palette',
  );
  const result = out.querySelector(':scope > .ev.rc');
  assert.ok(result, 'the transcript rendered the result card');
  const rcs = win.getComputedStyle(result);
  assert.strictEqual(
    rcs.backgroundColor,
    'rgba(0, 0, 0, 0)',
    'the result is prose on the canvas, with no fill of its own',
  );
  assert.strictEqual(rcs.borderRadius, '', 'the result has no card corner');
  // jsdom cannot read back a border shorthand with a var() colour, so
  // the hairline over the result is read from the stylesheet text.
  assert.match(
    lastCssRule(MAIN_CSS, '.rc'),
    /(?:^|[;\s])border-top:\s*1px solid var\(--border\)\s*;/,
    'a hairline sets the result off from the tool lines above it',
  );
  assert.strictEqual(
    rcs.width,
    'auto',
    'the result runs the full width of the chat column',
  );
  assert.strictEqual(
    rcs.marginLeft,
    '0px',
    'the result starts at the left edge',
  );
  const tool = out.querySelector(':scope > .ev.tc');
  const tcs = win.getComputedStyle(tool);
  assert.strictEqual(
    tcs.backgroundColor,
    'rgba(0, 0, 0, 0)',
    'a tool call is a line on the canvas, with no fill of its own',
  );
  assert.strictEqual(tcs.borderTopStyle, 'none', 'a tool call has no frame');
  assert.strictEqual(tcs.borderRadius, '', 'a tool call has no card corner');
  // Both tints resolve on this main.css page and on a remote page (whose
  // remote-codex.css re-derives them for its themes).
  const remote = harness.makePage({
    bodyClass: 'remote-chat',
    beforeScripts: harness.loadRemoteCss,
  }).win;
  for (const page of [win, remote]) {
    const root = page.getComputedStyle(page.document.documentElement);
    for (const token of ['--bubble-user-tint', '--panel-tint']) {
      assert.ok(
        root.getPropertyValue(token).includes('color-mix('),
        `${token} is a defined colour mix on the page`,
      );
    }
  }
  remote.close();
  win.close();
  console.log(
    '  ok - the user panels are bubbles, the agent panels plain on the canvas',
  );
}

testUserPanelsRightAgentPanelsLeft();
testOnlyTheUserPanelsAreBubbles();
testNestedPanelFillsItsParent();
testSummaryAdoptedPanelsFillTheSummary();
testThinkingIsPlainTextInsideThoughts();
console.log('transcriptPanelAlignment: all tests passed');
