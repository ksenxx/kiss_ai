// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// Chat page event panels (user requirement, 2026-10-12): the
// run_agent, run_parallel and run_commands_parallel tools take a
// `description` argument, and a collapsed panel of any of them reads
// "<tool> <description>" in its header, exactly like a Bash panel.
// Without a description the panel keeps the whole-body preview.  The
// transcript is rendered by the real media/main.js; share.js's
// collapsePreview is checked on the serialized DOM the way a shared
// page toggles it.

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');
const {
  makePage: makeChatPage,
  send,
  loadMainCss,
  MEDIA,
  readyTabId,
  output,
} = require('./simplify2_harness');

function makePage() {
  const {win, posted} = makeChatPage({modelName: ''});
  loadMainCss(win);
  win.__posted = posted;
  return win;
}

function startTask(win, task) {
  send(win, {
    type: 'task_events',
    events: [],
    task,
    tabId: readyTabId(win.__posted),
    chat_id: 'chat-' + task.replace(/\W+/g, '-'),
  });
  send(win, {type: 'status', running: true});
  send(win, {type: 'prompt', text: task});
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
  if (!panel.classList.contains('collapsed')) hdr.click();
  assert.ok(panel.classList.contains('collapsed'));
  return headerText(hdr);
}

/** The three fan-out / batch tool calls, as the daemon's tool_call events. */
const CALLS = [
  {
    name: 'run_agent',
    extras: {task: 'Send hello to #sorcar', agent: 'slack'},
    description: 'Post the greeting to Slack',
  },
  {
    name: 'run_parallel',
    extras: {
      tasks: '["review the diff", "check the tests"]',
      tool_profile: 'review',
    },
    description: 'Review the change from two angles',
  },
  {
    name: 'run_commands_parallel',
    extras: {
      commands: '["uv run pytest tests/a.py", "uv run pytest tests/b.py"]',
    },
    description: 'Run the two test splits',
  },
];

function testCollapsedHeaderIsDescription() {
  const win = makePage();
  startTask(win, 'Description headers');
  for (const call of CALLS) send(win, Object.assign({type: 'tool_call'}, call));
  const panels = Array.from(output(win).querySelectorAll('.tc'));
  assert.strictEqual(panels.length, CALLS.length, 'one panel per call');
  panels.forEach((panel, i) => {
    const call = CALLS[i];
    assert.ok(
      panel.classList.contains('tc-desc'),
      call.name + ': carries tc-desc',
    );
    // A fan-out panel is born open (its open panel keeps the sub-agent
    // tabs open); collapsedHeaderText folds whatever is still open.
    const txt = collapsedHeaderText(panel);
    assert.strictEqual(
      txt,
      call.name + ' ' + call.description,
      call.name + ': collapsed header is "<tool> <description>"',
    );
    assert.ok(
      !/description/.test(txt),
      call.name + ': no "description:" label in the header',
    );
    const argKey = Object.keys(call.extras)[0];
    assert.ok(
      !txt.includes(call.extras[argKey]),
      call.name + ': the ' + argKey + ' argument is not in the header',
    );
    // Unfolded, the header is the tool name and the body labels every argument.
    panel.querySelector('.tc-h').click();
    assert.ok(
      !panel.classList.contains('collapsed'),
      call.name + ': a header click unfolds it',
    );
    assert.strictEqual(headerText(panel.querySelector('.tc-h')), call.name);
    const body = panel.querySelector('.tc-b').textContent;
    assert.ok(
      new RegExp('description:\\s*' + call.description).test(body),
      call.name + ': expanded body labels the description',
    );
    assert.ok(
      body.includes(call.extras[argKey]),
      call.name + ': expanded body shows ' + argKey,
    );
  });
  win.close();
  console.log(
    '  ok - collapsed run_agent/run_parallel/run_commands_parallel headers read "<tool> <description>"',
  );
}

function testWithoutDescriptionPreviewsBody() {
  const win = makePage();
  startTask(win, 'No description');
  send(win, {
    type: 'tool_call',
    name: 'run_commands_parallel',
    extras: {commands: '["make lint"]'},
  });
  const panel = output(win).querySelector('.tc');
  assert.ok(!panel.classList.contains('tc-desc'), 'no description, no tc-desc');
  const txt = collapsedHeaderText(panel);
  assert.strictEqual(
    txt,
    'run_commands_parallel commands: ["make lint"]',
    'without a description the body is the preview',
  );
  win.close();
  console.log('  ok - a panel without description previews its body');
}

// A tool whose `description` argument is content, not a title: the
// Google Calendar SEA's gcal_update_event takes the event's body text.
// Its folded panel keeps the whole-body preview.
const GCAL_CALL = {
  name: 'gcal_update_event',
  extras: {event_id: 'event-123', summary: 'Renamed meeting'},
  description: 'Discuss revised agenda',
};
// The body renders the description argument before the extras.
const GCAL_HEADER =
  'gcal_update_event description: Discuss revised agenda ' +
  'event_id: event-123 summary: Renamed meeting';

function testOtherToolsDescriptionIsNotATitle() {
  const win = makePage();
  startTask(win, 'Calendar description');
  send(win, Object.assign({type: 'tool_call'}, GCAL_CALL));
  const panel = output(win).querySelector('.tc');
  assert.ok(
    !panel.classList.contains('tc-desc'),
    'a calendar panel carries no tc-desc',
  );
  assert.strictEqual(
    collapsedHeaderText(panel),
    GCAL_HEADER,
    'whole-body preview',
  );
  win.close();
  console.log(
    "  ok - a calendar tool's description stays a body argument, not the header",
  );
}

function testSharePagePreviewMatches() {
  const win = makePage();
  startTask(win, 'Share preview');
  for (const call of CALLS) send(win, Object.assign({type: 'tool_call'}, call));
  send(win, {
    type: 'tool_call',
    name: 'Bash',
    command: 'ls',
    description: 'list files',
  });
  send(win, Object.assign({type: 'tool_call'}, GCAL_CALL));
  const html =
    '<!doctype html><html><body><div id="output">' +
    output(win).innerHTML +
    '</div></body></html>';
  win.close();
  const dom = new JSDOM(html, {
    runScripts: 'dangerously',
    url: 'https://localhost/',
  });
  const sw = dom.window;
  sw.eval(fs.readFileSync(path.join(MEDIA, 'share.js'), 'utf8'));
  const panels = Array.from(sw.document.querySelectorAll('.tc'));
  assert.strictEqual(panels.length, CALLS.length + 2);
  const expect = CALLS.map(c => c.name + ' ' + c.description).concat([
    'Bash list files',
    GCAL_HEADER,
  ]);
  panels.forEach((p, i) => {
    const hdr = p.querySelector('.collapse-header');
    if (p.classList.contains('collapsed')) hdr.click();
    hdr.querySelector('.collapse-preview').textContent = 'stale';
    hdr.click();
    assert.ok(p.classList.contains('collapsed'), 'share.js folds the panel');
    assert.strictEqual(headerText(hdr), expect[i], 'shared page header ' + i);
  });
  sw.close();
  console.log('  ok - a shared page folds the panels to the same headers');
}

function main() {
  testCollapsedHeaderIsDescription();
  testWithoutDescriptionPreviewsBody();
  testOtherToolsDescriptionIsNotATitle();
  testSharePagePreviewMatches();
  console.log('descriptionHeaders: all tests passed');
}

main();
