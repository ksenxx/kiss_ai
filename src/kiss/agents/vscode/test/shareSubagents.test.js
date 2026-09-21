// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end tests for the shared chat page's sub-agent tabs and for
// the saved-page notification:
//
// * the share export (media/main.js buildShareableHtml) renders every
//   sub-agent of the `share_tasks` reply — grandchildren included —
//   into hidden .share-subagent sections and stamps each run_parallel
//   panel with the task ids it fanned out (data-rp-subagents),
// * on the exported page, share.js re-creates the webview's sub-agent
//   tab behaviour: expanded fan-outs open their tabs on load,
//   collapsing a fan-out closes its tabs, expanding reopens them, a
//   hand-closed tab stays closed until a collapse / expand cycle,
//   closing a tab takes its descendants' tabs with it, and selecting
//   a tab swaps the transcript on screen, and
// * a successful share_done reply raises a "Chat page saved to
//   <path>" toast in the webview.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

let passed = 0;
const failures = [];

async function test(name, fn) {
  try {
    await fn();
    passed++;
    console.log(`  \u2713 ${name}`);
  } catch (e) {
    failures.push({name, error: e});
    console.log(`  \u2717 ${name}`);
    console.log(`      ${e.message}`);
  }
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
  win.requestAnimationFrame = function (cb) {
    cb();
    return 0;
  };
  win.cancelAnimationFrame = function () {};

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
  win.eval(
    fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8') +
      '\n//# sourceURL=sharesub-main.js',
  );

  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function tabIdOf(wv) {
  const ready = wv.posted.find(m => m.type === 'ready');
  assert.ok(ready && ready.tabId, 'webview must post ready with a tabId');
  return ready.tabId;
}

function click(el) {
  el.dispatchEvent(
    new el.ownerDocument.defaultView.MouseEvent('click', {
      bubbles: true,
      cancelable: true,
    }),
  );
}

/**
 * Drive a task through the webview whose last panel is a run_parallel
 * fan-out declaring *taskDescs*, so the live transcript ends with an
 * EXPANDED fan-out panel exactly like a live run's screen.
 */
function runFanoutTask(wv, chatId, taskId, taskDescs) {
  const win = wv.win;
  const TAB = tabIdOf(wv);
  const now = Date.now();
  send(win, {type: 'clear', chat_id: chatId, tabId: TAB});
  send(win, {type: 'status', running: true, tabId: TAB, startTs: now});
  send(win, {type: 'setTaskText', text: 'fan out the work', tabId: TAB});
  send(win, {
    type: 'tool_call',
    name: 'run_parallel',
    extras: {tasks: JSON.stringify(taskDescs)},
    tabId: TAB,
    taskId: taskId,
    ts: now,
  });
  return TAB;
}

/**
 * Click the share button, answer the webview's shareChatTasks request
 * with *tasks*, and return the resulting shareChat command.
 */
function shareWithTasks(wv, tasks) {
  const win = wv.win;
  click(win.document.getElementById('share-btn'));
  const req = wv.posted.filter(m => m.type === 'shareChatTasks').pop();
  assert.ok(req, 'clicking share must ask the daemon for the chat tasks');
  send(win, {
    type: 'share_tasks',
    tabId: req.tabId,
    chatId: req.chatId,
    tasks: tasks || [],
    truncated: false,
  });
  return wv.posted.filter(m => m.type === 'shareChat').pop();
}

/**
 * Load share.js into a fresh page whose <body> is *bodyHtml* — the
 * document shape _build_share_page (web_server.py) writes.
 */
function makeSharePage(bodyHtml) {
  const dom = new JSDOM(
    '<!DOCTYPE html><html><body><div id="app">' +
      bodyHtml +
      '</div></body></html>',
    {runScripts: 'dangerously', pretendToBeVisual: true},
  );
  const win = dom.window;
  win.scrollTo = function () {};
  win.eval(
    fs.readFileSync(path.join(MEDIA, 'share.js'), 'utf8') +
      '\n//# sourceURL=sharesub-share.js',
  );
  return win;
}

const SUB_EVENTS_ONE = [
  {
    type: 'tool_call',
    name: 'run_parallel',
    extras: {tasks: JSON.stringify(['grand sub task'])},
  },
];
const SUB_EVENTS_TWO = [
  {type: 'tool_call', name: 'Bash', command: 'echo sub-two'},
  {type: 'tool_result', content: 'sub-two done', tool_name: 'Bash'},
];
const GRAND_EVENTS = [
  {type: 'tool_call', name: 'Bash', command: 'echo grand'},
  {type: 'tool_result', content: 'grand done', tool_name: 'Bash'},
];

/** One chat task with two sub-agents; the first ran a fan-out of its
 * own with one grandchild. */
function fanoutTaskEntry(taskId) {
  return {
    task: 'fan out the work',
    task_id: taskId,
    events: [],
    subagents: [
      {
        task: 'first sub task',
        task_id: 'sub-1',
        parent_task_id: taskId,
        events: SUB_EVENTS_ONE,
      },
      {
        task: 'second sub task',
        task_id: 'sub-2',
        parent_task_id: taskId,
        events: SUB_EVENTS_TWO,
      },
      {
        task: 'grand sub task',
        task_id: 'grand-1',
        parent_task_id: 'sub-1',
        events: GRAND_EVENTS,
      },
    ],
  };
}

/** Export the fan-out chat and return the shared page body html. */
function exportFanoutChat() {
  const wv = makeWebview();
  runFanoutTask(wv, 'chat-sub', 'task-live', [
    'first sub task',
    'second sub task',
  ]);
  const msg = shareWithTasks(wv, [fanoutTaskEntry('task-live')]);
  assert.ok(msg, 'the share_tasks reply must produce a shareChat command');
  return msg.html;
}

function tabsOf(win) {
  return Array.from(win.document.querySelectorAll('#tab-list .chat-tab'));
}

function tabLabels(win) {
  return tabsOf(win).map(
    el => el.querySelector('.chat-tab-label').textContent,
  );
}

function sectionByTask(win, taskId) {
  return win.document.querySelector(
    '.share-subagent[data-task-id="' + taskId + '"]',
  );
}

function fanoutPanel(win, root) {
  return (root || win.document).querySelector(
    '.tc-run-parallel[data-rp-subagents]',
  );
}

async function run() {
  await test('the export holds one hidden section per sub-agent', () => {
    const html = exportFanoutChat();
    const dom = new JSDOM('<div id="app">' + html + '</div>');
    const doc = dom.window.document;
    const sections = doc.querySelectorAll('.share-task.share-subagent');
    assert.strictEqual(sections.length, 3, 'two children plus a grandchild');
    for (const s of sections) {
      assert.ok(s.hasAttribute('hidden'), 'sub-agent sections start hidden');
    }
    const byId = {};
    for (const s of sections) byId[s.getAttribute('data-task-id')] = s;
    assert.ok(byId['sub-1'] && byId['sub-2'] && byId['grand-1']);
    assert.strictEqual(
      byId['sub-1'].getAttribute('data-parent-task-id'),
      'task-live',
    );
    assert.strictEqual(
      byId['grand-1'].getAttribute('data-parent-task-id'),
      'sub-1',
    );
    assert.strictEqual(
      byId['sub-1'].getAttribute('data-sub-title'),
      '1. first sub task',
      'the tab title matches the live webview\'s "N. description"',
    );
    assert.strictEqual(
      byId['sub-2'].getAttribute('data-sub-title'),
      '2. second sub task',
    );
    assert.strictEqual(
      byId['grand-1'].getAttribute('data-sub-title'),
      '1. grand sub task',
    );
    assert.ok(
      byId['sub-2'].textContent.includes('echo sub-two'),
      "the sub-agent's transcript is rendered into its section",
    );
    assert.ok(
      byId['sub-1'].querySelector('[id="task-panel"]'),
      'each sub-agent section carries its static task panel',
    );
    assert.strictEqual(
      byId['sub-1'].querySelector('[id^="task-panel-text"]').textContent,
      'first sub task',
      "the panel shows the sub-agent's own task text",
    );
  });

  await test('fan-out panels are stamped with their sub-agent ids', () => {
    const html = exportFanoutChat();
    const dom = new JSDOM('<div id="app">' + html + '</div>');
    const doc = dom.window.document;
    const taskSection = doc.querySelector(
      '.share-task:not(.share-subagent)',
    );
    const livePanel = taskSection.querySelector('.tc-run-parallel');
    assert.strictEqual(
      livePanel.getAttribute('data-rp-subagents'),
      'sub-1 sub-2',
      "the live task's fan-out names its two sub-agents",
    );
    const subSection = doc.querySelector(
      '.share-subagent[data-task-id="sub-1"]',
    );
    const grandPanel = subSection.querySelector('.tc-run-parallel');
    assert.strictEqual(
      grandPanel.getAttribute('data-rp-subagents'),
      'grand-1',
      "the sub-agent's own fan-out names its grandchild",
    );
  });

  await test('a replayed (non-live) task is stamped the same way', () => {
    const wv = makeWebview();
    runFanoutTask(wv, 'chat-replay', 'task-live', ['live only']);
    const replayed = fanoutTaskEntry('task-old');
    replayed.events = [
      {
        type: 'tool_call',
        name: 'run_parallel',
        extras: {tasks: JSON.stringify(['first sub task', 'second sub task'])},
      },
    ];
    const msg = shareWithTasks(wv, [
      replayed,
      {task: 'fan out the work', task_id: 'task-live', events: []},
    ]);
    assert.ok(msg);
    const dom = new JSDOM('<div id="app">' + msg.html + '</div>');
    const doc = dom.window.document;
    const sections = doc.querySelectorAll('.share-task:not(.share-subagent)');
    const oldPanel = sections[0].querySelector('.tc-run-parallel');
    assert.strictEqual(
      oldPanel.getAttribute('data-rp-subagents'),
      'sub-1 sub-2',
      'a replayed fan-out is stamped from its declared task count',
    );
  });

  await test('two fan-outs in one task split the sub-agents in order', () => {
    const wv = makeWebview();
    runFanoutTask(wv, 'chat-two', 'task-live', ['live only']);
    const entry = {
      task: 'two fanouts',
      task_id: 'task-old',
      events: [
        {
          type: 'tool_call',
          name: 'run_parallel',
          extras: {tasks: JSON.stringify(['a'])},
        },
        {
          type: 'tool_call',
          name: 'run_parallel',
          extras: {tasks: JSON.stringify(['b'])},
        },
      ],
      subagents: [
        {task: 'a', task_id: 'sub-a', parent_task_id: 'task-old', events: []},
        {task: 'b', task_id: 'sub-b', parent_task_id: 'task-old', events: []},
      ],
    };
    const msg = shareWithTasks(wv, [
      entry,
      {task: 'fan out the work', task_id: 'task-live', events: []},
    ]);
    const dom = new JSDOM('<div id="app">' + msg.html + '</div>');
    const panels = dom.window.document
      .querySelectorAll('.share-task:not(.share-subagent)')[0]
      .querySelectorAll('.tc-run-parallel');
    assert.strictEqual(panels[0].getAttribute('data-rp-subagents'), 'sub-a');
    assert.strictEqual(panels[1].getAttribute('data-rp-subagents'), 'sub-b');
  });

  await test('expanded fan-outs open their sub-agent tabs on load', () => {
    const win = makeSharePage(exportFanoutChat());
    const labels = tabLabels(win);
    assert.deepStrictEqual(
      labels,
      ['Chat', '1. first sub task', '2. second sub task'],
      "the live task's still-expanded fan-out opens its tabs; the " +
        "grandchild's fan-out was exported collapsed, so its tab stays " +
        'shut — exactly the layout the export captured',
    );
    const bar = win.document.getElementById('tab-bar');
    assert.notStrictEqual(bar.style.display, 'none', 'the strip is shown');
    const subTab = tabsOf(win)[1];
    assert.ok(
      subTab.classList.contains('subagent-tab'),
      'sub-agent tabs carry the webview tab class',
    );
    assert.ok(
      subTab.querySelector('.subagent-indicator.done.status-tick'),
      'an exported sub-agent is finished, so its indicator is the tick',
    );
    assert.ok(
      subTab.querySelector('.chat-tab-close'),
      'sub-agent tabs close like webview tabs',
    );
    assert.ok(
      !tabsOf(win)[0].querySelector('.chat-tab-close'),
      'the root chat tab has no close button',
    );
  });

  await test("expanding a sub-agent's own fan-out opens its child", () => {
    const win = makeSharePage(exportFanoutChat());
    click(tabsOf(win)[1]); // onto "1. first sub task"
    const grandPanel = fanoutPanel(win, sectionByTask(win, 'sub-1'));
    assert.ok(grandPanel.classList.contains('collapsed'));
    click(grandPanel.querySelector('.collapse-header'));
    assert.deepStrictEqual(tabLabels(win), [
      'Chat',
      '1. first sub task',
      '2. second sub task',
      '1. grand sub task',
    ]);
  });

  await test('selecting a tab swaps the transcript on screen', () => {
    const win = makeSharePage(exportFanoutChat());
    const doc = win.document;
    click(tabsOf(win)[1]); // 1. first sub task
    assert.ok(
      doc
        .querySelector('.share-task:not(.share-subagent)')
        .hasAttribute('hidden'),
      "the chat's own sections leave the screen",
    );
    assert.ok(!sectionByTask(win, 'sub-1').hasAttribute('hidden'));
    assert.ok(sectionByTask(win, 'sub-2').hasAttribute('hidden'));
    assert.strictEqual(
      tabsOf(win)[1].getAttribute('aria-selected'),
      'true',
    );
    click(tabsOf(win)[0]); // back to the chat
    assert.ok(
      !doc
        .querySelector('.share-task:not(.share-subagent)')
        .hasAttribute('hidden'),
    );
    assert.ok(sectionByTask(win, 'sub-1').hasAttribute('hidden'));
  });

  await test('collapsing a fan-out closes its tabs, expanding reopens', () => {
    const win = makeSharePage(exportFanoutChat());
    const doc = win.document;
    const panel = fanoutPanel(
      win,
      doc.querySelector('.share-task:not(.share-subagent)'),
    );
    // Open the grandchild too, so the collapse provably closes the
    // whole subtree, not just the direct children.
    click(tabsOf(win)[1]);
    const grandPanel = fanoutPanel(win, sectionByTask(win, 'sub-1'));
    click(grandPanel.querySelector('.collapse-header'));
    click(tabsOf(win)[0]);
    assert.strictEqual(tabLabels(win).length, 4);
    click(panel.querySelector('.collapse-header'));
    assert.ok(panel.classList.contains('collapsed'));
    assert.deepStrictEqual(
      tabLabels(win),
      ['Chat'],
      "collapsing the fan-out takes its sub-agents' tabs (and their " +
        "descendants' tabs) with it",
    );
    assert.strictEqual(
      doc.getElementById('tab-bar').style.display,
      'none',
      'the strip hides with only the root chat left',
    );
    click(panel.querySelector('.collapse-header'));
    assert.deepStrictEqual(tabLabels(win), [
      'Chat',
      '1. first sub task',
      '2. second sub task',
    ]);
  });

  await test('a hand-closed tab stays closed until collapse/expand', () => {
    const win = makeSharePage(exportFanoutChat());
    const doc = win.document;
    click(tabsOf(win)[2].querySelector('.chat-tab-close')); // 2. second...
    assert.deepStrictEqual(tabLabels(win), ['Chat', '1. first sub task']);
    const panel = fanoutPanel(
      win,
      doc.querySelector('.share-task:not(.share-subagent)'),
    );
    // The panel is still expanded: the hand-closed sub-agent must not
    // come back on its own.
    assert.ok(!panel.classList.contains('collapsed'));
    assert.deepStrictEqual(tabLabels(win), ['Chat', '1. first sub task']);
    click(panel.querySelector('.collapse-header')); // collapse forgives
    click(panel.querySelector('.collapse-header')); // expand reopens all
    assert.deepStrictEqual(tabLabels(win), [
      'Chat',
      '1. first sub task',
      '2. second sub task',
    ]);
  });

  await test("closing a tab closes its descendants' tabs too", () => {
    const win = makeSharePage(exportFanoutChat());
    // Open the grandchild's tab by expanding sub-1's own fan-out.
    click(tabsOf(win)[1]);
    const grandPanel = fanoutPanel(win, sectionByTask(win, 'sub-1'));
    click(grandPanel.querySelector('.collapse-header'));
    assert.strictEqual(tabLabels(win).length, 4);
    click(tabsOf(win)[0]);
    click(tabsOf(win)[1].querySelector('.chat-tab-close')); // 1. first...
    assert.deepStrictEqual(
      tabLabels(win),
      ['Chat', '2. second sub task'],
      "the grandchild's tab goes with its parent",
    );
    assert.ok(
      grandPanel.classList.contains('collapsed'),
      "a closed sub-agent's transcript folds its fan-outs, like a " +
        'webview replay would',
    );
  });

  await test('closing the selected tab picks the adjacent tab', () => {
    const win = makeSharePage(exportFanoutChat());
    const doc = win.document;
    click(tabsOf(win)[1]);
    assert.ok(!sectionByTask(win, 'sub-1').hasAttribute('hidden'));
    // The webview's user-close rule is index adjacency: whatever now
    // sits where the closed tab was comes forward — here sub-2.
    click(tabsOf(win)[1].querySelector('.chat-tab-close'));
    assert.ok(sectionByTask(win, 'sub-1').hasAttribute('hidden'));
    assert.ok(!sectionByTask(win, 'sub-2').hasAttribute('hidden'));
    assert.strictEqual(
      tabsOf(win)[1].getAttribute('aria-selected'),
      'true',
      "sub-2's tab slid into the closed slot and is selected",
    );
    // Closing the last sub-agent tab falls back to the root chat.
    click(tabsOf(win)[1].querySelector('.chat-tab-close'));
    assert.ok(
      !doc
        .querySelector('.share-task:not(.share-subagent)')
        .hasAttribute('hidden'),
      'the chat comes back on screen',
    );
    assert.strictEqual(
      tabsOf(win)[0].getAttribute('aria-selected'),
      'true',
    );
  });

  await test('a run_agent child gets an always-open orphan tab', () => {
    const wv = makeWebview();
    runFanoutTask(wv, 'chat-orphan', 'task-live', [
      'first sub task',
      'second sub task',
    ]);
    const entry = fanoutTaskEntry('task-live');
    entry.subagents.push({
      task: 'dispatched agent job',
      task_id: 'agent-1',
      parent_task_id: 'task-live',
      events: [{type: 'tool_call', name: 'Bash', command: 'echo agent'}],
    });
    const msg = shareWithTasks(wv, [entry]);
    assert.ok(msg);
    const dom = new JSDOM('<div id="app">' + msg.html + '</div>');
    const doc = dom.window.document;
    const orphan = doc.querySelector(
      '.share-subagent[data-task-id="agent-1"]',
    );
    assert.ok(orphan, 'the run_agent child is exported too');
    assert.ok(
      orphan.hasAttribute('data-sub-orphan'),
      'no fan-out declared it, so it is marked an orphan',
    );
    const taskSection = doc.querySelector(
      '.share-task:not(.share-subagent)',
    );
    assert.strictEqual(
      taskSection
        .querySelector('.tc-run-parallel')
        .getAttribute('data-rp-subagents'),
      'sub-1 sub-2',
      'the fan-out never swallows the run_agent child',
    );
    const win = makeSharePage(msg.html);
    assert.ok(
      tabLabels(win).indexOf('3. dispatched agent job') !== -1,
      "the orphan's tab opens with the chat, like the webview's " +
        'tab for a run_agent spawn',
    );
  });

  await test('declared task text beats reply order when stamping', () => {
    const wv = makeWebview();
    runFanoutTask(wv, 'chat-rev', 'task-live', ['x']);
    const entry = {
      task: 'reordered',
      task_id: 'task-old',
      events: [
        {
          type: 'tool_call',
          name: 'run_parallel',
          extras: {tasks: JSON.stringify(['alpha job', 'beta job'])},
        },
      ],
      subagents: [
        {
          task: 'beta job',
          task_id: 'sub-beta',
          parent_task_id: 'task-old',
          events: [],
        },
        {
          task: 'alpha job',
          task_id: 'sub-alpha',
          parent_task_id: 'task-old',
          events: [],
        },
      ],
    };
    const msg = shareWithTasks(wv, [
      entry,
      {task: 'x', task_id: 'task-live', events: []},
    ]);
    const dom = new JSDOM('<div id="app">' + msg.html + '</div>');
    const panel = dom.window.document
      .querySelectorAll('.share-task:not(.share-subagent)')[0]
      .querySelector('.tc-run-parallel');
    assert.strictEqual(
      panel.getAttribute('data-rp-subagents'),
      'sub-alpha sub-beta',
      'ids follow the declared task order, matched by task text',
    );
  });

  await test('a page without sub-agents never shows a tab strip', () => {
    const wv = makeWebview();
    const TAB = tabIdOf(wv);
    const now = Date.now();
    send(wv.win, {type: 'clear', chat_id: 'chat-plain', tabId: TAB});
    send(wv.win, {type: 'status', running: true, tabId: TAB, startTs: now});
    send(wv.win, {type: 'setTaskText', text: 'list files', tabId: TAB});
    send(wv.win, {
      type: 'tool_call',
      name: 'Bash',
      command: 'ls',
      tabId: TAB,
      taskId: 'task-p',
      ts: now,
    });
    const msg = shareWithTasks(wv, [
      {task: 'list files', task_id: 'task-p', events: [], subagents: []},
    ]);
    const win = makeSharePage(msg.html);
    const bar = win.document.getElementById('tab-bar');
    assert.ok(
      !bar || bar.style.display === 'none',
      'no sub-agent, no tab strip',
    );
  });

  await test('share_done raises a "Chat page saved to" toast', () => {
    const wv = makeWebview();
    runFanoutTask(wv, 'chat-toast', 'task-live', ['x']);
    const msg = shareWithTasks(wv, [fanoutTaskEntry('task-live')]);
    assert.ok(msg);
    send(wv.win, {
      type: 'share_done',
      tabId: msg.tabId,
      ok: true,
      path: '/work/reports/chat-toast.html',
    });
    const toast = wv.win.document.querySelector(
      '.kiss-notification .kiss-notification-message',
    );
    assert.ok(toast, 'a toast must appear');
    assert.strictEqual(
      toast.textContent,
      'Chat page saved to /work/reports/chat-toast.html',
    );
    const failWv = makeWebview();
    runFanoutTask(failWv, 'chat-toast2', 'task-live', ['x']);
    const failMsg = shareWithTasks(failWv, [fanoutTaskEntry('task-live')]);
    send(failWv.win, {
      type: 'share_done',
      tabId: failMsg.tabId,
      ok: false,
      error: 'disk full',
    });
    assert.ok(
      !failWv.win.document.querySelector('.kiss-notification'),
      'a failed share raises no saved-page toast',
    );
  });

  console.log(`\n${passed} passed, ${failures.length} failed`);
  // The JSDOM windows above loaded main.js, whose live timers would
  // otherwise keep the process alive until the runner's timeout.
  process.exit(failures.length > 0 ? 1 : 0);
}

run();
