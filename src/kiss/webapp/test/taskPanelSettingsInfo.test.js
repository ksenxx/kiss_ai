// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end tests for the task-settings rows of the Task Info panel
// (#meta-list in media/chat.html, painted by updateMetaTaskDetails in
// media/main.js) — Machine, Date, Base model, Chat id, Task id and
// (when the task has one) Parent task; there is no Parallel mode row —
// and for the task panels of a saved page ("Save chat"), which carry
// an info block each with EVERY row of the Task Info panel:
//
// * a live task's `task_settings` event paints the rows,
// * the Machine row names the tab's bound machine (`tabMachine`),
//   else this daemon's (`configData.machine`),
// * a new submit clears the previous task's rows ('—') until the new
//   task's settings arrive,
// * session replays (`task_events`) repopulate the rows from the
//   replayed stream's own task_settings event,
// * a spliced-in adjacent task's replay must NOT repaint the rows of
//   the task on screen, but scrolling onto the neighbour shows ITS
//   settings,
// * a background tab's task_settings land on that tab and show after a
//   switch, and
// * the share export (whose static page has no Task Info panel) gives
//   every synthesized task panel its OWN task's settings info block.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');
const {makeRunner} = require('./ui_antipattern_harness');
const {
  makePage: makeHarnessPage,
  send,
  clickTab,
  stubRaf,
  MEDIA,
} = require('./simplify2_harness');
const {test, report} = makeRunner();

function makePage() {
  return makeHarnessPage({
    sourceUrl: 'taskinfo-main.js',
    beforeScripts: stubRaf,
  });
}

function tabIdOf(wv) {
  const ready = wv.posted.find(m => m.type === 'ready');
  assert.ok(ready && ready.tabId, 'page must post ready with a tabId');
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

const DASH = '\u2014';

/** The Task Info panel's task-settings rows as {id: text}. */
function infoRows(win) {
  const doc = win.document;
  const val = id => doc.getElementById(id).textContent;
  return {
    machine: val('meta-machine'),
    date: val('meta-date'),
    model: val('meta-model'),
    chatId: val('meta-chat-id'),
    taskId: val('meta-task-id'),
    parent: val('meta-parent-id'),
    parentHidden: doc.getElementById('meta-parent-item').hidden,
  };
}

/** The rows' text joined, for "shows X" / "shows nothing" checks. */
function infoText(win) {
  const r = infoRows(win);
  return [r.date, r.model, r.chatId, r.taskId, r.parent]
    .filter(t => t !== DASH)
    .join(' | ');
}

const SETTINGS = {
  model: 'model-x',
  work_dir: '/repo',
  is_parallel: true,
  max_budget: 5,
  start_ts: Date.UTC(2026, 1, 3, 4, 5),
  chat_id: 'chat-abc',
  task_id: 'task-1',
  is_subagent: false,
};

test('live task_settings paints the info block; new submit clears it', () => {
  const wv = makePage();
  const win = wv.win;
  const TAB = tabIdOf(wv);
  send(win, {type: 'setTaskText', text: 'do the thing', tabId: TAB});
  assert.strictEqual(infoText(win), '', 'no settings known yet');
  send(win, {
    type: 'task_settings',
    settings: SETTINGS,
    tabId: TAB,
    taskId: 'task-1',
  });
  const rows = infoRows(win);
  const txt = infoText(win);
  assert.strictEqual(
    win.document.getElementById('meta-workdir').textContent,
    '/repo',
    'work dir shown',
  );
  assert.strictEqual(rows.model, 'model-x', 'model shown');
  assert.strictEqual(
    win.document.getElementById('meta-parallel'),
    null,
    'is_parallel paints no row: the Parallel mode item is gone',
  );
  assert.strictEqual(
    win.document.getElementById('meta-max-budget').textContent,
    '$5.00',
    'budget shown',
  );
  assert.strictEqual(
    rows.date,
    new Date(SETTINGS.start_ts).toLocaleString(undefined, {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    }),
    'start time shown localized',
  );
  assert.strictEqual(rows.chatId, 'chat-abc', 'chat id shown');
  assert.strictEqual(rows.taskId, 'task-1', 'task id shown');
  assert.strictEqual(rows.parent, DASH, 'no parent for a top-level task');
  assert.ok(rows.parentHidden, 'the Parent task row hides without one');
  assert.ok(!txt.includes('subagent'), 'not a subagent');
  assert.ok(
    !win.document.getElementById('task-panel-info'),
    'the live static task panel carries no settings block',
  );

  // A malformed event (no settings) must change nothing.
  send(win, {type: 'task_settings', tabId: TAB, taskId: 'task-1'});
  assert.strictEqual(infoText(win), txt);

  // Settings without a task id still paint the panel (they just are
  // not remembered for adjacent-task lookups).
  send(win, {
    type: 'task_settings',
    settings: {model: 'idless-model', chat_id: 'chat-abc'},
    tabId: TAB,
    taskId: 'task-1',
  });
  assert.ok(infoText(win).includes('idless-model'));

  // A queued follow-up echoes setTaskText for the SAME task: the
  // settings survive (only a real new run may drop them).
  send(win, {type: 'setTaskText', text: 'follow-up nudge', tabId: TAB});
  assert.ok(
    infoText(win).includes('idless-model'),
    'a queued follow-up must not clear the running task settings',
  );

  // A real replacement run announces itself with 'clear'.
  send(win, {type: 'clear', chat_id: 'chat-abc', tabId: TAB});
  assert.strictEqual(infoText(win), '', 'a new run clears the rows');
  assert.strictEqual(infoRows(win).model, DASH, 'an unknown row shows a dash');
});

test('a run with an agent script shows the Agent row; a plain run hides it', () => {
  const wv = makePage();
  const win = wv.win;
  const TAB = tabIdOf(wv);
  const doc = win.document;
  send(win, {type: 'setTaskText', text: 'plain', tabId: TAB});
  send(win, {
    type: 'task_settings',
    settings: SETTINGS,
    tabId: TAB,
    taskId: 'task-1',
  });
  assert.ok(
    doc.getElementById('meta-agent-item').hidden,
    'no script: row hidden',
  );
  assert.strictEqual(doc.getElementById('meta-agent').textContent, DASH);
  send(win, {
    type: 'task_settings',
    settings: {
      ...SETTINGS,
      task_id: 'task-3',
      sea: 'sh',
      tool_profile: 'bash',
      timeout: 3600,
      inherited: ['model', 'chat_id'],
      pinned: {use_worktree: [true, false], work_dir: ['', '/tmp/x']},
    },
    tabId: TAB,
    taskId: 'task-3',
  });
  assert.ok(!doc.getElementById('meta-agent-item').hidden, 'script: row shown');
  assert.strictEqual(
    doc.getElementById('meta-agent').textContent,
    'sh \u2022 tools bash \u2022 timeout 3600s \u2022 ' +
      'pinned use_worktree(true->false), work_dir(empty->/tmp/x)',
  );
  send(win, {type: 'clear', chat_id: 'chat-abc', tabId: TAB});
  assert.ok(
    doc.getElementById('meta-agent-item').hidden,
    'a new run clears the row',
  );
});

test('subagent settings show sequential and parentage', () => {
  const wv = makePage();
  const win = wv.win;
  const TAB = tabIdOf(wv);
  send(win, {type: 'setTaskText', text: 'sub work', tabId: TAB});
  send(win, {
    type: 'task_settings',
    settings: {
      model: 'model-y',
      is_parallel: false,
      chat_id: 'chat-abc',
      task_id: 'task-2',
      is_subagent: true,
      parent_task_id: 'task-1',
    },
    tabId: TAB,
    taskId: 'task-2',
  });
  const rows = infoRows(win);
  assert.strictEqual(rows.date, DASH, 'unknown start time is a dash');
  assert.strictEqual(rows.parent, 'task-1', 'parent shown');
  assert.ok(!rows.parentHidden, 'the Parent task row shows for a subagent');
  assert.strictEqual(rows.taskId, 'task-2 (subagent)', 'subagent marker');
  // Settings without a task id show dashes, not a guessed value.
  send(win, {
    type: 'task_settings',
    settings: {model: 'bare', is_subagent: true},
    tabId: TAB,
    taskId: 'task-2',
  });
  const bare = infoRows(win);
  assert.strictEqual(bare.taskId, DASH, 'no task id, no subagent suffix');
  assert.ok(bare.parentHidden, 'no parent id: the Parent task row hides');
  // A parent id shows whatever the subagent flag says: the row names
  // the task's parentage, not its kind.
  send(win, {
    type: 'task_settings',
    settings: {task_id: 'task-3', parent_task_id: 'task-1', is_subagent: false},
    tabId: TAB,
    taskId: 'task-3',
  });
  const child = infoRows(win);
  assert.strictEqual(child.parent, 'task-1');
  assert.ok(!child.parentHidden);
  assert.strictEqual(child.taskId, 'task-3', 'no subagent suffix');
});

test('task_events replay repopulates the info from its own stream', () => {
  const wv = makePage();
  const win = wv.win;
  const TAB = tabIdOf(wv);
  send(win, {
    type: 'task_events',
    tabId: TAB,
    chat_id: 'chat-abc',
    task_id: 'task-1',
    task: 'old task',
    events: [
      {type: 'task_settings', settings: SETTINGS},
      {type: 'system_output', text: 'hello\n'},
    ],
  });
  assert.strictEqual(
    infoRows(win).taskId,
    'task-1',
    'replayed settings must fill the rows: ' + infoText(win),
  );
  // A replay carrying no settings must not keep the previous task's.
  send(win, {
    type: 'task_events',
    tabId: TAB,
    chat_id: 'chat-abc',
    task_id: 'task-9',
    task: 'settings-less task',
    events: [{type: 'system_output', text: 'x\n'}],
  });
  assert.strictEqual(
    infoText(win),
    '',
    'a replay without settings clears the rows',
  );
});

test('adjacent replay never repaints the live panel; scrolling does', () => {
  const wv = makePage();
  const win = wv.win;
  const TAB = tabIdOf(wv);
  win._testApi.hideWelcome();
  send(win, {
    type: 'task_events',
    tabId: TAB,
    chat_id: 'chat-abc',
    task_id: 'task-2',
    task: 'live task',
    events: [
      {
        type: 'task_settings',
        settings: Object.assign({}, SETTINGS, {
          task_id: 'task-2',
          model: 'live-model',
        }),
      },
      {type: 'system_output', text: 'live\n'},
    ],
  });
  assert.ok(infoText(win).includes('live-model'));

  // A NEXT neighbour first: with everything at zero geometry the
  // visible-region scan keeps electing the first region (the tab's own
  // task), covering the "own settings" arm of updateVisibleTask.
  send(win, {
    type: 'adjacent_task_events',
    tabId: TAB,
    direction: 'next',
    task: 'newer neighbour',
    task_id: 'task-3',
    events: [{type: 'system_output', text: 'newer\n'}],
  });
  assert.ok(
    infoText(win).includes('live-model'),
    'a spliced-in neighbour must not steal the live task rows',
  );

  // A PREV neighbour becomes the first region, so the panel is lent to
  // it — with settings when its stream carries them.
  send(win, {
    type: 'adjacent_task_events',
    tabId: TAB,
    direction: 'prev',
    task: 'older neighbour',
    task_id: 'task-1',
    events: [
      {
        type: 'task_settings',
        settings: Object.assign({}, SETTINGS, {model: 'old-model'}),
      },
      {type: 'system_output', text: 'older\n'},
    ],
  });
  assert.ok(
    infoText(win).includes('old-model'),
    'parking on the neighbour shows ITS settings: ' + infoText(win),
  );
});

test('a neighbour without settings shows dashes', () => {
  const wv = makePage();
  const win = wv.win;
  const TAB = tabIdOf(wv);
  win._testApi.hideWelcome();
  send(win, {
    type: 'task_events',
    tabId: TAB,
    chat_id: 'chat-abc',
    task_id: 'task-2',
    task: 'live task',
    events: [
      {type: 'task_settings', settings: SETTINGS},
      {type: 'system_output', text: 'live\n'},
    ],
  });
  send(win, {
    type: 'adjacent_task_events',
    tabId: TAB,
    direction: 'prev',
    task: 'legacy neighbour',
    task_id: 'task-0',
    events: [{type: 'system_output', text: 'legacy\n'}],
  });
  assert.strictEqual(
    infoText(win),
    '',
    'a legacy neighbour with no known settings shows dashes only',
  );
});

test('a background tab keeps its settings and shows them on switch', () => {
  const wv = makePage();
  const win = wv.win;
  const TAB = tabIdOf(wv);
  send(win, {type: 'setTaskText', text: 'first task', tabId: TAB});
  send(win, {
    type: 'task_settings',
    settings: SETTINGS,
    tabId: TAB,
    taskId: 'task-1',
  });
  const SECOND = win._testApi.createNewTab();
  assert.notStrictEqual(win._testApi.getActiveTabId(), TAB);
  assert.strictEqual(infoText(win), '', 'a fresh tab has no settings');
  // A live event for the now-background first tab — a NEW task the
  // tab started while hidden, whose id the tab must adopt.
  send(win, {
    type: 'task_settings',
    settings: Object.assign({}, SETTINGS, {
      model: 'bg-model',
      task_id: 'task-7',
    }),
    tabId: TAB,
    taskId: 'task-7',
  });
  assert.strictEqual(
    infoText(win),
    '',
    'a background tab event must not repaint the visible rows',
  );
  clickTab(win, TAB);
  assert.ok(
    infoText(win).includes('bg-model'),
    'switching back shows the settings the tab received: ' + infoText(win),
  );
  clickTab(win, String(SECOND || win._testApi.getActiveTabId()));
});

test('share export gives every task panel its own settings info', () => {
  const wv = makePage();
  const win = wv.win;
  const TAB = tabIdOf(wv);
  const now = Date.now();
  send(win, {type: 'clear', chat_id: 'chat-777', tabId: TAB});
  send(win, {type: 'status', running: true, tabId: TAB, startTs: now});
  send(win, {type: 'setTaskText', text: 'live task', tabId: TAB});
  send(win, {
    type: 'task_settings',
    settings: Object.assign({}, SETTINGS, {
      task_id: 'task-b',
      model: 'live-model',
    }),
    tabId: TAB,
    taskId: 'task-b',
  });
  send(win, {
    type: 'system_output',
    text: 'live output\n',
    tabId: TAB,
    taskId: 'task-b',
    ts: now,
  });
  click(win.document.getElementById('share-btn'));
  const req = wv.posted.filter(m => m.type === 'shareChatTasks').pop();
  assert.ok(req, 'share click must request the chat tasks');
  send(win, {
    type: 'share_tasks',
    tabId: req.tabId,
    chatId: req.chatId,
    truncated: false,
    tasks: [
      {
        task: 'old task',
        task_id: 'task-a',
        events: [
          {
            type: 'task_settings',
            settings: Object.assign({}, SETTINGS, {
              task_id: 'task-a',
              model: 'old-model',
            }),
          },
          {type: 'system_output', text: 'old output\n'},
        ],
      },
      // The live task, listed WITHOUT its settings event: the export
      // must fall back to the settings the live panel holds.
      {task: 'live task', task_id: 'task-b', events: []},
    ],
  });
  const msg = wv.posted.filter(m => m.type === 'shareChat').pop();
  assert.ok(msg, 'the share_tasks reply must produce a shareChat command');
  const dom = new JSDOM('<div id="app">' + msg.html + '</div>');
  const infos = Array.from(
    dom.window.document.querySelectorAll('.task-panel-info'),
  ).map(el => el.textContent);
  assert.strictEqual(infos.length, 2, 'one info block per task panel');
  assert.ok(
    infos[0].includes('Base model:old-model') &&
      infos[0].includes('Task id:task-a'),
    'first section shows the OLD task settings: ' + infos[0],
  );
  assert.ok(
    infos[1].includes('Base model:live-model') &&
      infos[1].includes('Task id:task-b'),
    'second section shows the LIVE task settings: ' + infos[1],
  );
});

/** The labelled rows of one saved-page info block as {label: value}. */
function savedRows(infoEl) {
  const out = {};
  for (const line of infoEl.querySelectorAll('.task-panel-info-line')) {
    const label = line.querySelector('.task-panel-info-label').textContent;
    out[label.replace(/:$/, '')] = line.textContent.slice(label.length);
  }
  return out;
}

const PANEL_ROWS = [
  'Tokens',
  'Cost',
  'Steps',
  'Time',
  'Machine',
  'Workdir',
  'Max budget',
  'Date',
  'Base model',
  'Chat id',
  'Task id',
];

test('a saved page lists every Task Info row for every task', () => {
  const wv = makePage();
  const win = wv.win;
  const TAB = tabIdOf(wv);
  const doc = win.document;
  send(win, {type: 'configData', config: {}, machine: 'box-a', tabId: TAB});
  send(win, {type: 'clear', chat_id: 'chat-abc', tabId: TAB});
  send(win, {type: 'setTaskText', text: 'live task', tabId: TAB});
  send(win, {
    type: 'task_settings',
    settings: Object.assign({}, SETTINGS, {task_id: 'task-live'}),
    tabId: TAB,
    taskId: 'task-live',
  });
  send(win, {
    type: 'usage_info',
    tabId: TAB,
    taskId: 'task-live',
    total_tokens: 4321,
    cost_usd: 0.42,
    total_steps: 7,
  });
  send(win, {
    type: 'system_output',
    text: 'live output\n',
    tabId: TAB,
    taskId: 'task-live',
  });
  // The status bar figures the Task Info panel mirrors (through a
  // MutationObserver, which fires after this synchronous test's
  // checks) and the saved page must repeat.
  const statusText = id => doc.getElementById(id).textContent;
  assert.strictEqual(statusText('status-tokens'), 'Tokens: 4.32K');
  assert.strictEqual(statusText('status-budget'), 'Cost: $0.42');
  assert.strictEqual(statusText('status-steps'), 'Steps: 7');
  assert.strictEqual(statusText('meta-machine'), 'box-a');
  click(doc.getElementById('share-btn'));
  const req = wv.posted.filter(m => m.type === 'shareChatTasks').pop();
  // The status bar's spans are lent to a neighbouring task the reader
  // scrolls onto: the saved page must report the tab's OWN figures.
  doc.getElementById('status-tokens').textContent = 'Tokens: 999';
  doc.getElementById('status-budget').textContent = 'Cost: $9.99';
  doc.getElementById('status-steps').textContent = 'Steps: 99';
  const started = Date.UTC(2026, 1, 3, 4, 5);
  send(win, {
    type: 'share_tasks',
    tabId: req.tabId,
    chatId: req.chatId,
    truncated: false,
    tasks: [
      {
        task: 'finished task',
        task_id: 'task-done',
        events: [
          {
            type: 'task_settings',
            settings: Object.assign({}, SETTINGS, {
              task_id: 'task-done',
              sea: 'write_paper',
              kind: 'command',
            }),
          },
        ],
        metrics: {
          tokens: 1234567,
          cost: 1.5,
          steps: 12,
          start_ts: started,
          end_ts: started + 125000,
        },
        subagents: [
          {
            task: 'child task',
            task_id: 'task-child',
            parent_task_id: 'task-done',
            events: [
              {
                type: 'task_settings',
                settings: Object.assign({}, SETTINGS, {
                  task_id: 'task-child',
                  parent_task_id: 'task-done',
                  is_subagent: true,
                }),
              },
              // A running sub-agent's live recording: its newest
              // totals are in this event, not yet in its row.
              {
                type: 'usage_info',
                total_tokens: 10,
                cost_usd: 0.001,
                total_steps: 1,
              },
            ],
            // Still running: zero row totals, no end time yet.
            metrics: {tokens: 0, cost: 0, steps: 0, start_ts: started},
          },
        ],
      },
      {task: 'live task', task_id: 'task-live', events: []},
      // A task whose reply carries a usage event but no row metrics.
      {
        task: 'usage only',
        task_id: 'task-usage',
        events: [
          {type: 'usage_info', total_tokens: 500, cost: '$0.0500'},
          {type: 'result', total_tokens: 900, cost_usd: 0.09},
        ],
        metrics: null,
      },
    ],
  });
  const msg = wv.posted.filter(m => m.type === 'shareChat').pop();
  const dom = new JSDOM('<div id="app">' + msg.html + '</div>');
  const infos = Array.from(
    dom.window.document.querySelectorAll('.task-panel-info'),
  );
  assert.strictEqual(infos.length, 4, 'chat tasks, then the sub-agent');
  const done = savedRows(infos[0]);
  assert.deepStrictEqual(
    Object.keys(done),
    PANEL_ROWS.concat(['Agent']),
    'the finished task lists the panel rows in the panel order',
  );
  assert.strictEqual(done.Tokens, '1.23M');
  assert.strictEqual(done.Cost, '$1.50');
  assert.strictEqual(done.Steps, '12');
  assert.strictEqual(done.Time, 'Done (2m 5s)');
  assert.strictEqual(done.Machine, 'box-a');
  assert.strictEqual(done.Workdir, '/repo');
  assert.strictEqual(done['Max budget'], '$5.00');
  assert.strictEqual(done.Date, infoRows(win).date, 'same date text');
  assert.strictEqual(done['Base model'], 'model-x');
  assert.strictEqual(done['Chat id'], 'chat-abc');
  assert.strictEqual(done['Task id'], 'task-done');
  assert.ok(done.Agent.includes('write_paper'), done.Agent);

  const live = savedRows(infos[1]);
  assert.deepStrictEqual(Object.keys(live), PANEL_ROWS);
  assert.strictEqual(live.Tokens, '4.32K', 'the live task: the status bar');
  assert.strictEqual(live.Cost, '$0.42');
  assert.strictEqual(live.Steps, '7');
  assert.strictEqual(
    live.Time,
    statusText('status-text'),
    'the live task: the Time row as the panel shows it',
  );
  assert.strictEqual(live['Task id'], 'task-live');

  const usageOnly = savedRows(infos[2]);
  assert.deepStrictEqual(Object.keys(usageOnly), PANEL_ROWS);
  assert.strictEqual(usageOnly.Tokens, '900', 'the LAST usage event wins');
  assert.strictEqual(usageOnly.Cost, '$0.09');
  assert.strictEqual(usageOnly.Steps, '0', 'no step count in the event');
  assert.strictEqual(usageOnly.Time, DASH, 'no row: no start or end');

  const child = savedRows(infos[3]);
  assert.deepStrictEqual(
    Object.keys(child),
    PANEL_ROWS.concat(['Parent task']),
    'a sub-agent adds its Parent task row',
  );
  assert.strictEqual(child.Tokens, '10', "the live recording's totals");
  assert.strictEqual(child.Cost, '$0.0010');
  assert.strictEqual(child.Steps, '1');
  assert.strictEqual(child.Time, 'Running', 'no end time: still running');
  assert.strictEqual(child['Task id'], 'task-child (subagent)');
  assert.strictEqual(child['Parent task'], 'task-done');
});

test('the Machine row names the bound machine, else this daemon', () => {
  const wv = makePage();
  const win = wv.win;
  const TAB = tabIdOf(wv);
  assert.strictEqual(infoRows(win).machine, DASH, 'unknown until configData');
  send(win, {type: 'configData', config: {}, machine: 'box-a', tabId: TAB});
  assert.strictEqual(infoRows(win).machine, 'box-a', "this daemon's machine");
  send(win, {type: 'tabMachine', tabId: TAB, machine: 'remote-1'});
  assert.strictEqual(infoRows(win).machine, 'remote-1', 'the bound machine');
  send(win, {
    type: 'machines',
    machines: [
      {name: 'box-a', local: true, status: 'reachable'},
      {name: 'remote-1', label: 'Remote One', status: 'reachable'},
    ],
  });
  assert.strictEqual(
    infoRows(win).machine,
    'Remote One',
    'a machines reply relabels the row',
  );
  send(win, {
    type: 'history',
    sessions: [],
    machineLabels: {'remote-1': 'Build Host'},
  });
  assert.strictEqual(
    infoRows(win).machine,
    'Build Host',
    "a history reply's labels relabel the row too",
  );
  send(win, {type: 'tabMachine', tabId: TAB, machine: ''});
  assert.strictEqual(infoRows(win).machine, 'box-a', 'unbound: this daemon');
  send(win, {type: 'tabMachine', tabId: 'some-other-tab', machine: 'remote-1'});
  assert.strictEqual(
    infoRows(win).machine,
    'box-a',
    "another tab's binding leaves the active tab's row alone",
  );
});

test('share markup: the info block follows the task text', () => {
  const wv = makePage();
  const win = wv.win;
  const TAB = tabIdOf(wv);
  send(win, {type: 'clear', chat_id: 'chat-1', tabId: TAB});
  send(win, {type: 'setTaskText', text: 'a task', tabId: TAB});
  click(win.document.getElementById('share-btn'));
  const req = wv.posted.filter(m => m.type === 'shareChatTasks').pop();
  send(win, {
    type: 'share_tasks',
    tabId: req.tabId,
    chatId: req.chatId,
    truncated: false,
    // A task with no settings anywhere: the block is present but empty
    // (main.css hides an empty one), so every page has the same shape.
    tasks: [{task: 'a task', task_id: 'task-z', events: []}],
  });
  const msg = wv.posted.filter(m => m.type === 'shareChat').pop();
  const dom = new JSDOM('<div id="app">' + msg.html + '</div>');
  const doc = dom.window.document;
  const info = doc.querySelector('.task-panel .task-panel-info');
  assert.ok(info, 'the exported panel carries an info block');
  assert.strictEqual(info.textContent, '', 'no settings, empty block');
  assert.strictEqual(
    info.previousElementSibling.className,
    'task-panel-text',
    'the block follows the task text',
  );
  // The live transcript's own task panel shows no info block: the Task
  // Info panel carries the settings there.
  assert.ok(
    !win.document.querySelector('#output .task-panel-info'),
    'no info block on the live task panel',
  );
  const css = fs.readFileSync(path.join(MEDIA, 'main.css'), 'utf8');
  assert.ok(
    /\.task-panel-info:empty\s*\{\s*display:\s*none;/.test(css),
    'the info block hides when empty',
  );
});

report('taskPanelSettingsInfo');
// The pages leave live timers behind (the running-task clock, the
// button flash), so exit explicitly instead of draining them.
process.exit(0);
