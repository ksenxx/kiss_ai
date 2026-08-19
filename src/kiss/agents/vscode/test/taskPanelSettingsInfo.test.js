// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end tests for the static task panel's settings info block
// (#task-panel-info in media/chat.html, rendered by media/main.js):
//
// * a live task's `task_settings` event paints the info block with the
//   same fields the history sidebar shows (model, wt/no-wt,
//   parallel/sequential, budget, start time, chat/task/parent ids),
// * a new submit clears the previous task's info until the new task's
//   settings arrive,
// * session replays (`task_events`) repopulate the block from the
//   replayed stream's own task_settings event,
// * a spliced-in adjacent task's replay must NOT repaint the panel of
//   the task on screen, but scrolling onto the neighbour shows ITS
//   settings,
// * a background tab's task_settings land on that tab and show after a
//   switch, and
// * the share export gives every synthesized task panel its OWN task's
//   settings info.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

let passed = 0;
const failures = [];

function test(name, fn) {
  try {
    fn();
    passed++;
    console.log(`  \u2713 ${name}`);
  } catch (e) {
    failures.push({name, error: e});
    console.log(`  \u2717 ${name}`);
    console.log(`      ${e.stack || e.message}`);
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
      '\n//# sourceURL=taskinfo-main.js',
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

function clickTab(win, tabId) {
  const el = win.document.querySelector(
    `.chat-tab[data-tab-id=${JSON.stringify(tabId)}]`,
  );
  assert.ok(el, `tab ${tabId} must exist in the tab bar`);
  el.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

function infoText(win) {
  return win.document.getElementById('task-panel-info').textContent;
}

const SETTINGS = {
  model: 'model-x',
  work_dir: '/repo',
  is_worktree: true,
  is_parallel: true,
  max_budget: 5,
  start_ts: Date.UTC(2026, 1, 3, 4, 5),
  chat_id: 'chat-abc',
  task_id: 'task-1',
  is_subagent: false,
};

test('live task_settings paints the info block; new submit clears it', () => {
  const wv = makeWebview();
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
  const txt = infoText(win);
  assert.ok(txt.includes('/repo'), 'work dir shown: ' + txt);
  assert.ok(txt.includes('model-x'), 'model shown');
  assert.ok(txt.includes('wt'), 'worktree mode shown');
  assert.ok(txt.includes('parallel'), 'parallel mode shown');
  assert.ok(txt.includes('budget $5.00'), 'budget shown: ' + txt);
  assert.ok(txt.includes('started '), 'start time shown');
  assert.ok(txt.includes('chat chat-abc'), 'chat id shown');
  assert.ok(txt.includes('task task-1'), 'task id shown');
  assert.ok(!txt.includes('parent'), 'no parent for a top-level task');
  assert.ok(!txt.includes('subagent'), 'not a subagent');

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
  assert.strictEqual(infoText(win), '', 'a new run clears the info block');
});

test('subagent settings show no-wt, sequential and parentage', () => {
  const wv = makeWebview();
  const win = wv.win;
  const TAB = tabIdOf(wv);
  send(win, {type: 'setTaskText', text: 'sub work', tabId: TAB});
  send(win, {
    type: 'task_settings',
    settings: {
      model: 'model-y',
      is_worktree: false,
      is_parallel: false,
      chat_id: 'chat-abc',
      task_id: 'task-2',
      is_subagent: true,
      parent_task_id: 'task-1',
    },
    tabId: TAB,
    taskId: 'task-2',
  });
  const txt = infoText(win);
  assert.ok(txt.includes('no-wt'), 'no-wt shown: ' + txt);
  assert.ok(txt.includes('sequential'), 'sequential shown');
  assert.ok(!txt.includes('budget'), 'unknown budget omitted');
  assert.ok(!txt.includes('started'), 'unknown start time omitted');
  assert.ok(txt.includes('parent task-1'), 'parent shown');
  assert.ok(txt.includes('subagent'), 'subagent marker shown');
});

test('task_events replay repopulates the info from its own stream', () => {
  const wv = makeWebview();
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
  assert.ok(
    infoText(win).includes('task task-1'),
    'replayed settings must fill the info block: ' + infoText(win),
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
    'a replay without settings clears the block',
  );
});

test('adjacent replay never repaints the live panel; scrolling does', () => {
  const wv = makeWebview();
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
    'a spliced-in neighbour must not steal the live panel info',
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

test('a neighbour without settings shows an empty info block', () => {
  const wv = makeWebview();
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
    'a legacy neighbour with no known settings shows no info',
  );
});

test('a background tab keeps its settings and shows them on switch', () => {
  const wv = makeWebview();
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
    'a background tab event must not repaint the visible panel',
  );
  clickTab(win, TAB);
  assert.ok(
    infoText(win).includes('bg-model'),
    'switching back shows the settings the tab received: ' + infoText(win),
  );
  clickTab(win, String(SECOND || win._testApi.getActiveTabId()));
});

test('share export gives every task panel its own settings info', () => {
  const wv = makeWebview();
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
    dom.window.document.querySelectorAll('#task-panel-info'),
  ).map(el => el.textContent);
  assert.strictEqual(infos.length, 2, 'one info block per task panel');
  assert.ok(
    infos[0].includes('old-model') && infos[0].includes('task task-a'),
    'first section shows the OLD task settings: ' + infos[0],
  );
  assert.ok(
    infos[1].includes('live-model') && infos[1].includes('task task-b'),
    'second section shows the LIVE task settings: ' + infos[1],
  );
});

test('drawer markup: the info block lives inside the task panel', () => {
  const wv = makeWebview();
  const doc = wv.win.document;
  const info = doc.getElementById('task-panel-info');
  assert.ok(info, '#task-panel-info must exist');
  assert.strictEqual(
    info.parentElement,
    doc.getElementById('task-panel'),
    'the info block is a direct child of the panel',
  );
  const css = fs.readFileSync(path.join(MEDIA, 'main.css'), 'utf8');
  assert.ok(
    /#task-panel-info:empty,\s*#task-panel\.drawer-collapsed #task-panel-info\s*\{\s*display:\s*none;/.test(
      css,
    ),
    'the info block hides when empty or when the drawer is collapsed',
  );
});

console.log(`\n${passed} passed, ${failures.length} failed`);
// The webviews leave live timers behind (the running-task clock, the
// button flash), so exit explicitly instead of draining them.
process.exit(failures.length > 0 ? 1 : 0);
