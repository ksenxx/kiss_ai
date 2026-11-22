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
//   switch,
// * the Tags row shows the task's tags (`taskTags`) as chips and edits
//   them (`setTaskTags`), on the daemon that holds the task's row, and
// * the share export (whose static page has no Task Info panel) gives
//   every synthesized task panel its OWN task's settings info block.

'use strict';

const assert = require('assert');
const {JSDOM} = require('jsdom');
const {makeRunner, key} = require('./ui_antipattern_harness');
const {
  makePage: makeHarnessPage,
  send,
  clickTab,
  click,
  stubRaf,
  fakeMatchMedia,
  loadMainCss,
  readyTabId,
} = require('./simplify2_harness');
// Every page a test built: the runner closes them when it has ended.
const openWindows = [];
const {test, report} = makeRunner({windows: openWindows});

function makePage() {
  const wv = makeHarnessPage({
    sourceUrl: 'taskinfo-main.js',
    beforeScripts: stubRaf,
  });
  openWindows.push(wv.win);
  return wv;
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
  const TAB = readyTabId(wv.posted);
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
  const TAB = readyTabId(wv.posted);
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

test('subagent settings show parentage', () => {
  const wv = makePage();
  const win = wv.win;
  const TAB = readyTabId(wv.posted);
  send(win, {type: 'setTaskText', text: 'sub work', tabId: TAB});
  send(win, {
    type: 'task_settings',
    settings: {
      model: 'model-y',
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

/** What makes an id row's value a button, as {attribute: value}. */
function idButton(win, id) {
  const el = win.document.getElementById(id);
  return {
    role: el.getAttribute('role'),
    tabindex: el.getAttribute('tabindex'),
    tooltip: el.getAttribute('data-tooltip'),
    label: el.getAttribute('aria-label'),
  };
}

const NO_BUTTON = {role: null, tabindex: null, tooltip: null, label: null};

/**
 * What a mouse or a finger fires on *el* for one press and release: it
 * counts *presses*, one unless it is the second or third click of a
 * double- or triple-click.
 */
function pointerClick(win, el, presses = 1) {
  for (const type of ['mousedown', 'mouseup', 'click'])
    el.dispatchEvent(
      new win.MouseEvent(type, {
        bubbles: true,
        cancelable: true,
        detail: presses,
      }),
    );
}

/**
 * The message box with *text* in it and the selection *start*..*end*,
 * plus a count of the `input` events it fires from now on.
 */
function composerWith(win, text, start, end) {
  const inp = win.document.getElementById('task-input');
  inp.value = text;
  inp.setSelectionRange(start, end);
  const fired = {inputs: 0};
  inp.addEventListener('input', () => fired.inputs++);
  return {inp, fired};
}

test('a known id is a button that inserts the id into the message box at the caret', () => {
  const wv = makePage();
  const win = wv.win;
  const doc = win.document;
  const TAB = readyTabId(wv.posted);
  // Unknown ids are plain text, and a click on one inserts nothing.
  for (const id of ['meta-chat-id', 'meta-task-id', 'meta-parent-id'])
    assert.deepStrictEqual(idButton(win, id), NO_BUTTON);
  const {inp, fired} = composerWith(win, 'see  now', 4, 4);
  click(win, doc.getElementById('meta-chat-id'));
  assert.strictEqual(inp.value, 'see  now');
  assert.strictEqual(fired.inputs, 0);

  send(win, {
    type: 'task_settings',
    settings: SETTINGS,
    tabId: TAB,
    taskId: 'task-1',
  });
  assert.deepStrictEqual(idButton(win, 'meta-chat-id'), {
    role: 'button',
    tabindex: '0',
    tooltip: 'Insert into the message box',
    label: 'chat-abc: insert into the message box',
  });
  assert.deepStrictEqual(idButton(win, 'meta-task-id'), {
    role: 'button',
    tabindex: '0',
    tooltip: 'Insert into the message box',
    label: 'task-1: insert into the message box',
  });
  assert.deepStrictEqual(
    idButton(win, 'meta-parent-id'),
    NO_BUTTON,
    'a top-level task has no parent id to insert',
  );
  // At the caret, which ends after the id; the box takes the keyboard
  // and hears of the edit as it hears of typing.
  click(win, doc.getElementById('meta-chat-id'));
  assert.strictEqual(inp.value, 'see chat-abc now');
  assert.deepStrictEqual([inp.selectionStart, inp.selectionEnd], [12, 12]);
  assert.strictEqual(doc.activeElement, inp);
  assert.strictEqual(fired.inputs, 1);
  // A selection is replaced.
  inp.setSelectionRange(13, 16);
  click(win, doc.getElementById('meta-task-id'));
  assert.strictEqual(inp.value, 'see chat-abc task-1');
  assert.deepStrictEqual([inp.selectionStart, inp.selectionEnd], [19, 19]);
  assert.strictEqual(fired.inputs, 2);
  // Enter and Space press the button; another key does not.
  const taskId = doc.getElementById('meta-task-id');
  inp.setSelectionRange(0, 3);
  assert.ok(!key(win, taskId, 'Enter'), 'Enter is the button\u2019s');
  assert.strictEqual(inp.value, 'task-1 chat-abc task-1');
  inp.setSelectionRange(6, 6);
  assert.ok(!key(win, taskId, ' '), 'Space is the button\u2019s');
  assert.strictEqual(inp.value, 'task-1task-1 chat-abc task-1');
  assert.ok(key(win, taskId, 'a'));
  assert.strictEqual(fired.inputs, 4);
  // The pointer click that ends a drag over the id's text selected it
  // (to copy it): nothing is inserted.  Enter on the button inserts
  // all the same, and a selection elsewhere on the page does not hold
  // the pointer back.
  const selection = win.getSelection();
  selection.selectAllChildren(taskId);
  pointerClick(win, taskId);
  assert.strictEqual(fired.inputs, 4, 'the selection is what was asked for');
  inp.setSelectionRange(0, inp.value.length);
  assert.ok(!key(win, taskId, 'Enter'));
  assert.strictEqual(inp.value, 'task-1', 'a key press is not a drag');
  assert.strictEqual(fired.inputs, 5);
  selection.selectAllChildren(doc.getElementById('meta-model'));
  inp.setSelectionRange(0, 0);
  pointerClick(win, taskId);
  assert.strictEqual(inp.value, 'task-1task-1');
  assert.strictEqual(fired.inputs, 6);
  selection.removeAllRanges();

  // A new run forgets the ids: plain text again.
  send(win, {type: 'clear', chat_id: 'chat-abc', tabId: TAB});
  for (const id of ['meta-chat-id', 'meta-task-id', 'meta-parent-id'])
    assert.deepStrictEqual(idButton(win, id), NO_BUTTON);
  click(win, doc.getElementById('meta-chat-id'));
  assert.strictEqual(inp.value, 'task-1task-1');
  assert.strictEqual(fired.inputs, 6, 'nothing to insert');
});

test('a double-click on an id selects it and leaves the message box as it was: the first click inserts at once, the second takes that back', () => {
  const wv = makePage();
  const win = wv.win;
  const doc = win.document;
  const TAB = readyTabId(wv.posted);
  send(win, {
    type: 'task_settings',
    settings: SETTINGS,
    tabId: TAB,
    taskId: 'task-1',
  });
  const taskId = doc.getElementById('meta-task-id');
  const selection = win.getSelection();
  const {inp, fired} = composerWith(win, '', 0, 0);
  /** The message box as [text, selection start, selection end]. */
  const box = () => [inp.value, inp.selectionStart, inp.selectionEnd];
  /** The second click of a double-click on *el*: its press selects the word. */
  const secondClick = el => {
    selection.selectAllChildren(el);
    pointerClick(win, el, 2);
    el.dispatchEvent(
      new win.MouseEvent('dblclick', {bubbles: true, detail: 2}),
    );
  };

  // An empty box: the first click inserts with no wait, the second
  // one empties the box again and the id stays selected.
  pointerClick(win, taskId);
  assert.deepStrictEqual(box(), ['task-1', 6, 6]);
  secondClick(taskId);
  assert.deepStrictEqual(box(), ['', 0, 0]);
  assert.strictEqual(String(selection), 'task-1');
  assert.strictEqual(fired.inputs, 2, 'the composer settled both times');
  // The third click of a triple-click has nothing left to take back.
  pointerClick(win, taskId, 3);
  assert.deepStrictEqual(box(), ['', 0, 0]);
  assert.strictEqual(fired.inputs, 2);
  selection.removeAllRanges();

  // A draft with a selected range gets its text and the range back,
  // also when the rows moved and the second click hit another id.
  inp.value = 'keep this draft';
  inp.setSelectionRange(5, 9);
  pointerClick(win, taskId);
  assert.deepStrictEqual(box(), ['keep task-1 draft', 11, 11]);
  secondClick(doc.getElementById('meta-chat-id'));
  assert.deepStrictEqual(box(), ['keep this draft', 5, 9]);
  selection.removeAllRanges();

  // What the page put into the box between the two clicks stays, and
  // the id with it.
  pointerClick(win, taskId);
  inp.value = 'keep task-1 draft!';
  secondClick(taskId);
  assert.strictEqual(inp.value, 'keep task-1 draft!');
  selection.removeAllRanges();

  // So does a box that was edited between the two clicks and reads as
  // the first click left it all the same: words dictated into it and
  // taken back (voice.js fires `input` for each) ...
  inp.value = 'keep this draft';
  inp.setSelectionRange(5, 9);
  pointerClick(win, taskId);
  for (const spoken of ['keep task-1 draft and', 'keep task-1 draft']) {
    inp.value = spoken;
    inp.dispatchEvent(new win.Event('input', {bubbles: true}));
  }
  secondClick(taskId);
  assert.strictEqual(inp.value, 'keep task-1 draft');
  selection.removeAllRanges();

  // ... or the prompt sent and recalled with no key, mouse or `input`
  // event: a soft keyboard's Enter, then a swipe up over the box.
  inp.value = 'keep this draft';
  inp.setSelectionRange(5, 9);
  pointerClick(win, taskId);
  inp.dispatchEvent(
    new win.InputEvent('beforeinput', {inputType: 'insertLineBreak'}),
  );
  assert.strictEqual(inp.value, '');
  const swipe = {touchstart: 100, touchend: 40};
  for (const type in swipe) {
    const touch = new win.Event(type);
    touch.touches = touch.changedTouches = [{clientX: 9, clientY: swipe[type]}];
    inp.dispatchEvent(touch);
  }
  secondClick(taskId);
  assert.strictEqual(inp.value, 'keep task-1 draft');
  selection.removeAllRanges();

  // A double-click whose first click missed the id is a click sequence
  // of its own: the id an earlier click inserted is not its to remove.
  inp.value = '';
  pointerClick(win, taskId);
  pointerClick(win, taskId.parentNode);
  secondClick(taskId);
  assert.strictEqual(inp.value, 'task-1');
  selection.removeAllRanges();

  // An example command is a line to run, not a text to copy: a
  // double-click leaves it in the box.
  const example = doc.querySelector('#meta-machines .meta-example-cmd');
  inp.value = '';
  pointerClick(win, example);
  secondClick(example);
  assert.strictEqual(inp.value, '/machine add user@host');
  selection.removeAllRanges();

  // Nor is the text of another chat that came on screen meanwhile,
  // though its box holds the very same text.
  inp.value = '';
  pointerClick(win, taskId);
  win._testApi.createNewTab();
  inp.value = 'task-1';
  secondClick(taskId);
  assert.strictEqual(inp.value, 'task-1');
});

test('a sub-agent\u2019s Task id inserts the bare id, and its Parent task id is a button too', () => {
  const wv = makePage();
  const win = wv.win;
  const doc = win.document;
  const TAB = readyTabId(wv.posted);
  send(win, {
    type: 'task_settings',
    settings: {
      chat_id: 'chat-abc',
      task_id: 'task-2',
      is_subagent: true,
      parent_task_id: 'task-1',
    },
    tabId: TAB,
    taskId: 'task-2',
  });
  assert.strictEqual(infoRows(win).taskId, 'task-2 (subagent)');
  assert.strictEqual(
    idButton(win, 'meta-task-id').label,
    'task-2 (subagent): insert into the message box',
  );
  assert.strictEqual(
    idButton(win, 'meta-parent-id').label,
    'task-1: insert into the message box',
  );
  const {inp} = composerWith(win, '', 0, 0);
  click(win, doc.getElementById('meta-task-id'));
  assert.strictEqual(inp.value, 'task-2', 'without the " (subagent)" mark');
  click(win, doc.getElementById('meta-parent-id'));
  assert.strictEqual(inp.value, 'task-2task-1');
  assert.deepStrictEqual([inp.selectionStart, inp.selectionEnd], [12, 12]);
});

test('in the phone drawer an id closes the drawer, unfolds the message box and lands in it', () => {
  const wv = makeHarnessPage({
    bodyClass: 'remote-chat',
    sourceUrl: 'taskinfo-main.js',
    beforeScripts(win) {
      stubRaf(win);
      fakeMatchMedia(win, false);
    },
  });
  openWindows.push(wv.win);
  const win = wv.win;
  const doc = win.document;
  const panel = doc.getElementById('meta-panel');
  send(win, {
    type: 'task_settings',
    settings: SETTINGS,
    tabId: readyTabId(wv.posted),
    taskId: 'task-1',
  });
  const {inp} = composerWith(win, 'about ', 6, 6);
  // The message box is folded away by its handle.
  const handle = doc.getElementById('input-drawer-btn');
  const inputArea = doc.getElementById('input-area');
  click(win, handle);
  assert.ok(inputArea.classList.contains('drawer-collapsed'));
  click(win, doc.getElementById('meta-drawer-btn'));
  assert.ok(panel.classList.contains('open'), 'the drawer is open');
  const chatId = doc.getElementById('meta-chat-id');
  chatId.focus();
  click(win, chatId);
  assert.ok(!panel.classList.contains('open'), 'the drawer closed');
  assert.ok(
    !inputArea.classList.contains('drawer-collapsed'),
    'the box the id went into is on screen',
  );
  assert.strictEqual(handle.getAttribute('aria-expanded'), 'true');
  assert.strictEqual(inp.value, 'about chat-abc');
  assert.deepStrictEqual([inp.selectionStart, inp.selectionEnd], [14, 14]);
  assert.strictEqual(doc.activeElement, inp);
});

test('task_events replay repopulates the info from its own stream', () => {
  const wv = makePage();
  const win = wv.win;
  const TAB = readyTabId(wv.posted);
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
  const TAB = readyTabId(wv.posted);
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

  // Audit 2026-11-11 (W1): a neighbour's settings are its own.  Task
  // ids number each daemon's own history, so a task of another tab (on
  // another machine) may carry the neighbour's id: its settings must
  // not replace the rows of the neighbour the reader is parked on.
  send(win, {type: 'tabs_state', tabs: [{tabId: TAB}, {tabId: 'other-tab'}]});
  send(win, {
    type: 'task_settings',
    tabId: 'other-tab',
    taskId: 'task-1',
    settings: Object.assign({}, SETTINGS, {model: 'other-tab-model'}),
  });
  win.document
    .getElementById('output')
    .dispatchEvent(new win.Event('scroll'));
  assert.ok(
    infoText(win).includes('old-model'),
    "BUG - another tab's same-id task replaced the neighbour's rows: " +
      infoText(win),
  );
});

test('a neighbour without settings shows dashes', () => {
  const wv = makePage();
  const win = wv.win;
  const TAB = readyTabId(wv.posted);
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

test('a neighbour fetched by a same-task tab belongs to the tab on screen', () => {
  const wv = makePage();
  const win = wv.win;
  const first = readyTabId(wv.posted);
  win._testApi.hideWelcome();
  const own = {
    type: 'task_events',
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
    ],
  };
  send(win, Object.assign({tabId: first}, own));
  win._testApi.createNewTab();
  send(win, Object.assign({tabId: win._testApi.getActiveTabId()}, own));
  // The answer to a fetch the first tab made before the switch: both
  // tabs show the task, so it is spliced into the one on screen.
  send(win, {
    type: 'adjacent_task_events',
    tabId: first,
    direction: 'prev',
    task: 'older neighbour',
    task_id: 'task-1',
    events: [
      {
        type: 'task_settings',
        settings: Object.assign({}, SETTINGS, {
          task_id: 'task-1',
          model: 'old-model',
        }),
      },
    ],
  });
  assert.strictEqual(
    win.document.querySelectorAll('#output .adjacent-task').length,
    1,
    'the neighbour is spliced into the transcript on screen',
  );
  clickTab(win, first);
  const rows = infoRows(win);
  assert.deepStrictEqual(
    [rows.model, rows.taskId],
    ['live-model', 'task-2'],
    "the hidden tab keeps its own task, not the neighbour's",
  );
});

test('a background tab keeps its settings and shows them on switch', () => {
  const wv = makePage();
  const win = wv.win;
  const TAB = readyTabId(wv.posted);
  send(win, {type: 'setTaskText', text: 'first task', tabId: TAB});
  send(win, {
    type: 'task_settings',
    settings: SETTINGS,
    tabId: TAB,
    taskId: 'task-1',
  });
  win._testApi.createNewTab();
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
  // A replay REPLACES the hidden tab's transcript too.  One that
  // throws half way hands the old transcript back, settings included;
  // one carrying no settings must not keep the previous task's, as on
  // the visible tab.
  win._testApi.createNewTab();
  const other = win._testApi.getActiveTabId();
  const errors = [];
  win.addEventListener('error', e => {
    errors.push(e);
    e.preventDefault();
  });
  const replay = {
    type: 'task_events',
    tabId: TAB,
    chat_id: 'chat-abc',
    task_id: 'task-9',
    task: 'settings-less task',
  };
  send(win, {...replay, events: [null]});
  assert.strictEqual(errors.length, 1, 'the malformed replay threw');
  clickTab(win, TAB);
  assert.ok(
    infoText(win).includes('bg-model'),
    'a failed replay hands the old settings back: ' + infoText(win),
  );
  clickTab(win, other);
  send(win, {...replay, events: [{type: 'system_output', text: 'x\n'}]});
  clickTab(win, TAB);
  assert.strictEqual(
    infoText(win),
    '',
    'a hidden replay without settings clears the rows',
  );
});

test('share export gives every task panel its own settings info', () => {
  const wv = makePage();
  const win = wv.win;
  const TAB = readyTabId(wv.posted);
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
  click(win, win.document.getElementById('share-btn'));
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
  openWindows.push(dom.window);
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

test('a saved page lists every Task Info row but Tags for every task', () => {
  const wv = makePage();
  const win = wv.win;
  const TAB = readyTabId(wv.posted);
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
  click(win, doc.getElementById('share-btn'));
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
  openWindows.push(dom.window);
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
  assert.strictEqual(done.Time, '2m 5s');
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
  const TAB = readyTabId(wv.posted);
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

// The names a task can carry, as the daemon lists them (`allTags`).
const ALL_TAGS = ['work', 'personal', 'coding', 'research', 'testing', 'review'];

/** The Tags row: its text, chips, remove buttons, select and locked controls. */
function tagsRow(win) {
  const el = win.document.getElementById('meta-tags');
  const add = el.querySelector('.meta-tag-add');
  const controls = Array.from(el.querySelectorAll('button, select'));
  return {
    text: el.textContent,
    chips: Array.from(el.querySelectorAll('.meta-tag'), c => c.firstChild.data),
    removable: Array.from(
      el.querySelectorAll('.meta-tag-remove'),
      b => b.dataset.tag,
    ),
    options: add ? Array.from(add.options, o => o.textContent) : null,
    locked: controls.length > 0 && controls.every(c => c.disabled),
  };
}

/** The page's getTaskTags / setTaskTags commands so far, without `type`. */
function tagCommands(posted, type) {
  return posted
    .filter(m => m.type === type)
    .map(m => Object.assign({}, m, {type: undefined}))
    .map(m => JSON.parse(JSON.stringify(m)));
}

/** Deliver the daemon's taskTags of task-1: `tags`, editable when it has any. */
function sendTags(win, tags, extra) {
  send(
    win,
    Object.assign(
      {
        type: 'taskTags',
        taskId: 'task-1',
        tags: tags,
        allTags: ALL_TAGS,
        editable: tags.length > 0,
      },
      extra,
    ),
  );
}

/** A page showing task-1, whose tags the daemon was asked for. */
function makeTagsPage() {
  const wv = makePage();
  const TAB = readyTabId(wv.posted);
  send(wv.win, {type: 'daemonStatus', connected: true});
  send(wv.win, {type: 'task_settings', settings: SETTINGS, tabId: TAB});
  return Object.assign(wv, {TAB: TAB});
}

test('the Tags row shows the tags of the task as chips, removable but for a lone one', () => {
  const wv = makePage();
  const win = wv.win;
  const TAB = readyTabId(wv.posted);
  send(win, {type: 'daemonStatus', connected: true});
  send(win, {type: 'tasks_updated'});
  assert.strictEqual(tagsRow(win).text, DASH, 'no task: the dash');
  assert.deepStrictEqual(
    tagCommands(wv.posted, 'getTaskTags'),
    [],
    'and nothing to ask the daemon about',
  );
  send(win, {type: 'task_settings', settings: SETTINGS, tabId: TAB});
  assert.deepStrictEqual(
    tagCommands(wv.posted, 'getTaskTags'),
    [{taskId: 'task-1'}],
    'the task shown changed: its tags are asked of this daemon',
  );
  send(win, {type: 'configData', config: {}, machine: 'box-a', tabId: TAB});
  assert.strictEqual(
    tagCommands(wv.posted, 'getTaskTags').length,
    1,
    'a repaint for the same task asks nothing',
  );
  sendTags(win, ['work'], {taskId: 'task-9'});
  assert.strictEqual(tagsRow(win).text, DASH, "another task's tags are dropped");
  sendTags(win, []);
  assert.deepStrictEqual(
    tagsRow(win),
    {text: DASH, chips: [], removable: [], options: null, locked: false},
    'a task still running has no tags: the dash and no control',
  );
  sendTags(win, ['work', 'coding', 'testing'], {tabId: ''});
  const row = tagsRow(win);
  assert.deepStrictEqual(row.chips, ['work', 'coding', 'testing']);
  assert.deepStrictEqual(row.removable, ['work', 'coding', 'testing']);
  assert.deepStrictEqual(
    row.options,
    ['+ Add tag', 'personal', 'research', 'review'],
    'the select offers the names the task does not carry, in their order',
  );
  const remove = win.document.querySelector('.meta-tag-remove');
  assert.strictEqual(remove.getAttribute('aria-label'), 'Remove tag work');
  assert.strictEqual(remove.dataset.tooltip, 'Remove tag work');
  assert.strictEqual(remove.type, 'button');
  assert.strictEqual(
    win.document.querySelector('.meta-tag-add').getAttribute('aria-label'),
    'Add a tag',
  );
  sendTags(win, ['work']);
  assert.deepStrictEqual(
    [tagsRow(win).chips, tagsRow(win).removable, tagsRow(win).options.length],
    [['work'], [], ALL_TAGS.length],
    'a task keeps one tag: the lone chip has no remove button',
  );
  sendTags(win, ['work'], {allTags: ALL_TAGS.concat('docs')});
  assert.strictEqual(
    tagsRow(win).options.pop(),
    'docs',
    'the same tags from a daemon that knows one more name: the select offers it',
  );
  sendTags(win, ALL_TAGS);
  assert.deepStrictEqual(
    [tagsRow(win).removable, tagsRow(win).options],
    [ALL_TAGS, null],
    'no name left to add: no select',
  );
  sendTags(win, ['work', 'coding'], {editable: false});
  assert.deepStrictEqual(
    tagsRow(win),
    {
      text: 'workcoding',
      chips: ['work', 'coding'],
      removable: [],
      options: null,
      locked: false,
    },
    'tags the daemon will not take an edit of are chips without controls',
  );
  send(win, {type: 'daemonStatus', connected: false});
  assert.strictEqual(
    tagsRow(win).text,
    'workcoding',
    'an outage with no edit awaiting its answer leaves the row as it is',
  );
  send(win, {type: 'daemonStatus', connected: true});
  sendTags(win, ['work', 'coding']);
  assert.deepStrictEqual(
    [tagsRow(win).removable, tagsRow(win).options.length],
    [['work', 'coding'], ALL_TAGS.length - 1],
    'the task ended: the same tags, now with the controls that edit them',
  );
});

test('removing and adding a tag send the new set and lock the row until the answer', () => {
  const wv = makeTagsPage();
  const win = wv.win;
  const doc = win.document;
  sendTags(win, ['work', 'coding', 'testing']);
  click(win, doc.querySelector('.meta-tag'));
  assert.deepStrictEqual(
    tagCommands(wv.posted, 'setTaskTags'),
    [],
    'a click on a chip itself edits nothing',
  );
  const remove = doc.querySelector('.meta-tag-remove[data-tag="coding"]');
  remove.focus();
  click(win, remove);
  assert.deepStrictEqual(
    tagCommands(wv.posted, 'setTaskTags'),
    [{taskId: 'task-1', tags: ['work', 'testing']}],
    'the remove button sends the set without its tag',
  );
  assert.deepStrictEqual(
    [tagsRow(win).chips, tagsRow(win).locked],
    [['work', 'coding', 'testing'], true],
    'the row shows the stored tags, its controls locked, until the answer',
  );
  sendTags(win, ['work', 'coding', 'testing']);
  assert.strictEqual(
    tagsRow(win).locked,
    true,
    'a taskTags sent before the edit landed is not its answer',
  );
  sendTags(win, ['work', 'testing'], {tabId: ''});
  assert.deepStrictEqual(
    [tagsRow(win).chips, tagsRow(win).locked],
    [['work', 'testing'], false],
    'the set stored is the answer: the row shows it and works again',
  );
  const add = doc.querySelector('.meta-tag-add');
  assert.strictEqual(
    doc.activeElement,
    add,
    'the keyboard focus the edit took from the row comes back to it',
  );
  // The history reload a stored edit causes asks for the tags again.
  send(win, {type: 'tasks_updated'});
  sendTags(win, ['work', 'testing']);
  assert.strictEqual(
    doc.activeElement,
    add,
    'a reply repeating what the row shows leaves its elements, and the focus, in place',
  );
  add.dispatchEvent(new win.Event('change', {bubbles: true}));
  assert.strictEqual(
    tagCommands(wv.posted, 'setTaskTags').length,
    1,
    'the placeholder is no pick',
  );
  add.value = 'review';
  add.dispatchEvent(new win.Event('change', {bubbles: true}));
  assert.deepStrictEqual(
    tagCommands(wv.posted, 'setTaskTags')[1],
    {taskId: 'task-1', tags: ['work', 'testing', 'review']},
    'a pick sends the set with the name picked',
  );
  doc.getElementById('task-input').focus();
  sendTags(win, ['work', 'testing', 'review']);
  assert.deepStrictEqual(
    [tagsRow(win).chips, doc.querySelector('.meta-tag-add').value],
    [['work', 'testing', 'review'], ''],
    'the answer repaints the chips and the select is back on its placeholder',
  );
  assert.strictEqual(
    doc.activeElement.id,
    'task-input',
    'the focus stays where the user moved it meanwhile',
  );
});

test('a refused edit repaints the row from the stored tags and says why', () => {
  const wv = makeTagsPage();
  const win = wv.win;
  const doc = win.document;
  const toast = () =>
    doc.querySelector('.kiss-notification[data-notification-id="task-tags-error"]');
  sendTags(win, [], {error: 'Could not read the task\u2019s tags.'});
  assert.strictEqual(toast(), null, 'a failed read is not announced');
  sendTags(win, ['work', 'coding']);
  // A mouse click, which focuses the button it presses in Chromium.
  doc.querySelector('.meta-tag-remove[data-tag="work"]').focus();
  pointerClick(win, doc.querySelector('.meta-tag-remove[data-tag="work"]'));
  sendTags(win, ['work', 'coding'], {error: 'Could not save the tags.'});
  assert.deepStrictEqual(
    [tagsRow(win).chips, tagsRow(win).locked],
    [['work', 'coding'], false],
    'the row shows the tags the daemon kept and works again',
  );
  assert.strictEqual(
    doc.activeElement,
    doc.body,
    'a mouse edit hands no keyboard focus to the select, which a typed letter picks from',
  );
  assert.ok(
    toast().textContent.includes('Could not save the tags.'),
    'the reason is a notification',
  );
  // A registered machine that could not be asked lists no tags.
  click(win, doc.querySelector('.meta-tag-remove[data-tag="work"]'));
  send(win, {type: 'taskTags', taskId: 'task-1'});
  assert.strictEqual(
    tagsRow(win).locked,
    true,
    'a taskTags naming neither tags nor a reason is no answer',
  );
  send(win, {type: 'taskTags', taskId: 'task-1', error: 'box is unreachable'});
  assert.deepStrictEqual(
    [tagsRow(win).chips, tagsRow(win).locked],
    [['work', 'coding'], false],
    'a refusal without tags leaves the chips as they are',
  );
  assert.ok(toast().textContent.includes('box is unreachable'));
});

test('the Tags row follows the task, its machine and the daemon connection', () => {
  const wv = makeTagsPage();
  const win = wv.win;
  const doc = win.document;
  sendTags(win, ['work', 'coding']);
  send(win, {type: 'tabMachine', tabId: wv.TAB, machine: 'remote-1'});
  assert.deepStrictEqual(
    [tagCommands(wv.posted, 'getTaskTags').pop(), tagsRow(win).text],
    [{taskId: 'task-1', machine: 'remote-1'}, DASH],
    'a tab bound to a machine asks that machine: the row is its history\u2019s',
  );
  sendTags(win, ['work', 'coding']);
  assert.strictEqual(tagsRow(win).text, DASH, "this daemon's answer is dropped");
  sendTags(win, ['work', 'coding'], {machine: 'remote-1'});
  send(win, {
    type: 'taskTags',
    taskId: 'task-1',
    machine: 'remote-1',
    error: 'remote-1 is unreachable',
  });
  assert.deepStrictEqual(
    [
      tagsRow(win).chips,
      doc.querySelector('[data-notification-id="task-tags-error"]'),
    ],
    [['work', 'coding'], null],
    'a later read the machine could not answer: the tags stay, nothing is announced',
  );
  click(win, doc.querySelector('.meta-tag-remove[data-tag="work"]'));
  assert.deepStrictEqual(
    tagCommands(wv.posted, 'setTaskTags'),
    [{taskId: 'task-1', tags: ['coding'], machine: 'remote-1'}],
    'and stores the edit there',
  );
  // The answer to an edit sent on a dropped connection never comes.
  send(win, {type: 'daemonStatus', connected: false});
  assert.strictEqual(tagsRow(win).locked, false, 'an outage unlocks the row');
  const asked = tagCommands(wv.posted, 'getTaskTags').length;
  send(win, {type: 'daemonStatus', connected: true});
  assert.strictEqual(
    tagCommands(wv.posted, 'getTaskTags').length,
    asked + 1,
    'the connection is back: the tags are asked again',
  );
  send(win, {type: 'tasks_updated'});
  assert.strictEqual(
    tagCommands(wv.posted, 'getTaskTags').length,
    asked + 2,
    'task news (a task ended, an edit elsewhere) asks again',
  );
  // An edit awaiting its answer belongs to the task it was made on.
  click(win, doc.querySelector('.meta-tag-remove[data-tag="work"]'));
  send(win, {
    type: 'task_settings',
    settings: Object.assign({}, SETTINGS, {task_id: 'task-2'}),
    tabId: wv.TAB,
  });
  assert.deepStrictEqual(
    [tagCommands(wv.posted, 'getTaskTags').pop(), tagsRow(win).text],
    [{taskId: 'task-2', machine: 'remote-1'}, DASH],
    'another task: its tags are asked and the row starts from the dash',
  );
  sendTags(win, ['coding'], {machine: 'remote-1'});
  assert.strictEqual(tagsRow(win).text, DASH, 'the left task\u2019s answer is dropped');
  sendTags(win, ['review'], {machine: 'remote-1', taskId: 'task-2'});
  assert.deepStrictEqual(
    [tagsRow(win).chips, tagsRow(win).locked],
    [['review'], false],
    'the new task\u2019s row is not locked by the edit of the task left',
  );
});

test('share markup: the info block follows the task text', () => {
  const wv = makePage();
  const win = wv.win;
  const TAB = readyTabId(wv.posted);
  send(win, {type: 'clear', chat_id: 'chat-1', tabId: TAB});
  send(win, {type: 'setTaskText', text: 'a task', tabId: TAB});
  click(win, win.document.getElementById('share-btn'));
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
  openWindows.push(dom.window);
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
  loadMainCss(dom.window);
  assert.strictEqual(
    dom.window.getComputedStyle(info).display,
    'none',
    'the info block hides when empty',
  );
  info.textContent = 'model: m';
  assert.notStrictEqual(
    dom.window.getComputedStyle(info).display,
    'none',
    'a filled info block shows',
  );
});

report('taskPanelSettingsInfo');
