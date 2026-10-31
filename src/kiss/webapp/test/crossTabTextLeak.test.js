// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Text must never cross tabs. Two chat tabs are two conversations, and the
// only text a tab may show is the text that belongs to it. The page keeps
// exactly ONE copy of every surface -- #output, #task-input, the autocomplete
// dropdown, the ghost overlay, the file-link spans, the toast container -- and
// swaps tab snapshots in and out of them. So any handler that writes a
// surface without first proving the message belongs to the tab on screen
// attributes another task's text to the wrong conversation, durably (the next
// saveCurrentTab() bakes it into the wrong tab's snapshot).
//
// This file drives distinctly-marked traffic at one tab while another is on
// screen and pins down that neither tab ever shows the other's marker. It
// also pins down the two things the fix must NOT break: two tabs bound to the
// same backend chat legitimately share text, and genuinely global messages
// (daemon status, remote URL, models, welcome suggestions, history, and the
// user-initiated input commands) still reach the active tab.

'use strict';

const assert = require('assert');
const {
  makePage: makeChatPage,
  clickTab,
  send,
  stubRaf,
  twoTabs,
} = require('./simplify2_harness');
const {makeRunner} = require('./ui_antipattern_harness');
const {
  makeVoicePage,
  micBtn,
  startMic,
  wake,
  speak,
  fireTimer,
} = require('./voice_page_helpers');

const SOURCE_URL = {
  'main.js': 'tableak-main.js',
  'voice.js': 'tableak-voice.js',
};

// With `voice`, the page runs voice.js over the stubbed in-page voice
// pipeline (voice_page_helpers), its mic remembered ON so the pipeline
// starts on load.
function makePage(opts) {
  const options = opts || {};
  if (options.voice) {
    return makeVoicePage({enabled: true, sourceUrl: SOURCE_URL});
  }
  function beforeScripts(win) {
    stubRaf(win);
    win.__VOICE__ = {
      voskSrc: '/media/vosk.js',
      modelUrl: 'https://models.example.com/model.tar.gz',
    };
    win.localStorage.setItem('kissVoiceEnabled', '1');
  }
  const {win, posted} = makeChatPage({beforeScripts, sourceUrl: SOURCE_URL});
  return {win, posted};
}

// The label a tab carries (what its Chats-panel row / strip entry shows).
// Chat tabs have no row of their own any more, so the label is read from
// the tab records rather than from a rendered element.
function tabTitle(win, tabId) {
  const rec = win._testApi.openTabs().find(t => t.id === tabId);
  assert.ok(rec, `tab ${tabId} must be open`);
  return rec.title;
}

function visibleText(win) {
  return win.document.getElementById('output').textContent;
}

function taskInput(win) {
  return win.document.getElementById('task-input');
}

function setInput(win, text) {
  const inp = taskInput(win);
  inp.value = text;
  inp.setSelectionRange(text.length, text.length);
  return inp;
}

function autocompleteEl(win) {
  return win.document.getElementById('autocomplete');
}

function acTexts(win) {
  return Array.from(autocompleteEl(win).querySelectorAll('.ac-item')).map(el =>
    el.getAttribute('data-text'),
  );
}

function ghostText(win) {
  const overlay = win.document.getElementById('ghost-overlay');
  const el = overlay && overlay.querySelector('.ghost-text');
  return el ? el.textContent : '';
}

function toastText(win) {
  const container = win.document.getElementById('kiss-notification-container');
  return container ? container.textContent : '';
}

// Every test is queued and run in order by the runner at the bottom: the
// voice tests drive the async pipeline start, so they run one at a time
// after the synchronous ones.
const tests = [];

function test(name, fn) {
  tests.push({name, fn});
}

// ---------------------------------------------------------------------------
// Leak 1 -- pathsExist sweeps the whole document and ignores ev.tabId.
// ---------------------------------------------------------------------------

// Both tabs of a leak-1 test live in the same workspace: that is the
// overwhelmingly common case and the one where a reply for one tab can
// resolve another tab's spans.
function sameWorkspace(win) {
  send(win, {
    type: 'configData',
    config: {work_dir: '/tmp/wd-leak1'},
    apiKeys: {},
  });
}

test('a pathsExist reply for one tab never resolves another tab file links', () => {
  const {win} = makePage();
  sameWorkspace(win);
  const {first, second} = twoTabs(win);

  // Both tabs mention the same relative path under the same workDir. Only
  // tab `first` asked; the answer must not touch what tab `second` shows.
  send(win, {
    type: 'system_output',
    text: 'touched leak1dir/alpha_QK41.txt here',
    tabId: first,
  });
  send(win, {
    type: 'system_output',
    text: 'touched leak1dir/alpha_QK41.txt here',
    tabId: second,
  });

  // The reply is addressed to `first`; `second` is on screen.
  send(win, {
    type: 'pathsExist',
    workDir: '/tmp/wd-leak1',
    tabId: first,
    results: {'leak1dir/alpha_QK41.txt': true},
  });

  const promotedInSecond = win.document.querySelectorAll(
    '#output [data-path="leak1dir/alpha_QK41.txt"]',
  );
  assert.strictEqual(
    promotedInSecond.length,
    0,
    'a pathsExist reply addressed to a background tab must not promote ' +
      'the visible tab file-link candidates',
  );

  win.close();
});

test('a background tab keeps its own file-link candidates checkable', () => {
  const {win, posted} = makePage();
  sameWorkspace(win);
  const {first, second} = twoTabs(win);

  send(win, {
    type: 'system_output',
    text: 'wrote leak1dir/beta_QK42.txt now',
    tabId: second,
  });
  posted.length = 0;
  // The same path in another tab must still produce its own checkPaths
  // request -- the in-flight de-dup must not starve a second tab.
  send(win, {
    type: 'system_output',
    text: 'wrote leak1dir/beta_QK42.txt now',
    tabId: first,
  });
  const asked = posted.filter(
    m => m.type === 'checkPaths' && m.paths.includes('leak1dir/beta_QK42.txt'),
  );
  assert.ok(
    asked.length > 0,
    'a second tab must be allowed to check a path already in flight for ' +
      'another tab, otherwise its links stay permanently inert',
  );
  assert.ok(
    asked.every(m => m.tabId !== undefined),
    'every checkPaths request must name the tab that asked',
  );

  win.close();
});

// ---------------------------------------------------------------------------
// Leak 2 -- files / ghost / completions correlate on content, not identity.
// ---------------------------------------------------------------------------

test('an @-mention file list for one tab never opens over another tab', () => {
  const {win} = makePage();
  const {first, second} = twoTabs(win);

  // Both tabs happen to hold the same half-typed mention.
  setInput(win, 'look at @src');

  send(win, {
    type: 'files',
    prefix: 'src',
    tabId: first,
    files: [{type: 'file', text: 'src/leak2_files_QK43.py'}],
  });

  assert.ok(
    !acTexts(win).includes('src/leak2_files_QK43.py'),
    'a file list answered for a background tab must never render in the ' +
      'dropdown of the tab on screen',
  );
  assert.strictEqual(second, win._testApi.getActiveTabId());

  win.close();
});

test('the @-mention request names the tab that typed it', () => {
  const {win, posted} = makePage();
  const {second} = twoTabs(win);

  posted.length = 0;
  const inp = setInput(win, 'read @lib');
  inp.dispatchEvent(new win.Event('input', {bubbles: true}));

  const reqs = posted.filter(m => m.type === 'getFiles');
  assert.ok(reqs.length > 0, 'typing @ must ask the host for files');
  assert.ok(
    reqs.every(m => m.tabId === second),
    'every getFiles request must carry the id of the tab that typed it: ' +
      JSON.stringify(reqs),
  );

  win.close();
});

test('a ghost suggestion for one tab never paints over another tab input', () => {
  const {win} = makePage();
  const {first} = twoTabs(win);

  setInput(win, 'run the');

  send(win, {
    type: 'ghost',
    query: 'run the',
    tabId: first,
    suggestion: ' leak2_ghost_QK44 suite',
  });

  assert.ok(
    !ghostText(win).includes('leak2_ghost_QK44'),
    'a ghost completion computed for a background tab must never be ' +
      'painted into the visible input overlay',
  );

  win.close();
});

test('an inline completion list for one tab never opens over another tab', () => {
  const {win} = makePage();
  const {first} = twoTabs(win);

  setInput(win, 'deploy');

  send(win, {
    type: 'completions',
    query: 'deploy',
    tabId: first,
    completions: [{type: 'task', text: 'deploy leak2_comp_QK45 service'}],
  });

  assert.ok(
    !acTexts(win).includes('deploy leak2_comp_QK45 service'),
    'a completion list computed for a background tab must never render ' +
      'in the dropdown of the tab on screen',
  );

  win.close();
});

// ---------------------------------------------------------------------------
// Leak 3 -- fileContent ignores tabId and can steal the view.
// ---------------------------------------------------------------------------

test('a fileContent reply for a background tab never steals the view', () => {
  const {win} = makePage();
  const {first, second} = twoTabs(win);

  send(win, {
    type: 'fileContent',
    tabId: first,
    name: 'leak3_QK46.txt',
    path: '/tmp/leak3_QK46.txt',
    content: 'leak3 body QK46',
  });

  assert.strictEqual(
    win._testApi.getActiveTabId(),
    second,
    'a file opened for a background task must not pull the user away ' +
      'from the tab they are reading',
  );

  win.close();
});

test('a fileContent error for a background tab never raises a toast', () => {
  const {win} = makePage();
  const {first} = twoTabs(win);

  send(win, {
    type: 'fileContent',
    tabId: first,
    path: '/tmp/leak3_err_QK47.txt',
    error: 'cannot open leak3_err_QK47',
  });

  assert.ok(
    !toastText(win).includes('leak3_err_QK47'),
    'a file-open failure belonging to a background task must not be ' +
      'reported over the tab on screen',
  );

  win.close();
});

// ---------------------------------------------------------------------------
// Leak 4 -- followup_suggestion appends task text and fails OPEN.
// ---------------------------------------------------------------------------

test('an untagged followup suggestion never lands in the active tab', () => {
  const {win} = makePage();
  twoTabs(win);

  send(win, {type: 'followup_suggestion', text: 'leak4_untagged_QK48'});

  assert.ok(
    !visibleText(win).includes('leak4_untagged_QK48'),
    'a suggested-next task with no tab attribution must be dropped, not ' +
      'appended to whichever conversation happens to be on screen',
  );

  win.close();
});

test('a followup suggestion for a background tab never lands in the active tab', () => {
  const {win} = makePage();
  const {first, second} = twoTabs(win);

  send(win, {
    type: 'followup_suggestion',
    tabId: first,
    text: 'leak4_tagged_QK49',
  });

  assert.ok(
    !visibleText(win).includes('leak4_tagged_QK49'),
    'a suggested-next task for a background tab must not render in the ' +
      'visible transcript',
  );

  clickTab(win, first);
  clickTab(win, second);
  assert.ok(
    !visibleText(win).includes('leak4_tagged_QK49'),
    'switching tabs must not carry the suggestion into the other tab',
  );

  win.close();
});

test('a followup suggestion for the active tab still shows', () => {
  const {win} = makePage();
  const {second} = twoTabs(win);

  send(win, {
    type: 'followup_suggestion',
    tabId: second,
    text: 'leak4_ownn_QK50',
  });

  assert.ok(
    visibleText(win).includes('leak4_ownn_QK50'),
    'a tab must still see the suggestion computed for itself',
  );

  win.close();
});

// ---------------------------------------------------------------------------
// Leak 5 -- notification only checks the tab EXISTS.
// ---------------------------------------------------------------------------

test('a notification for a background tab never pops over the active tab', () => {
  const {win} = makePage();
  const {first} = twoTabs(win);

  send(win, {
    type: 'notification',
    tabId: first,
    id: 'leak5-toast',
    message: 'leak5_notify_QK51',
    severity: 'info',
  });

  assert.ok(
    !toastText(win).includes('leak5_notify_QK51'),
    'a toast belonging to a background task must not be shown over the ' +
      'tab the user is reading',
  );

  win.close();
});

test('an app-level notification with no tab still shows', () => {
  const {win} = makePage();
  twoTabs(win);

  send(win, {
    type: 'notification',
    id: 'global-toast',
    message: 'global_notify_QK52',
    severity: 'info',
  });

  assert.ok(
    toastText(win).includes('global_notify_QK52'),
    'window-level notifications must keep working',
  );

  win.close();
});

// ---------------------------------------------------------------------------
// Leak 6 -- the default branch streams untagged task events into the active
// tab.
// ---------------------------------------------------------------------------

// Each entry is a small burst of untagged task-scoped traffic whose marker
// becomes visible text in whatever container the handler picks.
const TASK_SCOPED_UNTAGGED = [
  {
    name: 'text_delta',
    evs: [{type: 'text_delta', text: 'leak6_text_QK53'}],
    marker: 'leak6_text_QK53',
  },
  {
    name: 'thinking_start/thinking_delta',
    evs: [
      {type: 'thinking_start'},
      {type: 'thinking_delta', text: 'leak6_think_QK54'},
    ],
    marker: 'leak6_think_QK54',
  },
  {
    name: 'tool_call',
    evs: [{type: 'tool_call', name: 'Bash', command: 'leak6_tool_QK56'}],
    marker: 'leak6_tool_QK56',
  },
  {
    name: 'tool_result',
    evs: [
      {type: 'tool_call', name: 'Read', path: '/tmp/leak6.txt'},
      {type: 'tool_result', content: 'leak6_result_QK57', is_error: false},
    ],
    marker: 'leak6_result_QK57',
  },
  {
    name: 'system_output',
    evs: [{type: 'system_output', text: 'leak6_sysout_QK58'}],
    marker: 'leak6_sysout_QK58',
  },
  {
    name: 'prompt',
    evs: [{type: 'prompt', text: 'leak6_prompt_QK59'}],
    marker: 'leak6_prompt_QK59',
  },
  {
    name: 'result',
    evs: [
      {
        type: 'result',
        summary: 'leak6_summary_QK75',
        success: true,
        total_tokens: 1,
        cost: '$0.01',
      },
    ],
    marker: 'leak6_summary_QK75',
  },
];

for (const c of TASK_SCOPED_UNTAGGED) {
  test(`an untagged ${c.name} never streams into the active tab`, () => {
    const {win} = makePage();
    twoTabs(win);

    for (const ev of c.evs) send(win, Object.assign({}, ev));

    assert.ok(
      !visibleText(win).includes(c.marker),
      `an untagged ${c.name} carries no tab attribution, so it must be ` +
        'dropped rather than attributed to whichever conversation is on ' +
        'screen',
    );

    win.close();
  });

  test(`a tagged ${c.name} still streams into its own tab`, () => {
    const {win} = makePage();
    const {second} = twoTabs(win);

    for (const ev of c.evs) {
      send(win, Object.assign({}, ev, {tabId: second}));
    }

    assert.ok(
      visibleText(win).includes(c.marker),
      `a ${c.name} addressed to the tab on screen must still render`,
    );

    win.close();
  });
}

test('a tagged task-scoped event still renders in its own tab', () => {
  const {win} = makePage();
  const {first, second} = twoTabs(win);

  send(win, {type: 'system_output', text: 'own_sysout_QK60', tabId: second});
  assert.ok(
    visibleText(win).includes('own_sysout_QK60'),
    'a tab must still see its own streaming output',
  );

  send(win, {type: 'system_output', text: 'bg_sysout_QK61', tabId: first});
  assert.ok(
    !visibleText(win).includes('bg_sysout_QK61'),
    'a background tab output must stay in that background tab',
  );

  clickTab(win, first);
  assert.ok(
    visibleText(win).includes('bg_sysout_QK61'),
    'the background tab must show its own output once visited',
  );
  assert.ok(
    !visibleText(win).includes('own_sysout_QK60'),
    'the other tab text must not have followed the user',
  );

  clickTab(win, second);
  assert.ok(
    visibleText(win).includes('own_sysout_QK60'),
    'the first tab must still hold its own output',
  );
  assert.ok(
    !visibleText(win).includes('bg_sysout_QK61'),
    'and must still be free of the other tab text',
  );

  win.close();
});

// A message with no tabId is still addressed when it names a task: the tab
// running that task may show it, any other tab may not.
test('an untagged event whose taskId matches the visible task renders', () => {
  const {win} = makePage();
  const {first, second} = twoTabs(win);

  // The visible tab adopts task QK70 by receiving one addressed event.
  send(win, {
    type: 'system_output',
    text: 'adopt_QK70',
    tabId: second,
    taskId: 'task-QK70',
  });
  send(win, {
    type: 'system_output',
    text: 'same_task_QK71',
    taskId: 'task-QK70',
  });
  assert.ok(
    visibleText(win).includes('same_task_QK71'),
    'a taskId that names the visible task addresses it as well as a tabId',
  );

  clickTab(win, first);
  assert.ok(
    !visibleText(win).includes('same_task_QK71'),
    'and the text must not have leaked into the other tab',
  );

  win.close();
});

test('an untagged event for a different task never lands in the visible tab', () => {
  const {win} = makePage();
  const {first, second} = twoTabs(win);

  send(win, {
    type: 'system_output',
    text: 'adopt_QK72',
    tabId: second,
    taskId: 'task-QK72',
  });
  send(win, {
    type: 'system_output',
    text: 'other_task_QK73',
    taskId: 'task-QK73',
  });
  assert.ok(
    !visibleText(win).includes('other_task_QK73'),
    'text of a task the visible tab is not running must be dropped',
  );

  clickTab(win, first);
  assert.ok(
    !visibleText(win).includes('other_task_QK73'),
    'and it must not have been parked in the other tab either',
  );

  win.close();
});

// ---------------------------------------------------------------------------
// Leak 6b -- an OWNERLESS visible tab. "I have adopted no task yet" is not
// proof of ownership, and accepting a foreign task id there is worse than a
// one-off leak: the streaming branch then ADOPTS that id, so the tab is
// permanently mis-bound to a task it never started.
//
// Two things must hold at once. A transcript type must be dropped, because
// only the tab that started the task can own its words. And no untagged
// message may cause adoption, because a tab learns which task it is running
// from the request it sent (pendingTaskId) or from an explicitly addressed
// reply -- never from a bare task id off the wire.
// ---------------------------------------------------------------------------

// The task a tab has claimed, as main.js reports it to voice.js: '' when the
// tab has adopted nothing.
function ownedTaskId(win) {
  return win.kissVoiceOwner().taskId;
}

test('an untagged foreign transcript never enters an unowned visible tab', () => {
  const {win} = makePage();
  const {first, second} = twoTabs(win);

  // Tab `first` is running task-A. Tab `second` is on screen and has
  // adopted nothing at all.
  send(win, {
    type: 'system_output',
    text: 'a_own_QK85',
    tabId: first,
    taskId: 'task-A-QK85',
  });
  clickTab(win, second);
  assert.strictEqual(
    ownedTaskId(win),
    '',
    'the probe requires the visible tab to own no task',
  );

  send(win, {
    type: 'system_output',
    taskId: 'task-A-QK85',
    text: 'FOREIGN_QK85',
  });

  assert.ok(
    !visibleText(win).includes('FOREIGN_QK85'),
    "another task's transcript must not render in a tab that never started " +
      'it, however little that tab knows about itself',
  );
  assert.strictEqual(
    ownedTaskId(win),
    '',
    'nor may the visible tab adopt a task id that arrived with no tabId: ' +
      'that mis-binds the tab permanently',
  );

  win.close();
});

test('an unowned visible tab still shows its own header counters', () => {
  const {win} = makePage();
  twoTabs(win);

  // The daemon reports header counters before the page has learned the
  // task id, and neither type is adopted, so both must still land.
  send(win, {
    type: 'usage_info',
    taskId: 'task-HDR-QK86',
    text: 'Steps: 3/100, Tokens: 7,000/400,000, Budget: $0.7000/$9.00, ',
    total_tokens: 7000,
    cost: '$0.7000',
    total_steps: 3,
  });

  assert.strictEqual(
    win.document.getElementById('status-tokens').textContent,
    'Tokens: 7.00K',
    'a pre-adoption usage_info must still drive the header of the tab it ' +
      'is reported for',
  );
  assert.strictEqual(
    ownedTaskId(win),
    '',
    'and reading a header counter must not bind the tab to that task',
  );

  win.close();
});

test('an untagged foreign result panel never enters an unowned tab', () => {
  const {win} = makePage();
  const {first, second} = twoTabs(win);

  // `result` may precede adoption, so it is admitted while the tab owns
  // nothing -- but the instant the tab owns a task, a foreign one is out.
  send(win, {
    type: 'system_output',
    text: 'own_start_QK87',
    tabId: second,
    taskId: 'task-MINE-QK87',
  });
  assert.strictEqual(ownedTaskId(win), 'task-MINE-QK87');

  send(win, {
    type: 'result',
    taskId: 'task-OTHER-QK87',
    summary: 'FOREIGN_RESULT_QK87',
    success: true,
    total_tokens: 5,
    cost: '$0.05',
  });

  assert.ok(
    !visibleText(win).includes('FOREIGN_RESULT_QK87'),
    'an owned tab must reject a foreign result panel',
  );
  clickTab(win, first);
  assert.ok(
    !visibleText(win).includes('FOREIGN_RESULT_QK87'),
    'and it must not have been parked in the other tab either',
  );

  win.close();
});

// Findings #12 and #13 route through the same predicate, so each of them is
// re-probed here with a taskId and NO tabId against an unowned visible tab.
const TASK_ONLY_DIAGNOSTICS = [
  {
    name: 'error',
    ev: {type: 'error', text: 'FOREIGN_ERR_QK88'},
    marker: 'FOREIGN_ERR_QK88',
  },
  {
    name: 'notice',
    ev: {type: 'notice', text: 'FOREIGN_NOTICE_QK88'},
    marker: 'FOREIGN_NOTICE_QK88',
  },
  {
    name: 'warning',
    ev: {type: 'warning', message: 'FOREIGN_WARN_QK88'},
    marker: 'FOREIGN_WARN_QK88',
  },
];

for (const c of TASK_ONLY_DIAGNOSTICS) {
  test(`a taskId-only ${c.name} never prints in an unowned visible tab`, () => {
    const {win} = makePage();
    const {first, second} = twoTabs(win);

    send(win, {
      type: 'system_output',
      text: 'diag_owner_QK88',
      tabId: first,
      taskId: 'task-DIAG-QK88',
    });
    clickTab(win, second);
    assert.strictEqual(ownedTaskId(win), '');

    send(win, Object.assign({taskId: 'task-DIAG-QK88'}, c.ev));

    assert.ok(
      !visibleText(win).includes(c.marker),
      `a ${c.name} belonging to another task must not print in a tab that ` +
        'happens to own nothing yet',
    );

    win.close();
  });
}

test('a taskId-only setTaskText never retitles an unowned visible tab', () => {
  const {win} = makePage();
  const {first, second} = twoTabs(win);

  send(win, {
    type: 'system_output',
    text: 'hdr_owner_QK89',
    tabId: first,
    taskId: 'task-HDR-QK89',
  });
  clickTab(win, second);
  assert.strictEqual(ownedTaskId(win), '');

  send(win, {
    type: 'setTaskText',
    taskId: 'task-HDR-QK89',
    text: 'foreign header QK89',
  });

  assert.ok(
    !tabTitle(win, second).includes('foreign header QK89'),
    "another task's header must not rename a tab that owns no task",
  );

  win.close();
});

test('a taskId-only adjacent transcript never replays in an unowned tab', () => {
  const {win} = makePage();
  const {first, second} = twoTabs(win);

  send(win, {
    type: 'system_output',
    text: 'adj_owner_QK90',
    tabId: first,
    taskId: 'task-ADJ-QK90',
  });
  clickTab(win, second);
  assert.strictEqual(ownedTaskId(win), '');

  send(win, {
    type: 'adjacent_task_events',
    direction: 'prev',
    taskId: 'task-ADJ-QK90',
    task: 'foreign neighbour QK90',
    task_id: 'task-ADJ-PREV-QK90',
    events: [{type: 'text_delta', text: 'ADJACENT_FOREIGN_QK90'}],
  });

  assert.ok(
    !visibleText(win).includes('ADJACENT_FOREIGN_QK90'),
    "another task's neighbour must not splice itself into a tab that owns " +
      'no task',
  );

  win.close();
});

// ---------------------------------------------------------------------------
// Leak 7 -- a transcript submitted after the user switched tabs.
//
// The page records the conversation on screen when a capture begins
// (beginCapture -> markSpeechStart), posts the audio as `voiceTranscribe`
// and pairs the daemon's `voiceSpeech` reply with that owner. These tests
// drive the real in-page pipeline: the mic is granted, the wake word fires
// through the recognizer's own result handler, and audio blocks run the
// real capture logic until the audio is posted. Only then does the tab
// switch happen and the transcript come back.  The mic is remembered ON
// (kissVoiceEnabled=1), so the pipeline starts on load and startMic only
// grants the microphone; wake / speak drive one round
// (voice_page_helpers).
// ---------------------------------------------------------------------------

test('speech started in one tab is never submitted as another tab task', async () => {
  const page = makePage({voice: true});
  const {win, posted} = page;
  const {first, second} = twoTabs(win);

  clickTab(win, first);
  await startMic(page);
  // The user wakes the mic and speaks while tab `first` is on screen ...
  wake(page);
  speak(page);
  // ... then switches to `second` while the audio is being transcribed.
  clickTab(win, second);
  posted.length = 0;
  send(win, {type: 'voiceSpeech', text: 'leak7_speech_QK62', speaker: 1});

  const submits = posted.filter(m => m.type === 'submit');
  assert.strictEqual(
    submits.length,
    0,
    'a transcript captured while another tab was on screen must not be ' +
      'submitted as the newly visible tab task: ' +
      JSON.stringify(posted),
  );
  assert.ok(
    !taskInput(win).value.includes('leak7_speech_QK62'),
    'nor may the transcript be typed into the newly visible tab input',
  );

  win.close();
});

test('speech started and finished in the same tab still submits', async () => {
  const page = makePage({voice: true});
  const {win, posted} = page;
  twoTabs(win);

  await startMic(page);
  wake(page);
  speak(page);
  posted.length = 0;
  send(win, {type: 'voiceSpeech', text: 'run the tests QK63', speaker: 1});

  const submits = posted.filter(m => m.type === 'submit');
  assert.strictEqual(
    submits.length,
    1,
    'voice dictation must keep working within one tab: ' +
      JSON.stringify(posted),
  );
  assert.ok(submits[0].prompt.includes('run the tests QK63'));

  win.close();
});

test('voice dictation while a content tab is in front submits under the owning chat', async () => {
  const page = makePage({voice: true});
  const {win, posted} = page;
  const api = win._testApi;
  const chatId = api.getActiveTabId();

  // A file view in front: voice still belongs to the conversation that
  // produced it, so the submit must land on the OWNING chat tab — and
  // the dictated text must survive the switch back to it (the switch
  // restores the chat's own saved composer text).
  send(win, {
    type: 'fileContent',
    tabId: chatId,
    name: 'leak_voice_ct_QK80.txt',
    path: '/tmp/leak_voice_ct_QK80.txt',
    content: 'content tab body QK80',
  });
  assert.notStrictEqual(
    api.getActiveTabId(),
    chatId,
    'the file tab is in front',
  );
  await startMic(page);
  wake(page);
  speak(page);
  posted.length = 0;
  send(win, {
    type: 'voiceSpeech',
    text: 'voice on file tab QK80',
    speaker: 1,
  });

  const submits = posted.filter(m => m.type === 'submit');
  assert.strictEqual(
    submits.length,
    1,
    'dictation on a file tab must still submit: ' + JSON.stringify(posted),
  );
  assert.strictEqual(
    submits[0].tabId,
    chatId,
    'the submit lands on the owning chat, never on the file view id',
  );
  assert.ok(submits[0].prompt.includes('voice on file tab QK80'));
  assert.strictEqual(
    api.getActiveTabId(),
    chatId,
    'the conversation is back on screen',
  );

  win.close();
});

test('a spoken answer started in one tab never answers another tab question', async () => {
  const page = makePage({voice: true});
  const {win, posted} = page;
  const {first, second} = twoTabs(win);

  clickTab(win, first);
  send(win, {type: 'askUser', question: 'Which color?', tabId: first});
  await startMic(page);
  wake(page);
  speak(page);
  clickTab(win, second);
  send(win, {type: 'askUser', question: 'Which port?', tabId: second});
  posted.length = 0;
  send(win, {type: 'voiceSpeech', text: 'leak7_answer_QK64'});

  const answers = posted.filter(m => m.type === 'userAnswer');
  assert.ok(
    answers.every(m => m.tabId !== second),
    'an answer spoken while another tab was on screen must never be sent ' +
      'as the newly visible tab answer: ' +
      JSON.stringify(posted),
  );

  win.close();
});

test('an answer spoken without leaving the tab still answers its question', async () => {
  const page = makePage({voice: true});
  const {win, posted} = page;
  const {first} = twoTabs(win);

  clickTab(win, first);
  send(win, {type: 'askUser', question: 'Which color?', tabId: first});
  await startMic(page);
  wake(page);
  speak(page);
  posted.length = 0;
  send(win, {type: 'voiceSpeech', text: 'blue QK64b'});

  const answers = posted.filter(m => m.type === 'userAnswer');
  assert.strictEqual(
    answers.length,
    1,
    'voice must still answer the question of the tab it was spoken in: ' +
      JSON.stringify(posted),
  );
  assert.ok(answers[0].answer.includes('blue QK64b'));

  win.close();
});

test('the second of two overlapping utterances never leaks either', async () => {
  const page = makePage({voice: true});
  const {win, posted} = page;
  const {first, second} = twoTabs(win);

  clickTab(win, first);
  await startMic(page);
  // Two rounds before either transcript comes back: the rounds overlap,
  // so a single owner slot would forget the second utterance's tab
  // entirely.
  wake(page);
  speak(page);
  wake(page);
  speak(page);
  clickTab(win, second);
  posted.length = 0;
  send(win, {type: 'voiceSpeech', text: 'FIRST_QK69'});
  send(win, {type: 'voiceSpeech', text: 'SECOND_LEAK_QK70'});

  assert.strictEqual(
    posted.filter(m => m.type === 'submit').length,
    0,
    'neither overlapping transcript may be submitted as the visible tab: ' +
      JSON.stringify(posted),
  );
  assert.ok(
    !taskInput(win).value.includes('SECOND_LEAK_QK70'),
    'the second transcript must not be typed into a stranger conversation',
  );

  win.close();
});

test('a transcript arriving after the mic was turned off is never attributed', async () => {
  const page = makePage({voice: true});
  const {win, posted} = page;
  const {first, second} = twoTabs(win);

  clickTab(win, first);
  await startMic(page);
  wake(page);
  speak(page);
  // The user turns the mic off while the audio is still being
  // transcribed: the round is cancelled and its owner with it.
  micBtn(win).click();
  assert.ok(micBtn(win).classList.contains('voice-off'));
  clickTab(win, second);
  posted.length = 0;
  send(win, {type: 'voiceSpeech', text: 'LATE_QK71'});

  assert.strictEqual(
    posted.filter(m => m.type === 'submit').length,
    0,
    'a cancelled round must not submit a late transcript: ' +
      JSON.stringify(posted),
  );
  assert.ok(
    !taskInput(win).value.includes('LATE_QK71'),
    'nor may it be typed into the visible tab input',
  );

  win.close();
});

test('a reply lost in the transcribe window never eats the next transcript', async () => {
  const page = makePage({voice: true});
  const {win, posted} = page;
  const {first} = twoTabs(win);

  clickTab(win, first);
  await startMic(page);
  // The daemon's reply to round A never comes (the socket dropped
  // mid-transcription), and 60 s later the transcribe-flash safety
  // timer cancels the round. The credit it leaves must not be charged
  // to the next words, whose reply names its round by the echoed tag.
  wake(page);
  speak(page); // round A: audio posted, the reply is lost
  fireTimer(page, 60000);
  assert.ok(!micBtn(win).classList.contains('voice-transcribing'));

  wake(page);
  speak(page); // round B, same tab
  const roundB = posted.filter(m => m.type === 'voiceTranscribe').pop().round;
  posted.length = 0;
  send(win, {type: 'voiceSpeech', text: 'after lost reply QK93', round: roundB});

  const submits = posted.filter(m => m.type === 'submit');
  assert.strictEqual(
    submits.length,
    1,
    "round B's words must reach the tab that spoke them: " +
      JSON.stringify(posted),
  );
  assert.ok(submits[0].prompt.includes('after lost reply QK93'));
  assert.strictEqual(submits[0].tabId, first);

  win.close();
});

test('speech carries across tabs that are showing the same task', async () => {
  const page = makePage({voice: true});
  const {win, posted} = page;
  const {first, second} = twoTabs(win);
  send(win, {type: 'task_events', tabId: first, task_id: 'task-QK72'});
  send(win, {type: 'task_events', tabId: second, task_id: 'task-QK72'});

  clickTab(win, first);
  await startMic(page);
  wake(page);
  speak(page);
  clickTab(win, second);
  posted.length = 0;
  send(win, {type: 'voiceSpeech', text: 'same task QK72'});

  const submits = posted.filter(m => m.type === 'submit');
  assert.strictEqual(
    submits.length,
    1,
    'two views of one task are one conversation, so speech must carry: ' +
      JSON.stringify(posted),
  );
  assert.ok(submits[0].prompt.includes('same task QK72'));

  win.close();
});

test('a live round still submits after an earlier round was cancelled', async () => {
  const page = makePage({voice: true});
  const {win, posted} = page;
  const {first, second} = twoTabs(win);

  clickTab(win, first);
  await startMic(page);
  wake(page);
  speak(page);
  // Round A is cancelled (mic off) while its audio is being transcribed;
  // the user moves to tab `second` and turns the mic back on. Round A's
  // late transcript lands before the next wake, while tab `second` is
  // on screen: it belongs to the cancelled round and must neither
  // submit nor be typed anywhere.
  micBtn(win).click();
  clickTab(win, second);
  micBtn(win).click();
  await startMic(page);
  posted.length = 0;
  send(win, {type: 'voiceSpeech', text: 'cancelled words QK91'});
  assert.strictEqual(
    posted.filter(m => m.type === 'submit').length,
    0,
    'a cancelled round must never submit into the tab now on screen: ' +
      JSON.stringify(posted),
  );
  assert.ok(!taskInput(win).value.includes('cancelled words QK91'));
  // Round B, spoken on tab `second`. A cancelled predecessor must not
  // poison it.
  wake(page);
  speak(page);
  send(win, {type: 'voiceSpeech', text: 'live round QK92'});

  const submits = posted.filter(m => m.type === 'submit');
  assert.strictEqual(
    submits.length,
    1,
    'the live round must still dictate into the tab it was started on: ' +
      JSON.stringify(posted),
  );
  assert.ok(submits[0].prompt.includes('live round QK92'));
  assert.ok(!submits[0].prompt.includes('QK91'));

  win.close();
});

test('an empty transcript closes its own round and nothing else', async () => {
  const page = makePage({voice: true});
  const {win, posted} = page;
  const {first, second} = twoTabs(win);

  clickTab(win, first);
  await startMic(page);
  wake(page);
  speak(page);
  clickTab(win, second);
  wake(page);
  speak(page);

  // Round A's audio was noise: the daemon reports an empty transcript.
  posted.length = 0;
  send(win, {type: 'voiceSpeech', text: ''});
  assert.strictEqual(
    posted.filter(m => m.type === 'submit').length,
    0,
    'an empty transcript submits nothing: ' + JSON.stringify(posted),
  );

  // Round B must still be intact and still owned by the visible tab.
  posted.length = 0;
  send(win, {type: 'voiceSpeech', text: 'after silence QK94'});
  const submits = posted.filter(m => m.type === 'submit');
  assert.strictEqual(
    submits.length,
    1,
    'an empty predecessor must not consume the next round owner: ' +
      JSON.stringify(posted),
  );
  assert.ok(submits[0].prompt.includes('after silence QK94'));
  assert.strictEqual(submits[0].tabId, second);

  win.close();
});

test('a transcript for a round this page never started still arrives', async () => {
  const page = makePage({voice: true});
  const {win, posted} = page;
  // A transcript can turn up without this page ever having begun a
  // round for it. It was never tied to a moment in time, so there is no
  // tab switch to detect and no reason to drop it.
  posted.length = 0;
  send(win, {type: 'voiceSpeech', text: 'HOST_SIDE_QK74'});

  const submits = posted.filter(m => m.type === 'submit');
  assert.strictEqual(
    submits.length,
    1,
    'an unsolicited transcript must still be delivered: ' +
      JSON.stringify(posted),
  );
  assert.ok(submits[0].prompt.includes('HOST_SIDE_QK74'));

  win.close();
});

test('voice fails closed when the page publishes no owner', async () => {
  const page = makePage({voice: true});
  const {win, posted} = page;
  twoTabs(win);

  await startMic(page);
  wake(page);
  speak(page);
  // No accessor at all any more: ownership cannot be proved, so the
  // transcript must be refused rather than typed somewhere arbitrary.
  delete win.kissVoiceOwner;
  posted.length = 0;
  send(win, {type: 'voiceSpeech', text: 'NO_OWNER_QK73'});

  assert.strictEqual(
    posted.filter(m => m.type === 'submit').length,
    0,
    'an unprovable transcript must not be submitted: ' + JSON.stringify(posted),
  );
  assert.ok(
    !taskInput(win).value.includes('NO_OWNER_QK73'),
    'nor may it be typed into whatever input happens to be mounted',
  );

  win.close();
});

// ---------------------------------------------------------------------------
// Exemption -- two tabs showing the SAME TASK legitimately share. A shared
// backend chat is NOT enough: one chat can host several tasks.
// ---------------------------------------------------------------------------

// Give both tabs the same backend chat and, optionally, a task each.
// Returns the two tab ids with the first tab left on screen.
function twoChatSiblings(win, chatId, firstTask, secondTask) {
  const api = win._testApi;
  const first = api.getActiveTabId();
  send(win, {type: 'clear', chat_id: chatId, tabId: first});
  api.createNewTab();
  const second = api.getActiveTabId();
  send(win, {type: 'clear', chat_id: chatId, tabId: second});
  if (firstTask) {
    send(win, {type: 'task_events', tabId: first, task_id: firstTask});
  }
  if (secondTask) {
    send(win, {type: 'task_events', tabId: second, task_id: secondTask});
  }
  return {first, second};
}

test('two tabs showing the same task still share their text', () => {
  const {win} = makePage();
  const {first} = twoChatSiblings(
    win,
    'shared-chat-QK65',
    'task-QK65',
    'task-QK65',
  );

  // A followup suggestion for the sibling tab is the SAME task, so it may
  // render on screen.
  send(win, {
    type: 'followup_suggestion',
    tabId: first,
    text: 'shared_followup_QK66',
  });
  assert.ok(
    visibleText(win).includes('shared_followup_QK66'),
    'tabs showing the same task must keep sharing text',
  );

  win.close();
});

test('two tabs on one chat running different tasks never share text', () => {
  const {win} = makePage();
  const {first} = twoChatSiblings(
    win,
    'shared-chat-QK65b',
    'task-A-QK65b',
    'task-B-QK65b',
  );

  send(win, {
    type: 'text_delta',
    tabId: first,
    taskId: 'task-A-QK65b',
    text: 'FOREIGN_TASK_QK66b',
  });
  assert.ok(
    !visibleText(win).includes('FOREIGN_TASK_QK66b'),
    'a shared backend chat must not let a different task leak on screen',
  );

  win.close();
});

test('a notification for a same-task sibling tab still shows', () => {
  const {win} = makePage();
  const {first} = twoChatSiblings(
    win,
    'shared-chat-QK67',
    'task-QK67',
    'task-QK67',
  );

  send(win, {
    type: 'notification',
    tabId: first,
    id: 'sibling-toast',
    message: 'sibling_notify_QK68',
    severity: 'info',
  });
  assert.ok(
    toastText(win).includes('sibling_notify_QK68'),
    'a toast from the same task must still be shown',
  );

  win.close();
});

test('a notification for a same-chat different-task tab never shows', () => {
  const {win} = makePage();
  const {first} = twoChatSiblings(
    win,
    'shared-chat-QK67b',
    'task-A-QK67b',
    'task-B-QK67b',
  );

  send(win, {
    type: 'notification',
    tabId: first,
    id: 'foreign-toast',
    message: 'foreign_notify_QK68b',
    severity: 'info',
  });
  assert.ok(
    !toastText(win).includes('foreign_notify_QK68b'),
    'a toast from another task must not pop over this conversation',
  );

  win.close();
});

// ---------------------------------------------------------------------------
// Genuinely global messages must keep working.
// ---------------------------------------------------------------------------

// One page for all of them: each has its own suite (the loading
// overlay, #remote-url, #model-name, the history sidebar, #new-chat-btn,
// focusInput, #stop-btn); this only checks the ownership guards left
// the app-global paths alone with two tabs open.
test('app-global messages and buttons still work with two tabs open', () => {
  const {win, posted} = makePage();
  const {first, second} = twoTabs(win);

  send(win, {type: 'daemonStatus', connected: false});
  const overlay = win.document.getElementById('kiss-server-loading');
  assert.strictEqual(
    overlay.style.display,
    '',
    'losing the daemon must still raise the loading overlay',
  );
  send(win, {type: 'daemonStatus', connected: true});
  assert.strictEqual(
    overlay.style.display,
    'none',
    'reconnecting must still drop the loading overlay',
  );

  send(win, {type: 'remote_url', url: 'https://global-url-QK69.example'});
  assert.ok(
    win.document
      .getElementById('remote-url')
      .textContent.includes('global-url-QK69'),
    'the remote URL is app-global and must keep rendering',
  );

  send(win, {
    type: 'models',
    models: [
      {name: 'model-QK70', inp: 15, out: 75, uses: 0, vendor: 'Anthropic'},
      {name: 'other-model', inp: 5, out: 25, uses: 0, vendor: 'OpenAI'},
    ],
    selected: 'model-QK70',
  });
  // The pill wraps its label in U+200E marks (leading-truncation CSS
  // keeps the line RTL; the marks pin the character order).
  assert.strictEqual(
    win.document
      .getElementById('model-name')
      .textContent.replace(/\u200e/g, ''),
    'model-QK70',
    'the model list is app-global and must keep rendering',
  );

  send(win, {
    type: 'history',
    offset: 0,
    generation: 0,
    sessions: [
      {
        id: 'c-QK72',
        task_id: 't-QK72',
        title: 'history-entry-QK72',
        preview: 'history-entry-QK72',
        ts: 1000,
      },
    ],
  });
  assert.ok(
    win.document
      .getElementById('history-list')
      .textContent.includes('history-entry-QK72'),
    'the history sidebar is shared by design and must keep rendering',
  );

  taskInput(win).blur();
  send(win, {type: 'focusInput'});
  assert.strictEqual(
    win.document.activeElement,
    taskInput(win),
    'the Focus Chat command must still focus the input',
  );

  posted.length = 0;
  win.document.getElementById('stop-btn').click();
  const stops = posted.filter(m => m.type === 'stop');
  assert.strictEqual(stops.length, 1, JSON.stringify(posted));
  assert.strictEqual(
    stops[0].tabId,
    second,
    'the Stop button must still target the tab the user is looking at',
  );

  // The New chat button reuses a chat that is still new, so the
  // daemon's task-text echo makes both chats "used" first.
  send(win, {type: 'setTaskText', text: 'first task', tabId: first});
  send(win, {type: 'setTaskText', text: 'second task', tabId: second});
  send(win, {type: 'clear', chat_id: 'chat-QK75', tabId: second});
  win.document.getElementById('new-chat-btn').click();
  const now = win._testApi.getActiveTabId();
  assert.ok(
    now !== first && now !== second,
    'the New chat button must still create a tab',
  );
  win.close();
});

// ---------------------------------------------------------------------------
// Leak 8 -- diagnostics, adjacent transcripts and the task header wrote the
// visible surfaces without proving ownership.
// ---------------------------------------------------------------------------

test('diagnostics for a background tab reach that tab, not the screen', () => {
  // Routed to the background tab's parked transcript, not dropped and
  // not printed into the visible conversation: a notice, a warning and
  // an error all show once the tab comes on screen.
  const {win} = makePage();
  const {first} = twoTabs(win);

  send(win, {type: 'notice', tabId: first, text: 'bg_notice_QK79'});
  send(win, {type: 'warning', tabId: first, message: 'bg_warning_QK79'});
  send(win, {type: 'error', tabId: first, text: 'bg_error_QK79'});
  const hidden = visibleText(win);
  assert.ok(!hidden.includes('bg_notice_QK79'), "another task's notice");
  assert.ok(!hidden.includes('bg_warning_QK79'), "another task's warning");
  assert.ok(!hidden.includes('bg_error_QK79'), "another task's failure");
  win._testApi.switchToTab(first);

  const shown = visibleText(win);
  assert.ok(shown.includes('Note: bg_notice_QK79'), 'the notice reached it');
  assert.ok(shown.includes('Warning: bg_warning_QK79'), 'the warning did');
  assert.ok(shown.includes('Error: bg_error_QK79'), 'the error did');

  win.close();
});

test('diagnostics for the visible tab still print', () => {
  const {win} = makePage();
  const {second} = twoTabs(win);

  send(win, {type: 'error', tabId: second, text: 'own_error_QK78'});
  send(win, {type: 'notice', tabId: second, text: 'own_notice_QK78'});
  send(win, {type: 'warning', tabId: second, message: 'own_warning_QK78'});

  const shown = visibleText(win);
  assert.ok(shown.includes('own_error_QK78'), 'own error must print');
  assert.ok(shown.includes('own_notice_QK78'), 'own notice must print');
  assert.ok(shown.includes('own_warning_QK78'), 'own warning must print');

  win.close();
});

test('an unaddressed diagnostic still reaches the only conversation', () => {
  const {win} = makePage();

  // One chat tab, nothing to confuse it with: a daemon-level diagnostic
  // carrying no ids must still be seen rather than silently swallowed.
  send(win, {type: 'error', text: 'global_error_QK79'});

  assert.ok(
    visibleText(win).includes('global_error_QK79'),
    'a window-level diagnostic must still reach the user',
  );

  win.close();
});

test('a neighbouring transcript for a background tab never replays on screen', () => {
  const {win} = makePage();
  const {first} = twoTabs(win);

  send(win, {
    type: 'adjacent_task_events',
    direction: 'prev',
    tabId: first,
    task: 'bg neighbour QK80',
    task_id: 'task-bg-QK80',
    events: [{type: 'text_delta', text: 'ADJACENT_LEAK_QK80'}],
  });

  assert.ok(
    !visibleText(win).includes('ADJACENT_LEAK_QK80'),
    'a neighbouring task of ANOTHER tab must not splice itself into this ' +
      'conversation output',
  );

  win.close();
});

test('a neighbouring transcript for the visible tab still replays', () => {
  const {win} = makePage();
  const {second} = twoTabs(win);

  send(win, {
    type: 'adjacent_task_events',
    direction: 'prev',
    tabId: second,
    task: 'own neighbour QK81',
    task_id: 'task-own-QK81',
    events: [{type: 'text_delta', text: 'ADJACENT_OWN_QK81'}],
  });

  assert.ok(
    visibleText(win).includes('ADJACENT_OWN_QK81'),
    'scrolling into this tab own neighbouring task must still work',
  );

  win.close();
});

test('a task header for a background tab never retitles the visible one', () => {
  const {win} = makePage();
  const {first, second} = twoTabs(win);

  send(win, {type: 'setTaskText', tabId: first, text: 'bg header QK82'});

  assert.ok(
    !tabTitle(win, second).includes('bg header QK82'),
    "a background task's header must not rename the tab on screen",
  );
  assert.ok(
    tabTitle(win, first).includes('bg header QK82'),
    'it must rename its OWN tab instead',
  );

  win.close();
});

test('a task header for the visible tab still retitles it', () => {
  const {win} = makePage();
  const {second} = twoTabs(win);

  send(win, {type: 'setTaskText', tabId: second, text: 'own header QK83'});

  assert.ok(
    tabTitle(win, second).includes('own header QK83'),
    'the visible tab must still be named after the task it is running',
  );

  win.close();
});

test('a file-open failure for the visible tab still raises a toast', () => {
  const {win} = makePage();
  const {second} = twoTabs(win);

  send(win, {
    type: 'fileContent',
    tabId: second,
    path: '/tmp/own_err_QK84.txt',
    error: 'cannot open own_err_QK84',
  });

  assert.ok(
    toastText(win).includes('own_err_QK84'),
    'the user must still learn that the file THEY opened failed',
  );

  win.close();
});

test('the visible tab id is published for voice', () => {
  const {win} = makePage();
  const {second} = twoTabs(win);

  assert.strictEqual(
    win.kissActiveTabId(),
    second,
    'voice.js reads the conversation on screen through this accessor',
  );

  win.close();
});

// ---------------------------------------------------------------------------
// Leak 9 -- a delayed `/ask` answer names a task its tab has moved on from.
// ---------------------------------------------------------------------------

// The answering agent runs for minutes on its own thread.  Its reply is
// stamped with the asking tab and the task it answers; by the time it
// lands, the same tab may be running a NEWER task.  mayAdoptTaskId trusts
// any tabId-stamped event, so without a guard the stale reply would render
// into the new task and re-bind the tab to the finished one.

test('a stale ask answer never enters the task that replaced it', () => {
  const {win} = makePage();
  const api = win._testApi;
  const tab = api.getActiveTabId();
  send(win, {type: 'text_delta', text: 'old task', tabId: tab, taskId: '100'});
  send(win, {type: 'text_end', tabId: tab, taskId: '100'});
  send(win, {type: 'text_delta', text: 'new task', tabId: tab, taskId: '200'});
  send(win, {type: 'text_end', tabId: tab, taskId: '200'});

  send(win, {
    type: 'ask_answer',
    question: 'stale question',
    text: 'STALE_ANSWER_L9A',
    success: true,
    tabId: tab,
    taskId: '100',
  });
  assert.ok(
    !visibleText(win).includes('STALE_ANSWER_L9A'),
    'an answer for task 100 must not show inside task 200',
  );
  // Still bound to task 200: a task-only addressed event of it renders.
  send(win, {type: 'prompt', text: 'STILL_200_L9B', taskId: '200'});
  assert.ok(
    visibleText(win).includes('STILL_200_L9B'),
    'the stale answer must not re-bind the tab to task 100',
  );

  win.close();
});

test('an ask answer for the task on screen still renders', () => {
  const {win} = makePage();
  const api = win._testApi;
  const tab = api.getActiveTabId();
  send(win, {type: 'text_delta', text: 'task', tabId: tab, taskId: '300'});
  send(win, {type: 'text_end', tabId: tab, taskId: '300'});

  send(win, {
    type: 'ask_answer',
    question: 'q',
    text: 'CURRENT_ANSWER_L9C',
    success: true,
    tabId: tab,
    taskId: '300',
  });
  assert.ok(
    visibleText(win).includes('CURRENT_ANSWER_L9C'),
    'the answer to the current task must render',
  );
  // An answer whose owner task never got a row carries no taskId: it is
  // shown in the tab it was asked from, there is nothing to compare.
  send(win, {
    type: 'ask_answer',
    question: 'q2',
    text: 'UNSTAMPED_ANSWER_L9D',
    success: false,
    tabId: tab,
  });
  assert.ok(
    visibleText(win).includes('UNSTAMPED_ANSWER_L9D'),
    'an answer without a taskId is shown in the asking tab',
  );

  win.close();
});

test('a stale ask answer is not buffered into a background tab either', () => {
  const {win} = makePage();
  const {first, second} = twoTabs(win);
  // A background tab is bound to its task by the task_settings event the
  // daemon emits at task start (a hidden tab adopts that id; streamed
  // deltas alone do not re-bind it).
  send(win, {type: 'task_settings', tabId: first, settings: {task_id: '100'}});
  send(win, {
    type: 'text_delta',
    text: 'first old',
    tabId: first,
    taskId: '100',
  });
  send(win, {type: 'text_end', tabId: first, taskId: '100'});
  send(win, {type: 'task_settings', tabId: first, settings: {task_id: '200'}});
  send(win, {
    type: 'text_delta',
    text: 'first new',
    tabId: first,
    taskId: '200',
  });
  send(win, {type: 'text_end', tabId: first, taskId: '200'});
  assert.strictEqual(win._testApi.getActiveTabId(), second);

  send(win, {
    type: 'ask_answer',
    question: 'q',
    text: 'STALE_BG_ANSWER_L9E',
    success: true,
    tabId: first,
    taskId: '100',
  });
  clickTab(win, first);
  assert.ok(
    !visibleText(win).includes('STALE_BG_ANSWER_L9E'),
    'the restored tab must not show an answer for the task it left',
  );

  win.close();
});

test("a stale ask answer is dropped in the replacement run's pre-adoption window", () => {
  // The tab still carries task 100's id after the new run's setTaskText +
  // clear, until the run's first id-bearing event arrives.  An answer for
  // task 100 landing in that window must not enter the cleared transcript.
  const {win} = makePage();
  const api = win._testApi;
  const tab = api.getActiveTabId();
  send(win, {type: 'text_delta', text: 'old task', tabId: tab, taskId: '100'});
  send(win, {type: 'text_end', tabId: tab, taskId: '100'});
  send(win, {type: 'setTaskText', text: 'new task', tabId: tab});
  send(win, {type: 'clear', tabId: tab, chat_id: 'chat-1'});

  send(win, {
    type: 'ask_answer',
    question: 'old q',
    text: 'STALE_PRE_ADOPTION_L9F',
    success: true,
    tabId: tab,
    taskId: '100',
  });
  assert.ok(
    !visibleText(win).includes('STALE_PRE_ADOPTION_L9F'),
    'an answer for the replaced task must not enter the new run',
  );
  // The new run then binds the tab to task 200 and streams normally.
  send(win, {
    type: 'text_delta',
    text: 'NEW_RUN_L9G',
    tabId: tab,
    taskId: '200',
  });
  send(win, {type: 'text_end', tabId: tab, taskId: '200'});
  assert.ok(visibleText(win).includes('NEW_RUN_L9G'), 'the new run renders');
  send(win, {
    type: 'ask_answer',
    question: 'new q',
    text: 'NEW_ANSWER_L9H',
    success: true,
    tabId: tab,
    taskId: '200',
  });
  assert.ok(
    visibleText(win).includes('NEW_ANSWER_L9H'),
    'an answer for the new run still renders',
  );

  win.close();
});

test("a stale ask answer is dropped in a background tab's pre-adoption window", () => {
  const {win} = makePage();
  const {first, second} = twoTabs(win);
  send(win, {type: 'task_settings', tabId: first, settings: {task_id: '100'}});
  send(win, {
    type: 'text_delta',
    text: 'first old',
    tabId: first,
    taskId: '100',
  });
  send(win, {type: 'text_end', tabId: first, taskId: '100'});
  send(win, {type: 'clear', tabId: first, chat_id: 'chat-1'});
  assert.strictEqual(win._testApi.getActiveTabId(), second);

  send(win, {
    type: 'ask_answer',
    question: 'q',
    text: 'STALE_BG_PRE_ADOPTION_L9I',
    success: true,
    tabId: first,
    taskId: '100',
  });
  send(win, {
    type: 'text_delta',
    text: 'first new',
    tabId: first,
    taskId: '200',
  });
  send(win, {type: 'text_end', tabId: first, taskId: '200'});
  clickTab(win, first);
  assert.ok(
    !visibleText(win).includes('STALE_BG_PRE_ADOPTION_L9I'),
    'the restored tab must not show an answer for the task it left',
  );
  assert.ok(visibleText(win).includes('first new'), 'the new run is shown');

  win.close();
});

(async () => {
  const runner = makeRunner();
  for (const t of tests) await runner.test(t.name, t.fn);
  runner.report('crossTabTextLeak');
  // main.js starts timers in every page opened above; exit explicitly.
  process.exit(0);
})();
