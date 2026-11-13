// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// A question is only answerable while its task is alive. The server emits
// `askUserDone` only after an answer is accepted, so a task that ends while a
// question is outstanding (error, stop, interrupt, or a plain done) would
// otherwise leave the composer stuck in answer mode and a '?' badge that never clears.

'use strict';

const assert = require('assert');
const harness = require('./simplify2_harness');
const {tabEl} = require('./subagent_tab_helpers');

function makePage() {
  return harness.makePage({beforeScripts: harness.stubRaf}).win;
}

const {send, clickTab, answering} = harness;

function tabRecord(win, tabId) {
  return win._testApi.openTabs().find(t => t.id === tabId) || null;
}

// The '?' attention mark a chat shows while it waits for an answer
// off-screen (a tab on screen never shows it: its question is the composer).
// Read from the tab's record: the chats below are root tabs of their own
// groups, and the strip renders only the group on screen, so their mark is
// never in the DOM.  testSubagentStripBadgeClearedOnEachTerminalEvent reads
// the rendered `.chat-tab-attention` of a sub-agent that shares the strip.
function attentionGlyph(win, tabId) {
  const rec = tabRecord(win, tabId);
  assert.ok(rec, `tab ${tabId} must be open`);
  return rec.askPending && tabId !== win._testApi.getActiveTabId() ? '?' : '';
}

// The rendered attention mark of a tab on the strip (null without one).
function stripBadge(win, tabId) {
  const el = tabEl(win, tabId);
  assert.ok(el, `tab ${tabId} must be on the strip`);
  return el.querySelector('.chat-tab-attention');
}

// The Question panel in the transcript is the whole notice: neither a
// question nor its retirement raises a toast.
function assertNoToast(win, where) {
  assert.strictEqual(
    win.document.querySelectorAll('.kiss-notification').length,
    0,
    'no toast ' + where,
  );
}

const TERMINAL_EVENTS = [
  'task_done',
  'task_error',
  'task_interrupted',
  'task_stopped',
];

// A task that ends while its own tab is on screen, like an accepted
// answer (askUserDone), must take the composer out of answer mode.
function testActiveTabAskClearedOnEachTerminalEvent() {
  for (const type of [...TERMINAL_EVENTS, 'askUserDone']) {
    const win = makePage();
    const api = win._testApi;
    const tabId = api.getActiveTabId();

    send(win, {type: 'askUser', question: 'Which branch?', tabId: tabId});
    assert.ok(
      answering(win),
      `${type}: the question must be showing before the task ends`,
    );
    assertNoToast(win, `${type}: while the question is pending`);

    send(win, {type: type, tabId: tabId, success: true});

    assert.strictEqual(
      answering(win),
      false,
      `${type} must take the composer out of answer mode`,
    );
    assert.strictEqual(
      tabRecord(win, tabId).askPending,
      false,
      `${type} must clear the attention indicator`,
    );
    assertNoToast(win, `${type}: after the question is retired`);

    win.close();
  }
  console.log(
    '  ok - an ending task or an answer clears its visible question, no toast',
  );
}

// The badge on a background tab must clear too, and the user must not be
// left with a stale question when they later visit that tab.
function testBackgroundAskClearedOnEachTerminalEvent() {
  for (const type of TERMINAL_EVENTS) {
    const win = makePage();
    const api = win._testApi;
    const askTab = api.getActiveTabId();
    api.createNewTab();
    const userTab = api.getActiveTabId();

    send(win, {type: 'askUser', question: 'Which branch?', tabId: askTab});
    // The question switched the client to its tab; the user goes back
    // to their own, which flags the waiting tab.
    assert.strictEqual(api.getActiveTabId(), askTab);
    clickTab(win, userTab);
    assert.strictEqual(
      attentionGlyph(win, askTab),
      '?',
      `${type}: the waiting background tab must be flagged first`,
    );

    send(win, {type: type, tabId: askTab, success: true});

    assert.strictEqual(
      tabRecord(win, askTab).askPending,
      false,
      `${type} must clear the background tab's attention indicator`,
    );
    assert.strictEqual(
      answering(win),
      false,
      `${type} must not put the composer of the tab the user is on into answer mode`,
    );

    clickTab(win, askTab);
    assert.strictEqual(
      answering(win),
      false,
      `${type}: visiting the finished tab must not resurrect the question`,
    );

    win.close();
  }
  console.log('  ok - an ending task clears its background question badge');
}

// A sub-agent shares its parent's strip, so its badge is a real element
// there: the terminal event must take that element away, not only the
// record's flag.
function testSubagentStripBadgeClearedOnEachTerminalEvent() {
  for (const type of TERMINAL_EVENTS) {
    const win = makePage();
    const api = win._testApi;
    const parentTab = api.getActiveTabId();
    const subTab = 'sub-' + type;
    send(win, {
      type: 'openSubagentTab',
      tab_id: subTab,
      parent_tab_id: parentTab,
      description: 'child',
      task_id: 'task-' + subTab,
      taskIndex: 0,
      isSubagentTab: true,
    });

    send(win, {type: 'askUser', question: 'Continue?', tabId: subTab});
    assert.strictEqual(api.getActiveTabId(), subTab);
    clickTab(win, parentTab);
    const badge = stripBadge(win, subTab);
    assert.ok(badge, `${type}: the waiting sub-agent must be flagged first`);
    assert.strictEqual(badge.textContent, '?');

    send(win, {type: type, tabId: subTab, success: true});

    assert.strictEqual(
      stripBadge(win, subTab),
      null,
      `${type} must take the badge off the sub-agent's strip entry`,
    );
    assert.strictEqual(
      tabRecord(win, subTab).askPending,
      false,
      `${type} must clear the sub-agent's record too`,
    );
    assert.strictEqual(answering(win), false);

    win.close();
  }
  console.log("  ok - an ending task clears its sub-agent's strip badge");
}

// Ending one task must not silence a question that another chat is still
// waiting on.
function testUnrelatedTabKeepsItsQuestion() {
  const win = makePage();
  const api = win._testApi;
  const askTab = api.getActiveTabId();
  api.createNewTab();
  const otherTab = api.getActiveTabId();
  api.createNewTab();

  send(win, {type: 'askUser', question: 'Which branch?', tabId: askTab});
  send(win, {type: 'askUser', question: 'Which remote?', tabId: otherTab});

  send(win, {type: 'task_done', tabId: askTab, success: true});

  assert.strictEqual(
    tabRecord(win, askTab).askPending,
    false,
    "the finished task's badge must clear",
  );
  assert.strictEqual(
    attentionGlyph(win, otherTab),
    '?',
    'an unrelated waiting task must keep its badge',
  );
  // A task that has FINISHED is allowed to pull focus - that is the one
  // case the no-auto-switch rule deliberately keeps.
  assert.strictEqual(
    api.getActiveTabId(),
    askTab,
    'a finished task may still be focused',
  );

  clickTab(win, otherTab);
  assert.ok(answering(win), 'the unrelated question must still be answerable');

  win.close();
  console.log('  ok - ending one task leaves other questions alone');
}

// Two tabs can be bound to the same backend chat (history click, resumed
// session). Each still runs its own task, so the end of ONE task may only
// retire that task's question - the sibling's question is still live and the
// backend is still blocked on it.
function testSameChatIdSiblingKeepsItsQuestion() {
  const win = makePage();
  const api = win._testApi;
  const doneTab = api.getActiveTabId();
  send(win, {type: 'clear', chat_id: 'shared-chat-id', tabId: doneTab});
  api.createNewTab();
  const siblingTab = api.getActiveTabId();
  send(win, {type: 'clear', chat_id: 'shared-chat-id', tabId: siblingTab});
  // The user sits on a third, uninvolved tab so nothing here is on screen
  // (each question pulls them over; they go back to it).
  api.createNewTab();
  const thirdTab = api.getActiveTabId();

  send(win, {type: 'askUser', question: 'Which branch?', tabId: doneTab});
  send(win, {type: 'askUser', question: 'Which remote?', tabId: siblingTab});
  assert.strictEqual(api.getActiveTabId(), siblingTab);
  clickTab(win, thirdTab);
  assert.strictEqual(
    attentionGlyph(win, siblingTab),
    '?',
    'the sibling tab must be flagged before the other task ends',
  );

  send(win, {type: 'task_done', tabId: doneTab, success: true});

  assert.strictEqual(
    tabRecord(win, doneTab).askPending,
    false,
    "the finished task's own question must be retired",
  );
  assert.strictEqual(
    attentionGlyph(win, siblingTab),
    '?',
    'a sibling tab sharing the backend chat id must keep its own question',
  );

  clickTab(win, siblingTab);
  assert.ok(answering(win), "the sibling's question must still be answerable");

  clickTab(win, doneTab);
  assert.strictEqual(
    answering(win),
    false,
    'the finished tab must not resurrect its dead question',
  );

  win.close();
  console.log('  ok - ending one task keeps a same-chat sibling question');
}

// Submitting an answer retiring the prompt in every tab bound to the same
// backend chat is pinned by ask_user_same_chat_tabs_clear.test.js.

// A terminal event for a tab with no outstanding question changes nothing.
function testTerminalEventWithoutQuestionIsHarmless() {
  const win = makePage();
  const api = win._testApi;
  const askTab = api.getActiveTabId();
  api.createNewTab();
  const userTab = api.getActiveTabId();

  send(win, {type: 'askUser', question: 'Which branch?', tabId: askTab});
  send(win, {type: 'task_done', tabId: userTab, success: true});

  assert.strictEqual(
    attentionGlyph(win, askTab),
    '?',
    'a terminal event for another tab must not clear this badge',
  );
  assert.strictEqual(
    answering(win),
    false,
    'the composer stays a prompt box for a tab with no question',
  );

  send(win, {type: 'task_done', success: true});
  assert.strictEqual(
    attentionGlyph(win, askTab),
    '?',
    'a terminal event without a tabId must not clear another tab',
  );

  win.close();
  console.log('  ok - terminal events without a question are harmless');
}

const TESTS = [
  testActiveTabAskClearedOnEachTerminalEvent,
  testBackgroundAskClearedOnEachTerminalEvent,
  testSubagentStripBadgeClearedOnEachTerminalEvent,
  testUnrelatedTabKeepsItsQuestion,
  testSameChatIdSiblingKeepsItsQuestion,
  testTerminalEventWithoutQuestionIsHarmless,
];

try {
  for (const test of TESTS) test();
  console.log(`\n${TESTS.length} passed, 0 failed`);
  process.exit(0);
} catch (err) {
  console.error('FAIL:', err && err.message ? err.message : err);
  process.exit(1);
}
