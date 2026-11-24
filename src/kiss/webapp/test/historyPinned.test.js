// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end (jsdom) tests for pins in the Chats panel (media/main.js):
// the Pin / Unpin button in the "…" menu of a task row and of a chat
// panel's header shows what the daemon says (`is_pinned` and
// `chat_is_pinned` on the rows), a press flips it at once and sends
// `setPinned` / `setChatPinned` to the machine that holds the row, the
// flip is shielded from the refetches that predate the write as a star
// is, and what is pinned is listed itself in a "Pinned" section between
// "Running" and the days: a pinned chat's panel, and the row of a
// pinned task on its own (out of its chat's panel, unless that chat is
// pinned too), so that the menu of every item listed there reads
// "Unpin ...".  Unpinned, an item is back in its place.
// historyProjects.test.js covers the chats and tasks of a project.
//
// jsdom has no Popover API, so the menu cannot be opened here;
// tests/webapp/test_history_pinned.py pins through the open menu in a
// real browser, against a real daemon.

'use strict';

const assert = require('assert');
const {
  makePage,
  send,
  click,
  stubRaf,
  ofType,
  plain,
  captureTimers,
  fakeMatchMedia,
} = require('./simplify2_harness');
const {
  sendHistory,
  makeRow,
  openChatsPage,
  chatHeader,
  chatTabs,
  rowByTitle,
  listShape,
  pressFilterOption,
  freezeClock,
} = require('./history_helpers');

const CACHE_KEY = 'sorcar-history-page1';
const T0 = 1_700_000_000;
const DAY_MIN = 24 * 60;
const FROZEN_NOW_MS = 1_700_500_000_000;

// The page's clock stands still: a press's pin is shielded from stale
// refetches for 10 s of it, so a stalled run must not let the host's
// clock end the shield between a press and the reply that follows it
// (the bound scenario moves the clock itself).
function openPage() {
  return openChatsPage(win => {
    stubRaf(win);
    freezeClock(win, FROZEN_NOW_MS);
  });
}

/** A row of chat `chat` titled `title` that finished `age` minutes before T0; `extra` on top. */
function row(chat, title, age, extra) {
  const end = T0 - age * 60;
  return makeRow(
    Object.assign(
      {
        id: chat,
        task_id: title,
        title,
        timestamp: end,
        startTs: (end - 10) * 1000,
        endTs: end * 1000,
      },
      extra || {},
    ),
  );
}

/**
 * The plain chat B (b1) and the older chat A (a2, a1), pinned or not on
 * the daemon; `extra` goes on a2, to make a page that differs.
 */
function page(pinned, extra) {
  return [
    row('B', 'b1', 1),
    row('A', 'a2', 2, Object.assign({chat_is_pinned: pinned}, extra || {})),
    row('A', 'a1', 3, {chat_is_pinned: pinned}),
  ];
}

const PLAIN = ['sep:day', 'chat:B[b1]', 'chat:A[a2,a1]'];
const A_PINNED = ['sep:Pinned', 'chat:A[a2,a1]', 'sep:day', 'chat:B[b1]'];

/** The visible list (listShape), every day's separator as `sep:day`. */
function shape(win) {
  return listShape(win).map(entry =>
    /^sep:(?!Running$|Pinned$)/.test(entry) ? 'sep:day' : entry,
  );
}

/** The pin in the menu of chat `chat`'s header, or null. */
function chatPin(win, chat) {
  return chatHeader(win, chat).querySelector('.history-chat-pin');
}

/** The pin in the menu of the task row titled `title`. */
function taskPin(win, title) {
  return rowByTitle(win, title).querySelector('.sidebar-item-pin');
}

/** What each row named in `titles` says of its task's pin and of its chat's. */
function flags(win, titles) {
  return titles.map(title => {
    const data = rowByTitle(win, title).dataset;
    return data.pinned + '/' + data.chatPinned;
  });
}

/** The pin commands the page sent. */
function writes(posted) {
  return plain(
    posted.filter(
      m => m && (m.type === 'setPinned' || m.type === 'setChatPinned'),
    ),
  );
}

/** Answer a `tasks_updated` nudge's refetch with `sessions`. */
function refresh(win, posted, sessions) {
  send(win, {type: 'tasks_updated'});
  sendHistory(win, posted, sessions);
}

function testThePinsShowWhatTheDaemonSays() {
  const {win, posted} = openPage();
  // B's row comes from a daemon that does not send the fields yet; the
  // last row belongs to no chat.
  sendHistory(win, posted, [
    row('A', 'a2', 1, {chat_is_pinned: true, is_pinned: true}),
    row('A', 'a1', 2, {chat_is_pinned: true}),
    row('B', 'b1', 3),
    row('', 'loose', 4),
  ]);
  [
    [chatPin(win, 'A'), 'Unpin chat', true],
    [chatPin(win, 'B'), 'Pin chat', false],
    [taskPin(win, 'a2'), 'Unpin task', true],
    [taskPin(win, 'a1'), 'Pin task', false],
  ].forEach(([pin, name, set]) => {
    assert.strictEqual(pin.getAttribute('aria-label'), name);
    assert.strictEqual(pin.dataset.tooltip, name);
    assert.strictEqual(pin.getAttribute('aria-pressed'), String(set), name);
    assert.strictEqual(pin.classList.contains('pinned'), set, name);
    // The filled glyph while set, its outline otherwise.
    assert.ok(
      pin.innerHTML.includes(set ? 'fill="currentColor"' : 'fill="none"'),
      name,
    );
    // In the row's menu, right after the favourite star.
    assert.ok(
      pin.parentElement.classList.contains('sidebar-item-actions'),
      name,
    );
    assert.ok(
      pin.previousElementSibling.classList.contains('sidebar-item-favorite'),
      name,
    );
  });
  assert.strictEqual(chatPin(win, ''), null, 'a row of no chat has no chat to pin');
  assert.deepStrictEqual(flags(win, ['a2', 'a1', 'b1', 'loose']), [
    '1/1',
    '0/1',
    '0/0',
    '0/0',
  ]);
  win.close();
  console.log('  ok - the pins show what the daemon says');
}

function testPinnedItemsAreListedBetweenRunningAndTheDays() {
  const {win, posted} = openPage();
  sendHistory(win, posted, [
    row('U', 'u1', 4, {is_running: true, endTs: 0}),
    // A pinned chat is under Pinned, also while a task of it runs.
    row('R', 'r1', 5, {chat_is_pinned: true, is_running: true, endTs: 0}),
    row('P', 'p1', 9, {chat_is_pinned: true}),
    // Chat T is not pinned; one of its tasks is.
    row('T', 't2', 7),
    row('T', 't1', 8, {is_pinned: true}),
    // The one task of chat S is pinned: nothing is left under its day.
    row('S', 's1', 6, {is_pinned: true}),
    // Chat Q is pinned, and so is one of its tasks, which stays in it.
    row('Q', 'q2', 10, {chat_is_pinned: true}),
    row('Q', 'q1', 11, {chat_is_pinned: true, is_pinned: true}),
    row('N', 'n1', 1),
    row('O', 'o1', 3 * DAY_MIN),
  ]);
  // What is pinned is listed itself -- a chat's panel, a task's row --
  // the newest first, as everywhere.
  assert.deepStrictEqual(shape(win), [
    'sep:Running',
    'chat:U[u1]',
    'sep:Pinned',
    'chat:R[r1]',
    'row:s1',
    'row:t1',
    'chat:P[p1]',
    'chat:Q[q2,q1]',
    'sep:day',
    'chat:N[n1]',
    'chat:T[t2]',
    'sep:day',
    'chat:O[o1]',
  ]);
  // So the menu of every item listed there unpins it.
  assert.deepStrictEqual(
    [
      chatPin(win, 'R'),
      taskPin(win, 's1'),
      taskPin(win, 't1'),
      chatPin(win, 'P'),
      chatPin(win, 'Q'),
    ].map(pin => pin.getAttribute('aria-label')),
    ['Unpin chat', 'Unpin task', 'Unpin task', 'Unpin chat', 'Unpin chat'],
  );
  const sep = win.document.querySelector(
    '#history-list > .history-day-sep[data-day="pinned"]',
  );
  assert.strictEqual(sep.textContent, 'Pinned');
  assert.strictEqual(sep.getAttribute('role'), 'separator');
  // The flat list has no chat to unpin: the pinned tasks are listed.
  pressFilterOption(win, 'group', 'None');
  assert.deepStrictEqual(shape(win), [
    'sep:Running',
    'row:u1',
    'row:r1',
    'sep:Pinned',
    'row:s1',
    'row:t1',
    'row:q1',
    'sep:day',
    'row:n1',
    'row:t2',
    'row:p1',
    'row:q2',
    'sep:day',
    'row:o1',
  ]);
  // Nothing pinned: no section.
  sendHistory(win, posted, [row('N', 'n1', 1), row('O', 'o1', 3 * DAY_MIN)]);
  assert.deepStrictEqual(shape(win), ['sep:day', 'row:n1', 'sep:day', 'row:o1']);
  win.close();
  console.log('  ok - pinned items are listed between Running and the days');
}

function testAPressPinsTheChatAtOnce() {
  const {win, posted} = openPage();
  sendHistory(win, posted, page(false));
  assert.deepStrictEqual(shape(win), PLAIN);
  const header = chatHeader(win, 'A');
  const tabsBefore = chatTabs(win).length;
  click(win, chatPin(win, 'A'));
  assert.deepStrictEqual(writes(posted), [
    {type: 'setChatPinned', chatId: 'A', isPinned: true},
  ]);
  assert.deepStrictEqual(shape(win), A_PINNED, 'the chat moved at once');
  assert.strictEqual(chatPin(win, 'A').getAttribute('aria-label'), 'Unpin chat');
  assert.strictEqual(chatPin(win, 'A').getAttribute('aria-pressed'), 'true');
  assert.deepStrictEqual(
    flags(win, ['a2', 'a1', 'b1']),
    ['0/1', '0/1', '0/0'],
    "the chat's rows, and only they, are a pinned chat's",
  );
  assert.strictEqual(chatTabs(win).length, tabsBefore, 'the pin opens no chat');
  assert.strictEqual(ofType(posted, 'resumeSession').length, 0);

  // A refetch from BEFORE the write landed: the daemon still says no.
  refresh(win, posted, page(false));
  assert.strictEqual(chatHeader(win, 'A'), header, 'the same page kept the header');
  assert.deepStrictEqual(shape(win), A_PINNED, 'the optimistic pin is still shown');
  // Another such refetch, whose changed rows rebuild the list.
  refresh(win, posted, page(false, {steps: 3}));
  assert.notStrictEqual(chatHeader(win, 'A'), header, 'the list was rebuilt');
  assert.deepStrictEqual(shape(win), A_PINNED);
  assert.strictEqual(chatPin(win, 'A').getAttribute('aria-pressed'), 'true');
  // The write landed: the daemon agrees.
  refresh(win, posted, page(true, {steps: 3}));
  assert.deepStrictEqual(shape(win), A_PINNED);
  // The daemon's word rules again at once: another surface unpinned
  // the chat, which goes back under its day.
  refresh(win, posted, page(false, {steps: 3}));
  assert.deepStrictEqual(shape(win), PLAIN);
  assert.strictEqual(chatPin(win, 'A').getAttribute('aria-pressed'), 'false');

  click(win, chatPin(win, 'A'));
  click(win, chatPin(win, 'A'));
  assert.deepStrictEqual(
    writes(posted).map(m => m.isPinned),
    [true, true, false],
    'a second press unpins',
  );
  assert.deepStrictEqual(shape(win), PLAIN, 'back under its day at once');
  assert.strictEqual(chatPin(win, 'A').getAttribute('aria-label'), 'Pin chat');
  win.close();
  console.log('  ok - a press pins the chat at once and tells the daemon');
}

function testAPressPinsATask() {
  const {win, posted} = openPage();
  sendHistory(win, posted, page(false));
  click(win, taskPin(win, 'a1'));
  assert.deepStrictEqual(writes(posted), [
    {type: 'setPinned', taskId: 'a1', isPinned: true},
  ]);
  assert.strictEqual(taskPin(win, 'a1').getAttribute('aria-label'), 'Unpin task');
  assert.deepStrictEqual(flags(win, ['a2', 'a1', 'b1']), ['0/0', '1/0', '0/0']);
  // The task is listed under Pinned on its own: its chat, which is
  // not pinned, keeps its other task under its day.
  const a1Pinned = ['sep:Pinned', 'row:a1', 'sep:day', 'chat:B[b1]', 'chat:A[a2]'];
  assert.deepStrictEqual(shape(win), a1Pinned);
  assert.strictEqual(chatPin(win, 'A').getAttribute('aria-pressed'), 'false');
  // A refetch from before the write, whose changed rows rebuild the
  // list, finds the pin shielded.
  refresh(win, posted, page(false, {steps: 3}));
  assert.deepStrictEqual(shape(win), a1Pinned);
  // Its chat pinned too: the chat is listed with every task of it,
  // and the task is on its own again once the chat's pin is lifted.
  click(win, chatPin(win, 'A'));
  assert.deepStrictEqual(shape(win), A_PINNED);
  assert.strictEqual(taskPin(win, 'a1').getAttribute('aria-label'), 'Unpin task');
  click(win, chatPin(win, 'A'));
  assert.deepStrictEqual(shape(win), a1Pinned);
  // The chat's other task pinned as well: nothing of the chat is left
  // under its day.  Unpinned, each is back in the chat's panel.
  click(win, taskPin(win, 'a2'));
  assert.deepStrictEqual(shape(win), [
    'sep:Pinned',
    'row:a2',
    'row:a1',
    'sep:day',
    'chat:B[b1]',
  ]);
  click(win, taskPin(win, 'a1'));
  click(win, taskPin(win, 'a2'));
  assert.deepStrictEqual(shape(win), PLAIN);
  click(win, taskPin(win, 'a1'));
  // The flat list has the pinned task alone under Pinned.
  pressFilterOption(win, 'group', 'None');
  const flat = ['sep:Pinned', 'row:a1', 'sep:day', 'row:b1', 'row:a2'];
  assert.deepStrictEqual(shape(win), flat);
  // The list was regrouped from its own rows, which is no word from
  // the daemon: a refetch from before the write still finds the pin
  // shielded.
  refresh(win, posted, page(false));
  assert.deepStrictEqual(shape(win), flat, 'the regrouped list keeps the pin');
  // There a press moves the row itself, either way.
  click(win, taskPin(win, 'b1'));
  click(win, taskPin(win, 'a1'));
  assert.deepStrictEqual(shape(win), [
    'sep:Pinned',
    'row:b1',
    'sep:day',
    'row:a2',
    'row:a1',
  ]);
  assert.deepStrictEqual(
    writes(posted).slice(-2),
    [
      {type: 'setPinned', taskId: 'b1', isPinned: true},
      {type: 'setPinned', taskId: 'a1', isPinned: false},
    ],
  );
  win.close();
  console.log('  ok - a press pins a task, in the grouped and in the flat list');
}

function testAPinTheDaemonNeverStoredGoesAtTheBound() {
  const {win, posted} = openPage();
  sendHistory(win, posted, page(false));
  const timers = captureTimers(win);
  click(win, chatPin(win, 'A'));
  // The write may fail and no refresh may ever arrive: the press itself
  // plans the refetch that brings the daemon's value back.
  const refetch = timers.filter(t => t.ms > 9500 && t.ms < 10500);
  assert.strictEqual(refetch.length, 1, 'the press planned the refetch');
  const realNow = win.Date.now;
  win.Date.now = () => realNow() + 10200;
  try {
    const asked = ofType(posted, 'getHistory').length;
    refetch[0].cb();
    assert.ok(
      ofType(posted, 'getHistory').length > asked,
      'the refetch went out with no page in between',
    );
    sendHistory(win, posted, page(false));
  } finally {
    win.Date.now = realNow;
  }
  assert.deepStrictEqual(shape(win), PLAIN, "past the bound the daemon's value rules");
  assert.strictEqual(chatPin(win, 'A').getAttribute('aria-pressed'), 'false');
  // So it does in a list regrouped from its own rows past the bound:
  // a pin set and lifted again leaves the task as the daemon has it,
  // unpinned, not with the pin that was lifted.
  click(win, taskPin(win, 'a1'));
  click(win, taskPin(win, 'a1'));
  win.Date.now = () => realNow() + 10200;
  try {
    pressFilterOption(win, 'group', 'None');
  } finally {
    win.Date.now = realNow;
  }
  assert.deepStrictEqual(shape(win), ['sep:day', 'row:b1', 'row:a2', 'row:a1']);
  assert.strictEqual(taskPin(win, 'a1').getAttribute('aria-pressed'), 'false');
  win.close();
  console.log('  ok - a pin the daemon never stored goes at the bound');
}

function testAPinIsWrittenOnTheMachineThatHoldsTheRow() {
  const {win, posted} = openPage();
  sendHistory(win, posted, [row('R', 'r1', 1, {machine: 'boxb'}), row('L', 'l1', 2)]);
  click(win, chatPin(win, 'R'));
  click(win, taskPin(win, 'r1'));
  click(win, taskPin(win, 'l1'));
  assert.deepStrictEqual(writes(posted), [
    {type: 'setChatPinned', chatId: 'R', isPinned: true, machine: 'boxb'},
    {type: 'setPinned', taskId: 'r1', isPinned: true, machine: 'boxb'},
    {type: 'setPinned', taskId: 'l1', isPinned: true},
  ]);
  win.close();
  console.log('  ok - a pin is written on the machine that holds the row');
}

function testFiltersApplyToPinnedItems() {
  const {win, posted} = openPage();
  sendHistory(win, posted, [
    row('F', 'f1', 1, {chat_is_pinned: true, failed: true}),
    row('N', 'n1', 2),
  ]);
  assert.deepStrictEqual(shape(win), [
    'sep:Pinned',
    'chat:F[f1]',
    'sep:day',
    'chat:N[n1]',
  ]);
  // A filter hides a pinned item as any other, and a section left
  // without an item shows no separator.
  pressFilterOption(win, 'status', 'Errored');
  assert.deepStrictEqual(shape(win), ['sep:day', 'chat:N[n1]']);
  pressFilterOption(win, 'status', 'Errored');
  pressFilterOption(win, 'status', 'Succeeded');
  assert.deepStrictEqual(shape(win), ['sep:Pinned', 'chat:F[f1]']);
  win.close();
  console.log('  ok - filters apply to pinned items');
}

function testTheKeyboardStaysOnThePin() {
  const {win, posted} = openPage();
  sendHistory(win, posted, page(false));
  // A press moves the chat: the keyboard stays on its pin.
  const pin = chatPin(win, 'A');
  pin.focus();
  click(win, pin);
  assert.deepStrictEqual(shape(win), A_PINNED);
  assert.strictEqual(win.document.activeElement, pin, 'the moved chat keeps the keyboard');
  // So it does on the pin of a task whose row leaves its chat's panel.
  // Back in the panel the row is out of sight while the panel is
  // folded: the chat's header takes the keyboard.  In an unfolded
  // panel the pin keeps it.
  const rowPin = taskPin(win, 'b1');
  const header = chatHeader(win, 'B');
  for (const back of [header, rowPin]) {
    for (const [listed, holder] of [['row:b1', rowPin], ['chat:B[b1]', back]]) {
      rowPin.focus();
      click(win, rowPin);
      assert.ok(shape(win).includes(listed), listed);
      assert.strictEqual(win.document.activeElement, holder, listed);
    }
    click(win, header.querySelector('.history-chat-collapse'));
  }
  // Changed data rebuilds the list: the pin of the same chat's fresh
  // header takes the focus over, so Space/Enter keeps doing what it did.
  pin.focus();
  refresh(win, posted, page(true, {steps: 2}));
  const rebuilt = chatPin(win, 'A');
  assert.notStrictEqual(rebuilt, pin, 'the list was rebuilt');
  assert.strictEqual(win.document.activeElement, rebuilt);
  win.close();
  console.log('  ok - the keyboard stays on the pin across a move and a rebuild');
}

function testTheCachedPageOneCarriesThePins() {
  // A desktop-width page opens its sidebar at boot and paints the page
  // one the previous visit left, its Pinned section included, before
  // the daemon answers.
  const {win, posted} = makePage({
    beforeScripts: w => {
      freezeClock(w, FROZEN_NOW_MS);
      fakeMatchMedia(w, true);
      w.localStorage.setItem(
        CACHE_KEY,
        JSON.stringify({sessions: page(true), machineLabels: {}}),
      );
    },
  });
  assert.deepStrictEqual(shape(win), A_PINNED);
  // The chat is unpinned, and a reply from before the write lands: the
  // page kept for the next load is the one on screen.
  sendHistory(win, posted, page(true));
  click(win, chatPin(win, 'A'));
  refresh(win, posted, page(true));
  assert.deepStrictEqual(shape(win), PLAIN);
  assert.deepStrictEqual(
    JSON.parse(win.localStorage.getItem(CACHE_KEY)).sessions.map(
      s => !!s.chat_is_pinned,
    ),
    [false, false, false],
  );
  win.close();
  console.log('  ok - the cached page one carries the pins');
}

function main() {
  testThePinsShowWhatTheDaemonSays();
  testPinnedItemsAreListedBetweenRunningAndTheDays();
  testAPressPinsTheChatAtOnce();
  testAPressPinsATask();
  testAPinTheDaemonNeverStoredGoesAtTheBound();
  testAPinIsWrittenOnTheMachineThatHoldsTheRow();
  testFiltersApplyToPinnedItems();
  testTheKeyboardStaysOnThePin();
  testTheCachedPageOneCarriesThePins();
  console.log('historyPinned.test.js: all tests passed');
}

main();
