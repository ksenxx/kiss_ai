// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end (jsdom) tests for a chat's favourite star in the Chats
// panel (media/main.js): the star on a chat panel's header shows what
// the daemon says of the chat (`chat_is_favorite` on its rows), a click
// flips it at once and sends `setChatFavorite` to the machine holding
// the chat, the flip is shielded from refetches that predate the write
// as a task's star is, and the Favorites chip keeps the tasks of a
// starred chat beside the starred tasks.

'use strict';

const assert = require('assert');
const {
  send,
  click,
  sleep,
  stubRaf,
  ofType,
  plain,
  byId,
  captureTimers,
} = require('./simplify2_harness');
const {
  sendHistory,
  makeRow,
  openChatsPage,
  chatHeader,
  chatTabs,
  rowByTitle,
  visibleTitles,
  pressFilterOption,
  freezeClock,
} = require('./history_helpers');

const FROZEN_NOW_MS = 1_700_500_000_000;

// The page's clock stands still: a click's star is shielded from stale
// refetches for 10 s of it, so a stalled run must not let the host's
// clock end the shield between a click and the reply that follows it
// (the bound scenarios move the clock themselves).
function openPage() {
  return openChatsPage(win => {
    stubRaf(win);
    freezeClock(win, FROZEN_NOW_MS);
  });
}

/** A finished row of chat `chat` titled `title`; `extra` on top. */
function row(chat, title, extra) {
  return makeRow(Object.assign({id: chat, task_id: title, title}, extra || {}));
}

/**
 * Chat A (tasks a2, a1) starred or not on the daemon, and the unstarred
 * chat B (b1); `extra` goes on a2, to make a page that differs.
 */
function page(starred, extra) {
  return [
    row('A', 'a2', Object.assign({chat_is_favorite: starred}, extra || {})),
    row('A', 'a1', {chat_is_favorite: starred}),
    row('B', 'b1', {chat_is_favorite: false}),
  ];
}

/** The star on chat `chat`'s header, or null. */
function star(win, chat) {
  return chatHeader(win, chat).querySelector('.history-chat-favorite');
}

/** Whether chat `chat`'s header star is lit. */
function lit(win, chat) {
  return star(win, chat).classList.contains('favorited');
}

/** What each row named in `titles` says of its chat's star. */
function chatFlags(win, titles) {
  return titles.map(title => rowByTitle(win, title).dataset.chatFavorite);
}

/** The `setChatFavorite` commands the page sent. */
function writes(posted) {
  return plain(ofType(posted, 'setChatFavorite'));
}

/** Answer a `tasks_updated` nudge's refetch with `sessions`. */
function refresh(win, posted, sessions) {
  send(win, {type: 'tasks_updated'});
  sendHistory(win, posted, sessions);
}

function testTheHeaderStarShowsTheChatsStar() {
  const {win, posted} = openPage();
  // B's row comes from a daemon that does not send the field yet; the
  // last row belongs to no chat.
  sendHistory(win, posted, [
    row('A', 'a2', {chat_is_favorite: true}),
    row('A', 'a1', {chat_is_favorite: true}),
    row('B', 'b1'),
    row('', 'loose'),
  ]);
  assert.ok(lit(win, 'A'), 'a starred chat shows a lit star');
  assert.strictEqual(star(win, 'A').getAttribute('aria-label'), 'Unfavourite chat');
  assert.strictEqual(star(win, 'A').dataset.tooltip, 'Unfavourite chat');
  assert.ok(!lit(win, 'B'), 'an unstarred chat shows an unlit star');
  assert.strictEqual(star(win, 'B').getAttribute('aria-label'), 'Favourite chat');
  assert.strictEqual(star(win, 'B').dataset.tooltip, 'Favourite chat');
  assert.notStrictEqual(
    star(win, 'A').innerHTML,
    star(win, 'B').innerHTML,
    'the lit star is the filled icon',
  );
  assert.strictEqual(star(win, ''), null, 'a row of no chat has no chat to star');
  assert.deepStrictEqual(chatFlags(win, ['a2', 'a1', 'b1', 'loose']), [
    '1',
    '1',
    '0',
    '0',
  ]);
  // Starring a chat marks its own rows only.
  click(win, star(win, 'B'));
  assert.deepStrictEqual(chatFlags(win, ['a2', 'a1', 'b1', 'loose']), [
    '1',
    '1',
    '1',
    '0',
  ]);
  win.close();
  console.log("  ok - the header star shows the chat's star");
}

function testAClickStarsTheChatAtOnce() {
  const {win, posted} = openPage();
  sendHistory(win, posted, page(false));
  const header = chatHeader(win, 'A');
  const tabsBefore = chatTabs(win).length;
  click(win, star(win, 'A'));
  assert.ok(lit(win, 'A'), 'the star flipped at once');
  assert.deepStrictEqual(writes(posted), [
    {type: 'setChatFavorite', chatId: 'A', isFavorite: true},
  ]);
  assert.deepStrictEqual(
    chatFlags(win, ['a2', 'a1', 'b1']),
    ['1', '1', '0'],
    "the chat's rows, and only they, are a starred chat's",
  );
  assert.strictEqual(chatTabs(win).length, tabsBefore, 'the star opens no chat');
  assert.strictEqual(ofType(posted, 'resumeSession').length, 0);

  // A refetch from BEFORE the write landed: the daemon still says no.
  refresh(win, posted, page(false));
  assert.strictEqual(chatHeader(win, 'A'), header, 'the same page kept the header');
  assert.ok(lit(win, 'A'), 'the optimistic star is still shown');
  // The write landed: the daemon agrees; still the same header.
  refresh(win, posted, page(true));
  assert.strictEqual(chatHeader(win, 'A'), header);
  assert.ok(lit(win, 'A'));
  // The daemon's word rules again at once: another surface took the
  // star off.
  refresh(win, posted, page(false));
  assert.ok(!lit(win, 'A'), 'the star another surface took off is gone');
  assert.deepStrictEqual(chatFlags(win, ['a2', 'a1']), ['0', '0']);

  click(win, star(win, 'A'));
  click(win, star(win, 'A'));
  assert.deepStrictEqual(
    writes(posted).map(m => m.isFavorite),
    [true, true, false],
    'a second click takes the star off',
  );
  assert.ok(!lit(win, 'A'));
  assert.strictEqual(star(win, 'A').getAttribute('aria-label'), 'Favourite chat');
  win.close();
  console.log('  ok - a click stars the chat at once and tells the daemon');
}

function testAStarTheDaemonNeverStoredGoesAtTheBound() {
  const {win, posted} = openPage();
  sendHistory(win, posted, page(false));
  const timers = captureTimers(win);
  click(win, star(win, 'A'));
  // The write may fail and no refresh may ever arrive: the click itself
  // plans the refetch that brings the daemon's value back.
  const refetch = timers.filter(t => t.ms > 9500 && t.ms < 10500);
  assert.strictEqual(refetch.length, 1, 'the click planned the refetch');
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
  assert.ok(!lit(win, 'A'), "past the bound the daemon's value rules");
  assert.deepStrictEqual(chatFlags(win, ['a2', 'a1']), ['0', '0']);
  win.close();
  console.log('  ok - a star the daemon never stored goes at the bound');
}

async function testAParkedPagePastTheBoundAgreesOnEveryRow() {
  const {win, posted} = openPage();
  sendHistory(win, posted, page(false));
  click(win, star(win, 'A'));
  // A page from before the write arrives while a row is pressed: it is
  // overlaid with the click's star and parked.
  const pressed = rowByTitle(win, 'b1');
  pressed.dispatchEvent(
    new win.MouseEvent('mousedown', {bubbles: true, cancelable: true}),
  );
  refresh(win, posted, page(false, {steps: 3}));
  assert.strictEqual(rowByTitle(win, 'b1'), pressed, 'the page is parked');
  // The press outlasts the overlay's bound: the page that lands shows
  // the daemon's value on EVERY row of the chat, not on the first only.
  const realNow = win.Date.now;
  win.Date.now = () => realNow() + 10200;
  try {
    pressed.dispatchEvent(
      new win.MouseEvent('mouseup', {bubbles: true, cancelable: true}),
    );
    await sleep(0);
  } finally {
    win.Date.now = realNow;
  }
  assert.notStrictEqual(rowByTitle(win, 'b1'), pressed, 'the page landed');
  assert.ok(!lit(win, 'A'));
  assert.deepStrictEqual(chatFlags(win, ['a2', 'a1']), ['0', '0']);
  win.close();
  console.log('  ok - a parked page past the bound agrees on every row');
}

function testAChatIsStarredOnTheMachineThatHoldsIt() {
  const {win, posted} = openPage();
  // R lives on boxb only; M has a task here and a newer one on boxb.
  const newer = {timestamp: 1_700_000_500, startTs: 1_700_000_500_000};
  const there = extra =>
    row('M', 'm-there', Object.assign({machine: 'boxb'}, newer, extra || {}));
  sendHistory(win, posted, [
    row('R', 'r1', {machine: 'boxb'}),
    there(),
    row('M', 'm-here'),
  ]);
  click(win, star(win, 'R'));
  click(win, star(win, 'M'));
  assert.deepStrictEqual(writes(posted), [
    {type: 'setChatFavorite', chatId: 'R', isFavorite: true, machine: 'boxb'},
    {type: 'setChatFavorite', chatId: 'M', isFavorite: true},
  ]);
  assert.ok(lit(win, 'R'));
  assert.ok(lit(win, 'M'), "the header shows this daemon's star of the chat");
  assert.deepStrictEqual(
    chatFlags(win, ['r1', 'm-here', 'm-there']),
    ['1', '1', '0'],
    "the copy on boxb keeps boxb's own star",
  );
  // boxb's copy of M is no word on this daemon's star: it neither
  // shows it nor, when it carries a star of its own, acknowledges it.
  refresh(win, posted, [there()]);
  assert.ok(!lit(win, 'M'), "boxb's copy shows boxb's star");
  refresh(win, posted, [there({chat_is_favorite: true})]);
  refresh(win, posted, [row('M', 'm-here')]);
  assert.ok(lit(win, 'M'), 'the star given here still awaits this daemon');
  win.close();
  console.log('  ok - a chat is starred on the machine that holds it');
}

function testTheFavoritesChipKeepsAStarredChatsTasks() {
  const {win, posted} = openPage();
  sendHistory(win, posted, [
    row('A', 'a2', {chat_is_favorite: true}),
    row('A', 'a1', {chat_is_favorite: true}),
    row('B', 'b2', {is_favorite: true}),
    row('B', 'b1'),
    row('C', 'c1'),
  ]);
  const chip = byId(win, 'hf-favorite');
  const setChip = on => {
    chip.checked = on;
    chip.dispatchEvent(new win.Event('change', {bubbles: true}));
  };
  setChip(true);
  assert.deepStrictEqual(
    visibleTitles(win),
    ['a1', 'a2', 'b2'],
    "a starred chat's tasks and the starred tasks",
  );
  assert.strictEqual(
    chatHeader(win, 'C').parentElement.style.display,
    'none',
    'a chat with neither goes',
  );
  // Taking the star off hides the chat's tasks at once.
  click(win, star(win, 'A'));
  assert.deepStrictEqual(visibleTitles(win), ['b2']);
  setChip(false);
  assert.deepStrictEqual(visibleTitles(win), ['a1', 'a2', 'b1', 'b2', 'c1']);
  click(win, star(win, 'C'));
  setChip(true);
  assert.deepStrictEqual(visibleTitles(win), ['b2', 'c1']);
  // The flat list keeps the same tasks.
  pressFilterOption(win, 'group', 'None');
  assert.strictEqual(win.document.querySelector('.history-chat-group'), null);
  assert.deepStrictEqual(visibleTitles(win), ['b2', 'c1']);
  win.close();
  console.log("  ok - the Favorites chip keeps a starred chat's tasks");
}

function testFocusStaysOnTheHeaderControlAcrossARebuild() {
  const {win, posted} = openPage();
  sendHistory(win, posted, page(false));
  ['history-chat-favorite', 'history-chat-collapse'].forEach((cls, i) => {
    const control = chatHeader(win, 'A').querySelector('.' + cls);
    control.focus();
    assert.strictEqual(win.document.activeElement, control);
    // Changed data rebuilds the list; the same control of the same
    // chat's fresh header takes the focus over, so Space/Enter keeps
    // doing what it did.
    refresh(win, posted, page(false, {steps: 2 + i}));
    const rebuilt = chatHeader(win, 'A').querySelector('.' + cls);
    assert.notStrictEqual(rebuilt, control, 'the list was rebuilt');
    assert.strictEqual(win.document.activeElement, rebuilt, cls);
  });
  win.close();
  console.log('  ok - focus stays on the header control across a rebuild');
}

async function main() {
  testTheHeaderStarShowsTheChatsStar();
  testAClickStarsTheChatAtOnce();
  testAStarTheDaemonNeverStoredGoesAtTheBound();
  await testAParkedPagePastTheBoundAgreesOnEveryRow();
  testAChatIsStarredOnTheMachineThatHoldsIt();
  testTheFavoritesChipKeepsAStarredChatsTasks();
  testFocusStaysOnTheHeaderControlAcrossARebuild();
  console.log('historyChatFavorite.test.js: all tests passed');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
