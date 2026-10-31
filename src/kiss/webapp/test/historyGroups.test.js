// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end (jsdom) tests for the remote webapp's chat-grouped task
// history (media/main.js): the history panel groups tasks by chat,
// newest chat (by its latest task) first, with "Today" / "Yesterday" /
// date separators, across pages and under the status filters; a press
// on a row opens its task, with the press guards and the favourite
// toggles of the rows.

'use strict';

const assert = require('assert');
const {
  makePage: makeChatPage,
  send,
  click,
  byId,
  sleep,
  stubRaf,
  fakeMatchMedia,
  ofType,
  stubClipboard,
} = require('./simplify2_harness');

const {test, report} = require('./ui_antipattern_harness').makeRunner();
const {all} = require('./ui_antipattern_harness');
const {
  todayNoonSec,
  groupedSession: session,
  sendHistoryPage: sendHistory,
  listShape,
} = require('./history_helpers');

function makePage() {
  return makeChatPage({
    bodyClass: 'remote-chat',
    beforeScripts: win => {
      stubRaf(win);
      fakeMatchMedia(win, true);
      stubClipboard(win);
      // Native dialogs are gone from main.js; any call is a regression.
      win.prompt = () => {
        throw new Error('window.prompt must not be called');
      };
      win.confirm = () => {
        throw new Error('window.confirm must not be called');
      };
    },
    extraScripts: [
      'marked.min.js',
      'contentContextMenu.js',
      'treeContextMenu.js',
    ],
  });
}

/**
 * Intercept the page's setTimeout calls whose delay lies in [min, max]
 * ms: they are recorded in `timers` (`{fn, ms, id}`) instead of being
 * scheduled, under fake ids (>= 1e9, truthy: main.js keeps the armed
 * timer's id and 0 would read as "no timer armed").  A clearTimeout of
 * a fake id lands in `cleared`.  Other delays run as usual.  `fire()`
 * runs every recorded timer not cleared, in order, and forgets them;
 * `restore()` puts the real functions back.
 */
function interceptTimers(win, min, max) {
  const origSetTimeout = win.setTimeout;
  const origClearTimeout = win.clearTimeout;
  const timers = [];
  const cleared = [];
  win.setTimeout = (fn, ms) => {
    if (ms >= min && ms <= max) {
      const id = 1e9 + timers.length + 1;
      timers.push({fn, ms, id});
      return id;
    }
    return origSetTimeout(fn, ms);
  };
  win.clearTimeout = id => {
    if (id >= 1e9) {
      cleared.push(id);
      return undefined;
    }
    return origClearTimeout(id);
  };
  return {
    timers,
    cleared,
    fire() {
      for (const t of timers.splice(0)) {
        if (!cleared.includes(t.id)) t.fn();
      }
    },
    restore() {
      win.setTimeout = origSetTimeout;
      win.clearTimeout = origClearTimeout;
    },
  };
}

async function main() {
  // ---- History grouping ----

  const DAY = 86400;
  const todayNoon = todayNoonSec();

  function groups(win) {
    return all(win, '#history-list > .history-chat-group');
  }

  /**
   * A history row click opens its task in a fresh chat tab and asks
   * the daemon to resume it there.  The chat that was on screen (an
   * idle new chat) is retired at the same time, so the tab count does
   * not grow: the new active tab and the resumeSession it posts are the
   * evidence that the click was honoured.
   */
  function assertRowOpenedTask(
    win,
    posted,
    activeBefore,
    resumesBefore,
    taskId,
  ) {
    const active = win._testApi.getActiveTabId();
    assert.notStrictEqual(
      active,
      activeBefore,
      'the click opens a new chat tab',
    );
    assert.ok(
      win._testApi.openTabs().some(t => t.id === active),
      'the new chat tab is open',
    );
    const resumes = ofType(posted, 'resumeSession');
    assert.strictEqual(
      resumes.length,
      resumesBefore + 1,
      'one resume is asked',
    );
    assert.strictEqual(resumes[resumes.length - 1].taskId, taskId);
    assert.strictEqual(resumes[resumes.length - 1].tabId, active);
  }

  await test('History: tasks grouped by chat, chats by latest task, day separators', async () => {
    const {win, posted} = makePage();
    const threeDaysAgo = todayNoon - 3 * DAY;
    const label = new Date(threeDaysAgo * 1000).toLocaleDateString(undefined, {
      weekday: 'short',
      month: 'short',
      day: 'numeric',
    });
    sendHistory(win, posted, 0, [
      session('A', 'a2', todayNoon),
      session('B', 'b2', todayNoon - 600),
      session('A', 'a1', todayNoon - 1200), // older task of A: joins A's block
      session('C', 'c1', todayNoon - DAY), // yesterday
      session('B', 'b1', todayNoon - DAY - 60),
      session('D', 'd1', threeDaysAgo),
      session('', 'e1', threeDaysAgo - 60), // no chat id: its own block
    ]);
    assert.deepStrictEqual(listShape(win), [
      'sep:Today',
      'chat:A[task a2,task a1]',
      'chat:B[task b2,task b1]',
      'sep:Yesterday',
      'chat:C[task c1]',
      'sep:' + label,
      'chat:D[task d1]',
      'chat:[task e1]',
    ]);
    assert.strictEqual(all(win, '#history-list .sidebar-item').length, 7);
    const sep = all(win, '.history-day-sep')[0];
    assert.strictEqual(sep.getAttribute('role'), 'separator');
    // Task panels carry NO per-chat colour any more.
    assert.strictEqual(
      groups(win)[0].style.getPropertyValue('--task-color'),
      '',
      'no per-chat colour on the block',
    );
    assert.strictEqual(
      all(win, '#history-list .sidebar-item')[0].style.getPropertyValue(
        '--task-color',
      ),
      '',
      'no per-chat colour on the rows',
    );
    // Every chat block is a collapsible panel: a header (the chat's
    // first task) above a body holding the task rows.
    const groupA = groups(win)[0];
    const headerA = groupA.querySelector(':scope > .history-chat-header');
    assert.ok(headerA, 'chat block has a clickable header');
    assert.strictEqual(
      headerA.querySelector('.history-chat-title').textContent,
      'task a1',
      "the header follows the chat's oldest loaded task when the daemon " +
        'sends no chat_first_task',
    );
    assert.ok(
      groupA.querySelector(':scope > .history-chat-body .sidebar-item'),
      'task rows live in the body container',
    );
    win.close();
  });

  await test('History: chat panels collapse by default, even while running, and remember the user toggle', async () => {
    const {win, posted} = makePage();
    const collapsed = g => g.classList.contains('collapsed');
    sendHistory(win, posted, 0, [
      Object.assign(session('A', 'a2', todayNoon), {
        chat_first_task: 'first task of A',
      }),
      Object.assign(session('B', 'b1', todayNoon - 600), {is_running: true}),
      Object.assign(session('A', 'a1', todayNoon - 1200), {
        chat_first_task: 'first task of A',
      }),
    ]);
    let [gB, gA] = groups(win); // The running chat precedes newer idle chats.
    // The daemon names the chat's FIRST task (which may be beyond the
    // loaded page); the header shows it, clamped by CSS to 3 lines.
    assert.strictEqual(
      gA.querySelector('.history-chat-title').textContent,
      'first task of A',
    );
    assert.ok(collapsed(gA), 'an idle chat starts collapsed');
    assert.ok(
      collapsed(gB),
      'a running task does not unfold its chat: only the user does',
    );
    assert.strictEqual(
      gB.dataset.hasRunning,
      '1',
      'the running chat is still marked for the Running section',
    );
    // The header is no toggle: the "Show details" chevron in its
    // action strip (as on a task panel) folds and unfolds the tasks.
    const chevronA = gA.querySelector(
      '.history-chat-header > .history-chat-actions > .history-chat-collapse',
    );
    assert.ok(chevronA, 'the chat panel carries a Show details chevron');
    assert.strictEqual(chevronA.getAttribute('aria-expanded'), 'false');
    assert.strictEqual(chevronA.dataset.tooltip, 'Show details');
    assert.strictEqual(
      gA.querySelector('.history-chat-header').getAttribute('aria-expanded'),
      null,
      'the header itself carries no expanded state',
    );
    // The user opens A: the choice survives a re-render.
    chevronA.click();
    assert.ok(!collapsed(gA), 'the chevron expands the panel');
    assert.strictEqual(chevronA.getAttribute('aria-expanded'), 'true');
    assert.strictEqual(chevronA.dataset.tooltip, 'Hide details');
    sendHistory(win, posted, 0, [
      Object.assign(session('A', 'a2', todayNoon), {
        chat_first_task: 'first task of A',
      }),
      session('B', 'b1', todayNoon - 600),
    ]);
    [gA, gB] = groups(win);
    assert.ok(!collapsed(gA), 'the explicit expand survives the rebuild');
    assert.ok(
      collapsed(gB),
      "B's task finished and the user never toggled it: collapsed again",
    );
    // Collapsing hides the body, not the block.
    gA.querySelector('.history-chat-collapse').click();
    assert.ok(collapsed(gA), 'the chevron collapses the panel again');
    assert.ok(
      gA.querySelector(':scope > .history-chat-body .sidebar-item'),
      'the rows stay in the DOM under the collapsed header',
    );
    win.close();
  });

  await test("History: the chat header shows the daemon's chat_summary, falling back to the first task until one exists", async () => {
    const {win, posted} = makePage();
    const title = g => g.querySelector('.history-chat-title').textContent;
    // A is summarised; B has only its first task; C has neither (an
    // older daemon) and follows its oldest loaded row.
    sendHistory(win, posted, 0, [
      Object.assign(session('A', 'a2', todayNoon), {
        chat_first_task: 'first task of A',
        chat_summary: 'Fix login and add tests',
      }),
      Object.assign(session('B', 'b1', todayNoon - 600), {
        chat_first_task: 'first task of B',
        chat_summary: '',
      }),
      session('C', 'c2', todayNoon - 1200),
      Object.assign(session('A', 'a1', todayNoon - 1800), {
        chat_first_task: 'first task of A',
        chat_summary: 'Fix login and add tests',
      }),
      session('C', 'c1', todayNoon - 2400),
    ]);
    let [gA, gB, gC] = groups(win);
    assert.strictEqual(title(gA), 'Fix login and add tests');
    assert.strictEqual(title(gB), 'first task of B');
    assert.strictEqual(title(gC), 'task c1');
    assert.strictEqual(
      gA.querySelector('.history-chat-header').getAttribute('aria-label'),
      'Fix login and add tests',
      "the summary is the header's accessible name (not the action strip)",
    );
    // B's task finishes and the daemon summarises the chat: the next
    // refresh renames the panel. A summary of only whitespace does not
    // count as one, so C keeps following its first task instead.
    sendHistory(win, posted, 0, [
      Object.assign(session('A', 'a2', todayNoon), {
        chat_first_task: 'first task of A',
        chat_summary: 'Fix login and add tests',
      }),
      Object.assign(session('B', 'b1', todayNoon - 600), {
        chat_first_task: 'first task of B',
        chat_summary: 'Refactor the billing module',
      }),
      Object.assign(session('C', 'c2', todayNoon - 1200), {
        chat_first_task: 'first task of C',
        chat_summary: '   \n',
      }),
    ]);
    [gA, gB, gC] = groups(win);
    assert.strictEqual(title(gA), 'Fix login and add tests');
    assert.strictEqual(title(gB), 'Refactor the billing module');
    assert.strictEqual(title(gC), 'first task of C');
    // Only the first line of a multi-line summary is shown, and a row
    // that arrives without the chat-level fields (mixed daemon
    // payloads) never overwrites a name the daemon already gave.  A
    // chat whose summary and first task are both blank is "Untitled".
    sendHistory(win, posted, 0, [
      Object.assign(session('A', 'a2', todayNoon), {
        chat_first_task: 'first task of A',
        chat_summary: '  Fix login\nand add tests',
      }),
      Object.assign(session('B', 'b1', todayNoon - 600), {
        chat_first_task: '\n\n   \n',
        chat_summary: '  \n',
      }),
      session('A', 'a1', todayNoon - 1800),
    ]);
    [gA, gB] = groups(win);
    assert.strictEqual(title(gA), 'Fix login');
    assert.strictEqual(title(gB), 'Untitled');
    // The header carries no tooltip (the full summary is not repeated
    // on hover), also after its Show details chevron repaints it.
    const header = gA.querySelector('.history-chat-header');
    const chevron = gA.querySelector('.history-chat-collapse');
    for (const state of ['true', 'false']) {
      chevron.click();
      assert.strictEqual(chevron.getAttribute('aria-expanded'), state);
      assert.ok(!header.hasAttribute('data-tooltip'), 'no data-tooltip');
      assert.ok(!header.hasAttribute('title'), 'no title attribute');
    }
    win.close();
  });

  await test('History: a search expands every chat — past saved collapses and through the identical-refresh fast path — and clearing it restores the defaults', async () => {
    const {win, posted} = makePage();
    const collapsed = g => g.classList.contains('collapsed');
    const searchBox = byId(win, 'history-search');
    const setSearch = value => {
      searchBox.value = value;
      searchBox.dispatchEvent(new win.Event('input', {bubbles: true}));
    };
    const page = () => [
      session('A', 'a1', todayNoon),
      session('B', 'b1', todayNoon - 600),
    ];
    sendHistory(win, posted, 0, page());
    let [gA, gB] = groups(win);
    // The user explicitly collapses A (expand + collapse stores the
    // choice), which must NOT hide A's matches inside a later search.
    gA.querySelector('.history-chat-collapse').click();
    gA.querySelector('.history-chat-collapse').click();
    assert.ok(collapsed(gA));
    setSearch('task');
    // The search's results equal the loaded page, so the reply lands
    // in the identical-refresh fast path: no rebuild, same nodes.
    sendHistory(win, posted, 0, page());
    [gA, gB] = groups(win);
    assert.ok(
      !collapsed(gA),
      'a search expands A even though the user collapsed it before',
    );
    assert.ok(!collapsed(gB), 'B expands for the search too');
    // A collapse DURING the search applies to the search view only.
    gA.querySelector('.history-chat-collapse').click();
    assert.ok(collapsed(gA), 'the panel can still be folded mid-search');
    // A changed-data rebuild while the SAME query stands (any task on
    // the daemon persisted a result) keeps the in-search fold.
    send(win, {type: 'tasks_updated'});
    sendHistory(win, posted, 0, [
      session('A', 'a2', todayNoon + 60),
      session('A', 'a1', todayNoon),
      session('B', 'b1', todayNoon - 600),
    ]);
    [gA, gB] = groups(win);
    assert.ok(
      collapsed(gA),
      'the in-search fold survives a changed-data rebuild',
    );
    assert.ok(!collapsed(gB), 'unfolded chats stay expanded for the search');
    setSearch('');
    sendHistory(win, posted, 0, page());
    [gA, gB] = groups(win);
    assert.ok(collapsed(gA), "clearing the search restores A's saved collapse");
    assert.ok(collapsed(gB), 'clearing the search restores the default');
    // A NEW search starts from expanded matches again: the fold made
    // inside the previous search does not carry over.
    setSearch('task');
    sendHistory(win, posted, 0, page());
    [gA, gB] = groups(win);
    assert.ok(!collapsed(gA), 'a fresh search drops the old in-search fold');
    assert.ok(!collapsed(gB));
    win.close();
  });

  await test('History: keyboard focus on a chat header survives a changed-data rebuild', async () => {
    const {win, posted} = makePage();
    sendHistory(win, posted, 0, [
      session('A', 'a1', todayNoon),
      session('B', 'b1', todayNoon - 600),
    ]);
    const header = groups(win)[0].querySelector('.history-chat-header');
    header.focus();
    assert.strictEqual(win.document.activeElement, header);
    // Changed data (a new task appears) wipes and rebuilds the list;
    // the fresh header of the SAME chat must take the focus over.
    send(win, {type: 'tasks_updated'});
    sendHistory(win, posted, 0, [
      session('A', 'a2', todayNoon + 60),
      session('A', 'a1', todayNoon),
      session('B', 'b1', todayNoon - 600),
    ]);
    const headerAfter = groups(win)[0].querySelector('.history-chat-header');
    assert.notStrictEqual(headerAfter, header, 'the list was rebuilt');
    assert.strictEqual(
      groups(win)[0].dataset.chatId,
      'A',
      'A still leads (its newest task is the youngest)',
    );
    assert.strictEqual(
      win.document.activeElement,
      headerAfter,
      'focus lands on the rebuilt header of the same chat',
    );
    win.close();
  });

  function firstRow(win) {
    return win.document.querySelector('#history-list .sidebar-item');
  }

  // The 300 ms touch release-grace timer, intercepted so a test fires
  // it by hand (the zero-delay apply hop behind it still runs for real).
  function interceptTouchGrace(win) {
    return interceptTimers(win, 300, 300);
  }

  await test('History: an identical refresh keeps the same DOM rows (no lost click, no lost focus)', async () => {
    const {win, posted} = makePage();
    const page1 = () => [
      session('A', 'a1', todayNoon),
      session('B', 'b1', todayNoon - 600),
    ];
    sendHistory(win, posted, 0, page1());
    const rowBefore = firstRow(win);
    rowBefore.focus();
    // A `tasks_updated` broadcast (ANY task on the daemon persisted a
    // result, or the post-ready nudge) refetches the first page.  When
    // it comes back identical, the rendered rows must be KEPT: wiping
    // and rebuilding them swallows a click in flight (mousedown on the
    // old row, mouseup on its replacement) and drops keyboard focus.
    const asksBefore = ofType(posted, 'getHistory').length;
    send(win, {type: 'tasks_updated'});
    const asks = ofType(posted, 'getHistory');
    assert.ok(
      asks.length > asksBefore,
      'tasks_updated refetches the open history panel',
    );
    sendHistory(win, posted, 0, page1());
    assert.strictEqual(
      firstRow(win),
      rowBefore,
      'identical refresh must keep the existing row node',
    );
    assert.strictEqual(
      win.document.activeElement,
      rowBefore,
      'keyboard focus survives an identical refresh',
    );
    // The KEPT row's handler is alive: clicking it opens its task in a
    // tab (this is the click the old wipe-and-rebuild used to swallow).
    const activeBefore = win._testApi.getActiveTabId();
    const resumesBefore = ofType(posted, 'resumeSession').length;
    click(win, rowBefore);
    assertRowOpenedTask(win, posted, activeBefore, resumesBefore, 'a1');
    win.close();
  });

  await test('History: changed data still rebuilds the rows', async () => {
    const {win, posted} = makePage();
    sendHistory(win, posted, 0, [
      session('A', 'a1', todayNoon),
      session('B', 'b1', todayNoon - 600),
    ]);
    const rowBefore = firstRow(win);
    send(win, {type: 'tasks_updated'});
    sendHistory(win, posted, 0, [
      session('B', 'b2', todayNoon + 60),
      session('A', 'a1', todayNoon),
      session('B', 'b1', todayNoon - 600),
    ]);
    assert.notStrictEqual(
      firstRow(win),
      rowBefore,
      'changed data must rebuild the rows',
    );
    assert.deepStrictEqual(listShape(win), [
      'sep:Today',
      'chat:B[task b2,task b1]',
      'chat:A[task a1]',
    ]);
    win.close();
  });

  await test('History: an identical refresh retires every stale pagination loader', async () => {
    const {win, posted} = makePage();
    const page1 = () => [
      session('A', 'a1', todayNoon),
      session('B', 'b1', todayNoon - 600),
    ];
    sendHistory(win, posted, 0, page1());
    const rowBefore = firstRow(win);
    // Loaders can pile up: the scroll handler appends one, then a
    // broadcast-driven refresh resets historyLoading while it is still
    // on screen, so a second scroll appends another (duplicate ids).
    // The old offset-0 wipe cleared them as a side effect; the reply
    // itself must now retire ALL of them even on the fast path.
    const list = byId(win, 'history-list');
    for (let i = 0; i < 2; i++) {
      const loader = win.document.createElement('div');
      loader.className = 'sidebar-loading';
      loader.id = 'history-loader';
      loader.textContent = 'Loading...';
      list.appendChild(loader);
    }
    send(win, {type: 'tasks_updated'});
    sendHistory(win, posted, 0, page1());
    assert.strictEqual(
      all(win, '#history-loader').length,
      0,
      'every loader row is removed by the identical refresh',
    );
    assert.strictEqual(firstRow(win), rowBefore, 'rows still kept');
    win.close();
  });

  await test('History: a huge identical page is still kept (field comparison, no serialization)', async () => {
    const {win, posted} = makePage();
    // Rows are compared field by field (huge prompt strings directly),
    // so even multi-megabyte pages take the fast path when unchanged.
    const big = 'x'.repeat(600 * 1024);
    const page1 = () => [
      session('A', 'a1', todayNoon, {title: big, preview: big}),
      session('B', 'b1', todayNoon - 600),
    ];
    sendHistory(win, posted, 0, page1());
    const rowBefore = firstRow(win);
    send(win, {type: 'tasks_updated'});
    sendHistory(win, posted, 0, page1());
    assert.strictEqual(
      firstRow(win),
      rowBefore,
      'a huge identical page must still keep its row nodes',
    );
    win.close();
  });

  await test('History: a kept row with a ticking duration gets its metrics refreshed in place', async () => {
    const {win, posted} = makePage();
    // A running task's duration is `Date.now() - startTs`; the fast
    // path must keep the node (so clicks/focus survive) but refresh the
    // metrics text so the duration keeps ticking.  This is the shape of
    // every running row and of completed rows lacking a recorded end.
    const startTs = Date.now() - 59000;
    const page1 = () => [
      session('A', 'a1', todayNoon, {is_running: true, startTs}),
      session('B', 'b1', todayNoon - 600),
    ];
    sendHistory(win, posted, 0, page1());
    const rowBefore = firstRow(win);
    const metricsEl = rowBefore.querySelector('.running-item-metrics');
    const textBefore = metricsEl.textContent;
    assert.ok(/00:00:59|00:01:0\d/.test(textBefore), textBefore);
    const realNow = win.Date.now;
    win.Date.now = () => realNow() + 60000;
    try {
      send(win, {type: 'tasks_updated'});
      sendHistory(win, posted, 0, page1());
      assert.strictEqual(
        firstRow(win),
        rowBefore,
        'the ticking row node is kept',
      );
      const textAfter = rowBefore.querySelector(
        '.running-item-metrics',
      ).textContent;
      assert.notStrictEqual(
        textAfter,
        textBefore,
        'the duration must be recomputed in place',
      );
      assert.ok(/00:01:59|00:02:0\d/.test(textAfter), textAfter);
    } finally {
      win.Date.now = realNow;
    }
    win.close();
  });

  await test('History: a rebuild with CHANGED data is deferred while the mouse is pressed on a row', async () => {
    const {win, posted} = makePage();
    sendHistory(win, posted, 0, [
      session('A', 'a1', todayNoon),
      session('B', 'b1', todayNoon - 600),
    ]);
    const rowBefore = firstRow(win);
    // Press the row, and let a refresh with DIFFERENT data land while
    // the button is still down: rebuilding now would swallow the click.
    rowBefore.dispatchEvent(
      new win.MouseEvent('mousedown', {bubbles: true, cancelable: true}),
    );
    send(win, {type: 'tasks_updated'});
    sendHistory(win, posted, 0, [
      session('C', 'c1', todayNoon + 60),
      session('A', 'a1', todayNoon),
      session('B', 'b1', todayNoon - 600),
    ]);
    assert.strictEqual(
      firstRow(win),
      rowBefore,
      'the rebuild is deferred while the press is held',
    );
    rowBefore.dispatchEvent(
      new win.MouseEvent('mouseup', {bubbles: true, cancelable: true}),
    );
    // The rebuild must not run synchronously inside the mouseup
    // dispatch either: the browser synthesizes the click right after
    // mouseup, so a rebuild there would still swallow it.
    assert.strictEqual(
      firstRow(win),
      rowBefore,
      'no rebuild during the mouseup dispatch',
    );
    const activeBefore = win._testApi.getActiveTabId();
    const resumesBefore = ofType(posted, 'resumeSession').length;
    click(win, rowBefore);
    assertRowOpenedTask(win, posted, activeBefore, resumesBefore, 'a1');
    await sleep(0);
    assert.notStrictEqual(
      firstRow(win),
      rowBefore,
      'the deferred rebuild lands right after the click',
    );
    assert.deepStrictEqual(listShape(win), [
      'sep:Today',
      'chat:C[task c1]',
      'chat:A[task a1]',
      'chat:B[task b1]',
    ]);
    win.close();
  });

  await test('History: a rebuild is deferred during a TOUCH press (pointer events)', async () => {
    const {win, posted} = makePage();
    sendHistory(win, posted, 0, [
      session('A', 'a1', todayNoon),
      session('B', 'b1', todayNoon - 600),
    ]);
    const rowBefore = firstRow(win);
    const grace = interceptTouchGrace(win);
    // Touch: only pointer events fire while the finger is down; the
    // compatibility mouse events arrive after the finger lifts.
    rowBefore.dispatchEvent(
      new win.MouseEvent('pointerdown', {bubbles: true, cancelable: true}),
    );
    send(win, {type: 'tasks_updated'});
    sendHistory(win, posted, 0, [
      session('C', 'c1', todayNoon + 60),
      session('A', 'a1', todayNoon),
      session('B', 'b1', todayNoon - 600),
    ]);
    assert.strictEqual(
      firstRow(win),
      rowBefore,
      'the rebuild is deferred while the touch is held',
    );
    // On a tap, Chromium dispatches pointerup FIRST and the
    // compatibility mousedown/mouseup/click AFTER it: the parked page
    // must not land in that gap, or the click would hit whatever row
    // the rebuild put under the finger.
    const upEvent = new win.MouseEvent('pointerup', {
      bubbles: true,
      cancelable: true,
    });
    Object.defineProperty(upEvent, 'pointerType', {value: 'touch'});
    rowBefore.dispatchEvent(upEvent);
    assert.strictEqual(grace.timers.length, 1, 'the lift arms the grace timer');
    await sleep(0);
    assert.strictEqual(
      firstRow(win),
      rowBefore,
      'the parked page must wait out the compatibility-event gap',
    );
    const activeBefore = win._testApi.getActiveTabId();
    const resumesBefore = ofType(posted, 'resumeSession').length;
    click(win, rowBefore);
    assertRowOpenedTask(win, posted, activeBefore, resumesBefore, 'a1');
    grace.fire();
    await sleep(0);
    assert.notStrictEqual(
      firstRow(win),
      rowBefore,
      'the deferred rebuild lands after the grace period',
    );
    win.close();
  });

  await test('History: a reused touch pointer ID does not release a newer touch', async () => {
    const {win, posted} = makePage();
    sendHistory(win, posted, 0, [
      session('A', 'a1', todayNoon),
      session('B', 'b1', todayNoon - 600),
    ]);
    const rowBefore = firstRow(win);
    const grace = interceptTouchGrace(win);
    const pev = (type, id) => {
      const e = new win.MouseEvent(type, {bubbles: true, cancelable: true});
      Object.defineProperty(e, 'pointerId', {value: id});
      Object.defineProperty(e, 'pointerType', {value: 'touch'});
      return e;
    };
    // Touch 1 presses, a changed page lands and is parked, the finger
    // lifts (arming the 300ms compatibility-event grace timer) ...
    rowBefore.dispatchEvent(pev('pointerdown', 7));
    send(win, {type: 'tasks_updated'});
    sendHistory(win, posted, 0, [
      session('C', 'c1', todayNoon + 60),
      session('A', 'a1', todayNoon),
      session('B', 'b1', todayNoon - 600),
    ]);
    rowBefore.dispatchEvent(pev('pointerup', 7));
    // ... and touch 2 begins INSIDE the grace period, with the browser
    // reusing pointer ID 7 for the new contact.  Touch 1's grace timer
    // must not release touch 2's press.
    rowBefore.dispatchEvent(pev('pointerdown', 7));
    grace.fire();
    await sleep(0);
    assert.strictEqual(
      firstRow(win),
      rowBefore,
      'a reused pointer ID must not release a newer touch',
    );
    // Touch 2 lifts; its own grace period ends; the parked page lands.
    rowBefore.dispatchEvent(pev('pointerup', 7));
    grace.fire();
    await sleep(0);
    assert.notStrictEqual(
      firstRow(win),
      rowBefore,
      'the parked page lands after the second touch releases',
    );
    win.close();
  });

  await test('History: a parked page stays parked when a second press begins before the deferred apply', async () => {
    const {win, posted} = makePage();
    sendHistory(win, posted, 0, [
      session('A', 'a1', todayNoon),
      session('B', 'b1', todayNoon - 600),
    ]);
    const rowBefore = firstRow(win);
    rowBefore.dispatchEvent(
      new win.MouseEvent('mousedown', {bubbles: true, cancelable: true}),
    );
    send(win, {type: 'tasks_updated'});
    sendHistory(win, posted, 0, [
      session('C', 'c1', todayNoon + 60),
      session('A', 'a1', todayNoon),
      session('B', 'b1', todayNoon - 600),
    ]);
    // Release press 1 (queues the zero-delay apply), then start press 2
    // synchronously, BEFORE that macrotask can run: the apply must find
    // the new press held and re-park the page instead of rebuilding.
    rowBefore.dispatchEvent(
      new win.MouseEvent('mouseup', {bubbles: true, cancelable: true}),
    );
    rowBefore.dispatchEvent(
      new win.MouseEvent('mousedown', {bubbles: true, cancelable: true}),
    );
    await sleep(0);
    assert.strictEqual(
      firstRow(win),
      rowBefore,
      'the deferred apply must re-park while a second press is held',
    );
    rowBefore.dispatchEvent(
      new win.MouseEvent('mouseup', {bubbles: true, cancelable: true}),
    );
    await sleep(0);
    assert.notStrictEqual(
      firstRow(win),
      rowBefore,
      'the re-parked page lands after the second release',
    );
    assert.deepStrictEqual(listShape(win), [
      'sep:Today',
      'chat:C[task c1]',
      'chat:A[task a1]',
      'chat:B[task b1]',
    ]);
    win.close();
  });

  await test('History: a rebuild is deferred while a keyboard activation (Space) is held', async () => {
    const {win, posted} = makePage();
    sendHistory(win, posted, 0, [
      session('A', 'a1', todayNoon),
      session('B', 'b1', todayNoon - 600),
    ]);
    const rowBefore = firstRow(win);
    const fav = rowBefore.querySelector('.sidebar-item-favorite');
    fav.focus();
    // A native button activates Space on KEYUP: a rebuild landing
    // between keydown and keyup would detach the button and swallow
    // the activation, exactly like a swallowed mouse click.
    fav.dispatchEvent(
      new win.KeyboardEvent('keydown', {
        key: ' ',
        bubbles: true,
        cancelable: true,
      }),
    );
    send(win, {type: 'tasks_updated'});
    sendHistory(win, posted, 0, [
      session('C', 'c1', todayNoon + 60),
      session('A', 'a1', todayNoon),
      session('B', 'b1', todayNoon - 600),
    ]);
    assert.strictEqual(
      firstRow(win),
      rowBefore,
      'the rebuild is deferred while Space is held on a list control',
    );
    // Overlapping activation keys: pressing and RELEASING Enter while
    // Space is still held must not release the guard.
    fav.dispatchEvent(
      new win.KeyboardEvent('keydown', {
        key: 'Enter',
        bubbles: true,
        cancelable: true,
      }),
    );
    fav.dispatchEvent(
      new win.KeyboardEvent('keyup', {
        key: 'Enter',
        bubbles: true,
        cancelable: true,
      }),
    );
    await sleep(0);
    assert.strictEqual(
      firstRow(win),
      rowBefore,
      'releasing Enter must not release the still-held Space',
    );
    // Chromium dispatches the Space keyup FIRST and synthesizes the
    // button's click right after it (before any queued macrotask): the
    // guard's release must not rebuild synchronously inside the keyup
    // dispatch, or the click would hit a detached button.
    fav.dispatchEvent(
      new win.KeyboardEvent('keyup', {
        key: ' ',
        bubbles: true,
        cancelable: true,
      }),
    );
    assert.ok(
      fav.isConnected,
      'no rebuild during the keyup dispatch (the activation click is next)',
    );
    click(win, fav);
    await sleep(0);
    assert.notStrictEqual(
      firstRow(win),
      rowBefore,
      'the parked page lands after the keyboard activation',
    );
    const rebuiltA = all(win, '#history-list .sidebar-item')[1];
    assert.ok(
      rebuiltA
        .querySelector('.sidebar-item-favorite')
        .classList.contains('favorited'),
      'the activation took effect (optimistic star on the rebuilt row)',
    );
    win.close();
  });

  await test('History: a press whose release never arrives cannot park the panel forever', async () => {
    const {win, posted} = makePage();
    sendHistory(win, posted, 0, [
      session('A', 'a1', todayNoon),
      session('B', 'b1', todayNoon - 600),
    ]);
    const rowBefore = firstRow(win);
    // Fold the Source Control section: its own task-news debounce
    // timer is not the one under test.
    click(win, byId(win, 'meta-scm').querySelector('.meta-section-toggle'));
    const {timers, restore} = interceptTimers(win, 250, 5500);
    const realNow = win.Date.now;
    const T = realNow();
    try {
      win.Date.now = () => T;
      // A press with no matching release (the release event was eaten:
      // e.g. a drag released outside the page's iframe).
      rowBefore.dispatchEvent(
        new win.MouseEvent('mousedown', {bubbles: true, cancelable: true}),
      );
      send(win, {type: 'tasks_updated'});
      sendHistory(win, posted, 0, [
        session('C', 'c1', todayNoon + 60),
        session('A', 'a1', todayNoon),
        session('B', 'b1', todayNoon - 600),
      ]);
      assert.strictEqual(
        firstRow(win),
        rowBefore,
        'parked while the press appears held',
      );
      assert.strictEqual(
        timers.length,
        1,
        'a stale-press safety timer is armed with the parked page',
      );
      // A NEW press begins two seconds in: when the old timer fires at
      // the bound it must NOT clear the younger press's latches — it
      // re-arms for that press's own remainder instead.
      win.Date.now = () => T + 2000;
      firstRow(win).dispatchEvent(
        new win.MouseEvent('mousedown', {bubbles: true, cancelable: true}),
      );
      win.Date.now = () => T + 5000;
      timers[0].fn();
      await sleep(0);
      assert.strictEqual(
        firstRow(win),
        rowBefore,
        'the safety timer must not release a press younger than the bound',
      );
      assert.strictEqual(timers.length, 2, 're-armed for the remainder');
      assert.ok(
        timers[1].ms <= 2000,
        `re-armed for the remainder, got ${timers[1].ms}ms`,
      );
      // The younger press too never releases: at ITS bound the latches
      // are dropped and the parked page finally lands.
      win.Date.now = () => T + 7100;
      timers[1].fn();
      await sleep(0);
      assert.notStrictEqual(
        firstRow(win),
        rowBefore,
        'the safety timer drops the stale latches and lands the page',
      );
    } finally {
      restore();
      win.Date.now = realNow;
    }
    win.close();
  });

  await test('History: a favourite click alone arms the reconciliation refetch', async () => {
    const {win, posted} = makePage();
    sendHistory(win, posted, 0, [
      session('A', 'a1', todayNoon),
      session('B', 'b1', todayNoon - 600),
    ]);
    const {timers, restore} = interceptTimers(win, 9500, 10500);
    const realNow = win.Date.now;
    try {
      // The write has no acknowledgement and may FAIL: with no other
      // refresh ever arriving, the click itself must plan the refetch
      // that lets the daemon's value reappear at the bound.
      click(win, firstRow(win).querySelector('.sidebar-item-favorite'));
      assert.strictEqual(
        timers.length,
        1,
        'the click armed the reconciliation refetch',
      );
      win.Date.now = () => realNow() + 10200;
      const asked = ofType(posted, 'getHistory').length;
      timers[0].fn();
      assert.ok(
        ofType(posted, 'getHistory').length > asked,
        'the refetch was posted with no page ever delivered in between',
      );
      // The (failed) write never happened on the daemon: past the
      // bound its contrary value rules again.
      sendHistory(win, posted, 0, [
        session('A', 'a1', todayNoon),
        session('B', 'b1', todayNoon - 600),
      ]);
      assert.ok(
        !firstRow(win)
          .querySelector('.sidebar-item-favorite')
          .classList.contains('favorited'),
        'the daemon value shows once the overlay expired',
      );
    } finally {
      restore();
      win.Date.now = realNow;
    }
    win.close();
  });

  await test('History: hiding the page mid-press releases the guard', async () => {
    const {win, posted} = makePage();
    sendHistory(win, posted, 0, [
      session('A', 'a1', todayNoon),
      session('B', 'b1', todayNoon - 600),
    ]);
    const rowBefore = firstRow(win);
    const pev = (type, id) => {
      const e = new win.MouseEvent(type, {bubbles: true, cancelable: true});
      Object.defineProperty(e, 'pointerId', {value: id});
      Object.defineProperty(e, 'pointerType', {value: 'touch'});
      return e;
    };
    // A touch press, then the page is hidden before any release event.
    rowBefore.dispatchEvent(pev('pointerdown', 3));
    send(win, {type: 'tasks_updated'});
    sendHistory(win, posted, 0, [
      session('C', 'c1', todayNoon + 60),
      session('A', 'a1', todayNoon),
      session('B', 'b1', todayNoon - 600),
    ]);
    assert.strictEqual(firstRow(win), rowBefore, 'parked during the press');
    Object.defineProperty(win.document, 'hidden', {
      value: true,
      configurable: true,
    });
    win.document.dispatchEvent(new win.Event('visibilitychange'));
    await sleep(0);
    assert.notStrictEqual(
      firstRow(win),
      rowBefore,
      'hiding the page drops the latches and lands the parked page',
    );
    win.close();
  });

  await test('History: a cancelled press (drag/scroll) releases the guard without a mouseup', async () => {
    const {win, posted} = makePage();
    sendHistory(win, posted, 0, [
      session('A', 'a1', todayNoon),
      session('B', 'b1', todayNoon - 600),
    ]);
    const rowBefore = firstRow(win);
    // Chromium can emit pointerdown -> mousedown -> dragstart ->
    // pointercancel with NO mouseup (dragging selected row text): the
    // cancel must release both flags or rendering stays parked forever.
    rowBefore.dispatchEvent(
      new win.MouseEvent('pointerdown', {bubbles: true, cancelable: true}),
    );
    rowBefore.dispatchEvent(
      new win.MouseEvent('mousedown', {bubbles: true, cancelable: true}),
    );
    rowBefore.dispatchEvent(
      new win.MouseEvent('pointercancel', {bubbles: true, cancelable: true}),
    );
    send(win, {type: 'tasks_updated'});
    sendHistory(win, posted, 0, [
      session('C', 'c1', todayNoon + 60),
      session('A', 'a1', todayNoon),
      session('B', 'b1', todayNoon - 600),
    ]);
    assert.notStrictEqual(
      firstRow(win),
      rowBefore,
      'rendering is not latched after a cancelled press',
    );
    win.close();
  });

  await test('History: a duplicate later page is dropped instead of duplicating rows', async () => {
    const {win, posted} = makePage();
    const fullPage = (tag, base) =>
      Array.from({length: 50}, (_, i) =>
        session('C' + tag + i, 't' + tag + i, base - i * 60),
      );
    sendHistory(win, posted, 0, fullPage('x', todayNoon));
    sendHistory(win, posted, 50, fullPage('y', todayNoon - 5000));
    assert.strictEqual(all(win, '#history-list .sidebar-item').length, 100);
    // A second offset-50 reply (an overlapped same-generation request)
    // no longer extends at the cursor (now 100): it must be dropped.
    sendHistory(win, posted, 50, fullPage('z', todayNoon - 9000));
    assert.strictEqual(
      all(win, '#history-list .sidebar-item').length,
      100,
      'the stale duplicate page is dropped',
    );
    byId(win, 'history-list').dispatchEvent(new win.Event('scroll'));
    const asks = ofType(posted, 'getHistory');
    assert.strictEqual(
      asks[asks.length - 1].offset,
      100,
      'pagination continues at the rendered row count',
    );
    win.close();
  });

  await test('History: no second same-generation request can overlap a pending refresh', async () => {
    const {win, posted} = makePage();
    const fullPage = tag =>
      Array.from({length: 50}, (_, i) =>
        session('C' + tag + i, 't' + tag + i, todayNoon - i * 60),
      );
    sendHistory(win, posted, 0, fullPage('x'));
    // A broadcast resets pagination and refetches; until that reply
    // lands, a bottom-scroll must NOT fire an overlapping request of
    // the same generation (its late reply would re-order pagination).
    send(win, {type: 'tasks_updated'});
    const asked = ofType(posted, 'getHistory').length;
    byId(win, 'history-list').dispatchEvent(new win.Event('scroll'));
    assert.strictEqual(
      ofType(posted, 'getHistory').length,
      asked,
      'the scroll is parked while the refresh is in flight',
    );
    sendHistory(win, posted, 0, fullPage('x'));
    byId(win, 'history-list').dispatchEvent(new win.Event('scroll'));
    assert.strictEqual(
      ofType(posted, 'getHistory').length,
      asked + 1,
      'after the reply, the scroll paginates again',
    );
    win.close();
  });

  await test('History: a same-day clock change alone does not rebuild, a day change does', async () => {
    const {win, posted} = makePage();
    const page1 = () => [
      session('A', 'a1', todayNoon),
      session('B', 'b1', todayNoon - 600),
    ];
    sendHistory(win, posted, 0, page1());
    const rowBefore = firstRow(win);
    // A wall-clock correction across local midnight must relabel
    // "Today"/"Yesterday", so the fast path is off on a new local day.
    const origToDateString = win.Date.prototype.toDateString;
    win.Date.prototype.toDateString = function () {
      return 'Fri Jan 01 2100';
    };
    try {
      send(win, {type: 'tasks_updated'});
      sendHistory(win, posted, 0, page1());
      assert.notStrictEqual(
        firstRow(win),
        rowBefore,
        'a new local day must rebuild (separator labels changed)',
      );
    } finally {
      win.Date.prototype.toDateString = origToDateString;
    }
    win.close();
  });

  await test('History: the favourite overlay plans a refetch so its expiry is visible without broadcasts', async () => {
    const {win, posted} = makePage();
    sendHistory(win, posted, 0, [
      session('A', 'a1', todayNoon),
      session('B', 'b1', todayNoon - 600),
    ]);
    // Capture the expiry timer main.js arms at the toggle (and keeps
    // serving while a page still disagrees inside the bound).
    const {timers: longTimers, fire, restore} = interceptTimers(
      win,
      5001,
      Infinity,
    );
    const realNow = win.Date.now;
    try {
      click(win, firstRow(win).querySelector('.sidebar-item-favorite'));
      send(win, {type: 'tasks_updated'});
      sendHistory(win, posted, 0, [
        session('A', 'a1', todayNoon, {steps: 3}),
        session('B', 'b1', todayNoon - 600),
      ]);
      // At least the expiry refetch is armed (the midnight-relabel
      // timer is captured by the same >5s filter).
      assert.ok(longTimers.length >= 1, 'an expiry refetch is armed');
      // At the deadline the timer refetches; the daemon's (contrary)
      // value now rules because the overlay has expired.
      win.Date.now = () => realNow() + 11000;
      // The tasks_updated above was this tick's refresh; a second one
      // in the same tick folds into the end-of-tick refetch, so let the
      // tick end before the timer fires.
      await sleep(0);
      const asked = ofType(posted, 'getHistory').length;
      fire();
      assert.ok(
        ofType(posted, 'getHistory').length > asked,
        'the expiry timer refetches the history',
      );
      sendHistory(win, posted, 0, [
        session('A', 'a1', todayNoon, {steps: 3}),
        session('B', 'b1', todayNoon - 600),
      ]);
      assert.ok(
        !firstRow(win)
          .querySelector('.sidebar-item-favorite')
          .classList.contains('favorited'),
        'the daemon value shows at the deadline without any broadcast',
      );
    } finally {
      win.Date.now = realNow;
      restore();
    }
    win.close();
  });

  await test('History: the expiry refetch serves the EARLIEST pending favourite deadline', async () => {
    const {win, posted} = makePage();
    sendHistory(win, posted, 0, [
      session('A', 'a1', todayNoon),
      session('B', 'b1', todayNoon - 600),
    ]);
    const realNow = win.Date.now;
    const {timers, restore} = interceptTimers(win, 3001, 19999);
    try {
      const T = realNow();
      const rows = all(win, '#history-list .sidebar-item');
      // A is starred at T, B five seconds later: A's overlay expires at
      // T+10s, B's at T+15s.  The first click arms the timer; the
      // second has a LATER deadline and must not move it.
      win.Date.now = () => T;
      click(win, rows[0].querySelector('.sidebar-item-favorite'));
      assert.strictEqual(timers.length, 1, "A's click armed the timer");
      win.Date.now = () => T + 5000;
      click(win, rows[1].querySelector('.sidebar-item-favorite'));
      assert.strictEqual(
        timers.length,
        1,
        "B's later deadline must not move the armed timer",
      );
      // The timer fires (both overlays still pending) and refetches; the
      // reply's PAGE ORDER puts the later deadline (B) first: the next
      // timer must be re-armed for A's earlier deadline (~4.1s away at
      // T+6s), not left at B's (~9.1s away).
      win.Date.now = () => T + 6000;
      timers[0].fn();
      sendHistory(win, posted, 0, [
        session('B', 'b1', todayNoon - 600, {steps: 3}),
        session('A', 'a1', todayNoon, {steps: 3}),
      ]);
      assert.strictEqual(timers.length, 3, 'armed for B, re-armed for A');
      assert.ok(
        timers[2].ms >= 4000 && timers[2].ms <= 4300,
        `re-armed for the earliest pending deadline, got ${timers[2].ms}ms`,
      );
      // At A's deadline the timer refetches; A's overlay has expired
      // (daemon value rules) while B's still holds.
      win.Date.now = () => T + 10200;
      // Same tick as timers[0]'s refresh: let it end so this one is not
      // folded into its end-of-tick refetch.
      await sleep(0);
      const asked = ofType(posted, 'getHistory').length;
      timers[2].fn();
      assert.ok(
        ofType(posted, 'getHistory').length > asked,
        'the earliest-deadline timer refetches the history',
      );
      sendHistory(win, posted, 0, [
        session('A', 'a1', todayNoon, {steps: 3}),
        session('B', 'b1', todayNoon - 600, {steps: 3}),
      ]);
      const after = all(win, '#history-list .sidebar-item');
      assert.ok(
        !after[0]
          .querySelector('.sidebar-item-favorite')
          .classList.contains('favorited'),
        "A's overlay expired at its own 10s bound",
      );
      assert.ok(
        after[1]
          .querySelector('.sidebar-item-favorite')
          .classList.contains('favorited'),
        "B's younger overlay still holds",
      );
    } finally {
      win.Date.now = realNow;
      restore();
    }
    win.close();
  });

  await test('History: an agreeing page retires the overlay and cancels its refetch', async () => {
    const {win, posted} = makePage();
    sendHistory(win, posted, 0, [
      session('A', 'a1', todayNoon),
      session('B', 'b1', todayNoon - 600),
    ]);
    const {timers: armed, cleared, restore} = interceptTimers(
      win,
      9500,
      10500,
    );
    try {
      click(win, firstRow(win).querySelector('.sidebar-item-favorite'));
      assert.strictEqual(armed.length, 1, 'the click armed the refetch');
      // The daemon's own data AGREES with the toggle: the overlay is
      // retired and the now-pointless reconciliation refetch cancelled.
      send(win, {type: 'tasks_updated'});
      sendHistory(win, posted, 0, [
        session('A', 'a1', todayNoon, {is_favorite: true}),
        session('B', 'b1', todayNoon - 600),
      ]);
      assert.deepStrictEqual(
        cleared,
        [armed[0].id],
        'the agreeing page cancelled the armed refetch',
      );
    } finally {
      restore();
    }
    win.close();
  });

  await test('History: a parked page cannot acknowledge its own favourite overlay', async () => {
    const {win, posted} = makePage();
    sendHistory(win, posted, 0, [
      session('A', 'a1', todayNoon),
      session('B', 'b1', todayNoon - 600),
    ]);
    const realNow = win.Date.now;
    const {timers: armed, cleared, restore} = interceptTimers(
      win,
      9500,
      10500,
    );
    try {
      click(win, firstRow(win).querySelector('.sidebar-item-favorite'));
      assert.strictEqual(armed.length, 1, 'the click armed the refetch');
      // A CONTRARY page (the daemon never saw the write) arrives while
      // the row is pressed: it is merged (overlay applied), parked, and
      // merged AGAIN when it lands after the release.  The second merge
      // must not mistake the overlay for daemon agreement.
      const rowBefore = firstRow(win);
      rowBefore.dispatchEvent(
        new win.MouseEvent('mousedown', {bubbles: true, cancelable: true}),
      );
      send(win, {type: 'tasks_updated'});
      sendHistory(win, posted, 0, [
        session('A', 'a1', todayNoon, {steps: 3}),
        session('B', 'b1', todayNoon - 600),
      ]);
      rowBefore.dispatchEvent(
        new win.MouseEvent('mouseup', {bubbles: true, cancelable: true}),
      );
      await sleep(0);
      assert.notStrictEqual(firstRow(win), rowBefore, 'parked page landed');
      assert.ok(
        firstRow(win)
          .querySelector('.sidebar-item-favorite')
          .classList.contains('favorited'),
        'the optimistic star survives the parked page landing',
      );
      assert.strictEqual(
        cleared.length,
        0,
        'landing the parked page must not cancel the reconciliation refetch',
      );
      // The write really failed: at the bound the refetch lets the
      // daemon's contrary value rule again.
      win.Date.now = () => realNow() + 10200;
      const asked = ofType(posted, 'getHistory').length;
      armed[0].fn();
      assert.ok(
        ofType(posted, 'getHistory').length > asked,
        'the reconciliation refetch still fires at the bound',
      );
      sendHistory(win, posted, 0, [
        session('A', 'a1', todayNoon, {steps: 3}),
        session('B', 'b1', todayNoon - 600),
      ]);
      assert.ok(
        !firstRow(win)
          .querySelector('.sidebar-item-favorite')
          .classList.contains('favorited'),
        'the daemon value rules once the overlay expired',
      );
    } finally {
      restore();
      win.Date.now = realNow;
    }
    win.close();
  });

  await test('History: the favourite overlay expires so the daemon becomes authoritative again', async () => {
    const {win, posted} = makePage();
    sendHistory(win, posted, 0, [
      session('A', 'a1', todayNoon),
      session('B', 'b1', todayNoon - 600),
    ]);
    click(win, firstRow(win).querySelector('.sidebar-item-favorite'));
    // Within the bound, a contrary (pre-write) page keeps the star.
    send(win, {type: 'tasks_updated'});
    sendHistory(win, posted, 0, [
      session('A', 'a1', todayNoon, {steps: 3}),
      session('B', 'b1', todayNoon - 600),
    ]);
    assert.ok(
      firstRow(win)
        .querySelector('.sidebar-item-favorite')
        .classList.contains('favorited'),
      'the optimistic star holds within the bound',
    );
    // Past the bound (another client may have toggled it back, or the
    // write may have failed), the daemon's value rules.
    const realNow = win.Date.now;
    win.Date.now = () => realNow() + 11000;
    try {
      send(win, {type: 'tasks_updated'});
      sendHistory(win, posted, 0, [
        session('A', 'a1', todayNoon, {steps: 4}),
        session('B', 'b1', todayNoon - 600),
      ]);
      assert.ok(
        !firstRow(win)
          .querySelector('.sidebar-item-favorite')
          .classList.contains('favorited'),
        'after the bound the server value rules',
      );
    } finally {
      win.Date.now = realNow;
    }
    win.close();
  });

  await test('History: an optimistic favourite survives a page deferred across its own click', async () => {
    const {win, posted} = makePage();
    sendHistory(win, posted, 0, [
      session('A', 'a1', todayNoon),
      session('B', 'b1', todayNoon - 600),
    ]);
    const favBtn = firstRow(win).querySelector('.sidebar-item-favorite');
    // Press the star; while the button is held, a refresh lands that
    // predates the toggle (star still off) with an unrelated change.
    favBtn.dispatchEvent(
      new win.MouseEvent('mousedown', {bubbles: true, cancelable: true}),
    );
    send(win, {type: 'tasks_updated'});
    sendHistory(win, posted, 0, [
      session('A', 'a1', todayNoon, {steps: 9}),
      session('B', 'b1', todayNoon - 600),
    ]);
    click(win, favBtn);
    favBtn.dispatchEvent(
      new win.MouseEvent('mouseup', {bubbles: true, cancelable: true}),
    );
    await sleep(0);
    const favAfter = firstRow(win).querySelector('.sidebar-item-favorite');
    assert.notStrictEqual(favAfter, favBtn, 'the row was rebuilt');
    assert.ok(
      favAfter.classList.contains('favorited'),
      'the optimistic star is preserved over the stale page',
    );
    // Once the daemon's own data agrees, the overlay retires and the
    // server value keeps ruling.
    send(win, {type: 'tasks_updated'});
    sendHistory(win, posted, 0, [
      session('A', 'a1', todayNoon, {steps: 9, is_favorite: true}),
      session('B', 'b1', todayNoon - 600),
    ]);
    send(win, {type: 'tasks_updated'});
    sendHistory(win, posted, 0, [
      session('A', 'a1', todayNoon, {steps: 9, is_favorite: false}),
      session('B', 'b1', todayNoon - 600),
    ]);
    assert.ok(
      !firstRow(win)
        .querySelector('.sidebar-item-favorite')
        .classList.contains('favorited'),
      'after the daemon agreed once, later server data rules again',
    );
    win.close();
  });

  await test('History: a duplicate offset-0 reply in one generation does not corrupt pagination', async () => {
    const {win, posted} = makePage();
    const fullPage = tag =>
      Array.from({length: 50}, (_, i) =>
        session('C' + tag + i, 't' + tag + i, todayNoon - i * 60),
      );
    sendHistory(win, posted, 0, fullPage('x'));
    // A second offset-0 reply in the SAME generation (a scroll's
    // duplicate racing a deferred rebuild) must assign the cursor, not
    // advance it past what is on screen.
    sendHistory(win, posted, 0, fullPage('y'));
    byId(win, 'history-list').dispatchEvent(new win.Event('scroll'));
    const asks = ofType(posted, 'getHistory');
    const last = asks[asks.length - 1];
    assert.strictEqual(
      last.offset,
      50,
      'the next page is requested at the rendered row count',
    );
    win.close();
  });

  await test('History: keyboard focus moves to the same task after a changed-data rebuild', async () => {
    const {win, posted} = makePage();
    sendHistory(win, posted, 0, [
      session('A', 'a1', todayNoon),
      session('B', 'b1', todayNoon - 600),
    ]);
    const rows = all(win, '#history-list .sidebar-item');
    rows[1].focus();
    send(win, {type: 'tasks_updated'});
    sendHistory(win, posted, 0, [
      session('A', 'a1', todayNoon, {steps: 7}),
      session('B', 'b1', todayNoon - 600),
    ]);
    const rowsAfter = all(win, '#history-list .sidebar-item');
    assert.notStrictEqual(rowsAfter[1], rows[1], 'rows were rebuilt');
    assert.strictEqual(
      win.document.activeElement,
      rowsAfter[1],
      "focus lands on task b1's fresh row",
    );
    // Focus on an inline control (the favourite star) survives as the
    // SAME control on the rebuilt row: the next Space/Enter must repeat
    // the control's action, not open the task.
    rowsAfter[0].querySelector('.sidebar-item-favorite').focus();
    send(win, {type: 'tasks_updated'});
    sendHistory(win, posted, 0, [
      session('A', 'a1', todayNoon, {steps: 8}),
      session('B', 'b1', todayNoon - 600),
    ]);
    const rowsFinal = all(win, '#history-list .sidebar-item');
    assert.strictEqual(
      win.document.activeElement,
      rowsFinal[0].querySelector('.sidebar-item-favorite'),
      "focus lands on task a1's fresh favourite control",
    );
    win.close();
  });

  await test('History: a locale change makes the next refresh rebuild (row times are localized)', async () => {
    const {win, posted} = makePage();
    const page1 = () => [
      session('A', 'a1', todayNoon),
      session('B', 'b1', todayNoon - 600),
    ];
    sendHistory(win, posted, 0, page1());
    const rowBefore = firstRow(win);
    const origDtf = win.Intl.DateTimeFormat;
    win.Intl.DateTimeFormat = function () {
      return {resolvedOptions: () => ({locale: 'de-DE', timeZone: 'UTC'})};
    };
    try {
      send(win, {type: 'tasks_updated'});
      sendHistory(win, posted, 0, page1());
      assert.notStrictEqual(
        firstRow(win),
        rowBefore,
        'a locale change must rebuild the rows',
      );
    } finally {
      win.Intl.DateTimeFormat = origDtf;
    }
    win.close();
  });

  await test('History: a time-zone change makes the next refresh rebuild (row times are localized)', async () => {
    const {win, posted} = makePage();
    const page1 = () => [
      session('A', 'a1', todayNoon),
      session('B', 'b1', todayNoon - 600),
    ];
    sendHistory(win, posted, 0, page1());
    const rowBefore = firstRow(win);
    // Same data, different clock localization: the rows' timestamps
    // were rendered under the old zone, so the fast path must not keep
    // them.
    const origOffset = win.Date.prototype.getTimezoneOffset;
    win.Date.prototype.getTimezoneOffset = function () {
      return origOffset.call(this) + 60;
    };
    try {
      send(win, {type: 'tasks_updated'});
      sendHistory(win, posted, 0, page1());
      assert.notStrictEqual(
        firstRow(win),
        rowBefore,
        'a zone change must rebuild the rows',
      );
    } finally {
      win.Date.prototype.getTimezoneOffset = origOffset;
    }
    win.close();
  });

  await test('History: a later page adds older tasks to existing blocks and opens older ones after', async () => {
    const {win, posted} = makePage();
    sendHistory(win, posted, 0, [
      session('A', 'a2', todayNoon),
      session('B', 'b1', todayNoon - 60),
    ]);
    sendHistory(win, posted, 2, [
      session('A', 'a1', todayNoon - 120),
      session('C', 'c1', todayNoon - DAY),
    ]);
    assert.deepStrictEqual(listShape(win), [
      'sep:Today',
      'chat:A[task a2,task a1]',
      'chat:B[task b1]',
      'sep:Yesterday',
      'chat:C[task c1]',
    ]);
    // A fresh first page replaces everything (no duplicate blocks).
    sendHistory(win, posted, 0, [
      session('B', 'b2', todayNoon + 1),
      session('A', 'a2', todayNoon),
    ]);
    assert.deepStrictEqual(listShape(win), [
      'sep:Today',
      'chat:B[task b2]',
      'chat:A[task a2]',
    ]);
    assert.strictEqual(all(win, '.history-day-sep').length, 1);
    // An empty first page shows the empty note, no separators.
    sendHistory(win, posted, 0, []);
    assert.strictEqual(all(win, '.history-day-sep').length, 0);
    assert.ok(
      /No conversations yet/.test(byId(win, 'history-list').textContent),
    );
    win.close();
  });

  await test('History: filters hide blocks and separators left without a visible task', async () => {
    const {win, posted} = makePage();
    sendHistory(win, posted, 0, [
      session('A', 'a1', todayNoon, {failed: true}),
      session('B', 'b1', todayNoon - DAY),
      session('C', 'c1', todayNoon - 2 * DAY, {failed: true}),
    ]);
    assert.strictEqual(listShape(win).length, 6);
    // Hide failed tasks: chats A and C go, with the Today separator and
    // the two-days-ago one; Yesterday / B stay.
    const errors = byId(win, 'hf-errors');
    assert.ok(errors, 'errors filter chip');
    click(win, errors);
    const shape = listShape(win);
    assert.deepStrictEqual(shape, ['sep:Yesterday', 'chat:B[task b1]']);
    // Show them again: everything is back.
    click(win, errors);
    assert.strictEqual(listShape(win).length, 6);
    win.close();
  });

  await test('History: epoch 0 is a date, a missing timestamp is "Undated"; other years show the year', async () => {
    const {win, posted} = makePage();
    const lastYear = new Date();
    lastYear.setFullYear(lastYear.getFullYear() - 1);
    lastYear.setHours(12, 0, 0, 0);
    const lastYearTs = Math.floor(lastYear.getTime() / 1000);
    const missing = session('C', 'c1', 0);
    delete missing.timestamp;
    sendHistory(win, posted, 0, [
      session('A', 'a1', lastYearTs),
      session('B', 'b1', 0),
      missing,
      session('D', 'd1', 'not a number'),
      session('E', 'e1', 1e300),
    ]);
    const seps = all(win, '.history-day-sep').map(el => el.textContent);
    const fmt = d =>
      d.toLocaleDateString(undefined, {
        weekday: 'short',
        month: 'short',
        day: 'numeric',
        year: 'numeric',
      });
    // Epoch zero is 1970 (the date filter treats it the same way); the
    // missing, the unparsable and the beyond-Date timestamps share one
    // "Undated".
    assert.deepStrictEqual(seps, [fmt(lastYear), fmt(new Date(0)), 'Undated']);
    win.close();
  });

  await test('History: separators are rebuilt when the page becomes visible again (midnight, time zone)', async () => {
    const {win, posted} = makePage();
    sendHistory(win, posted, 0, [session('A', 'a1', todayNoon)]);
    let sep = all(win, '.history-day-sep')[0];
    assert.strictEqual(sep.textContent, 'Today');
    assert.strictEqual(sep.dataset.ts, String(todayNoon));
    // A label computed on another day is recomputed when the page wakes.
    sep.textContent = 'Yesterday';
    win.document.dispatchEvent(new win.Event('visibilitychange'));
    sep = all(win, '.history-day-sep')[0];
    assert.strictEqual(all(win, '.history-day-sep').length, 1);
    assert.strictEqual(sep.textContent, 'Today');
    // A time-zone change can move a chat to another local day: the
    // partition is rebuilt, not just the words.  2026-01-10T00:30Z and
    // 2026-01-09T23:30Z are two days in UTC, one day in Los Angeles.
    const savedTz = process.env.TZ;
    process.env.TZ = 'UTC';
    try {
      const t1 = Date.UTC(2026, 0, 10, 0, 30) / 1000;
      const t2 = Date.UTC(2026, 0, 9, 23, 30) / 1000;
      sendHistory(win, posted, 0, [
        session('B', 'b1', t1),
        session('C', 'c1', t2),
      ]);
      assert.deepStrictEqual(
        all(win, '.history-day-sep').map(el => el.dataset.day),
        ['2026-1-10', '2026-1-9'],
      );
      process.env.TZ = 'America/Los_Angeles';
      win.document.dispatchEvent(new win.Event('visibilitychange'));
      assert.deepStrictEqual(
        all(win, '.history-day-sep').map(el => el.dataset.day),
        ['2026-1-9'],
        'one local day after the zone change',
      );
      assert.deepStrictEqual(listShape(win).slice(0, 3), [
        'sep:' +
          new Date(t1 * 1000).toLocaleDateString(undefined, {
            weekday: 'short',
            month: 'short',
            day: 'numeric',
          }),
        'chat:B[task b1]',
        'chat:C[task c1]',
      ]);
    } finally {
      if (savedTz === undefined) delete process.env.TZ;
      else process.env.TZ = savedTz;
    }
    win.close();
  });

  report('historyGroups');
}

main().catch(e => {
  console.error(e);
  process.exit(1);
});
