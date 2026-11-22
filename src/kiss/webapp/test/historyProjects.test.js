// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end (jsdom) tests of the project panels of the Chats list: the
// real chat.html, main.js and projects.js over a recording transport.
//
// With Group by: Project (the default) and at least one project:
//
// * every project is a foldable panel, under Running while a task of
//   one of its chats runs, under Pinned when it is pinned, else under
//   Projects; the chats that are in no project stay under their days;
// * the panel's header is the Projects icon, the name, the fold button
//   and "…": unfolding lists every chat of the project (a page of its
//   own, asked from the daemon), "…" opens the project's settings and
//   a click on the header starts a new chat in the project;
// * a panel whose page was a full one ends with "Show more chats";
// * Group by: Chat or None, a search, a tag and the list of another
//   machine's tasks show no panel.
//
// projectsPage.test.js covers the Projects page and the chat header's
// project drop-down.

'use strict';

const assert = require('assert');
const {
  makePage,
  send,
  click,
  byId,
  ofType,
  typeInto,
  pickMachine,
  sleep,
  captureTimers,
  fireCapturedTimers,
  recordTimers,
  fireTimers,
} = require('./simplify2_harness');
const {makeRunner, key} = require('./ui_antipattern_harness');
const {MACHINES} = require('./settings_helpers');
const {
  groupedSession,
  todayNoonSec,
  freezeClock,
  pressFilterOption,
  tickedFilterOptions,
  selectFilter,
  rowByTitle,
  chatHeader,
} = require('./history_helpers');
const {
  answerWorkDirCheck,
  currentDir,
  errorNotifications,
} = require('./workdir_helpers');

const windows = [];
const {test, report} = makeRunner({windows});

// Noon today: every row of these tests ran today.  The pages' clock
// stands at that instant (openList), so a run that crosses midnight
// still lists the rows under Today.
const T = todayNoonSec();
// The rows of one project the list asks for at a time.
const PAGE = 200;

/** A project as the daemon's `projects` event lists it. */
function project(id, name, extra) {
  return Object.assign(
    {
      id,
      name,
      description: '',
      settings: {},
      isPinned: false,
      createdAt: T - 9000,
      updatedAt: T - 9000,
    },
    extra,
  );
}

const PAPER = project('p1', 'Paper');
const SITE = project('p2', 'Site', {isPinned: true});
// Chats c1 and c2 are in Paper, c3 is in Site, c4 is in no project.
const CHAT_PROJECTS = {c1: 'p1', c2: 'p1', c3: 'p2'};
const STREAM = [
  groupedSession('c4', 14, T - 10),
  groupedSession('c1', 11, T - 20),
  groupedSession('c3', 13, T - 30),
  groupedSession('c2', 12, T - 40),
  groupedSession('c1', 10, T - 50),
];
// The page of Paper's rows: the tasks of c1 and c2.
const PAPER_ROWS = [STREAM[1], STREAM[3], STREAM[4]];

/** Plain copy of a value made in the page's realm. */
function plain(value) {
  return JSON.parse(JSON.stringify(value));
}

/**
 * A page with the Chats drawer open that knows `projects` and the
 * chats in them and lists `stream` (default STREAM).  `pageOpts` are
 * further options of the page (`beforeScripts`, `url`); its
 * `beforeScripts` runs after the page's clock is frozen at T.
 */
function openList(projects, chatProjects, stream, pageOpts) {
  const {beforeScripts, ...opts} = pageOpts || {};
  const page = makePage(
    Object.assign(
      {
        bodyClass: 'remote-chat',
        extraScripts: ['projects.js'],
        sourceUrl: {'main.js': 'projects-main.js', 'projects.js': 'projects-page.js'},
        beforeScripts: w => {
          freezeClock(w, T * 1000);
          if (beforeScripts) beforeScripts(w);
        },
      },
      opts,
    ),
  );
  windows.push(page.win);
  byId(page.win, 'sidebar').classList.add('open');
  send(page.win, {type: 'configData', config: {work_dir: '/work/ws'}, apiKeys: {}});
  send(page.win, {type: 'projects', projects, chatProjects});
  sendStream(page.win, page.posted, stream || STREAM);
  return page;
}

/** The page's requests for the list's own rows. */
function streamRequests(posted) {
  return ofType(posted, 'getHistory').filter(m => !m.project);
}

/** The page's requests for the rows of projects, from `posted[from]` on. */
function projectRequests(posted, from) {
  return plain(ofType(posted.slice(from || 0), 'getHistory').filter(m => m.project));
}

/**
 * The generation of the list's view: what its last request for its own
 * rows carries, and the daemon's answers to the view's requests echo.
 */
function generation(posted) {
  const req = streamRequests(posted).pop();
  return req ? req.generation : 0;
}

/** What the page sends for `limit` rows of project `id` from row `offset` on. */
function pageRequest(posted, id, workDir, offset, limit) {
  return {
    type: 'getHistory',
    project: id,
    workDir,
    offset,
    limit,
    generation: generation(posted),
  };
}

/** Answer the list's last request for its own rows with `sessions`. */
function sendStream(win, posted, sessions, extra) {
  send(
    win,
    Object.assign(
      {type: 'history', offset: 0, generation: generation(posted), sessions},
      extra,
    ),
  );
}

/**
 * The daemon's page of project `id`'s rows from row `offset` on, asked
 * to hold `limit` rows (default: one page), for the list's view.
 */
function sendProjectPage(win, posted, id, sessions, offset, limit) {
  send(win, {
    type: 'history',
    project: id,
    offset: offset || 0,
    limit: limit || PAGE,
    generation: generation(posted),
    sessions,
  });
}

/** `count` rows of chat `chat`, newest first: tasks `first`, `first` - 1, ... */
function rowsOf(chat, first, count) {
  const rows = [];
  for (let i = 0; i < count; i++) {
    rows.push(groupedSession(chat, first - i, T - 5000 + first - i));
  }
  return rows;
}

/**
 * A full page of rows, newest first: tasks `first`, `first` - 1, ...,
 * each the one task of a chat of its own (`<chat>-0`, `<chat>-1`, ...).
 */
function pageOf(chat, first) {
  const rows = [];
  for (let i = 0; i < PAGE; i++) {
    rows.push(groupedSession(chat + '-' + i, first - i, T - 5000 + first - i));
  }
  return rows;
}

/** How many task rows the panel of project `id` holds. */
function rowCount(win, id) {
  return panelOf(win, id).querySelectorAll('.sidebar-item').length;
}

/** "Show more chats" of project `id`'s panel, or null. */
function moreRows(win, id) {
  return panelOf(win, id).querySelector('.history-project-more-rows');
}

/** Have the chat on screen show task `taskId` of chat `chatId`. */
function showTask(win, chatId, taskId) {
  send(win, {
    type: 'task_settings',
    tabId: win._testApi.getActiveTabId(),
    taskId: String(taskId),
    settings: {model: 'm', chat_id: chatId, task_id: taskId, start_ts: 1},
  });
}

/** The panel of project `id` in the list. */
function panelOf(win, id) {
  return win.document.querySelector(
    '#history-list .history-project[data-project-id="' + id + '"]',
  );
}

/** Press the button `.cls` of project `id`'s panel. */
function press(win, id, cls) {
  click(win, panelOf(win, id).querySelector('.' + cls));
}

/** A chat panel as `chat:<id>[<its visible rows' titles>]`. */
function chatShape(group) {
  const titles = Array.from(group.querySelectorAll('.sidebar-item'))
    .filter(r => r.style.display !== 'none')
    .map(r => r.querySelector('.sidebar-item-text').textContent);
  return 'chat:' + group.dataset.chatId + '[' + titles.join(',') + ']';
}

/** What the body of a project's `panel` shows, in order. */
function bodyShape(panel) {
  const body = panel.querySelector(':scope > .history-project-body');
  const chats = Array.from(body.querySelectorAll(':scope > .history-chat-group'))
    .filter(g => g.style.display !== 'none')
    .map(chatShape);
  const parts = chats.length
    ? chats
    : [body.querySelector('.history-project-note').textContent];
  if (body.querySelector('.history-project-more-rows')) parts.push('more');
  return parts;
}

/**
 * The visible top-level children of the list: `sep:<label>`, a chat
 * panel (chatShape), `row:<title>` and a project's panel, which reads
 * `project:<id>` while folded and `project:<id>{<its body>}` unfolded.
 */
function shape(win) {
  return Array.from(byId(win, 'history-list').children)
    .filter(el => el.style.display !== 'none')
    .map(el => {
      if (el.classList.contains('history-day-sep')) return 'sep:' + el.textContent;
      if (el.classList.contains('history-chat-group')) return chatShape(el);
      if (el.classList.contains('sidebar-item')) {
        return 'row:' + el.querySelector('.sidebar-item-text').textContent;
      }
      const name = 'project:' + el.dataset.projectId;
      return el.classList.contains('collapsed')
        ? name
        : name + '{' + bodyShape(el).join(' ') + '}';
    });
}

test('a project is a foldable panel of the list: unfolding lists its chats, "\u2026" opens its settings and its header starts a new chat in it', () => {
  const {win, posted} = openList([PAPER, SITE], CHAT_PROJECTS);
  // The pinned project, the other one, then the chat that is in none.
  assert.deepStrictEqual(shape(win), [
    'sep:Pinned',
    'project:p2',
    'sep:Projects',
    'project:p1',
    'sep:Today',
    'chat:c4[task 14]',
  ]);
  const header = panelOf(win, 'p1').querySelector('.history-project-header');
  assert.deepStrictEqual(
    Array.from(header.children, c => c.className.split(' ').pop()),
    [
      'history-project-icon',
      'history-project-title',
      'history-project-collapse',
      'history-project-more',
    ],
  );
  assert.strictEqual(header.querySelector('.history-project-title').textContent, 'Paper');
  assert.ok(header.querySelector('.history-project-icon svg'), 'the Projects icon');
  const fold = header.querySelector('.history-project-collapse');
  assert.deepStrictEqual(
    [fold.getAttribute('aria-expanded'), fold.dataset.tooltip],
    ['false', 'Show chats'],
  );

  // Unfolding asks the daemon for the project's rows, and lists every
  // chat of it: the tasks stay grouped by chat.
  let from = posted.length;
  click(win, fold);
  assert.deepStrictEqual(projectRequests(posted, from), [
    pageRequest(posted, 'p1', '', 0, PAGE),
  ]);
  assert.deepStrictEqual(
    [fold.getAttribute('aria-expanded'), fold.dataset.tooltip],
    ['true', 'Hide chats'],
  );
  assert.strictEqual(shape(win)[3], 'project:p1{Loading...}');
  sendProjectPage(win, posted, 'p1', PAPER_ROWS);
  assert.deepStrictEqual(shape(win), [
    'sep:Pinned',
    'project:p2',
    'sep:Projects',
    'project:p1{chat:c1[task 11,task 10] chat:c2[task 12]}',
    'sep:Today',
    'chat:c4[task 14]',
  ]);
  // A rebuild keeps the keyboard on a task's row under the project.
  rowByTitle(win, 'task 11').focus();
  sendProjectPage(win, posted, 'p1', PAPER_ROWS.concat(groupedSession('c2', 9, T - 60)));
  assert.strictEqual(shape(win)[3], 'project:p1{chat:c1[task 11,task 10] chat:c2[task 12,task 9]}');
  assert.strictEqual(win.document.activeElement, rowByTitle(win, 'task 11'));
  sendProjectPage(win, posted, 'p1', PAPER_ROWS);

  // Folding keeps the rows; unfolding asks again, and a page that
  // changed nothing leaves the list as it is.
  const chat = win.document.querySelector('.history-chat-group[data-chat-id="c1"]');
  press(win, 'p1', 'history-project-collapse');
  assert.strictEqual(shape(win)[3], 'project:p1');
  from = posted.length;
  press(win, 'p1', 'history-project-collapse');
  assert.strictEqual(projectRequests(posted, from).length, 1);
  sendProjectPage(win, posted, 'p1', PAPER_ROWS);
  assert.strictEqual(
    win.document.querySelector('.history-chat-group[data-chat-id="c1"]'),
    chat,
  );
  // A refresh of the list asks for the unfolded project's rows again,
  // not for those of a folded one; a project without a chat says so.
  from = posted.length;
  send(win, {type: 'tasks_updated'});
  assert.deepStrictEqual(projectRequests(posted, from).map(r => r.project), ['p1']);
  press(win, 'p2', 'history-project-collapse');
  // A reply that lists nothing.
  send(win, {
    type: 'history',
    project: 'p2',
    limit: PAGE,
    generation: generation(posted),
  });
  assert.strictEqual(shape(win)[1], 'project:p2{No chats yet}');
  // The rows of a project folded since they were asked for are dropped.
  press(win, 'p2', 'history-project-collapse');
  sendProjectPage(win, posted, 'p2', [STREAM[2]]);
  press(win, 'p2', 'history-project-collapse');
  assert.strictEqual(shape(win)[1], 'project:p2{No chats yet}');

  // "…" opens the Projects page at the project's settings.
  press(win, 'p1', 'history-project-more');
  const settings = win.document.querySelector(
    '.projects-tab-view .project-panel[data-project-id="p1"]',
  );
  assert.strictEqual(win.document.activeElement, settings.querySelector('.project-name'));
  assert.ok(!byId(win, 'sidebar').classList.contains('open'), 'in front of the drawer');

  // A click on the header: a new chat, in the project.
  click(win, panelOf(win, 'p1').querySelector('.history-project-header'));
  assert.strictEqual(byId(win, 'chat-project').value, 'p1');
  assert.ok(!byId(win, 'chat-project-wrap').hidden);
  // The chat on screen is that new chat: its project is the one marked
  // in the list.  A first task that goes to a registered machine takes
  // the chat into no project of this daemon.
  const marked = () =>
    ['p1', 'p2'].filter(id =>
      panelOf(win, id).classList.contains('history-active-project'),
    );
  assert.deepStrictEqual(marked(), ['p1']);
  send(win, {type: 'machines', machines: MACHINES});
  pickMachine(win, 'gpu-box');
  assert.deepStrictEqual(marked(), []);
  pickMachine(win, '');
  assert.deepStrictEqual(marked(), ['p1']);
  // The mark goes with the chat on screen, here another new chat that
  // is in no project, and follows a pick in the header.
  const inPaper = win._testApi.getActiveTabId();
  win._testApi.createNewTab();
  assert.deepStrictEqual(marked(), []);
  win._testApi.switchToTab(inPaper);
  assert.deepStrictEqual(marked(), ['p1']);
  selectFilter(win, 'chat-project', 'p2');
  assert.deepStrictEqual(marked(), ['p2']);
  selectFilter(win, 'chat-project', '');
  assert.deepStrictEqual(marked(), []);
});

test('a panel whose page was a full one ends with "Show more chats"; a refresh asks for every row it holds', () => {
  const first = pageOf('a', 1000);
  const second = pageOf('b', 800);
  const inPaper = {c1: 'p1', c2: 'p1'};
  for (const row of first.concat(second)) inPaper[row.id] = 'p1';
  const {win, posted} = openList([PAPER], inPaper, []);
  // A project is listed while there is no task at all.
  assert.deepStrictEqual(shape(win), ['sep:Projects', 'project:p1']);
  // Group by: Chat has nothing to list then; Project brings the panel back.
  pressFilterOption(win, 'group', 'Chat');
  assert.strictEqual(
    byId(win, 'history-list').textContent.trim(),
    'No conversations yet',
  );
  pressFilterOption(win, 'group', 'Project');
  assert.deepStrictEqual(shape(win), ['sep:Projects', 'project:p1']);
  press(win, 'p1', 'history-project-collapse');
  sendProjectPage(win, posted, 'p1', first);
  const body = panelOf(win, 'p1').querySelector('.history-project-body');
  assert.strictEqual(moreRows(win, 'p1').textContent, 'Show more chats');
  assert.strictEqual(body.lastElementChild, moreRows(win, 'p1'), 'after the chats');
  assert.strictEqual(rowCount(win, 'p1'), PAGE);

  // It asks for the page after the rows held.  A full page keeps the
  // button, and the keyboard on it.
  let from = posted.length;
  moreRows(win, 'p1').focus();
  click(win, moreRows(win, 'p1'));
  assert.deepStrictEqual(projectRequests(posted, from), [
    pageRequest(posted, 'p1', '', PAGE, PAGE),
  ]);
  sendProjectPage(win, posted, 'p1', second, PAGE);
  assert.strictEqual(rowCount(win, 'p1'), 2 * PAGE);
  assert.strictEqual(win.document.activeElement, moreRows(win, 'p1'));
  assert.strictEqual(
    panelOf(win, 'p1').querySelector('.history-project-body').lastElementChild,
    moreRows(win, 'p1'),
    'still after the chats',
  );
  // A page that does not continue the rows held (it was asked for
  // twice, or before they were replaced) is dropped.
  sendProjectPage(win, posted, 'p1', second, PAGE);
  sendProjectPage(win, posted, 'p1', second, 3 * PAGE);
  assert.strictEqual(rowCount(win, 'p1'), 2 * PAGE);

  // A refresh asks for every row held, in one page: the same rows
  // leave the list as it is, with what "Show more chats" added.
  from = posted.length;
  send(win, {type: 'tasks_updated'});
  assert.deepStrictEqual(projectRequests(posted, from), [
    pageRequest(posted, 'p1', '', 0, 2 * PAGE),
  ]);
  const chat = win.document.querySelector('.history-chat-group[data-chat-id="b-0"]');
  assert.ok(chat);
  sendProjectPage(win, posted, 'p1', first.concat(second), 0, 2 * PAGE);
  assert.strictEqual(
    win.document.querySelector('.history-chat-group[data-chat-id="b-0"]'),
    chat,
  );
  assert.ok(moreRows(win, 'p1'));

  // A page with fewer rows than were asked for is the last: the button
  // goes, and the keyboard focus it held with it.
  moreRows(win, 'p1').focus();
  click(win, moreRows(win, 'p1'));
  sendProjectPage(win, posted, 'p1', rowsOf('c2', 600, 1), 2 * PAGE);
  assert.strictEqual(rowCount(win, 'p1'), 2 * PAGE + 1);
  assert.strictEqual(moreRows(win, 'p1'), null);
  assert.ok(!byId(win, 'history-list').contains(win.document.activeElement));

  // A refresh that brings other rows replaces the rows held.
  send(win, {type: 'tasks_updated'});
  sendProjectPage(win, posted, 'p1', rowsOf('c1', 1000, 2), 0, 2 * PAGE + 1);
  assert.deepStrictEqual(shape(win), [
    'sep:Projects',
    'project:p1{chat:c1[task 1000,task 999]}',
  ]);

  // The WORKDIR filter narrows a project's rows as it does the list's.
  sendStream(win, posted, [], {workDirs: ['/work/ws', '/work/other']});
  from = posted.length;
  const unfiltered = generation(posted);
  selectFilter(win, 'hf-workdir', '/work/other');
  assert.deepStrictEqual(projectRequests(posted, from), [
    pageRequest(posted, 'p1', '/work/other', 0, PAGE),
  ]);
  sendProjectPage(win, posted, 'p1', []);
  assert.deepStrictEqual(shape(win), ['sep:Projects', 'project:p1{No matching chats}']);
  // The answer to a request of the view before the filter, coming
  // late, is not this view's: the panel keeps this view's rows.
  send(win, {
    type: 'history',
    project: 'p1',
    offset: 0,
    limit: PAGE,
    generation: unfiltered,
    sessions: rowsOf('c1', 1000, 2),
  });
  assert.strictEqual(rowCount(win, 'p1'), 0);

  // The last project gone, with no task at all: nothing to list.
  send(win, {type: 'projects', projects: [], chatProjects: {}});
  assert.strictEqual(
    byId(win, 'history-list').textContent.trim(),
    'No conversations yet',
  );
});

test('an older chat that joins a project whose rows are all listed is listed in its panel', () => {
  // A project of one full page of rows, and of one row more: both
  // panels list every row, so neither ends with "Show more chats".
  for (const beyond of [0, 1]) {
    const rows = pageOf('a', 1000).concat(rowsOf('c1', 700, beyond));
    const inPaper = {c1: 'p1'};
    for (const row of rows) inPaper[row.id] = 'p1';
    const {win, posted} = openList([PAPER], inPaper, []);
    press(win, 'p1', 'history-project-collapse');
    sendProjectPage(win, posted, 'p1', rows.slice(0, PAGE));
    click(win, moreRows(win, 'p1'));
    sendProjectPage(win, posted, 'p1', rows.slice(PAGE), PAGE);
    assert.strictEqual(moreRows(win, 'p1'), null);
    assert.strictEqual(rowCount(win, 'p1'), rows.length);

    // A refresh asks for one row more than the panel holds: the rows
    // it holds, coming back alone, leave the panel as it is.
    let from = posted.length;
    send(win, {type: 'tasks_updated'});
    assert.deepStrictEqual(projectRequests(posted, from), [
      pageRequest(posted, 'p1', '', 0, rows.length + 1),
    ]);
    sendProjectPage(win, posted, 'p1', rows.slice(), 0, rows.length + 1);
    assert.strictEqual(moreRows(win, 'p1'), null);
    assert.strictEqual(rowCount(win, 'p1'), rows.length);

    // An older chat joins the project (from the chat header's project
    // drop-down of another window, say): its row is the one more that
    // is asked for, and a page as long as was asked for may have
    // another after it.
    from = posted.length;
    send(win, {
      type: 'projects',
      projects: [PAPER],
      chatProjects: Object.assign({old: 'p1'}, inPaper),
    });
    assert.deepStrictEqual(projectRequests(posted, from), [
      pageRequest(posted, 'p1', '', 0, rows.length + 1),
    ]);
    const older = groupedSession('old', 1, T - 9000);
    sendProjectPage(win, posted, 'p1', rows.concat(older), 0, rows.length + 1);
    assert.ok(
      panelOf(win, 'p1').contains(rowByTitle(win, 'task 1')),
      'BUG: the older chat of the project is not listed',
    );
    assert.strictEqual(rowCount(win, 'p1'), rows.length + 1);
    assert.ok(moreRows(win, 'p1'));
  }
});

test('the projects changing while the first page of the list is on its way leaves the next page to be asked for', () => {
  // `held`: a press on the list is held when the projects change, so
  // the list is built again when the press ends.
  for (const held of [false, true]) {
    const timers = [];
    const {win, posted} = openList([PAPER], {c1: 'p1'}, [], {
      beforeScripts: w => captureTimers(w, timers),
    });
    const list = byId(win, 'history-list');
    // A refresh: its page is on its way when another project appears,
    // and the list is built again from the rows it holds (none).
    send(win, {type: 'tasks_updated'});
    if (held) list.dispatchEvent(new win.MouseEvent('mousedown', {bubbles: true}));
    send(win, {type: 'projects', projects: [PAPER, SITE], chatProjects: {c1: 'p1'}});
    if (held) {
      assert.strictEqual(panelOf(win, 'p2'), null, 'not under a held press');
      const released = timers.length;
      win.dispatchEvent(new win.MouseEvent('mouseup', {bubbles: true}));
      fireCapturedTimers(win, timers, released);
    }
    assert.ok(panelOf(win, 'p2'));
    // The page: a full one, so another may follow it.
    sendStream(win, posted, rowsOf('c4', 100, 50));
    const from = posted.length;
    list.dispatchEvent(new win.Event('scroll'));
    assert.deepStrictEqual(
      streamRequests(posted.slice(from)).map(m => m.offset),
      [50],
      (held ? 'held: ' : '') + 'BUG: the end of the list asks for no more rows',
    );
    if (!held) continue;
    // That next page lands while another rebuild waits for a press
    // that never ends: the list built when the press is given up (5 s)
    // holds its rows too.
    list.dispatchEvent(new win.MouseEvent('mousedown', {bubbles: true}));
    send(win, {type: 'projects', projects: [PAPER], chatProjects: {c1: 'p1'}});
    sendStream(win, posted, rowsOf('c5', 50, 50), {offset: 50});
    // 5 s on the page's clock, which openList froze at T (not this
    // process's clock: before noon it is behind T).
    win.Date.now = () => T * 1000 + 5000;
    timers.find(t => t.ms === 5000 && !t.cleared && !t.fired).cb();
    assert.strictEqual(panelOf(win, 'p2'), null);
    assert.strictEqual(
      list.querySelectorAll('.sidebar-item').length,
      100,
      'BUG: the page that landed under the press is gone',
    );
  }
});

test('Group by offers Project, Chat and None; a search and the list of another machine show no panel', () => {
  const {win, posted} = openList([PAPER, SITE], CHAT_PROJECTS);
  assert.deepStrictEqual(tickedFilterOptions(win, 'group'), ['Project']);
  press(win, 'p1', 'history-project-collapse');
  sendProjectPage(win, posted, 'p1', PAPER_ROWS);
  const byProject = shape(win);
  assert.strictEqual(byProject[3], 'project:p1{chat:c1[task 11,task 10] chat:c2[task 12]}');

  // Chat: every chat on its own under its day, no panel, and no
  // project's rows asked for.
  let from = posted.length;
  pressFilterOption(win, 'group', 'Chat');
  assert.deepStrictEqual(shape(win), [
    'sep:Today',
    'chat:c4[task 14]',
    'chat:c1[task 11,task 10]',
    'chat:c3[task 13]',
    'chat:c2[task 12]',
  ]);
  assert.strictEqual(win.localStorage.getItem('kissSorcar.historyByProject'), '0');
  send(win, {type: 'tasks_updated'});
  assert.strictEqual(projectRequests(posted, from).length, 0);
  sendStream(win, posted, STREAM);

  // None: the flat list.
  pressFilterOption(win, 'group', 'None');
  assert.deepStrictEqual(shape(win), [
    'sep:Today',
    'row:task 14',
    'row:task 11',
    'row:task 13',
    'row:task 12',
    'row:task 10',
  ]);

  // Project again: the unfolded project's rows are asked for anew and
  // the list is rebuilt from the rows it holds.
  from = posted.length;
  pressFilterOption(win, 'group', 'Project');
  assert.strictEqual(win.localStorage.getItem('kissSorcar.historyByProject'), '1');
  assert.deepStrictEqual(projectRequests(posted, from).map(r => r.project), ['p1']);
  assert.strictEqual(streamRequests(posted.slice(from)).length, 0, 'no refetch of the list');
  assert.deepStrictEqual(shape(win), byProject);

  // A search lists its matches by chat, and its end brings the panels
  // back.
  const search = byId(win, 'history-search');
  typeInto(win, search, 'task 1');
  sendStream(win, posted, STREAM);
  assert.ok(!shape(win).some(s => s.startsWith('project:')), 'no panel in a search');
  assert.ok(shape(win).some(s => s.startsWith('chat:c1[')));
  typeInto(win, search, '');
  sendStream(win, posted, STREAM);
  assert.deepStrictEqual(shape(win), byProject);

  // The projects are this machine's: the list of a registered
  // machine's tasks shows none, the list of this machine's shows them.
  const machines = {machines: ['gpu-box']};
  sendStream(win, posted, STREAM, machines);
  from = posted.length;
  selectFilter(win, 'hf-machine', 'gpu-box');
  sendStream(
    win,
    posted,
    [groupedSession('c1', 90, T - 5, {machine: 'gpu-box'})],
    machines,
  );
  assert.deepStrictEqual(shape(win), ['sep:Today', 'chat:c1[task 90]']);
  assert.strictEqual(projectRequests(posted, from).length, 0);
  selectFilter(win, 'hf-machine', '__local__');
  sendStream(win, posted, STREAM, machines);
  assert.deepStrictEqual(shape(win), byProject);

  // The stored choice is the one a page starts with: Chat, or the flat
  // list.  A page whose storage cannot be read starts with Project.
  const again = openList([PAPER], CHAT_PROJECTS, STREAM, {
    beforeScripts: w => w.localStorage.setItem('kissSorcar.historyByProject', '0'),
  });
  assert.deepStrictEqual(tickedFilterOptions(again.win, 'group'), ['Chat']);
  assert.ok(!shape(again.win).some(s => s.startsWith('project:')));
  const flat = openList([PAPER], CHAT_PROJECTS, STREAM, {
    beforeScripts: w => w.localStorage.setItem('kissSorcar.historyLegacyView', '1'),
  });
  assert.deepStrictEqual(tickedFilterOptions(flat.win, 'group'), ['None']);
  const opaque = openList([PAPER], CHAT_PROJECTS, STREAM, {url: null});
  assert.deepStrictEqual(tickedFilterOptions(opaque.win, 'group'), ['Project']);
  assert.ok(shape(opaque.win).includes('project:p1'));
});

test('a project with a running task is listed under Running; the project of the chat on screen is marked, and the list follows the chats in a project', () => {
  const scrolled = [];
  const running = STREAM.map(s =>
    s.task_id === 12 ? Object.assign({}, s, {is_running: true}) : s,
  );
  const {win, posted} = openList([PAPER, SITE], CHAT_PROJECTS, running, {
    beforeScripts: w => {
      w.Element.prototype.scrollIntoView = function () {
        scrolled.push(this);
      };
    },
  });
  // Paper has a task that runs: under Running, a spinner for its icon.
  assert.deepStrictEqual(shape(win), [
    'sep:Running',
    'project:p1',
    'sep:Pinned',
    'project:p2',
    'sep:Today',
    'chat:c4[task 14]',
  ]);
  const icon = panelOf(win, 'p1').querySelector('.history-project-icon');
  assert.ok(icon.classList.contains('status-spinner'));
  assert.strictEqual(icon.querySelector('svg'), null);
  assert.ok(panelOf(win, 'p2').querySelector('.history-project-icon svg'));

  // The chat on screen is in Paper: its panel is marked, folded or
  // not, and while it is folded its header stands for the task's row.
  showTask(win, 'c1', 10);
  assert.ok(panelOf(win, 'p1').classList.contains('history-active-project'));
  assert.ok(!panelOf(win, 'p2').classList.contains('history-active-project'));
  press(win, 'p1', 'history-project-collapse');
  sendProjectPage(win, posted, 'p1', [running[1], running[3], running[4]]);
  press(win, 'p1', 'history-project-collapse');
  scrolled.length = 0;
  showTask(win, 'c1', 11);
  // The header in the list, by identity: an element built like it is
  // deep-equal to it.
  assert.strictEqual(scrolled.length, 1);
  assert.strictEqual(
    scrolled[0],
    panelOf(win, 'p1').querySelector('.history-project-header'),
  );
  press(win, 'p1', 'history-project-collapse');

  // A chat that enters a project is listed under it once its rows
  // came; one that left is listed under its day again.
  let from = posted.length;
  send(win, {
    type: 'projects',
    projects: [PAPER, SITE],
    chatProjects: {c1: 'p1', c3: 'p2', c4: 'p1'},
  });
  assert.deepStrictEqual(projectRequests(posted, from).map(r => r.project), ['p1']);
  assert.deepStrictEqual(shape(win), [
    'sep:Running',
    'chat:c2[task 12]',
    'sep:Pinned',
    'project:p2',
    'sep:Projects',
    'project:p1{chat:c1[task 11,task 10]}',
  ]);
  sendProjectPage(win, posted, 'p1', [running[0], running[1], running[4]]);
  assert.strictEqual(shape(win)[5], 'project:p1{chat:c4[task 14] chat:c1[task 11,task 10]}');

  // A project's page holds the tasks that ran on this machine: a row
  // of one of its chats that another machine's history brought is
  // listed from the list's own rows, with its chat.
  sendStream(win, posted, [
    groupedSession('c1', 91, T - 5, {machine: 'gpu-box'}),
    running[2],
  ]);
  assert.deepStrictEqual(shape(win), [
    'sep:Pinned',
    'project:p2',
    'sep:Projects',
    'project:p1{chat:c1[task 91,task 11,task 10] chat:c4[task 14]}',
  ]);

  // A project that is deleted takes its panel and its fold with it:
  // its chats are listed under their days.
  send(win, {type: 'projects', projects: [SITE], chatProjects: {c3: 'p2'}});
  sendStream(win, posted, STREAM);
  assert.deepStrictEqual(shape(win), [
    'sep:Pinned',
    'project:p2',
    'sep:Today',
    'chat:c4[task 14]',
    'chat:c1[task 11,task 10]',
    'chat:c2[task 12]',
  ]);
  send(win, {type: 'projects', projects: [PAPER, SITE], chatProjects: CHAT_PROJECTS});
  assert.strictEqual(shape(win)[3], 'project:p1', 'a project of the same id starts folded');

  // A later page of the list: a row of a project's chat is counted in
  // the project's panel, any other row is listed.
  sendStream(
    win,
    posted,
    [groupedSession('c2', 9, T - 100), groupedSession('c5', 8, T - 110)],
    {offset: STREAM.length},
  );
  assert.deepStrictEqual(shape(win).slice(-3), [
    'sep:Today',
    'chat:c4[task 14]',
    'chat:c5[task 8]',
  ]);

  // A chat the daemon bound to a registered machine is in no project
  // of this one, whatever its id.
  assert.ok(panelOf(win, 'p1').classList.contains('history-active-project'));
  send(win, {
    type: 'tabMachine',
    tabId: win._testApi.getActiveTabId(),
    machine: 'gpu-box',
  });
  showTask(win, 'c1', 10);
  assert.ok(!panelOf(win, 'p1').classList.contains('history-active-project'));
});

/** The folders the page asked the daemon to check before adopting them, in order. */
function workDirChecks(posted) {
  return ofType(posted, 'listDir')
    .filter(m => String(m.token).startsWith('workdir:'))
    .map(m => m.path);
}

test('a chat opened from the list comes on screen in its default folder: its project\u2019s, else the Settings sheet\u2019s', () => {
  const paper = project('p1', 'Paper', {settings: {default_work_dir: '/work/paper'}});
  const elsewhere = groupedSession('c9', 15, T - 60, {machine: 'gpu-box'});
  const {win, posted} = openList([paper, SITE], CHAT_PROJECTS, STREAM.concat(elsewhere));
  // The timers that give an unanswered check up (10 s), fired by hand.
  const timers = recordTimers(win, 10000);
  press(win, 'p1', 'history-project-collapse');
  sendProjectPage(win, posted, 'p1', PAPER_ROWS);

  // A task's row of a chat in Paper: the folder is checked, then
  // adopted, as a pick of it in the chat header is.
  click(win, rowByTitle(win, 'task 11'));
  assert.deepStrictEqual(workDirChecks(posted), ['/work/paper']);
  // A chat in no project, while the Settings sheet names no default,
  // has no folder of its own: nothing is asked, and the check in
  // flight is still the one whose answer counts.
  click(win, chatHeader(win, 'c4'));
  answerWorkDirCheck(win, posted, '/work/paper');
  assert.strictEqual(currentDir(win), '/work/paper');
  // The chat's panel of Paper: the folder is the working directory, so
  // nothing is asked.
  click(win, chatHeader(win, 'c1'));
  assert.deepStrictEqual(workDirChecks(posted), ['/work/paper']);

  // The Settings sheet's default is that of a chat in no project and of
  // one whose project (Site) has no folder of its own.
  send(win, {
    type: 'configData',
    config: {work_dir: '/work/paper', default_work_dir: '/work/default'},
    apiKeys: {},
  });
  click(win, chatHeader(win, 'c4'));
  assert.deepStrictEqual(workDirChecks(posted).slice(1), ['/work/default']);
  press(win, 'p2', 'history-project-collapse');
  sendProjectPage(win, posted, 'p2', [STREAM[2]]);
  click(win, rowByTitle(win, 'task 13'));
  assert.deepStrictEqual(workDirChecks(posted).slice(2), ['/work/default']);
  // That check is still in flight when a chat of Paper is opened, whose
  // folder is the working directory already: nothing is asked, and the
  // late answer for the chat left behind does not move the folder.
  click(win, chatHeader(win, 'c1'));
  assert.strictEqual(workDirChecks(posted).length, 3);
  answerWorkDirCheck(win, posted, '/work/default');
  assert.strictEqual(currentDir(win), '/work/paper');
  assert.deepStrictEqual(
    ofType(posted, 'setWorkDir').map(m => m.workDir),
    ['/work/paper'],
  );
  // A chat of another machine: the defaults are this machine's folders.
  click(win, rowByTitle(win, 'task 15'));
  assert.strictEqual(workDirChecks(posted).length, 3);
  // Every check ended, the one a chat left behind too: none is given
  // up later for want of an answer.
  fireTimers(timers);
  assert.deepStrictEqual(errorNotifications(win), []);
});

test("a project's rows read what the daemon last said once the star of a click is over", () => {
  // The rows of an unfolded project come from the project's own page,
  // which the pages of the list do not replace: the star of a click
  // whose bound passed stayed on them however the list was built again
  // (audit 2026-11-21, W13/F2).
  const timers = [];
  let now = T * 1000;
  const row = favorite => [groupedSession('c1', 11, T - 20, {is_favorite: favorite})];
  const {win, posted} = openList([PAPER], {c1: 'p1'}, row(false), {
    beforeScripts: w => captureTimers(w, timers),
  });
  win.Date.now = () => now;
  const pending = () => win._testApi.pendingHistoryFlagCount();
  const star = () => panelOf(win, 'p1').querySelector('[data-flag="is_favorite"]');
  const boundPasses = () => {
    now += 10100;
    timers.find(t => t.ms === 10100 && !t.cleared && !t.fired).cb();
    assert.strictEqual(pending(), 0, 'the click is over at its bound');
  };
  // The list built again from the rows it holds, the project's too.
  const rebuilt = () => {
    for (const id of ['hf-group-chat', 'hf-group-project']) {
      byId(win, id).checked = true;
      byId(win, id).dispatchEvent(new win.Event('change', {bubbles: true}));
    }
    return star().getAttribute('aria-pressed');
  };
  press(win, 'p1', 'history-project-collapse');
  sendProjectPage(win, posted, 'p1', row(false));

  // No word of the daemon's by the bound: its last one is back.
  star().click();
  assert.strictEqual(star().getAttribute('aria-pressed'), 'true');
  boundPasses();
  assert.strictEqual(rebuilt(), 'false', 'BUG: the star of a click that is over was built again');

  // The daemon's word at the bound is the click's, in a page that reads
  // like the rows held: it stays.
  star().click();
  boundPasses();
  sendProjectPage(win, posted, 'p1', row(true));
  assert.strictEqual(rebuilt(), 'true', 'BUG: the star the daemon confirmed was taken back');

  // A page of the list agrees within the bound: so does the row of the
  // project's page, whose own value it replaces.
  star().click();
  sendStream(win, posted, row(false));
  assert.strictEqual(pending(), 0, 'the list agreed');
  assert.strictEqual(rebuilt(), 'false', 'BUG: the star the daemon agreed to drop was built again');
});

test("a flag the daemon agreed to is on the chat's other copy too, which Group by shows next", () => {
  // The list's own page and the page of an unfolded project each hold
  // the rows of a chat in the project, and a click marks the copy on
  // screen only: the daemon's agreement left the other copy, which no
  // click had marked, with the flag it had before (audit 2026-11-21,
  // W13/F2).
  const flags = ['is_favorite', 'is_pinned', 'chat_is_favorite', 'chat_is_pinned'];
  const row = value => [
    groupedSession('c1', 11, T - 20, Object.fromEntries(flags.map(f => [f, value]))),
  ];
  const {win, posted} = openList([PAPER], {c1: 'p1'}, row(false));
  const button = flag => byId(win, 'history-list').querySelector('[data-flag="' + flag + '"]');
  // Every flag is clicked and `confirm` brings the daemon's agreement;
  // then Group by `id` is chosen: what the copy it shows reads.
  const clickedThenGroupedBy = (confirm, id) => {
    flags.forEach(flag => button(flag).click());
    confirm();
    assert.strictEqual(win._testApi.pendingHistoryFlagCount(), 0, 'the daemon agreed');
    byId(win, id).checked = true;
    byId(win, id).dispatchEvent(new win.Event('change', {bubbles: true}));
    return flags.map(flag => button(flag).getAttribute('aria-pressed'));
  };
  press(win, 'p1', 'history-project-collapse');
  sendProjectPage(win, posted, 'p1', row(false));

  // Set on the project's copy, confirmed by the project's page: the
  // list's copy is shown, and no page of the list is asked for.
  assert.deepStrictEqual(
    clickedThenGroupedBy(() => sendProjectPage(win, posted, 'p1', row(true)), 'hf-group-chat'),
    flags.map(() => 'true'),
    "BUG: the list's copy lost the flags the daemon confirmed",
  );
  // Cleared on the list's copy, confirmed by the list's page: the
  // project's copy is shown before the page asked for it comes.
  assert.deepStrictEqual(
    clickedThenGroupedBy(() => sendStream(win, posted, row(false)), 'hf-group-project'),
    flags.map(() => 'false'),
    "BUG: the project's copy kept the flags the daemon agreed to drop",
  );
});

// The last test is asynchronous (a parked page lands a timer turn
// after the release): the totals wait for it.
(async () => {
  await test('a page of the list parked for a held press or an open rename box outlives the project page that lands meanwhile', async () => {
    const inPaper = groupedSession('c1', 11, T - 20);
    // A full page of the list's own rows: another may follow it.
    const loose = rowsOf('c4', 100, 50);
    const added = groupedSession('c5', 15, T - 5);
    // `late`: the project's page lands after the release, before the
    // parked page does.
    for (const [hold, late] of [
      ['press', false],
      ['rename', false],
      ['press', true],
    ]) {
      const {win, posted} = openList([PAPER], {c1: 'p1'}, loose.concat(inPaper));
      press(win, 'p1', 'history-project-collapse');
      sendProjectPage(win, posted, 'p1', [inPaper]);
      let release;
      if (hold === 'press') {
        rowByTitle(win, 'task 100').dispatchEvent(
          new win.MouseEvent('mousedown', {bubbles: true}),
        );
        release = () =>
          win.dispatchEvent(new win.MouseEvent('mouseup', {bubbles: true}));
      } else {
        click(win, chatHeader(win, 'c4').querySelector('.history-chat-rename'));
        const box = chatHeader(win, 'c4').querySelector('.history-chat-title-input');
        release = () => key(win, box, 'Escape');
      }
      // A refresh brings a chat that is in no project: its page waits
      // for the release.  The page of the project's rows, which the
      // refresh asked for too, follows with a changed row.
      send(win, {type: 'tasks_updated'});
      sendStream(win, posted, [added].concat(loose, inPaper));
      assert.strictEqual(rowByTitle(win, 'task 15'), null, 'parked');
      if (late) release();
      sendProjectPage(win, posted, 'p1', [
        Object.assign({}, inPaper, {title: 'task 11 again'}),
      ]);
      if (!late) {
        assert.strictEqual(rowByTitle(win, 'task 15'), null, 'still parked');
        release();
      }
      await sleep(0);
      const what = hold + (late ? ', released first' : '');
      assert.ok(
        rowByTitle(win, 'task 15'),
        what + ': BUG: the newer page of the list is gone',
      );
      assert.ok(
        panelOf(win, 'p1').contains(rowByTitle(win, 'task 11 again')),
        what + ": the project's page is listed",
      );
      // No page is on its way any more: the end of the list asks for
      // the next one.
      const from = posted.length;
      byId(win, 'history-list').dispatchEvent(new win.Event('scroll'));
      assert.deepStrictEqual(
        streamRequests(posted.slice(from)).map(m => m.offset),
        [52],
        what + ': BUG: the list is left waiting for a page that is not coming',
      );
    }
  });
  report('historyProjects');
})();
