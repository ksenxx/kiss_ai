// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end (jsdom) tests for the history panel's two views
// (media/main.js, media/chat.html):
//
// * the collapsible chat-panel headers carry no tooltip (neither
//   data-tooltip nor title) and show one line of text — the first line
//   of the chat's first task,
// * the button right of the search box switches to the legacy flat
//   list: every task newest first, no chat panels, no day separators,
//   each row stamped with its chat's --task-color (rows of one chat
//   share a color, rows of different chats differ),
// * the switch is remembered in localStorage and honoured on load,
// * switching rebuilds the loaded rows without a refetch — through the
//   identical-refresh fast path, which must not keep the old view — and
//   a later identical refresh keeps the flat rows, and
// * the status filters and the keyboard focus restore work in the flat
//   list too.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');
const VIEW_KEY = 'kissSorcar.historyLegacyView';

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

function makeWebview(beforeMain) {
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
  html = html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');
  html = html.replace('<body', '<body class="remote-chat"');
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
  win.matchMedia = function (query) {
    return {
      matches: query === '(min-width: 900px)',
      media: query,
      addEventListener: () => {},
      removeEventListener: () => {},
      addListener: () => {},
      removeListener: () => {},
    };
  };
  if (beforeMain) beforeMain(win);
  win.eval(fs.readFileSync(path.join(MEDIA, 'marked.min.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'contentContextMenu.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'treeContextMenu.js'), 'utf8'));
  win.eval(
    fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8') +
      '\n//# sourceURL=history-legacy-main.js',
  );
  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function byId(win, id) {
  return win.document.getElementById(id);
}

function all(win, sel) {
  return Array.from(win.document.querySelectorAll(sel));
}

const DAY = 86400;
const todayNoon = (() => {
  const d = new Date();
  d.setHours(12, 0, 0, 0);
  return Math.floor(d.getTime() / 1000);
})();

function session(chat, task, ts, extra) {
  return Object.assign(
    {
      id: chat,
      task_id: task,
      title: 'task ' + task,
      preview: 'task ' + task,
      has_events: true,
      timestamp: ts,
      tokens: 1,
      cost: 0.1,
      steps: 1,
    },
    extra || {},
  );
}

/** Deliver a history page stamped with the generation the panel last asked for. */
function sendHistory(win, posted, offset, sessions) {
  const asked = posted.filter(m => m.type === 'getHistory');
  send(win, {
    type: 'history',
    offset,
    generation: asked.length ? asked[asked.length - 1].generation : 0,
    sessions,
  });
}

/** The visible top-level children of #history-list, described. */
function listShape(win) {
  return Array.from(byId(win, 'history-list').children)
    .filter(el => el.style.display !== 'none')
    .map(el =>
      el.classList.contains('history-day-sep')
        ? 'sep:' + el.textContent
        : el.classList.contains('history-chat-group')
          ? 'chat:' +
            el.dataset.chatId +
            '[' +
            Array.from(el.querySelectorAll('.sidebar-item'))
              .map(r => r.querySelector('.sidebar-item-text').textContent)
              .join(',') +
            ']'
          : el.classList.contains('sidebar-item')
            ? 'row:' + el.querySelector('.sidebar-item-text').textContent
            : el.className,
    );
}

function page() {
  return [
    session('A', 'a2', todayNoon),
    session('B', 'b2', todayNoon - 600),
    session('A', 'a1', todayNoon - 1200),
    session('C', 'c1', todayNoon - DAY),
    session('B', 'b1', todayNoon - DAY - 60),
  ];
}

const GROUPED = [
  'sep:Today',
  'chat:A[task a2,task a1]',
  'chat:B[task b2,task b1]',
  'sep:Yesterday',
  'chat:C[task c1]',
];
const FLAT = [
  'row:task a2',
  'row:task b2',
  'row:task a1',
  'row:task c1',
  'row:task b1',
];

function taskColor(row) {
  return row.style.getPropertyValue('--task-color');
}

test('chat headers carry no tooltip and show the first line of the first task', () => {
  const {win, posted} = makeWebview();
  sendHistory(win, posted, 0, [
    Object.assign(session('A', 'a2', todayNoon), {
      chat_first_task: '  Fix the login bug\n\nSteps:\n1. reproduce\n2. fix',
    }),
    Object.assign(session('B', 'b1', todayNoon - 600), {
      chat_first_task: '\n\n   \n',
    }),
    session('C', 'c1', todayNoon - 1200),
  ]);
  const headers = all(win, '#history-list .history-chat-header');
  assert.strictEqual(headers.length, 3);
  headers.forEach(h => {
    assert.ok(!h.hasAttribute('data-tooltip'), 'no data-tooltip on a header');
    assert.ok(!h.hasAttribute('title'), 'no title attribute on a header');
    assert.strictEqual(
      h.querySelector('.history-chat-title').textContent.indexOf('\n'),
      -1,
      'the title is one line',
    );
  });
  const titles = headers.map(
    h => h.querySelector('.history-chat-title').textContent,
  );
  assert.deepStrictEqual(titles, ['Fix the login bug', 'Untitled', 'task c1']);
  // Toggling the panel repaints the ARIA state only — still no tooltip.
  headers[0].click();
  assert.strictEqual(headers[0].getAttribute('aria-expanded'), 'true');
  assert.ok(!headers[0].hasAttribute('data-tooltip'));
  assert.ok(!headers[0].hasAttribute('title'));
  headers[0].click();
  assert.strictEqual(headers[0].getAttribute('aria-expanded'), 'false');
  assert.ok(!headers[0].hasAttribute('data-tooltip'));
  win.close();
});

test('the view toggle sits right of the search box and swaps grouped <-> flat without a refetch', () => {
  const {win, posted} = makeWebview();
  const toggle = byId(win, 'history-view-toggle');
  assert.ok(toggle, '#history-view-toggle exists');
  const searchWrap = byId(win, 'history-search').closest('.search-wrap');
  assert.strictEqual(
    searchWrap.nextElementSibling,
    toggle,
    'the toggle is the search box\u2019s next sibling (to its right)',
  );
  assert.strictEqual(toggle.getAttribute('aria-pressed'), 'false');
  assert.ok(!byId(win, 'history-list').classList.contains('legacy-view'));

  sendHistory(win, posted, 0, page());
  assert.deepStrictEqual(listShape(win), GROUPED);
  const fetchesBefore = posted.filter(m => m.type === 'getHistory').length;

  toggle.click();
  assert.strictEqual(toggle.getAttribute('aria-pressed'), 'true');
  assert.ok(toggle.classList.contains('active'));
  assert.ok(byId(win, 'history-list').classList.contains('legacy-view'));
  assert.strictEqual(win.localStorage.getItem(VIEW_KEY), '1');
  assert.strictEqual(
    posted.filter(m => m.type === 'getHistory').length,
    fetchesBefore,
    'switching views rebuilds locally, without a refetch',
  );
  assert.deepStrictEqual(listShape(win), FLAT, 'flat, newest first');
  assert.strictEqual(all(win, '#history-list .history-chat-group').length, 0);
  assert.strictEqual(all(win, '#history-list .history-day-sep').length, 0);

  // Per-chat colors: one hue per chat, distinct across chats.
  const rows = all(win, '#history-list > .sidebar-item');
  const colorOf = task =>
    taskColor(
      rows.find(
        r => r.querySelector('.sidebar-item-text').textContent === task,
      ),
    );
  assert.ok(/^hsl\(/.test(colorOf('task a2')), 'rows carry --task-color');
  assert.strictEqual(
    colorOf('task a2'),
    colorOf('task a1'),
    'same chat, same color',
  );
  assert.strictEqual(colorOf('task b2'), colorOf('task b1'));
  assert.notStrictEqual(
    colorOf('task a2'),
    colorOf('task b2'),
    'different chats differ',
  );
  assert.notStrictEqual(colorOf('task a2'), colorOf('task c1'));

  // An identical refresh keeps the flat rows (fast path, same view).
  const rowsBefore = all(win, '#history-list > .sidebar-item');
  sendHistory(win, posted, 0, page());
  assert.deepStrictEqual(listShape(win), FLAT);
  assert.strictEqual(
    all(win, '#history-list > .sidebar-item')[0],
    rowsBefore[0],
    'an identical refresh keeps the DOM rows',
  );

  // Back to chat panels: the same data, regrouped, colors gone.
  toggle.click();
  assert.strictEqual(toggle.getAttribute('aria-pressed'), 'false');
  assert.strictEqual(win.localStorage.getItem(VIEW_KEY), '0');
  assert.ok(!byId(win, 'history-list').classList.contains('legacy-view'));
  assert.deepStrictEqual(listShape(win), GROUPED);
  all(win, '#history-list .sidebar-item').forEach(r => {
    assert.strictEqual(taskColor(r), '', 'grouped rows carry no color');
  });
  win.close();
});

test('the flat list is remembered across loads and used for the first page', () => {
  const {win, posted} = makeWebview(w => {
    w.localStorage.setItem(VIEW_KEY, '1');
  });
  const toggle = byId(win, 'history-view-toggle');
  assert.strictEqual(toggle.getAttribute('aria-pressed'), 'true');
  assert.ok(byId(win, 'history-list').classList.contains('legacy-view'));
  sendHistory(win, posted, 0, page());
  assert.deepStrictEqual(listShape(win), FLAT);
  // Later pages extend the flat list in order.
  sendHistory(win, posted, 5, [session('D', 'd1', todayNoon - 3 * DAY)]);
  assert.deepStrictEqual(listShape(win), FLAT.concat(['row:task d1']));
  // A row without a chat id still gets the neutral fallback color.
  sendHistory(win, posted, 6, [session('', 'e1', todayNoon - 4 * DAY)]);
  const last = all(win, '#history-list > .sidebar-item').pop();
  assert.strictEqual(taskColor(last), 'hsl(0, 0%, 75%)');
  win.close();
});

test('toggling with nothing loaded only flips the view; filters and focus work in the flat list', () => {
  const {win, posted} = makeWebview();
  const toggle = byId(win, 'history-view-toggle');
  toggle.click();
  assert.strictEqual(
    byId(win, 'history-list').querySelectorAll('.sidebar-item').length,
    0,
  );
  assert.ok(byId(win, 'history-list').classList.contains('legacy-view'));
  sendHistory(win, posted, 0, [
    session('A', 'a2', todayNoon),
    Object.assign(session('B', 'b1', todayNoon - 600), {failed: true}),
    Object.assign(session('A', 'a1', todayNoon - 1200), {is_running: true}),
  ]);
  assert.deepStrictEqual(listShape(win), [
    'row:task a2',
    'row:task b1',
    'row:task a1',
  ]);
  // The "Errored" chip hides the failed row only.
  const hfErrors = byId(win, 'hf-errors');
  hfErrors.checked = false;
  hfErrors.dispatchEvent(new win.Event('change', {bubbles: true}));
  assert.deepStrictEqual(listShape(win), ['row:task a2', 'row:task a1']);
  hfErrors.checked = true;
  hfErrors.dispatchEvent(new win.Event('change', {bubbles: true}));
  // Keyboard focus on a row survives a changed-data rebuild.
  const rowB = all(win, '#history-list > .sidebar-item')[1];
  rowB.focus();
  sendHistory(win, posted, 0, [
    session('A', 'a3', todayNoon + 60),
    session('A', 'a2', todayNoon),
    Object.assign(session('B', 'b1', todayNoon - 600), {failed: true}),
    session('A', 'a1', todayNoon - 1200),
  ]);
  assert.deepStrictEqual(listShape(win), [
    'row:task a3',
    'row:task a2',
    'row:task b1',
    'row:task a1',
  ]);
  const active = win.document.activeElement;
  assert.strictEqual(
    active.querySelector('.sidebar-item-text').textContent,
    'task b1',
    'focus follows the same task after the rebuild',
  );
  win.close();
});

console.log(`\n${passed} passed, ${failures.length} failed`);
process.exit(failures.length ? 1 : 0);
