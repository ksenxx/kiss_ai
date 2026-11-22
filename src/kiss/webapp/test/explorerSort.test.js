// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end (jsdom) tests for the Explorer's Sort submenu (media/main.js):
// a folder's context menu offers six orders, the one the folder is
// listed in checked; the pick is sent as the `sort` of listDir (the
// daemon sorts), kept per machine and folder, inherited by the folders
// beneath it, and remembered in localStorage across reloads.

'use strict';

const assert = require('assert');
const {
  makePage: buildPage,
  send,
  click,
  byId,
  stubRaf,
  fakeMatchMedia,
  ofType,
  pickMachine,
} = require('./simplify2_harness');
const {
  all,
  showSection,
  menuItem,
  rightClick,
  forbidNativeDialogs,
  makeRunner,
} = require('./ui_antipattern_harness');
const {rootRows} = require('./workdir_helpers');

// Every page a test built: the runner closes them when the test ends.
const windows = [];
const {test, report} = makeRunner({windows});

const WD = '/ws/repo';
const STORE = 'kiss-explorer-sort';
const LABELS = [
  'Alphabetical (Ascending)',
  'Alphabetical (Descending)',
  'Created (Ascending)',
  'Created (Descending)',
  'Modified (Ascending)',
  'Modified (Descending)',
];

function dir(name) {
  return {name, isDir: true};
}

function file(name) {
  return {name, isDir: false};
}

// What the daemon lists, by folder.
const TREE = {
  [WD]: [dir('docs'), dir('src'), file('a.txt'), file('b.txt')],
  [WD + '/docs']: [file('x.md')],
  [WD + '/src']: [dir('lib'), dir('pkg'), file('a.py'), file('z.py')],
  [WD + '/src/lib']: [],
  [WD + '/src/pkg']: [file('m.py')],
};

/**
 * A page whose Explorer shows WD.  *storage* is what localStorage holds
 * when the page loads (key -> value), as after a reload.
 */
function makePage(storage) {
  const {win, posted} = buildPage({
    bodyClass: 'remote-chat',
    extraScripts: ['contentContextMenu.js', 'treeContextMenu.js'],
    beforeScripts: w => {
      stubRaf(w);
      fakeMatchMedia(w, true);
      forbidNativeDialogs(w);
      Object.keys(storage || {}).forEach(k => {
        w.localStorage.setItem(k, storage[k]);
      });
    },
  });
  windows.push(win);
  send(win, {type: 'configData', config: {work_dir: WD}});
  showSection(win, 'meta-explorer');
  return {win, posted};
}

/** The listDir requests in the ledger, as `path order`. */
function listed(posted) {
  return ofType(posted, 'listDir').map(m => m.path + ' ' + m.sort);
}

/** Answer the listDir request *req* with *entries*. */
function reply(win, req, entries) {
  const ev = {
    type: 'dirListing',
    token: req.token,
    path: req.path,
    root: req.workDir,
    entries,
  };
  if (req.machine) ev.machine = req.machine;
  send(win, ev);
}

/** Answer the latest listing of each of *paths* with its TREE entries. */
function answerAll(win, posted, paths) {
  paths.forEach(p => {
    const req = ofType(posted, 'listDir')
      .filter(m => m.path === p)
      .pop();
    assert.ok(req, 'a listDir for ' + p + ' was sent');
    reply(win, req, TREE[p]);
  });
}

function row(win, p) {
  return all(win, '.explorer-row').find(r => r.dataset.explorerPath === p);
}

/** Expand the folder *p* and answer its first listing. */
function expand(win, posted, p) {
  click(win, row(win, p));
  answerAll(win, posted, [p]);
}

/** Right-click *r* and open the Sort submenu of its menu; its items. */
function openSort(win, r) {
  rightClick(win, r);
  click(win, menuItem(win, 'Sort'));
  return all(win, '#sidebar-context-submenu .tree-ctx-item');
}

/** The order the Sort submenu of *r* shows checked. */
function checkedOrder(win, r) {
  const checked = openSort(win, r).filter(
    i => i.getAttribute('aria-checked') === 'true',
  );
  assert.strictEqual(checked.length, 1, 'exactly one order is checked');
  return checked[0].textContent;
}

function pickSort(win, r, label) {
  click(
    win,
    openSort(win, r).find(i => i.textContent === label),
  );
}

function saved(win) {
  return JSON.parse(win.localStorage.getItem(STORE));
}

test("A folder's menu offers Sort in a group of its own; files and multi-selections do not", () => {
  const {win, posted} = makePage();
  answerAll(win, posted, [WD]);
  const src = row(win, WD + '/src');
  rightClick(win, src);
  const rows = Array.from(byId(win, 'sidebar-context-menu').children);
  const at = rows.indexOf(menuItem(win, 'Sort'));
  assert.deepStrictEqual(
    rows.slice(at - 2, at + 3).map(r => {
      const label = r.querySelector('.tree-ctx-label');
      return label ? label.textContent : '-';
    }),
    ['New Folder...', '-', 'Sort', '-', 'Find in Folder...'],
  );
  // The six orders; a folder nobody sorted is listed by name.
  const items = openSort(win, src);
  assert.deepStrictEqual(
    items.map(i => i.textContent),
    LABELS,
  );
  assert.deepStrictEqual(
    items.map(i => i.getAttribute('aria-checked') === 'true'),
    [true, false, false, false, false, false],
  );
  assert.strictEqual(checkedOrder(win, rootRows(win)[0]), LABELS[0]);
  // A file's menu has no Sort, nor has a multi-selection's.
  rightClick(win, row(win, WD + '/a.txt'));
  assert.ok(menuItem(win, 'Open to the Side'));
  assert.strictEqual(menuItem(win, 'Sort'), undefined);
  src.dispatchEvent(
    new win.MouseEvent('click', {bubbles: true, ctrlKey: true}),
  );
  rightClick(win, src);
  assert.ok(menuItem(win, 'Delete'), 'the multi-selection menu is open');
  assert.strictEqual(menuItem(win, 'New File...'), undefined);
  assert.strictEqual(menuItem(win, 'Sort'), undefined);
  assert.deepStrictEqual(
    listed(posted),
    [WD + ' name-asc'],
    'nothing re-listed',
  );
});

test('Picking an order re-lists the folder and the listed folders beneath it; rows follow the reply', () => {
  const {win, posted} = makePage();
  answerAll(win, posted, [WD]);
  ['/docs', '/src', '/src/pkg'].forEach(p => expand(win, posted, WD + p));
  const src = row(win, WD + '/src');
  const pkg = row(win, WD + '/src/pkg');
  // Collapsed again, pkg stays listed.
  click(win, pkg);
  assert.deepStrictEqual(
    listed(posted),
    ['', '/docs', '/src', '/src/pkg'].map(p => WD + p + ' name-asc'),
  );

  posted.length = 0;
  pickSort(win, src, 'Modified (Descending)');
  assert.deepStrictEqual(listed(posted), [
    WD + '/src modified-desc',
    WD + '/src/pkg modified-desc',
  ]);
  assert.ok(
    !byId(win, 'sidebar-context-menu').childElementCount,
    'menu closed',
  );
  // The rows take the order of the reply; a kept folder keeps its row.
  reply(win, ofType(posted, 'listDir')[0], [
    dir('pkg'),
    dir('lib'),
    file('z.py'),
    file('a.py'),
  ]);
  assert.deepStrictEqual(
    Array.from(src.nextSibling.querySelectorAll(':scope > .explorer-row')).map(
      r => r.textContent,
    ),
    ['pkg', 'lib', 'z.py', 'a.py'],
  );
  assert.strictEqual(row(win, WD + '/src/pkg'), pkg);
  answerAll(win, posted, [WD + '/src/pkg']);
  // The check mark is the folder's own order, else its nearest ancestor's
  // (lib has not been listed yet).
  assert.strictEqual(checkedOrder(win, src), 'Modified (Descending)');
  assert.strictEqual(checkedOrder(win, pkg), 'Modified (Descending)');
  assert.strictEqual(
    checkedOrder(win, row(win, WD + '/src/lib')),
    'Modified (Descending)',
  );
  assert.strictEqual(checkedOrder(win, row(win, WD + '/docs')), LABELS[0]);
  assert.strictEqual(checkedOrder(win, rootRows(win)[0]), LABELS[0]);

  // A folder's own choice wins over its ancestor's.
  posted.length = 0;
  pickSort(win, pkg, 'Alphabetical (Descending)');
  assert.deepStrictEqual(listed(posted), [WD + '/src/pkg name-desc']);
  answerAll(win, posted, [WD + '/src/pkg']);
  // A folder first expanded later is listed in the inherited order.
  posted.length = 0;
  click(win, row(win, WD + '/src/lib'));
  assert.deepStrictEqual(listed(posted), [WD + '/src/lib modified-desc']);
  const libFirst = ofType(posted, 'listDir')[0];
  // The top-level folder's pick reaches every listed folder, each in its
  // effective order; lib, whose first listing is in flight, is asked
  // again when that reply lands.
  posted.length = 0;
  pickSort(win, rootRows(win)[0], 'Created (Ascending)');
  assert.deepStrictEqual(listed(posted), [
    WD + ' created-asc',
    WD + '/docs created-asc',
    WD + '/src modified-desc',
    WD + '/src/pkg name-desc',
  ]);
  answerAll(win, posted, [WD, WD + '/docs', WD + '/src', WD + '/src/pkg']);
  posted.length = 0;
  reply(win, libFirst, []);
  assert.deepStrictEqual(listed(posted), [WD + '/src/lib modified-desc']);
  answerAll(win, posted, [WD + '/src/lib']);
  // The refresh button still re-lists every listed folder.
  posted.length = 0;
  click(win, byId(win, 'explorer-refresh'));
  assert.deepStrictEqual(listed(posted), [
    WD + ' created-asc',
    WD + '/docs created-asc',
    WD + '/src modified-desc',
    WD + '/src/lib modified-desc',
    WD + '/src/pkg name-desc',
  ]);
});

test('A folder shown under two top-level folders is re-listed under both', () => {
  const {win, posted} = makePage({
    'kiss-explorer-roots': JSON.stringify([WD + '/src']),
  });
  answerAll(win, posted, [WD]);
  const top = rootRows(win)[1];
  const nested = all(win, '.explorer-row').find(
    r => r.dataset.explorerPath === WD + '/src' && r !== top,
  );
  click(win, nested);
  click(win, top);
  ofType(posted, 'listDir')
    .slice(-2)
    .forEach(req => reply(win, req, TREE[WD + '/src']));
  posted.length = 0;
  pickSort(win, nested, 'Created (Descending)');
  assert.deepStrictEqual(
    ofType(posted, 'listDir')
      .map(m => [m.path, m.workDir, m.sort].join(' '))
      .sort(),
    [
      WD + '/src ' + WD + ' created-desc',
      WD + '/src ' + WD + '/src created-desc',
    ],
  );
  assert.strictEqual(checkedOrder(win, top), 'Created (Descending)');
});

test('The choices survive a reload, the 200 most recent of them', () => {
  let {win, posted} = makePage();
  answerAll(win, posted, [WD]);
  expand(win, posted, WD + '/src');
  pickSort(win, row(win, WD + '/src'), 'Modified (Descending)');
  pickSort(win, rootRows(win)[0], 'Created (Ascending)');
  // A folder never listed has nothing to list again: its pick is used
  // when it is expanded.
  posted.length = 0;
  pickSort(win, row(win, WD + '/docs'), 'Modified (Ascending)');
  assert.deepStrictEqual(listed(posted), []);
  click(win, row(win, WD + '/docs'));
  assert.deepStrictEqual(listed(posted), [WD + '/docs modified-asc']);
  // Chosen again, a folder's choice becomes the most recent one.
  pickSort(win, row(win, WD + '/src'), 'Alphabetical (Descending)');
  assert.deepStrictEqual(saved(win), [
    ['\0' + WD, 'created-asc'],
    ['\0' + WD + '/docs', 'modified-asc'],
    ['\0' + WD + '/src', 'name-desc'],
  ]);

  ({win, posted} = makePage({[STORE]: win.localStorage.getItem(STORE)}));
  assert.deepStrictEqual(listed(posted), [WD + ' created-asc']);
  answerAll(win, posted, [WD]);
  click(win, row(win, WD + '/src'));
  assert.strictEqual(listed(posted)[1], WD + '/src name-desc');
  assert.strictEqual(
    checkedOrder(win, rootRows(win)[0]),
    'Created (Ascending)',
  );

  const old = Array.from({length: 200}, (_, i) => ['\0/old/' + i, 'name-desc']);
  ({win, posted} = makePage({[STORE]: JSON.stringify(old)}));
  answerAll(win, posted, [WD]);
  pickSort(win, rootRows(win)[0], 'Modified (Ascending)');
  assert.strictEqual(saved(win).length, 200);
  assert.deepStrictEqual(saved(win)[0], old[1], 'the oldest choice is dropped');
  assert.deepStrictEqual(saved(win)[199], ['\0' + WD, 'modified-asc']);
});

test('Unreadable or unknown saved choices are dropped; a full storage keeps the pick for the session', () => {
  let {win, posted} = makePage({[STORE]: '{oops'});
  assert.deepStrictEqual(listed(posted), [WD + ' name-asc']);

  const stored = JSON.stringify([
    ['\0' + WD, 'by-size'],
    ['\0' + WD + '/src', 'modified-asc'],
  ]);
  ({win, posted} = makePage({[STORE]: stored}));
  assert.deepStrictEqual(listed(posted), [WD + ' name-asc']);
  answerAll(win, posted, [WD]);
  expand(win, posted, WD + '/src');
  assert.strictEqual(listed(posted)[1], WD + '/src modified-asc');

  // Fill the storage to its quota (jsdom's is 5,000,000 code units of
  // keys and values), so the next save is refused.
  const ls = win.localStorage;
  let used = 'filler'.length;
  for (let i = 0; i < ls.length; i++)
    used += ls.key(i).length + ls.getItem(ls.key(i)).length;
  ls.setItem('filler', 'x'.repeat(5000000 - used));
  posted.length = 0;
  pickSort(win, rootRows(win)[0], 'Created (Descending)');
  assert.deepStrictEqual(listed(posted), [
    WD + ' created-desc',
    WD + '/src modified-asc',
  ]);
  assert.strictEqual(ls.getItem(STORE), stored, 'nothing could be saved');
  assert.strictEqual(
    checkedOrder(win, rootRows(win)[0]),
    'Created (Descending)',
  );
});

test("A machine's folders keep choices of their own", () => {
  const {win, posted} = makePage();
  answerAll(win, posted, [WD]);
  pickSort(win, rootRows(win)[0], 'Created (Descending)');
  // The chat moves to a registered machine whose working directory has
  // the same path: the tree is that machine's, listed by name.
  send(win, {type: 'configData', config: {work_dir: WD}, machine: 'laptop'});
  send(win, {
    type: 'machines',
    machines: [
      {name: 'laptop', local: true, status: 'reachable', address: 'https://a'},
      {name: 'gpu-box', status: 'reachable', address: 'https://b'},
    ],
    checkedAt: 1,
  });
  posted.length = 0;
  pickMachine(win, 'gpu-box');
  send(win, {
    type: 'machineWorkDirs',
    machine: 'gpu-box',
    workDir: WD,
    recentWorkDirs: [{path: WD, ts: 1}],
    error: '',
  });
  const first = ofType(posted, 'listDir').pop();
  assert.deepStrictEqual(
    [first.machine, first.path, first.sort],
    ['gpu-box', WD, 'name-asc'],
  );
  reply(win, first, TREE[WD]);
  posted.length = 0;
  pickSort(win, rootRows(win)[0], 'Modified (Ascending)');
  assert.deepStrictEqual(
    ofType(posted, 'listDir').map(m => [m.machine, m.path, m.sort]),
    [['gpu-box', WD, 'modified-asc']],
  );
  assert.deepStrictEqual(saved(win), [
    ['\0' + WD, 'created-desc'],
    ['gpu-box\0' + WD, 'modified-asc'],
  ]);
});

report('explorerSort');
