// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// The history panel's tag dropdown (#hf-tag): its options are the
// classification vocabulary, a chosen tag travels to the daemon on every
// getHistory request (page one AND later pages), and rows already on
// screen that lack the tag hide at once.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

const ALL_TAGS = [
  'work',
  'personal',
  'secret',
  'chore',
  'question',
  'coding',
  'testing',
  'debugging',
  'review',
  'research',
  'paper',
  'writing',
  'docs',
  'data',
  'devops',
  'messaging',
  'browsing',
  'shopping',
  'finance',
  'scheduling',
  'failed',
];

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
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));

  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function visibleTitles(win) {
  const list = win.document.getElementById('history-list');
  const out = [];
  list.querySelectorAll('.sidebar-item').forEach(r => {
    if (r.style.display !== 'none') {
      out.push(r.querySelector('.sidebar-item-text').textContent);
    }
  });
  return out;
}

function session(id, title, timestamp, tags) {
  return {
    id,
    task_id: Number(id.slice(4)),
    title,
    timestamp,
    preview: title,
    has_events: false,
    failed: false,
    is_running: false,
    tokens: 0,
    cost: 0,
    steps: 0,
    is_favorite: false,
    work_dir: '',
    startTs: timestamp * 1000,
    endTs: timestamp * 1000 + 10_000,
    tags,
  };
}

const SESSIONS = [
  session('chat1', 'fix the parser bug', 1_700_000_000, 'work,coding,debugging'),
  session('chat2', 'buy milk', 1_700_000_100, 'personal,shopping'),
  session('chat3', 'write the paper', 1_700_000_200, 'work,paper,writing'),
];

function selectTag(win, value) {
  const sel = win.document.getElementById('hf-tag');
  sel.value = value;
  sel.dispatchEvent(new win.Event('change', {bubbles: true}));
}

function getHistoryPosts(posted) {
  return posted.filter(m => m && m.type === 'getHistory');
}

function testDropdownMarkup() {
  const {win} = makeWebview();
  const doc = win.document;
  const sel = doc.getElementById('hf-tag');
  assert.ok(sel, '#hf-tag must exist');
  assert.strictEqual(sel.tagName, 'SELECT', '#hf-tag is a <select>');
  assert.ok(
    doc.getElementById('history-filters-body').contains(sel),
    'the tag dropdown lives inside the collapsible Filters body',
  );
  const pill = sel.parentElement;
  assert.ok(
    pill.classList.contains('history-filter-tag'),
    'the select sits in the .history-filter-tag pill',
  );
  assert.strictEqual(
    pill.parentElement,
    doc.querySelector('.history-filter-bar'),
    '.history-filter-tag is a direct child of the filter bar',
  );
  const options = Array.from(sel.options).map(o => o.value);
  assert.deepStrictEqual(
    options,
    [''].concat(ALL_TAGS),
    'options are "All tags" then every listable tag in classification order ' +
      '("subagent" is left out: sub-agent runs are never listed)',
  );
  assert.strictEqual(sel.value, '', 'no tag is selected by default');
  assert.strictEqual(
    sel.options[0].textContent,
    'All tags',
    'the empty option reads "All tags"',
  );
  const lbl = doc.querySelector('label[for="hf-tag"]');
  assert.ok(lbl, 'the select has a <label for="hf-tag">');
  win.close();
  console.log('  ok - #hf-tag dropdown lists every tag');
}

function testTagFiltersLoadedRowsAndRefetches() {
  const {win, posted} = makeWebview();
  send(win, {type: 'history', sessions: SESSIONS, offset: 0});
  assert.strictEqual(visibleTitles(win).length, 3, 'all rows shown at start');
  const pill = win.document.getElementById('hf-tag').parentElement;
  assert.ok(!pill.classList.contains('active'), 'pill inactive with no tag');

  const before = getHistoryPosts(posted).length;
  selectTag(win, 'work');
  assert.deepStrictEqual(
    visibleTitles(win).sort(),
    ['fix the parser bug', 'write the paper'],
    'rows without the tag hide before the daemon answers',
  );
  assert.ok(pill.classList.contains('active'), 'pill marks the active tag');
  const posts = getHistoryPosts(posted);
  assert.strictEqual(posts.length, before + 1, 'one refetch per change');
  const req = posts[posts.length - 1];
  assert.strictEqual(req.tag, 'work', 'the request carries the tag');
  assert.strictEqual(req.offset, 0, 'a changed tag restarts at page one');
  assert.strictEqual(req.query, '', 'the (empty) search text rides along');

  selectTag(win, 'shopping');
  assert.deepStrictEqual(
    visibleTitles(win),
    ['buy milk'],
    'a tag matches anywhere in the comma list, not only first',
  );

  selectTag(win, 'finance');
  assert.deepStrictEqual(visibleTitles(win), [], 'no row carries finance');
  assert.ok(
    win.document.querySelector('#history-list .sidebar-empty-filter'),
    'the "no tasks match" placeholder appears',
  );

  selectTag(win, '');
  assert.strictEqual(visibleTitles(win).length, 3, '"All tags" restores rows');
  assert.ok(!pill.classList.contains('active'), 'pill inactive again');
  assert.strictEqual(
    getHistoryPosts(posted)[getHistoryPosts(posted).length - 1].tag,
    '',
    'clearing the tag refetches without one',
  );
  win.close();
  console.log('  ok - choosing a tag hides other rows and refetches page one');
}

function testTagRidesOnSearchAndScrollRequests() {
  const {win, posted} = makeWebview();
  selectTag(win, 'coding');
  const search = win.document.getElementById('history-search');
  search.value = 'parser';
  search.dispatchEvent(new win.Event('input', {bubbles: true}));
  let req = getHistoryPosts(posted).pop();
  assert.strictEqual(req.query, 'parser', 'search text is sent');
  assert.strictEqual(req.tag, 'coding', 'the tag rides on a search request');

  // Fill a full page so the list believes more rows exist, then scroll.
  const page = [];
  for (let i = 0; i < 50; i++) {
    page.push(session(`chat${i + 10}`, `task ${i}`, 1_700_000_000 + i, 'work,coding'));
  }
  // A reply is only rendered when it answers the CURRENT request
  // generation (each search/tag change starts a new one).
  send(win, {type: 'history', sessions: page, offset: 0, generation: req.generation});
  const list = win.document.getElementById('history-list');
  Object.defineProperty(list, 'scrollHeight', {value: 1000, configurable: true});
  Object.defineProperty(list, 'clientHeight', {value: 500, configurable: true});
  list.scrollTop = 600;
  list.dispatchEvent(new win.Event('scroll'));
  req = getHistoryPosts(posted).pop();
  assert.strictEqual(req.offset, 50, 'the scroll asks for page two');
  assert.strictEqual(req.tag, 'coding', 'the tag rides on the page-two request');
  assert.strictEqual(req.query, 'parser', 'and so does the search text');
  win.close();
  console.log('  ok - the tag is sent with search and pagination requests');
}

function testTagExpandsChatGroups() {
  const {win, posted} = makeWebview();
  // Two tasks in one chat, none running: the group folds by default.
  const twoTasks = [
    session('chat1', 'fix the parser bug', 1_700_000_000, 'work,coding'),
    Object.assign(session('chat1', 'add a test', 1_700_000_050, 'work,testing'), {
      task_id: 11,
    }),
  ];
  send(win, {type: 'history', sessions: twoTasks, offset: 0});
  const group = win.document.querySelector('#history-list .history-chat-group');
  assert.ok(group, 'a chat group is rendered');
  assert.ok(group.classList.contains('collapsed'), 'folded without a filter');

  selectTag(win, 'testing');
  send(win, {
    type: 'history',
    sessions: [twoTasks[1]],
    offset: 0,
    generation: getHistoryPosts(posted).pop().generation,
  });
  const filtered = win.document.querySelector('#history-list .history-chat-group');
  assert.ok(
    !filtered.classList.contains('collapsed'),
    'a tag filter unfolds chat groups like a search does',
  );
  assert.deepStrictEqual(visibleTitles(win), ['add a test']);
  win.close();
  console.log('  ok - a tag filter unfolds chat groups');
}

function testTagCss() {
  const css = fs.readFileSync(path.join(MEDIA, 'main.css'), 'utf8');
  for (const sel of [
    '.history-filter-tag {',
    '.history-filter-tag.active {',
    '.history-filter-tag-select {',
  ]) {
    assert.ok(css.includes(sel), `main.css must style ${sel}`);
  }
  console.log('  ok - tag dropdown is styled');
}

function main() {
  testDropdownMarkup();
  testTagFiltersLoadedRowsAndRefetches();
  testTagRidesOnSearchAndScrollRequests();
  testTagExpandsChatGroups();
  testTagCss();
  console.log('historyTagFilter.test.js: all assertions passed.');
}

main();
