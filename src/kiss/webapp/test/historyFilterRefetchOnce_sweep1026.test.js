// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end (jsdom) test for the Chats list's daemon-applied filters
// (media/main.js: the tag, machine and work-dir dropdowns).  A change
// hides the loaded rows the filter excludes and refetches page one.
// When the rows left no longer fill the list, the visibility pass used
// to ask for the NEXT page of the old view first (old generation, old
// offset: a wasted daemon query whose reply is discarded) and only then
// restart at page one.  One request per change, page one, the current
// generation.

'use strict';

const assert = require('assert');
const {makePage, send} = require('./simplify2_harness');
const {makeRow, selectFilter} = require('./history_helpers');

function session(i, tags) {
  return makeRow({
    id: 'chat' + i,
    task_id: i,
    title: 'task ' + i,
    timestamp: 1_700_000_000 + i,
    tokens: 0,
    steps: 0,
    startTs: (1_700_000_000 + i) * 1000,
    endTs: (1_700_000_000 + i) * 1000 + 10_000,
    tags,
  });
}

function getHistoryPosts(posted) {
  return posted.filter(m => m && m.type === 'getHistory');
}

function bootFullPage() {
  const {win, posted} = makePage({bodyClass: 'remote-chat'});
  send(win, {type: 'configData', config: {work_dir: ''}, apiKeys: {}});
  win.document.getElementById('sidebar').classList.add('open');
  // A full page: the list believes more rows exist.
  const page = [];
  for (let i = 1; i <= 50; i++)
    page.push(session(i, i % 2 ? 'work,coding' : 'personal'));
  const first = getHistoryPosts(posted).pop();
  send(win, {
    type: 'history',
    sessions: page,
    offset: 0,
    generation: first ? first.generation : 0,
  });
  // Laid out, and taller than what is left once half the rows hide.
  const list = win.document.getElementById('history-list');
  Object.defineProperty(list, 'scrollHeight', {value: 100, configurable: true});
  Object.defineProperty(list, 'clientHeight', {value: 500, configurable: true});
  return {win, posted};
}

function checkOneRequest(filterId, value, field) {
  const {win, posted} = bootFullPage();
  const before = getHistoryPosts(posted).length;
  selectFilter(win, filterId, value);
  const posts = getHistoryPosts(posted).slice(before);
  assert.strictEqual(
    posts.length,
    1,
    filterId + ': one request per change; got ' + JSON.stringify(posts),
  );
  assert.strictEqual(posts[0].offset, 0, filterId + ': page one');
  assert.strictEqual(posts[0][field], value, filterId + ': carries the filter');
  assert.strictEqual(
    win.document.getElementById('history-loader'),
    null,
    filterId + ': no "Loading..." row of a next-page request',
  );
  win.close();
  console.log('  ok - a ' + filterId + ' change sends one page-one request');
}

function main() {
  checkOneRequest('hf-tag', 'work', 'tag');
  checkOneRequest('hf-machine', '__local__', 'machine');
  console.log('historyFilterRefetchOnce_sweep1026.test.js: all tests passed');
}

main();
