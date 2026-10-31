// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

/* global require, console, process */

// Audit 2026-10-16 (review-mainjs I2): a synthetic result view (a
// commit's patch, a file diff, a search) opened from a gitShow / fsResult
// reply a registered machine answered must stay on that machine: the
// daemon stamps the reply with `machine`
// (sidebar_panels.machine_workspace_reply), and the same repository
// path / revision on two machines is two results, so each keeps its own
// content tab.  Before the fix openTextResultTab() and openGitDiffTab()
// dropped the stamp, so a remote reply re-rendered the local tab.  The
// mainjs-1 F1 fix stays: the header drop-down's unsent pick is never
// consulted (audit1016b_file_content_machine.test.js).
// main.js against the real chat.html, through the ui_antipattern harness.

const assert = require('assert');
const H = require('./ui_antipattern_harness');

const {WD, SHA_A, makePage, send, sleep, ofType, runMenuItem} = H;

/** The open content tabs, copied into this realm's Array for deepStrictEqual. */
function contentTabs(win) {
  return [...win._testApi.openTabs()].filter(t => t.isContentTab);
}

/**
 * Give the page's recording Monaco stub the diff editor a gitShow diff
 * opens; each diff editor is recorded in *created* with the text of
 * its modified side as `value`.
 */
function stubDiffEditor(created) {
  const editor = created.monaco.editor;
  editor.createModel = text => ({getValue: () => text, dispose: () => {}});
  editor.createDiffEditor = holder => {
    const entry = {holder, value: undefined};
    created.push(entry);
    return {
      setModel: models => {
        entry.value = models ? models.modified.getValue() : undefined;
      },
      dispose: () => {},
    };
  };
}

/**
 * The text each content tab's viewer shows, in tab order: the newest
 * editor created in the tab's view.  Monaco resolves asynchronously, so
 * this lets the page settle a tick first.
 */
async function shownTexts(win, created) {
  await sleep(0);
  return contentTabs(win).map(t => {
    const view = win.document.querySelector(
      `.content-tab-view[data-tab-id="${t.id}"]`,
    );
    const mine = created.filter(c => view && view.contains(c.holder));
    assert.ok(mine.length, `content tab ${t.id} shows an editor`);
    return mine[mine.length - 1].value;
  });
}

/** Run "Open Changes" on *row* and return the gitShow request it posted. */
function askOpenChanges(win, posted, row) {
  const before = ofType(posted, 'gitShow').length;
  runMenuItem(win, row, 'Open Changes');
  const reqs = ofType(posted, 'gitShow');
  assert.strictEqual(reqs.length, before + 1, 'one gitShow was posted');
  return reqs[reqs.length - 1];
}

function showReply(req, machine, text) {
  const ev = {
    type: 'gitShow',
    token: req.token,
    tabId: req.tabId,
    repo: WD,
    sha: SHA_A,
    subject: 'Title line',
    text: text,
  };
  if (machine) ev.machine = machine;
  return ev;
}

function diffReply(req, machine, modified) {
  const ev = {
    type: 'gitShow',
    token: req.token,
    tabId: req.tabId,
    mode: 'diff',
    repo: WD,
    sha: SHA_A,
    parent: 'b'.repeat(40),
    path: WD + '/src/a.py',
    original: 'old\n',
    modified: modified,
  };
  if (machine) ev.machine = machine;
  return ev;
}

async function testCommitPatchPerMachine() {
  const {win, posted, created} = makePage();
  const row = H.openScm(win, posted);
  // Local first.
  send(win, showReply(askOpenChanges(win, posted, row), '', 'local patch'));
  let tabs = contentTabs(win);
  assert.strictEqual(tabs.length, 1, 'the local patch opened a tab');
  assert.strictEqual(tabs[0].ownerBrowseMachine, '', 'on this machine');
  // The same commit of the same repository path answered by gpu-box is
  // another result: a second tab, bound to gpu-box.
  send(
    win,
    showReply(askOpenChanges(win, posted, row), 'gpu-box', 'remote patch'),
  );
  tabs = contentTabs(win);
  assert.strictEqual(
    tabs.length,
    2,
    'a remote reply for the same commit opened its own tab',
  );
  assert.strictEqual(tabs[0].ownerBrowseMachine, '');
  assert.strictEqual(tabs[1].ownerBrowseMachine, 'gpu-box');
  assert.strictEqual(tabs[0].contentPath, tabs[1].contentPath);
  assert.deepStrictEqual(await shownTexts(win, created), [
    'local patch',
    'remote patch',
  ]);
  // Repeating either refreshes its own tab, never the other's.
  send(win, showReply(askOpenChanges(win, posted, row), '', 'local again'));
  assert.deepStrictEqual(
    await shownTexts(win, created),
    ['local again', 'remote patch'],
    'the local repeat refreshed the local tab only',
  );
  send(
    win,
    showReply(askOpenChanges(win, posted, row), 'gpu-box', 'remote again'),
  );
  assert.deepStrictEqual(
    await shownTexts(win, created),
    ['local again', 'remote again'],
    'the gpu-box repeat refreshed the gpu-box tab only',
  );
  tabs = contentTabs(win);
  assert.deepStrictEqual(
    tabs.map(t => t.ownerBrowseMachine),
    ['', 'gpu-box'],
    'repeats reuse the tab of their machine',
  );
  win.close();
  console.log('  ok - a commit patch keeps one tab per machine');
}

async function testFileDiffPerMachine() {
  const {win, posted, created} = makePage();
  stubDiffEditor(created);
  const row = H.openScm(win, posted);
  send(win, diffReply(askOpenChanges(win, posted, row), '', 'local\n'));
  assert.strictEqual(contentTabs(win).length, 1, 'the local diff opened');
  send(win, diffReply(askOpenChanges(win, posted, row), 'gpu-box', 'remote\n'));
  const tabs = contentTabs(win);
  assert.strictEqual(
    tabs.length,
    2,
    'a remote diff of the same file and commit is a second tab',
  );
  assert.deepStrictEqual(
    tabs.map(t => t.ownerBrowseMachine),
    ['', 'gpu-box'],
  );
  assert.strictEqual(tabs[0].contentPath, tabs[1].contentPath);
  assert.deepStrictEqual(await shownTexts(win, created), [
    'local\n',
    'remote\n',
  ]);
  send(win, diffReply(askOpenChanges(win, posted, row), 'gpu-box', 'again\n'));
  assert.strictEqual(contentTabs(win).length, 2, 'a repeat reuses its tab');
  assert.deepStrictEqual(
    await shownTexts(win, created),
    ['local\n', 'again\n'],
    'the gpu-box repeat refreshed the gpu-box diff only',
  );
  win.close();
  console.log('  ok - a file diff keeps one tab per machine');
}

(async () => {
  await testCommitPatchPerMachine();
  await testFileDiffPerMachine();
  console.log('audit1016b_virtual_result_machine: all passed');
  process.exit(0);
})().catch(e => {
  console.error(e);
  process.exit(1);
});
