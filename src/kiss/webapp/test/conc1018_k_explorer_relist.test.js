// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

/* global require, console */

// Explorer re-listings that remove rows (handleDirListing):
// 1. an error re-listing of a folder drops the registered nodes of the
//    sub-folders it removes, as a successful re-listing does, so a later
//    refresh does not ask the daemon to list folders no longer in the tree;
// 2. a Rename box whose entry vanished mid-edit is cancelled with its row,
//    so a now-empty folder paints its "(empty)" note.

const assert = require('assert');
const {
  makePage,
  send,
  stubRaf,
  click,
  fakeMatchMedia,
} = require('./simplify2_harness');
const {showSection, explorerRow: row} = require('./ui_antipattern_harness');

const WD = '/ws/repo';

function makeExplorerPage() {
  const {win, posted} = makePage({
    bodyClass: 'remote-chat',
    beforeScripts: w => {
      stubRaf(w);
      fakeMatchMedia(w, true);
    },
  });
  send(win, {type: 'configData', config: {work_dir: WD}});
  showSection(win, 'meta-explorer');
  return {win, posted};
}

function listings(posted) {
  return posted.filter(m => m.type === 'listDir');
}

function answer(win, req, entries) {
  send(win, {
    type: 'dirListing',
    token: req.token,
    path: req.path,
    root: WD,
    entries,
  });
}

function notes(win) {
  return Array.from(win.document.querySelectorAll('.explorer-note')).map(
    e => e.textContent,
  );
}

function testErrorRelistingDropsSubfolders() {
  const {win, posted} = makeExplorerPage();
  const lst = () => listings(posted);
  answer(win, lst()[0], [{name: 'src', isDir: true}]);
  click(win, row(win, WD + '/src'));
  answer(win, lst()[1], [{name: 'pkg', isDir: true}]);
  const pkg = row(win, WD + '/src/pkg');
  click(win, pkg);
  answer(win, lst()[2], [{name: 'a.py', isDir: false}]);
  assert.strictEqual(lst().length, 3);

  // Refresh: root, src and pkg are re-listed; src now fails.
  click(win, win.document.getElementById('explorer-refresh'));
  const first = lst().slice(3);
  assert.deepStrictEqual(first.map(m => m.path).sort(), [
    WD,
    WD + '/src',
    WD + '/src/pkg',
  ]);
  const srcReq = first.find(m => m.path === WD + '/src');
  send(win, {
    type: 'dirListing',
    token: srcReq.token,
    path: WD + '/src',
    root: WD,
    error: 'permission denied',
  });
  assert.ok(!pkg.isConnected, 'pkg row is gone with the error');
  assert.ok(notes(win).includes('permission denied'));
  for (const m of first) {
    if (m.path === WD) answer(win, m, [{name: 'src', isDir: true}]);
    if (m.path === WD + '/src/pkg') answer(win, m, [{name: 'a.py'}]);
  }

  // Refresh again: only folders still in the tree are re-listed.
  const before = lst().length;
  click(win, win.document.getElementById('explorer-refresh'));
  const second = lst().slice(before);
  assert.deepStrictEqual(
    second.map(m => m.path).sort(),
    [WD, WD + '/src'],
    'the detached pkg folder is not re-listed',
  );
  win.close();
  console.log('  ok - an error re-listing drops the removed sub-folders');
}

function testErrorRelistingKeepsEditedFolder() {
  // The Rename box on an expanded folder survives an error re-listing
  // of its parent with the folder's contents, and that folder stays
  // registered: a later refresh re-lists it.
  const {win, posted} = makeExplorerPage();
  const lst = () => listings(posted);
  answer(win, lst()[0], [
    {name: 'src', isDir: true},
    {name: 'doc', isDir: true},
  ]);
  click(win, row(win, WD + '/src'));
  answer(win, lst()[1], [{name: 'a.py'}]);
  click(win, row(win, WD + '/doc'));
  answer(win, lst()[2], [{name: 'b.md'}]);
  const src = row(win, WD + '/src');
  src.focus();
  src.dispatchEvent(
    new win.KeyboardEvent('keydown', {
      key: 'F2',
      bubbles: true,
      cancelable: true,
    }),
  );
  assert.ok(src.querySelector('.explorer-input'), 'rename box on src');
  click(win, win.document.getElementById('explorer-refresh'));
  const first = lst().slice(3);
  const rootReq = first.find(m => m.path === WD);
  send(win, {
    type: 'dirListing',
    token: rootReq.token,
    path: WD,
    root: WD,
    error: 'boom',
  });
  assert.ok(src.isConnected, 'the edited folder row survives');
  assert.ok(row(win, WD + '/src/a.py'), 'with its listed contents');
  assert.ok(!row(win, WD + '/doc'), 'the other folder is gone');
  for (const m of first) {
    if (m.path !== WD) answer(win, m, [{name: 'x'}]);
  }
  const before = lst().length;
  click(win, win.document.getElementById('explorer-refresh'));
  assert.deepStrictEqual(
    lst()
      .slice(before)
      .map(m => m.path)
      .sort(),
    [WD, WD + '/src'],
    'src is still registered, doc is not',
  );
  win.close();
  console.log('  ok - an error re-listing keeps the folder being renamed');
}

function testVanishedRenameRowIsCancelled() {
  const {win, posted} = makeExplorerPage();
  const lst = () => listings(posted);
  answer(win, lst()[0], [{name: 'a.py', isDir: false}]);
  const a = row(win, WD + '/a.py');
  a.focus();
  a.dispatchEvent(
    new win.KeyboardEvent('keydown', {
      key: 'F2',
      bubbles: true,
      cancelable: true,
    }),
  );
  const input = win.document.querySelector('.explorer-input');
  assert.ok(input, 'rename box open');
  assert.strictEqual(input.value, 'a.py');

  // Another surface deleted a.py; the refresh re-lists the root without it.
  click(win, win.document.getElementById('explorer-refresh'));
  const last = lst()[lst().length - 1];
  answer(win, last, []);
  assert.ok(!a.isConnected, 'the row is gone');
  assert.ok(!input.isConnected, 'and the box with it');
  assert.deepStrictEqual(notes(win), ['(empty)'], 'the empty folder says so');

  // The folder fills up again and another entry is renamed: one box.
  click(win, win.document.getElementById('explorer-refresh'));
  answer(win, lst()[lst().length - 1], [{name: 'b.py'}]);
  assert.deepStrictEqual(notes(win), [], 'the note is gone with the entry');
  const b = row(win, WD + '/b.py');
  b.focus();
  b.dispatchEvent(
    new win.KeyboardEvent('keydown', {
      key: 'F2',
      bubbles: true,
      cancelable: true,
    }),
  );
  const boxes = win.document.querySelectorAll('.explorer-input');
  assert.strictEqual(boxes.length, 1, 'exactly one inline box');
  assert.strictEqual(boxes[0].value, 'b.py');
  win.close();
  console.log('  ok - a vanished rename row is cancelled with its entry');
}

testErrorRelistingDropsSubfolders();
testErrorRelistingKeepsEditedFolder();
testVanishedRenameRowIsCancelled();
