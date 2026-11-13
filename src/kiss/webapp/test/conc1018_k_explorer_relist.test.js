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

// A26_astra-6 / A26_fable-5: a re-listing of the parent keeps the row of
// a folder still there, and must also take over the folder's real
// location from the fresh listing: a symlink that pointed back at an
// ancestor (a cycle, never listed) and now points at an ordinary folder
// lists that folder on its next expansion instead of repeating the old
// "(symbolic link cycle)" verdict.
function testRelistingUpdatesASymlinkTarget(retarget) {
  const {win, posted} = makeExplorerPage();
  const lst = () => listings(posted);
  // `link` -> the workspace root itself: a cycle.
  answer(win, lst()[0], [{name: 'link', isDir: true, real: WD}]);
  click(win, win.document.getElementById('explorer-refresh'));
  // The link now points elsewhere (an explicit target, or none: a plain
  // folder at the parent's real path + name).
  answer(win, lst()[1], [Object.assign({name: 'link', isDir: true}, retarget)]);
  const link = row(win, WD + '/link');
  assert.ok(link, 'the link row survives the re-listing');
  click(win, link);
  const asked = lst().slice(2).map(m => m.path);
  assert.deepStrictEqual(
    asked,
    [WD + '/link'],
    'BUG - the retargeted link was still judged a cycle: ' +
      JSON.stringify(notes(win)),
  );
  answer(win, lst()[2], [{name: 'x.py'}]);
  assert.ok(row(win, WD + '/link/x.py'), 'the new target is listed');
  assert.ok(
    !notes(win).includes('(symbolic link cycle)'),
    'no cycle note for a link that no longer points at an ancestor',
  );
  win.close();
  console.log(
    '  ok - a re-listing takes over a symlink\'s new target (' +
      (retarget.real ? 'explicit' : 'derived') +
      ')',
  );
}

// Audit 2026-11-08 (W5 F1): a cycle refused on expansion is not a
// listed folder.  Refresh re-lists listed folders only, so it must not
// ask the daemon for the cyclic link (whose reply would paint a copy of
// the ancestor under it, link included, one level deeper per refresh);
// the cycle note stays while the link cycles, and once the parent's
// fresh listing retargets the link at an ordinary folder, expanding it
// again lists that folder.
function testRefreshKeepsABlockedCycleUnlisted() {
  const {win, posted} = makeExplorerPage();
  const lst = () => listings(posted);
  answer(win, lst()[0], [{name: 'loop', isDir: true, real: WD}]);
  const loop = () => row(win, WD + '/loop');
  click(win, loop());
  assert.deepStrictEqual(notes(win), ['(symbolic link cycle)']);
  assert.strictEqual(lst().length, 1, 'the cycle is not listed');
  click(win, win.document.getElementById('explorer-refresh'));
  assert.deepStrictEqual(
    lst().slice(1).map(m => m.path),
    [WD],
    'BUG - Refresh asked the daemon to list the cyclic link',
  );
  answer(win, lst()[1], [{name: 'loop', isDir: true, real: WD}]);
  assert.ok(loop().classList.contains('expanded'), 'the link row stays expanded');
  assert.deepStrictEqual(notes(win), ['(symbolic link cycle)'], 'the note stays');
  assert.strictEqual(row(win, WD + '/loop/loop'), null, 'no copy of the ancestor');
  // The link now points at an ordinary folder: the next expansion
  // lists it.
  click(win, win.document.getElementById('explorer-refresh'));
  answer(win, lst()[2], [{name: 'loop', isDir: true, real: WD + '/real'}]);
  click(win, loop());
  click(win, loop());
  assert.deepStrictEqual(
    lst().slice(3).map(m => m.path),
    [WD + '/loop'],
    'the retargeted link is listed on its expansion',
  );
  answer(win, lst()[3], [{name: 'x.py'}]);
  assert.ok(row(win, WD + '/loop/x.py'), 'the new target is listed');
  assert.deepStrictEqual(notes(win), []);
  win.close();
  console.log('  ok - Refresh leaves a blocked cycle unlisted; a retarget lists it');
}

// Audit 2026-11-11 (W1): a successful re-listing that removes the row
// the keyboard is on (its file was deleted by another process) hands
// the focus and the tree's tab stop to the folder's own row, as an
// error re-listing does; they used to fall to <body>, which ended the
// arrow-key navigation until the user clicked back into the tree.
function testRelistingWithoutTheFocusedRowKeepsFocusInTheTree() {
  const {win, posted} = makeExplorerPage();
  const lst = () => listings(posted);
  answer(win, lst()[0], [{name: 'gone.txt'}, {name: 'stays.txt'}]);
  const gone = row(win, WD + '/gone.txt');
  gone.focus();
  assert.strictEqual(win.document.activeElement, gone, 'setup: row focused');
  click(win, win.document.getElementById('explorer-refresh'));
  answer(win, lst()[1], [{name: 'stays.txt'}]);
  assert.strictEqual(row(win, WD + '/gone.txt'), null, 'the row is gone');
  const root = row(win, WD);
  assert.strictEqual(
    win.document.activeElement,
    root,
    'BUG - the focus fell out of the tree with the removed row',
  );
  assert.strictEqual(root.tabIndex, 0, 'the folder row is the tab stop');
  assert.strictEqual(row(win, WD + '/stays.txt').tabIndex, -1);

  // A row that survives keeps the focus, as before.
  const stays = row(win, WD + '/stays.txt');
  stays.focus();
  click(win, win.document.getElementById('explorer-refresh'));
  answer(win, lst()[2], [{name: 'new.txt'}, {name: 'stays.txt'}]);
  assert.strictEqual(win.document.activeElement, stays);
  assert.strictEqual(stays.tabIndex, 0);
  win.close();
  console.log('  ok - a re-listing without the focused row keeps the focus');
}

testErrorRelistingDropsSubfolders();
testRelistingWithoutTheFocusedRowKeepsFocusInTheTree();
testErrorRelistingKeepsEditedFolder();
testVanishedRenameRowIsCancelled();
testRelistingUpdatesASymlinkTarget({real: WD + '/real-folder'});
testRelistingUpdatesASymlinkTarget({});
testRefreshKeepsABlockedCycleUnlisted();
