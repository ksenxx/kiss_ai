// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end (jsdom) test for "Set as Working Directory" in the Explorer
// (media/main.js setExplorerWorkDir / handleWorkDirListing): the folder
// that was the working directory is kept as an added Explorer folder
// only once the daemon has accepted the pick.  It used to be persisted
// to localStorage the moment the check mark was clicked, so a refused
// pick (the daemon answers with an error, or the canonical path is a
// root) left the unchanged working directory ALSO listed as a persisted
// extra root.

'use strict';

const assert = require('assert');
const {
  makePage: makeChatPage,
  send,
  click,
  stubRaf,
  fakeMatchMedia,
  ofType,
} = require('./simplify2_harness');
const {all, showSection} = require('./ui_antipattern_harness');

const WD = '/ws/repo';
const OTHER = '/data/other';
const ROOTS_KEY = 'kiss-explorer-roots';

function makePage() {
  return makeChatPage({
    bodyClass: 'remote-chat',
    beforeScripts: win => {
      stubRaf(win);
      fakeMatchMedia(win, true);
      // OTHER is already an added folder when the page loads.
      win.localStorage.setItem(ROOTS_KEY, JSON.stringify([OTHER]));
    },
    extraScripts: [
      'marked.min.js',
      'contentContextMenu.js',
      'treeContextMenu.js',
    ],
  });
}

function savedRoots(win) {
  return JSON.parse(win.localStorage.getItem(ROOTS_KEY) || '[]');
}

function rootRows(win) {
  return all(win, '#explorer-tree > .explorer-row.is-root');
}

/** Answer the last listDir for *p* with an empty listing. */
function answerListing(win, posted, p) {
  const reqs = ofType(posted, 'listDir').filter(m => m.path === p);
  const req = reqs[reqs.length - 1];
  assert.ok(req, 'a listDir for ' + p + ' was sent');
  send(win, {
    type: 'dirListing',
    token: req.token,
    path: p,
    root: req.workDir,
    entries: [],
  });
}

/** Answer the Explorer's pending 'workdir:' check for *dir*. */
function answerWorkDirCheck(win, posted, dir, error) {
  const checks = ofType(posted, 'listDir').filter(
    m => String(m.token).startsWith('workdir:') && m.path === dir,
  );
  const req = checks[checks.length - 1];
  assert.ok(req, 'a workdir: listDir check for ' + dir + ' was sent');
  if (error) {
    send(win, {type: 'dirListing', token: req.token, path: dir, error});
  } else {
    send(win, {
      type: 'dirListing',
      token: req.token,
      path: dir,
      root: dir,
      entries: [],
    });
  }
}

function clickSetWorkDir(win) {
  const row = rootRows(win).find(r => r.dataset.explorerPath === OTHER);
  assert.ok(row, 'the OTHER root row exists');
  click(win, row.querySelector('.explorer-root-set'));
}

function testRefusedPickKeepsRootsUnchanged() {
  const {win, posted} = makePage();
  send(win, {type: 'configData', config: {work_dir: WD}});
  showSection(win, 'meta-explorer');
  answerListing(win, posted, WD);
  assert.deepStrictEqual(savedRoots(win), [OTHER], 'precondition');

  // The daemon refuses the pick: nothing changed, so nothing is persisted.
  clickSetWorkDir(win);
  assert.deepStrictEqual(
    savedRoots(win),
    [OTHER],
    'the old working directory must not be persisted before the daemon confirms',
  );
  answerWorkDirCheck(win, posted, OTHER, 'Not a directory: ' + OTHER);
  assert.deepStrictEqual(
    savedRoots(win),
    [OTHER],
    'a refused pick must not leave the unchanged working directory as an extra root',
  );
  assert.deepStrictEqual(
    rootRows(win).map(r => r.dataset.explorerPath),
    [WD, OTHER],
  );

  // The canonical spelling turns out to be a root: refused the same way.
  clickSetWorkDir(win);
  const checks = ofType(posted, 'listDir').filter(m =>
    String(m.token).startsWith('workdir:'),
  );
  send(win, {
    type: 'dirListing',
    token: checks[checks.length - 1].token,
    path: '/',
    root: '/',
    entries: [],
  });
  assert.deepStrictEqual(
    savedRoots(win),
    [OTHER],
    'a root answer is a refusal too',
  );
  assert.ok(
    win.document.querySelector('.kiss-notification'),
    'the refusal is reported',
  );

  // An accepted pick: the old working directory stays listed, persisted.
  clickSetWorkDir(win);
  answerWorkDirCheck(win, posted, OTHER);
  assert.strictEqual(ofType(posted, 'setWorkDir').length, 1, 'one setWorkDir');
  assert.deepStrictEqual(
    savedRoots(win),
    [OTHER, WD],
    'the old work dir is kept once accepted',
  );
  assert.deepStrictEqual(
    rootRows(win).map(r => r.dataset.explorerPath),
    [OTHER, WD],
  );
  win.close();
  console.log(
    '  ok - a refused "Set as Working Directory" persists no extra root',
  );
}

function testAcceptedPickOfTheSameFolderAddsNothing() {
  const {win, posted} = makePage();
  send(win, {type: 'configData', config: {work_dir: WD}});
  showSection(win, 'meta-explorer');
  answerListing(win, posted, WD);
  // Set OTHER, accepted; then set WD (now an extra root) back: WD is
  // already listed, so it must not be pushed a second time.
  clickSetWorkDir(win);
  answerWorkDirCheck(win, posted, OTHER);
  assert.deepStrictEqual(savedRoots(win), [OTHER, WD]);
  const wdRow = rootRows(win).find(r => r.dataset.explorerPath === WD);
  click(win, wdRow.querySelector('.explorer-root-set'));
  answerWorkDirCheck(win, posted, WD);
  assert.deepStrictEqual(savedRoots(win), [OTHER, WD], 'no duplicate root');
  win.close();
  console.log('  ok - an accepted pick never duplicates a listed root');
}

function main() {
  testRefusedPickKeepsRootsUnchanged();
  testAcceptedPickOfTheSameFolderAddsNothing();
  console.log('explorerWorkDirRefused_sweep1026.test.js: all tests passed');
}

main();
