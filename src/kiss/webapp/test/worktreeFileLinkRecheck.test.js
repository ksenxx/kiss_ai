// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const {
  makePage: makeChatPage,
  send,
  click,
  answerCheckPaths,
  stubRaf,
} = require('./simplify2_harness');
const {
  findLinks,
  findMissing,
  checkPathsOnRealFs,
} = require('./file_link_helpers');

// A real temp workspace; each test creates the files it links to.
const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-wtrecheck-'));

// makePage loads the real chat page (chat.html + api.js + main.js)
// in jsdom.  Every posted `checkPaths` command is answered with a
// `pathsExist` reply computed against the REAL filesystem, exactly like
// the remote web server does.
function makePage() {
  return makeChatPage({
    beforeScripts: win => answerCheckPaths(win, checkPathsOnRealFs(tmpDir)),
  });
}

// End-to-end tests of recheckFileLinksForTab(): a worktree_result event
// (merge or discard finished) must re-verify the tab's file links, so a
// report committed on the task's branch becomes clickable the moment
// the merge lands it in the checkout, and a link that resolved against
// the discarded worktree copy greys out.

function testMergeFlipsMissingResultLinkClickable() {
  const {win, posted} = makePage();
  const report = path.join(tmpDir, 'reports', 'analysis.html');
  try {
    // The task's result panel names the committed report, but the
    // branch is not merged yet: the file is not in the checkout.
    send(win, {
      type: 'result',
      success: true,
      summary: '<p>Committed at <code>' + report + '</code></p>',
      total_tokens: 1,
      cost: 'N/A',
    });
    assert.strictEqual(
      findLinks(win, report).length,
      0,
      'the un-merged report must not be clickable yet',
    );
    assert.strictEqual(
      findMissing(win, report).length,
      1,
      'the un-merged report must be marked missing',
    );

    // The merge lands the file, and the daemon reports it done.
    fs.mkdirSync(path.dirname(report), {recursive: true});
    fs.writeFileSync(report, '<h1>report</h1>\n');
    send(win, {
      type: 'worktree_result',
      success: true,
      message: "Merged branch 'kiss/wt-1'.",
    });

    const links = findLinks(win, report);
    assert.strictEqual(
      links.length,
      1,
      'the merged report must become clickable',
    );
    assert.strictEqual(
      findMissing(win, report).length,
      0,
      'the missing marker must be gone after the merge',
    );
    click(win, links[0]);
    const opens = posted.filter(m => m.type === 'openFile');
    assert.strictEqual(opens.length, 1, 'click must post one openFile');
    assert.strictEqual(opens[0].path, report);
  } finally {
    fs.rmSync(path.dirname(report), {recursive: true, force: true});
  }
  win.close();
  console.log('  ok - merge flips a missing result link to clickable');
}

function testDiscardDemotesLinkWhoseFileVanished() {
  const {win} = makePage();
  const fleeting = path.join(tmpDir, 'discarded.txt');
  fs.writeFileSync(fleeting, 'x\n');
  try {
    send(win, {type: 'prompt', text: 'wrote ' + fleeting});
    assert.strictEqual(
      findLinks(win, fleeting).length,
      1,
      'the file must be clickable while it exists',
    );
    // The discard deletes the only copy of the file.
    fs.rmSync(fleeting);
    send(win, {
      type: 'worktree_result',
      success: true,
      message: "Discarded branch 'kiss/wt-2'.",
    });
    assert.strictEqual(
      findLinks(win, fleeting).length,
      0,
      'the link must grey out once the discard removed the file',
    );
    assert.strictEqual(
      findMissing(win, fleeting).length,
      1,
      'the span must be marked missing after the discard',
    );
  } finally {
    fs.rmSync(fleeting, {force: true});
  }
  win.close();
  console.log('  ok - discard demotes a link whose file vanished');
}

function testFailedWorktreeResultDoesNotRecheck() {
  const {win} = makePage();
  const late = path.join(tmpDir, 'failed-merge.txt');
  try {
    send(win, {type: 'prompt', text: 'see ' + late});
    assert.strictEqual(findLinks(win, late).length, 0);
    fs.writeFileSync(late, 'x\n');
    send(win, {
      type: 'worktree_result',
      success: false,
      message: 'merge failed: conflict',
    });
    assert.strictEqual(
      findLinks(win, late).length,
      0,
      'a failed worktree action must not re-verify links',
    );
  } finally {
    fs.rmSync(late, {force: true});
  }
  win.close();
  console.log('  ok - a failed worktree_result does not recheck');
}

// 2026-11-04 W4 F1: the worktree's completion re-asks about a path
// whose first check is still unanswered (the daemon checks the file
// system off the loop).  The replies come in the order asked: the
// first is the older observation and must not settle the span for
// good -- the fresher one does.
function testOlderReplyDoesNotSettleARecheckedLink() {
  for (const kind of ['merge', 'discard']) {
    const {win, posted} = makeChatPage({beforeScripts: stubRaf});
    const tabId = win._testApi.getActiveTabId();
    const file = path.join(tmpDir, 'verdict-' + kind + '.txt');
    win._testApi.hideWelcome();
    send(win, {type: 'prompt', tabId, text: 'See ' + file});
    const checks = () =>
      posted.filter(m => m.type === 'checkPaths' && m.paths.includes(file));
    assert.strictEqual(checks().length, 1, 'the path is checked');
    send(win, {
      type: 'worktree_result',
      tabId,
      success: true,
      message: kind + ' done',
    });
    assert.strictEqual(checks().length, 2, 'the completion rechecks');
    const [old, fresh] = checks();
    // A merge lands the file (the old check saw none); a discard
    // removes the worktree copy (the old check saw it).
    const oldExists = kind === 'discard';
    const answer = (req, verdict) =>
      send(win, {
        type: 'pathsExist',
        tabId: req.tabId,
        workDir: req.workDir,
        results: {[file]: verdict},
      });
    answer(old, oldExists);
    answer(fresh, !oldExists);
    assert.strictEqual(
      findLinks(win, file).length,
      oldExists ? 0 : 1,
      kind + ': the fresher verdict decides the link',
    );
    assert.strictEqual(findMissing(win, file).length, oldExists ? 1 : 0);
    win.close();
  }
  console.log('  ok - an older reply does not settle a re-asked link');
}

try {
  testMergeFlipsMissingResultLinkClickable();
  testDiscardDemotesLinkWhoseFileVanished();
  testFailedWorktreeResultDoesNotRecheck();
  testOlderReplyDoesNotSettleARecheckedLink();
  console.log('worktreeFileLinkRecheck.test.js: all tests passed');
} finally {
  fs.rmSync(tmpDir, {recursive: true, force: true});
}
