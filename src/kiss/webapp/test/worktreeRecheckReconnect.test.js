// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// End-to-end test of reissueFileLinkChecks() after a post-worktree
// recheck: when the daemon dies before answering the checkPaths that
// recheckFileLinksForTab() sent, the reconnect must ask about the
// affected spans again even though they carry data-path /
// data-path-missing (a resolved verdict) instead of
// data-path-candidate. Before the fix the reissue skipped them and the
// links stayed frozen in their pre-merge / pre-discard state forever.

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
// makePage loads the real chat page (chat.html + api.js + main.js)
// in jsdom. Replies are NOT automatic: the test answers each posted
// `checkPaths` explicitly so it can swallow the one a disconnect would
// swallow.
const {makePage, send} = require('./simplify2_harness');
const {findLinks, findMissing} = require('./file_link_helpers');

const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-wtreconnect-'));

function checkPathsPosts(posted) {
  return posted.filter(m => m.type === 'checkPaths');
}

function replyOnRealFs(win, msg) {
  const results = {};
  for (const p of msg.paths) {
    let ok = false;
    try {
      const st = fs.statSync(p);
      ok = st.isFile() || st.isDirectory();
    } catch {
      ok = false;
    }
    results[p] = ok;
  }
  send(win, {
    type: 'pathsExist',
    results,
    workDir: msg.workDir,
    tabId: msg.tabId,
  });
}

function testReconnectReissuesPostWorktreeRecheck() {
  const {win, posted} = makePage();
  const discarded = path.join(tmpDir, 'discarded.txt');
  const merged = path.join(tmpDir, 'reports', 'merged.html');
  fs.writeFileSync(discarded, 'x\n');
  try {
    // Two links resolve to opposite verdicts: `discarded` exists (the
    // span is promoted to data-path), `merged` does not (the span is
    // demoted to data-path-missing).
    send(win, {type: 'prompt', text: 'wrote ' + discarded + ' see ' + merged});
    assert.strictEqual(checkPathsPosts(posted).length, 1);
    replyOnRealFs(win, checkPathsPosts(posted)[0]);
    assert.strictEqual(findLinks(win, discarded).length, 1);
    assert.strictEqual(findMissing(win, merged).length, 1);

    // A discard removes one file while a merge lands the other, and the
    // worktree_result recheck goes out ...
    fs.rmSync(discarded);
    fs.mkdirSync(path.dirname(merged), {recursive: true});
    fs.writeFileSync(merged, '<h1>report</h1>\n');
    send(win, {
      type: 'worktree_result',
      success: true,
      message: "Merged branch 'kiss/wt-1'.",
    });
    assert.strictEqual(
      checkPathsPosts(posted).length,
      2,
      'worktree_result must send a recheck',
    );

    // ... but the daemon dies before answering it. The links still
    // show the stale pre-worktree verdicts.
    send(win, {type: 'daemonStatus', connected: false});
    assert.strictEqual(findLinks(win, discarded).length, 1);
    assert.strictEqual(findMissing(win, merged).length, 1);

    // The reconnect must ask about both resolved spans again.
    const before = checkPathsPosts(posted).length;
    send(win, {type: 'daemonStatus', connected: true});
    const reissued = checkPathsPosts(posted).slice(before);
    const asked = [];
    for (const msg of reissued) asked.push(...msg.paths);
    assert.ok(
      asked.includes(discarded),
      'the reconnect must re-ask about the promoted span',
    );
    assert.ok(
      asked.includes(merged),
      'the reconnect must re-ask about the missing span',
    );

    // Answering the reissued checks flips both links to the truth.
    for (const msg of reissued) replyOnRealFs(win, msg);
    assert.strictEqual(
      findLinks(win, discarded).length,
      0,
      'the discarded file must grey out after the reconnect',
    );
    assert.strictEqual(
      findMissing(win, discarded).length,
      1,
      'the discarded file must be marked missing after the reconnect',
    );
    assert.strictEqual(
      findLinks(win, merged).length,
      1,
      'the merged report must become clickable after the reconnect',
    );
  } finally {
    fs.rmSync(discarded, {force: true});
    fs.rmSync(path.dirname(merged), {recursive: true, force: true});
  }
  win.close();
  console.log('  ok - reconnect reissues the post-worktree recheck');
}

function testReconnectStillReissuesCandidateSpans() {
  const {win, posted} = makePage();
  const pending = path.join(tmpDir, 'pending.txt');
  fs.writeFileSync(pending, 'x\n');
  try {
    // The candidate's very first check is swallowed by the outage.
    send(win, {type: 'prompt', text: 'see ' + pending});
    assert.strictEqual(checkPathsPosts(posted).length, 1);
    send(win, {type: 'daemonStatus', connected: false});
    const before = checkPathsPosts(posted).length;
    send(win, {type: 'daemonStatus', connected: true});
    const reissued = checkPathsPosts(posted).slice(before);
    const asked = [];
    for (const msg of reissued) asked.push(...msg.paths);
    assert.ok(
      asked.includes(pending),
      'the reconnect must still re-ask about unresolved candidates',
    );
    for (const msg of reissued) replyOnRealFs(win, msg);
    assert.strictEqual(
      findLinks(win, pending).length,
      1,
      'the candidate must resolve from the reissued check',
    );
  } finally {
    fs.rmSync(pending, {force: true});
  }
  win.close();
  console.log('  ok - reconnect still reissues unresolved candidates');
}

try {
  testReconnectReissuesPostWorktreeRecheck();
  testReconnectStillReissuesCandidateSpans();
  console.log('worktreeRecheckReconnect.test.js: all tests passed');
} finally {
  fs.rmSync(tmpDir, {recursive: true, force: true});
}
