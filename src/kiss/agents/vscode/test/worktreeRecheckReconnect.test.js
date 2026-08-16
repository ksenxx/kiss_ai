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
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-wtreconnect-'));

// makeWebview loads the real chat webview (chat.html + api.js + main.js)
// in jsdom. Replies are NOT automatic: the test answers each posted
// `checkPaths` explicitly so it can swallow the one a disconnect would
// swallow.
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
      postMessage: msg => {
        posted.push(msg);
      },
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

function checkPathsPosts(posted) {
  return posted.filter(m => m.type === 'checkPaths');
}

function replyOnRealFs(win, msg) {
  const results = {};
  for (const p of msg.paths) {
    let ok = false;
    try {
      ok = fs.statSync(p).isFile();
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

function findLinks(win, p) {
  return Array.from(
    win.document.querySelectorAll('#output [data-path]'),
  ).filter(el => el.dataset.path === p);
}

function findMissing(win, p) {
  return Array.from(
    win.document.querySelectorAll('#output [data-path-missing]'),
  ).filter(el => el.getAttribute('data-path-missing') === p);
}

function testReconnectReissuesPostWorktreeRecheck() {
  const {win, posted} = makeWebview();
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
  const {win, posted} = makeWebview();
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
