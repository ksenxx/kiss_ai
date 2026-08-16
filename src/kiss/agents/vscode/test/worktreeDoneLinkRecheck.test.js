// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

// End-to-end tests: worktree_created / worktree_done must re-verify the
// tab's file links.  A result rendered while the host had no
// pending-worktree fallback (e.g. a reconnect replay dropped it) demotes
// the report link as missing; the worktree event is the moment the host
// records the fallback dir, so the webview must re-ask — otherwise the
// link stays dead until the user merges.

const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-wtdone-'));
const wtDir = path.join(tmpDir, '.kiss-worktrees', 'kiss_wt-1');

// Emulates the extension host's resolution: the span's workDir first,
// then the tab's recorded pending-worktree fallback dir.
const fallbackDirs = {};

function checkPathsLikeHost(msg) {
  const results = {};
  const roots = [msg.workDir || tmpDir];
  const fallback = fallbackDirs[String(msg.tabId)];
  if (fallback) roots.push(fallback);
  for (const p of msg.paths) {
    let ok = false;
    for (const root of roots) {
      const abs = path.isAbsolute(p) ? p : path.resolve(root, p);
      try {
        ok = fs.statSync(abs).isFile();
      } catch {
        ok = false;
      }
      if (ok) break;
    }
    results[p] = ok;
  }
  return results;
}

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
        if (msg.type === 'checkPaths') {
          send(win, {
            type: 'pathsExist',
            results: checkPathsLikeHost(msg),
            workDir: msg.workDir,
            tabId: msg.tabId,
          });
        }
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

function clickEl(win, el) {
  el.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

function testWorktreeDoneRevivesDemotedLink() {
  const {win, posted} = makeWebview();
  const rel = 'reports/live.html';
  const wtCopy = path.join(wtDir, rel);
  try {
    // The result names the report, but the host has no fallback for
    // the tab yet (a reconnect replay dropped it): the link demotes.
    send(win, {
      type: 'result',
      success: true,
      summary: '<p>Report at <code>' + rel + '</code></p>',
      total_tokens: 1,
      cost: 'N/A',
    });
    assert.strictEqual(findLinks(win, rel).length, 0);
    assert.strictEqual(
      findMissing(win, rel).length,
      1,
      'the report must be demoted while the fallback is unknown',
    );

    // worktree_done is the moment the host records the fallback.
    fs.mkdirSync(path.dirname(wtCopy), {recursive: true});
    fs.writeFileSync(wtCopy, '<h1>live</h1>\n');
    // The demoted span is stamped with its owning tab id.
    const tabId = findMissing(win, rel)[0].getAttribute('data-path-tab');
    fallbackDirs[String(tabId)] = wtDir;
    const checksBefore = posted.filter(m => m.type === 'checkPaths').length;
    send(win, {
      type: 'worktree_done',
      branch: 'kiss/wt-1',
      worktreeDir: wtDir,
      originalBranch: 'main',
      changedFiles: [rel],
      tabId,
    });

    const checksAfter = posted.filter(m => m.type === 'checkPaths').length;
    assert.ok(
      checksAfter > checksBefore,
      'worktree_done must re-issue checkPaths for the tab',
    );
    const links = findLinks(win, rel);
    assert.strictEqual(
      links.length,
      1,
      'the report must become clickable once the fallback is known',
    );
    assert.strictEqual(findMissing(win, rel).length, 0);
    clickEl(win, links[0]);
    const opens = posted.filter(m => m.type === 'openFile');
    assert.strictEqual(opens.length, 1, 'click must post one openFile');
    assert.strictEqual(opens[0].path, rel);
  } finally {
    delete fallbackDirs[Object.keys(fallbackDirs)[0]];
    fs.rmSync(wtDir, {recursive: true, force: true});
  }
  win.close();
  console.log('  ok - worktree_done revives a demoted report link');
}

function testWorktreeCreatedRechecksToo() {
  const {win, posted} = makeWebview();
  const rel = 'notes/plan.md';
  const wtCopy = path.join(wtDir, rel);
  try {
    send(win, {type: 'prompt', text: 'see ' + rel});
    assert.strictEqual(findLinks(win, rel).length, 0);

    fs.mkdirSync(path.dirname(wtCopy), {recursive: true});
    fs.writeFileSync(wtCopy, 'plan\n');
    // The demoted span is stamped with its owning tab id.
    const missingSpans = Array.from(
      win.document.querySelectorAll('#output [data-path-missing]'),
    ).filter(el => el.getAttribute('data-path-missing') === rel);
    const tabId = missingSpans[0].getAttribute('data-path-tab');
    fallbackDirs[String(tabId)] = wtDir;
    const checksBefore = posted.filter(m => m.type === 'checkPaths').length;
    send(win, {
      type: 'worktree_created',
      worktreeDir: wtDir,
      branch: 'kiss/wt-1',
      tabId,
    });
    const checksAfter = posted.filter(m => m.type === 'checkPaths').length;
    assert.ok(
      checksAfter > checksBefore,
      'worktree_created must re-issue checkPaths for the tab',
    );
    assert.strictEqual(
      findLinks(win, rel).length,
      1,
      'the link must resolve once the fallback is recorded',
    );
  } finally {
    delete fallbackDirs[Object.keys(fallbackDirs)[0]];
    fs.rmSync(wtDir, {recursive: true, force: true});
  }
  win.close();
  console.log('  ok - worktree_created rechecks links too');
}

try {
  testWorktreeDoneRevivesDemotedLink();
  testWorktreeCreatedRechecksToo();
  console.log('worktreeDoneLinkRecheck.test.js: all tests passed');
} finally {
  fs.rmSync(tmpDir, {recursive: true, force: true});
}
