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

// A real temp workspace with one real file and no "missing" file.
const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-wtrecheck-'));
const realFile = path.join(tmpDir, 'real.txt');
fs.writeFileSync(realFile, 'hello\n');
const missingFile = path.join(tmpDir, 'missing.txt');
fs.mkdirSync(path.join(tmpDir, 'somedir'));

function checkPathsOnRealFs(msg) {
  const results = {};
  for (const p of msg.paths) {
    let abs = p;
    if (p.startsWith('~/')) {
      abs = path.join(os.homedir(), p.slice(2));
    } else if (!path.isAbsolute(p)) {
      abs = path.resolve(msg.workDir || tmpDir, p);
    }
    let ok = false;
    try {
      ok = fs.statSync(abs).isFile();
    } catch {
      ok = false;
    }
    results[p] = ok;
  }
  return results;
}

// makeWebview loads the real chat webview (chat.html + api.js + main.js)
// in jsdom.  When autoReply is true, every posted `checkPaths` command is
// answered with a `pathsExist` reply computed against the REAL filesystem,
// exactly like the extension host / remote web server do.
function makeWebview(opts) {
  const autoReply = !opts || opts.autoReply !== false;
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
        if (autoReply && msg.type === 'checkPaths') {
          send(win, {
            type: 'pathsExist',
            results: checkPathsOnRealFs(msg),
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

function clickEl(win, el) {
  el.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}


// End-to-end tests of recheckFileLinksForTab(): a worktree_result event
// (merge or discard finished) must re-verify the tab's file links, so a
// report committed on the task's branch becomes clickable the moment
// the merge lands it in the checkout, and a link that resolved against
// the discarded worktree copy greys out.

function findMissing(win, p) {
  return Array.from(
    win.document.querySelectorAll('#output [data-path-missing]'),
  ).filter(el => el.getAttribute('data-path-missing') === p);
}

function testMergeFlipsMissingResultLinkClickable() {
  const {win, posted} = makeWebview();
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
    clickEl(win, links[0]);
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
  const {win} = makeWebview();
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
  const {win} = makeWebview();
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

try {
  testMergeFlipsMissingResultLinkClickable();
  testDiscardDemotesLinkWhoseFileVanished();
  testFailedWorktreeResultDoesNotRecheck();
  console.log('worktreeFileLinkRecheck.test.js: all tests passed');
} finally {
  fs.rmSync(tmpDir, {recursive: true, force: true});
}
