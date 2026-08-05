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
const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-filelink-'));
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

function testExistingPathIsClickable() {
  const {win, posted} = makeWebview();
  send(win, {type: 'prompt', text: 'see ' + realFile + ' for details'});
  const links = findLinks(win, realFile);
  assert.strictEqual(links.length, 1, 'existing path must become clickable');
  assert.ok(
    links[0].classList.contains('kiss-filelink'),
    'verified link must carry the kiss-filelink class',
  );
  clickEl(win, links[0]);
  const opens = posted.filter(m => m.type === 'openFile');
  assert.strictEqual(opens.length, 1, 'click must post one openFile');
  assert.strictEqual(opens[0].path, realFile);
  win.close();
  console.log('  ok - existing path is clickable and opens');
}

function testMissingPathIsNotClickable() {
  const {win, posted} = makeWebview();
  send(win, {
    type: 'prompt',
    text: 'compare ' + realFile + ' with ' + missingFile + ' now',
  });
  assert.strictEqual(
    findLinks(win, realFile).length,
    1,
    'existing path must be clickable',
  );
  assert.strictEqual(
    findLinks(win, missingFile).length,
    0,
    'missing path must NOT be clickable (no [data-path])',
  );
  const missing = Array.from(
    win.document.querySelectorAll('#output [data-path-missing]'),
  ).filter(el => el.textContent === missingFile);
  assert.strictEqual(
    missing.length,
    1,
    'missing path text must still be visible as plain text',
  );
  assert.ok(
    !missing[0].classList.contains('kiss-filelink'),
    'missing path must not be styled as a link',
  );
  clickEl(win, missing[0]);
  const opens = posted.filter(m => m.type === 'openFile');
  assert.strictEqual(
    opens.length,
    0,
    'clicking a missing path must NOT post openFile',
  );
  win.close();
  console.log('  ok - missing path is not clickable');
}

function testDirectoryPathIsNotClickable() {
  const {win} = makeWebview();
  const dirPath = path.join(tmpDir, 'somedir');
  send(win, {type: 'prompt', text: 'look in ' + dirPath + ' please'});
  assert.strictEqual(
    findLinks(win, dirPath).length,
    0,
    'directory path must NOT be clickable',
  );
  win.close();
  console.log('  ok - directory path is not clickable');
}

function testLineSuffixCheckedAgainstStrippedPath() {
  const {win, posted} = makeWebview();
  send(win, {type: 'prompt', text: 'crash at ' + realFile + ':42 today'});
  const checks = posted.filter(m => m.type === 'checkPaths');
  assert.strictEqual(checks.length, 1, 'one checkPaths must be posted');
  assert.deepStrictEqual(
    Array.from(checks[0].paths),
    [realFile],
    'checkPaths must send the path with the :line suffix stripped',
  );
  const links = findLinks(win, realFile + ':42');
  assert.strictEqual(links.length, 1, 'path:line must be clickable');
  clickEl(win, links[0]);
  const opens = posted.filter(m => m.type === 'openFile');
  assert.strictEqual(opens.length, 1);
  assert.strictEqual(opens[0].path, realFile);
  assert.strictEqual(opens[0].line, 42, 'line must be parsed on click');
  win.close();
  console.log('  ok - :line suffix stripped for check, kept for open');
}

function testMissingPathWithLineSuffixNotClickable() {
  const {win} = makeWebview();
  send(win, {
    type: 'tool_call',
    name: 'Read',
    path: realFile,
  });
  send(win, {
    type: 'tool_result',
    content: 'no such file: ' + missingFile + ':7',
    is_error: true,
  });
  assert.strictEqual(
    findLinks(win, missingFile + ':7').length,
    0,
    'missing path with :line in tool error must NOT be clickable',
  );
  win.close();
  console.log('  ok - missing path with :line in tool error not clickable');
}

function testCheckPathsDeduplicatesWithinOneMessage() {
  const {win, posted} = makeWebview();
  send(win, {
    type: 'prompt',
    text: 'open ' + realFile + ' twice ' + realFile,
  });
  const checks = posted.filter(m => m.type === 'checkPaths');
  assert.strictEqual(checks.length, 1, 'one checkPaths for one panel');
  const occurrences = checks[0].paths.filter(p => p === realFile);
  assert.strictEqual(
    occurrences.length,
    1,
    'the same path must be checked only once per message',
  );
  assert.strictEqual(
    findLinks(win, realFile).length,
    2,
    'both occurrences must become clickable',
  );
  win.close();
  console.log('  ok - checkPaths dedupes paths within a message');
}

function testFileCreatedMidRunBecomesClickableInNewPanel() {
  // Existence results must NOT be cached forever: a file the agent
  // creates mid-run must be clickable in panels rendered afterwards.
  const {win, posted} = makeWebview();
  const lateFile = path.join(tmpDir, 'late.txt');
  try {
    send(win, {type: 'prompt', text: 'will write ' + lateFile + ' soon'});
    assert.strictEqual(
      findLinks(win, lateFile).length,
      0,
      'path must not be clickable before the file exists',
    );
    fs.writeFileSync(lateFile, 'created later\n');
    send(win, {type: 'prompt', text: 'wrote ' + lateFile + ' now'});
    const checks = posted.filter(
      m => m.type === 'checkPaths' && m.paths.indexOf(lateFile) >= 0,
    );
    assert.strictEqual(
      checks.length,
      2,
      'a NEW panel must re-check existence (no stale cache)',
    );
    assert.strictEqual(
      findLinks(win, lateFile).length,
      1,
      'file created mid-run must be clickable in the new panel',
    );
  } finally {
    fs.rmSync(lateFile, {force: true});
  }
  win.close();
  console.log('  ok - file created mid-run becomes clickable in new panels');
}

function testFileDeletedMidRunNotClickableInNewPanel() {
  const {win} = makeWebview();
  const fleeting = path.join(tmpDir, 'fleeting.txt');
  fs.writeFileSync(fleeting, 'x\n');
  try {
    send(win, {type: 'prompt', text: 'first ' + fleeting});
    assert.strictEqual(
      findLinks(win, fleeting).length,
      1,
      'existing file must be clickable at first',
    );
    fs.rmSync(fleeting);
    send(win, {type: 'prompt', text: 'again ' + fleeting});
    assert.strictEqual(
      findLinks(win, fleeting).length,
      1,
      'only the pre-deletion panel keeps its link',
    );
    const missing = Array.from(
      win.document.querySelectorAll('#output [data-path-missing]'),
    ).filter(el => el.textContent === fleeting);
    assert.strictEqual(
      missing.length,
      1,
      'file deleted mid-run must render as plain text in the new panel',
    );
  } finally {
    fs.rmSync(fleeting, {force: true});
  }
  win.close();
  console.log('  ok - file deleted mid-run is not clickable in new panels');
}

function testUnverifiedPathIsNotClickable() {
  // No pathsExist reply ever arrives (e.g. host lost the message):
  // the path must stay non-clickable rather than optimistically open.
  const {win, posted} = makeWebview({autoReply: false});
  send(win, {type: 'prompt', text: 'open ' + realFile + ' now'});
  assert.strictEqual(
    findLinks(win, realFile).length,
    0,
    'unverified path must NOT be clickable before the pathsExist reply',
  );
  const cand = Array.from(
    win.document.querySelectorAll('#output [data-path-candidate]'),
  ).filter(el => el.dataset.pathCandidate === realFile);
  assert.strictEqual(cand.length, 1, 'candidate span must hold the text');
  clickEl(win, cand[0]);
  assert.strictEqual(
    posted.filter(m => m.type === 'openFile').length,
    0,
    'clicking an unverified path must NOT post openFile',
  );
  win.close();
  console.log('  ok - unverified path is not clickable');
}

function testLateReplyPromotesCandidates() {
  const {win, posted} = makeWebview({autoReply: false});
  send(win, {type: 'prompt', text: 'open ' + realFile + ' now'});
  const checks = posted.filter(m => m.type === 'checkPaths');
  assert.strictEqual(checks.length, 1);
  // Deliver the reply later, as the real host does asynchronously.
  send(win, {
    type: 'pathsExist',
    results: checkPathsOnRealFs(checks[0]),
    workDir: checks[0].workDir,
    tabId: checks[0].tabId,
  });
  const links = findLinks(win, realFile);
  assert.strictEqual(
    links.length,
    1,
    'late pathsExist reply must promote the candidate to a link',
  );
  clickEl(win, links[0]);
  assert.strictEqual(
    posted.filter(m => m.type === 'openFile').length,
    1,
    'promoted link must open on click',
  );
  win.close();
  console.log('  ok - late pathsExist reply promotes candidates');
}

function testPendingPathNotResentWhileAwaitingReply() {
  const {win, posted} = makeWebview({autoReply: false});
  send(win, {type: 'prompt', text: 'open ' + realFile + ' now'});
  send(win, {type: 'prompt', text: 'still ' + realFile + ' here'});
  const checks = posted.filter(
    m => m.type === 'checkPaths' && m.paths.indexOf(realFile) >= 0,
  );
  assert.strictEqual(
    checks.length,
    1,
    'a pending path must not be re-sent while its reply is in flight',
  );
  win.close();
  console.log('  ok - pending paths are not re-sent');
}

function testPartialReplyLeavesOtherPathsPending() {
  const {win, posted} = makeWebview({autoReply: false});
  send(win, {type: 'prompt', text: 'see ' + realFile + ' and ' + missingFile});
  // A reply about unrelated paths must not resolve these candidates.
  send(win, {
    type: 'pathsExist',
    results: {'/unrelated/other.txt': true},
    workDir: '',
  });
  assert.strictEqual(findLinks(win, realFile).length, 0);
  // The real reply then resolves both candidates.
  const checks = posted.filter(m => m.type === 'checkPaths');
  send(win, {
    type: 'pathsExist',
    results: checkPathsOnRealFs(checks[0]),
    workDir: checks[0].workDir,
  });
  assert.strictEqual(findLinks(win, realFile).length, 1);
  assert.strictEqual(findLinks(win, missingFile).length, 0);
  win.close();
  console.log('  ok - partial replies leave unrelated candidates pending');
}

function testMalformedPathsExistIgnored() {
  const {win} = makeWebview({autoReply: false});
  send(win, {type: 'prompt', text: 'open ' + realFile + ' now'});
  send(win, {type: 'pathsExist'});
  send(win, {type: 'pathsExist', results: null});
  send(win, {type: 'pathsExist', results: 'bogus'});
  send(win, {type: 'pathsExist', results: {}, workDir: 123});
  assert.strictEqual(
    findLinks(win, realFile).length,
    0,
    'malformed pathsExist replies must be ignored',
  );
  win.close();
  console.log('  ok - malformed pathsExist replies are ignored');
}

function testLinkifierIdempotentOnCandidatesAndMissing() {
  const {win} = makeWebview();
  // Render the same panel content twice: missing paths stay demoted and
  // no nested candidate/link spans are created.
  send(win, {
    type: 'result',
    summary: 'see ' + realFile + ' and ' + missingFile,
    success: true,
  });
  send(win, {
    type: 'prompt',
    text: 'see ' + realFile + ' and ' + missingFile,
  });
  for (const el of win.document.querySelectorAll(
    '#output [data-path], #output [data-path-candidate], ' +
      '#output [data-path-missing]',
  )) {
    assert.strictEqual(
      el.querySelectorAll(
        '[data-path], [data-path-candidate], [data-path-missing]',
      ).length,
      0,
      'no nested filelink spans after repeated linkify passes',
    );
  }
  assert.strictEqual(findLinks(win, missingFile).length, 0);
  win.close();
  console.log('  ok - linkifier is idempotent with existence checks');
}

function testToolCallExistingPathArgIsClickable() {
  const {win, posted} = makeWebview();
  send(win, {type: 'tool_call', name: 'Read', path: realFile});
  const links = findLinks(win, realFile);
  assert.strictEqual(
    links.length,
    1,
    'existing tool_call path arg must be clickable',
  );
  clickEl(win, links[0]);
  assert.ok(
    posted.some(m => m.type === 'openFile' && m.path === realFile),
    'click on existing tool_call path arg must post openFile',
  );
  win.close();
  console.log('  ok - existing tool_call path arg is clickable');
}

function testToolCallMissingPathArgNotClickable() {
  // The direct tool_call path hook must go through the same existence
  // gate as linkified text: nonexistent paths stay plain text.
  const {win, posted} = makeWebview();
  send(win, {type: 'tool_call', name: 'Read', path: missingFile});
  assert.strictEqual(
    findLinks(win, missingFile).length,
    0,
    'missing tool_call path arg must NOT be clickable',
  );
  const spans = Array.from(
    win.document.querySelectorAll('#output .tp[data-path-missing]'),
  ).filter(el => el.textContent === missingFile);
  assert.strictEqual(spans.length, 1, 'path arg text must stay visible');
  clickEl(win, spans[0]);
  assert.strictEqual(
    posted.filter(m => m.type === 'openFile').length,
    0,
    'clicking a missing tool_call path arg must NOT post openFile',
  );
  win.close();
  console.log('  ok - missing tool_call path arg is not clickable');
}

function testWorkDirMismatchedReplyDoesNotResolve() {
  // A pathsExist reply is only valid for spans checked under the SAME
  // workDir; a reply for another workDir must not flip candidates.
  const {win, posted} = makeWebview({autoReply: false});
  send(win, {type: 'prompt', text: 'open ' + realFile + ' now'});
  const checks = posted.filter(m => m.type === 'checkPaths');
  assert.strictEqual(checks.length, 1);
  const results = {};
  results[realFile] = true;
  send(win, {
    type: 'pathsExist',
    results,
    workDir: '/somewhere/else',
  });
  assert.strictEqual(
    findLinks(win, realFile).length,
    0,
    'reply correlated to a different workDir must not promote spans',
  );
  send(win, {
    type: 'pathsExist',
    results,
    workDir: checks[0].workDir,
  });
  assert.strictEqual(
    findLinks(win, realFile).length,
    1,
    'reply for the matching workDir must promote the span',
  );
  win.close();
  console.log('  ok - pathsExist replies are correlated by workDir');
}

function testReplayResolvesAgainstOwnerTabWorkDir() {
  // Replayed task events must verify paths against the workDir of the
  // tab that owns them (from task extras), not the default workDir.
  const wd2 = path.join(tmpDir, 'wd2');
  fs.mkdirSync(path.join(wd2, 'sub'), {recursive: true});
  fs.writeFileSync(path.join(wd2, 'sub', 'rel.txt'), 'x\n');
  const {win, posted} = makeWebview();
  send(win, {
    type: 'task_events',
    task: 'replayed task',
    extra: JSON.stringify({work_dir: wd2}),
    events: [
      {
        type: 'result',
        summary: 'edited sub/rel.txt and sub/gone.txt',
        success: true,
        total_tokens: 1,
        cost: '$0.01',
      },
    ],
  });
  const checks = posted.filter(
    m => m.type === 'checkPaths' && m.paths.indexOf('sub/rel.txt') >= 0,
  );
  assert.ok(checks.length >= 1, 'replayed panel must be verified');
  for (const c of checks) {
    assert.strictEqual(
      c.workDir,
      wd2,
      'replayed panels must check against the owner tab workDir',
    );
  }
  assert.strictEqual(
    findLinks(win, 'sub/rel.txt').length,
    1,
    'relative path existing under the tab workDir must be clickable',
  );
  assert.strictEqual(
    findLinks(win, 'sub/gone.txt').length,
    0,
    'relative path missing under the tab workDir must not be clickable',
  );
  win.close();
  console.log('  ok - replayed events resolve against owner tab workDir');
}

function testBackgroundTabResolvesAgainstItsOwnWorkDir() {
  // Events replayed into a NON-active tab must use that tab's workDir,
  // not the active tab's.
  const wd3 = path.join(tmpDir, 'wd3');
  fs.mkdirSync(path.join(wd3, 'bgsub'), {recursive: true});
  fs.writeFileSync(path.join(wd3, 'bgsub', 'bg.txt'), 'x\n');
  const {win, posted} = makeWebview();
  const doc = win.document;
  const tabIdsBefore = Array.from(
    doc.querySelectorAll('.chat-tab[data-tab-id]'),
  ).map(el => el.dataset.tabId);
  const addBtn = doc.querySelector('.chat-tab-add');
  assert.ok(addBtn, 'tab add button must exist');
  clickEl(win, addBtn);
  const tabIdsAfter = Array.from(
    doc.querySelectorAll('.chat-tab[data-tab-id]'),
  ).map(el => el.dataset.tabId);
  const tab2Id = tabIdsAfter.filter(id => tabIdsBefore.indexOf(id) < 0)[0];
  assert.ok(tab2Id, 'a second tab must have been created');
  const tab1Id = tabIdsBefore[0];
  const tab1El = doc.querySelector('.chat-tab[data-tab-id="' + tab1Id + '"]');
  clickEl(win, tab1El);
  send(win, {
    type: 'task_events',
    tabId: tab2Id,
    task: 'background task',
    extra: JSON.stringify({work_dir: wd3}),
    events: [
      {
        type: 'result',
        summary: 'wrote bgsub/bg.txt and bgsub/none.txt',
        success: true,
        total_tokens: 1,
        cost: '$0.01',
      },
    ],
  });
  const checks = posted.filter(
    m => m.type === 'checkPaths' && m.paths.indexOf('bgsub/bg.txt') >= 0,
  );
  assert.ok(checks.length >= 1, 'background tab panel must be verified');
  for (const c of checks) {
    assert.strictEqual(
      c.workDir,
      wd3,
      'background tab panels must check against THEIR tab workDir',
    );
  }
  // Switch to the background tab: its replayed panel must show the
  // existing relative path as clickable and the missing one as text.
  const tab2El = doc.querySelector('.chat-tab[data-tab-id="' + tab2Id + '"]');
  clickEl(win, tab2El);
  assert.strictEqual(findLinks(win, 'bgsub/bg.txt').length, 1);
  assert.strictEqual(findLinks(win, 'bgsub/none.txt').length, 0);
  win.close();
  console.log('  ok - background tab panels resolve against their workDir');
}

function runTests() {
  testExistingPathIsClickable();
  testMissingPathIsNotClickable();
  testDirectoryPathIsNotClickable();
  testLineSuffixCheckedAgainstStrippedPath();
  testMissingPathWithLineSuffixNotClickable();
  testCheckPathsDeduplicatesWithinOneMessage();
  testFileCreatedMidRunBecomesClickableInNewPanel();
  testFileDeletedMidRunNotClickableInNewPanel();
  testUnverifiedPathIsNotClickable();
  testLateReplyPromotesCandidates();
  testPendingPathNotResentWhileAwaitingReply();
  testPartialReplyLeavesOtherPathsPending();
  testMalformedPathsExistIgnored();
  testLinkifierIdempotentOnCandidatesAndMissing();
  testToolCallExistingPathArgIsClickable();
  testToolCallMissingPathArgNotClickable();
  testWorkDirMismatchedReplyDoesNotResolve();
  testReplayResolvesAgainstOwnerTabWorkDir();
  testBackgroundTabResolvesAgainstItsOwnWorkDir();
}

try {
  runTests();
  console.log('clickFilePathExistsOnly.test.js: all tests passed');
} finally {
  fs.rmSync(tmpDir, {recursive: true, force: true});
}
