// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Regression test: absolute file paths in an `ls -l` listing inside a
// result panel's <pre><code> block must be clickable even when
// highlight.js scatters each path over several token spans.
//
// A GNU `ls -ld "$PWD"/*` listing ("-rw-rw-r--  1 ksen ksen  71830
// Sep 15 07:29 /home/ksen/kiss/API.md") auto-detects as Swift, whose
// grammar tokenizes "/home/" and "/kiss/" as regexp literals.  The
// path then lives in five text nodes, linkifyFilePaths() only ever saw
// the fragments, and the reader got an inert path (plus a useless
// "/home/" link).  The whole path must be one link, and the block must
// stay highlighted.

'use strict';

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

// Fixture tree on disk, addressed through the virtual prefix
// VIRTUAL_ROOT ("/home/tester/kiss") that the real listing used: hljs
// auto-detection is content-driven, and it is the "/home/<user>/..."
// listing that scores as Swift and gets its paths tokenized.  The fake
// host maps VIRTUAL_ROOT onto the tmp tree and stats the real files,
// so existence verdicts still come from the filesystem.
const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-splitpath-'));
const VIRTUAL_ROOT = '/home/tester/kiss';
const apiFile = VIRTUAL_ROOT + '/API.md';
const readmeFile = VIRTUAL_ROOT + '/README.md';
const dockerFile = VIRTUAL_ROOT + '/Dockerfile';
const cacheDir = VIRTUAL_ROOT + '/__pycache__';
const missingFile = VIRTUAL_ROOT + '/missing.lock';

function realPath(virtual) {
  return path.join(tmpDir, virtual.slice(VIRTUAL_ROOT.length));
}

fs.writeFileSync(realPath(apiFile), 'api\n');
fs.writeFileSync(realPath(readmeFile), 'readme\n');
fs.writeFileSync(realPath(dockerFile), 'FROM scratch\n');
fs.mkdirSync(realPath(cacheDir));

function checkPathsOnRealFs(msg) {
  const results = {};
  for (const p of msg.paths) {
    let ok = false;
    if (p.startsWith(VIRTUAL_ROOT + '/')) {
      try {
        const st = fs.statSync(realPath(p));
        ok = st.isFile() || st.isDirectory();
      } catch {
        ok = false;
      }
    }
    results[p] = ok;
  }
  return results;
}

// Loads the real chat webview (chat.html + api.js + main.js) in jsdom
// WITH the real highlight.js, answering every checkPaths against the
// real filesystem exactly like the extension host / web server do.
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
  win.eval(fs.readFileSync(path.join(MEDIA, 'highlight.min.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'marked.min.js'), 'utf8'));
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

function allLinkPaths(win) {
  return Array.from(win.document.querySelectorAll('#output [data-path]')).map(
    el => el.dataset.path,
  );
}

function clickEl(win, el) {
  el.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

// The exact shape of the GNU `ls -ld "$PWD"/*` listing that rendered
// with inert paths: no "@" after the mode, two-space column padding,
// "Mon DD HH:MM" dates.
function lsListing() {
  return (
    '-rw-rw-r--  1 ksen ksen  71830 Sep 15 07:29 ' +
    apiFile +
    '\n-rw-rw-r--  1 ksen ksen   2639 Sep  1 18:25 ' +
    dockerFile +
    '\n-rw-rw-r--  1 ksen ksen  49260 Sep 15 23:23 ' +
    readmeFile +
    '\ndrwxr-xr-x  2 ksen ksen   4096 Sep 12 01:18 ' +
    cacheDir +
    '\n-rw-rw-r--  1 ksen ksen 662614 Sep 15 03:58 ' +
    missingFile +
    '\n'
  );
}

function summaryHtml() {
  return (
    '<h3>Output of <code>ls -ld "$PWD"/*</code></h3>' +
    '<pre><code>' +
    lsListing() +
    '</code></pre>' +
    '<p><strong>5 entries</strong>, each shown with its full path.</p>'
  );
}

function assertListingLinks(win, label) {
  const code = win.document.querySelector('#output .rc-body pre code');
  assert.ok(code, label + ': result code block must exist');
  assert.ok(
    code.classList.contains('hljs'),
    label + ': the block must still be highlighted by hljs',
  );
  // The listing auto-detects as Swift; that grammar is what tokenizes
  // "/home/" and "/kiss/" as regexp literals and splits the paths.  If
  // a highlight.js upgrade changes the verdict this fixture no longer
  // exercises the split and must be re-tuned, so pin it.
  assert.ok(
    code.classList.contains('language-swift'),
    label + ': fixture must auto-detect as Swift, got ' + code.className,
  );
  assert.ok(
    code.querySelector('span[class^="hljs-"]'),
    label + ': hljs token spans must survive (highlighting is kept)',
  );
  assert.strictEqual(
    code.textContent,
    lsListing(),
    label + ': merging split paths must not alter the listing text',
  );
  for (const p of [apiFile, dockerFile, readmeFile, cacheDir]) {
    const links = findLinks(win, p);
    assert.strictEqual(
      links.length,
      1,
      label + ': ' + p + ' must be exactly one clickable link',
    );
    assert.strictEqual(
      links[0].textContent,
      p,
      label + ': the link text must be the whole path',
    );
    assert.ok(links[0].closest('pre'), label + ': link lives in the block');
  }
  assert.strictEqual(
    findLinks(win, missingFile).length,
    0,
    label + ': a missing path must NOT be clickable',
  );
  const partial = allLinkPaths(win).filter(
    p => p.endsWith('/') || !p.startsWith(VIRTUAL_ROOT + '/'),
  );
  assert.deepStrictEqual(
    partial,
    [],
    label + ': no path-fragment links such as "/home/" or "/kiss/"',
  );
  assert.strictEqual(
    win.document.querySelectorAll('#output [data-path-candidate]').length,
    0,
    label + ': no unresolved candidate spans may remain',
  );
}

function testLiveLsListingPathsClickable() {
  const {win, posted} = makeWebview();
  send(win, {type: 'result', summary: summaryHtml(), success: true});
  assertListingLinks(win, 'live');
  clickEl(win, findLinks(win, readmeFile)[0]);
  const opens = posted.filter(m => m.type === 'openFile');
  assert.strictEqual(opens.length, 1, 'live: link must open on click');
  assert.strictEqual(opens[0].path, readmeFile);
  win.close();
  console.log('  ok - live ls listing paths are whole clickable links');
}

function testReplayedLsListingPathsClickable() {
  const {win, posted} = makeWebview();
  send(win, {
    type: 'task_events',
    task: 'replayed task',
    extra: JSON.stringify({work_dir: VIRTUAL_ROOT}),
    events: [
      {
        type: 'result',
        summary: summaryHtml(),
        success: true,
        total_tokens: 1,
        cost: '$0.01',
      },
    ],
  });
  assertListingLinks(win, 'replay');
  clickEl(win, findLinks(win, cacheDir)[0]);
  const opens = posted.filter(m => m.type === 'openFile');
  assert.strictEqual(opens.length, 1, 'replay: directory link must open');
  assert.strictEqual(opens[0].path, cacheDir);
  win.close();
  console.log('  ok - replayed ls listing paths are whole clickable links');
}

function testCollapsedTextPanelExpandLinksWholePaths() {
  // Deferred highlighting (collapsed panel expanded later) goes through
  // highlightPending -> highlightBlockPreservingLinks as well.
  const {win} = makeWebview();
  send(win, {
    type: 'task_events',
    task: 'replayed task',
    extra: JSON.stringify({work_dir: VIRTUAL_ROOT}),
    events: [
      {
        type: 'text_delta',
        text: 'listing:\n```\n' + lsListing() + '```\n',
      },
      {type: 'text_end'},
      {
        type: 'result',
        summary: '<p>done</p>',
        success: true,
        total_tokens: 1,
        cost: '$0.01',
      },
    ],
  });
  const collapsed = Array.from(
    win.document.querySelectorAll('#output .collapsible.collapsed'),
  );
  assert.ok(collapsed.length >= 1, 'replay must collapse non-result panels');
  for (const panel of collapsed) {
    const header = panel.querySelector('.collapse-header');
    if (header) clickEl(win, header);
  }
  const links = findLinks(win, apiFile);
  assert.strictEqual(
    links.length,
    1,
    'expanded text panel: the whole path must be one link',
  );
  assert.strictEqual(links[0].textContent, apiFile);
  assert.ok(links[0].closest('.collapsible'));
  win.close();
  console.log('  ok - expanded collapsed panel links whole paths');
}

function testUnsplitPathsUntouched() {
  // A block hljs leaves intact (no token boundary inside the path) must
  // link the same way as before; the merge step is a no-op there.
  const {win} = makeWebview();
  send(win, {
    type: 'result',
    summary: '<pre><code>' + apiFile + '\n' + missingFile + '\n</code></pre>',
    success: true,
  });
  assert.strictEqual(findLinks(win, apiFile).length, 1);
  assert.strictEqual(findLinks(win, missingFile).length, 0);
  const code = win.document.querySelector('#output .rc-body pre code');
  assert.strictEqual(code.textContent, apiFile + '\n' + missingFile + '\n');
  win.close();
  console.log('  ok - unsplit paths link as before');
}

function testShellVariablePathsKeepHighlighting() {
  // "$HOME/kiss" is a shell expansion, not a path named HOME/kiss.  The
  // whole-text scan must not treat it as one, or the merge would pull
  // "HOME" out of its hljs-variable span and strip the variable colour
  // from every shell snippet; nor may "HOME/kiss" become a candidate.
  const {win, posted} = makeWebview();
  send(win, {
    type: 'result',
    summary:
      '<pre><code class="language-bash">cd $HOME/kiss &amp;&amp; ls $PWD/src\n' +
      'export P=$HOME/bin:$PATH\n' +
      'cat ' +
      apiFile +
      '\n</code></pre>',
    success: true,
  });
  const code = win.document.querySelector('#output .rc-body pre code');
  const vars = Array.from(code.querySelectorAll('.hljs-variable')).map(
    el => el.textContent,
  );
  assert.deepStrictEqual(
    vars,
    ['$HOME', '$PWD', '$HOME', '$PATH'],
    'shell variables must keep their whole hljs-variable span',
  );
  const checked = [];
  for (const m of posted) {
    if (m.type === 'checkPaths') checked.push(...m.paths);
  }
  assert.deepStrictEqual(
    checked.filter(p => /^(HOME|PWD)\//.test(p)),
    [],
    'HOME/kiss, PWD/src, HOME/bin must not be path candidates',
  );
  assert.strictEqual(findLinks(win, apiFile).length, 1);
  win.close();
  console.log('  ok - $VAR/path keeps hljs variable spans, no bogus links');
}

function runTests() {
  testLiveLsListingPathsClickable();
  testReplayedLsListingPathsClickable();
  testCollapsedTextPanelExpandLinksWholePaths();
  testUnsplitPathsUntouched();
  testShellVariablePathsKeepHighlighting();
}

try {
  runTests();
  console.log('highlightSplitPathFileLinks.test.js: all tests passed');
} finally {
  fs.rmSync(tmpDir, {recursive: true, force: true});
}
