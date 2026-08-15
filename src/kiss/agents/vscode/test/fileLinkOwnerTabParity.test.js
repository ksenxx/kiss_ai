// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) test of the file-link OWNERSHIP invariant.
//
// linkifyFilePaths(root, workDir, ownerTabId) / verifyFileLinkCandidates
// exist so that a transcript rendered into a detached BACKGROUND
// fragment stamps its candidate spans with the tab that owns them --
// "ownerTabId names the tab whose transcript `root` belongs to. It is
// not always the active tab" (main.js). Nine of the eleven call sites
// thread the owner through; two dropped it and silently fell back to
// activeTabId, so a hidden tab's checkPaths went out under the VISIBLE
// tab's id and collapsed the per-tab dedup namespace into it.
//
// The two sites are the `path:` argument span of every Read/Write/Edit
// tool_call, and the summary body of the result panel.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');
const WORK_DIR = '/tmp/kiss-owner-parity';

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
  let state;
  win.acquireVsCodeApi = function () {
    return {
      postMessage: msg => posted.push(msg),
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

function clickTab(win, tabId) {
  const el = win.document.querySelector(
    `.chat-tab[data-tab-id=${JSON.stringify(tabId)}]`,
  );
  assert.ok(el, `tab ${tabId} must exist in the tab bar`);
  el.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

// The commands are built inside the JSDOM realm, so their `paths` array
// is a foreign Array; copy it into this realm before comparing.
function checkPathCommands(posted) {
  return posted
    .filter(m => m && m.type === 'checkPaths')
    .map(m => ({type: m.type, tabId: m.tabId, workDir: m.workDir,
      paths: Array.from(m.paths || [])}));
}

// Two tabs; the SECOND one is left on screen so the first is a genuine
// background tab whose transcript is rendered into a detached fragment.
function makeTwoTabs() {
  const {win, posted} = makeWebview();
  send(win, {type: 'configData', config: {work_dir: WORK_DIR}, apiKeys: {}});
  win._testApi.endLaunch();
  const tabA = win._testApi.getActiveTabId();
  win._testApi.createNewTab();
  const tabB = win._testApi.getActiveTabId();
  assert.notStrictEqual(tabA, tabB, 'a second tab must have been created');
  clickTab(win, tabA);
  assert.strictEqual(
    win._testApi.getActiveTabId(),
    tabA,
    'tab A must be the visible tab',
  );
  posted.length = 0;
  return {win, posted, visible: tabA, hidden: tabB};
}

function testToolCallPathSpanCarriesOwnerTab() {
  const {win, posted, visible, hidden} = makeTwoTabs();

  send(win, {
    type: 'tool_call',
    name: 'Read',
    path: 'src/foo.py',
    tabId: hidden,
    workDir: WORK_DIR,
  });

  const cmds = checkPathCommands(posted);
  assert.strictEqual(
    cmds.length,
    1,
    `exactly one checkPaths expected, got ${JSON.stringify(cmds)}`,
  );
  assert.deepStrictEqual(cmds[0].paths, ['src/foo.py']);
  assert.strictEqual(
    cmds[0].tabId,
    hidden,
    "the tool_call `path:` span belongs to the hidden tab, so its " +
      'checkPaths must name that tab — not the tab that happens to be ' +
      `on screen (${visible})`,
  );

  // The stamp on the span must agree, or the reply cannot resolve it.
  clickTab(win, hidden);
  const span = win.document.querySelector('#output [data-path-candidate]');
  assert.ok(span, 'the path argument must render a candidate span');
  assert.strictEqual(
    span.getAttribute('data-path-tab'),
    hidden,
    'the candidate span must be stamped with its owning tab',
  );

  // And the reply must still promote it: the fix must not make links inert.
  send(win, {
    type: 'pathsExist',
    tabId: hidden,
    workDir: WORK_DIR,
    results: {'src/foo.py': true},
  });
  assert.ok(
    win.document.querySelector('#output [data-path="src/foo.py"]'),
    'the pathsExist reply must promote the span to a clickable link',
  );
  win.close();
  console.log('  ok - tool_call path span checks under its own tab');
}

function testResultPanelSummaryCarriesOwnerTab() {
  const {win, posted, visible, hidden} = makeTwoTabs();

  send(win, {
    type: 'result',
    summary: 'Edited ./src/bar.py and moved on.',
    success: true,
    tabId: hidden,
    workDir: WORK_DIR,
  });

  const cmds = checkPathCommands(posted);
  assert.ok(cmds.length >= 1, 'the result summary must trigger a checkPaths');
  for (const cmd of cmds) {
    assert.strictEqual(
      cmd.tabId,
      hidden,
      "the result panel belongs to the hidden tab, so its checkPaths must " +
        `name that tab — not the visible one (${visible})`,
    );
  }
  assert.ok(
    cmds.some(c => c.paths.includes('./src/bar.py')),
    `the summary path must be checked, got ${JSON.stringify(cmds)}`,
  );

  clickTab(win, hidden);
  const span = win.document.querySelector('.rc-body [data-path-candidate]');
  assert.ok(span, 'the result summary must render a candidate span');
  assert.strictEqual(
    span.getAttribute('data-path-tab'),
    hidden,
    'the result summary span must be stamped with its owning tab',
  );

  send(win, {
    type: 'pathsExist',
    tabId: hidden,
    workDir: WORK_DIR,
    results: {'./src/bar.py': true},
  });
  assert.ok(
    win.document.querySelector('.rc-body [data-path="./src/bar.py"]'),
    'the pathsExist reply must promote the result summary span',
  );
  win.close();
  console.log('  ok - result panel summary checks under its own tab');
}

// Control: the same two paths rendered into the VISIBLE tab must keep
// naming the visible tab, so the fix is not just "always use ev.tabId".
function testVisibleTabStillNamesItself() {
  const {win, posted, visible} = makeTwoTabs();
  send(win, {
    type: 'tool_call',
    name: 'Read',
    path: 'src/foo.py',
    tabId: visible,
    workDir: WORK_DIR,
  });
  send(win, {
    type: 'result',
    summary: 'Edited ./src/bar.py and moved on.',
    success: true,
    tabId: visible,
    workDir: WORK_DIR,
  });
  const cmds = checkPathCommands(posted);
  assert.ok(cmds.length >= 2, 'both sites must post a checkPaths');
  for (const cmd of cmds) {
    assert.strictEqual(
      cmd.tabId,
      visible,
      'a transcript rendered into the visible tab must check under it',
    );
  }
  win.close();
  console.log('  ok - the visible tab still checks under its own id');
}

function main() {
  testToolCallPathSpanCarriesOwnerTab();
  testResultPanelSummaryCarriesOwnerTab();
  testVisibleTabStillNamesItself();
  console.log('fileLinkOwnerTabParity.test.js: all tests passed');
}

main();
