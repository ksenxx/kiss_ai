// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM + compiled host) regression tests for a late
// "Working directory" reply landing in another tab's panel.
//
// Bug: tab A asks the VS Code host for a folder (openWorkDir /
// pickWorkDir).  Before the answer arrives the user closes the panel,
// switches to tab B and opens B's panel, typing a path there.  A's
// `workDirPicked` pinned A correctly but also closed B's panel (losing
// the typed path), and A's `workDirError` was shown in B's panel.
//
// Fixed behaviour: a reply touches the panel only when it answers the
// request the open panel is waiting on; a `workDirError` without a
// `tabId` (an older host build) keeps the previous behaviour.  The host
// stamps `workDirError` with the requesting tab, like `workDirPicked`.

/* global require, __dirname, console, process, global */

'use strict';

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const Module = require('module');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

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
  send(win, {type: 'configData', config: {work_dir: '/work/ws'}});
  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function click(win, el) {
  el.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

function byId(win, id) {
  const node = win.document.getElementById(id);
  assert.ok(node, `#${id} must exist`);
  return node;
}

function typeInto(win, id, value) {
  const node = byId(win, id);
  node.value = value;
  node.dispatchEvent(new win.Event('input', {bubbles: true}));
}

function pressEnter(win, id) {
  byId(win, id).dispatchEvent(
    new win.KeyboardEvent('keydown', {key: 'Enter', bubbles: true}),
  );
}

function msgs(posted, type) {
  return posted.filter(m => m && m.type === type);
}

function panelOpen(win) {
  return byId(win, 'workdir-panel').classList.contains('open');
}

function errorText(win) {
  const el = byId(win, 'workdir-error');
  return el.hidden ? '' : el.textContent;
}

function openPanelViaMenu(win) {
  click(win, byId(win, 'more-btn'));
  click(win, byId(win, 'workdir-btn'));
  assert.ok(panelOpen(win), 'the menu item opens the panel');
}

function switchToTab(win, tabId) {
  click(win, win.document.querySelector(`.chat-tab[data-tab-id="${tabId}"]`));
  assert.strictEqual(win._testApi.getActiveTabId(), tabId);
}

function submitPrompt(win, posted, text) {
  const before = msgs(posted, 'submit').length;
  byId(win, 'task-input').value = text;
  click(win, byId(win, 'send-btn'));
  const sent = msgs(posted, 'submit');
  assert.strictEqual(sent.length, before + 1, 'one submit is posted');
  return sent[sent.length - 1];
}

/**
 * Tab A asks for a folder (through *ask*), the user closes the panel,
 * creates tab B and opens B's panel with a path typed into it.
 * Returns the two tab ids.
 */
function askForAThenOpenBsPanel(win, posted, ask) {
  const tabA = win._testApi.getActiveTabId();
  openPanelViaMenu(win);
  ask(win, posted, tabA);
  click(win, byId(win, 'workdir-panel-close'));
  assert.ok(!panelOpen(win), 'the user closed A\'s panel');

  win._testApi.createNewTab();
  const tabB = win._testApi.getActiveTabId();
  assert.notStrictEqual(tabB, tabA);
  openPanelViaMenu(win);
  typeInto(win, 'workdir-input', '/typed/for-b');
  assert.strictEqual(byId(win, 'workdir-input').value, '/typed/for-b');
  return {tabA, tabB};
}

function askViaPickButton(win, posted, tabA) {
  click(win, byId(win, 'workdir-pick-btn'));
  const picks = msgs(posted, 'pickWorkDir');
  assert.strictEqual(picks.length, 1);
  assert.strictEqual(picks[0].tabId, tabA);
}

function askViaTypedPath(win, posted, tabA) {
  typeInto(win, 'workdir-input', '/asked/by-a');
  pressEnter(win, 'workdir-input');
  const opens = msgs(posted, 'openWorkDir');
  assert.strictEqual(opens.length, 1);
  assert.strictEqual(opens[0].tabId, tabA);
}

// ---------------------------------------------------------------------------

function testLatePickForAKeepsBsPanelOpen() {
  const {win, posted} = makeWebview();
  const {tabA, tabB} = askForAThenOpenBsPanel(win, posted, askViaPickButton);

  send(win, {type: 'workDirPicked', path: '/picked/for-a', tabId: tabA});
  assert.ok(panelOpen(win), 'A\'s reply must not close B\'s panel');
  assert.strictEqual(
    byId(win, 'workdir-input').value,
    '/typed/for-b',
    'B\'s typed path is intact',
  );
  assert.strictEqual(errorText(win), '');

  // A was still pinned by its reply; B was not.
  click(win, byId(win, 'workdir-panel-close'));
  let sub = submitPrompt(win, posted, 'from b');
  assert.strictEqual(sub.tabId, tabB);
  assert.strictEqual(sub.workDir, undefined, 'tab B was not pinned');
  switchToTab(win, tabA);
  sub = submitPrompt(win, posted, 'from a');
  assert.strictEqual(sub.workDir, '/picked/for-a', 'tab A runs where it asked');
}

function testLateErrorForAStaysOutOfBsPanel() {
  const {win, posted} = makeWebview();
  const {tabA} = askForAThenOpenBsPanel(win, posted, askViaTypedPath);

  send(win, {
    type: 'workDirError',
    text: 'Not a directory: /asked/by-a',
    tabId: tabA,
  });
  assert.ok(panelOpen(win));
  assert.strictEqual(errorText(win), '', 'A\'s error is not shown to B');
  assert.strictEqual(byId(win, 'workdir-input').value, '/typed/for-b');
}

function testLatePickForATurnedContentTabShowsNoErrorInBsPanel() {
  const {win, posted} = makeWebview();
  const {tabA} = askForAThenOpenBsPanel(win, posted, askViaPickButton);
  // Tab A is busy by the time its folder arrives: the pin is refused,
  // and the refusal is A's business, not B's panel's.
  send(win, {type: 'status', running: true, tabId: tabA});
  send(win, {type: 'workDirPicked', path: '/picked/for-a', tabId: tabA});
  assert.ok(panelOpen(win));
  assert.strictEqual(errorText(win), '');
  assert.strictEqual(byId(win, 'workdir-input').value, '/typed/for-b');
}

function testReplyForTheAskingPanelStillCloses() {
  const {win, posted} = makeWebview();
  const tabA = win._testApi.getActiveTabId();
  openPanelViaMenu(win);
  askViaTypedPath(win, posted, tabA);
  send(win, {type: 'workDirPicked', path: '/asked/by-a', tabId: tabA});
  assert.ok(!panelOpen(win), 'the tab that asked closes its own panel');
  const sub = submitPrompt(win, posted, 'go');
  assert.strictEqual(sub.workDir, '/asked/by-a');
}

function testErrorForTheAskingPanelIsShown() {
  const {win, posted} = makeWebview();
  const tabA = win._testApi.getActiveTabId();
  openPanelViaMenu(win);
  askViaTypedPath(win, posted, tabA);
  send(win, {type: 'workDirError', text: 'Not a directory: x', tabId: tabA});
  assert.ok(panelOpen(win), 'an error leaves the panel on screen');
  assert.strictEqual(errorText(win), 'Not a directory: x');

  // The refusal of a pick for a tab that started running meanwhile is
  // shown in that tab's own panel.
  send(win, {type: 'status', running: true, tabId: tabA});
  send(win, {type: 'workDirPicked', path: '/late', tabId: tabA});
  assert.ok(panelOpen(win));
  assert.match(errorText(win), /running task keeps its working directory/);
}

function testPanelOpenedForAThenBAsksClosesOnBsReplyOnly() {
  // The panel stays open across a tab switch; the request it then
  // sends names the now-active tab, and only that tab's reply closes it.
  const {win, posted} = makeWebview();
  const tabA = win._testApi.getActiveTabId();
  openPanelViaMenu(win);
  askViaPickButton(win, posted, tabA);
  win._testApi.createNewTab();
  const tabB = win._testApi.getActiveTabId();
  assert.ok(panelOpen(win));
  typeInto(win, 'workdir-input', '/for/b');
  pressEnter(win, 'workdir-input');
  const opens = msgs(posted, 'openWorkDir');
  assert.strictEqual(opens[opens.length - 1].tabId, tabB);

  send(win, {type: 'workDirPicked', path: '/picked/for-a', tabId: tabA});
  assert.ok(panelOpen(win), 'A\'s stale reply leaves B\'s pending panel');
  send(win, {type: 'workDirPicked', path: '/for/b', tabId: tabB});
  assert.ok(!panelOpen(win), 'B\'s reply closes the panel B is waiting on');
}

function testErrorWithoutTabIdKeepsOldBehaviour() {
  const {win, posted} = makeWebview();
  askForAThenOpenBsPanel(win, posted, askViaTypedPath);
  send(win, {type: 'workDirError', text: 'Not a directory: legacy'});
  assert.ok(panelOpen(win));
  assert.strictEqual(
    errorText(win),
    'Not a directory: legacy',
    'an older host without tabId still reports into the open panel',
  );
}

// ---- host: workDirError names the requesting tab ------------------------

class EventEmitterLite {
  constructor() {
    this._subs = [];
  }
  event = cb => {
    this._subs.push(cb);
    return {dispose() {}};
  };
  fire(v) {
    for (const cb of this._subs) cb(v);
  }
  dispose() {}
}

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-workdir-reply-tab-'));
fs.writeFileSync(path.join(tmp, 'a-file.txt'), 'not a folder\n');

global.__kissVscodeStub = {
  Uri: {
    file: p => ({fsPath: p, scheme: 'file'}),
    joinPath: (base, ...parts) => ({
      fsPath: path.join(base.fsPath, ...parts),
      scheme: 'file',
    }),
    parse: s => ({fsPath: s.replace(/^file:\/\//, ''), scheme: 'file'}),
  },
  Position: class {},
  Range: class {},
  Selection: class {},
  TextEditorRevealType: {InCenter: 2},
  ViewColumn: {One: 1},
  ProgressLocation: {Notification: 15},
  ConfigurationTarget: {Global: 1, Workspace: 2, WorkspaceFolder: 3},
  EventEmitter: EventEmitterLite,
  TabInputText: class {},
  window: {
    visibleTextEditors: [],
    activeTextEditor: undefined,
    tabGroups: {all: [], close: () => Promise.resolve(true)},
    createTextEditorDecorationType: () => ({dispose() {}}),
    onDidChangeVisibleTextEditors: () => ({dispose() {}}),
    showTextDocument: () => Promise.resolve({}),
    showInformationMessage: () => undefined,
    showWarningMessage: () => undefined,
    showErrorMessage: () => undefined,
    showOpenDialog: () => Promise.resolve(undefined),
    withProgress: (_opts, task) =>
      task({report() {}}, {onCancellationRequested: () => ({dispose() {}})}),
    createTerminal: () => ({show() {}, sendText() {}}),
  },
  workspace: {
    isTrusted: true,
    workspaceFolders: [{uri: {fsPath: tmp, scheme: 'file'}}],
    getConfiguration: () => ({get: () => undefined}),
    onWillSaveTextDocument: () => ({dispose() {}}),
    onDidSaveTextDocument: () => ({dispose() {}}),
    onDidChangeWorkspaceFolders: () => ({dispose() {}}),
    textDocuments: [],
    openTextDocument: () => Promise.resolve({}),
    saveAll: () => Promise.resolve(true),
    applyEdit: () => Promise.resolve(true),
  },
  commands: {executeCommand: () => Promise.resolve()},
};
const origResolve = Module._resolveFilename;
Module._resolveFilename = function (request, parent, ...rest) {
  if (request === 'vscode') return require.resolve('./_vscode-stub.js');
  return origResolve.call(this, request, parent, ...rest);
};

async function testHostStampsWorkDirErrorWithTheTab() {
  const outDir = path.join(__dirname, '..', 'out');
  assert.ok(
    fs.existsSync(path.join(outDir, 'SorcarSidebarView.js')),
    'compiled extension missing — run `npm run compile` first',
  );
  const {SorcarSidebarView} = require(path.join(outDir, 'SorcarSidebarView.js'));
  const view = new SorcarSidebarView({fsPath: path.join(tmp, 'ext')});
  const forwarded = [];
  view._api = {
    forward: cmd => forwarded.push(cmd),
    run: () => {},
    getConfig: () => {},
    setWorkDir: () => {},
  };
  const posted = [];
  view._view = {
    visible: true,
    webview: {postMessage: m => posted.push(m)},
    show() {},
  };
  view._disposed = false;

  await view._handleMessage({
    type: 'openWorkDir',
    path: path.join(tmp, 'a-file.txt'),
    tabId: 'tab-a',
  });
  await view._handleMessage({type: 'openWorkDir', path: '/', tabId: 'tab-b'});
  const errors = posted.filter(m => m.type === 'workDirError');
  assert.strictEqual(errors.length, 2);
  assert.match(errors[0].text, /^Not a directory: /);
  assert.strictEqual(errors[0].tabId, 'tab-a');
  assert.match(errors[1].text, /file-system root/);
  assert.strictEqual(errors[1].tabId, 'tab-b');
  assert.strictEqual(forwarded.length, 0, 'nothing recorded for a refusal');
}

// ---------------------------------------------------------------------------

const tests = [
  ['late workDirPicked for A keeps B\'s panel and typed path', testLatePickForAKeepsBsPanelOpen],
  ['late workDirError for A is not shown in B\'s panel', testLateErrorForAStaysOutOfBsPanel],
  [
    'late pick for a now-running A shows no error in B\'s panel',
    testLatePickForATurnedContentTabShowsNoErrorInBsPanel,
  ],
  ['a reply for the asking tab still closes the panel', testReplyForTheAskingPanelStillCloses],
  ['an error for the asking tab is shown in its panel', testErrorForTheAskingPanelIsShown],
  [
    'panel kept open across a tab switch closes on the new request\'s reply only',
    testPanelOpenedForAThenBAsksClosesOnBsReplyOnly,
  ],
  ['workDirError without tabId keeps the old behaviour', testErrorWithoutTabIdKeepsOldBehaviour],
  ['host: workDirError carries the requesting tabId', testHostStampsWorkDirErrorWithTheTab],
];

(async () => {
  let failed = 0;
  for (const [name, fn] of tests) {
    try {
      await fn();
      console.log('ok - ' + name);
    } catch (e) {
      failed += 1;
      console.log('not ok - ' + name);
      console.log(e && e.stack ? e.stack : String(e));
    }
  }
  fs.rmSync(tmp, {recursive: true, force: true});
  if (failed) {
    console.log(`${failed} of ${tests.length} tests failed`);
    process.exit(1);
  }
  console.log(`all ${tests.length} audit0924_workdir_reply_tab tests passed`);
  process.exit(0);
})();
