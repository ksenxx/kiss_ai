// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// E2E tests for SorcarSidebarView routing/lifecycle findings:
// VS-011: openFile/checkPaths must resolve against the tab's workDir.
// VS-012: symlinks must not escape the workspace containment check.
// VS-014: merge all-done must echo the merge_data work_dir.
// VS-016: closeTab must release per-tab host resources.
// VS-018: stopTask without a resolved webview must stop via the API.

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const Module = require('module');

const opened = [];

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

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-sidebar-route-'));
const wsRoot = path.join(tmp, 'workspace');
const tabRepo = path.join(tmp, 'tab-repo');
const outside = path.join(tmp, 'outside');
fs.mkdirSync(wsRoot, {recursive: true});
fs.mkdirSync(tabRepo, {recursive: true});
fs.mkdirSync(outside, {recursive: true});
fs.writeFileSync(path.join(tabRepo, 'inrepo.txt'), 'tab repo file\n');
fs.writeFileSync(path.join(wsRoot, 'inws.txt'), 'workspace file\n');
fs.writeFileSync(path.join(outside, 'secret.txt'), 'SECRET\n');
if (process.platform !== 'win32') {
  fs.symlinkSync(outside, path.join(wsRoot, 'link-out'));
}

const vscodeStub = {
  Uri: {
    file: p => ({fsPath: p, scheme: 'file'}),
    joinPath: (base, ...parts) => ({
      fsPath: path.join(base.fsPath, ...parts),
      scheme: 'file',
    }),
    parse: s => ({fsPath: s.replace(/^file:\/\//, ''), scheme: 'file'}),
  },
  Position: class {
    constructor(line, character) {
      this.line = line;
      this.character = character;
    }
  },
  Range: class {
    constructor(a, b, c, d) {
      this.start = a && a.line !== undefined ? a : {line: a, character: b};
      this.end = b && b.line !== undefined ? b : {line: c, character: d};
    }
  },
  Selection: class {
    constructor(a, b) {
      this.anchor = a;
      this.active = b;
    }
  },
  TextEditorRevealType: {InCenter: 2},
  ViewColumn: {One: 1},
  ProgressLocation: {Notification: 15},
  EventEmitter: EventEmitterLite,
  TabInputText: class {},
  window: {
    visibleTextEditors: [],
    activeTextEditor: undefined,
    tabGroups: {all: [], close: () => Promise.resolve(true)},
    createTextEditorDecorationType: () => ({dispose() {}}),
    onDidChangeVisibleTextEditors: () => ({dispose() {}}),
    showTextDocument: doc => {
      opened.push(doc.uri.fsPath);
      return Promise.resolve({
        document: doc,
        selection: null,
        revealRange() {},
        setDecorations() {},
        edit: () => Promise.resolve(true),
      });
    },
    showInformationMessage: () => undefined,
    showWarningMessage: () => undefined,
    showErrorMessage: () => undefined,
    withProgress: (_opts, task) =>
      task({report() {}}, {onCancellationRequested: () => ({dispose() {}})}),
    createTerminal: () => ({show() {}, sendText() {}}),
  },
  workspace: {
    isTrusted: true,
    workspaceFolders: [{uri: {fsPath: wsRoot, scheme: 'file'}}],
    getConfiguration: () => ({get: () => undefined}),
    onWillSaveTextDocument: () => ({dispose() {}}),
    onDidSaveTextDocument: () => ({dispose() {}}),
    onDidChangeWorkspaceFolders: () => ({dispose() {}}),
    textDocuments: [],
    openTextDocument: uri =>
      Promise.resolve({
        uri,
        getText: () => '',
        lineCount: 1,
        lineAt: () => ({text: '', range: {end: {line: 0, character: 0}}}),
        isDirty: false,
      }),
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
global.__kissVscodeStub = vscodeStub;

const outDir = path.join(__dirname, '..', 'out');
assert.ok(
  fs.existsSync(path.join(outDir, 'SorcarSidebarView.js')),
  'compiled extension missing — run `npm run compile` first',
);
const {SorcarSidebarView} = require(path.join(outDir, 'SorcarSidebarView.js'));

function makeView() {
  const view = new SorcarSidebarView({fsPath: path.join(tmp, 'ext')});
  const sent = [];
  // Replace the daemon client with a recorder (no real daemon in tests).
  view._api = {
    _sent: sent,
    forward: c => sent.push(c),
    run: c => sent.push({type: 'run', ...c}),
    stop: tabId => sent.push({type: 'stop', tabId}),
    complete: c => sent.push({type: 'complete', ...c}),
    mergeAction: (action, tabId, workDir) =>
      sent.push({type: 'mergeAction', action, tabId, workDir}),
    closeTab: tabId => sent.push({type: 'closeTab', tabId}),
    getModels: () => {},
    getInputHistory: () => {},
    getConfig: () => {},
    setWorkDir: () => {},
    recordFileUsage: () => {},
    selectModel: () => {},
    userAnswer: () => {},
    resumeSession: () => {},
    worktreeAction: () => {},
    autocommitAction: () => {},
    generateCommitMessage: () => {},
    serverReset: () => {},
    appendUserMessage: () => {},
  };
  return {view, sent};
}

function delay(ms) {
  return new Promise(r => setTimeout(r, ms));
}

// VS-011: openFile resolves against the tab's workDir, not workspace 0.
async function testOpenFileHonorsTabWorkDir() {
  const {view} = makeView();
  opened.length = 0;
  await view._handleMessage({
    type: 'openFile',
    path: 'inrepo.txt',
    workDir: tabRepo,
    tabId: 't1',
  });
  assert.deepStrictEqual(
    opened,
    [path.join(tabRepo, 'inrepo.txt')],
    'VS-011: openFile must resolve relative paths against the tab workDir',
  );
  view.dispose();
  console.log('ok - VS-011 openFile honors the tab workDir');
}

// VS-011: checkPaths resolves against the supplied workDir.
async function testCheckPathsHonorsTabWorkDir() {
  const {view} = makeView();
  const posted = [];
  view._view = {
    visible: true,
    webview: {postMessage: m => posted.push(m)},
    show() {},
  };
  view._disposed = false;
  await view._handleMessage({
    type: 'checkPaths',
    paths: ['inrepo.txt', 'inws.txt'],
    workDir: tabRepo,
    tabId: 't1',
  });
  const reply = posted.find(m => m.type === 'pathsExist');
  assert.ok(reply, 'pathsExist reply expected');
  assert.strictEqual(
    reply.results['inrepo.txt'],
    true,
    'VS-011: file in the tab repo must be reported as existing',
  );
  assert.strictEqual(
    reply.results['inws.txt'],
    false,
    'VS-011: file that only exists in workspace 0 must not match the tab repo',
  );
  view.dispose();
  console.log('ok - VS-011 checkPaths honors the tab workDir');
}

// VS-012: symlink escape must be refused for openFile and checkPaths.
async function testSymlinkEscapeRefused() {
  if (process.platform === 'win32') return;
  const {view} = makeView();
  opened.length = 0;
  const posted = [];
  view._view = {
    visible: true,
    webview: {postMessage: m => posted.push(m)},
    show() {},
  };
  view._disposed = false;
  await view._handleMessage({
    type: 'openFile',
    path: 'link-out/secret.txt',
    tabId: 't1',
  });
  assert.deepStrictEqual(
    opened,
    [],
    'VS-012: a workspace symlink must not open a file outside the workspace',
  );
  await view._handleMessage({
    type: 'checkPaths',
    paths: ['link-out/secret.txt'],
    tabId: 't1',
  });
  const reply = posted.find(m => m.type === 'pathsExist');
  assert.strictEqual(
    reply.results['link-out/secret.txt'],
    false,
    'VS-012: checkPaths must not report symlink-escaping paths as openable',
  );
  view.dispose();
  console.log('ok - VS-012 symlink escape refused');
}

// VS-014: merge all-done echoes the merge_data work_dir.
async function testMergeAllDoneUsesMergeWorkDir() {
  const {view, sent} = makeView();
  view._ownTabs.add('tabX');
  view._mergeWorkDirs.set('tabX', tabRepo);
  view.sendMergeAllDone('tabX');
  const msg = sent.find(m => m.type === 'mergeAction');
  assert.ok(msg, 'mergeAction sent');
  assert.strictEqual(
    msg.workDir,
    tabRepo,
    'VS-014: all-done must carry the merge work_dir, not workspace 0',
  );
  assert.ok(
    !view._mergeWorkDirs.has('tabX'),
    'merge workDir entry must be consumed',
  );
  view.dispose();
  console.log('ok - VS-014 merge all-done echoes merge_data work_dir');
}

// VS-016: closeTab releases per-tab resources.
async function testCloseTabCleansResources() {
  const {view, sent} = makeView();
  view._ownTabs.add('tabY');
  view._runningTabs.add('tabY');
  view._commitPendingTabs.add('tabY');
  view._worktreeDirs.set('tabY', tabRepo);
  view._mergeWorkDirs.set('tabY', tabRepo);
  view._preMergeOpenFiles.set('tabY', new Set());
  let resolved = false;
  view._worktreeActionResolves.set('tabY', () => (resolved = true));
  view._worktreeProgresses.set('tabY', {report() {}});
  const mgr = view._getOrCreateMergeManager('tabY');
  assert.ok(view._mergeManagers.has('tabY'));
  await view._handleMessage({type: 'closeTab', tabId: 'tabY'});
  assert.ok(!view._mergeManagers.has('tabY'), 'VS-016: merge manager leaked');
  assert.ok(!view._runningTabs.has('tabY'), 'VS-016: runningTabs leaked');
  assert.ok(
    !view._commitPendingTabs.has('tabY'),
    'VS-016: commitPendingTabs leaked',
  );
  assert.ok(!view._worktreeDirs.has('tabY'), 'VS-016: worktreeDirs leaked');
  assert.ok(!view._mergeWorkDirs.has('tabY'), 'VS-016: mergeWorkDirs leaked');
  assert.ok(
    !view._preMergeOpenFiles.has('tabY'),
    'VS-016: preMergeOpenFiles leaked',
  );
  assert.ok(resolved, 'VS-016: worktree progress resolver not resolved');
  assert.ok(
    !view._worktreeActionResolves.has('tabY') &&
      !view._worktreeProgresses.has('tabY'),
    'VS-016: worktree progress maps leaked',
  );
  assert.ok(
    sent.some(m => m.type === 'closeTab' && m.tabId === 'tabY'),
    'closeTab still forwarded to the daemon',
  );
  void mgr;
  view.dispose();
  console.log('ok - VS-016 closeTab releases per-tab resources');
}

// VS-018: stopTask without a resolved webview stops via the API.
async function testStopTaskWithoutWebview() {
  const {view, sent} = makeView();
  view._runningTabs.add('run1');
  view._runningTabs.add('run2');
  view.stopTask(); // no view resolved — must fall back to direct stops
  const stops = sent.filter(m => m.type === 'stop').map(m => m.tabId);
  assert.deepStrictEqual(
    stops.sort(),
    ['run1', 'run2'],
    'VS-018: stopTask must stop running tabs when no webview is resolved',
  );
  view.dispose();
  console.log('ok - VS-018 stopTask stops via API without a webview');
}

// VS-013: ghost completion must use the webview-supplied tabId, not the
// host's stale notion of the active tab.
async function testCompleteUsesMessageTabId() {
  const {view, sent} = makeView();
  view._activeTabId = 'stale-tab';
  await view._handleMessage({type: 'complete', query: 'fix bug', tabId: 'fresh-tab'});
  const msg = sent.find(m => m.type === 'complete');
  assert.ok(msg, 'complete forwarded');
  assert.strictEqual(
    msg.tabId,
    'fresh-tab',
    'VS-013: completion routed to the stale host-side active tab',
  );
  view.dispose();
  console.log('ok - VS-013 completion uses the message tabId');
}

// VS-015: post-merge restore closes only editors the merge itself opened.
async function testRestoreClosesOnlyMergeOpenedEditors() {
  const {view} = makeView();
  const vscode = global.__kissVscodeStub;
  const mkTab = fp => {
    const input = new vscode.TabInputText();
    input.uri = {fsPath: fp, scheme: 'file'};
    return {input};
  };
  const userFile = path.join(wsRoot, 'user-opened.txt');
  const mergeFile = path.join(wsRoot, 'merge-opened.txt');
  const closed = [];
  vscode.window.tabGroups = {
    all: [{tabs: [mkTab(userFile), mkTab(mergeFile)]}],
    close: tabs => {
      for (const t of tabs) closed.push(t.input.uri.fsPath);
      return Promise.resolve(true);
    },
  };
  view._preMergeOpenFiles.set('tabR', new Set()); // nothing open pre-merge
  await view._restorePreMergeEditors('tabR', new Set([mergeFile]));
  assert.deepStrictEqual(
    closed,
    [mergeFile],
    'VS-015: restore must close only editors the merge review opened',
  );
  vscode.window.tabGroups = {all: [], close: () => Promise.resolve(true)};
  view.dispose();
  console.log('ok - VS-015 restore keeps user-opened editors');
}

async function run() {
  await testOpenFileHonorsTabWorkDir();
  await testCompleteUsesMessageTabId();
  await testRestoreClosesOnlyMergeOpenedEditors();
  await testCheckPathsHonorsTabWorkDir();
  await testSymlinkEscapeRefused();
  await testMergeAllDoneUsesMergeWorkDir();
  await testCloseTabCleansResources();
  await testStopTaskWithoutWebview();
  await delay(10);
  console.log('\nAll sidebar workDir-routing tests passed');
}

run().then(
  () => {
    fs.rmSync(tmp, {recursive: true, force: true});
    process.exit(0);
  },
  err => {
    console.error('FAIL:', err);
    fs.rmSync(tmp, {recursive: true, force: true});
    process.exit(1);
  },
);
