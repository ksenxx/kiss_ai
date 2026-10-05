// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// E2E tests for the work-dir fallback of a window with NO folder open.
//
// Bug: a Dock/Finder-launched VS Code window with no workspace folder
// runs its extension host with cwd at the filesystem root ('/'), and
// `_getWorkDir()` fell back to `process.cwd()` — so the window seeded
// the daemon with '/', every task it submitted recorded work_dir '/',
// and the @-mention file picker scanned the whole disk ("@ shows
// files from different folders").
//
// The working directory is ONE global value every task on every surface
// runs in (the daemon's config.json work_dir).  The window's folder only
// SEEDS it: the connect preamble is `setWorkDir {workDir, ifUnset:true}`
// with the workspace folder, else the host cwd unless that is a root
// (`_workspaceDir()`), and the daemon ignores the seed once a directory
// is persisted.  `_getWorkDir()` -- the fallback for path-taking webview
// messages that name no workDir -- prefers the daemon-reported directory
// (`configData.config.work_dir` / `workDirChanged`, cached in
// `_daemonWorkDir`) over `_workspaceDir()`, and `configData` reaches the
// webview untouched.
//
// Covered through the real compiled _handleMessage(), _getClient() and
// _installClientListener() code paths (the daemon API client is replaced
// by a recorder, as in the other extension-host tests; the daemon side of
// the empty-workDir seed is covered separately by the Python tests):
//  1. no folder + root cwd      -> the seed is '' (the daemon keeps its
//     own fallback), the forwarded submit carries no workDir of its own,
//     configData is not rewritten, and the daemon's directory becomes
//     `_getWorkDir()`.
//  2. no folder + normal cwd    -> the seed is the cwd (`code file.txt`
//     launched from a terminal keeps the shell's directory) and it is
//     the host fallback only until the daemon reports a directory.
//  3. folder open               -> the seed is the folder, which the
//     daemon-reported directory likewise outranks.

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const Module = require('module');

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

const workspaceChangeHandlers = [];

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-root-cwd-'));
const wsRoot = path.join(tmp, 'workspace');
fs.mkdirSync(wsRoot, {recursive: true});

const vscodeStub = {
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
    withProgress: (_opts, task) =>
      task({report() {}}, {onCancellationRequested: () => ({dispose() {}})}),
    createTerminal: () => ({show() {}, sendText() {}}),
  },
  workspace: {
    isTrusted: true,
    // The scenario under test: a window with NO folder open.
    workspaceFolders: undefined,
    getConfiguration: () => ({get: () => undefined}),
    onWillSaveTextDocument: () => ({dispose() {}}),
    onDidSaveTextDocument: () => ({dispose() {}}),
    // Recorded: the window must no longer react to folder changes (the
    // folder only seeds the daemon on connect).
    onDidChangeWorkspaceFolders: cb => {
      workspaceChangeHandlers.push(cb);
      return {dispose() {}};
    },
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

const fsRoot = path.parse(process.cwd()).root; // '/' on POSIX, 'C:\' on win32

function makeView() {
  const view = new SorcarSidebarView({fsPath: path.join(tmp, 'ext')});
  const sent = [];
  view._api = {
    _sent: sent,
    forward: c => sent.push(c),
    submit: c => sent.push({type: 'submit', ...c}),
    stop: tabId => sent.push({type: 'stop', tabId}),
    complete: c => sent.push({type: 'complete', ...c}),
    closeTab: tabId => sent.push({type: 'closeTab', tabId}),
    getModels: () => {},
    getInputHistory: () => {},
    getConfig: () => {},
    setWorkDir: wd => sent.push({type: 'setWorkDir', workDir: wd}),
    recordFileUsage: () => {},
    selectModel: () => {},
    userAnswer: () => {},
    resumeSession: () => {},
    worktreeAction: () => {},
    generateCommitMessage: () => {},
    serverReset: () => {},
    appendUserMessage: () => {},
  };
  // What the host actually forwards to the webview.
  const toWebview = [];
  view._view = {
    visible: true,
    webview: {postMessage: m => toWebview.push(m)},
    show() {},
  };
  return {view, sent, toWebview};
}

// The seed the window hands the daemon on every connect: the preamble
// `_getClient()` installs on the real AgentClient (never connected here).
function connectSeed(view) {
  const preamble = view._getClient()._preamble;
  assert.ok(preamble, 'the window must install a connect preamble');
  assert.strictEqual(preamble.type, 'setWorkDir');
  assert.strictEqual(
    preamble.ifUnset,
    true,
    'the seed must be ifUnset: opening a window never overrides the ' +
      "user's persisted global directory",
  );
  return preamble.workDir;
}

// Submit from a tab and check the forwarded command carries no workDir
// of its own: the daemon runs every task in the global directory.
async function submitCarriesNoWorkDir(view, sent, tabId) {
  await view._handleMessage({
    type: 'submit',
    prompt: 'explain the fallback\nsecond line so no path lookup happens',
    model: 'test-model',
    tabId,
  });
  const submit = sent.find(m => m.type === 'submit');
  assert.ok(submit, 'submit must reach the daemon as a submit command');
  assert.strictEqual(
    submit.workDir,
    undefined,
    'the host must not stamp a workDir; the daemon applies the global one',
  );
}

// Deliver a daemon message through the real client listener and return
// the message the host forwarded to the webview for it.
function deliver(view, toWebview, msg) {
  const handlers = {};
  view._installClientListener({
    on: (evt, cb) => {
      handlers[evt] = cb;
    },
  });
  const before = toWebview.length;
  handlers.message(msg);
  const forwarded = toWebview.slice(before).find(m => m.type === msg.type);
  assert.ok(forwarded, `the host must forward ${msg.type} to the webview`);
  return forwarded;
}

// 1a. No folder open, host cwd at the filesystem root: the submitted
//     task must NOT be rooted at the whole disk, and neither may the
//     seed the window hands the daemon.
async function testRootCwdSeedsEmptyWorkDir() {
  process.chdir(fsRoot);
  const {view, sent} = makeView();
  await submitCarriesNoWorkDir(view, sent, 't-root');
  assert.strictEqual(
    view._getWorkDir(),
    '',
    "no-folder window with root cwd must fall back to '' " +
      `(got ${JSON.stringify(view._getWorkDir())})`,
  );
  assert.strictEqual(
    connectSeed(view),
    '',
    "no-folder window with root cwd must seed the daemon with '', not the root",
  );
  view.dispose();
  console.log('ok - root cwd: the window seeds an empty workDir, not the root');
}

// 1b. Same window: configData must reach the webview untouched, and the
//     daemon's directory becomes the host fallback from then on, as does
//     every later workDirChanged broadcast.
function testRootCwdAdoptsDaemonWorkDir() {
  process.chdir(fsRoot);
  const {view, toWebview} = makeView();
  const msg = deliver(view, toWebview, {
    type: 'configData',
    config: {work_dir: '/Users/u/proj'},
  });
  assert.strictEqual(
    msg.config.work_dir,
    '/Users/u/proj',
    'configData must keep the daemon work_dir: the webview shows the global value',
  );
  assert.strictEqual(
    view._getWorkDir(),
    '/Users/u/proj',
    'the daemon-reported directory is the host fallback',
  );
  const changed = deliver(view, toWebview, {
    type: 'workDirChanged',
    workDir: '/Users/u/other',
  });
  assert.strictEqual(
    changed.workDir,
    '/Users/u/other',
    'the broadcast reaches the webview so it re-scopes too',
  );
  assert.strictEqual(
    view._getWorkDir(),
    '/Users/u/other',
    'a workDirChanged broadcast (a pick on any surface) moves the fallback',
  );
  const blank = deliver(view, toWebview, {
    type: 'configData',
    config: {work_dir: ''},
  });
  assert.strictEqual(blank.config.work_dir, '', 'still not rewritten');
  assert.strictEqual(
    view._getWorkDir(),
    '/Users/u/other',
    'a config reply without a directory does not forget the last reported one',
  );
  view.dispose();
  console.log('ok - root cwd: the daemon work_dir is adopted, never rewritten');
}

// 2. No folder open, normal cwd (VS Code launched from a terminal via
//    `code file.txt`): the shell's directory is the seed and the host
//    fallback -- until the daemon reports the global directory.
async function testTerminalCwdSeedsAndYieldsToDaemon() {
  process.chdir(wsRoot);
  const cwd = process.cwd(); // may differ from wsRoot via symlinks
  const {view, sent, toWebview} = makeView();
  await submitCarriesNoWorkDir(view, sent, 't-term');
  assert.strictEqual(
    connectSeed(view),
    cwd,
    'no-folder window with a normal cwd must seed the daemon with that cwd',
  );
  assert.strictEqual(
    view._getWorkDir(),
    cwd,
    'and use it as the host fallback before the daemon reports a directory',
  );
  const msg = deliver(view, toWebview, {
    type: 'configData',
    config: {work_dir: '/Users/u/proj'},
  });
  assert.strictEqual(
    msg.config.work_dir,
    '/Users/u/proj',
    'configData must not be rewritten with the cwd: the global value is shown',
  );
  assert.strictEqual(
    view._getWorkDir(),
    '/Users/u/proj',
    'the daemon-reported global directory outranks the cwd',
  );
  view.dispose();
  console.log('ok - terminal cwd: seeds the daemon, yields to its directory');
}

// 3. A window WITH a folder open seeds the folder, and the daemon's
//    global directory likewise outranks it.
async function testWorkspaceFolderSeedsAndYieldsToDaemon() {
  process.chdir(fsRoot);
  vscodeStub.workspace.workspaceFolders = [
    {uri: {fsPath: wsRoot, scheme: 'file'}},
  ];
  const {view, sent, toWebview} = makeView();
  await submitCarriesNoWorkDir(view, sent, 't-ws');
  assert.strictEqual(
    connectSeed(view),
    wsRoot,
    'a window with a folder open must seed the daemon with that folder',
  );
  assert.strictEqual(
    view._getWorkDir(),
    wsRoot,
    'and use it as the host fallback before the daemon reports a directory',
  );
  const msg = deliver(view, toWebview, {
    type: 'configData',
    config: {work_dir: '/Users/u/proj'},
  });
  assert.strictEqual(msg.config.work_dir, '/Users/u/proj', 'not rewritten');
  assert.strictEqual(
    view._getWorkDir(),
    '/Users/u/proj',
    'the global directory wins over the workspace folder: every task on ' +
      'every surface runs in the one directory',
  );
  assert.strictEqual(
    workspaceChangeHandlers.length,
    0,
    'the window must not watch onDidChangeWorkspaceFolders: the folder ' +
      'only seeds the daemon on connect (ifUnset); changing it later must ' +
      "not move the user's global directory",
  );
  view.dispose();
  vscodeStub.workspace.workspaceFolders = undefined;
  console.log(
    'ok - workspace folder: seeds the daemon, yields to its directory',
  );
}

async function main() {
  const origCwd = process.cwd();
  try {
    await testRootCwdSeedsEmptyWorkDir();
    testRootCwdAdoptsDaemonWorkDir();
    await testTerminalCwdSeedsAndYieldsToDaemon();
    await testWorkspaceFolderSeedsAndYieldsToDaemon();
  } finally {
    process.chdir(origCwd);
    fs.rmSync(tmp, {recursive: true, force: true});
  }
  console.log('all workDirRootCwdFallback tests passed');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
