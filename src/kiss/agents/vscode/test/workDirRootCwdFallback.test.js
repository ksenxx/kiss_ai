// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// E2E tests for the work-dir fallback of a window with NO folder open.
//
// Bug: a Dock/Finder-launched VS Code window with no workspace folder
// runs its extension host with cwd at the filesystem root ('/'), and
// `_getWorkDir()` fell back to `process.cwd()` — so the window pinned
// the daemon to '/', every task it submitted recorded work_dir '/',
// and the @-mention file picker scanned the whole disk ("@ shows
// files from different folders").
//
// Fixed behaviour, covered through the real compiled _handleMessage()
// and _installClientListener() code paths (the daemon API client is
// replaced by a recorder, as in the other extension-host tests; the
// daemon side of the empty-workDir fallback is covered separately by
// src/kiss/tests/server/test_per_window_work_dir.py):
//  1. no folder + root cwd      -> submit carries workDir '' (which the
//     daemon resolves to its own fallback folder) and configData keeps
//     the daemon's work_dir instead of overwriting it with the root.
//  2. no folder + normal cwd    -> cwd still wins (`code file.txt`
//     launched from a terminal keeps the shell's directory).
//  3. folder open               -> the folder still wins everywhere.

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

const fsRoot = path.parse(process.cwd()).root; // '/' on POSIX, 'C:\' on win32

function makeView() {
  const view = new SorcarSidebarView({fsPath: path.join(tmp, 'ext')});
  const sent = [];
  view._api = {
    _sent: sent,
    forward: c => sent.push(c),
    run: c => sent.push({type: 'run', ...c}),
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
  return {view, sent};
}

async function submitAndGetRun(view, sent, tabId) {
  await view._handleMessage({
    type: 'submit',
    prompt: 'explain the fallback\nsecond line so no path lookup happens',
    model: 'test-model',
    tabId,
  });
  const run = sent.find(m => m.type === 'run');
  assert.ok(run, 'submit must reach the daemon as a run command');
  return run;
}

// 1a. No folder open, host cwd at the filesystem root: the submitted
//     task must NOT be rooted at the whole disk.
async function testRootCwdSubmitsEmptyWorkDir() {
  process.chdir(fsRoot);
  const {view, sent} = makeView();
  const run = await submitAndGetRun(view, sent, 't-root');
  assert.strictEqual(
    run.workDir,
    '',
    "no-folder window with root cwd must submit workDir '' " +
      `(got ${JSON.stringify(run.workDir)})`,
  );
  view.dispose();
  console.log('ok - root cwd: submit carries empty workDir, not the root');
}

// 1b. Same window: the configData rewrite must keep the daemon's
//     work_dir instead of blanking it or replacing it with the root.
function testRootCwdKeepsDaemonConfigWorkDir() {
  process.chdir(fsRoot);
  const {view} = makeView();
  const handlers = {};
  view._installClientListener({
    on: (evt, cb) => {
      handlers[evt] = cb;
    },
  });
  const msg = {type: 'configData', config: {work_dir: '/Users/u/proj'}};
  handlers.message(msg);
  assert.strictEqual(
    msg.config.work_dir,
    '/Users/u/proj',
    'configData must keep the daemon work_dir when the window has none',
  );
  view.dispose();
  console.log('ok - root cwd: configData keeps the daemon work_dir');
}

// 2. No folder open, normal cwd (VS Code launched from a terminal via
//    `code file.txt`): the shell's directory must still win.
async function testTerminalCwdStillWins() {
  process.chdir(wsRoot);
  const cwd = process.cwd(); // may differ from wsRoot via symlinks
  const {view, sent} = makeView();
  const run = await submitAndGetRun(view, sent, 't-term');
  assert.strictEqual(
    run.workDir,
    cwd,
    'no-folder window with a normal cwd must keep submitting that cwd',
  );
  const handlers = {};
  view._installClientListener({
    on: (evt, cb) => {
      handlers[evt] = cb;
    },
  });
  const msg = {type: 'configData', config: {work_dir: '/Users/u/proj'}};
  handlers.message(msg);
  assert.strictEqual(
    msg.config.work_dir,
    cwd,
    'configData must show the non-root cwd when the window has one',
  );
  view.dispose();
  console.log('ok - terminal cwd: non-root cwd fallback preserved');
}

// 3. A window WITH a folder open is untouched by the fix.
async function testWorkspaceFolderStillWins() {
  process.chdir(fsRoot);
  vscodeStub.workspace.workspaceFolders = [
    {uri: {fsPath: wsRoot, scheme: 'file'}},
  ];
  const {view, sent} = makeView();
  const run = await submitAndGetRun(view, sent, 't-ws');
  assert.strictEqual(
    run.workDir,
    wsRoot,
    'a window with a folder open must keep submitting that folder',
  );
  view.dispose();
  vscodeStub.workspace.workspaceFolders = undefined;
  console.log('ok - workspace folder: still wins over any cwd');
}

async function main() {
  const origCwd = process.cwd();
  try {
    await testRootCwdSubmitsEmptyWorkDir();
    testRootCwdKeepsDaemonConfigWorkDir();
    await testTerminalCwdStillWins();
    await testWorkspaceFolderStillWins();
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
