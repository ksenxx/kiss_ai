// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// The extension-host half of the "Working directory" panel: the webview
// posts `openWorkDir {path}` (typed path or a row of the opened-so-far
// list) or `pickWorkDir` (the folder button), and the compiled
// SorcarSidebarView either runs `vscode.openFolder` on the folder or
// answers `workDirError` -- for a path that is not a directory, a
// file-system root (also one reached through `..` or a symlink), the
// folder this window already shows, or an open VS Code rejects.
//
// Runs the compiled extension (out/SorcarSidebarView.js) against a
// minimal `vscode` stub; run `npm run compile` first.

/* global require, __dirname, console, process, global */

'use strict';

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

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-workdir-host-'));
const wsRoot = path.join(tmp, 'workspace');
const other = path.join(tmp, 'other');
fs.mkdirSync(wsRoot, {recursive: true});
fs.mkdirSync(other, {recursive: true});
fs.writeFileSync(path.join(tmp, 'a-file.txt'), 'not a folder\n');
const rootLink = path.join(tmp, 'root-link');
const wsLink = path.join(tmp, 'ws-link');
if (process.platform !== 'win32') {
  fs.symlinkSync('/', rootLink);
  fs.symlinkSync(wsRoot, wsLink);
}

const executed = [];
let dialogAnswer = undefined;
let rejectOpen = false;

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
    showOpenDialog: opts => {
      executed.push({dialog: opts});
      return Promise.resolve(dialogAnswer);
    },
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
    openTextDocument: () => Promise.resolve({}),
    saveAll: () => Promise.resolve(true),
    applyEdit: () => Promise.resolve(true),
  },
  commands: {
    executeCommand: (cmd, ...args) => {
      executed.push({cmd, args});
      if (cmd === 'vscode.openFolder' && rejectOpen) {
        return Promise.reject(new Error('refused by the editor'));
      }
      return Promise.resolve();
    },
  },
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
  view._api = {
    forward: () => {},
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
  return {view, posted};
}

function opens() {
  return executed
    .filter(e => e.cmd === 'vscode.openFolder')
    .map(e => e.args[0].fsPath);
}

function errors(posted) {
  return posted.filter(m => m.type === 'workDirError').map(m => m.text);
}

async function testOpenWorkDirOpensAFolder() {
  const {view, posted} = makeView();
  executed.length = 0;
  await view._handleMessage({type: 'openWorkDir', path: other});
  assert.deepStrictEqual(opens(), [other], 'a real folder is opened');
  assert.deepStrictEqual(errors(posted), [], 'and nothing is reported');
  view.dispose();
}

async function testOpenWorkDirRefusals() {
  const {view, posted} = makeView();
  executed.length = 0;
  const missing = path.join(tmp, 'missing');
  await view._handleMessage({type: 'openWorkDir', path: missing});
  await view._handleMessage({
    type: 'openWorkDir',
    path: path.join(tmp, 'a-file.txt'),
  });
  await view._handleMessage({type: 'openWorkDir', path: '   '});
  await view._handleMessage({type: 'openWorkDir', path: '/'});
  // A root reached through ".." segments.
  await view._handleMessage({
    type: 'openWorkDir',
    path: tmp + '/..'.repeat(12),
  });
  // The folder this window already shows, in another spelling.
  await view._handleMessage({
    type: 'openWorkDir',
    path: wsRoot + path.sep + '.',
  });
  if (process.platform !== 'win32') {
    await view._handleMessage({type: 'openWorkDir', path: rootLink});
    await view._handleMessage({type: 'openWorkDir', path: wsLink});
  }
  assert.deepStrictEqual(opens(), [], 'none of these opens a folder');
  const texts = errors(posted);
  assert.strictEqual(texts.length, process.platform !== 'win32' ? 8 : 6);
  assert.strictEqual(texts[0], 'Not a directory: ' + missing);
  assert.strictEqual(
    texts[1],
    'Not a directory: ' + path.join(tmp, 'a-file.txt'),
  );
  assert.strictEqual(texts[2], 'Not a directory: (empty path)');
  assert.ok(/root/.test(texts[3]), 'a literal root is refused');
  assert.ok(/root/.test(texts[4]), 'a root spelled with .. is refused');
  assert.ok(/already the working directory/.test(texts[5]));
  if (process.platform !== 'win32') {
    assert.ok(/root/.test(texts[6]), 'a symlink to the root is refused');
    assert.ok(
      /already the working directory/.test(texts[7]),
      'a symlink to the open folder is recognised',
    );
  }
  view.dispose();
}

async function testRejectedOpenIsReported() {
  const {view, posted} = makeView();
  executed.length = 0;
  rejectOpen = true;
  try {
    await view._handleMessage({type: 'openWorkDir', path: other});
  } finally {
    rejectOpen = false;
  }
  assert.deepStrictEqual(opens(), [other], 'the open was attempted');
  const texts = errors(posted);
  assert.strictEqual(texts.length, 1);
  assert.ok(/Could not open .*refused by the editor/.test(texts[0]));
  view.dispose();
}

async function testPickWorkDirUsesTheEditorDialog() {
  const {view, posted} = makeView();
  executed.length = 0;
  // Cancelled dialog: nothing happens.
  dialogAnswer = undefined;
  await view._handleMessage({type: 'pickWorkDir'});
  const dialog = executed.find(e => e.dialog).dialog;
  assert.strictEqual(dialog.canSelectFolders, true);
  assert.strictEqual(dialog.canSelectFiles, false);
  assert.strictEqual(dialog.canSelectMany, false);
  assert.strictEqual(dialog.defaultUri.fsPath, wsRoot);
  assert.deepStrictEqual(opens(), []);
  assert.deepStrictEqual(errors(posted), []);

  // A picked folder goes through the same checks as a typed one.
  dialogAnswer = [{fsPath: other}];
  await view._handleMessage({type: 'pickWorkDir'});
  assert.deepStrictEqual(opens(), [other]);
  dialogAnswer = [{fsPath: '/'}];
  await view._handleMessage({type: 'pickWorkDir'});
  assert.deepStrictEqual(opens(), [other], 'a picked root is not opened');
  assert.ok(/root/.test(errors(posted)[0]));
  view.dispose();
}

async function main() {
  const tests = [
    ['openWorkDir opens a real folder', testOpenWorkDirOpensAFolder],
    ['openWorkDir refusals are reported to the panel', testOpenWorkDirRefusals],
    ['a rejected vscode.openFolder is reported', testRejectedOpenIsReported],
    ['pickWorkDir uses the editor dialog', testPickWorkDirUsesTheEditorDialog],
  ];
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
  console.log(`all ${tests.length} workDirPanelHost tests passed`);
  process.exit(0);
}

main();
