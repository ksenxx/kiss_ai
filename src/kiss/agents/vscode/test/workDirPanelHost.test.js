// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// The extension-host half of the "Working directory" panel: the webview
// posts `openWorkDir {path}` (typed path or a row of the opened-so-far
// list) or `pickWorkDir` (the folder button), and the compiled
// SorcarSidebarView answers `workDirPicked {path}` with the folder's
// real path (and records it in the daemon's opened-so-far list through
// `recordWorkDir`) -- or `workDirError` for a path that is not a
// directory or a file-system root (also one reached through `..` or a
// symlink).  The host never opens the folder as the window's workspace:
// `vscode.openFolder` must not run, and the window's own folder is as
// valid a pick as any other.  A `submit` carrying the webview's
// `tabScopeWorkDir` passes it on to the daemon's `run`.
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
const otherLink = path.join(tmp, 'other-link');
if (process.platform !== 'win32') {
  fs.symlinkSync('/', rootLink);
  fs.symlinkSync(other, otherLink);
}
// The realpath of the temp dir (macOS puts it under a /private symlink).
const realOther = fs.realpathSync(other);
const realWsRoot = fs.realpathSync(wsRoot);

const executed = [];
let dialogAnswer = undefined;

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
  const forwarded = [];
  const runs = [];
  view._api = {
    forward: cmd => forwarded.push(cmd),
    run: fields => runs.push(fields),
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
  return {view, posted, forwarded, runs};
}

function opens() {
  return executed.filter(e => e.cmd === 'vscode.openFolder');
}

function errors(posted) {
  return posted.filter(m => m.type === 'workDirError').map(m => m.text);
}

function picks(posted) {
  return posted.filter(m => m.type === 'workDirPicked').map(m => m.path);
}

/** The tab named by every workDirPicked reply (must echo the request). */
function pickTabs(posted) {
  return posted.filter(m => m.type === 'workDirPicked').map(m => m.tabId);
}

function recorded(forwarded) {
  return forwarded.filter(c => c.type === 'recordWorkDir').map(c => c.path);
}

async function testOpenWorkDirPicksAFolderForTheTab() {
  const {view, posted, forwarded} = makeView();
  executed.length = 0;
  await view._handleMessage({type: 'openWorkDir', path: other, tabId: 'tab-a'});
  // The window's own folder is a legitimate pick too (the tab may be
  // brought back from another folder), in any spelling.
  await view._handleMessage({
    type: 'openWorkDir',
    path: wsRoot + path.sep + '.',
    tabId: 'tab-a',
  });
  if (process.platform !== 'win32') {
    await view._handleMessage({
      type: 'openWorkDir',
      path: otherLink,
      tabId: 'tab-a',
    });
  }
  assert.deepStrictEqual(
    opens(),
    [],
    'the folder is never opened as the workspace',
  );
  assert.deepStrictEqual(errors(posted), [], 'and nothing is reported');
  const expected = [realOther, realWsRoot];
  if (process.platform !== 'win32') expected.push(realOther);
  assert.deepStrictEqual(
    picks(posted),
    expected,
    'each pick is answered with the real path',
  );
  assert.deepStrictEqual(
    pickTabs(posted),
    expected.map(() => 'tab-a'),
    'each reply names the tab that asked',
  );
  assert.deepStrictEqual(
    recorded(forwarded),
    expected,
    'each pick lands in the daemon opened-so-far list',
  );
  assert.ok(
    !forwarded.some(c => c.type === 'setWorkDir'),
    'the connection pin is left alone',
  );
  view.dispose();
}

async function testOpenWorkDirRefusals() {
  const {view, posted, forwarded} = makeView();
  executed.length = 0;
  const missing = path.join(tmp, 'missing');
  await view._handleMessage({
    type: 'openWorkDir',
    path: missing,
    tabId: 'tab-a',
  });
  await view._handleMessage({
    type: 'openWorkDir',
    path: path.join(tmp, 'a-file.txt'),
    tabId: 'tab-a',
  });
  await view._handleMessage({type: 'openWorkDir', path: '   ', tabId: 'tab-a'});
  await view._handleMessage({type: 'openWorkDir', path: '/', tabId: 'tab-a'});
  // A root reached through ".." segments.
  await view._handleMessage({
    type: 'openWorkDir',
    path: tmp + '/..'.repeat(12),
    tabId: 'tab-a',
  });
  if (process.platform !== 'win32') {
    await view._handleMessage({
      type: 'openWorkDir',
      path: rootLink,
      tabId: 'tab-a',
    });
  }
  assert.deepStrictEqual(opens(), [], 'nothing is opened');
  assert.deepStrictEqual(picks(posted), [], 'none of these is picked');
  assert.deepStrictEqual(recorded(forwarded), [], 'nor recorded');
  const texts = errors(posted);
  assert.strictEqual(texts.length, process.platform !== 'win32' ? 6 : 5);
  assert.strictEqual(texts[0], 'Not a directory: ' + missing);
  assert.strictEqual(
    texts[1],
    'Not a directory: ' + path.join(tmp, 'a-file.txt'),
  );
  assert.strictEqual(texts[2], 'Not a directory: (empty path)');
  assert.ok(/root/.test(texts[3]), 'a literal root is refused');
  assert.ok(/root/.test(texts[4]), 'a root spelled with .. is refused');
  if (process.platform !== 'win32') {
    assert.ok(/root/.test(texts[5]), 'a symlink to the root is refused');
  }
  view.dispose();
}

async function testPickWorkDirUsesTheEditorDialog() {
  const {view, posted, forwarded} = makeView();
  executed.length = 0;
  // Cancelled dialog: nothing happens.
  dialogAnswer = undefined;
  await view._handleMessage({type: 'pickWorkDir', tabId: 'tab-b'});
  const dialog = executed.find(e => e.dialog).dialog;
  assert.strictEqual(dialog.canSelectFolders, true);
  assert.strictEqual(dialog.canSelectFiles, false);
  assert.strictEqual(dialog.canSelectMany, false);
  assert.strictEqual(dialog.defaultUri.fsPath, wsRoot);
  assert.ok(
    !/open/i.test(dialog.openLabel),
    'the button does not promise to open the folder: ' + dialog.openLabel,
  );
  assert.deepStrictEqual(picks(posted), []);
  assert.deepStrictEqual(errors(posted), []);

  // A picked folder goes through the same checks as a typed one.
  dialogAnswer = [{fsPath: other}];
  await view._handleMessage({type: 'pickWorkDir', tabId: 'tab-b'});
  assert.deepStrictEqual(picks(posted), [realOther]);
  assert.deepStrictEqual(pickTabs(posted), ['tab-b']);
  assert.deepStrictEqual(recorded(forwarded), [realOther]);
  dialogAnswer = [{fsPath: '/'}];
  await view._handleMessage({type: 'pickWorkDir', tabId: 'tab-b'});
  assert.deepStrictEqual(picks(posted), [realOther], 'a root is not picked');
  assert.ok(/root/.test(errors(posted)[0]));
  assert.deepStrictEqual(opens(), [], 'no vscode.openFolder either way');
  view.dispose();
}

async function testSubmitPassesTheTabScope() {
  const {view, runs} = makeView();
  await view._handleMessage({
    type: 'submit',
    prompt: 'list files',
    model: 'm',
    attachments: [],
    tabId: 'tab-1',
    workDir: other,
    tabScopeWorkDir: wsRoot,
  });
  await view._handleMessage({
    type: 'submit',
    prompt: 'list files',
    model: 'm',
    attachments: [],
    tabId: 'tab-2',
  });
  assert.strictEqual(runs.length, 2);
  assert.strictEqual(runs[0].workDir, other, 'the tab dir is the run dir');
  assert.strictEqual(
    runs[0].tabScopeWorkDir,
    wsRoot,
    'the tab stays scoped to the window workspace',
  );
  assert.strictEqual(runs[1].workDir, wsRoot, 'no tab dir: the workspace');
  assert.strictEqual(
    runs[1].tabScopeWorkDir,
    undefined,
    'no scope override for an ordinary run',
  );
  view.dispose();
}

async function main() {
  const tests = [
    [
      'openWorkDir answers workDirPicked without opening a workspace',
      testOpenWorkDirPicksAFolderForTheTab,
    ],
    ['openWorkDir refusals are reported to the panel', testOpenWorkDirRefusals],
    ['pickWorkDir uses the editor dialog', testPickWorkDirUsesTheEditorDialog],
    ['submit passes tabScopeWorkDir to run', testSubmitPassesTheTabScope],
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
