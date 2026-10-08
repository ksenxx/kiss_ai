// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// The extension-host half of the "Working directory" panel: the webview
// posts `openWorkDir {path}` (typed path or a row of the opened-so-far
// list) or `pickWorkDir` (the folder button) -- neither names a tab, the
// working directory is ONE global value -- and the compiled
// SorcarSidebarView makes the folder's real path the daemon's global
// working directory (a `setWorkDir` command, nothing is sent as
// `recordWorkDir`) and answers `workDirPicked {path}` without a tabId;
// a path that is not a directory or is a file-system root (also one
// reached through `..` or a symlink) gets `workDirError {text}` instead
// and nothing goes to the daemon.  The host never opens the folder as
// the window's workspace: `vscode.openFolder` must not run, and the
// window's own folder is as valid a pick as any other.  The host caches
// the daemon's value (its own pick, `configData.config.work_dir`,
// `workDirChanged`) as the fallback directory of path-taking messages
// and the folder dialog's starting folder.  A `submit` is forwarded
// bare: the host adds no workDir, the daemon runs it in the global value.
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
  // Commands sent to the daemon, split by kind: submits, completions,
  // every directory sent as the global value, and everything else
  // (getConfig refreshes aside) as raw forwards.
  const forwarded = [];
  const runs = [];
  const completes = [];
  const sets = [];
  view._send = cmd => {
    const {type, ...fields} = cmd;
    if (type === 'submit') runs.push(fields);
    else if (type === 'complete') completes.push(fields);
    else if (type === 'setWorkDir') sets.push(cmd.workDir);
    else if (type !== 'getConfig') forwarded.push(cmd);
  };
  const posted = [];
  view._view = {
    visible: true,
    webview: {postMessage: m => posted.push(m)},
    show() {},
  };
  view._disposed = false;
  return {view, posted, forwarded, runs, completes, sets};
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

/** No workDirPicked / workDirError reply names a tab: the pick is global. */
function assertNoTabIds(posted) {
  const replies = posted.filter(
    m => m.type === 'workDirPicked' || m.type === 'workDirError',
  );
  assert.ok(
    replies.every(m => !('tabId' in m)),
    'replies carry no tabId: ' + JSON.stringify(replies),
  );
}

async function testOpenWorkDirMakesTheFolderGlobal() {
  const {view, posted, forwarded, sets} = makeView();
  executed.length = 0;
  assert.strictEqual(view._getWorkDir(), wsRoot, 'the window folder at first');
  await view._handleMessage({type: 'openWorkDir', path: other});
  assert.strictEqual(
    view._getWorkDir(),
    realOther,
    'the pick is the fallback directory from now on',
  );
  // The window's own folder is a legitimate pick too (the user may come
  // back to it from another folder), in any spelling.
  await view._handleMessage({
    type: 'openWorkDir',
    path: wsRoot + path.sep + '.',
  });
  if (process.platform !== 'win32') {
    await view._handleMessage({type: 'openWorkDir', path: otherLink});
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
  assertNoTabIds(posted);
  assert.deepStrictEqual(
    sets,
    expected,
    'each pick is sent to the daemon as the global working directory',
  );
  assert.deepStrictEqual(forwarded, [], 'nothing is forwarded raw');
  view.dispose();
}

async function testOpenWorkDirRefusals() {
  const {view, posted, forwarded, sets} = makeView();
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
  if (process.platform !== 'win32') {
    await view._handleMessage({type: 'openWorkDir', path: rootLink});
  }
  assert.deepStrictEqual(opens(), [], 'nothing is opened');
  assert.deepStrictEqual(picks(posted), [], 'none of these is picked');
  assert.deepStrictEqual(sets, [], 'nor sent to the daemon');
  assert.deepStrictEqual(forwarded, []);
  assert.strictEqual(view._getWorkDir(), wsRoot, 'the fallback is unchanged');
  assertNoTabIds(posted);
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
  const {view, posted, sets} = makeView();
  executed.length = 0;
  // Cancelled dialog: nothing happens.
  dialogAnswer = undefined;
  await view._handleMessage({type: 'pickWorkDir'});
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

  assert.deepStrictEqual(sets, []);

  // A picked folder goes through the same checks as a typed one.
  dialogAnswer = [{fsPath: other}];
  await view._handleMessage({type: 'pickWorkDir'});
  assert.deepStrictEqual(picks(posted), [realOther]);
  assert.deepStrictEqual(sets, [realOther]);
  assertNoTabIds(posted);
  dialogAnswer = [{fsPath: '/'}];
  await view._handleMessage({type: 'pickWorkDir'});
  assert.deepStrictEqual(picks(posted), [realOther], 'a root is not picked');
  assert.deepStrictEqual(sets, [realOther]);
  assert.ok(/root/.test(errors(posted)[0]));
  // The next dialog starts in the global directory, not the window's.
  const last = executed.filter(e => e.dialog).pop().dialog;
  assert.strictEqual(last.defaultUri.fsPath, realOther);
  assert.deepStrictEqual(opens(), [], 'no vscode.openFolder either way');
  view.dispose();
}

async function testDaemonValueIsTheHostFallback() {
  // The daemon's global value reaches the host through the client
  // messages configData (config.work_dir, passed on untouched) and
  // workDirChanged; both set the fallback directory and the folder
  // dialog's starting folder.
  const {view, posted} = makeView();
  const listeners = {};
  view._installClientListener({
    on: (event, cb) => {
      listeners[event] = cb;
    },
  });
  assert.strictEqual(view._getWorkDir(), wsRoot);
  listeners.message({type: 'configData', config: {work_dir: '/from/config'}});
  assert.strictEqual(view._getWorkDir(), '/from/config');
  listeners.message({type: 'configData', config: {work_dir: ''}});
  assert.strictEqual(
    view._getWorkDir(),
    '/from/config',
    'an empty work_dir does not erase the cached value',
  );
  listeners.message({type: 'workDirChanged', workDir: '/from/broadcast'});
  assert.strictEqual(view._getWorkDir(), '/from/broadcast');
  // Both messages reach the webview as the daemon sent them: the host
  // no longer rewrites work_dir to the window's workspace folder.
  const configs = posted.filter(m => m.type === 'configData');
  assert.deepStrictEqual(
    configs.map(m => m.config.work_dir),
    ['/from/config', ''],
    'configData.config.work_dir is passed on untouched',
  );
  const changes = posted.filter(m => m.type === 'workDirChanged');
  assert.deepStrictEqual(
    changes.map(m => m.workDir),
    ['/from/broadcast'],
  );
  executed.length = 0;
  dialogAnswer = undefined;
  await view._handleMessage({type: 'pickWorkDir'});
  const dialog = executed.find(e => e.dialog).dialog;
  assert.strictEqual(dialog.defaultUri.fsPath, '/from/broadcast');
  assert.deepStrictEqual(picks(posted), []);
  view.dispose();
}

async function testSubmitIsForwardedWithoutAWorkDir() {
  const {view, runs} = makeView();
  await view._handleMessage({type: 'openWorkDir', path: other});
  await view._handleMessage({
    type: 'submit',
    prompt: 'list files',
    model: 'm',
    attachments: [],
    tabId: 'tab-1',
  });
  assert.strictEqual(runs.length, 1);
  assert.strictEqual(runs[0].prompt, 'list files');
  assert.strictEqual(runs[0].tabId, 'tab-1');
  // The host stamps no directory of its own, not even the one it just
  // picked: the daemon runs every task in the global value.
  assert.ok(!('workDir' in runs[0]), 'the submit is forwarded bare');
  view.dispose();
}

async function testWebviewEditorContextFallsBackToTheFileTab() {
  // Without a visible VS Code editor the webview's own file tab (the
  // Monaco buffer the remote webapp also has) is the editor context of
  // a run and of a completion; with one, the native editor wins.
  const {view, runs, completes} = makeView();
  await view._handleMessage({
    type: 'submit',
    prompt: 'explain',
    model: 'm',
    attachments: [],
    tabId: 'tab-1',
    activeFile: '/ws/notes.md',
  });
  assert.strictEqual(runs[0].activeFile, '/ws/notes.md');
  await view._handleMessage({
    type: 'complete',
    query: 'exp',
    tabId: 'tab-1',
    activeFile: '/ws/notes.md',
    activeFileContent: 'buffer text',
  });
  assert.strictEqual(completes[0].activeFile, '/ws/notes.md');
  assert.strictEqual(completes[0].activeFileContent, 'buffer text');
  await view._handleMessage({type: 'complete', query: 'exp', tabId: 'tab-1'});
  assert.strictEqual(
    completes[1].activeFile,
    undefined,
    'no file tab: no context',
  );
  assert.strictEqual(completes[1].activeFileContent, undefined);

  const native = path.join(wsRoot, 'native.ts');
  vscodeStub.window.activeTextEditor = {document: {uri: {fsPath: native}}};
  vscodeStub.workspace.textDocuments = [
    {uri: {fsPath: native}, getText: () => 'native text'},
  ];
  try {
    await view._handleMessage({
      type: 'submit',
      prompt: 'explain',
      model: 'm',
      attachments: [],
      tabId: 'tab-3',
      activeFile: '/ws/notes.md',
    });
    assert.strictEqual(runs[1].activeFile, native, 'the visible editor wins');
    await view._handleMessage({
      type: 'complete',
      query: 'exp',
      tabId: 'tab-1',
      activeFile: '/ws/notes.md',
      activeFileContent: 'buffer text',
    });
    assert.strictEqual(completes[2].activeFile, native);
    assert.strictEqual(completes[2].activeFileContent, 'native text');
  } finally {
    vscodeStub.window.activeTextEditor = undefined;
    vscodeStub.workspace.textDocuments = [];
  }
  view.dispose();
}

async function main() {
  const tests = [
    [
      'openWorkDir sends setWorkDir and answers workDirPicked without opening a workspace',
      testOpenWorkDirMakesTheFolderGlobal,
    ],
    ['openWorkDir refusals are reported to the panel', testOpenWorkDirRefusals],
    ['pickWorkDir uses the editor dialog', testPickWorkDirUsesTheEditorDialog],
    [
      'configData / workDirChanged set the host fallback directory',
      testDaemonValueIsTheHostFallback,
    ],
    [
      'submit is forwarded without a workDir',
      testSubmitIsForwardedWithoutAWorkDir,
    ],
    [
      'the webview file tab is the editor context unless an editor is visible',
      testWebviewEditorContextFallsBackToTheFileTab,
    ],
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
