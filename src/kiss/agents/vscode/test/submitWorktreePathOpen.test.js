// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// End-to-end test of the path-only submit shortcut against a pending
// worktree, through the REAL compiled extension host and the REAL
// daemon (test/_real_daemon.py) over the real local WSS endpoint: typing just
// `reports/analysis.html` and pressing Send must open the tab's worktree
// copy of the file in the editor — not launch an unintended agent run.
// The host forwards the webview's `submit` untouched; the daemon
// classifies it and answers this window with `promptOpened` and
// `openResolvedFile`, which the host opens natively.

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const Module = require('module');
const {startRealDaemon} = require('./_realDaemon.js');

// Absent from get_available_models(): a prompt that must still start a
// task ends at task_runner's "No model available" guard, no LLM call.
const UNAVAILABLE_MODEL = 'kiss-test-no-such-model';

class StubEventEmitter {
  constructor() {
    this._listeners = [];
    this.event = cb => {
      this._listeners.push(cb);
      return {
        dispose: () => {
          const i = this._listeners.indexOf(cb);
          if (i >= 0) this._listeners.splice(i, 1);
        },
      };
    };
  }
  fire(arg) {
    for (const cb of this._listeners.slice()) cb(arg);
  }
  dispose() {
    this._listeners = [];
  }
}

class StubCancellationTokenSource {
  constructor() {
    this.token = {onCancellationRequested: () => ({dispose: () => {}})};
  }
  dispose() {}
}

function makeUri(fsPath) {
  return {fsPath, scheme: 'file', toString: () => `file://${fsPath}`};
}

let workspaceFolders = [];

const vscodeStub = {
  workspace: {
    get workspaceFolders() {
      return workspaceFolders;
    },
    getConfiguration: () => ({get: () => 'stub-default-model'}),
    onDidChangeWorkspaceFolders: () => ({dispose: () => {}}),
    openTextDocument: () => Promise.resolve({getText: () => ''}),
    textDocuments: [],
  },
  EventEmitter: StubEventEmitter,
  CancellationTokenSource: StubCancellationTokenSource,
  Uri: {
    file: p => makeUri(p),
    joinPath: (base, ...parts) => makeUri(path.join(base.fsPath, ...parts)),
    parse: s => makeUri(s),
  },
  Position: class {},
  Range: class {},
  Selection: class {},
  TextEditorRevealType: {InCenter: 2, AtTop: 3},
  ProgressLocation: {Notification: 15},
  ViewColumn: {One: 1},
  window: {
    withProgress: (_opts, task) =>
      task(
        {report: () => {}},
        {onCancellationRequested: () => ({dispose: () => {}})},
      ),
    showInformationMessage: () => Promise.resolve(undefined),
    showWarningMessage: () => Promise.resolve(undefined),
    showErrorMessage: () => Promise.resolve(undefined),
    showTextDocument: () => Promise.resolve({}),
    activeTextEditor: undefined,
    tabGroups: {all: []},
  },
  commands: {
    executeCommand: () => Promise.resolve(),
  },
  extensions: {
    getExtension: () => undefined,
  },
};

const origResolve = Module._resolveFilename;
Module._resolveFilename = function (request, parent, ...rest) {
  if (request === 'vscode') return require.resolve('./_vscode-stub.js');
  return origResolve.call(this, request, parent, ...rest);
};
global.__kissVscodeStub = vscodeStub;

const tmpHome = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-wtsubmit-'));
const tmpDirs = [tmpHome];
process.env.HOME = tmpHome;
process.env.USERPROFILE = tmpHome;
process.env.KISS_HOME = path.join(tmpHome, '.kiss');
fs.mkdirSync(process.env.KISS_HOME, {recursive: true});

const {findUvPath} = require(path.join(__dirname, '..', 'out', 'kissPaths.js'));
const UV = findUvPath();
assert.ok(UV, 'this suite needs a real uv binary to run the real daemon');
let daemon = null;

function makeWebviewView() {
  const recv = new StubEventEmitter();
  const posted = [];
  const webview = {
    options: {},
    html: '',
    cspSource: 'vscode-resource:',
    asWebviewUri: uri => makeUri(uri.fsPath),
    postMessage: msg => {
      posted.push(msg);
      return Promise.resolve(true);
    },
    onDidReceiveMessage: cb => recv.event(cb),
  };
  const webviewView = {
    webview,
    visible: true,
    show: () => {},
    onDidChangeVisibility: () => ({dispose: () => {}}),
    onDidDispose: () => ({dispose: () => {}}),
  };
  return {webviewView, posted, fireMessage: m => recv.fire(m)};
}

async function waitFor(predicate, message, timeoutMs = 1500) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    const value = predicate();
    if (value) return value;
    await new Promise(r => setTimeout(r, 10));
  }
  throw new Error(message || 'waitFor timed out');
}

async function runTests() {
  const sourcePath = path.join(__dirname, '..', 'out', 'SorcarSidebarView.js');
  assert.ok(
    fs.existsSync(sourcePath),
    `compiled extension missing: ${sourcePath}`,
  );
  delete require.cache[require.resolve(sourcePath)];
  const {SorcarSidebarView} = require(sourcePath);

  const ws = fs.realpathSync(
    fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-wtsubmit-ws-')),
  );
  tmpDirs.push(ws);
  workspaceFolders = [{uri: makeUri(ws)}];

  // The report exists only in the tab's pending worktree.
  const wt = path.join(ws, '.kiss-worktrees', 'kiss_wt-1');
  fs.mkdirSync(path.join(wt, 'reports'), {recursive: true});
  const wtReport = path.join(wt, 'reports', 'analysis.html');
  fs.writeFileSync(wtReport, '<h1>report</h1>\n');

  daemon = await startRealDaemon(UV, ws, process.env);

  // reports/analysis.html is an HTML file, so a path-only submit renders
  // it in a webview panel tab (exactly like a clicked file link — both
  // routes share _openResolvedFile). The panel's asWebviewUri call
  // receives the opened file's directory and the panel title is its
  // basename, which together identify the file that was opened.
  const opened = [];
  vscodeStub.window.createWebviewPanel = (viewType, title) => ({
    viewType,
    title,
    webview: {
      html: '',
      asWebviewUri: uri => {
        opened.push(path.join(uri.fsPath, title));
        return makeUri(uri.fsPath);
      },
    },
    reveal: () => {},
    onDidDispose: () => ({dispose: () => {}}),
    dispose: () => {},
  });

  const view = new SorcarSidebarView(makeUri(path.join(__dirname, '..')));
  const wv = makeWebviewView();
  view.resolveWebviewView(wv.webviewView, {}, {});
  wv.fireMessage({type: 'ready', tabId: 'tab1', restoredTabs: []});
  await waitFor(
    () => wv.posted.find(m => m.type === 'tabs_state'),
    'the real daemon must answer ready',
    15000,
  );

  const started = tabId =>
    wv.posted.filter(
      m => m.type === 'status' && m.running === true && m.tabId === tabId,
    );
  const submit = (prompt, tabId) =>
    wv.fireMessage({
      type: 'submit',
      prompt,
      model: UNAVAILABLE_MODEL,
      attachments: [],
      useWorktree: false,
      workDir: ws,
      tabId,
    });

  // The worktree task's own event, emitted by the daemon: it records the
  // tab's pending worktree for path resolution and reaches this window.
  daemon.broadcast({
    type: 'worktree_created',
    worktreeDir: wt,
    branch: 'kiss/wt-1',
    tabId: 'tab1',
  });
  await waitFor(
    () => wv.posted.find(m => m.type === 'worktree_created'),
    'worktree_created must reach the webview',
    5000,
  );

  // 1. A path-only submit opens the pending worktree copy — no run.
  submit('reports/analysis.html', 'tab1');
  await waitFor(() => opened.length === 1, 'submit must open the file', 5000);
  assert.strictEqual(
    opened[0],
    wtReport,
    'a path-only submit must open the pending worktree copy',
  );
  await waitFor(
    () => wv.posted.find(m => m.type === 'promptOpened'),
    'the webview must be told the prompt opened a file, not a task',
  );
  assert.strictEqual(
    wv.posted.find(m => m.type === 'promptOpened').tabId,
    'tab1',
    'promptOpened must address the submitting tab',
  );
  assert.strictEqual(
    started('tab1').length,
    0,
    'a path-only submit that resolves must not start an agent task',
  );
  assert.ok(
    !wv.posted.some(m => m.type === 'openResolvedFile'),
    'the daemon\'s openResolvedFile is for the host, not the webview',
  );
  console.log('  ok - path-only submit opens the worktree copy, no run');

  // 1b. A workspace DIRECTORY must not shadow a pending-worktree FILE
  // at the same relative path: the daemon resolves a path-only submit
  // for regular files only, so the directory candidate is skipped and
  // the worktree copy opens.
  fs.mkdirSync(path.join(ws, 'shadow.html'), {recursive: true});
  fs.writeFileSync(path.join(wt, 'shadow.html'), '<h1>wt copy</h1>\n');
  submit('shadow.html', 'tab1');
  await waitFor(
    () => opened.length === 2,
    'submit must open the worktree copy behind the directory',
    5000,
  );
  assert.strictEqual(
    opened[1],
    path.join(wt, 'shadow.html'),
    'a workspace directory must not shadow the pending-worktree file',
  );
  assert.strictEqual(
    started('tab1').length,
    0,
    'the shadowed path-only submit must not start an agent task',
  );
  console.log('  ok - workspace dir does not shadow the worktree file');

  // 2. A non-path prompt still starts a task (which ends at once: the
  // model is unavailable).  The daemon reports `running: true` twice per
  // run (an immediate ack, then the start carrying startTs), so wait for
  // at least one.
  submit('summarize the repo', 'tab1');
  await waitFor(
    () => started('tab1').length >= 1,
    'a non-path prompt must start an agent task',
    15000,
  );
  assert.strictEqual(opened.length, 2, 'no extra file must be opened');
  console.log('  ok - a non-path prompt still starts a task');

  // 3. A prompt naming a DIRECTORY starts a task: directories resolve
  // for clickable links (checkPaths/openFile), but the path-only submit
  // shortcut is for regular files only — "reports" must not be swallowed
  // by an Explorer reveal.  tab2: tab1's task may still be winding down,
  // and a submit to a running tab is a follow-up, not a new run.
  wv.fireMessage({type: 'openTab', tabId: 'tab2', workDir: wt});
  submit('reports', 'tab2');
  await waitFor(
    () => started('tab2').length >= 1,
    'a directory-path prompt must start an agent task',
    15000,
  );
  assert.strictEqual(opened.length, 2, 'no extra file must be opened');
  console.log('  ok - a directory-path prompt still starts a task');

  // Let both tasks reach their end before the daemon is stopped.
  await waitFor(
    () =>
      ['tab1', 'tab2'].every(t =>
        wv.posted.some(
          m => m.type === 'status' && m.running === false && m.tabId === t,
        ),
      ),
    'both refused tasks must end',
    30000,
  );
  view.dispose();
}

runTests()
  .then(() => {
    console.log('submitWorktreePathOpen.test.js: all tests passed');
  })
  .catch(err => {
    console.error('FAIL:', err && err.message ? err.message : err);
    process.exitCode = 1;
  })
  .finally(async () => {
    if (daemon) await daemon.stop();
    for (const dir of tmpDirs.slice().reverse()) {
      fs.rmSync(dir, {recursive: true, force: true});
    }
  });
