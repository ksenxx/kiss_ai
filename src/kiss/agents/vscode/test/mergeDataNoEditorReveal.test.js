// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// E2E: a merge produced by a task running in a BACKGROUND chat tab must not
// yank the editor away from the user. The real compiled MergeManager is
// driven through the real compiled SorcarSidebarView over a real
// Unix-domain-socket daemon; only the `vscode` host surface is stubbed
// (it is not available outside a real extension host).

const assert = require('assert');
const fs = require('fs');
const net = require('net');
const os = require('os');
const path = require('path');
const Module = require('module');

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

function makeUri(fsPath) {
  return {fsPath, scheme: 'file', toString: () => `file://${fsPath}`};
}

// Every way the extension can pull an editor in front of the user.
const reveals = [];

function recordReveal(kind, fsPath, opts) {
  reveals.push({kind, fsPath, preserveFocus: !!(opts && opts.preserveFocus)});
}

function makeDoc(fp) {
  return {
    uri: makeUri(fp),
    isDirty: false,
    getText: () => fs.readFileSync(fp, 'utf8'),
    get lineCount() {
      return fs.readFileSync(fp, 'utf8').split('\n').length;
    },
    lineAt(i) {
      const t = fs.readFileSync(fp, 'utf8').split('\n')[i] || '';
      return {text: t, range: {end: {line: i, character: t.length}}};
    },
  };
}

let workspaceFolders = [];

const vscodeStub = {
  Uri: {
    file: p => makeUri(p),
    joinPath: (base, ...parts) => makeUri(path.join(base.fsPath, ...parts)),
    parse: s => makeUri(s),
  },
  Position: class {
    constructor(line, character) {
      this.line = line;
      this.character = character;
    }
  },
  Range: class {
    constructor(a, b, c, d) {
      this.start = {line: a, character: b};
      this.end = {line: c, character: d};
    }
  },
  Selection: class {
    constructor(a, b, c, d) {
      this.anchor = {line: a, character: b};
      this.active = {line: c, character: d};
    }
  },
  EventEmitter: StubEventEmitter,
  TextEditorRevealType: {InCenter: 2},
  ViewColumn: {One: 1},
  ProgressLocation: {Notification: 15},
  WorkspaceEdit: class {
    constructor() {
      this._textOps = [];
      this._deletes = [];
    }
    get size() {
      return this._textOps.length;
    }
    insert() {}
    replace() {}
    deleteFile() {}
  },
  window: {
    visibleTextEditors: [],
    activeTextEditor: undefined,
    tabGroups: {all: []},
    createTextEditorDecorationType: () => ({dispose() {}}),
    onDidChangeVisibleTextEditors: () => ({dispose() {}}),
    withProgress: (_opts, task) =>
      task(
        {report: () => {}},
        {onCancellationRequested: () => ({dispose: () => {}})},
      ),
    showInformationMessage: () => undefined,
    showWarningMessage: () => undefined,
    showErrorMessage: () => undefined,
    showTextDocument: (doc, opts) => {
      recordReveal('showTextDocument', doc.uri.fsPath, opts);
      return Promise.resolve({
        document: doc,
        selection: null,
        revealRange: () => recordReveal('revealRange', doc.uri.fsPath, null),
        setDecorations: () => {},
        edit: () => Promise.resolve(true),
      });
    },
  },
  workspace: {
    get workspaceFolders() {
      return workspaceFolders;
    },
    textDocuments: [],
    getConfiguration: () => ({get: () => 'stub-default-model'}),
    onDidChangeWorkspaceFolders: () => ({dispose: () => {}}),
    onWillSaveTextDocument: () => ({dispose() {}}),
    onDidSaveTextDocument: () => ({dispose() {}}),
    openTextDocument: uri =>
      Promise.resolve(makeDoc(uri.fsPath !== undefined ? uri.fsPath : uri)),
    saveAll: () => Promise.resolve(true),
    applyEdit: () => Promise.resolve(true),
  },
  commands: {
    executeCommand: (cmd, uri, opts) => {
      if (cmd === 'vscode.open') {
        recordReveal('vscode.open', uri && uri.fsPath, opts);
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

const tmpHome = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-mergereveal-'));
process.env.HOME = tmpHome;
process.env.USERPROFILE = tmpHome;
fs.mkdirSync(path.join(tmpHome, '.kiss'), {recursive: true});
const sockPath = path.join(tmpHome, '.kiss', 'sorcar.sock');

if (process.platform === 'win32') {
  console.log('  skipped on win32 (UDS test)');
  fs.rmSync(tmpHome, {recursive: true, force: true});
  process.exit(0);
}

let lastServerSock = null;
const server = net.createServer(sock => {
  lastServerSock = sock;
  sock.on('data', () => {});
});

function daemonSend(msg) {
  assert.ok(lastServerSock, 'daemon has no connected client socket');
  lastServerSock.write(JSON.stringify(msg) + '\n');
  return new Promise(r => setTimeout(r, 120));
}

async function waitForClient() {
  for (let i = 0; i < 100 && !lastServerSock; i++) {
    await new Promise(r => setTimeout(r, 20));
  }
  assert.ok(lastServerSock, 'client never connected to daemon');
}

function makeWebviewView() {
  const posted = [];
  const shows = [];
  const recvEmitter = new StubEventEmitter();
  const disposeEmitter = new StubEventEmitter();
  const visEmitter = new StubEventEmitter();
  const webview = {
    options: {},
    html: '',
    cspSource: 'vscode-resource:',
    asWebviewUri: uri => makeUri(uri.fsPath),
    postMessage: msg => {
      posted.push(msg);
      return Promise.resolve(true);
    },
    onDidReceiveMessage: cb => recvEmitter.event(cb),
  };
  const webviewView = {
    webview,
    visible: true,
    show: preserveFocus => shows.push(preserveFocus),
    onDidChangeVisibility: cb => visEmitter.event(cb),
    onDidDispose: cb => disposeEmitter.event(cb),
  };
  return {webviewView, posted, shows, fireMessage: m => recvEmitter.fire(m)};
}

// Build a real on-disk merge payload: a work file plus its base snapshot.
function makeMergePayload(work, mergeDir, name) {
  const cur = path.join(work, name);
  const base = path.join(mergeDir, 'base', name);
  fs.writeFileSync(base, 'alpha\nbeta\ngamma\n');
  fs.writeFileSync(cur, 'alpha\nBETA\ngamma\n');
  return {
    work_dir: work,
    files: [
      {
        name: name,
        base: base,
        current: cur,
        target: cur,
        // One replaced line: base line 1 -> current line 1.
        hunks: [{bs: 1, bc: 1, cs: 1, cc: 1}],
      },
    ],
  };
}

async function startView() {
  const sourcePath = path.join(__dirname, '..', 'out', 'SorcarSidebarView.js');
  assert.ok(
    fs.existsSync(sourcePath),
    `compiled extension missing: ${sourcePath} — run \`tsc -p .\` first`,
  );
  delete require.cache[require.resolve(sourcePath)];
  const {SorcarSidebarView} = require(sourcePath);

  const view = new SorcarSidebarView(makeUri(path.join(__dirname, '..')));
  const wv = makeWebviewView();
  view.resolveWebviewView(wv.webviewView, {}, {});
  return {view, wv};
}

async function testBackgroundMergeDoesNotRevealEditor(work, mergeDir) {
  reveals.length = 0;
  const {view, wv} = await startView();

  const BG_TAB = 'tab-bg-merge';
  const USER_TAB = 'tab-user-front';

  wv.fireMessage({type: 'ready', tabId: BG_TAB, restoredTabs: []});
  await waitForClient();
  wv.fireMessage({
    type: 'submit',
    prompt: 'long background task',
    model: 'm',
    tabId: BG_TAB,
  });
  await daemonSend({
    type: 'status',
    running: true,
    tabId: BG_TAB,
    startTs: Date.now(),
  });

  // The user opens and works in a different chat tab.
  wv.fireMessage({type: 'activeTabChanged', tabId: USER_TAB});

  // The still-running background task produces a merge.
  await daemonSend({
    type: 'merge_data',
    tabId: BG_TAB,
    hunk_count: 1,
    data: makeMergePayload(work, mergeDir, 'bg.txt'),
  });
  await new Promise(r => setTimeout(r, 250));

  const stealing = reveals.filter(r => !r.preserveFocus);
  assert.deepStrictEqual(
    stealing,
    [],
    'a merge from a background chat tab must not pull an editor in front ' +
      `of the user (got ${JSON.stringify(reveals)})`,
  );
  assert.strictEqual(
    wv.shows.length,
    0,
    'a background merge must not force-reveal the sidebar either',
  );

  if (typeof view.dispose === 'function') view.dispose();
  console.log('  ok - background merge_data never reveals an editor');
}

async function testActiveTabMergeStillRevealsEditor(work, mergeDir) {
  reveals.length = 0;
  const {view, wv} = await startView();

  const TAB = 'tab-active-merge';
  wv.fireMessage({type: 'ready', tabId: TAB, restoredTabs: []});
  await waitForClient();
  wv.fireMessage({
    type: 'submit',
    prompt: 'foreground task',
    model: 'm',
    tabId: TAB,
  });
  await daemonSend({
    type: 'status',
    running: true,
    tabId: TAB,
    startTs: Date.now(),
  });
  wv.fireMessage({type: 'activeTabChanged', tabId: TAB});

  await daemonSend({
    type: 'merge_data',
    tabId: TAB,
    hunk_count: 1,
    data: makeMergePayload(work, mergeDir, 'fg.txt'),
  });
  await new Promise(r => setTimeout(r, 250));

  assert.ok(
    reveals.some(r => r.kind === 'showTextDocument' && !r.preserveFocus),
    'a merge for the tab the user is looking at must still open the diff ' +
      `(got ${JSON.stringify(reveals)})`,
  );

  if (typeof view.dispose === 'function') view.dispose();
  console.log('  ok - active-tab merge_data still reveals the editor');
}

async function runTests() {
  await new Promise((res, rej) =>
    server.listen(sockPath, err => (err ? rej(err) : res())),
  );

  const ws = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-mergereveal-ws-'));
  workspaceFolders = [{uri: makeUri(ws)}];
  const work = path.join(ws, 'work');
  const mergeDir = path.join(ws, 'merge');
  fs.mkdirSync(work, {recursive: true});
  fs.mkdirSync(path.join(mergeDir, 'base'), {recursive: true});

  await testBackgroundMergeDoesNotRevealEditor(work, mergeDir);
  await testActiveTabMergeStillRevealsEditor(work, mergeDir);

  fs.rmSync(ws, {recursive: true, force: true});
}

function cleanup() {
  try {
    if (lastServerSock) lastServerSock.destroy();
  } catch {}
  try {
    server.close();
  } catch {}
  try {
    fs.unlinkSync(sockPath);
  } catch {}
  fs.rmSync(tmpHome, {recursive: true, force: true});
}

runTests().then(
  () => {
    cleanup();
    console.log('\nAll tests passed');
    process.exit(0);
  },
  err => {
    console.error('FAIL:', err && err.message ? err.message : err);
    cleanup();
    process.exit(1);
  },
);
