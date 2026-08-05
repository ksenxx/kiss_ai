// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// E2E race regression: the permission to pull an editor in front of the
// user must be a LIVE decision, not a snapshot taken when the merge
// started. `_doOpenMerge` awaits several host operations (saveAll,
// openTextDocument, applyEdit) before it opens the diff; if the user
// switches to another chat tab -- or closes the merging tab -- during
// those awaits, nothing may be revealed any more.
//
// The real compiled MergeManager is driven through the real compiled
// SorcarSidebarView over a real Unix-domain-socket daemon; only the
// `vscode` host surface is stubbed.

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

// A gate the test closes so one awaited host call hangs mid-merge.
let saveAllGate = null;

function openSaveAllGate() {
  let release;
  const promise = new Promise(r => {
    release = r;
  });
  let entered;
  const entry = new Promise(r => {
    entered = r;
  });
  saveAllGate = {promise, release, entry, entered, hits: 0};
  return saveAllGate;
}

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
    saveAll: () => {
      if (saveAllGate) {
        saveAllGate.hits += 1;
        saveAllGate.entered();
        return saveAllGate.promise.then(() => true);
      }
      return Promise.resolve(true);
    },
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

const tmpHome = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-mergerace-'));
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

// The user starts a merge on the tab they are looking at, then switches
// away while the merge is still awaiting a host operation. By the time
// the diff would be opened the permission is gone, so nothing may be
// revealed.
async function testTabSwitchDuringMergeCancelsReveal(work, mergeDir) {
  reveals.length = 0;
  const {view, wv} = await startView();

  const MERGE_TAB = 'tab-merging';
  const OTHER_TAB = 'tab-user-moved-to';

  wv.fireMessage({type: 'ready', tabId: MERGE_TAB, restoredTabs: []});
  await waitForClient();
  wv.fireMessage({
    type: 'submit',
    prompt: 'task that merges',
    model: 'm',
    tabId: MERGE_TAB,
  });
  await daemonSend({
    type: 'status',
    running: true,
    tabId: MERGE_TAB,
    startTs: Date.now(),
  });
  wv.fireMessage({type: 'activeTabChanged', tabId: MERGE_TAB});

  const gate = openSaveAllGate();
  await daemonSend({
    type: 'merge_data',
    tabId: MERGE_TAB,
    hunk_count: 1,
    data: makeMergePayload(work, mergeDir, 'race.txt'),
  });
  // The merge is now parked inside `saveAll`, before any editor work.
  await gate.entry;
  assert.deepStrictEqual(
    reveals,
    [],
    'nothing may be revealed before the merge gets past saveAll',
  );

  // The user moves to a different chat tab while the merge is blocked.
  wv.fireMessage({type: 'activeTabChanged', tabId: OTHER_TAB});

  gate.release();
  saveAllGate = null;
  await new Promise(r => setTimeout(r, 300));

  assert.deepStrictEqual(
    reveals,
    [],
    'the user switched chat tabs while the merge was still preparing, so ' +
      'the merge must no longer pull an editor in front of them ' +
      `(got ${JSON.stringify(reveals)})`,
  );
  assert.strictEqual(
    wv.shows.length,
    0,
    'a merge that lost the foreground must not force-reveal the sidebar',
  );

  if (typeof view.dispose === 'function') view.dispose();
  console.log('  ok - tab switch during a blocked merge cancels the reveal');
}

// The mirror case: a merge queued behind an in-flight one (the
// `_pendingMerge` re-entrancy path) must also re-check the live
// permission rather than replay the snapshot taken when it was queued.
async function testQueuedMergeRechecksLivePermission(work, mergeDir) {
  reveals.length = 0;
  const {view, wv} = await startView();

  const MERGE_TAB = 'tab-merging-queued';
  const OTHER_TAB = 'tab-user-elsewhere';

  wv.fireMessage({type: 'ready', tabId: MERGE_TAB, restoredTabs: []});
  await waitForClient();
  wv.fireMessage({
    type: 'submit',
    prompt: 'task that merges twice',
    model: 'm',
    tabId: MERGE_TAB,
  });
  await daemonSend({
    type: 'status',
    running: true,
    tabId: MERGE_TAB,
    startTs: Date.now(),
  });
  wv.fireMessage({type: 'activeTabChanged', tabId: MERGE_TAB});

  const gate = openSaveAllGate();
  await daemonSend({
    type: 'merge_data',
    tabId: MERGE_TAB,
    hunk_count: 1,
    data: makeMergePayload(work, mergeDir, 'queued-a.txt'),
  });
  await gate.entry;

  // A second merge arrives for the same tab while the first is parked;
  // it is stored as the pending merge with its own reveal permission.
  await daemonSend({
    type: 'merge_data',
    tabId: MERGE_TAB,
    hunk_count: 1,
    data: makeMergePayload(work, mergeDir, 'queued-b.txt'),
  });

  wv.fireMessage({type: 'activeTabChanged', tabId: OTHER_TAB});

  gate.release();
  saveAllGate = null;
  await new Promise(r => setTimeout(r, 400));

  assert.deepStrictEqual(
    reveals,
    [],
    'neither the in-flight merge nor the one queued behind it may reveal ' +
      `an editor after the user left the tab (got ${JSON.stringify(reveals)})`,
  );

  if (typeof view.dispose === 'function') view.dispose();
  console.log('  ok - queued merge re-checks the live reveal permission');
}

// Control case: if the user never leaves, the merge still opens the diff
// after the blocked host operation completes.
async function testStayingOnTabStillReveals(work, mergeDir) {
  reveals.length = 0;
  const {view, wv} = await startView();

  const TAB = 'tab-stays-front';
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

  const gate = openSaveAllGate();
  await daemonSend({
    type: 'merge_data',
    tabId: TAB,
    hunk_count: 1,
    data: makeMergePayload(work, mergeDir, 'stay.txt'),
  });
  await gate.entry;
  gate.release();
  saveAllGate = null;
  await new Promise(r => setTimeout(r, 300));

  assert.ok(
    reveals.some(r => r.kind === 'showTextDocument' && !r.preserveFocus),
    'a merge for the tab the user stayed on must still open the diff ' +
      `(got ${JSON.stringify(reveals)})`,
  );

  if (typeof view.dispose === 'function') view.dispose();
  console.log('  ok - staying on the merging tab still reveals the diff');
}

async function runTests() {
  await new Promise((res, rej) =>
    server.listen(sockPath, err => (err ? rej(err) : res())),
  );

  const ws = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-mergerace-ws-'));
  workspaceFolders = [{uri: makeUri(ws)}];
  const work = path.join(ws, 'work');
  const mergeDir = path.join(ws, 'merge');
  fs.mkdirSync(work, {recursive: true});
  fs.mkdirSync(path.join(mergeDir, 'base'), {recursive: true});

  await testTabSwitchDuringMergeCancelsReveal(work, mergeDir);
  await testQueuedMergeRechecksLivePermission(work, mergeDir);
  await testStayingOnTabStillReveals(work, mergeDir);

  fs.rmSync(ws, {recursive: true, force: true});
}

function cleanup() {
  saveAllGate = null;
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
