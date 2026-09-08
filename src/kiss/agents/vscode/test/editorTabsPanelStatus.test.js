// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end tests for editor-tabs mode's TAB-TITLE STATUS CIRCLES and
// finished-task reveal (out/SorcarPanelManager.js +
// out/SorcarSidebarView.js) against a real Unix-domain-socket daemon
// stub:
//  - `panelTitle {state:'running'}` paints a green circle that PULSES
//    (alternates with a hollow circle on the shared 750 ms clock);
//  - `state:'ok'` paints a steady solid green circle, `state:'fail'` a
//    steady solid red one, and no state paints no circle;
//  - `revealPanel` from the webview reveals the hosting editor tab
//    without stealing focus (preserveFocus);
//  - a serializer-revived panel drops the status circle a previous
//    session persisted into its title.

'use strict';

const assert = require('assert');
const fs = require('fs');
const net = require('net');
const os = require('os');
const path = require('path');
const Module = require('module');

const EXT_ROOT = path.join(__dirname, '..');
const OUT_DIR = path.join(EXT_ROOT, 'out');
assert.ok(
  fs.existsSync(path.join(OUT_DIR, 'SorcarPanelManager.js')),
  'compiled extension missing — run `npm run compile` first',
);

if (process.platform === 'win32') {
  console.log('  skipped on win32 (UDS test)');
  process.exit(0);
}

const GREEN = '\u{1F7E2} ';
const RED = '\u{1F534} ';
const HOLLOW = '\u{26AA} ';

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

const createdPanels = [];
let registeredSerializer = null;

function makeFakePanel(viewType, title) {
  const recvEmitter = new StubEventEmitter();
  const disposeEmitter = new StubEventEmitter();
  const viewStateEmitter = new StubEventEmitter();
  const panel = {
    viewType,
    title,
    iconPath: undefined,
    visible: true,
    active: true,
    revealCalls: [],
    disposed: false,
    webview: {
      options: {},
      html: '',
      cspSource: 'vscode-resource:',
      asWebviewUri: uri => makeUri(uri.fsPath),
      postMessage: () => Promise.resolve(true),
      onDidReceiveMessage: cb => recvEmitter.event(cb),
    },
    reveal: (column, preserveFocus) => {
      panel.revealCalls.push({column, preserveFocus});
    },
    onDidChangeViewState: cb => viewStateEmitter.event(cb),
    onDidDispose: cb => disposeEmitter.event(cb),
    dispose: () => {
      if (panel.disposed) return;
      panel.disposed = true;
      disposeEmitter.fire();
    },
    _recv: recvEmitter,
  };
  return panel;
}

const vscodeStub = {
  workspace: {
    workspaceFolders: [],
    getConfiguration: () => ({
      get: key => (key === 'editorTabsMode' ? true : ''),
      update: () => Promise.resolve(),
    }),
    onDidChangeWorkspaceFolders: () => ({dispose: () => {}}),
    openTextDocument: () =>
      Promise.resolve({uri: makeUri('/x'), getText: () => ''}),
    textDocuments: [],
  },
  EventEmitter: StubEventEmitter,
  Uri: {
    file: p => makeUri(p),
    joinPath: (base, ...parts) => makeUri(path.join(base.fsPath, ...parts)),
    parse: s => makeUri(s),
  },
  ProgressLocation: {Notification: 15},
  ViewColumn: {Active: -1, One: 1},
  window: {
    createWebviewPanel: (viewType, title) => {
      const panel = makeFakePanel(viewType, title);
      createdPanels.push(panel);
      return panel;
    },
    registerWebviewPanelSerializer: (viewType, serializer) => {
      registeredSerializer = {viewType, serializer};
      return {dispose: () => {}};
    },
    withProgress: (_opts, task) =>
      task(
        {report: () => {}},
        {onCancellationRequested: () => ({dispose: () => {}})},
      ),
    showInformationMessage: () => {},
    showWarningMessage: () => {},
    showErrorMessage: () => {},
    activeTextEditor: undefined,
    tabGroups: {all: []},
  },
  commands: {executeCommand: () => Promise.resolve()},
};

const origResolve = Module._resolveFilename;
Module._resolveFilename = function (request, parent, ...rest) {
  if (request === 'vscode') return require.resolve('./_vscode-stub.js');
  return origResolve.call(this, request, parent, ...rest);
};
global.__kissVscodeStub = vscodeStub;

const tmpHome = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-edstatus-'));
process.env.HOME = tmpHome;
process.env.USERPROFILE = tmpHome;
fs.mkdirSync(path.join(tmpHome, '.kiss'), {recursive: true});
const sockPath = path.join(tmpHome, '.kiss', 'sorcar.sock');

const server = net.createServer(sock => {
  sock.on('data', () => {});
  sock.on('error', () => {});
});

async function waitFor(predicate, message) {
  for (let i = 0; i < 200; i++) {
    if (predicate()) return;
    await new Promise(r => setTimeout(r, 20));
  }
  throw new Error(message || 'waitFor timed out');
}

const {SorcarPanelManager, CHAT_PANEL_VIEW_TYPE} = require(
  path.join(OUT_DIR, 'SorcarPanelManager.js'),
);

async function runTest() {
  server.listen(sockPath);
  const manager = new SorcarPanelManager(vscodeStub.Uri.file(EXT_ROOT));
  manager.registerSerializer();
  assert.strictEqual(registeredSerializer.viewType, CHAT_PANEL_VIEW_TYPE);

  manager.openNewChat();
  const panel = createdPanels[0];
  const m = /data-kiss-tab-id="([^"]+)"/.exec(panel.webview.html);
  assert.ok(m, 'panel html must carry data-kiss-tab-id');
  const tabId = m[1];

  // --- no state: plain title -------------------------------------------
  panel._recv.fire({type: 'panelTitle', title: 'fix the bug', tabId, state: ''});
  await waitFor(
    () => panel.title === 'fix the bug',
    'a stateless panelTitle must set the plain title',
  );

  // --- running: green circle that PULSES -------------------------------
  panel._recv.fire({
    type: 'panelTitle',
    title: 'fix the bug',
    tabId,
    state: 'running',
  });
  await waitFor(
    () => panel.title === GREEN + 'fix the bug',
    'a running task must paint the green circle immediately',
  );
  await waitFor(
    () => panel.title === HOLLOW + 'fix the bug',
    'the running circle must pulse to its dim phase',
  );
  await waitFor(
    () => panel.title === GREEN + 'fix the bug',
    'the running circle must pulse back to its bright phase',
  );

  // --- ok: steady solid green circle ------------------------------------
  panel._recv.fire({
    type: 'panelTitle',
    title: 'fix the bug',
    tabId,
    state: 'ok',
  });
  await waitFor(
    () => panel.title === GREEN + 'fix the bug',
    'a successful task must paint the solid green circle',
  );
  // The pulse clock must be gone: the title stays green past a period.
  await new Promise(r => setTimeout(r, 900));
  assert.strictEqual(
    panel.title,
    GREEN + 'fix the bug',
    'the ok circle must not pulse',
  );

  // --- fail: steady solid red circle -------------------------------------
  panel._recv.fire({
    type: 'panelTitle',
    title: 'fix the bug',
    tabId,
    state: 'fail',
  });
  await waitFor(
    () => panel.title === RED + 'fix the bug',
    'a failed task must paint the solid red circle',
  );
  await new Promise(r => setTimeout(r, 900));
  assert.strictEqual(
    panel.title,
    RED + 'fix the bug',
    'the fail circle must not pulse',
  );

  // --- revealPanel brings the tab forward without stealing focus --------
  assert.strictEqual(panel.revealCalls.length, 0);
  panel._recv.fire({type: 'revealPanel'});
  await waitFor(
    () => panel.revealCalls.length === 1,
    'revealPanel must reveal the hosting editor tab',
  );
  assert.strictEqual(
    panel.revealCalls[0].preserveFocus,
    true,
    'the reveal must not steal keyboard focus',
  );

  // --- a legitimate circle-leading title is never corrupted --------------
  panel._recv.fire({
    type: 'panelTitle',
    title: GREEN + 'deploy status',
    tabId,
    state: 'fail',
  });
  await waitFor(
    () => panel.title === RED + GREEN + 'deploy status',
    'a chat legitimately titled with a leading circle must keep it',
  );

  // --- a revived panel drops the persisted status circle ----------------
  const revived = makeFakePanel(CHAT_PANEL_VIEW_TYPE, GREEN + 'old chat');
  revived.active = false;
  await registeredSerializer.serializer.deserializeWebviewPanel(revived, {
    editorRootTabId: 'revived-tab-1',
  });
  assert.strictEqual(
    revived.title,
    'old chat',
    'revival must strip the previous session status circle',
  );

  // --- ... but strips only the ONE decoration it added itself ------------
  const revived2 = makeFakePanel(
    CHAT_PANEL_VIEW_TYPE,
    RED + GREEN + 'deploy status',
  );
  revived2.active = false;
  await registeredSerializer.serializer.deserializeWebviewPanel(revived2, {
    editorRootTabId: 'revived-tab-2',
  });
  assert.strictEqual(
    revived2.title,
    GREEN + 'deploy status',
    'revival must keep a legitimate leading circle in the chat title',
  );

  manager.dispose();
}

runTest()
  .then(() => {
    console.log('editorTabsPanelStatus: all tests passed');
    server.close();
    process.exit(0);
  })
  .catch(err => {
    console.error(err);
    process.exit(1);
  });
