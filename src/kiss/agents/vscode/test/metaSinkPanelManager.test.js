// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end tests for the host side of the Task Info relay
// (out/SorcarPanelManager.js + out/SorcarSidebarView.js) against a
// REAL local-WSS daemon stub:
//  - setMetaSink pushes the current state right away (the (null, null)
//    placeholders when no panel exists or none reported yet);
//  - the chat panel webview's `metaUpdate` {values, taskUpdate} reaches
//    the sink, and the sink falls back to the placeholders when the
//    panel closes (no `closeTab` goes to the daemon: closing the panel
//    closes the surface only);
//  - open() reveals the one existing panel instead of creating another;
//  - refreshActiveTaskUpdate posts `refreshTaskUpdate` to the panel's
//    webview, and is a no-op without a panel;
//  - the panel webview's `getTaskUpdate` is forwarded to the daemon with
//    exactly {tabId, knownSig, token, refresh}, and the daemon's direct
//    `taskUpdate` reply is relayed back into the webview;
//  - SorcarSidebarView.postMetaState before the webview resolves is
//    caught up on the webview's `ready` (the meta view's late-resolve
//    path), and its webview's `metaRefresh` calls onMetaRefresh.

'use strict';

const assert = require('assert');
const fs = require('fs');
const net = require('net');
const os = require('os');
const path = require('path');
const Module = require('module');
const {createFakeDaemon} = require('./fakeDaemon');

const EXT_ROOT = path.join(__dirname, '..');
const OUT_DIR = path.join(EXT_ROOT, 'out');
assert.ok(
  fs.existsSync(path.join(OUT_DIR, 'SorcarPanelManager.js')),
  'compiled extension missing — run `npm run compile` first',
);

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

function makeFakePanel(viewType, title) {
  const recvEmitter = new StubEventEmitter();
  const disposeEmitter = new StubEventEmitter();
  const viewStateEmitter = new StubEventEmitter();
  const posted = [];
  const panel = {
    viewType,
    title,
    iconPath: undefined,
    visible: true,
    active: true,
    disposed: false,
    webview: {
      options: {},
      html: '',
      cspSource: 'vscode-resource:',
      asWebviewUri: uri => makeUri(uri.fsPath),
      postMessage: msg => {
        posted.push(msg);
        return Promise.resolve(true);
      },
      onDidReceiveMessage: cb => recvEmitter.event(cb),
    },
    reveal: () => {},
    onDidChangeViewState: cb => viewStateEmitter.event(cb),
    onDidDispose: cb => disposeEmitter.event(cb),
    dispose: () => {
      if (panel.disposed) return;
      panel.disposed = true;
      disposeEmitter.fire();
    },
    _posted: posted,
    _recv: recvEmitter,
    _viewState: viewStateEmitter,
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
    registerWebviewPanelSerializer: () => ({dispose: () => {}}),
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

const tmpHome = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-metasink-'));
process.env.HOME = tmpHome;
process.env.USERPROFILE = tmpHome;
fs.mkdirSync(path.join(tmpHome, '.kiss'), {recursive: true});
const endpointPath = path.join(tmpHome, '.kiss', 'sorcar-local.json');

const serverSockets = [];
const daemonCommands = [];
const server = createFakeDaemon(sock => {
  serverSockets.push(sock);
  let buf = '';
  sock.on('data', chunk => {
    buf += chunk.toString('utf8');
    let nl;
    while ((nl = buf.indexOf('\n')) >= 0) {
      const line = buf.slice(0, nl);
      buf = buf.slice(nl + 1);
      if (line.trim()) daemonCommands.push(JSON.parse(line));
    }
  });
  sock.on('error', () => {});
});

function daemonBroadcast(msg) {
  const line = JSON.stringify(msg) + '\n';
  for (const sock of serverSockets) {
    if (!sock.destroyed) sock.write(line);
  }
  return new Promise(r => setTimeout(r, 80));
}

async function waitFor(predicate, message) {
  for (let i = 0; i < 150; i++) {
    if (predicate()) return;
    await new Promise(r => setTimeout(r, 20));
  }
  throw new Error(message || 'waitFor timed out');
}

const {SorcarPanelManager} = require(
  path.join(OUT_DIR, 'SorcarPanelManager.js'),
);
const {SorcarSidebarView} = require(
  path.join(OUT_DIR, 'SorcarSidebarView.js'),
);

async function runTest() {
  server.listen(endpointPath);
  const manager = new SorcarPanelManager(vscodeStub.Uri.file(EXT_ROOT));

  // --- setMetaSink pushes the placeholder state right away -------------
  const sinkCalls = [];
  manager.setMetaSink((values, taskUpdate) => {
    sinkCalls.push({values, taskUpdate});
  });
  assert.deepStrictEqual(
    sinkCalls,
    [{values: null, taskUpdate: null}],
    'no panels: the sink starts on the placeholder state',
  );

  // --- refreshActiveTaskUpdate without a panel is a no-op --------------
  manager.refreshActiveTaskUpdate();
  assert.strictEqual(createdPanels.length, 0, 'no panel is created by a refresh');

  // --- the panel's metaUpdate reaches the sink --------------------------
  const controllerA = manager.open();
  assert.strictEqual(createdPanels.length, 1, 'open() creates the panel');
  assert.strictEqual(
    manager.controller(),
    controllerA,
    'controller() is the panel controller open() returned',
  );
  const panelA = createdPanels[0];
  assert.strictEqual(panelA.viewType, 'kissSorcar.chatTab');
  assert.deepStrictEqual(
    sinkCalls[sinkCalls.length - 1],
    {values: null, taskUpdate: null},
    'a fresh panel has no cached values yet',
  );
  const valuesA = {
    tokens: '1.00K',
    cost: '$0.10',
    steps: '3',
    time: '5.0s',
    timeColor: 'var(--red)',
    workdir: '/a',
    maxBudget: '$10.00',
  };
  const updateA = {
    content: '<h4>So far</h4><ul><li>read the spec</li></ul>',
    running: false,
    updatedAt: 1790000000000,
    cost: 0.04,
    error: '',
  };
  panelA._recv.fire({type: 'metaUpdate', values: valuesA, taskUpdate: updateA});
  await waitFor(
    () =>
      sinkCalls.length > 0 &&
      sinkCalls[sinkCalls.length - 1].taskUpdate !== null,
    'the active panel metaUpdate must reach the sink',
  );
  assert.deepStrictEqual(sinkCalls[sinkCalls.length - 1].values, valuesA);
  assert.deepStrictEqual(
    sinkCalls[sinkCalls.length - 1].taskUpdate,
    updateA,
    'the task-update state rides along with the values',
  );

  // --- refreshActiveTaskUpdate reaches the panel ------------------------
  manager.refreshActiveTaskUpdate();
  assert.deepStrictEqual(
    panelA._posted.filter(m => m.type === 'refreshTaskUpdate'),
    [{type: 'refreshTaskUpdate'}],
    'the panel gets one refreshTaskUpdate',
  );

  // --- open() again reveals the same panel; the sink keeps its values --
  let reveals = 0;
  panelA.reveal = () => reveals++;
  assert.strictEqual(manager.open(), controllerA, 'open() reuses the panel');
  assert.strictEqual(createdPanels.length, 1, 'no second panel is created');
  assert.strictEqual(reveals, 1, 'the existing panel is revealed');
  assert.deepStrictEqual(sinkCalls[sinkCalls.length - 1].values, valuesA);

  // A later report repaints the sink with the new values.
  const updateA2 = {...updateA, content: '<p>a2</p>', running: true};
  panelA._recv.fire({
    type: 'metaUpdate',
    values: {...valuesA, tokens: '1.50K'},
    taskUpdate: updateA2,
  });
  await waitFor(
    () =>
      sinkCalls[sinkCalls.length - 1].values &&
      sinkCalls[sinkCalls.length - 1].values.tokens === '1.50K',
    'a later metaUpdate must repaint the sink',
  );
  assert.deepStrictEqual(sinkCalls[sinkCalls.length - 1].taskUpdate, updateA2);

  // --- closing the panel falls back to the placeholders -----------------
  panelA.dispose();
  await waitFor(
    () =>
      sinkCalls[sinkCalls.length - 1].values === null &&
      sinkCalls[sinkCalls.length - 1].taskUpdate === null,
    'closing the panel must show the placeholders',
  );
  assert.strictEqual(manager.controller(), undefined, 'no controller left');
  manager.refreshActiveTaskUpdate();
  assert.strictEqual(
    panelA._posted.filter(m => m.type === 'refreshTaskUpdate').length,
    1,
    'a refresh after the close posts nothing',
  );
  await new Promise(r => setTimeout(r, 200));
  assert.ok(
    !daemonCommands.some(c => c.type === 'closeTab'),
    'closing the panel closes the surface only: no closeTab to the daemon',
  );

  // --- getTaskUpdate forwards to the daemon; taskUpdate relays back -----
  // A new panel after the close: the chats stayed in the registry, so
  // the webview addresses the one on screen by its registry tab id.
  manager.open();
  assert.strictEqual(createdPanels.length, 2, 'open() after a close creates a panel');
  const panelB = createdPanels[1];
  const tabB = 'tab-b';
  panelB._recv.fire({
    type: 'getTaskUpdate',
    tabId: tabB,
    knownSig: 'old-sig',
    token: '7',
    refresh: true,
    workDir: '/stale-field',
  });
  await waitFor(
    () => daemonCommands.some(c => c.type === 'getTaskUpdate'),
    'getTaskUpdate must be forwarded to the daemon',
  );
  const fwd = daemonCommands.find(c => c.type === 'getTaskUpdate');
  assert.deepStrictEqual(
    Object.keys(fwd).sort(),
    ['knownSig', 'refresh', 'tabId', 'token', 'type'],
    'exactly tabId, knownSig, token and refresh are forwarded',
  );
  assert.strictEqual(fwd.tabId, tabB);
  assert.strictEqual(fwd.knownSig, 'old-sig');
  assert.strictEqual(fwd.token, '7');
  assert.strictEqual(fwd.refresh, true);
  await daemonBroadcast({
    type: 'taskUpdate',
    tabId: tabB,
    token: '7',
    taskId: 'task-b',
    exists: true,
    sig: '9:9',
    content: '<p>report text</p>',
    error: '',
    running: false,
    cost: 0.07,
    updatedAt: 1790000100000,
  });
  await waitFor(
    () => panelB._posted.some(m => m.type === 'taskUpdate'),
    'the taskUpdate reply must be relayed into the webview',
  );
  const reply = panelB._posted.find(m => m.type === 'taskUpdate');
  assert.strictEqual(reply.content, '<p>report text</p>');
  assert.strictEqual(reply.token, '7');
  assert.strictEqual(reply.sig, '9:9');
  assert.strictEqual(reply.running, false);
  assert.strictEqual(reply.cost, 0.07);
  assert.strictEqual(reply.updatedAt, 1790000100000);
  assert.strictEqual(reply.exists, true);
  assert.strictEqual(reply.error, '');

  // --- postMetaState before the webview resolves is flushed on ready ----
  const metaView = new SorcarSidebarView(vscodeStub.Uri.file(EXT_ROOT), {
    rootTabId: 'meta-panel',
    bodyAttrs: ' class="editor-tab-mode meta-panel-mode"',
    onEvent: () => {},
  });
  const valuesB = {...valuesA, tokens: '2.00K', workdir: '/b'};
  const cachedUpdate = {
    content: '<p>cached</p>',
    running: false,
    updatedAt: 1790000200000,
    cost: 0.01,
    error: '',
  };
  metaView.postMetaState(valuesB, cachedUpdate);
  const host = makeFakePanel('kissSorcar.metaViewSecondary', 'Task Info');
  metaView.attachWebviewHost(
    {
      webview: host.webview,
      get visible() {
        return true;
      },
      show: () => {},
      onDidChangeVisibility: host.onDidChangeViewState,
      onDidDispose: host.onDidDispose,
    },
    ' class="editor-tab-mode meta-panel-mode"',
  );
  assert.strictEqual(
    host._posted.filter(m => m.type === 'metaState').length,
    0,
    'nothing was posted before the webview reported ready',
  );
  host._recv.fire({type: 'ready', tabId: 'meta-panel'});
  await waitFor(
    () => host._posted.some(m => m.type === 'metaState'),
    'ready must flush the cached metaState',
  );
  const flushed = host._posted.find(m => m.type === 'metaState');
  assert.deepStrictEqual(flushed.values, valuesB);
  assert.deepStrictEqual(flushed.taskUpdate, cachedUpdate);

  // A later postMetaState reaches the resolved webview directly.
  metaView.postMetaState(null, null);
  await waitFor(
    () =>
      host._posted.filter(m => m.type === 'metaState').length >= 2 &&
      host._posted[host._posted.length - 1].values === null,
    'a live postMetaState must reach the webview',
  );
  assert.strictEqual(host._posted[host._posted.length - 1].taskUpdate, null);

  // --- the view's metaRefresh message calls onMetaRefresh ---------------
  let refreshCalls = 0;
  metaView.onMetaRefresh = () => {
    refreshCalls += 1;
  };
  host._recv.fire({type: 'metaRefresh'});
  await waitFor(() => refreshCalls === 1, 'metaRefresh must call onMetaRefresh');
  assert.strictEqual(
    daemonCommands.filter(c => c.type === 'getTaskUpdate').length,
    1,
    'metaRefresh is a host hook, not a daemon command',
  );

  metaView.dispose();
  manager.dispose();
  console.log('metaSinkPanelManager: all tests passed');
}

runTest()
  .then(() => {
    server.close();
    process.exit(0);
  })
  .catch(err => {
    console.error(err);
    process.exit(1);
  });
