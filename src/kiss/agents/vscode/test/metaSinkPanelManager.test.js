// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end tests for the host side of the Task Info relay
// (out/SorcarPanelManager.js + out/SorcarSidebarView.js) against a
// REAL Unix-domain-socket daemon stub:
//  - setMetaSink pushes the current state right away (placeholders
//    when no panel reported yet);
//  - a panel webview's `metaUpdate` reaches the sink while that panel
//    is active, is cached while it is not, and the cache repaints the
//    sink on panel activation and on the active panel's close;
//  - a panel webview's `getInfoFile` is forwarded to the daemon, and
//    the daemon's direct `infoFile` reply is relayed back into the
//    webview;
//  - SorcarSidebarView.postMetaState before the webview resolves is
//    caught up on the webview's `ready` (the meta view's late-resolve
//    path).

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
const sockPath = path.join(tmpHome, '.kiss', 'sorcar.sock');

const serverSockets = [];
const daemonCommands = [];
const server = net.createServer(sock => {
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

function tabIdOf(panel) {
  const m = /data-kiss-tab-id="([^"]+)"/.exec(panel.webview.html);
  assert.ok(m, 'panel html must carry data-kiss-tab-id');
  return m[1];
}

async function runTest() {
  server.listen(sockPath);
  const manager = new SorcarPanelManager(vscodeStub.Uri.file(EXT_ROOT));

  // --- setMetaSink pushes the placeholder state right away -------------
  const sinkCalls = [];
  manager.setMetaSink((values, progressMd) => {
    sinkCalls.push({values, progressMd});
  });
  assert.deepStrictEqual(
    sinkCalls,
    [{values: null, progressMd: ''}],
    'no panels: the sink starts on the placeholder state',
  );

  // --- the active panel's metaUpdate reaches the sink -------------------
  manager.openNewChat();
  const panelA = createdPanels[0];
  const tabA = tabIdOf(panelA);
  assert.deepStrictEqual(
    sinkCalls[sinkCalls.length - 1],
    {values: null, progressMd: ''},
    'a fresh active panel has no cached values yet',
  );
  const valuesA = {
    tokens: '1.00K',
    cost: '$0.10',
    steps: '3',
    time: '5.0s',
    timeColor: 'var(--red)',
    machine: 'hostA',
    workdir: '/a',
    maxBudget: '$10.00',
  };
  panelA._recv.fire({type: 'metaUpdate', values: valuesA, progressMd: '# a'});
  await waitFor(
    () =>
      sinkCalls.length > 0 &&
      sinkCalls[sinkCalls.length - 1].progressMd === '# a',
    'the active panel metaUpdate must reach the sink',
  );
  assert.deepStrictEqual(sinkCalls[sinkCalls.length - 1].values, valuesA);

  // --- a second (active) panel repaints the sink; a BACKGROUND panel's
  // report is cached without repainting --------------------------------
  manager.openNewChat();
  const panelB = createdPanels[1];
  panelA.active = false;
  assert.deepStrictEqual(
    sinkCalls[sinkCalls.length - 1],
    {values: null, progressMd: ''},
    'activating an unreported panel shows the placeholders',
  );
  const valuesB = {...valuesA, tokens: '2.00K', workdir: '/b'};
  panelB._recv.fire({type: 'metaUpdate', values: valuesB, progressMd: ''});
  await waitFor(
    () =>
      sinkCalls[sinkCalls.length - 1].values &&
      sinkCalls[sinkCalls.length - 1].values.tokens === '2.00K',
    'the new active panel metaUpdate must reach the sink',
  );
  const sinkLenAfterB = sinkCalls.length;
  panelA._recv.fire({
    type: 'metaUpdate',
    values: {...valuesA, tokens: '1.50K'},
    progressMd: '# a2',
  });
  await new Promise(r => setTimeout(r, 200));
  assert.strictEqual(
    sinkCalls.length,
    sinkLenAfterB,
    'a background panel report must not repaint the sink',
  );

  // --- switching back to panel A pushes its cached values ---------------
  panelB.active = false;
  panelA.active = true;
  panelA._viewState.fire({webviewPanel: panelA});
  assert.strictEqual(sinkCalls[sinkCalls.length - 1].values.tokens, '1.50K');
  assert.strictEqual(sinkCalls[sinkCalls.length - 1].progressMd, '# a2');

  // --- closing the active panel repaints from the remaining one ---------
  panelA.dispose();
  await waitFor(
    () =>
      sinkCalls[sinkCalls.length - 1].values &&
      sinkCalls[sinkCalls.length - 1].values.tokens === '2.00K',
    'closing the active panel must fall back to the remaining panel',
  );
  const closeTabCmd = () =>
    daemonCommands.find(c => c.type === 'closeTab' && c.tabId === tabA);
  await waitFor(closeTabCmd, 'the user close retires the chat tab');

  // --- getInfoFile forwards to the daemon; infoFile relays back ---------
  const tabB = tabIdOf(panelB);
  panelB._recv.fire({
    type: 'getInfoFile',
    workDir: '/b',
    tabId: tabB,
    knownSig: 'old-sig',
    token: '7',
  });
  await waitFor(
    () => daemonCommands.some(c => c.type === 'getInfoFile'),
    'getInfoFile must be forwarded to the daemon',
  );
  const fwd = daemonCommands.find(c => c.type === 'getInfoFile');
  assert.strictEqual(fwd.workDir, '/b');
  assert.strictEqual(fwd.tabId, tabB);
  assert.strictEqual(fwd.knownSig, 'old-sig');
  assert.strictEqual(fwd.token, '7');
  await daemonBroadcast({
    type: 'infoFile',
    tabId: tabB,
    token: '7',
    exists: true,
    sig: '9:9',
    content: 'progress text',
  });
  await waitFor(
    () => panelB._posted.some(m => m.type === 'infoFile'),
    'the infoFile reply must be relayed into the webview',
  );
  const reply = panelB._posted.find(m => m.type === 'infoFile');
  assert.strictEqual(reply.content, 'progress text');
  assert.strictEqual(reply.token, '7');

  // --- postMetaState before the webview resolves is flushed on ready ----
  const metaView = new SorcarSidebarView(vscodeStub.Uri.file(EXT_ROOT), {
    rootTabId: 'meta-panel',
    bodyAttrs: ' class="editor-tab-mode meta-panel-mode"',
    onEvent: () => {},
  });
  metaView.postMetaState(valuesB, '# cached');
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
  assert.strictEqual(flushed.progressMd, '# cached');

  // A later postMetaState reaches the resolved webview directly.
  metaView.postMetaState(null, '');
  await waitFor(
    () =>
      host._posted.filter(m => m.type === 'metaState').length >= 2 &&
      host._posted[host._posted.length - 1].values === null,
    'a live postMetaState must reach the webview',
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
