// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end tests for the host side of the history panel's
// active-task relay (out/SorcarPanelManager.js +
// out/SorcarSidebarView.js) against a REAL Unix-domain-socket daemon
// stub:
//  - setActiveTaskSink pushes the current state right away ('' ids
//    when no panel reported yet);
//  - a panel webview's `activeTask` reaches the sink while that panel
//    is active, is cached while it is not, and the cache repaints the
//    sink on panel activation and on the active panel's close;
//  - SorcarSidebarView.onActiveTask fires for a plain (sidebar) view's
//    `activeTask` message;
//  - SorcarSidebarView.postActiveTask before the webview resolves is
//    caught up on the webview's `ready` (the history view's
//    late-resolve path) and a later one reaches the webview directly.

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

const tmpHome = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-activetask-'));
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
const {SorcarSidebarView} = require(path.join(OUT_DIR, 'SorcarSidebarView.js'));

function tabIdOf(panel) {
  const m = /data-kiss-tab-id="([^"]+)"/.exec(panel.webview.html);
  assert.ok(m, 'panel html must carry data-kiss-tab-id');
  return m[1];
}

async function runTest() {
  server.listen(sockPath);
  const manager = new SorcarPanelManager(vscodeStub.Uri.file(EXT_ROOT));

  // --- setActiveTaskSink pushes the empty state right away -------------
  const sinkCalls = [];
  manager.setActiveTaskSink((chatId, taskId) => {
    sinkCalls.push({chatId, taskId});
  });
  assert.deepStrictEqual(
    sinkCalls,
    [{chatId: '', taskId: ''}],
    'no panels: the sink starts empty',
  );

  // --- the active panel's activeTask reaches the sink -------------------
  manager.openNewChat();
  const panelA = createdPanels[0];
  assert.deepStrictEqual(
    sinkCalls[sinkCalls.length - 1],
    {chatId: '', taskId: ''},
    'a fresh active panel has reported nothing yet',
  );
  panelA._recv.fire({type: 'activeTask', chatId: 'chat-a', taskId: 'task-1'});
  await waitFor(
    () => sinkCalls[sinkCalls.length - 1].chatId === 'chat-a',
    'the active panel activeTask must reach the sink',
  );
  assert.deepStrictEqual(sinkCalls[sinkCalls.length - 1], {
    chatId: 'chat-a',
    taskId: 'task-1',
  });

  // --- a second (active) panel repaints the sink; a BACKGROUND panel's
  // report is cached without repainting --------------------------------
  manager.openNewChat();
  const panelB = createdPanels[1];
  panelA.active = false;
  assert.deepStrictEqual(
    sinkCalls[sinkCalls.length - 1],
    {chatId: '', taskId: ''},
    'activating an unreported panel clears the highlight',
  );
  panelB._recv.fire({type: 'activeTask', chatId: 'chat-b', taskId: 'task-2'});
  await waitFor(
    () => sinkCalls[sinkCalls.length - 1].chatId === 'chat-b',
    'the new active panel activeTask must reach the sink',
  );
  const sinkLenAfterB = sinkCalls.length;
  panelA._recv.fire({type: 'activeTask', chatId: 'chat-a', taskId: 'task-3'});
  await new Promise(r => setTimeout(r, 200));
  assert.strictEqual(
    sinkCalls.length,
    sinkLenAfterB,
    'a background panel report must not repaint the sink',
  );

  // --- switching back to panel A pushes its cached ids ------------------
  panelB.active = false;
  panelA.active = true;
  panelA._viewState.fire({webviewPanel: panelA});
  assert.deepStrictEqual(sinkCalls[sinkCalls.length - 1], {
    chatId: 'chat-a',
    taskId: 'task-3',
  });

  // --- closing the active panel repaints from the remaining one ---------
  panelA.dispose();
  await waitFor(
    () => sinkCalls[sinkCalls.length - 1].chatId === 'chat-b',
    'closing the active panel must fall back to the remaining panel',
  );
  assert.strictEqual(sinkCalls[sinkCalls.length - 1].taskId, 'task-2');

  // --- a plain (sidebar) view fires onActiveTask --------------------------
  const sidebar = new SorcarSidebarView(vscodeStub.Uri.file(EXT_ROOT));
  const sidebarCalls = [];
  sidebar.onActiveTask = (chatId, taskId) =>
    sidebarCalls.push({chatId, taskId});
  const sidebarHost = makeFakePanel('kissSorcar.chatViewSecondary', 'Chat');
  sidebar.attachWebviewHost(
    {
      webview: sidebarHost.webview,
      get visible() {
        return true;
      },
      show: () => {},
      onDidChangeVisibility: sidebarHost.onDidChangeViewState,
      onDidDispose: sidebarHost.onDidDispose,
    },
    '',
  );
  sidebarHost._recv.fire({
    type: 'activeTask',
    chatId: 'chat-s',
    taskId: 'task-9',
  });
  await waitFor(
    () => sidebarCalls.length === 1,
    'the sidebar view must fire onActiveTask',
  );
  assert.deepStrictEqual(sidebarCalls, [{chatId: 'chat-s', taskId: 'task-9'}]);

  // --- postActiveTask before the webview resolves is flushed on ready ---
  const historyView = new SorcarSidebarView(vscodeStub.Uri.file(EXT_ROOT), {
    rootTabId: 'history-panel',
    bodyAttrs: ' class="editor-tab-mode history-panel-mode"',
    onEvent: () => {},
  });
  historyView.postActiveTask('chat-b', 'task-2');
  const host = makeFakePanel('kissSorcar.historyView', 'History');
  historyView.attachWebviewHost(
    {
      webview: host.webview,
      get visible() {
        return true;
      },
      show: () => {},
      onDidChangeVisibility: host.onDidChangeViewState,
      onDidDispose: host.onDidDispose,
    },
    ' class="editor-tab-mode history-panel-mode"',
  );
  assert.strictEqual(
    host._posted.filter(m => m.type === 'activeTask').length,
    0,
    'nothing was posted before the webview reported ready',
  );
  host._recv.fire({type: 'ready', tabId: 'history-panel'});
  await waitFor(
    () => host._posted.some(m => m.type === 'activeTask'),
    'ready must flush the cached activeTask',
  );
  const flushed = host._posted.find(m => m.type === 'activeTask');
  assert.strictEqual(flushed.chatId, 'chat-b');
  assert.strictEqual(flushed.taskId, 'task-2');

  // A later postActiveTask reaches the resolved webview directly.
  historyView.postActiveTask('', '');
  await waitFor(() => {
    const msgs = host._posted.filter(m => m.type === 'activeTask');
    return msgs.length >= 2 && msgs[msgs.length - 1].chatId === '';
  }, 'a live postActiveTask must reach the webview');

  sidebar.dispose();
  historyView.dispose();
  manager.dispose();
  console.log('activeTaskSinkPanelManager: all tests passed');
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
