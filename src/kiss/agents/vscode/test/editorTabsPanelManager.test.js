// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end tests for editor-tabs mode's host side
// (out/SorcarPanelManager.js + out/SorcarSidebarView.js) against a
// REAL Unix-domain-socket daemon stub:
//  - openNewChat creates a WebviewPanel whose chat HTML is pinned to a
//    fresh root tab (editor-tab-mode body class + data-kiss-tab-id);
//  - the webview's `panelTitle` message retitles the editor tab;
//  - `openChatPanel` from a webview opens a second panel carrying the
//    resume data attributes, and — once the daemon's `tabs_state`
//    binds that panel's tab to the chat — a repeat open REVEALS the
//    existing panel instead of stacking a third;
//  - a USER panel close retires the chat tab (`closeTab` reaches the
//    daemon), while closeAll (mode switch off) closes panels WITHOUT
//    retiring their chats;
//  - the serializer re-adopts a revived panel from its persisted
//    editorRootTabId, and disposes revived panels when the mode is
//    off.

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

let editorTabsMode = true;
const createdPanels = [];
let registeredSerializer = null;

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
    reveals: 0,
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
    reveal: () => {
      panel.reveals += 1;
    },
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
      get: key => (key === 'editorTabsMode' ? editorTabsMode : ''),
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
    createWebviewPanel: (viewType, title, _column, _options) => {
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

const tmpHome = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-edtabs-'));
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

const {SorcarPanelManager, CHAT_PANEL_VIEW_TYPE} = require(
  path.join(OUT_DIR, 'SorcarPanelManager.js'),
);

function tabIdOf(panel) {
  const m = /data-kiss-tab-id="([^"]+)"/.exec(panel.webview.html);
  assert.ok(m, 'panel html must carry data-kiss-tab-id');
  return m[1];
}

async function runTest() {
  server.listen(sockPath);
  const manager = new SorcarPanelManager(vscodeStub.Uri.file(EXT_ROOT));
  vscodeStub._ignored = manager.registerSerializer();
  assert.strictEqual(registeredSerializer.viewType, CHAT_PANEL_VIEW_TYPE);

  // --- openNewChat pins a fresh root tab -------------------------------
  manager.openNewChat();
  assert.strictEqual(createdPanels.length, 1);
  const panelA = createdPanels[0];
  assert.strictEqual(panelA.viewType, CHAT_PANEL_VIEW_TYPE);
  assert.ok(
    panelA.webview.html.includes('class="editor-tab-mode"'),
    'chat html must be in editor-tab mode',
  );
  const tabA = tabIdOf(panelA);
  assert.ok(panelA.iconPath, 'panel gets the KISS icon');
  await waitFor(
    () => serverSockets.length >= 1,
    'panel controller must connect to the daemon',
  );

  // --- panelTitle retitles the editor tab ------------------------------
  panelA._recv.fire({type: 'panelTitle', title: 'fix the bug', tabId: tabA});
  await waitFor(
    () => panelA.title === 'fix the bug',
    'panelTitle must retitle the panel',
  );

  // --- openChatPanel opens a resuming second panel ---------------------
  panelA._recv.fire({
    type: 'openChatPanel',
    chatId: 'chat-B',
    taskId: 7,
    title: 'resumed chat',
  });
  await waitFor(() => createdPanels.length === 2, 'second panel must open');
  const panelB = createdPanels[1];
  assert.strictEqual(panelB.title, 'resumed chat');
  assert.ok(panelB.webview.html.includes('data-kiss-resume-chat-id="chat-B"'));
  assert.ok(panelB.webview.html.includes('data-kiss-resume-task-id="7"'));
  const tabB = tabIdOf(panelB);
  assert.notStrictEqual(tabA, tabB);
  await waitFor(
    () => serverSockets.length >= 2,
    'second panel controller must connect',
  );

  // --- one chat, one panel: repeat open reveals ------------------------
  await daemonBroadcast({
    type: 'tabs_state',
    tabs: [
      {
        tabId: tabB,
        chatId: 'chat-B',
        title: 'resumed chat',
        workDir: '',
        scopeWorkDir: '',
      },
    ],
  });
  panelA._recv.fire({type: 'openChatPanel', chatId: 'chat-B'});
  await new Promise(r => setTimeout(r, 100));
  assert.strictEqual(
    createdPanels.length,
    2,
    'an already-open chat must not get a second panel',
  );
  await waitFor(
    () => panelB.reveals >= 1,
    'the existing chat panel must be revealed instead',
  );

  // --- a user close retires the chat tab -------------------------------
  panelB.dispose();
  await waitFor(
    () => daemonCommands.some(c => c.type === 'closeTab' && c.tabId === tabB),
    'user panel close must send closeTab for its chat tab',
  );

  // --- closeAll (mode off) keeps the chats registered ------------------
  manager.closeAll();
  assert.ok(panelA.disposed, 'closeAll must dispose the panels');
  await new Promise(r => setTimeout(r, 150));
  assert.ok(
    !daemonCommands.some(c => c.type === 'closeTab' && c.tabId === tabA),
    'closeAll must NOT retire the chat from the registry',
  );

  // --- serializer: revive adopts the persisted root tab ----------------
  const revived = makeFakePanel(CHAT_PANEL_VIEW_TYPE, 'KISS Sorcar');
  await registeredSerializer.serializer.deserializeWebviewPanel(revived, {
    editorRootTabId: 'restored-tab-9',
  });
  assert.ok(
    revived.webview.html.includes('data-kiss-tab-id="restored-tab-9"'),
    'revived panel must re-adopt its persisted root tab',
  );
  assert.ok(!revived.disposed);

  // --- serializer: mode off disposes revived panels --------------------
  editorTabsMode = false;
  const staleRevive = makeFakePanel(CHAT_PANEL_VIEW_TYPE, 'KISS Sorcar');
  await registeredSerializer.serializer.deserializeWebviewPanel(staleRevive, {
    editorRootTabId: 'restored-tab-10',
  });
  assert.ok(
    staleRevive.disposed,
    'a panel revived with the mode off must be disposed',
  );

  // --- closePanel(retire) from the webview retires the chat ------------
  editorTabsMode = true;
  manager.openNewChat();
  const panelC = createdPanels[createdPanels.length - 1];
  const tabC = tabIdOf(panelC);
  await waitFor(
    () => serverSockets.filter(s => !s.destroyed).length >= 1,
    'third panel controller must connect',
  );
  panelC._recv.fire({type: 'closePanel', retire: true});
  await waitFor(() => panelC.disposed, 'closePanel must dispose the panel');
  await waitFor(
    () => daemonCommands.some(c => c.type === 'closeTab' && c.tabId === tabC),
    'closePanel(retire) must retire the chat tab',
  );

  // --- enterMode marks registry-born panels ----------------------------
  const before = createdPanels.length;
  manager.enterMode(
    [
      {
        tabId: 'reg-tab-1',
        chatId: 'reg-chat-1',
        title: 'registry chat',
        workDir: '/some/ws',
        scopeWorkDir: '',
      },
    ],
    '/some/ws',
  );
  assert.strictEqual(createdPanels.length, before + 1);
  const panelD = createdPanels[createdPanels.length - 1];
  assert.ok(
    panelD.webview.html.includes('data-kiss-in-registry="1"'),
    'registry-born panels boot as already registered',
  );
  assert.ok(panelD.webview.html.includes('data-kiss-tab-id="reg-tab-1"'));

  // --- terminal dispose leaves panels standing for revival -------------
  const closeTabsBefore = daemonCommands.filter(
    c => c.type === 'closeTab',
  ).length;
  manager.dispose();
  assert.ok(
    !panelD.disposed,
    'teardown must NOT close editor tabs (the workbench persists them ' +
      'for the serializer)',
  );
  await new Promise(r => setTimeout(r, 120));
  assert.strictEqual(
    daemonCommands.filter(c => c.type === 'closeTab').length,
    closeTabsBefore,
    'teardown must not retire any chat',
  );

  console.log('editorTabsPanelManager: all tests passed');
}

runTest()
  .then(() => {
    server.close();
    fs.rmSync(tmpHome, {recursive: true, force: true});
    process.exit(0);
  })
  .catch(err => {
    console.error(err);
    process.exit(1);
  });
