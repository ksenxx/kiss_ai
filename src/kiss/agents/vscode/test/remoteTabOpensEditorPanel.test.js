// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end tests for editor-tabs mode adopting registry tabs other
// clients create — the remote web app running a task must open an
// editor tab in this window (out/SorcarSidebarView.js
// onRegistryTabsState + out/SorcarPanelManager.js adoptRegistryTabs),
// mirroring how sidebar mode's webview adopts the same tab into its
// internal strip from the same `tabs_state` snapshot. Runs against a
// REAL Unix-domain-socket daemon stub:
//  - the window's long-lived controller requests a baseline snapshot
//    (`getTabsState`) on daemon connect; per-panel controllers do not;
//  - a reloaded window (serialized chat placeholder present) filters
//    the FIRST snapshot against the previous session's persisted
//    panel tab ids: the placeholders' own tabs are skipped, a tab a
//    remote client registered in the meantime still opens;
//  - a tab that APPEARS in a later snapshot opens a panel pinned to
//    the same tab id (data-kiss-in-registry, preserveFocus, no
//    keyboard-focus steal);
//  - a re-broadcast of the same snapshot opens no duplicate;
//  - a known tab that later gains its FIRST chat binding (a task ran
//    in an idle remote tab) opens a panel;
//  - tabs scoped to another workspace are skipped;
//  - a tab bound to a chat some open panel already shows is skipped
//    while that panel's tab is still registered (resume race) — but
//    ADOPTED when the registry displaced the panel's tab (one tab per
//    chat: the newest bind wins and the old panel closes itself);
//  - the displacement decision holds even when the panel's OWN daemon
//    socket never delivered its snapshots (the manager syncs
//    registryBound from the controller's snapshot stream itself);
//  - the manager reports the open panels' tab ids on every change,
//    except during terminal teardown (the serializer's tabs survive);
//  - with editor-tabs mode off the extension wiring does nothing;
//  - WITHOUT a persisted panel-id record (pre-upgrade session) a
//    first snapshot is fully suppressed while a chat tab exists;
//  - an EMPTY window (no chat editor tabs at all) adopts even the
//    first snapshot: the daemon only broadcasts on mutations, webview
//    `ready`s and getTabsState requests, so a remote task's own
//    mutation may be the first snapshot this window ever sees.

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
const WORKSPACE_DIR = '/ws/proj';
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
    workspaceFolders: [{uri: makeUri(WORKSPACE_DIR)}],
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
    createWebviewPanel: (viewType, title, showOptions, _options) => {
      const panel = makeFakePanel(viewType, title);
      panel._showOptions = showOptions;
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

const tmpHome = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-remotetab-'));
process.env.HOME = tmpHome;
process.env.USERPROFILE = tmpHome;
fs.mkdirSync(path.join(tmpHome, '.kiss'), {recursive: true});
const sockPath = path.join(tmpHome, '.kiss', 'sorcar.sock');

const serverSockets = [];
// Every command line any client sent to the daemon stub.
const receivedCommands = [];
const server = net.createServer(sock => {
  serverSockets.push(sock);
  let buf = '';
  sock.on('data', chunk => {
    buf += chunk.toString('utf8');
    let nl;
    while ((nl = buf.indexOf('\n')) >= 0) {
      const line = buf.slice(0, nl);
      buf = buf.slice(nl + 1);
      try {
        receivedCommands.push(JSON.parse(line));
      } catch {
        /* partial or non-JSON noise */
      }
    }
  });
  sock.on('error', () => {});
});

function getTabsStateRequests() {
  return receivedCommands.filter(c => c.type === 'getTabsState').length;
}

/** Broadcast *msg* to *socks* (default: every connected client). */
function daemonBroadcast(msg, socks) {
  const line = JSON.stringify(msg) + '\n';
  for (const sock of socks || serverSockets) {
    if (!sock.destroyed) sock.write(line);
  }
  return new Promise(r => setTimeout(r, 120));
}

async function waitFor(predicate, message) {
  for (let i = 0; i < 150; i++) {
    if (predicate()) return;
    await new Promise(r => setTimeout(r, 20));
  }
  throw new Error(message || 'waitFor timed out');
}

const {SorcarPanelManager} = require(path.join(OUT_DIR, 'SorcarPanelManager.js'));
const {SorcarSidebarView} = require(path.join(OUT_DIR, 'SorcarSidebarView.js'));

function entry(tabId, chatId, title, workDir, scopeWorkDir) {
  return {
    tabId,
    chatId: chatId || '',
    title: title || '',
    workDir: workDir || '',
    scopeWorkDir: scopeWorkDir || '',
  };
}

function tabIdOf(panel) {
  const m = /data-kiss-tab-id="([^"]+)"/.exec(panel.webview.html);
  assert.ok(m, 'panel html must carry data-kiss-tab-id');
  return m[1];
}

/** How many editor tabs host a chat (live panels + placeholders). */
function chatEditorTabCount() {
  let count = 0;
  for (const group of vscodeStub.window.tabGroups.all) {
    for (const tab of group.tabs) {
      const viewType = tab.input && tab.input.viewType;
      if (
        typeof viewType === 'string' &&
        viewType.includes('kissSorcar.chatTab')
      ) {
        count += 1;
      }
    }
  }
  return count;
}

/** The extension.ts wiring under test (see activate()). */
function wire(sidebar, manager, priorPanelTabIds) {
  return sidebar.onRegistryTabsState(delta => {
    if (!editorTabsMode) return;
    let toAdopt = delta.added;
    if (
      delta.firstSnapshot &&
      (manager.panelCount > 0 || chatEditorTabCount() > 0)
    ) {
      const tabCount = chatEditorTabCount();
      if (tabCount >= 0 && tabCount === manager.panelCount) {
        // live panels only — adopt wholesale, the manager dedupes
      } else if (priorPanelTabIds) {
        const prior = new Set(priorPanelTabIds);
        toAdopt = delta.added.filter(e => !prior.has(e.tabId));
      } else {
        toAdopt = [];
      }
    }
    manager.adoptRegistryTabs(toAdopt, WORKSPACE_DIR, delta.listed);
  });
}

const PLACEHOLDER_TAB_GROUPS = {
  all: [{tabs: [{input: {viewType: 'mainThreadWebview-kissSorcar.chatTab'}}]}],
};

async function runTest() {
  server.listen(sockPath);

  // ===== Phase A: a RELOADED window (serialized chat placeholders,
  // previous session's panel tab ids persisted as T0 + T-idle) =======
  vscodeStub.window.tabGroups = PLACEHOLDER_TAB_GROUPS;
  // The extension's workspaceState-backed record (see extension.ts
  // recordPanelTab): per-tab add/remove deltas on the stored array.
  let recordedIds = ['T0', 'T-idle'];
  const recordPanelTab = (tabId, open) => {
    recordedIds = open
      ? recordedIds.includes(tabId)
        ? recordedIds
        : [...recordedIds, tabId]
      : recordedIds.filter(id => id !== tabId);
  };
  const sidebarView = new SorcarSidebarView(vscodeStub.Uri.file(EXT_ROOT));
  const manager = new SorcarPanelManager(
    vscodeStub.Uri.file(EXT_ROOT),
    tabId => sidebarView.closeChatTab(tabId),
    recordPanelTab,
  );
  const sub = wire(sidebarView, manager, ['T0', 'T-idle']);
  sidebarView.syncWorkDir();
  await waitFor(
    () => serverSockets.length >= 1,
    'sidebar controller must connect to the daemon',
  );
  const controllerSocket = serverSockets[0];

  // --- the long-lived controller requests a baseline on connect -------
  await waitFor(
    () => getTabsStateRequests() === 1,
    'the controller must request the registry snapshot on connect',
  );

  // --- first snapshot: the placeholders' tabs are filtered out, a
  // tab a remote client registered before this window's baseline
  // request (the exact revival race) still opens ----------------------
  await daemonBroadcast({
    type: 'tabs_state',
    tabs: [
      entry('T0', 'chat-0', 'old chat', WORKSPACE_DIR),
      entry('T-idle', '', 'idle remote tab', WORKSPACE_DIR),
      entry('T-race', 'chat-race', 'raced remote task', WORKSPACE_DIR),
    ],
    tabId: '',
  });
  await waitFor(
    () => createdPanels.length === 1,
    'a remote tab unknown to the previous session must open even on ' +
      'the first snapshot',
  );
  assert.strictEqual(
    tabIdOf(createdPanels[0]),
    'T-race',
    'the placeholder tabs (T0, T-idle) must not be adopted',
  );
  assert.deepStrictEqual(
    recordedIds,
    ['T0', 'T-idle', 'T-race'],
    'an adopted panel must be ADDED to the record without touching ' +
      "the previous session's ids (their placeholders still exist)",
  );

  // --- a NEW tab in a later snapshot opens an editor tab --------------
  await daemonBroadcast({
    type: 'tabs_state',
    tabs: [
      entry('T0', 'chat-0', 'old chat', WORKSPACE_DIR),
      entry('T-idle', '', 'idle remote tab', WORKSPACE_DIR),
      entry('T-race', 'chat-race', 'raced remote task', WORKSPACE_DIR),
      entry('T1', 'chat-1', 'remote task', path.join(WORKSPACE_DIR, 'sub')),
    ],
    tabId: '',
  });
  await waitFor(
    () => createdPanels.length === 2,
    'a remote-created tab must open an editor tab',
  );
  const panel1 = createdPanels[1];
  assert.strictEqual(tabIdOf(panel1), 'T1', 'panel pins the registry tab id');
  assert.strictEqual(panel1.title, 'remote task');
  assert.ok(
    panel1.webview.html.includes('data-kiss-in-registry="1"'),
    'an adopted registry tab must be marked in-registry',
  );
  assert.deepStrictEqual(
    panel1._showOptions,
    {viewColumn: vscodeStub.ViewColumn.Active, preserveFocus: true},
    'an adopted tab must open without stealing keyboard focus',
  );
  await waitFor(
    () => serverSockets.length >= 3,
    'the adopted panel controllers must connect',
  );
  assert.strictEqual(
    getTabsStateRequests(),
    1,
    'per-panel controllers must NOT request registry snapshots',
  );

  // --- a re-broadcast of the same snapshot opens no duplicate ---------
  await daemonBroadcast({
    type: 'tabs_state',
    tabs: [
      entry('T0', 'chat-0', 'old chat', WORKSPACE_DIR),
      entry('T-idle', '', 'idle remote tab', WORKSPACE_DIR),
      entry('T-race', 'chat-race', 'raced remote task', WORKSPACE_DIR),
      entry('T1', 'chat-1', 'remote task', path.join(WORKSPACE_DIR, 'sub')),
    ],
    tabId: '',
  });
  assert.strictEqual(
    createdPanels.length,
    2,
    'an unchanged snapshot must not open duplicate panels',
  );

  // --- a known tab gaining its FIRST chat binding opens a panel -------
  // (a task run in an idle remote tab binds the chat id)
  await daemonBroadcast({
    type: 'tabs_state',
    tabs: [
      entry('T0', 'chat-0', 'old chat', WORKSPACE_DIR),
      entry('T-idle', 'chat-idle', 'first task here', WORKSPACE_DIR),
      entry('T-race', 'chat-race', 'raced remote task', WORKSPACE_DIR),
      entry('T1', 'chat-1', 'remote task', path.join(WORKSPACE_DIR, 'sub')),
    ],
    tabId: '',
  });
  await waitFor(
    () => createdPanels.length === 3,
    'a first task run in a known idle tab must open an editor tab',
  );
  assert.strictEqual(tabIdOf(createdPanels[2]), 'T-idle');

  // --- tabs scoped to another workspace are skipped -------------------
  await daemonBroadcast({
    type: 'tabs_state',
    tabs: [
      entry('T0', 'chat-0', 'old chat', WORKSPACE_DIR),
      entry('T-idle', 'chat-idle', 'first task here', WORKSPACE_DIR),
      entry('T-race', 'chat-race', 'raced remote task', WORKSPACE_DIR),
      entry('T1', 'chat-1', 'remote task', path.join(WORKSPACE_DIR, 'sub')),
      entry('T-other', 'chat-2', 'other repo', '/elsewhere/repo'),
      entry('T-pinned', 'chat-3', 'pinned away', WORKSPACE_DIR, '/elsewhere'),
    ],
    tabId: '',
  });
  await new Promise(r => setTimeout(r, 150));
  assert.strictEqual(
    createdPanels.length,
    3,
    'tabs of another workspace must not open panels here',
  );

  // --- resume race: a still-registering panel's chat is not doubled ---
  // This window resumed chat-R from history (the panel knows its chat
  // id from the open, but its openTab has not been confirmed by any
  // snapshot), then another client resumed the SAME chat into its own
  // new registry tab: no second panel may open for it.
  manager.openChat({chatId: 'chat-R', title: 'resumed here'});
  assert.strictEqual(createdPanels.length, 4);
  await daemonBroadcast({
    type: 'tabs_state',
    tabs: [
      entry('T0', 'chat-0', 'old chat', WORKSPACE_DIR),
      entry('T-idle', 'chat-idle', 'first task here', WORKSPACE_DIR),
      entry('T-race', 'chat-race', 'raced remote task', WORKSPACE_DIR),
      entry('T1', 'chat-1', 'remote task', path.join(WORKSPACE_DIR, 'sub')),
      entry('T-r2', 'chat-R', 'resumed here', WORKSPACE_DIR),
    ],
    tabId: '',
  });
  await new Promise(r => setTimeout(r, 150));
  assert.strictEqual(
    createdPanels.length,
    4,
    'a chat still registering in an open panel must not get a second panel',
  );

  // --- unpinned tabs (no workDir/scope) are adopted (enterMode parity)
  await daemonBroadcast({
    type: 'tabs_state',
    tabs: [
      entry('T0', 'chat-0', 'old chat', WORKSPACE_DIR),
      entry('T-idle', 'chat-idle', 'first task here', WORKSPACE_DIR),
      entry('T-race', 'chat-race', 'raced remote task', WORKSPACE_DIR),
      entry('T1', 'chat-1', 'remote task', path.join(WORKSPACE_DIR, 'sub')),
      entry('T-r2', 'chat-R', 'resumed here', WORKSPACE_DIR),
      entry('T-unpinned', 'chat-4', 'everywhere tab'),
    ],
    tabId: '',
  });
  await waitFor(
    () => createdPanels.length === 5,
    'an unpinned tab must be adopted like enterMode does',
  );
  assert.strictEqual(tabIdOf(createdPanels[4]), 'T-unpinned');

  // --- displacement: the replacement tab of a displaced chat IS
  // adopted. panel1's root tab T1 is registry-confirmed (listed by
  // the snapshots above); another client re-binding chat-1 to a NEW
  // tab makes the daemon drop T1 (one tab per chat, newest bind wins)
  // — panel1 will close itself, so the replacement must get a panel
  // or the chat would vanish from this window.
  await daemonBroadcast({
    type: 'tabs_state',
    tabs: [
      entry('T0', 'chat-0', 'old chat', WORKSPACE_DIR),
      entry('T-idle', 'chat-idle', 'first task here', WORKSPACE_DIR),
      entry('T-race', 'chat-race', 'raced remote task', WORKSPACE_DIR),
      entry('T1b', 'chat-1', 'remote task moved', WORKSPACE_DIR),
      entry('T-r2', 'chat-R', 'resumed here', WORKSPACE_DIR),
      entry('T-unpinned', 'chat-4', 'everywhere tab'),
    ],
    tabId: '',
  });
  await waitFor(
    () => createdPanels.length === 6,
    'the displaced chat-1 replacement tab must be adopted',
  );
  assert.strictEqual(tabIdOf(createdPanels[5]), 'T1b');

  // --- displacement with a LAGGING panel socket: the manager must
  // confirm a panel's registration from the CONTROLLER's snapshot
  // stream, not from the panel's own daemon connection. The panel's
  // socket here never receives any snapshot (broadcasts go to the
  // controller socket only), yet chat-S's replacement tab must still
  // be adopted after the controller saw the panel's tab listed once.
  manager.openChat({chatId: 'chat-S', title: 'slow-socket resume'});
  await waitFor(
    () => createdPanels.length === 7,
    'the chat-S resume opens its own panel',
  );
  const slowPanelTabId = tabIdOf(createdPanels[6]);
  const baseTabs = [
    entry('T0', 'chat-0', 'old chat', WORKSPACE_DIR),
    entry('T-idle', 'chat-idle', 'first task here', WORKSPACE_DIR),
    entry('T-race', 'chat-race', 'raced remote task', WORKSPACE_DIR),
    entry('T1b', 'chat-1', 'remote task moved', WORKSPACE_DIR),
    entry('T-r2', 'chat-R', 'resumed here', WORKSPACE_DIR),
    entry('T-unpinned', 'chat-4', 'everywhere tab'),
  ];
  await daemonBroadcast(
    {
      type: 'tabs_state',
      tabs: [
        ...baseTabs,
        entry(slowPanelTabId, 'chat-S', 'slow-socket resume', WORKSPACE_DIR),
      ],
      tabId: '',
    },
    [controllerSocket],
  );
  assert.strictEqual(createdPanels.length, 7, 'own tab: nothing to adopt');
  await daemonBroadcast(
    {
      type: 'tabs_state',
      tabs: [
        ...baseTabs,
        entry('T-s2', 'chat-S', 'slow-socket resume moved', WORKSPACE_DIR),
      ],
      tabId: '',
    },
    [controllerSocket],
  );
  await waitFor(
    () => createdPanels.length === 8,
    'the displaced chat-S replacement tab must be adopted even though ' +
      "the old panel's own socket saw no snapshot",
  );
  assert.strictEqual(tabIdOf(createdPanels[7]), 'T-s2');

  // --- persisted tab ids: user closes update the record ---------------
  assert.deepStrictEqual(
    [...recordedIds].sort(),
    [
      'T0', // previous session's ids stay until their panels close
      'T-idle',
      'T-race',
      'T-s2',
      'T-unpinned',
      'T1',
      'T1b',
      tabIdOf(createdPanels[3]), // the chat-R resume panel's random id
      slowPanelTabId,
    ].sort(),
    'the record lists every open panel tab id',
  );
  createdPanels[7].dispose(); // user closes the T-s2 panel
  assert.ok(
    !recordedIds.includes('T-s2'),
    'a user close must drop the tab id from the record',
  );

  // --- editor-tabs mode off: the wiring does nothing ------------------
  editorTabsMode = false;
  await daemonBroadcast({
    type: 'tabs_state',
    tabs: [
      entry('T0', 'chat-0', 'old chat', WORKSPACE_DIR),
      entry('T-sidebar', 'chat-5', 'sidebar adopts me', WORKSPACE_DIR),
    ],
    tabId: '',
  });
  await new Promise(r => setTimeout(r, 150));
  assert.strictEqual(
    createdPanels.length,
    8,
    'with editor-tabs mode off the host must not open panels',
  );
  editorTabsMode = true;

  // --- terminal teardown must NOT rewrite the record: the editor tabs
  // survive the reload for the serializer, so their ids must too.
  const recordedAtShutdown = [...recordedIds];
  manager.markShutdown();
  createdPanels[6].dispose(); // the workbench closing tabs on reload
  assert.deepStrictEqual(
    recordedIds,
    recordedAtShutdown,
    'disposals during shutdown must not touch the record',
  );

  sub.dispose();
  manager.dispose();
  sidebarView.dispose();
  await new Promise(r => setTimeout(r, 100));
  assert.deepStrictEqual(
    recordedIds,
    recordedAtShutdown,
    'manager.dispose() must not touch the record',
  );

  // ===== Phase B: NO persisted record (pre-upgrade session) — a
  // first snapshot is fully suppressed while a chat tab exists ========
  vscodeStub.window.tabGroups = PLACEHOLDER_TAB_GROUPS;
  const panelsBeforeB = createdPanels.length;
  const sidebarB = new SorcarSidebarView(vscodeStub.Uri.file(EXT_ROOT));
  const managerB = new SorcarPanelManager(vscodeStub.Uri.file(EXT_ROOT));
  const subB = wire(sidebarB, managerB, undefined);
  sidebarB.syncWorkDir();
  await waitFor(
    () => getTabsStateRequests() === 2,
    'the second controller must request its baseline snapshot',
  );
  await daemonBroadcast({
    type: 'tabs_state',
    tabs: [entry('T-b1', 'chat-b1', 'maybe a placeholder', WORKSPACE_DIR)],
    tabId: '',
  });
  await new Promise(r => setTimeout(r, 150));
  assert.strictEqual(
    createdPanels.length,
    panelsBeforeB,
    'no persisted record: the first snapshot must not be adopted while ' +
      'a serialized chat placeholder may duplicate its tabs',
  );
  await daemonBroadcast({
    type: 'tabs_state',
    tabs: [
      entry('T-b1', 'chat-b1', 'maybe a placeholder', WORKSPACE_DIR),
      entry('T-b2', 'chat-b2', 'later remote task', WORKSPACE_DIR),
    ],
    tabId: '',
  });
  await waitFor(
    () => createdPanels.length === panelsBeforeB + 1,
    'a later snapshot must be adopted normally',
  );
  assert.strictEqual(tabIdOf(createdPanels[panelsBeforeB]), 'T-b2');
  subB.dispose();
  managerB.dispose();
  sidebarB.dispose();
  await new Promise(r => setTimeout(r, 100));

  // ===== Phase C: an EMPTY window adopts even the FIRST snapshot ======
  // The daemon broadcasts no snapshot until a mutation, a webview
  // `ready` or a getTabsState request, so a remote task's own
  // registration can be the first snapshot this window sees; with no
  // chat editor tab anywhere there is nothing to duplicate and the
  // tab must open.
  vscodeStub.window.tabGroups = {all: []};
  const panelsBeforeC = createdPanels.length;
  const socketsBeforeC = serverSockets.length;
  const sidebarC = new SorcarSidebarView(vscodeStub.Uri.file(EXT_ROOT));
  const managerC = new SorcarPanelManager(vscodeStub.Uri.file(EXT_ROOT));
  const subC = wire(sidebarC, managerC, undefined);
  sidebarC.syncWorkDir();
  await waitFor(
    () => serverSockets.length > socketsBeforeC,
    'the empty window controller must connect',
  );
  await daemonBroadcast({
    type: 'tabs_state',
    tabs: [entry('T-fresh', 'chat-9', 'remote task in empty window', WORKSPACE_DIR)],
    tabId: '',
  });
  await waitFor(
    () => createdPanels.length === panelsBeforeC + 1,
    'an empty window must adopt the first snapshot it ever sees',
  );
  assert.strictEqual(tabIdOf(createdPanels[panelsBeforeC]), 'T-fresh');

  subC.dispose();
  managerC.dispose();
  sidebarC.dispose();
  await new Promise(r => setTimeout(r, 50));

  // ===== Phase D: daemon RECONNECT — the controller re-requests the
  // snapshot, so a tab created during the outage is adopted ===========
  vscodeStub.window.tabGroups = {all: []};
  const panelsBeforeD = createdPanels.length;
  const socketsBeforeD = serverSockets.length;
  const sidebarD = new SorcarSidebarView(vscodeStub.Uri.file(EXT_ROOT));
  const managerD = new SorcarPanelManager(vscodeStub.Uri.file(EXT_ROOT));
  const subD = wire(sidebarD, managerD, undefined);
  sidebarD.syncWorkDir();
  await waitFor(
    () => serverSockets.length > socketsBeforeD,
    'the phase D controller must connect',
  );
  const reqBeforeOutage = getTabsStateRequests();
  await daemonBroadcast({type: 'tabs_state', tabs: [], tabId: ''});
  // The daemon "restarts": the controller's socket drops, the client
  // reconnects (base backoff 500ms) and must request a fresh snapshot
  // — the only way to learn of tabs registered while it was away.
  serverSockets[socketsBeforeD].destroy();
  await waitFor(
    () => getTabsStateRequests() > reqBeforeOutage,
    'the controller must re-request the snapshot after a reconnect',
  );
  await daemonBroadcast({
    type: 'tabs_state',
    tabs: [entry('T-out', 'chat-out', 'created during outage', WORKSPACE_DIR)],
    tabId: '',
  });
  await waitFor(
    () => createdPanels.length === panelsBeforeD + 1,
    'a tab created during the outage must open after the reconnect',
  );
  assert.strictEqual(tabIdOf(createdPanels[panelsBeforeD]), 'T-out');
  subD.dispose();
  managerD.dispose();
  sidebarD.dispose();
  await new Promise(r => setTimeout(r, 100));

  // ===== Phase F: chat editor tabs exist but ALL of them are LIVE
  // panels (no serialized placeholder pending revival) — the first
  // snapshot is adopted wholesale even without a persisted record:
  // the manager dedupes against the open panels, so a fresh chat
  // opened while the daemon was down cannot suppress a remote tab ===
  const panelsBeforeF = createdPanels.length;
  const socketsBeforeF = serverSockets.length;
  const sidebarF = new SorcarSidebarView(vscodeStub.Uri.file(EXT_ROOT));
  const managerF = new SorcarPanelManager(vscodeStub.Uri.file(EXT_ROOT));
  const subF = wire(sidebarF, managerF, undefined);
  sidebarF.syncWorkDir();
  await waitFor(
    () => serverSockets.length > socketsBeforeF,
    'the phase F controller must connect',
  );
  managerF.openNewChat(); // opened before the daemon answered
  assert.strictEqual(createdPanels.length, panelsBeforeF + 1);
  // The live panel IS an editor tab; mirror it in the tabGroups stub.
  vscodeStub.window.tabGroups = PLACEHOLDER_TAB_GROUPS;
  await daemonBroadcast({
    type: 'tabs_state',
    tabs: [
      entry(
        tabIdOf(createdPanels[panelsBeforeF]),
        '',
        'offline chat',
        WORKSPACE_DIR,
      ),
      entry('T-r9', 'chat-r9', 'raced remote task 2', WORKSPACE_DIR),
    ],
    tabId: '',
  });
  await waitFor(
    () => createdPanels.length === panelsBeforeF + 2,
    'live-panels-only window: the first snapshot must be adopted',
  );
  assert.strictEqual(tabIdOf(createdPanels[panelsBeforeF + 1]), 'T-r9');
  subF.dispose();
  managerF.dispose();
  sidebarF.dispose();
  await new Promise(r => setTimeout(r, 100));
  vscodeStub.window.tabGroups = {all: []};

  // ===== Phase E: a panel whose registry registration is DROPPED
  // must release its chat claim so the chat's real tab can open ========
  vscodeStub.window.tabGroups = {all: []};
  const panelsBeforeE = createdPanels.length;
  const socketsBeforeE = serverSockets.length;
  const sidebarE = new SorcarSidebarView(vscodeStub.Uri.file(EXT_ROOT));
  const managerE = new SorcarPanelManager(vscodeStub.Uri.file(EXT_ROOT));
  const subE = wire(sidebarE, managerE, undefined);
  sidebarE.syncWorkDir();
  await waitFor(
    () => serverSockets.length > socketsBeforeE,
    'the phase E controller must connect',
  );
  const controllerE = serverSockets[socketsBeforeE];
  await daemonBroadcast({type: 'tabs_state', tabs: [], tabId: ''}, [
    controllerE,
  ]);
  // The daemon stops accepting connections (the controller's OWN
  // socket survives — only NEW connections fail), so every panel
  // opened below queues its registration commands.
  await new Promise(r => {
    server.close(() => {});
    setTimeout(r, 50);
  });

  /** Queue a resumeSession for *panel*, then overflow it out of the
   *  bounded queue (MAX_PENDING_SENDS=256) with harmless commands —
   *  the fast deterministic way to make the client DROP it. */
  const dropRegistration = (panel, chatId) => {
    panel._recv.fire({
      type: 'resumeSession',
      chatId,
      tabId: tabIdOf(panel),
    });
    for (let i = 0; i < 300; i += 1) {
      panel._recv.fire({type: 'getInputHistory'});
    }
  };

  // --- E1: the registration drops BEFORE the chat's real tab is seen.
  managerE.openChat({chatId: 'chat-X', title: 'never registers'});
  assert.strictEqual(createdPanels.length, panelsBeforeE + 1);
  dropRegistration(createdPanels[panelsBeforeE], 'chat-X');
  await new Promise(r => setTimeout(r, 100));
  await daemonBroadcast(
    {
      type: 'tabs_state',
      tabs: [entry('T-x2', 'chat-X', 'the real chat-X tab', WORKSPACE_DIR)],
      tabId: '',
    },
    [controllerE],
  );
  await waitFor(
    () => createdPanels.length === panelsBeforeE + 2,
    "chat-X's real tab must open once the local claim is void",
  );
  assert.strictEqual(tabIdOf(createdPanels[panelsBeforeE + 1]), 'T-x2');

  // --- E2: the chat's real tab is seen (and skipped as a duplicate of
  // the in-flight claim) BEFORE the registration drops — the exact
  // reviewed race. The skip must be remembered and resolved.
  managerE.openChat({chatId: 'chat-Y', title: 'never registers either'});
  assert.strictEqual(createdPanels.length, panelsBeforeE + 3);
  await daemonBroadcast(
    {
      type: 'tabs_state',
      tabs: [
        entry('T-x2', 'chat-X', 'the real chat-X tab', WORKSPACE_DIR),
        entry('T-y2', 'chat-Y', 'the real chat-Y tab', WORKSPACE_DIR),
      ],
      tabId: '',
    },
    [controllerE],
  );
  await new Promise(r => setTimeout(r, 150));
  assert.strictEqual(
    createdPanels.length,
    panelsBeforeE + 3,
    'while the local claim looks in-flight the real tab is skipped',
  );
  dropRegistration(createdPanels[panelsBeforeE + 2], 'chat-Y');
  await waitFor(
    () => createdPanels.length === panelsBeforeE + 4,
    'the remembered chat-Y tab must be adopted once the claim drops',
  );
  assert.strictEqual(tabIdOf(createdPanels[panelsBeforeE + 3]), 'T-y2');

  // --- E3: closing the claiming panel also frees the remembered tab.
  managerE.openChat({chatId: 'chat-Z', title: 'closed by the user'});
  assert.strictEqual(createdPanels.length, panelsBeforeE + 5);
  await daemonBroadcast(
    {
      type: 'tabs_state',
      tabs: [
        entry('T-x2', 'chat-X', 'the real chat-X tab', WORKSPACE_DIR),
        entry('T-y2', 'chat-Y', 'the real chat-Y tab', WORKSPACE_DIR),
        entry('T-z2', 'chat-Z', 'the real chat-Z tab', WORKSPACE_DIR),
      ],
      tabId: '',
    },
    [controllerE],
  );
  await new Promise(r => setTimeout(r, 150));
  assert.strictEqual(
    createdPanels.length,
    panelsBeforeE + 5,
    'the in-flight claim skips the real chat-Z tab for now',
  );
  createdPanels[panelsBeforeE + 4].dispose(); // user closes the claim
  await waitFor(
    () => createdPanels.length === panelsBeforeE + 6,
    'closing the claiming panel must open the remembered tab',
  );
  assert.strictEqual(tabIdOf(createdPanels[panelsBeforeE + 5]), 'T-z2');

  subE.dispose();
  managerE.dispose();
  sidebarE.dispose();
  await new Promise(r => setTimeout(r, 50));
  for (const sock of serverSockets) {
    if (!sock.destroyed) sock.destroy();
  }
  console.log('remoteTabOpensEditorPanel: all assertions passed');
}

runTest()
  .then(() => process.exit(0))
  .catch(err => {
    console.error(err);
    process.exit(1);
  });
