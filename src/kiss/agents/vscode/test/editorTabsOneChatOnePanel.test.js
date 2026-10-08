// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end tests for the editor-tabs surface invariant "one chat id,
// at most one open chat webview" (out/SorcarPanelManager.js). Runs
// against a REAL local-WSS daemon stub and the real extension-host
// wiring (out/SorcarSidebarView.js onRegistryTabsState):
//  - a panel adopted from a `tabs_state` snapshot is known as its
//    chat's panel IMMEDIATELY: a history click on that chat reveals it
//    instead of opening a second panel while the adopted panel's own
//    daemon socket is still connecting;
//  - the same holds for panels enterMode materializes when the user
//    switches editor-tabs mode on;
//  - when the registry displaces a panel's tab (another client
//    re-bound the chat to a new tab), adopting the replacement closes
//    the displaced panel on the host right away — WITHOUT retiring
//    the chat from the registry, without opening a fresh "last tab
//    closed" chat, and without waiting for the displaced webview's
//    own snapshot — so the chat never has two panels in the window;
//  - a panel whose own resume claim is still in flight is NOT
//    displaced by the registry tab it is about to displace (resume
//    race): that tab is skipped, not the panel;
//  - a panel the serializer revives takes its chat binding from the
//    latest controller snapshot, so a history click right after a
//    window reload reveals it instead of opening a second panel;
//  - the displacement sweep also covers a replacement tab this window
//    opened itself (already a panel, so nothing is adopted).

'use strict';

const assert = require('assert');
const fs = require('fs');
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

const WORKSPACE_DIR = '/ws/proj';
const createdPanels = [];
let registeredSerializer;

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
    reveals: 0,
    disposed: false,
    webview: {
      options: {},
      html: '',
      cspSource: 'vscode-resource:',
      asWebviewUri: uri => makeUri(uri.fsPath),
      postMessage: () => Promise.resolve(true),
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
  };
  return panel;
}

const vscodeStub = {
  workspace: {
    workspaceFolders: [{uri: makeUri(WORKSPACE_DIR)}],
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
    registerWebviewPanelSerializer: (_viewType, serializer) => {
      registeredSerializer = serializer;
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

const tmpHome = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-onechat-'));
process.env.HOME = tmpHome;
process.env.USERPROFILE = tmpHome;
fs.mkdirSync(path.join(tmpHome, '.kiss'), {recursive: true});
const endpointPath = path.join(tmpHome, '.kiss', 'sorcar-local.json');

const serverSockets = [];
const receivedCommands = [];
const server = createFakeDaemon(sock => {
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

/** Broadcast *msg* to every connected client and let it settle. */
function daemonBroadcast(msg) {
  const line = JSON.stringify(msg) + '\n';
  for (const sock of serverSockets) {
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

function entry(tabId, chatId, title) {
  return {
    tabId,
    chatId: chatId || '',
    title: title || '',
    workDir: WORKSPACE_DIR,
    scopeWorkDir: '',
  };
}

function tabIdOf(panel) {
  const m = /data-kiss-tab-id="([^"]+)"/.exec(panel.webview.html);
  assert.ok(m, 'panel html must carry data-kiss-tab-id');
  return m[1];
}

/** Open panels (not disposed) in creation order. */
function openPanels() {
  return createdPanels.filter(p => !p.disposed);
}

/** `closeTab` commands any client sent the daemon for *tabId*. */
function closeTabCommandsFor(tabId) {
  return receivedCommands.filter(
    c => c.type === 'closeTab' && String(c.tabId || '') === tabId,
  );
}

/** The extension.ts wiring under test (an empty window: no
 * placeholders, so every snapshot is adopted wholesale). */
function wire(sidebar, manager) {
  return sidebar.onRegistryTabsState(delta =>
    manager.adoptRegistryTabs(delta.added, delta.listed),
  );
}

async function runTest() {
  server.listen(endpointPath);
  const sidebarView = new SorcarSidebarView(vscodeStub.Uri.file(EXT_ROOT));
  const manager = new SorcarPanelManager(vscodeStub.Uri.file(EXT_ROOT));
  const sub = wire(sidebarView, manager);
  sidebarView.syncWorkDir();
  await waitFor(
    () => serverSockets.length >= 1,
    'the sidebar controller must connect to the daemon',
  );

  // --- 1. an adopted panel is its chat's panel at once ----------------
  await daemonBroadcast({
    type: 'tabs_state',
    tabs: [entry('T1', 'chat-1', 'remote task')],
    tabId: '',
  });
  await waitFor(() => createdPanels.length === 1, 'T1 must be adopted');
  const panel1 = createdPanels[0];
  assert.strictEqual(tabIdOf(panel1), 'T1');
  // The history view's click lands before the new panel's own daemon
  // socket delivered any snapshot (`chatBound`): the binding the
  // adopting snapshot listed must already count.
  manager.openChat({chatId: 'chat-1', title: 'remote task'});
  assert.strictEqual(
    createdPanels.length,
    1,
    'a history open of an adopted chat must not open a second panel',
  );
  assert.strictEqual(panel1.reveals, 1, 'the adopted panel is revealed');
  // An expanded history panel (onlyIfMissing) wants the chat on screen
  // somewhere: a panel already bound to it is neither revealed nor
  // moved to the task.
  manager.openChat({chatId: 'chat-1', taskId: 7, onlyIfMissing: true});
  assert.strictEqual(createdPanels.length, 1, 'onlyIfMissing opens nothing');
  assert.strictEqual(panel1.reveals, 1, 'onlyIfMissing reveals nothing');

  // --- 2. enterMode panels are bound too -------------------------------
  const managerB = new SorcarPanelManager(vscodeStub.Uri.file(EXT_ROOT));
  managerB.enterMode([entry('T2', 'chat-2', 'sidebar chat')]);
  assert.strictEqual(createdPanels.length, 2, 'enterMode opens T2');
  const panel2 = createdPanels[1];
  assert.strictEqual(tabIdOf(panel2), 'T2');
  managerB.openChat({chatId: 'chat-2'});
  assert.strictEqual(
    createdPanels.length,
    2,
    'a history open of an enterMode chat must reveal, not duplicate',
  );
  assert.strictEqual(panel2.reveals, 1);
  managerB.dispose();
  // Terminal teardown leaves the panel standing; drop it for the
  // open-panel bookkeeping below.
  panel2.dispose();
  await new Promise(r => setTimeout(r, 50));

  // --- 3. displacement closes the displaced panel on the host ---------
  // Another client re-binds chat-1 to T1b; the registry drops T1.
  await daemonBroadcast({
    type: 'tabs_state',
    tabs: [entry('T1b', 'chat-1', 'remote task moved')],
    tabId: '',
  });
  await waitFor(
    () => createdPanels.length === 3,
    'the replacement tab T1b must be adopted',
  );
  const panel1b = createdPanels[2];
  assert.strictEqual(tabIdOf(panel1b), 'T1b');
  assert.ok(
    panel1.disposed,
    'the displaced panel must close as soon as its replacement opens',
  );
  assert.deepStrictEqual(
    openPanels().map(tabIdOf),
    ['T1b'],
    'exactly one panel shows chat-1 — and no fresh chat was opened ' +
      'for the displaced panel closing',
  );
  assert.strictEqual(manager.panelCount, 1);
  await new Promise(r => setTimeout(r, 150));
  assert.strictEqual(
    closeTabCommandsFor('T1').length,
    0,
    'closing a displaced panel must not retire anything: the registry ' +
      'already dropped its tab',
  );
  manager.openChat({chatId: 'chat-1'});
  assert.strictEqual(createdPanels.length, 3, 'chat-1 has its panel');
  assert.strictEqual(panel1b.reveals, 1, 'the replacement is revealed');

  // --- 4. a resume race does not close the claiming panel ------------
  // The user resumes chat-3 here while the registry still lists the
  // chat on T3 (registered by another client): the local claim is in
  // flight, so T3 is skipped and the claiming panel stays.
  manager.openChat({chatId: 'chat-3', title: 'resumed here'});
  assert.strictEqual(createdPanels.length, 4);
  const panel3 = createdPanels[3];
  await daemonBroadcast({
    type: 'tabs_state',
    tabs: [entry('T1b', 'chat-1', 'remote task moved'), entry('T3', 'chat-3')],
    tabId: '',
  });
  await new Promise(r => setTimeout(r, 150));
  assert.strictEqual(createdPanels.length, 4, 'T3 is skipped');
  assert.ok(!panel3.disposed, 'the in-flight claim is not displaced');
  // The daemon then accepts the local bind: T3 is gone, the claiming
  // panel's tab is listed — nothing opens, nothing closes.
  await daemonBroadcast({
    type: 'tabs_state',
    tabs: [
      entry('T1b', 'chat-1', 'remote task moved'),
      entry(tabIdOf(panel3), 'chat-3', 'resumed here'),
    ],
    tabId: '',
  });
  await new Promise(r => setTimeout(r, 150));
  assert.deepStrictEqual(openPanels().map(tabIdOf), ['T1b', tabIdOf(panel3)]);

  // --- 5. a revived panel knows its chat from the latest snapshot ----
  // Window reload: the controller's first snapshot listed T5 (chat-5)
  // but adoption skipped it because a serialized placeholder holds
  // that tab (extension.ts filters by the persisted panel ids); the
  // workbench then revives the placeholder. A history click on chat-5
  // before the revived panel's own socket reports anything must
  // reveal it, not open a second panel.
  manager.registerSerializer();
  assert.ok(registeredSerializer, 'the manager registers its serializer');
  await daemonBroadcast({
    type: 'tabs_state',
    tabs: [
      entry('T1b', 'chat-1', 'remote task moved'),
      entry(tabIdOf(panel3), 'chat-3', 'resumed here'),
      entry('T5', 'chat-5', 'reloaded chat'),
    ],
    tabId: '',
  });
  // (The wiring here adopts wholesale — undo T5's adoption to model
  // the filtered first snapshot of a reloaded window.)
  const adoptedT5 = createdPanels.find(p => tabIdOf(p) === 'T5');
  assert.ok(adoptedT5, 'T5 adopted by the unfiltered wiring');
  adoptedT5.dispose();
  await new Promise(r => setTimeout(r, 150));
  const revived = makeFakePanel('kissSorcar.chatTab', '✅ reloaded chat');
  await registeredSerializer.deserializeWebviewPanel(revived, {
    editorRootTabId: 'T5',
  });
  assert.strictEqual(tabIdOf(revived), 'T5');
  const panelsBeforeRevivalClick = createdPanels.length;
  manager.openChat({chatId: 'chat-5', title: 'reloaded chat'});
  assert.strictEqual(
    createdPanels.length,
    panelsBeforeRevivalClick,
    'a history open of a revived chat must reveal the revived panel',
  );
  assert.strictEqual(revived.reveals, 1);

  // --- 6. the sweep closes a displaced panel even when the chat's
  // new tab is one this window opened itself ---------------------------
  // A fresh panel of this window has chat-5 resumed into its root tab
  // (the daemon rebinds the chat to that tab and drops T5). The new
  // tab is already a panel, so nothing is adopted — the revived panel
  // must still close: nothing retired, no fresh chat.
  manager.openNewChat();
  const freshPanel = createdPanels[createdPanels.length - 1];
  const freshTabId = tabIdOf(freshPanel);
  assert.strictEqual(createdPanels.length, panelsBeforeRevivalClick + 1);
  // (The manual close of the adopted T5 panel above was a user close
  // and retired T5 once; the displacement below must add nothing.)
  const t5ClosesBefore = closeTabCommandsFor('T5').length;
  await daemonBroadcast({
    type: 'tabs_state',
    tabs: [
      entry('T1b', 'chat-1', 'remote task moved'),
      entry(tabIdOf(panel3), 'chat-3', 'resumed here'),
      entry(freshTabId, 'chat-5', 'reloaded chat'),
    ],
    tabId: '',
  });
  assert.ok(revived.disposed, 'the displaced revived panel must close');
  assert.ok(!freshPanel.disposed, 'the listed panel stays');
  await new Promise(r => setTimeout(r, 150));
  assert.strictEqual(
    closeTabCommandsFor('T5').length,
    t5ClosesBefore,
    'the displaced revived panel must not retire chat-5',
  );
  assert.deepStrictEqual(
    openPanels().map(tabIdOf).sort(),
    ['T1b', tabIdOf(panel3), freshTabId].sort(),
    'one panel per chat, no fresh chat opened',
  );
  manager.openChat({chatId: 'chat-5'});
  assert.strictEqual(
    createdPanels.length,
    panelsBeforeRevivalClick + 1,
    'chat-5 has exactly one panel to reveal',
  );
  assert.strictEqual(freshPanel.reveals, 1);

  // --- 7. history open BEFORE the revival: the husk is dropped -------
  // The inverse ordering of phase 5: the snapshot lists T7 (chat-7)
  // held by a placeholder, the user clicks chat-7 in history before
  // the workbench revives it (openChat finds no panel: a resume panel
  // opens), then the placeholder revives. One chat, one panel: the
  // revived husk is dropped, and T7 is remembered so the chat's
  // registry tab reopens if the resume claim goes away.
  await daemonBroadcast({
    type: 'tabs_state',
    tabs: [
      entry('T1b', 'chat-1', 'remote task moved'),
      entry(tabIdOf(panel3), 'chat-3', 'resumed here'),
      entry(freshTabId, 'chat-5', 'reloaded chat'),
      entry('T7', 'chat-7', 'reloaded chat 7'),
    ],
    tabId: '',
  });
  const adoptedT7 = createdPanels.find(p => tabIdOf(p) === 'T7');
  assert.ok(adoptedT7, 'T7 adopted by the unfiltered wiring');
  adoptedT7.dispose(); // model the filtered first snapshot again
  await new Promise(r => setTimeout(r, 150));
  manager.openChat({chatId: 'chat-7', title: 'reloaded chat 7'});
  const resume7 = createdPanels[createdPanels.length - 1];
  assert.notStrictEqual(tabIdOf(resume7), 'T7', 'a resume panel opened');
  const revived7 = makeFakePanel('kissSorcar.chatTab', 'reloaded chat 7');
  const panelsBeforeRevival7 = createdPanels.length;
  await registeredSerializer.deserializeWebviewPanel(revived7, {
    editorRootTabId: 'T7',
  });
  assert.ok(revived7.disposed, 'the revived husk of a claimed chat drops');
  assert.ok(!resume7.disposed, 'the resume panel stays');
  assert.strictEqual(createdPanels.length, panelsBeforeRevival7);
  manager.openChat({chatId: 'chat-7'});
  assert.strictEqual(createdPanels.length, panelsBeforeRevival7);
  assert.strictEqual(resume7.reveals, 1, 'chat-7 has one panel');
  // The user closes the resume panel: the remembered registry tab
  // T7 is adopted so chat-7 keeps an editor tab.
  resume7.dispose();
  await waitFor(
    () => createdPanels.length === panelsBeforeRevival7 + 1,
    'the remembered T7 must be adopted once the claim is gone',
  );
  assert.strictEqual(tabIdOf(createdPanels[panelsBeforeRevival7]), 'T7');

  sub.dispose();
  manager.dispose();
  sidebarView.dispose();
  await new Promise(r => setTimeout(r, 50));
  for (const sock of serverSockets) {
    if (!sock.destroyed) sock.destroy();
  }
  server.close();
  console.log('editorTabsOneChatOnePanel: all assertions passed');
}

runTest()
  .then(() => process.exit(0))
  .catch(err => {
    console.error(err);
    process.exit(1);
  });
