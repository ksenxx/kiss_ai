// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end tests for the editor-tabs-mode invariant "at least one
// chat editor tab is always open" (out/SorcarPanelManager.js against
// a stubbed VS Code API and a REAL Unix-domain-socket daemon stub) —
// the editor-tab analogue of the sidebar strip never running out of
// chat tabs (main.js closeTab -> createNewTab, reconcileTabs'
// placeholder):
//  - the user closing the LAST chat panel (editor tab X) opens a fresh
//    chat that takes the focus; closing one of several opens nothing;
//  - the webview's closePanel (root chat closed inside the panel, or
//    the registry dropping the tab) on the last panel opens a fresh
//    chat — focused for a user close, in the background otherwise;
//  - a restored placeholder tab (tabGroups) counts as an open chat, so
//    closing the last LIVE panel next to one opens nothing, while the
//    placeholder itself closing (tabGroups.onDidChangeTabs — it never
//    became a panel) opens a fresh chat in the background;
//  - one live close reaching the host twice (tabs-model event and
//    panel dispose, in either order) opens exactly ONE replacement;
//  - a pending same-chat registry adoption triggered by the close
//    satisfies the invariant by itself (no extra empty chat);
//  - enterMode with nothing to migrate opens one chat;
//  - closeAll (mode off), terminal dispose and a switched-off mode
//    never spawn a replacement.

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
const tabsChanged = new StubEventEmitter();
let tabGroupsAll = [];

function makeFakePanel(viewType, title, showOptions) {
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
    // How the panel was opened: {viewColumn, preserveFocus}.
    showOptions,
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
  };
  return panel;
}

const CHAT_TAB = {input: {viewType: 'mainThreadWebview-kissSorcar.chatTab'}};
const FILE_TAB = {input: {uri: makeUri('/ws/a.py')}};

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
    createWebviewPanel: (viewType, title, showOptions, _options) => {
      const panel = makeFakePanel(viewType, title, showOptions);
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
    tabGroups: {
      get all() {
        return tabGroupsAll;
      },
      onDidChangeTabs: cb => tabsChanged.event(cb),
    },
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
const sockPath = path.join(tmpHome, '.kiss', 'sorcar.sock');

const daemonCommands = [];
const server = net.createServer(sock => {
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

async function waitFor(predicate, message) {
  for (let i = 0; i < 150; i++) {
    if (predicate()) return;
    await new Promise(r => setTimeout(r, 20));
  }
  throw new Error(message || 'waitFor timed out');
}

const sleep = ms => new Promise(r => setTimeout(r, ms));

const {SorcarPanelManager, CHAT_PANEL_VIEW_TYPE} = require(
  path.join(OUT_DIR, 'SorcarPanelManager.js'),
);

function tabIdOf(panel) {
  const m = /data-kiss-tab-id="([^"]+)"/.exec(panel.webview.html);
  assert.ok(m, 'panel html must carry data-kiss-tab-id');
  return m[1];
}

function livePanels() {
  return createdPanels.filter(p => !p.disposed);
}

function lastPanel() {
  return createdPanels[createdPanels.length - 1];
}

/** The fresh chat took the focus: revealed with focus AND told to focus
 * its composer (SorcarSidebarView.focusChatInput). */
async function assertFocusedReplacement(panel) {
  assert.strictEqual(panel.showOptions.preserveFocus, false);
  await waitFor(
    () => panel._posted.some(m => m.type === 'focusInput'),
    'a user-close replacement must focus its composer',
  );
  assert.ok(panel.reveals >= 1, 'a user-close replacement is revealed');
}

async function assertBackgroundReplacement(panel) {
  assert.strictEqual(
    panel.showOptions.preserveFocus,
    true,
    'an automatic replacement opens without stealing focus',
  );
  await sleep(200);
  assert.ok(
    !panel._posted.some(m => m.type === 'focusInput'),
    'an automatic replacement must not focus its composer',
  );
}

async function runTest() {
  server.listen(sockPath);
  const retired = [];
  const manager = new SorcarPanelManager(
    vscodeStub.Uri.file(EXT_ROOT),
    tabId => retired.push(tabId),
  );
  const watcher = manager.watchEditorTabs();
  assert.strictEqual(typeof watcher.dispose, 'function');

  // --- ensureChatOpen opens the very first chat (activation) -----------
  const first = manager.ensureChatOpen({preserveFocus: true});
  assert.ok(first, 'no chat tab anywhere: ensureChatOpen opens one');
  assert.strictEqual(createdPanels.length, 1);
  assert.strictEqual(createdPanels[0].viewType, CHAT_PANEL_VIEW_TYPE);
  await assertBackgroundReplacement(createdPanels[0]);
  assert.strictEqual(
    manager.ensureChatOpen(),
    undefined,
    'a chat is open: ensureChatOpen is a no-op',
  );
  assert.strictEqual(createdPanels.length, 1);

  // --- user closes the LAST panel (editor tab X): a fresh, focused chat
  const panelA = createdPanels[0];
  const tabA = tabIdOf(panelA);
  panelA.dispose();
  assert.strictEqual(createdPanels.length, 2, 'last close opens a chat');
  const panelB = lastPanel();
  assert.ok(!panelB.disposed);
  assert.notStrictEqual(tabIdOf(panelB), tabA, 'the replacement is fresh');
  assert.deepStrictEqual(retired, [tabA], 'the closed chat is retired');
  assert.strictEqual(manager.panelCount, 1);
  await assertFocusedReplacement(panelB);
  assert.strictEqual(
    manager.activeController(),
    manager.revealActiveOrCreate(),
    'the replacement is the active panel',
  );

  // --- closing one of two panels opens nothing ------------------------
  manager.openNewChat();
  assert.strictEqual(createdPanels.length, 3);
  const panelC = lastPanel();
  const tabC = tabIdOf(panelC);
  panelC.dispose();
  await sleep(50);
  assert.strictEqual(createdPanels.length, 3, 'a chat remains: no open');
  assert.strictEqual(manager.panelCount, 1);
  assert.ok(!panelB.disposed);
  assert.deepStrictEqual(retired, [tabA, tabC], 'the user close retires');

  // --- closePanel from the webview WITHOUT retire (registry dropped the
  // tab / another client closed it): background replacement -----------
  panelB._recv.fire({type: 'closePanel'});
  await waitFor(() => panelB.disposed, 'closePanel disposes the panel');
  assert.strictEqual(createdPanels.length, 4, 'remote close: chat reopened');
  const panelD = lastPanel();
  await assertBackgroundReplacement(panelD);
  assert.deepStrictEqual(retired, [tabA, tabC], 'a remote close retires nothing');

  // --- closePanel WITH retire (user closed the root chat inside the
  // webview): focused replacement, chat retired -----------------------
  const tabD = tabIdOf(panelD);
  panelD._recv.fire({type: 'closePanel', retire: true});
  await waitFor(() => panelD.disposed, 'closePanel(retire) disposes');
  assert.strictEqual(createdPanels.length, 5);
  const panelE = lastPanel();
  await assertFocusedReplacement(panelE);
  assert.deepStrictEqual(retired, [tabA, tabC, tabD]);

  // --- a restored placeholder counts as an open chat ------------------
  // After a window reload the workbench lists the chat tabs it restored
  // long before the manager revives them; closing the only LIVE panel
  // next to one must not open a duplicate.
  tabGroupsAll = [{tabs: [FILE_TAB, CHAT_TAB]}];
  assert.strictEqual(manager.chatEditorTabCount(), 1);
  assert.ok(manager.hasChatEditorTab());
  panelE.dispose();
  await sleep(50);
  assert.strictEqual(createdPanels.length, 5, 'placeholder standing: no open');
  assert.strictEqual(manager.panelCount, 0);
  assert.ok(manager.hasChatEditorTab(), 'the placeholder still counts');
  assert.strictEqual(manager.ensureChatOpen(), undefined);

  // --- the placeholder itself closes (never revived: no panel dispose
  // fires) — the tabGroups backstop opens a background chat -----------
  tabGroupsAll = [{tabs: [FILE_TAB]}];
  tabsChanged.fire({opened: [], closed: [FILE_TAB], changed: []});
  assert.strictEqual(createdPanels.length, 5, 'a file tab close: nothing');
  tabsChanged.fire({opened: [], closed: [CHAT_TAB], changed: []});
  assert.strictEqual(createdPanels.length, 6, 'placeholder closed: chat opened');
  const panelF = lastPanel();
  await assertBackgroundReplacement(panelF);
  assert.strictEqual(manager.panelCount, 1);
  // The live panel is an editor tab too; a further chat-tab close event
  // with it standing opens nothing.
  tabGroupsAll = [{tabs: [CHAT_TAB]}];
  tabsChanged.fire({opened: [], closed: [CHAT_TAB], changed: []});
  assert.strictEqual(createdPanels.length, 6);
  tabGroupsAll = [];

  // --- one live close, two notifications, ONE replacement --------------
  // A live panel's close reaches the host twice: the tabGroups model
  // update and the WebviewPanel dispose. Whichever lands first opens
  // the replacement; the other must find it and do nothing — in both
  // orders.
  // (a) tabs model first (the workbench's actual order): the panel is
  // still live when the tab-close event fires.
  tabGroupsAll = [];
  tabsChanged.fire({opened: [], closed: [CHAT_TAB], changed: []});
  assert.strictEqual(createdPanels.length, 6, 'panel still live: no open');
  panelF.dispose();
  assert.strictEqual(createdPanels.length, 7, 'dispose opens the one chat');
  const panelF2 = lastPanel();
  await assertFocusedReplacement(panelF2);
  // (b) dispose first while the tabs model still lists the closed tab:
  // the dispose path sees a chat tab standing and defers to the
  // backstop, which opens exactly one chat once the model catches up.
  tabGroupsAll = [{tabs: [CHAT_TAB]}];
  panelF2.dispose();
  assert.strictEqual(createdPanels.length, 7, 'stale model: dispose waits');
  assert.strictEqual(manager.panelCount, 0);
  tabGroupsAll = [];
  tabsChanged.fire({opened: [], closed: [CHAT_TAB], changed: []});
  assert.strictEqual(createdPanels.length, 8, 'backstop opens the one chat');
  await assertBackgroundReplacement(lastPanel());
  tabsChanged.fire({opened: [], closed: [CHAT_TAB], changed: []});
  assert.strictEqual(createdPanels.length, 8, 'a repeat event: no dup');
  // Hand the rest of the script a live panel named F again.
  const panelF3 = lastPanel();

  // --- a pending same-chat registry adoption satisfies the invariant --
  // Panel X resumes chat-X (registration unconfirmed); the registry
  // then lists chat-X under another tab id, whose adoption the claim
  // blocks. Closing X releases the claim: the remembered registry tab
  // opens as X's successor — and THAT is the one open chat, no empty
  // extra.
  // The live panel (F3) is empty; open the resuming panel through it.
  panelF3._recv.fire({type: 'openChatPanel', chatId: 'chat-X', title: 'X'});
  await waitFor(() => createdPanels.length === 9, 'resume panel opens');
  const panelX = lastPanel();
  assert.ok(panelX.webview.html.includes('data-kiss-resume-chat-id="chat-X"'));
  panelF3.dispose();
  await sleep(50);
  assert.strictEqual(createdPanels.length, 9, 'X remains: no open');
  manager.adoptRegistryTabs(
    [{tabId: 'reg-x', chatId: 'chat-X', title: 'X', workDir: '', scopeWorkDir: ''}],
    '',
  );
  assert.strictEqual(createdPanels.length, 9, 'same-chat tab: adoption blocked');
  panelX.dispose();
  assert.strictEqual(
    createdPanels.length,
    10,
    'exactly ONE panel opens: the remembered registry tab, no empty extra',
  );
  const panelRegX = lastPanel();
  assert.strictEqual(tabIdOf(panelRegX), 'reg-x');
  assert.strictEqual(manager.panelCount, 1);

  // --- closeAll (mode switched off) spawns no replacement -------------
  manager.closeAll();
  assert.ok(panelRegX.disposed);
  await sleep(50);
  assert.strictEqual(createdPanels.length, 10, 'closeAll: no replacement');
  assert.strictEqual(manager.panelCount, 0);

  // --- mode off: ensureChatOpen and the backstop open nothing ---------
  editorTabsMode = false;
  assert.strictEqual(manager.ensureChatOpen(), undefined);
  tabsChanged.fire({opened: [], closed: [CHAT_TAB], changed: []});
  assert.strictEqual(createdPanels.length, 10, 'mode off: nothing opens');
  editorTabsMode = true;

  // --- enterMode with nothing to migrate opens one chat ---------------
  manager.enterMode([], '/ws');
  assert.strictEqual(createdPanels.length, 11, 'enterMode: one chat');
  await assertFocusedReplacement(lastPanel());
  manager.enterMode([], '/ws');
  assert.strictEqual(createdPanels.length, 11, 'enterMode again: no dup');

  // --- terminal dispose: panels stand, nothing new opens --------------
  manager.dispose();
  await sleep(50);
  assert.strictEqual(createdPanels.length, 11, 'teardown: no open');
  assert.ok(!lastPanel().disposed, 'teardown leaves the editor tab');
  assert.strictEqual(
    manager.ensureChatOpen(),
    undefined,
    'after teardown ensureChatOpen is inert',
  );
  lastPanel().dispose();
  assert.strictEqual(createdPanels.length, 11, 'post-teardown close: inert');
  watcher.dispose();

  // --- watchEditorTabs without the tabGroups API is a no-op -----------
  const savedTabGroups = vscodeStub.window.tabGroups;
  vscodeStub.window.tabGroups = undefined;
  const bare = new SorcarPanelManager(vscodeStub.Uri.file(EXT_ROOT));
  const noop = bare.watchEditorTabs();
  assert.strictEqual(typeof noop.dispose, 'function');
  noop.dispose();
  assert.strictEqual(bare.chatEditorTabCount(), -1);
  assert.strictEqual(
    bare.hasChatEditorTab(),
    false,
    'no API and no panel: nothing is open',
  );
  vscodeStub.window.tabGroups = savedTabGroups;

  console.log('editorTabsAlwaysOneChat: all tests passed');
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
