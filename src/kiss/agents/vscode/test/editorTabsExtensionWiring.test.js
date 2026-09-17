// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end tests for extension.ts's editor-tabs-mode command routing
// (out/extension.js activated against a stubbed VS Code API):
//  - kissSorcar.openSettings is registered; with the mode ON it opens
//    the panel manager's settings, with the mode OFF the sidebar's;
//  - newConversation / openPanel / stopTask / runSelection /
//    insertSelectionToChat go to the panel manager's controllers when
//    the mode is ON and to the sidebar view when OFF;
//  - flipping kissSorcar.editorTabsMode ON migrates the registry's
//    tabs into panels (enterMode) and closes the secondary sidebar
//    (workbench.action.closeAuxiliaryBar); OFF closes all panels and
//    opens the secondary sidebar on the KISS Sorcar chat view, focusing
//    its composer (sidebarView.focusChatInput);
//  - the KS activity-bar button's dummy tree (non-editor mode): on
//    becoming visible it closes the primary sidebar and reveals the
//    chat in the secondary sidebar without creating a chat — except
//    right after an editorTabsMode flip, whose config handler already
//    revealed the chat (no second, competing reveal).
//
// The panel manager and sidebar view are replaced by instrumented
// fakes (the real ones have their own end-to-end suites:
// editorTabsPanelManager.test.js and the sidebar suites).

'use strict';

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const Module = require('module');

const EXT_ROOT = path.join(__dirname, '..');
const OUT_DIR = path.join(EXT_ROOT, 'out');
const extensionPath = path.join(OUT_DIR, 'extension.js');
assert.ok(
  fs.existsSync(extensionPath),
  'compiled extension missing — run `npm run compile` first',
);

function makeDisposable() {
  return {dispose: () => {}};
}

function makeMemento() {
  const store = new Map();
  return {
    get: (key, def) => (store.has(key) ? store.get(key) : def),
    update: (key, value) => {
      if (value === undefined) store.delete(key);
      else store.set(key, value);
      return Promise.resolve();
    },
  };
}

let editorTabsMode = false;
const commands = new Map();
const executedCommands = [];
const configListeners = [];
const viewProviders = new Map();
const treeVisibilityListeners = [];

const vscodeStub = {
  window: {
    registerWebviewViewProvider: (id, provider) => {
      viewProviders.set(id, provider);
      return makeDisposable();
    },
    createTreeView: () => ({
      onDidChangeVisibility: cb => {
        treeVisibilityListeners.push(cb);
        return makeDisposable();
      },
      dispose: () => {},
    }),
    showInformationMessage: () => Promise.resolve(undefined),
    showErrorMessage: () => Promise.resolve(undefined),
    showWarningMessage: () => Promise.resolve(undefined),
    activeTextEditor: undefined,
  },
  commands: {
    registerCommand: (id, fn) => {
      commands.set(id, fn);
      return makeDisposable();
    },
    executeCommand: (cmd, ...args) => {
      executedCommands.push({cmd, args});
      return Promise.resolve();
    },
  },
  workspace: {
    workspaceFolders: [{uri: {fsPath: '/ws/project'}}],
    asRelativePath: p => String(p),
    getConfiguration: () => ({
      get: key => (key === 'editorTabsMode' ? editorTabsMode : ''),
      update: () => Promise.resolve(),
    }),
    onDidChangeConfiguration: cb => {
      configListeners.push(cb);
      return makeDisposable();
    },
  },
  Uri: {
    file: p => ({fsPath: p, scheme: 'file', toString: () => `file://${p}`}),
  },
  EventEmitter: class {
    constructor() {
      this._listeners = [];
      this.event = cb => {
        this._listeners.push(cb);
        return makeDisposable();
      };
    }
    fire(arg) {
      for (const cb of this._listeners.slice()) cb(arg);
    }
    dispose() {
      this._listeners = [];
    }
  },
  TreeItem: class {
    constructor(label) {
      this.label = label;
    }
  },
};

const origLoad = Module._load;
Module._load = function (request, parent, isMain) {
  if (request === 'vscode') return vscodeStub;
  return origLoad.call(this, request, parent, isMain);
};

const calls = {
  sidebar: {
    focusChatInput: 0,
    newConversation: 0,
    stopTask: 0,
    submitTask: [],
    appendToInput: [],
    openSettingsUI: 0,
    gitCommit: 0,
    widenToOneThird: 0,
  },
  manager: {
    openNewChat: 0,
    revealActiveOrCreate: 0,
    openSettings: 0,
    enterMode: [],
    adoptRegistryTabs: [],
    closeAll: 0,
    openChat: [],
    watchEditorTabs: 0,
    ensureChatOpen: [],
  },
  controller: {
    focusChatInput: 0,
    newConversation: 0,
    stopTask: 0,
    submitTask: [],
    appendToInput: [],
    gitCommit: 0,
  },
};

const registryEntries = [
  {tabId: 't1', chatId: 'c1', title: 'one', workDir: '/ws/project', scopeWorkDir: ''},
];

const registryTabsAddedListeners = [];

const sidebarInstances = [];
// The one-time widening's first-resolve hook, when armed.
let firstResolveCb = null;

class FakeSidebarView {
  constructor(_uri, panelHooks) {
    this.hasFocus = false;
    this.panelHooks = panelHooks;
    this.resolvedViews = [];
    sidebarInstances.push(this);
  }
  resolveWebviewView(view) {
    this.resolvedViews.push(view);
  }
  syncWorkDir() {}
  focusChatInput() {
    calls.sidebar.focusChatInput += 1;
    return Promise.resolve();
  }
  newConversation() {
    calls.sidebar.newConversation += 1;
  }
  stopTask() {
    calls.sidebar.stopTask += 1;
  }
  submitTask(text) {
    calls.sidebar.submitTask.push(text);
    return Promise.resolve();
  }
  appendToInput(text) {
    calls.sidebar.appendToInput.push(text);
    return Promise.resolve();
  }
  openSettingsUI() {
    calls.sidebar.openSettingsUI += 1;
    return Promise.resolve();
  }
  gitCommit() {
    calls.sidebar.gitCommit += 1;
    return Promise.resolve();
  }
  getRegistryTabEntries() {
    return registryEntries;
  }
  onRegistryTabsState(cb) {
    registryTabsAddedListeners.push(cb);
    return makeDisposable();
  }
  onCommitMessage() {
    return makeDisposable();
  }
  generateCommitMessage() {
    return Promise.resolve();
  }
  onFirstResolve(cb) {
    firstResolveCb = cb;
  }
  widenToOneThird() {
    calls.sidebar.widenToOneThird += 1;
    return Promise.resolve();
  }
  runUpdate() {}
  dispose() {}
}

class FakeController {
  constructor() {
    this.hasFocus = false;
  }
  focusChatInput() {
    calls.controller.focusChatInput += 1;
    return Promise.resolve();
  }
  newConversation() {
    calls.controller.newConversation += 1;
  }
  stopTask() {
    calls.controller.stopTask += 1;
  }
  submitTask(text) {
    calls.controller.submitTask.push(text);
    return Promise.resolve();
  }
  appendToInput(text) {
    calls.controller.appendToInput.push(text);
    return Promise.resolve();
  }
  gitCommit() {
    calls.controller.gitCommit += 1;
    return Promise.resolve();
  }
}

const fakeController = new FakeController();

let fakePanelCount = 0;

class FakePanelManager {
  constructor(_uri, _retireTab, recordPanelTab) {
    // The extension's workspaceState-backed panel-id recorder, so the
    // test can drive its add/remove semantics directly.
    calls.manager.recordPanelTab = recordPanelTab;
  }
  static modeEnabled() {
    return editorTabsMode;
  }
  get panelCount() {
    return fakePanelCount;
  }
  registerSerializer() {
    return makeDisposable();
  }
  watchEditorTabs() {
    calls.manager.watchEditorTabs += 1;
    return makeDisposable();
  }
  // Mirrors the real manager: live panels first, then the tabGroups
  // model (restored placeholders), -1 / false when the API is absent.
  chatEditorTabCount() {
    const groups = vscodeStub.window.tabGroups && vscodeStub.window.tabGroups.all;
    if (!groups) return -1;
    let count = 0;
    for (const group of groups) {
      for (const tab of group.tabs) {
        const viewType = tab.input && tab.input.viewType;
        if (typeof viewType === 'string' && viewType.includes('kissSorcar.chatTab')) {
          count += 1;
        }
      }
    }
    return count;
  }
  hasChatEditorTab() {
    if (fakePanelCount > 0) return true;
    return this.chatEditorTabCount() > 0;
  }
  ensureChatOpen(opts) {
    calls.manager.ensureChatOpen.push(!!(opts && opts.preserveFocus));
    if (!editorTabsMode) return undefined;
    if (this.hasChatEditorTab()) return undefined;
    const controller = this.openNewChat();
    if (!(opts && opts.preserveFocus)) void controller.focusChatInput();
    return controller;
  }
  openNewChat() {
    calls.manager.openNewChat += 1;
    fakePanelCount += 1;
    return fakeController;
  }
  openChat(event) {
    calls.manager.openChat.push(event);
  }
  revealActiveOrCreate() {
    calls.manager.revealActiveOrCreate += 1;
    return fakeController;
  }
  activeController() {
    // Mirrors the real manager: undefined when no panel exists.
    return fakePanelCount > 0 ? fakeController : undefined;
  }
  openSettings() {
    calls.manager.openSettings += 1;
    return Promise.resolve();
  }
  enterMode(entries, workspaceDir) {
    calls.manager.enterMode.push({entries, workspaceDir});
  }
  adoptRegistryTabs(entries, workspaceDir, listed) {
    calls.manager.adoptRegistryTabs.push({entries, workspaceDir, listed});
  }
  closeAll() {
    calls.manager.closeAll += 1;
    fakePanelCount = 0;
  }
  markShutdown() {}
  dispose() {}
}

function stubModule(filePath, exports) {
  const fakeMod = new Module(filePath);
  fakeMod.filename = filePath;
  fakeMod.loaded = true;
  fakeMod.exports = exports;
  require.cache[filePath] = fakeMod;
}

stubModule(path.join(OUT_DIR, 'SorcarSidebarView.js'), {
  SorcarSidebarView: FakeSidebarView,
});
stubModule(path.join(OUT_DIR, 'SorcarPanelManager.js'), {
  SorcarPanelManager: FakePanelManager,
  CHAT_PANEL_VIEW_TYPE: 'kissSorcar.chatTab',
});
stubModule(path.join(OUT_DIR, 'DependencyInstaller.js'), {
  ensureLocalBinInPath: () => {},
  ensureDependencies: () => Promise.resolve(),
});
stubModule(path.join(OUT_DIR, 'gitApi.js'), {
  getGitApi: () => Promise.resolve(undefined),
});
stubModule(path.join(OUT_DIR, 'reloadGuard.js'), {
  isReloadReady: () => ({codeReady: false, socketUp: false, size: 0}),
});
stubModule(path.join(OUT_DIR, 'kissPaths.js'), {
  findKissProject: () => '/fake/kiss_project',
});
stubModule(path.join(OUT_DIR, 'WebviewNotifications.js'), {
  showInformationNotification: () => Promise.resolve(undefined),
  showWarningNotification: () => Promise.resolve(undefined),
  showErrorNotification: () => Promise.resolve(undefined),
});
stubModule(path.join(OUT_DIR, 'UpdateChecker.js'), {
  checkForExtensionUpdate: () => Promise.resolve({checked: false}),
  snoozeUpdateNotification: () => ({}),
});
stubModule(path.join(OUT_DIR, 'SorcarTab.js'), {
  resetTipsOnExtensionUpdate: () => {},
  HISTORY_PANEL_TAB_ID: 'history-panel',
  historyPanelBodyAttrs: () =>
    ' class="editor-tab-mode history-panel-mode"' +
    ' data-kiss-tab-id="history-panel"',
});

delete require.cache[require.resolve(extensionPath)];
const extension = require(extensionPath);

async function fireConfigChange() {
  for (const cb of configListeners) {
    cb({affectsConfiguration: s => s === 'kissSorcar.editorTabsMode'});
  }
  await new Promise(r => setTimeout(r, 20));
}

async function runTest() {
  const tmpExtPath = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-edwire-'));
  const ctx = {
    extensionUri: vscodeStub.Uri.file(tmpExtPath),
    extensionPath: tmpExtPath,
    subscriptions: [],
    workspaceState: makeMemento(),
    globalState: makeMemento(),
  };
  // The one-time flows (auto-open, widen) are not under test here.
  await ctx.workspaceState.update('firstLaunchDone', true);
  await ctx.workspaceState.update('sidebarWidened', true);
  // The previous session persisted its open panel tab ids: first
  // snapshots may then be FILTERED against them instead of being
  // suppressed wholesale (see the first-snapshot scenarios below).
  await ctx.workspaceState.update('kissSorcar.editorPanelTabIds', ['t-prior']);

  extension.activate(ctx);
  assert.ok(commands.has('kissSorcar.openSettings'), 'openSettings command');

  // --- the one-chat invariant at activation ----------------------------
  // The tabGroups backstop is always subscribed; the activation open is
  // asked for in the background and, with the mode OFF, opens nothing.
  assert.strictEqual(calls.manager.watchEditorTabs, 1, 'tabs backstop wired');
  assert.deepStrictEqual(
    calls.manager.ensureChatOpen,
    [true],
    'activation checks the invariant without stealing focus',
  );
  assert.strictEqual(calls.manager.openNewChat, 0, 'mode off: no chat tab');

  // --- sidebar mode (default) -----------------------------------------
  await commands.get('kissSorcar.openSettings')();
  assert.strictEqual(calls.sidebar.openSettingsUI, 1);
  assert.strictEqual(calls.manager.openSettings, 0);

  await commands.get('kissSorcar.newConversation')();
  assert.strictEqual(calls.sidebar.newConversation, 1);
  assert.strictEqual(calls.manager.openNewChat, 0);

  await commands.get('kissSorcar.stopTask')();
  assert.strictEqual(calls.sidebar.stopTask, 1);

  await commands.get('kissSorcar.gitCommit')();
  assert.strictEqual(calls.sidebar.gitCommit, 1);
  assert.strictEqual(calls.controller.gitCommit, 0);

  // The KS button's command in sidebar mode: just focus the chat.
  const sidebarFocusBefore = calls.sidebar.focusChatInput;
  await commands.get('kissSorcar.showHistory')();
  assert.strictEqual(
    calls.sidebar.focusChatInput,
    sidebarFocusBefore + 1,
    'sidebar mode: showHistory focuses the chat',
  );
  assert.strictEqual(calls.manager.openNewChat, 0);

  // --- the KS activity-bar button (sidebar mode): its dummy tree -------
  // Becoming visible closes the primary sidebar (nothing — history
  // panel included — may stay open there) and reveals the chat in the
  // secondary sidebar, creating no chat.
  assert.strictEqual(treeVisibilityListeners.length, 1, 'tree handler');
  const treeFocusBefore = calls.sidebar.focusChatInput;
  executedCommands.length = 0;
  for (const cb of treeVisibilityListeners) cb({visible: true});
  await new Promise(r => setTimeout(r, 120));
  assert.deepStrictEqual(
    executedCommands.map(e => e.cmd),
    ['workbench.action.closeSidebar'],
    'KS activity-bar click: the primary sidebar closes',
  );
  assert.strictEqual(
    calls.sidebar.focusChatInput,
    treeFocusBefore + 1,
    'KS activity-bar click: the secondary-sidebar chat is revealed',
  );
  assert.strictEqual(calls.manager.openNewChat, 0, 'no chat created');

  // A hidden tree does nothing.
  executedCommands.length = 0;
  for (const cb of treeVisibilityListeners) cb({visible: false});
  await new Promise(r => setTimeout(r, 120));
  assert.deepStrictEqual(executedCommands, []);
  assert.strictEqual(calls.sidebar.focusChatInput, treeFocusBefore + 1);

  // --- registry tabs other clients create: sidebar mode ignores them
  // (its webview adopts them itself), editor-tabs mode materializes
  // them through the panel manager, scoped to this workspace.
  const remoteEntries = [
    {tabId: 'rt', chatId: 'rc', title: 'remote', workDir: '/ws/project', scopeWorkDir: ''},
  ];
  // The tab behind the previous session's persisted panel id: on a
  // first snapshot it may be a serialized placeholder of this window.
  const priorEntry = {
    tabId: 't-prior',
    chatId: 'pc',
    title: 'prior',
    workDir: '/ws/project',
    scopeWorkDir: '',
  };
  const remoteListed = [remoteEntries[0], priorEntry];
  const fireDelta = (firstSnapshot, added = remoteEntries) => {
    for (const cb of registryTabsAddedListeners) {
      cb({added, listed: remoteListed, firstSnapshot});
    }
  };
  assert.strictEqual(
    registryTabsAddedListeners.length,
    1,
    'activation subscribes to the sidebar controller registry deltas',
  );
  fireDelta(false);
  assert.strictEqual(
    calls.manager.adoptRegistryTabs.length,
    0,
    'sidebar mode: a remote tab must not open an editor tab',
  );

  // --- flip the mode ON: registry tabs migrate to panels and the
  // secondary sidebar (which hosted the sidebar chat) closes ------------
  editorTabsMode = true;
  executedCommands.length = 0;
  await fireConfigChange();
  assert.strictEqual(calls.manager.enterMode.length, 1);
  assert.deepStrictEqual(calls.manager.enterMode[0].entries, registryEntries);
  assert.strictEqual(calls.manager.enterMode[0].workspaceDir, '/ws/project');
  assert.ok(
    executedCommands.some(
      e => e.cmd === 'workbench.action.closeAuxiliaryBar',
    ),
    'mode on closes the secondary sidebar',
  );

  fireDelta(false);
  assert.strictEqual(
    calls.manager.adoptRegistryTabs.length,
    1,
    'editor-tabs mode: a remote tab opens through the panel manager',
  );
  assert.deepStrictEqual(calls.manager.adoptRegistryTabs[0], {
    entries: remoteEntries,
    workspaceDir: '/ws/project',
    listed: remoteListed,
  });

  // A FIRST snapshot may duplicate this window's own chat tabs
  // (serialized placeholders the panel manager has not revived yet).
  // With no chat tab anywhere it is adopted wholesale; with one, only
  // the tabs the previous session's persisted panel ids do NOT cover
  // survive the filter — a remote tab registered before this window's
  // baseline snapshot still opens.
  fireDelta(true, [remoteEntries[0], priorEntry]);
  assert.strictEqual(
    calls.manager.adoptRegistryTabs.length,
    2,
    'first snapshot with no chat tabs anywhere: adopted wholesale',
  );
  assert.deepStrictEqual(calls.manager.adoptRegistryTabs[1].entries, [
    remoteEntries[0],
    priorEntry,
  ]);
  vscodeStub.window.tabGroups = {
    all: [{tabs: [{input: {viewType: 'mainThreadWebview-kissSorcar.chatTab'}}]}],
  };
  fireDelta(true, [remoteEntries[0], priorEntry]);
  assert.strictEqual(
    calls.manager.adoptRegistryTabs.length,
    3,
    'first snapshot with a serialized chat placeholder: filtered',
  );
  assert.deepStrictEqual(
    calls.manager.adoptRegistryTabs[2].entries,
    remoteEntries,
    "the previous session's own tab is filtered out, the remote one kept",
  );
  vscodeStub.window.tabGroups = {all: []};
  fakePanelCount = 1;
  fireDelta(true, [priorEntry]);
  assert.strictEqual(
    calls.manager.adoptRegistryTabs.length,
    4,
    'open chat panel: the filtered snapshot still syncs the manager',
  );
  assert.deepStrictEqual(
    calls.manager.adoptRegistryTabs[3].entries,
    [],
    'nothing beyond the previous session tabs: nothing to adopt',
  );
  // Every chat editor tab is a LIVE panel (tab count == panel count):
  // no placeholder can duplicate anything, so the first snapshot is
  // adopted wholesale and the manager dedupes against open panels.
  vscodeStub.window.tabGroups = {
    all: [{tabs: [{input: {viewType: 'mainThreadWebview-kissSorcar.chatTab'}}]}],
  };
  fireDelta(true, [remoteEntries[0], priorEntry]);
  assert.strictEqual(
    calls.manager.adoptRegistryTabs.length,
    5,
    'live-panels-only window: first snapshot adopted',
  );
  assert.deepStrictEqual(
    calls.manager.adoptRegistryTabs[4].entries,
    [remoteEntries[0], priorEntry],
    'live-panels-only window: nothing is filtered out',
  );
  vscodeStub.window.tabGroups = {all: []};
  fakePanelCount = 0;

  // --- the persisted panel-id record: per-tab add/remove deltas --------
  const record = calls.manager.recordPanelTab;
  assert.strictEqual(typeof record, 'function');
  record('x1', true);
  record('x2', true);
  record('x1', true); // idempotent
  record('x1', false);
  assert.deepStrictEqual(
    ctx.workspaceState.get('kissSorcar.editorPanelTabIds'),
    ['t-prior', 'x2'],
    'opens append (once), closes remove, other ids are untouched',
  );
  record('t-prior', false);
  record('x2', false);
  assert.deepStrictEqual(
    ctx.workspaceState.get('kissSorcar.editorPanelTabIds'),
    [],
    'closing every panel empties the record',
  );

  // --- editor-tabs mode routing ----------------------------------------
  await commands.get('kissSorcar.openSettings')();
  assert.strictEqual(calls.manager.openSettings, 1);
  assert.strictEqual(calls.sidebar.openSettingsUI, 1, 'sidebar untouched');

  // No chat panel open yet: Cmd+T opens a fresh editor tab directly.
  await commands.get('kissSorcar.newConversation')();
  assert.strictEqual(calls.manager.openNewChat, 1);
  assert.strictEqual(calls.controller.focusChatInput, 1);
  assert.strictEqual(calls.sidebar.newConversation, 1, 'sidebar untouched');
  assert.strictEqual(
    calls.controller.newConversation,
    0,
    'no panel to route through yet',
  );

  // With a panel open, Cmd+T routes through the ACTIVE panel's webview
  // (clearChat -> createNewTab posts openChatPanel with the composer
  // draft) so the drafted text is carried into the new editor tab.
  await commands.get('kissSorcar.newConversation')();
  assert.strictEqual(calls.controller.newConversation, 1);
  assert.strictEqual(calls.controller.focusChatInput, 2);
  assert.strictEqual(
    calls.manager.openNewChat,
    1,
    'the panel webview opens the new tab, not openNewChat',
  );
  assert.strictEqual(calls.sidebar.newConversation, 1, 'sidebar untouched');

  await commands.get('kissSorcar.openPanel')();
  assert.strictEqual(calls.manager.revealActiveOrCreate, 1);
  assert.strictEqual(calls.controller.focusChatInput, 3);

  await commands.get('kissSorcar.stopTask')();
  assert.strictEqual(calls.controller.stopTask, 1);
  assert.strictEqual(calls.sidebar.stopTask, 1, 'sidebar untouched');

  // The editor-title git-commit button: reveal (or open) a chat panel
  // and run its manual Git Commit through that panel's controller.
  await commands.get('kissSorcar.gitCommit')();
  assert.strictEqual(calls.controller.gitCommit, 1);
  assert.strictEqual(calls.manager.revealActiveOrCreate, 2);
  assert.strictEqual(calls.sidebar.gitCommit, 1, 'sidebar untouched');

  vscodeStub.window.activeTextEditor = {
    document: {getText: () => 'selected text'},
    selection: {},
  };
  await commands.get('kissSorcar.runSelection')();
  assert.deepStrictEqual(calls.controller.submitTask, ['selected text']);
  assert.deepStrictEqual(calls.sidebar.submitTask, []);

  await commands.get('kissSorcar.insertSelectionToChat')();
  await new Promise(r => setTimeout(r, 20));
  assert.deepStrictEqual(calls.controller.appendToInput, ['selected text']);
  assert.deepStrictEqual(calls.sidebar.appendToInput, []);

  // --- the KS button: showHistory in editor-tabs mode -------------------
  // A chat panel is open (newConversation above): only focus the view.
  executedCommands.length = 0;
  await commands.get('kissSorcar.showHistory')();
  assert.deepStrictEqual(
    executedCommands.map(e => e.cmd),
    ['kissSorcar.historyView.focus'],
    'showHistory focuses the primary-sidebar history view',
  );
  assert.strictEqual(
    calls.manager.openNewChat,
    1,
    'a chat tab is already open: no extra chat',
  );

  // With no chat tab open, the same click also opens a fresh chat.
  fakePanelCount = 0;
  executedCommands.length = 0;
  await commands.get('kissSorcar.showHistory')();
  assert.deepStrictEqual(executedCommands.map(e => e.cmd), [
    'kissSorcar.historyView.focus',
  ]);
  assert.strictEqual(calls.manager.openNewChat, 2, 'no chat tab: one opened');
  assert.strictEqual(calls.controller.focusChatInput, 4);

  // --- the history view: resolve + visibility both ensure a chat --------
  const historyProvider = viewProviders.get('kissSorcar.historyView');
  assert.ok(historyProvider, 'history view provider registered');
  const historyController = sidebarInstances.find(
    v => v.panelHooks && v.panelHooks.rootTabId === 'history-panel',
  );
  assert.ok(historyController, 'history controller owns the fixed root tab');
  assert.ok(
    historyController.panelHooks.bodyAttrs.includes('history-panel-mode'),
    'history controller carries the history-panel body attrs',
  );

  const visibilityListeners = [];
  const fakeView = {
    webview: {},
    visible: true,
    show: () => {},
    onDidChangeVisibility: cb => {
      visibilityListeners.push(cb);
      return makeDisposable();
    },
    onDidDispose: () => makeDisposable(),
  };
  fakePanelCount = 0;
  historyProvider.resolveWebviewView(fakeView, {}, {});
  assert.deepStrictEqual(
    historyController.resolvedViews,
    [fakeView],
    'provider delegates to the history controller',
  );
  assert.strictEqual(
    calls.manager.openNewChat,
    3,
    'resolving with no chat tab opens one',
  );

  // Visible again with a panel open: no duplicate chat.
  for (const cb of visibilityListeners) cb();
  assert.strictEqual(calls.manager.openNewChat, 3);

  // Visible again with none open: a fresh chat.
  fakePanelCount = 0;
  for (const cb of visibilityListeners) cb();
  assert.strictEqual(calls.manager.openNewChat, 4);

  // A hidden view must not open chats.
  fakePanelCount = 0;
  fakeView.visible = false;
  for (const cb of visibilityListeners) cb();
  assert.strictEqual(calls.manager.openNewChat, 4);
  fakeView.visible = true;
  fakePanelCount = 1;

  // A history click routes through the panel manager's openChat; other
  // panel events are not the history hook's business.
  historyController.panelHooks.onEvent({
    kind: 'openChat',
    chatId: 'c9',
    taskId: 7,
    title: 'resumed',
  });
  assert.deepStrictEqual(calls.manager.openChat, [
    {kind: 'openChat', chatId: 'c9', taskId: 7, title: 'resumed'},
  ]);
  historyController.panelHooks.onEvent({kind: 'title', title: 'ignored'});
  assert.strictEqual(calls.manager.openChat.length, 1);

  // --- flip the mode OFF: panels close, the secondary sidebar opens on
  // the KISS Sorcar chat view with its composer focused -----------------
  const focusBeforeModeOff = calls.sidebar.focusChatInput;
  executedCommands.length = 0;
  editorTabsMode = false;
  await fireConfigChange();
  assert.strictEqual(calls.manager.closeAll, 1);
  assert.strictEqual(
    calls.sidebar.focusChatInput,
    focusBeforeModeOff + 1,
    'mode off reveals the secondary-sidebar chat and focuses its composer',
  );
  assert.ok(
    !executedCommands.some(
      e => e.cmd === 'workbench.action.closeAuxiliaryBar',
    ),
    'mode off must NOT close the secondary sidebar it just revealed',
  );

  // The flip pops the dummy tree up in the still-open primary sidebar
  // (the history panel hides, the tree takes its spot): the handler
  // closes the primary sidebar but must not issue a second, competing
  // reveal of the secondary-sidebar chat — the config handler already
  // did that; this is a mode flip, not a KS click.
  executedCommands.length = 0;
  for (const cb of treeVisibilityListeners) cb({visible: true});
  await new Promise(r => setTimeout(r, 120));
  assert.deepStrictEqual(
    executedCommands.map(e => e.cmd),
    ['workbench.action.closeSidebar'],
    'tree shown by the mode flip: primary sidebar closes',
  );
  assert.strictEqual(
    calls.sidebar.focusChatInput,
    focusBeforeModeOff + 1,
    'tree shown by the mode flip: no second reveal of the sidebar chat',
  );

  // --- a session with NO persisted panel-id record (pre-upgrade
  // workspace state): while any chat tab exists the ids behind the
  // serialized placeholders are unknowable, so nothing of a first
  // snapshot is adopted (the manager is still called so the snapshot
  // syncs open panels' bindings); later snapshots adopt normally.
  editorTabsMode = true;
  registryTabsAddedListeners.length = 0;
  const ctx2 = {
    extensionUri: vscodeStub.Uri.file(tmpExtPath),
    extensionPath: tmpExtPath,
    subscriptions: [],
    workspaceState: makeMemento(),
    globalState: makeMemento(),
  };
  await ctx2.workspaceState.update('firstLaunchDone', true);
  await ctx2.workspaceState.update('sidebarWidened', true);
  // Mode ON, no chat editor tab anywhere: activation opens the one
  // chat tab the invariant demands, in the background.
  fakePanelCount = 0;
  vscodeStub.window.tabGroups = {all: []};
  const openBeforeActivate2 = calls.manager.openNewChat;
  const ensureBeforeActivate2 = calls.manager.ensureChatOpen.length;
  extension.activate(ctx2);
  assert.strictEqual(calls.manager.watchEditorTabs, 2, 'backstop re-wired');
  assert.deepStrictEqual(
    calls.manager.ensureChatOpen.slice(ensureBeforeActivate2),
    [true],
    'activation asks for a background chat',
  );
  assert.strictEqual(
    calls.manager.openNewChat,
    openBeforeActivate2 + 1,
    'mode on with no chat tab: activation opens one',
  );
  assert.strictEqual(fakePanelCount, 1);
  vscodeStub.window.tabGroups = {
    all: [{tabs: [{input: {viewType: 'mainThreadWebview-kissSorcar.chatTab'}}]}],
  };
  fakePanelCount = 0;
  const adoptCallsBefore = calls.manager.adoptRegistryTabs.length;
  fireDelta(true);
  assert.strictEqual(
    calls.manager.adoptRegistryTabs.length,
    adoptCallsBefore + 1,
    'no persisted record: the snapshot still reaches the manager',
  );
  assert.deepStrictEqual(
    calls.manager.adoptRegistryTabs[adoptCallsBefore].entries,
    [],
    'no persisted record: nothing of a first snapshot is adopted while ' +
      'a chat tab exists',
  );
  fireDelta(false);
  assert.strictEqual(
    calls.manager.adoptRegistryTabs.length,
    adoptCallsBefore + 2,
    'a later snapshot is adopted normally',
  );
  assert.deepStrictEqual(
    calls.manager.adoptRegistryTabs[adoptCallsBefore + 1].entries,
    remoteEntries,
  );
  // The recorder starts from an EMPTY record when none was persisted.
  calls.manager.recordPanelTab('y1', true);
  assert.deepStrictEqual(
    ctx2.workspaceState.get('kissSorcar.editorPanelTabIds'),
    ['y1'],
    'the first recorded open seeds the record',
  );
  vscodeStub.window.tabGroups = {all: []};
  for (const d of ctx2.subscriptions) {
    try {
      if (d && typeof d.dispose === 'function') d.dispose();
    } catch {
      /* fs watchers on tmp dirs */
    }
  }

  for (const d of ctx.subscriptions) {
    try {
      if (d && typeof d.dispose === 'function') d.dispose();
    } catch {
      /* fs watchers on tmp dirs */
    }
  }

  // --- the mode-OFF reveal must END with the chat composer focused,
  // even when the secondary sidebar has never been widened: a window
  // that started in editor-tabs mode resolves the sidebar view for the
  // first time on that reveal, which arms the one-time widening; its
  // focus handoff must go back to the chat, not the editor group.
  // The earlier activations' config listeners must not fire here (the
  // stub's disposables are no-ops), so start from clean listener sets.
  configListeners.length = 0;
  treeVisibilityListeners.length = 0;
  registryTabsAddedListeners.length = 0;
  firstResolveCb = null;
  editorTabsMode = true;
  const activateUnwidened = async () => {
    const c = {
      extensionUri: vscodeStub.Uri.file(tmpExtPath),
      extensionPath: tmpExtPath,
      subscriptions: [],
      workspaceState: makeMemento(),
      globalState: makeMemento(),
    };
    await c.workspaceState.update('firstLaunchDone', true);
    // `sidebarWidened` unset: the widening is armed on first resolve.
    fakePanelCount = 0;
    vscodeStub.window.tabGroups = {all: []};
    extension.activate(c);
    assert.ok(firstResolveCb, 'the widening hooks the first resolve');
    return c;
  };
  const disposeCtx = c => {
    for (const d of c.subscriptions) {
      try {
        if (d && typeof d.dispose === 'function') d.dispose();
      } catch {
        /* fs watchers on tmp dirs */
      }
    }
  };
  const runWidening = async () => {
    firstResolveCb();
    firstResolveCb = null;
    // The widening waits 500ms after the resolve, then runs its awaits.
    await new Promise(r => setTimeout(r, 700));
  };

  const ctx3 = await activateUnwidened();
  const focusBeforeOff3 = calls.sidebar.focusChatInput;
  const widenBefore3 = calls.sidebar.widenToOneThird;
  executedCommands.length = 0;
  editorTabsMode = false;
  await fireConfigChange();
  assert.strictEqual(
    calls.sidebar.focusChatInput,
    focusBeforeOff3 + 1,
    'mode off (never widened): the reveal focuses the chat',
  );
  // The reveal resolved the view: the widening runs, then hands focus
  // back to the chat composer instead of the editor group.
  await runWidening();
  assert.strictEqual(
    calls.sidebar.widenToOneThird,
    widenBefore3 + 1,
    'first resolve on the mode-off reveal: the sidebar is widened once',
  );
  assert.ok(
    executedCommands.some(e => e.cmd === 'workbench.action.focusAuxiliaryBar'),
    'the widening still focuses the secondary sidebar for its resizes',
  );
  assert.ok(
    !executedCommands.some(
      e => e.cmd === 'workbench.action.focusFirstEditorGroup',
    ),
    'widening after the mode-off reveal must NOT move focus to the editor',
  );
  assert.strictEqual(
    calls.sidebar.focusChatInput,
    focusBeforeOff3 + 2,
    'widening after the mode-off reveal refocuses the chat composer',
  );
  assert.strictEqual(
    ctx3.workspaceState.get('sidebarWidened'),
    true,
    'the widening is still recorded as done',
  );
  disposeCtx(ctx3);

  // Contrast: a first resolve NOT caused by a mode flip (the user opened
  // the sidebar chat in a window that was in sidebar mode all along)
  // keeps the widening's original handoff to the editor group.
  configListeners.length = 0;
  treeVisibilityListeners.length = 0;
  registryTabsAddedListeners.length = 0;
  editorTabsMode = false;
  const ctx4 = await activateUnwidened();
  const focusBefore4 = calls.sidebar.focusChatInput;
  executedCommands.length = 0;
  await runWidening();
  assert.ok(
    executedCommands.some(
      e => e.cmd === 'workbench.action.focusFirstEditorGroup',
    ),
    'widening without a mode flip hands focus to the editor group',
  );
  assert.strictEqual(
    calls.sidebar.focusChatInput,
    focusBefore4,
    'widening without a mode flip does not refocus the chat',
  );
  assert.strictEqual(ctx4.workspaceState.get('sidebarWidened'), true);
  disposeCtx(ctx4);

  fs.rmSync(tmpExtPath, {recursive: true, force: true});
  console.log('editorTabsExtensionWiring: all tests passed');
}

runTest()
  .then(() => process.exit(0))
  .catch(err => {
    console.error(err);
    process.exit(1);
  });
