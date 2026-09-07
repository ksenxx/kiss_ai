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
//    tabs into panels (enterMode) and OFF closes all panels and
//    refocuses the sidebar.
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

const vscodeStub = {
  window: {
    registerWebviewViewProvider: () => makeDisposable(),
    createTreeView: () => ({
      onDidChangeVisibility: () => makeDisposable(),
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
  },
  manager: {
    openNewChat: 0,
    revealActiveOrCreate: 0,
    openSettings: 0,
    enterMode: [],
    closeAll: 0,
  },
  controller: {
    focusChatInput: 0,
    stopTask: 0,
    submitTask: [],
    appendToInput: [],
  },
};

const registryEntries = [
  {tabId: 't1', chatId: 'c1', title: 'one', workDir: '/ws/project', scopeWorkDir: ''},
];

class FakeSidebarView {
  constructor() {
    this.hasFocus = false;
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
  getRegistryTabEntries() {
    return registryEntries;
  }
  onCommitMessage() {
    return makeDisposable();
  }
  generateCommitMessage() {
    return Promise.resolve();
  }
  onFirstResolve() {}
  widenToOneThird() {
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
}

const fakeController = new FakeController();

class FakePanelManager {
  static modeEnabled() {
    return editorTabsMode;
  }
  registerSerializer() {
    return makeDisposable();
  }
  openNewChat() {
    calls.manager.openNewChat += 1;
    return fakeController;
  }
  revealActiveOrCreate() {
    calls.manager.revealActiveOrCreate += 1;
    return fakeController;
  }
  activeController() {
    return fakeController;
  }
  openSettings() {
    calls.manager.openSettings += 1;
    return Promise.resolve();
  }
  enterMode(entries, workspaceDir) {
    calls.manager.enterMode.push({entries, workspaceDir});
  }
  closeAll() {
    calls.manager.closeAll += 1;
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

  extension.activate(ctx);
  assert.ok(commands.has('kissSorcar.openSettings'), 'openSettings command');

  // --- sidebar mode (default) -----------------------------------------
  await commands.get('kissSorcar.openSettings')();
  assert.strictEqual(calls.sidebar.openSettingsUI, 1);
  assert.strictEqual(calls.manager.openSettings, 0);

  await commands.get('kissSorcar.newConversation')();
  assert.strictEqual(calls.sidebar.newConversation, 1);
  assert.strictEqual(calls.manager.openNewChat, 0);

  await commands.get('kissSorcar.stopTask')();
  assert.strictEqual(calls.sidebar.stopTask, 1);

  // --- flip the mode ON: registry tabs migrate to panels ---------------
  editorTabsMode = true;
  await fireConfigChange();
  assert.strictEqual(calls.manager.enterMode.length, 1);
  assert.deepStrictEqual(calls.manager.enterMode[0].entries, registryEntries);
  assert.strictEqual(calls.manager.enterMode[0].workspaceDir, '/ws/project');

  // --- editor-tabs mode routing ----------------------------------------
  await commands.get('kissSorcar.openSettings')();
  assert.strictEqual(calls.manager.openSettings, 1);
  assert.strictEqual(calls.sidebar.openSettingsUI, 1, 'sidebar untouched');

  await commands.get('kissSorcar.newConversation')();
  assert.strictEqual(calls.manager.openNewChat, 1);
  assert.strictEqual(calls.controller.focusChatInput, 1);
  assert.strictEqual(calls.sidebar.newConversation, 1, 'sidebar untouched');

  await commands.get('kissSorcar.openPanel')();
  assert.strictEqual(calls.manager.revealActiveOrCreate, 1);
  assert.strictEqual(calls.controller.focusChatInput, 2);

  await commands.get('kissSorcar.stopTask')();
  assert.strictEqual(calls.controller.stopTask, 1);
  assert.strictEqual(calls.sidebar.stopTask, 1, 'sidebar untouched');

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

  // --- flip the mode OFF: panels close, sidebar comes back --------------
  editorTabsMode = false;
  await fireConfigChange();
  assert.strictEqual(calls.manager.closeAll, 1);
  assert.ok(
    calls.sidebar.focusChatInput >= 1,
    'mode off refocuses the sidebar chat',
  );

  for (const d of ctx.subscriptions) {
    try {
      if (d && typeof d.dispose === 'function') d.dispose();
    } catch {
      /* fs watchers on tmp dirs */
    }
  }
  fs.rmSync(tmpExtPath, {recursive: true, force: true});
  console.log('editorTabsExtensionWiring: all tests passed');
}

runTest()
  .then(() => process.exit(0))
  .catch(err => {
    console.error(err);
    process.exit(1);
  });
