// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end regression test for review-vscode.md #4: the KS activity-bar
// tree's `visible: true` continuation in extension.ts awaited
// workbench.action.closeSidebar plus a 50 ms settle, then focused the
// Sorcar chat WITHOUT re-checking anything:
//
//   - a user action that flipped the tree's visibility again during the
//     settle window (hiding it, switching primary-sidebar views) was
//     overridden by the stale continuation, which stole focus back;
//   - deactivate() during either await cleared `sidebarView`, and the
//     continuation's `sidebarView!.focusChatInput()` then threw a
//     TypeError as an unhandled rejection.
//
// The fix snapshots a visibility generation after the close and bails if
// a newer flip arrived or the extension was deactivated.
//
// Same scaffolding as editorTabsExtensionWiring.test.js: the real
// compiled out/extension.js is activated against a stubbed vscode module,
// with the sidebar view / panel manager replaced by instrumented fakes.

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

const commands = new Map();
const executedCommands = [];
const treeVisibilityListeners = [];

const vscodeStub = {
  window: {
    registerWebviewViewProvider: () => makeDisposable(),
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
    tabGroups: {all: []},
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
      get: () => '',
      update: () => Promise.resolve(),
    }),
    onDidChangeConfiguration: () => makeDisposable(),
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

const calls = {focusChatInput: 0};

class FakeSidebarView {
  constructor() {
    this.hasFocus = false;
  }
  resolveWebviewView() {}
  syncWorkDir() {}
  focusChatInput() {
    calls.focusChatInput += 1;
    return Promise.resolve();
  }
  newConversation() {}
  stopTask() {}
  submitTask() {
    return Promise.resolve();
  }
  appendToInput() {
    return Promise.resolve();
  }
  openSettingsUI() {
    return Promise.resolve();
  }
  gitCommit() {
    return Promise.resolve();
  }
  getRegistryTabEntries() {
    return [];
  }
  onRegistryTabsState() {
    return makeDisposable();
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

class FakePanelManager {
  static modeEnabled() {
    return false;
  }
  get panelCount() {
    return 0;
  }
  registerSerializer() {
    return makeDisposable();
  }
  watchEditorTabs() {
    return makeDisposable();
  }
  ensureChatOpen() {
    return undefined;
  }
  activeController() {
    return undefined;
  }
  openChat() {}
  adoptRegistryTabs() {}
  enterMode() {}
  closeAll() {}
  setMetaSink() {}
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
  historyPanelBodyAttrs: () => '',
  META_PANEL_TAB_ID: 'meta-panel',
  metaPanelBodyAttrs: () => '',
});

delete require.cache[require.resolve(extensionPath)];
const extension = require(extensionPath);

const rejections = [];
process.on('unhandledRejection', err => {
  rejections.push(err);
});

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

function fireVisibility(visible) {
  for (const cb of treeVisibilityListeners.slice()) cb({visible});
}

async function runTest() {
  const tmpExtPath = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-treevis-'));
  const ctx = {
    extensionUri: vscodeStub.Uri.file(tmpExtPath),
    extensionPath: tmpExtPath,
    subscriptions: [],
    workspaceState: makeMemento(),
    globalState: makeMemento(),
  };
  await ctx.workspaceState.update('firstLaunchDone', true);
  await ctx.workspaceState.update('sidebarWidened', true);

  extension.activate(ctx);
  assert.strictEqual(treeVisibilityListeners.length, 1, 'tree handler');

  // --- happy path: the KS button click still reveals the chat ----------
  executedCommands.length = 0;
  fireVisibility(true);
  await sleep(150);
  assert.deepStrictEqual(
    executedCommands.map(e => e.cmd),
    ['workbench.action.closeSidebar'],
    'the primary sidebar closes',
  );
  assert.strictEqual(
    calls.focusChatInput,
    1,
    'the KS button click focuses the chat',
  );

  // --- a newer user action during the settle window wins ---------------
  // The user re-opens the tree and hides it again (switches to Explorer)
  // while the first continuation is still parked in its 50 ms wait: the
  // stale continuation must NOT steal focus back.
  executedCommands.length = 0;
  fireVisibility(true);
  await sleep(10); // inside the settle window
  fireVisibility(false); // the user moved on
  await sleep(150);
  assert.strictEqual(
    calls.focusChatInput,
    1,
    'a stale visible:true continuation focused the chat after the user ' +
      'had already hidden the tree again',
  );

  // A follow-up click still works (the guard is per continuation).
  fireVisibility(true);
  await sleep(150);
  assert.strictEqual(calls.focusChatInput, 2, 'the next click focuses');

  // --- deactivation during the waits must not crash or focus -----------
  fireVisibility(true);
  await sleep(10); // continuation parked in the settle wait
  extension.deactivate();
  await sleep(150);
  assert.strictEqual(
    calls.focusChatInput,
    2,
    'a continuation must not focus a disposed sidebar surface',
  );
  assert.deepStrictEqual(
    rejections.map(e => String(e && e.message)),
    [],
    'deactivation during the waits raised an unhandled rejection',
  );

  console.log('audit0911_ext_tree_visibility_stale: OK');
}

runTest()
  .then(() => process.exit(0))
  .catch(err => {
    console.error(err && err.stack ? err.stack : err);
    process.exit(1);
  });
