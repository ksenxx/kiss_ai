// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end tests for moving the four Sorcar editor-title buttons
// (+ new chat, git commit, settings gear, KS history) from the
// top-right of the editor window into the window title bar above it.
//
// VS Code cannot place extension commands in the title bar directly;
// the extension instead drives the workbench setting
// `workbench.editor.editorActionsLocation` (src/editorActionsLocation
// .ts, wired in extension.ts). Covered behavior, activation + config
// flips through the real compiled extension host wiring:
//  - activation with editor-tabs mode ON sets the global value to
//    "titleBar" and remembers the user's prior global value;
//  - activation with the mode OFF leaves the setting alone even when
//    a restore record exists (another window may have the mode ON;
//    restores happen only on explicit mode flips);
//  - flipping the mode OFF restores the remembered value (undefined
//    when the user had none) and clears the memory;
//  - a user's own prior value (e.g. "hidden") round-trips;
//  - `window.customTitleBarVisibility` flipping to "never" while the
//    extension owns the title-bar value restores it (the title bar
//    would HIDE the actions, not fall back), and flipping back
//    re-applies it; the visibility listener is inert in sidebar mode;
//  - a user-picked "titleBar" is never recorded nor reverted;
//  - a user override made while the mode was on wins over the restore;
//  - back-to-back mode flips are serialized: with slow configuration
//    writes the LAST flip still determines the final state;
//  - host stubs without the configuration surface are a no-op.

'use strict';

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const Module = require('module');

const EXT_ROOT = path.join(__dirname, '..');
const OUT_DIR = path.join(EXT_ROOT, 'out');
const extensionPath = path.join(OUT_DIR, 'extension.js');
const EDITOR_ACTIONS = 'workbench.editor.editorActionsLocation';

assert.ok(
  fs.existsSync(extensionPath),
  `compiled extension missing: ${extensionPath} — run \`npm run compile\` first`,
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

// Mutable host-side state the stub configuration reads and writes.
const state = {
  editorTabsMode: true,
  customTitleBarVisibility: 'auto',
  editorActionsGlobal: undefined,
  // >0 makes configuration writes slow (and apply their effect only on
  // completion), for the serialization test.
  updateDelayMs: 0,
  // Makes the next configuration write reject, for the chain-recovery
  // test.
  failNextUpdate: false,
};
const updates = [];
const configListeners = [];

function makeConfig(section) {
  const full = key => (section ? `${section}.${key}` : key);
  return {
    get: key => {
      const k = full(key);
      if (k === 'kissSorcar.editorTabsMode') return state.editorTabsMode;
      if (k === 'window.customTitleBarVisibility') {
        return state.customTitleBarVisibility;
      }
      if (k === EDITOR_ACTIONS) return state.editorActionsGlobal;
      return undefined;
    },
    inspect: key => {
      if (full(key) === EDITOR_ACTIONS) {
        return {globalValue: state.editorActionsGlobal};
      }
      return undefined;
    },
    update: async (key, value, target) => {
      if (state.updateDelayMs > 0) {
        await new Promise(r => setTimeout(r, state.updateDelayMs));
      }
      if (state.failNextUpdate) {
        state.failNextUpdate = false;
        throw new Error('synthetic configuration write failure');
      }
      updates.push({key: full(key), value, target});
      if (full(key) === EDITOR_ACTIONS) state.editorActionsGlobal = value;
    },
  };
}

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
    registerCommand: () => makeDisposable(),
    executeCommand: () => Promise.resolve(),
  },
  workspace: {
    workspaceFolders: [{uri: {fsPath: '/ws/project'}}],
    asRelativePath: p => String(p),
    getConfiguration: section => makeConfig(section),
    onDidChangeConfiguration: cb => {
      configListeners.push(cb);
      return makeDisposable();
    },
  },
  ConfigurationTarget: {Global: 1, Workspace: 2, WorkspaceFolder: 3},
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

class FakeSidebarView {
  constructor() {
    this.hasFocus = false;
  }
  syncWorkDir() {}
  focusChatInput() {
    return Promise.resolve();
  }
  newConversation() {}
  stopTask() {}
  submitTask() {}
  appendToInput() {
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
  openSettingsUI() {
    return Promise.resolve();
  }
  closeChatTab() {}
  dispose() {}
}

class FakePanelManager {
  static modeEnabled() {
    return state.editorTabsMode;
  }
  constructor() {
    this.panelCount = 0;
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
  enterMode() {}
  closeAll() {}
  setMetaSink() {}
  markShutdown() {}
  adoptRegistryTabs() {}
  activeController() {
    return undefined;
  }
  revealActiveOrCreate() {
    return new FakeSidebarView();
  }
  openNewChat() {
    return new FakeSidebarView();
  }
  openChat() {}
  openSettings() {
    return Promise.resolve();
  }
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
  CHAT_PANEL_VIEW_TYPE: 'kissSorcar.chatTab',
  SorcarPanelManager: FakePanelManager,
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
  checkForExtensionUpdate: () =>
    Promise.resolve({checked: false, notified: false, reason: 'test'}),
  snoozeUpdateNotification: () => ({snoozeUntilMs: 0}),
});

delete require.cache[require.resolve(extensionPath)];
const extension = require(extensionPath);
const {PRIOR_LOCATION_KEY, syncEditorActionsLocation} = require(
  path.join(OUT_DIR, 'editorActionsLocation.js'),
);

function makeContext() {
  const tmpExtPath = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-ext-eal-'));
  return {
    extensionUri: vscodeStub.Uri.file(tmpExtPath),
    extensionPath: tmpExtPath,
    subscriptions: [],
    workspaceState: makeMemento(),
    globalState: makeMemento(),
    _tmpExtPath: tmpExtPath,
  };
}

function disposeContext(ctx) {
  for (const d of ctx.subscriptions) {
    try {
      if (d && typeof d.dispose === 'function') d.dispose();
    } catch {
    }
  }
  fs.rmSync(ctx._tmpExtPath, {recursive: true, force: true});
}

async function settle() {
  for (let i = 0; i < 8; i++) await new Promise(r => setTimeout(r, 10));
}

function fireModeChange(enabled) {
  state.editorTabsMode = enabled;
  for (const cb of configListeners.slice()) {
    cb({affectsConfiguration: s => s === 'kissSorcar.editorTabsMode'});
  }
}

function fireTitleBarVisibilityChange(visibility) {
  state.customTitleBarVisibility = visibility;
  for (const cb of configListeners.slice()) {
    cb({
      affectsConfiguration: s => s === 'window.customTitleBarVisibility',
    });
  }
}

function actionUpdates() {
  return updates.filter(u => u.key === EDITOR_ACTIONS);
}

async function runTest() {
  // --- activation with mode ON moves the actions to the title bar ---
  const ctx = makeContext();
  extension.activate(ctx);
  await settle();
  assert.deepStrictEqual(
    actionUpdates(),
    [{key: EDITOR_ACTIONS, value: 'titleBar', target: 1}],
    'activation in editor-tabs mode must set editorActionsLocation to titleBar globally',
  );
  assert.deepStrictEqual(
    ctx.globalState.get(PRIOR_LOCATION_KEY),
    {prior: null},
    'the absent prior user value must be remembered as null',
  );

  // A repeated ON sync while already "titleBar" must be a no-op that
  // PRESERVES the restore record.
  updates.length = 0;
  fireModeChange(true);
  await settle();
  assert.deepStrictEqual(actionUpdates(), [], 'already titleBar: no update');
  assert.deepStrictEqual(ctx.globalState.get(PRIOR_LOCATION_KEY), {
    prior: null,
  });

  // --- custom title bar disabled: the owned value is restored -------
  // (with "never" VS Code hides title-bar editor actions entirely)
  updates.length = 0;
  fireTitleBarVisibilityChange('never');
  await settle();
  assert.deepStrictEqual(
    actionUpdates(),
    [{key: EDITOR_ACTIONS, value: undefined, target: 1}],
    'visibility "never" must undo the extension-owned titleBar value',
  );
  assert.strictEqual(ctx.globalState.get(PRIOR_LOCATION_KEY), undefined);

  // Mode flipping ON while the custom title bar is off must not move.
  updates.length = 0;
  fireModeChange(false);
  fireModeChange(true);
  await settle();
  assert.deepStrictEqual(
    actionUpdates(),
    [],
    'no move while the custom title bar is disabled',
  );

  // --- custom title bar re-enabled: the move is re-applied ----------
  updates.length = 0;
  fireTitleBarVisibilityChange('auto');
  await settle();
  assert.deepStrictEqual(actionUpdates(), [
    {key: EDITOR_ACTIONS, value: 'titleBar', target: 1},
  ]);
  assert.deepStrictEqual(ctx.globalState.get(PRIOR_LOCATION_KEY), {
    prior: null,
  });

  // --- flipping the mode OFF restores the (absent) user value -------
  updates.length = 0;
  fireModeChange(false);
  await settle();
  assert.deepStrictEqual(
    actionUpdates(),
    [{key: EDITOR_ACTIONS, value: undefined, target: 1}],
    'mode OFF must remove the global override the extension made',
  );
  assert.strictEqual(
    ctx.globalState.get(PRIOR_LOCATION_KEY),
    undefined,
    'the restore record must be cleared after restoring',
  );

  // The visibility listener is inert in sidebar (mode OFF) state.
  state.editorActionsGlobal = 'titleBar'; // user's own choice
  updates.length = 0;
  fireTitleBarVisibilityChange('never');
  await settle();
  assert.deepStrictEqual(
    actionUpdates(),
    [],
    'visibility changes in sidebar mode must not touch the setting',
  );
  fireTitleBarVisibilityChange('auto');
  await settle();
  updates.length = 0;

  // --- a user-picked titleBar is not recorded when the mode turns ON
  fireModeChange(true);
  await settle();
  assert.deepStrictEqual(actionUpdates(), [], 'user already at titleBar');
  assert.strictEqual(
    ctx.globalState.get(PRIOR_LOCATION_KEY),
    undefined,
    'no record may be created for a value the extension did not set',
  );

  // --- mode OFF without a record leaves everything alone ------------
  updates.length = 0;
  fireModeChange(false);
  await settle();
  assert.deepStrictEqual(
    actionUpdates(),
    [],
    "no record: the user's own titleBar value must not be reverted",
  );

  // --- a prior user value ("hidden") round-trips ---------------------
  state.editorActionsGlobal = 'hidden';
  updates.length = 0;
  fireModeChange(true);
  await settle();
  assert.deepStrictEqual(actionUpdates(), [
    {key: EDITOR_ACTIONS, value: 'titleBar', target: 1},
  ]);
  assert.deepStrictEqual(ctx.globalState.get(PRIOR_LOCATION_KEY), {
    prior: 'hidden',
  });
  updates.length = 0;
  fireModeChange(false);
  await settle();
  assert.deepStrictEqual(
    actionUpdates(),
    [{key: EDITOR_ACTIONS, value: 'hidden', target: 1}],
    "mode OFF must restore the user's remembered value",
  );
  assert.strictEqual(ctx.globalState.get(PRIOR_LOCATION_KEY), undefined);

  // --- user override while the mode is on wins over the restore -----
  state.editorActionsGlobal = undefined;
  fireModeChange(true);
  await settle();
  assert.deepStrictEqual(ctx.globalState.get(PRIOR_LOCATION_KEY), {
    prior: null,
  });
  state.editorActionsGlobal = 'hidden'; // user changed it themselves
  updates.length = 0;
  fireModeChange(false);
  await settle();
  assert.deepStrictEqual(
    actionUpdates(),
    [],
    'a user override made meanwhile must not be clobbered by the restore',
  );
  assert.strictEqual(
    ctx.globalState.get(PRIOR_LOCATION_KEY),
    undefined,
    'the stale record must still be cleared',
  );

  // --- back-to-back flips are serialized; the LAST flip wins --------
  state.editorActionsGlobal = undefined;
  state.updateDelayMs = 25; // configuration writes now finish late
  updates.length = 0;
  fireModeChange(true);
  fireModeChange(false); // queued before the ON sync's writes finish
  await settle();
  state.updateDelayMs = 0;
  assert.deepStrictEqual(
    actionUpdates(),
    [
      {key: EDITOR_ACTIONS, value: 'titleBar', target: 1},
      {key: EDITOR_ACTIONS, value: undefined, target: 1},
    ],
    'the two slow syncs must apply strictly in arrival order',
  );
  assert.strictEqual(
    state.editorActionsGlobal,
    undefined,
    'after ON immediately followed by OFF the setting must be restored',
  );
  assert.strictEqual(ctx.globalState.get(PRIOR_LOCATION_KEY), undefined);

  // --- a rejected write is caught and the chain recovers ------------
  const rejections = [];
  const onRejection = reason => rejections.push(reason);
  process.on('unhandledRejection', onRejection);
  state.failNextUpdate = true;
  updates.length = 0;
  fireModeChange(true); // this sync's configuration write rejects
  await settle();
  assert.strictEqual(
    state.editorActionsGlobal,
    undefined,
    'the failed write must leave the setting untouched',
  );
  fireModeChange(true); // the chain must still be alive and retry
  await settle();
  process.removeListener('unhandledRejection', onRejection);
  assert.deepStrictEqual(rejections, [], 'no unhandled rejection may escape');
  assert.deepStrictEqual(
    actionUpdates(),
    [{key: EDITOR_ACTIONS, value: 'titleBar', target: 1}],
    'the sync after a failed write must succeed',
  );
  assert.deepStrictEqual(ctx.globalState.get(PRIOR_LOCATION_KEY), {
    prior: null,
  });
  fireModeChange(false); // leave the shared state clean for later phases
  await settle();

  extension.deactivate();
  disposeContext(ctx);
  console.log('editorActionsTitleBar.test.js: extension wiring tests passed');

  // --- activation with mode OFF must not restore --------------------
  // (the record may belong to another window whose mode is still ON;
  // restores happen only on explicit flips)
  configListeners.length = 0;
  state.editorTabsMode = false;
  state.editorActionsGlobal = 'titleBar';
  const ctx2 = makeContext();
  await ctx2.globalState.update(PRIOR_LOCATION_KEY, {prior: null});
  updates.length = 0;
  extension.activate(ctx2);
  await settle();
  assert.deepStrictEqual(
    actionUpdates(),
    [],
    'activation in sidebar mode must leave the setting and record alone',
  );
  assert.deepStrictEqual(ctx2.globalState.get(PRIOR_LOCATION_KEY), {
    prior: null,
  });
  extension.deactivate();
  disposeContext(ctx2);
  state.editorActionsGlobal = undefined;
  console.log('editorActionsTitleBar.test.js: activation-OFF tests passed');

  // --- guard branches: hosts without the configuration surface ------
  // (Reachable only under a host stub by definition; exercised through
  // the real module against the same stubbed host as everything else.)
  const guardCtx = makeContext();
  updates.length = 0;
  const cfgFn = vscodeStub.workspace.getConfiguration;
  vscodeStub.workspace.getConfiguration = undefined;
  await syncEditorActionsLocation(guardCtx, true);
  vscodeStub.workspace.getConfiguration = () => ({get: () => undefined});
  await syncEditorActionsLocation(guardCtx, true);
  vscodeStub.workspace.getConfiguration = () => ({
    get: () => undefined,
    inspect: () => undefined, // update absent
  });
  await syncEditorActionsLocation(guardCtx, true);
  vscodeStub.workspace.getConfiguration = cfgFn;
  assert.deepStrictEqual(
    actionUpdates(),
    [],
    'hosts without getConfiguration/inspect/update must be a no-op',
  );
  assert.strictEqual(guardCtx.globalState.get(PRIOR_LOCATION_KEY), undefined);
  disposeContext(guardCtx);
  console.log('editorActionsTitleBar.test.js: all assertions passed');
}

runTest().then(
  () => {
    Module._load = origLoad;
    process.exit(0);
  },
  err => {
    console.error('FAIL:', err && err.stack ? err.stack : err);
    process.exit(1);
  },
);
