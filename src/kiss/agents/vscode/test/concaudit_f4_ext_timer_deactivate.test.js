// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end regression test for the two activation timers in
// extension.ts (the one-time sidebar widening and the first-launch chat
// open).  Both checked `sidebarView` only BEFORE awaiting a workbench
// command; clearing the timer subscription cannot cancel a callback
// that has already started, so deactivate() during the await cleared
// the module globals and the continuation then:
//
//   - widen timer: called `sidebarView.widenToOneThird()` on undefined
//     -> TypeError, unhandled rejection;
//   - first-launch timer: `chatController(true)!` dereferenced the
//     cleared `sidebarView!` / `panelManager!` -> TypeError, unhandled
//     rejection, and (had it survived) wrote workspace state after
//     teardown.
//
// The fix captures the controller before the await, re-checks it after
// every await and catches the callback body.
//
// Same scaffolding as audit0911_ext_tree_visibility_stale.test.js: the
// real compiled out/extension.js is activated against a stubbed vscode
// module (there is no extension host outside VS Code), with the sidebar
// view / panel manager replaced by instrumented stand-ins so the test
// can observe what the continuation touches after deactivation.  The
// interleaving is forced for real: the stubbed workbench command
// returns a promise the test holds open until deactivate() has run.

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

// Keep the update-marker watcher and any other ~/.kiss lookups off the
// developer's real state.
const tmpKissHome = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-exttimer-'));
process.env.KISS_HOME = tmpKissHome;

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

// Workbench commands the test holds open: cmd -> resolve function.
const gates = new Map();
const executedCommands = [];

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
    tabGroups: {all: []},
  },
  commands: {
    registerCommand: () => makeDisposable(),
    executeCommand: (cmd, ...args) => {
      executedCommands.push({cmd, args});
      if (gates.has(cmd)) {
        return new Promise(resolve => {
          gates.get(cmd).push(resolve);
        });
      }
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

const calls = {focusChatInput: 0, widenToOneThird: 0};
let firstResolveCb = null;

class FakeSidebarView {
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
  onFirstResolve(cb) {
    firstResolveCb = cb;
  }
  widenToOneThird() {
    calls.widenToOneThird += 1;
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
  revealActiveOrCreate() {
    return undefined;
  }
  openChat() {}
  adoptRegistryTabs() {}
  enterMode() {}
  closeAll() {}
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

async function waitForCommand(cmd, ms) {
  const deadline = Date.now() + ms;
  while (Date.now() < deadline) {
    if (executedCommands.some(e => e.cmd === cmd)) return true;
    await sleep(10);
  }
  return false;
}

function makeContext() {
  const tmpExtPath = fs.mkdtempSync(
    path.join(os.tmpdir(), 'kiss-exttimer-ext-'),
  );
  return {
    extensionUri: vscodeStub.Uri.file(tmpExtPath),
    extensionPath: tmpExtPath,
    subscriptions: [],
    workspaceState: makeMemento(),
    globalState: makeMemento(),
  };
}

function deactivateViaContext(ctx) {
  // What the host does on deactivation: dispose the subscriptions (the
  // timer handles among them) and call deactivate().
  for (const d of ctx.subscriptions) {
    try {
      d.dispose();
    } catch {}
  }
  extension.deactivate();
}

// --- Scenario A: widen timer, deactivated while focusAuxiliaryBar is
// pending ------------------------------------------------------------
async function testWidenTimer() {
  executedCommands.length = 0;
  gates.set('workbench.action.focusAuxiliaryBar', []);
  const ctx = makeContext();
  await ctx.workspaceState.update('firstLaunchDone', true);
  // `sidebarWidened` unset: the widen timer is armed on first resolve.
  extension.activate(ctx);
  assert.ok(firstResolveCb, 'the widen timer hooks the first resolve');
  firstResolveCb();
  firstResolveCb = null;

  assert.ok(
    await waitForCommand('workbench.action.focusAuxiliaryBar', 3000),
    'the widen timer never fired',
  );
  // The continuation is parked on the gated command: deactivate now.
  deactivateViaContext(ctx);
  for (const release of gates.get('workbench.action.focusAuxiliaryBar')) {
    release();
  }
  gates.delete('workbench.action.focusAuxiliaryBar');
  await sleep(100);

  assert.strictEqual(
    calls.widenToOneThird,
    0,
    'the widen continuation acted on a disposed sidebar view',
  );
  assert.strictEqual(
    executedCommands.filter(
      e => e.cmd === 'workbench.action.focusFirstEditorGroup',
    ).length,
    0,
    'the widen continuation kept driving the workbench after deactivation',
  );
  assert.strictEqual(
    ctx.workspaceState.get('sidebarWidened'),
    undefined,
    'workspace state was written after teardown',
  );
  console.log('  ok - widen timer bails cleanly when deactivated mid-await');
}

// --- Scenario B: first-launch timer, deactivated while
// closeAuxiliaryBar is pending ---------------------------------------
async function testFirstLaunchTimer() {
  executedCommands.length = 0;
  gates.set('workbench.action.closeAuxiliaryBar', []);
  const ctx = makeContext();
  await ctx.workspaceState.update('sidebarWidened', true);
  // `firstLaunchDone` unset: a genuine first launch arms the auto-open.
  extension.activate(ctx);

  assert.ok(
    await waitForCommand('workbench.action.closeAuxiliaryBar', 3000),
    'the first-launch timer never fired',
  );
  deactivateViaContext(ctx);
  for (const release of gates.get('workbench.action.closeAuxiliaryBar')) {
    release();
  }
  gates.delete('workbench.action.closeAuxiliaryBar');
  await sleep(100);

  assert.strictEqual(
    calls.focusChatInput,
    0,
    'the first-launch continuation focused a chat surface after deactivation',
  );
  assert.strictEqual(
    ctx.workspaceState.get('firstLaunchDone'),
    undefined,
    'firstLaunchDone was recorded after teardown',
  );
  console.log(
    '  ok - first-launch timer bails cleanly when deactivated mid-await',
  );
}

// --- Scenario C: the timers still do their job when NOT interrupted ---
async function testHappyPath() {
  executedCommands.length = 0;
  calls.focusChatInput = 0;
  calls.widenToOneThird = 0;
  const ctx = makeContext();
  extension.activate(ctx);
  assert.ok(firstResolveCb, 'widen timer armed');
  firstResolveCb();
  firstResolveCb = null;
  await sleep(1500);
  assert.strictEqual(calls.widenToOneThird, 1, 'the sidebar was widened');
  assert.strictEqual(ctx.workspaceState.get('sidebarWidened'), true);
  assert.strictEqual(calls.focusChatInput, 1, 'the chat was opened');
  assert.strictEqual(ctx.workspaceState.get('firstLaunchDone'), true);
  deactivateViaContext(ctx);
  console.log('  ok - both timers still complete when left alone');
}

async function runTest() {
  await testWidenTimer();
  await testFirstLaunchTimer();
  await testHappyPath();
  await sleep(50);
  assert.deepStrictEqual(
    rejections.map(e => String(e && e.message)),
    [],
    'deactivation during the timer awaits raised an unhandled rejection',
  );
  console.log('concaudit_f4_ext_timer_deactivate: OK');
}

runTest()
  .then(() => {
    fs.rmSync(tmpKissHome, {recursive: true, force: true});
    process.exit(0);
  })
  .catch(err => {
    fs.rmSync(tmpKissHome, {recursive: true, force: true});
    console.error(err && err.stack ? err.stack : err);
    process.exit(1);
  });
