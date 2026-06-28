// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end regression test for the activation-time update notification.
//
// Requirement: clicking the update button in an update notification must take
// the same action as clicking the Update button in the settings panel.  The
// settings-panel button posts `{type: 'runUpdate'}` and SorcarSidebarView runs
// the updater by locating `~/kiss_ai/install.sh`, showing an install
// notification, opening the `KISS Sorcar Update` terminal, and sending the
// installer command.  The activation-time notification used to handle its
// `Update now` action by opening a blank terminal via
// `workbench.action.terminal.new`, which did not run the updater at all.
//
// This test drives the real compiled `out/extension.js` activate() end-to-end
// with only VS Code and side-effectful dependencies stubbed.  It forces
// `UpdateChecker.checkForExtensionUpdate()` to report an available update,
// forces the notification promise to resolve as if the user clicked
// `Update now`, and asserts that the same SorcarSidebarView updater entrypoint
// used by the settings-panel `runUpdate` path is invoked, with no blank
// terminal command executed.

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

const executedCommands = [];
const notifications = [];
const calls = {
  runUpdate: 0,
  updateChecks: 0,
};

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
    executeCommand: (cmd, ...args) => {
      executedCommands.push({cmd, args});
      return Promise.resolve();
    },
  },
  workspace: {
    asRelativePath: p => String(p),
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
  onCommitMessage() {
    return makeDisposable();
  }
  generateCommitMessage() {
    return Promise.resolve();
  }
  handleMergeCommand() {}
  onFirstResolve() {}
  widenToOneThird() {
    return Promise.resolve();
  }
  runUpdate() {
    calls.runUpdate += 1;
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
  MERGE_ACTIONS: {accept: 'mergeAccept'},
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
  showInformationNotification: (message, ...actions) => {
    notifications.push({kind: 'info', message, actions});
    return Promise.resolve('Update now');
  },
  showWarningNotification: (message, ...actions) => {
    notifications.push({kind: 'warning', message, actions});
    return Promise.resolve(undefined);
  },
  showErrorNotification: (message, ...actions) => {
    notifications.push({kind: 'error', message, actions});
    return Promise.resolve(undefined);
  },
});
stubModule(path.join(OUT_DIR, 'UpdateChecker.js'), {
  checkForExtensionUpdate: async opts => {
    calls.updateChecks += 1;
    opts.notify({latest: '2099.1.1', current: '2026.6.31'});
    return {
      checked: true,
      notified: true,
      latest: '2099.1.1',
      current: '2026.6.31',
      reason: 'update-available',
    };
  },
});

delete require.cache[require.resolve(extensionPath)];
const extension = require(extensionPath);

function makeContext() {
  const tmpExtPath = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-ext-upd-'));
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
      // ignore cleanup failures
    }
  }
  fs.rmSync(ctx._tmpExtPath, {recursive: true, force: true});
}

async function waitFor(predicate, message) {
  for (let i = 0; i < 100; i++) {
    if (predicate()) return;
    await new Promise(r => setTimeout(r, 20));
  }
  throw new Error(message || 'waitFor timed out');
}

async function runTest() {
  const ctx = makeContext();
  try {
    extension.activate(ctx);
    await waitFor(
      () => calls.updateChecks === 1 && notifications.length === 1,
      'activation must check for updates and show the update notification',
    );
    await waitFor(
      () => calls.runUpdate === 1,
      'clicking Update now in the update notification must invoke the settings-panel updater path',
    );

    assert.strictEqual(calls.updateChecks, 1);
    assert.strictEqual(calls.runUpdate, 1);
    assert.strictEqual(notifications[0].kind, 'info');
    assert.deepStrictEqual(notifications[0].actions, ['Update now']);
    assert.ok(
      notifications[0].message.includes('2099.1.1'),
      'notification must mention the available version',
    );
    assert.deepStrictEqual(
      executedCommands.filter(c => c.cmd === 'workbench.action.terminal.new'),
      [],
      'update notification action must not open a blank terminal instead of running the updater',
    );
    console.log('\nAll activation update notification tests passed');
  } finally {
    extension.deactivate();
    disposeContext(ctx);
    Module._load = origLoad;
  }
}

runTest().then(
  () => process.exit(0),
  err => {
    console.error('FAIL:', err && err.stack ? err.stack : err);
    Module._load = origLoad;
    process.exit(1);
  },
);
