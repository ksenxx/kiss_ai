// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// Shared harness for the ui_antipattern_*.test.js host tests: a vscode
// API stub with a live configuration map, a recording notification
// relay (each toast is resolved by the TEST, so a test can click
// 'Skip this version' on one toast and dismiss the next), recorded
// executeCommand calls and the registered command handlers, plus the
// sibling out/ modules the compiled extension needs but the test under
// question does not exercise (sidebar view, git API, reload guard).
//
// Not a test itself (no .test.js suffix): run-all.js skips it.

const fs = require('fs');
const os = require('os');
const path = require('path');
const Module = require('module');

const EXT_ROOT = path.join(__dirname, '..');
const OUT_DIR = path.join(EXT_ROOT, 'out');

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

function stubModule(filePath, exports) {
  const fakeMod = new Module(filePath);
  fakeMod.filename = filePath;
  fakeMod.loaded = true;
  fakeMod.exports = exports;
  require.cache[filePath] = fakeMod;
}

class FakeSidebarView {
  constructor() {
    this.hasFocus = false;
    this.runUpdates = 0;
    this.updateWhenIdles = 0;
  }
  postMetaState() {}
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
  runUpdate() {
    this.runUpdates += 1;
  }
  updateWhenIdle() {
    this.updateWhenIdles += 1;
  }
  dispose() {}
}

/**
 * Build the vscode stub and load the compiled extension with the given
 * sibling-module overrides.
 *
 * @param opts.config flat map of `section.key` -> value backing
 *     getConfiguration(); tests mutate it between activations.
 * @param opts.modules extra {relativeOutFile: exports} stubs applied
 *     before the extension loads (e.g. 'UpdateChecker.js').
 * @param opts.notifyThrough when true the real WebviewNotifications
 *     module is used (the tests then set the poster); otherwise
 *     notifications are recorded in `notifications` and resolved by
 *     the test through their `resolve` function.
 */
function loadExtension(opts) {
  const o = opts || {};
  const config = o.config || {};
  const configUpdates = [];
  const executedCommands = [];
  const registeredCommands = new Map();
  const notifications = [];
  const openedDocuments = [];
  const shownDocuments = [];

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
      showTextDocument: doc => {
        shownDocuments.push(doc);
        return Promise.resolve({document: doc});
      },
      activeTextEditor: undefined,
    },
    commands: {
      registerCommand: (id, handler) => {
        registeredCommands.set(id, handler);
        return makeDisposable();
      },
      executeCommand: (cmd, ...args) => {
        executedCommands.push({cmd, args});
        return Promise.resolve();
      },
    },
    workspace: {
      asRelativePath: p => String(p),
      workspaceFolders: undefined,
      getConfiguration: section => ({
        get: (key, def) => {
          const full = section ? `${section}.${key}` : key;
          return full in config ? config[full] : def;
        },
        inspect: key => {
          const full = section ? `${section}.${key}` : key;
          return {globalValue: config[full]};
        },
        update: (key, value, target) => {
          const full = section ? `${section}.${key}` : key;
          configUpdates.push({key: full, value, target});
          if (value === undefined) delete config[full];
          else config[full] = value;
          return Promise.resolve();
        },
      }),
      onDidChangeConfiguration: () => makeDisposable(),
      openTextDocument: arg => {
        openedDocuments.push(arg);
        return Promise.resolve({uri: arg && arg.fsPath ? arg : undefined, arg});
      },
    },
    Uri: {
      file: p => ({fsPath: p, scheme: 'file', toString: () => `file://${p}`}),
      joinPath: (base, ...parts) =>
        vscodeStub.Uri.file(path.join(base.fsPath, ...parts)),
    },
    ConfigurationTarget: {Global: 1, Workspace: 2},
    ProgressLocation: {Notification: 15},
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

  stubModule(path.join(OUT_DIR, 'SorcarSidebarView.js'), {
    SorcarSidebarView: FakeSidebarView,
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

  function record(kind) {
    return (message, ...items) => {
      const actions = items.filter(i => typeof i === 'string');
      return new Promise(resolve => {
        notifications.push({kind, message, actions, resolve});
      });
    };
  }
  if (!o.notifyThrough) {
    stubModule(path.join(OUT_DIR, 'WebviewNotifications.js'), {
      setWebviewNotificationPoster: () => {},
      clearWebviewNotificationPoster: () => {},
      resolveWebviewNotificationAction: () => {},
      showInformationNotification: record('info'),
      showWarningNotification: record('warning'),
      showErrorNotification: record('error'),
      withWebviewNotificationProgress: (_opts, task) =>
        Promise.resolve(
          task(
            {report: () => {}},
            {
              isCancellationRequested: false,
              onCancellationRequested: () => makeDisposable(),
            },
          ),
        ),
    });
  }
  for (const [file, exports] of Object.entries(o.modules || {})) {
    stubModule(path.join(OUT_DIR, file), exports);
  }

  const extensionPath = path.join(OUT_DIR, 'extension.js');
  if (!fs.existsSync(extensionPath)) {
    throw new Error(
      `compiled extension missing: ${extensionPath} — run \`npm run compile\` first`,
    );
  }
  delete require.cache[require.resolve(extensionPath)];
  const extension = require(extensionPath);

  function makeContext() {
    const tmpExtPath = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-ext-uap-'));
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
      } catch {}
    }
    fs.rmSync(ctx._tmpExtPath, {recursive: true, force: true});
  }

  return {
    vscodeStub,
    extension,
    config,
    configUpdates,
    executedCommands,
    registeredCommands,
    notifications,
    openedDocuments,
    shownDocuments,
    makeContext,
    disposeContext,
  };
}

async function waitFor(predicate, message, tries = 150) {
  for (let i = 0; i < tries; i++) {
    if (predicate()) return;
    await new Promise(r => setTimeout(r, 20));
  }
  throw new Error(message || 'waitFor timed out');
}

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

module.exports = {
  OUT_DIR,
  loadExtension,
  makeMemento,
  stubModule,
  waitFor,
  sleep,
};
