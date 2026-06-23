// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// Integration test for the per-workspace secondary-sidebar bootstrap.
//
// REQUIREMENT (the bug): when a NEW workspace is opened in VS Code,
// the secondary sidebar must
//   (1) auto-widen to approximately one-third of the VS Code window,
//   (2) auto-focus the KISS Sorcar tab (i.e. select it in the
//       secondary panel and put the cursor in the chat input).
//
// Before the fix, both behaviours were gated on
// ``context.globalState`` flags (``sidebarWidened`` and
// ``firstLaunchDone``).  Those flags are PER-INSTALL, not
// per-workspace, so any second or later workspace opened on the same
// machine inherited the flags from the first workspace and the
// activate() handler skipped both the widen call AND the focusChat
// call.  The user saw a narrow secondary sidebar with the KISS Sorcar
// tab unselected on every fresh workspace.
//
// The fix: gate the same logic on ``context.workspaceState`` so each
// workspace gets the one-shot bootstrap exactly once.
//
// This test drives the REAL compiled ``out/extension.js`` activate()
// with stubs for the dependent modules (``SorcarSidebarView``,
// ``DependencyInstaller``, ``gitApi``, ``reloadGuard``) and asserts
// that widening + focusChatInput fire on EVERY new workspace, not
// just the first one on the machine.
//
// Run directly with ``node``:
//
//     node src/kiss/agents/vscode/test/secondarySidebarPerWorkspace.test.js

'use strict';

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const Module = require('module');

const EXT_ROOT = path.join(__dirname, '..');
const OUT_DIR = path.join(EXT_ROOT, 'out');

assert.ok(
  fs.existsSync(path.join(OUT_DIR, 'extension.js')),
  `compiled extension missing: ${OUT_DIR}/extension.js — run \`npm run compile\` first`,
);

// ---- 1. vscode module stub ------------------------------------------
// Captures executeCommand invocations so we can assert that the
// secondary-sidebar focus command runs once per workspace, and exposes
// minimal versions of every API extension.ts touches in activate().

const executedCommands = [];

function makeDisposable() {
  return {dispose: () => {}};
}

const vscodeStub = {
  window: {
    registerWebviewViewProvider: () => makeDisposable(),
    createTreeView: () => ({
      onDidChangeVisibility: () => makeDisposable(),
      dispose: () => {},
    }),
    showInformationMessage: () => {},
    showErrorMessage: () => {},
    showWarningMessage: () => {},
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

// Redirect every ``require('vscode')`` to a tiny on-disk shim that
// returns ``global.__kissVscodeStub`` (same pattern as the other
// bughunt tests in this directory).
const stubPath = path.join(__dirname, '_vscode-stub-secondary.js');
fs.writeFileSync(
  stubPath,
  `'use strict';\nmodule.exports = global.__kissVscodeStub || {};\n`,
);
global.__kissVscodeStub = vscodeStub;
const origResolve = Module._resolveFilename;
Module._resolveFilename = function (request, parent, ...rest) {
  if (request === 'vscode') return require.resolve(stubPath);
  return origResolve.call(this, request, parent, ...rest);
};

// ---- 2. Stub SorcarSidebarView, DependencyInstaller, gitApi, reloadGuard
// Pre-populate require.cache so the compiled extension picks up these
// fakes instead of the real modules (which would pull jsdom, vscode,
// the daemon socket, etc.).

const calls = {
  widenToOneThird: 0,
  focusChatInput: 0,
  onFirstResolveCount: 0,
  syncWorkDir: 0,
};

class FakeSidebarView {
  constructor() {
    this.hasFocus = false;
    this._onCommitListeners = [];
    this._firstResolveCb = undefined;
  }
  syncWorkDir() {
    calls.syncWorkDir += 1;
  }
  focusChatInput() {
    calls.focusChatInput += 1;
    return Promise.resolve();
  }
  newConversation() {}
  stopTask() {}
  submitTask() {}
  appendToInput() {
    return Promise.resolve();
  }
  onCommitMessage(cb) {
    this._onCommitListeners.push(cb);
    return makeDisposable();
  }
  generateCommitMessage() {
    return Promise.resolve();
  }
  handleMergeCommand() {}
  onFirstResolve(cb) {
    calls.onFirstResolveCount += 1;
    this._firstResolveCb = cb;
    // Simulate the secondary sidebar resolving immediately (the real
    // SorcarSidebarView fires this callback the first time VS Code
    // calls resolveWebviewView).  Running synchronously is fine — the
    // extension code itself wraps the widen work in a setTimeout.
    setImmediate(() => {
      try {
        cb();
      } catch {
        // ignore
      }
    });
  }
  widenToOneThird() {
    calls.widenToOneThird += 1;
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
  // ``activate()`` registers a command for every value in
  // MERGE_ACTIONS — keep one entry so registerCommand runs at least
  // once but doesn't bloat the test.
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

// Force a fresh load of the compiled extension so the stubs above win.
const extensionPath = path.join(OUT_DIR, 'extension.js');
delete require.cache[require.resolve(extensionPath)];
const extension = require(extensionPath);
assert.strictEqual(
  typeof extension.activate,
  'function',
  'compiled extension must export activate()',
);
assert.strictEqual(
  typeof extension.deactivate,
  'function',
  'compiled extension must export deactivate()',
);

// ---- 3. Build a fake ExtensionContext ------------------------------

function makeMemento() {
  const store = new Map();
  return {
    get: (key, def) => (store.has(key) ? store.get(key) : def),
    update: (key, value) => {
      if (value === undefined) store.delete(key);
      else store.set(key, value);
      return Promise.resolve();
    },
    keys: () => Array.from(store.keys()),
    _store: store,
  };
}

function makeContext(workspaceState, globalState) {
  const tmpExtPath = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-ext-'));
  // ``activate()`` only stats these paths via fs.watchFile, which is a
  // no-op when the file is missing, so we just need the dir to exist.
  return {
    extensionUri: vscodeStub.Uri.file(tmpExtPath),
    extensionPath: tmpExtPath,
    subscriptions: [],
    workspaceState,
    globalState,
    _tmpExtPath: tmpExtPath,
  };
}

function disposeContext(ctx) {
  for (const d of ctx.subscriptions) {
    try {
      d.dispose && d.dispose();
    } catch {
      // ignore
    }
  }
  fs.rmSync(ctx._tmpExtPath, {recursive: true, force: true});
}

const sleep = ms => new Promise(r => setTimeout(r, ms));

// The widen path is gated behind ``setTimeout(..., 500)`` and the
// auto-open path behind ``setTimeout(..., 1000)`` inside activate(),
// so allow a little headroom for both to fire before we assert.
const WAIT_MS = 1500;

async function runTests() {
  let failures = 0;

  // --- Workspace 1 ---------------------------------------------------
  // Fresh machine: both flags absent.  Both widen + focus must fire.
  const globalState = makeMemento();
  const ws1State = makeMemento();
  const ctx1 = makeContext(ws1State, globalState);
  const widenBefore1 = calls.widenToOneThird;
  const focusBefore1 = calls.focusChatInput;
  extension.activate(ctx1);
  await sleep(WAIT_MS);
  const widen1 = calls.widenToOneThird - widenBefore1;
  const focus1 = calls.focusChatInput - focusBefore1;
  try {
    assert.strictEqual(
      widen1,
      1,
      `workspace 1: widenToOneThird must fire exactly once (got ${widen1})`,
    );
    assert.ok(
      focus1 >= 1,
      `workspace 1: focusChatInput must fire at least once (got ${focus1})`,
    );
    console.log('  ok - workspace 1 widens secondary sidebar and focuses chat');
  } catch (err) {
    failures += 1;
    console.log(`  FAIL - workspace 1 bootstrap: ${err.message}`);
  }
  extension.deactivate();
  disposeContext(ctx1);

  // --- Workspace 2 ---------------------------------------------------
  // Different folder opened in a new window.  ``workspaceState`` is
  // therefore fresh, but ``globalState`` is the SAME memento (same
  // machine, same VS Code install).  The bug: extension.ts gates on
  // globalState, so the second workspace inherits the "already done"
  // flag from workspace 1 and skips both widen + focus.  After the
  // fix the gate is workspaceState-based and both fire again.
  const ws2State = makeMemento();
  const ctx2 = makeContext(ws2State, globalState);
  const widenBefore2 = calls.widenToOneThird;
  const focusBefore2 = calls.focusChatInput;
  extension.activate(ctx2);
  await sleep(WAIT_MS);
  const widen2 = calls.widenToOneThird - widenBefore2;
  const focus2 = calls.focusChatInput - focusBefore2;
  try {
    assert.strictEqual(
      widen2,
      1,
      `workspace 2: widenToOneThird must fire exactly once on a NEW ` +
        `workspace (got ${widen2}). Likely cause: the widen gate is on ` +
        `globalState, which was already set by workspace 1, so the new ` +
        `workspace inherits the "already widened" flag and leaves the ` +
        `secondary sidebar at its default narrow width.`,
    );
    assert.ok(
      focus2 >= 1,
      `workspace 2: focusChatInput must fire at least once on a NEW ` +
        `workspace (got ${focus2}). Likely cause: the auto-open gate is ` +
        `on globalState ("firstLaunchDone"), so the new workspace ` +
        `inherits the flag from workspace 1 and never selects the ` +
        `KISS Sorcar tab in the secondary panel.`,
    );
    console.log('  ok - workspace 2 widens secondary sidebar and focuses chat');
  } catch (err) {
    failures += 1;
    console.log(`  FAIL - workspace 2 bootstrap: ${err.message}`);
  }
  extension.deactivate();
  disposeContext(ctx2);

  // --- Workspace 2 reopened ------------------------------------------
  // Same workspaceState memento as workspace 2 (the user closed and
  // reopened the same folder).  Widen + focus must NOT fire again —
  // the user has already tuned the sidebar width for this workspace.
  const ctx2b = makeContext(ws2State, globalState);
  const widenBefore3 = calls.widenToOneThird;
  const focusBefore3 = calls.focusChatInput;
  extension.activate(ctx2b);
  await sleep(WAIT_MS);
  const widen3 = calls.widenToOneThird - widenBefore3;
  const focus3 = calls.focusChatInput - focusBefore3;
  try {
    assert.strictEqual(
      widen3,
      0,
      `workspace 2 reopened: widenToOneThird must NOT fire again ` +
        `(got ${widen3}). The widen gate must persist per-workspace ` +
        `so we don't overwrite the user's manual width adjustments.`,
    );
    assert.strictEqual(
      focus3,
      0,
      `workspace 2 reopened: focusChatInput auto-open must NOT fire ` +
        `again (got ${focus3}). The first-launch gate must persist ` +
        `per-workspace.`,
    );
    console.log('  ok - workspace 2 reopened does not re-bootstrap');
  } catch (err) {
    failures += 1;
    console.log(`  FAIL - workspace 2 reopen idempotency: ${err.message}`);
  }
  extension.deactivate();
  disposeContext(ctx2b);

  return failures;
}

runTests().then(
  failures => {
    try {
      fs.unlinkSync(stubPath);
    } catch {
      // ignore
    }
    if (failures > 0) {
      console.error(`\n${failures} test(s) failed`);
      process.exit(1);
    }
    console.log('\nAll tests passed');
    process.exit(0);
  },
  err => {
    try {
      fs.unlinkSync(stubPath);
    } catch {
      // ignore
    }
    console.error('FAIL:', err && err.stack ? err.stack : err);
    process.exit(1);
  },
);
