// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// Regression test for the install-time "^C aborted my install" bug.
//
// The bug
// -------
// ``install.sh``'s step [5/6] runs ``tsc`` which rewrites the
// running-extension's ``out/extension.js`` ~5–15 s BEFORE the rest of
// the install (``copy-kiss.sh``, daemon restart, ``code
// --install-extension``) completes.  The extension used to also watch
// ``out/extension.js`` with a 2 s ``fs.watchFile`` poll, so the
// mid-install rewrite triggered ``workbench.action.reloadWindow``.
// VS Code's ``reloadWindow`` tears down the integrated terminal that
// is running ``install.sh``; the terminal shutdown writes ``\x03`` to
// the PTY (which is why users saw an unexplained ``^C``) and the
// install aborted.
//
// The fix
// -------
// ``extension.ts`` now watches ONLY ``~/.kiss/.extension-updated``.
// All install scripts touch that marker as their *final* step (after
// ``code --install-extension`` has returned and the kiss-web daemon
// has been restarted), so a stat change on it signals "extension is
// fully installed; safe to reload".  Modifying ``out/extension.js``
// alone must NOT trigger a reload, because doing so races with an
// in-flight ``install.sh``.
//
// What this test exercises
// ------------------------
// The REAL compiled ``out/extension.js`` activate(), with stubs for
// the dependent modules, against a real on-disk filesystem.  We
// redirect ``$HOME`` to a tmpdir so the marker we touch is the one
// the extension is watching, and we record every
// ``vscode.commands.executeCommand`` call so we can assert whether
// ``workbench.action.reloadWindow`` was issued.
//
// Run directly with ``node``:
//
//     node src/kiss/agents/vscode/test/reloadOnMarkerOnly.test.js

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

// ---- 1. Redirect HOME so the marker path lives in our tmpdir --------
// ``extension.ts`` derives ``markerPath`` from ``os.homedir()`` at
// activate() time, so we must override before requiring it.
const TMP_HOME = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-reload-home-'));
const ORIG_HOME = process.env.HOME;
process.env.HOME = TMP_HOME;
fs.mkdirSync(path.join(TMP_HOME, '.kiss'), {recursive: true});
const markerPath = path.join(TMP_HOME, '.kiss', '.extension-updated');
const sockPath = path.join(TMP_HOME, '.kiss', 'sorcar.sock');
// A pre-existing daemon socket lets the settle gate's ``socketUp``
// branch fire immediately so the reload is observed quickly instead
// of having to wait the full 3 s grace window.
fs.writeFileSync(sockPath, '');

// ---- 2. vscode module stub ------------------------------------------
// Captures executeCommand invocations.
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
  workspace: {asRelativePath: p => String(p)},
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

const stubPath = path.join(__dirname, '_vscode-stub-reload.js');
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

// ---- 3. Stub the heavy dependency modules ----------------------------
class FakeSidebarView {
  constructor() {}
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
// The real reloadGuard's ``isReloadReady`` checks an extension.js
// path; we want the settle gate to clear immediately once it polls,
// so always report ``codeReady`` + ``socketUp``.
stubModule(path.join(OUT_DIR, 'reloadGuard.js'), {
  isReloadReady: () => ({
    ready: true,
    codeReady: true,
    socketUp: true,
    size: 1,
  }),
});

// Force a fresh load.
const extensionPath = path.join(OUT_DIR, 'extension.js');
delete require.cache[require.resolve(extensionPath)];
const extension = require(extensionPath);

// ---- 4. ExtensionContext helpers ------------------------------------
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
  };
}

function makeContext() {
  const tmpExtPath = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-ext-'));
  // Mimic the on-disk layout of an installed extension: ``out/extension.js``
  // must be a real file so the watcher (had we kept it) and the settle
  // gate's stat probes can see it.
  const extOutDir = path.join(tmpExtPath, 'out');
  fs.mkdirSync(extOutDir, {recursive: true});
  const installedExtJs = path.join(extOutDir, 'extension.js');
  fs.writeFileSync(installedExtJs, '// compiled bundle');
  return {
    extensionUri: vscodeStub.Uri.file(tmpExtPath),
    extensionPath: tmpExtPath,
    subscriptions: [],
    workspaceState: makeMemento(),
    globalState: makeMemento(),
    _tmpExtPath: tmpExtPath,
    _installedExtJs: installedExtJs,
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

function countReloads() {
  return executedCommands.filter(
    e => e.cmd === 'workbench.action.reloadWindow',
  ).length;
}

// The watcher polls at 2000 ms; allow generous headroom so a stray
// reload, if any, has time to fire.
const POLL_INTERVAL_MS = 2000;
const WAIT_AFTER_TOUCH_MS = POLL_INTERVAL_MS * 3 + 1000;
// After writing the marker we also need the settle gate (500 ms ticks)
// to clear, which our stub returns ready immediately on the first poll.
const WAIT_AFTER_MARKER_MS = POLL_INTERVAL_MS * 2 + 1500;

async function runTests() {
  let failures = 0;

  // ------------------------------------------------------------------
  // Test 1: rewriting ``out/extension.js`` must NOT trigger a reload.
  // This is what install.sh's tsc step did mid-install, killing the
  // terminal running install.sh.
  // ------------------------------------------------------------------
  const ctx = makeContext();
  extension.activate(ctx);
  const beforeTouch = countReloads();

  // Simulate ``tsc`` rewriting the file: change mtime + content + size
  // so any naive watchFile callback (mtime, ino, size) would fire.
  await sleep(50);
  fs.writeFileSync(ctx._installedExtJs, '// recompiled bundle larger now');
  const nowSec = Date.now() / 1000;
  fs.utimesSync(ctx._installedExtJs, nowSec, nowSec);

  await sleep(WAIT_AFTER_TOUCH_MS);
  const afterTouch = countReloads();
  try {
    assert.strictEqual(
      afterTouch - beforeTouch,
      0,
      `rewriting out/extension.js must NOT trigger workbench.action.reloadWindow ` +
        `(got ${afterTouch - beforeTouch} reload(s)). ` +
        `If this fires, the extension is racing with install.sh's tsc step ` +
        `and will tear down the terminal running install.sh mid-install.`,
    );
    console.log('  ok - rewriting out/extension.js does not trigger reload');
  } catch (err) {
    failures += 1;
    console.log(`  FAIL - extension.js touch trigger: ${err.message}`);
  }

  // ------------------------------------------------------------------
  // Test 2: writing the ``~/.kiss/.extension-updated`` marker MUST
  // trigger a reload.  install.sh writes this marker as its final
  // step, when the extension is fully installed and the daemon is
  // back up — exactly when a reload is wanted.
  // ------------------------------------------------------------------
  const beforeMarker = countReloads();
  fs.writeFileSync(markerPath, new Date().toISOString());

  await sleep(WAIT_AFTER_MARKER_MS);
  const afterMarker = countReloads();
  try {
    assert.ok(
      afterMarker - beforeMarker >= 1,
      `writing ${markerPath} MUST trigger workbench.action.reloadWindow ` +
        `(got ${afterMarker - beforeMarker} reload(s)). ` +
        `If this never fires, install.sh's "all done" signal is being ` +
        `ignored and the user is stranded on the old code until they ` +
        `restart VS Code by hand.`,
    );
    console.log('  ok - writing .extension-updated marker triggers reload');
  } catch (err) {
    failures += 1;
    console.log(`  FAIL - marker reload: ${err.message}`);
  }

  extension.deactivate();
  disposeContext(ctx);
  return failures;
}

runTests().then(
  failures => {
    try {
      fs.unlinkSync(stubPath);
    } catch {
      // ignore
    }
    fs.rmSync(TMP_HOME, {recursive: true, force: true});
    if (ORIG_HOME === undefined) delete process.env.HOME;
    else process.env.HOME = ORIG_HOME;
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
    fs.rmSync(TMP_HOME, {recursive: true, force: true});
    if (ORIG_HOME === undefined) delete process.env.HOME;
    else process.env.HOME = ORIG_HOME;
    console.error('FAIL:', err && err.stack ? err.stack : err);
    process.exit(1);
  },
);
