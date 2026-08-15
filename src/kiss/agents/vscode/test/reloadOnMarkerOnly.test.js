// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

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

const TMP_HOME = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-reload-home-'));
const ORIG_HOME = process.env.HOME;
process.env.HOME = TMP_HOME;
fs.mkdirSync(path.join(TMP_HOME, '.kiss'), {recursive: true});
const markerPath = path.join(TMP_HOME, '.kiss', '.extension-updated');
const sockPath = path.join(TMP_HOME, '.kiss', 'sorcar.sock');
fs.writeFileSync(sockPath, '');

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
});
stubModule(path.join(OUT_DIR, 'DependencyInstaller.js'), {
  ensureLocalBinInPath: () => {},
  ensureDependencies: () => Promise.resolve(),
});
stubModule(path.join(OUT_DIR, 'gitApi.js'), {
  getGitApi: () => Promise.resolve(undefined),
});
stubModule(path.join(OUT_DIR, 'reloadGuard.js'), {
  isReloadReady: () => ({
    ready: true,
    codeReady: true,
    socketUp: true,
    size: 1,
  }),
});

const extensionPath = path.join(OUT_DIR, 'extension.js');
delete require.cache[require.resolve(extensionPath)];
const extension = require(extensionPath);

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

const POLL_INTERVAL_MS = 2000;
const WAIT_AFTER_TOUCH_MS = POLL_INTERVAL_MS * 3 + 1000;
const WAIT_AFTER_MARKER_MS = POLL_INTERVAL_MS * 2 + 1500;

async function runTests() {
  let failures = 0;

  const ctx = makeContext();
  extension.activate(ctx);
  const beforeTouch = countReloads();

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
    }
    fs.rmSync(TMP_HOME, {recursive: true, force: true});
    if (ORIG_HOME === undefined) delete process.env.HOME;
    else process.env.HOME = ORIG_HOME;
    console.error('FAIL:', err && err.stack ? err.stack : err);
    process.exit(1);
  },
);
