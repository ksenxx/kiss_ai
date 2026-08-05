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

const TMP_HOME = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-tips-rebuild-'));
const KISS_HOME = path.join(TMP_HOME, '.kiss');
const ORIG_ENV = {
  HOME: process.env.HOME,
  KISS_HOME: process.env.KISS_HOME,
  KISS_TIPS_PATH: process.env.KISS_TIPS_PATH,
  KISS_PROJECT_PATH: process.env.KISS_PROJECT_PATH,
};
process.env.HOME = TMP_HOME;
process.env.KISS_HOME = KISS_HOME;
delete process.env.KISS_PROJECT_PATH;
fs.mkdirSync(KISS_HOME, {recursive: true});

const tipsFile = path.join(TMP_HOME, 'TIPS.md');
fs.writeFileSync(tipsFile, '# Tip\n\nHello **rebuild** tips.\n');
process.env.KISS_TIPS_PATH = tipsFile;

const markerPath = path.join(KISS_HOME, '.extension-updated');
const tipsShownPath = path.join(KISS_HOME, 'TIPS_SHOWN');

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
    executeCommand: () => Promise.resolve(),
  },
  workspace: {
    isTrusted: true,
    workspaceFolders: [],
    asRelativePath: p => String(p),
    getConfiguration() {
      return {get: () => undefined};
    },
  },
  Uri: {
    file: p => ({fsPath: p, scheme: 'file', toString: () => `file://${p}`}),
    joinPath(base, ...parts) {
      return {fsPath: path.join(base.fsPath, ...parts)};
    },
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

const stubPath = path.join(__dirname, '_vscode-stub-tips-rebuild.js');
fs.writeFileSync(
  stubPath,
  "'use strict';\nmodule.exports = global.__kissVscodeStub || {};\n",
);
global.__kissVscodeStub = vscodeStub;
const origResolve = Module._resolveFilename;
Module._resolveFilename = function (request, parent, ...rest) {
  if (request === 'vscode') return require.resolve(stubPath);
  return origResolve.call(this, request, parent, ...rest);
};

class FakeSidebarView {
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
  ensureDependencies: () => {
    fs.rmSync(markerPath, {force: true});
    return Promise.resolve();
  },
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
stubModule(path.join(OUT_DIR, 'UpdateChecker.js'), {
  checkForExtensionUpdate: () => Promise.resolve(),
});

const extension = require(path.join(OUT_DIR, 'extension.js'));
const {buildChatHtml} = require(path.join(OUT_DIR, 'SorcarTab.js'));

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
  fs.writeFileSync(path.join(extOutDir, 'extension.js'), '// compiled');
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
      d.dispose && d.dispose();
    } catch {
    }
  }
  fs.rmSync(ctx._tmpExtPath, {recursive: true, force: true});
}

function renderTipsConfig() {
  const webview = {
    cspSource: 'vscode-webview://stub',
    asWebviewUri(uri) {
      return {toString: () => 'vscode-webview://' + uri.fsPath};
    },
  };
  const html = buildChatHtml(webview, {fsPath: EXT_ROOT}, 'test-model');
  const m = html.match(/window\.__TIPS__\s*=\s*(\{.*?\});<\/script>/);
  assert.ok(m, 'window.__TIPS__ must be assigned a JSON object literal');
  return JSON.parse(m[1].replace(/<\\\//g, '</'));
}

function writeExtensionUpdateMarker() {
  fs.writeFileSync(markerPath, new Date().toISOString() + '\n');
  assert.ok(
    fs.existsSync(markerPath) && fs.statSync(markerPath).size > 0,
    'the update marker must exist and be non-empty',
  );
}

let failures = 0;

function check(name, fn) {
  try {
    fn();
    console.log(`  ok - ${name}`);
  } catch (err) {
    failures += 1;
    console.log(`  FAIL - ${name}: ${err && err.message}`);
  }
}

async function run() {
  const ctx1 = makeContext();
  extension.activate(ctx1);
  check('fresh install: tips auto-open on first chat render', () => {
    assert.deepStrictEqual(renderTipsConfig(), {
      tips: ['Hello **rebuild** tips.'],
      show: true,
    });
    assert.ok(fs.existsSync(tipsShownPath), 'TIPS_SHOWN marker written');
  });
  check('fresh install: tips stay closed on later renders', () => {
    assert.strictEqual(renderTipsConfig().show, false);
  });
  extension.deactivate();
  disposeContext(ctx1);

  const ctx2 = makeContext();
  extension.activate(ctx2);
  check('plain reload without rebuild: tips stay closed', () => {
    assert.strictEqual(renderTipsConfig().show, false);
  });
  extension.deactivate();
  disposeContext(ctx2);

  writeExtensionUpdateMarker();

  const ctx3 = makeContext();
  extension.activate(ctx3);
  check('after extension update marker: tips re-open', () => {
    assert.deepStrictEqual(renderTipsConfig(), {
      tips: ['Hello **rebuild** tips.'],
      show: true,
    });
  });
  check('after update: tips shown once, closed on later renders', () => {
    assert.strictEqual(renderTipsConfig().show, false);
  });
  extension.deactivate();
  disposeContext(ctx3);

  fs.rmSync(markerPath, {force: true});
  const ctx4 = makeContext();
  extension.activate(ctx4);
  check('next plain reload after update: tips stay closed', () => {
    assert.strictEqual(renderTipsConfig().show, false);
  });
  extension.deactivate();
  disposeContext(ctx4);
}

run().then(
  () => {
    try {
      fs.unlinkSync(stubPath);
    } catch {
    }
    fs.rmSync(TMP_HOME, {recursive: true, force: true});
    for (const [k, v] of Object.entries(ORIG_ENV)) {
      if (v === undefined) delete process.env[k];
      else process.env[k] = v;
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
    }
    fs.rmSync(TMP_HOME, {recursive: true, force: true});
    console.error('FAIL:', err && err.stack ? err.stack : err);
    process.exit(1);
  },
);
