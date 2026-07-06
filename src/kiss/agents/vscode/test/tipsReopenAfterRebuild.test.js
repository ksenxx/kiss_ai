// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end regression test: the Tips window must re-open after
// ``./scripts/build-extension.sh`` rebuilds/reinstalls the extension
// and kiss-web restarts.
//
// The bug
// -------
// ``consumeTipsFirstRun()`` persists ``~/.kiss/TIPS_SHOWN`` forever, so
// the Tips window auto-opened only once per machine — ever.  After
// ``./scripts/build-extension.sh`` reinstalled the extension, killed
// the kiss-web daemon and wrote ``~/.kiss/.extension-updated`` (which
// makes every VS Code window reload), the reloaded extension rendered
// the chat webview with ``show:false`` because ``TIPS_SHOWN`` still
// existed.  The Tips window therefore never opened after a rebuild.
//
// The fix
// -------
// ``activate()`` calls ``resetTipsOnExtensionUpdate()``: when the
// ``.extension-updated`` marker is present at activation (i.e. the
// window just reloaded because of a rebuild/reinstall), the
// ``TIPS_SHOWN`` marker is removed so the next chat webview render
// auto-opens the Tips window exactly once again.
//
// What this test exercises
// ------------------------
// The REAL compiled ``out/extension.js`` ``activate()`` and the REAL
// ``out/SorcarTab.js`` ``buildChatHtml()`` against a real on-disk
// filesystem, with ``$HOME``/``$KISS_HOME`` redirected to a tmpdir.
// The rebuild is simulated by executing the actual marker-writing
// command extracted from ``scripts/build-extension.sh`` plus the
// daemon-restart side effect (socket removal), i.e. exactly what the
// script does as its final steps.
//
// Run directly with ``node`` (after ``npm run compile``):
//
//     node src/kiss/agents/vscode/test/tipsReopenAfterRebuild.test.js

'use strict';

const assert = require('assert');
const {execFileSync} = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');
const Module = require('module');

const EXT_ROOT = path.join(__dirname, '..');
const OUT_DIR = path.join(EXT_ROOT, 'out');
const REPO_ROOT = path.resolve(EXT_ROOT, '..', '..', '..', '..');
const BUILD_SCRIPT = path.join(REPO_ROOT, 'scripts', 'build-extension.sh');

assert.ok(
  fs.existsSync(path.join(OUT_DIR, 'extension.js')),
  `compiled extension missing: ${OUT_DIR}/extension.js — run \`npm run compile\` first`,
);
assert.ok(
  fs.existsSync(BUILD_SCRIPT),
  `build script missing: ${BUILD_SCRIPT}`,
);

// ---- 1. Redirect HOME / KISS_HOME so all markers live in a tmpdir ----
// ``extension.ts`` derives the ``.extension-updated`` path from
// ``os.homedir()`` and ``SorcarTab.ts`` derives ``TIPS_SHOWN`` from
// ``kissHomeDir()`` (KISS_HOME || ~/.kiss), so both env vars are set
// before the compiled modules are required.
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
const sockPath = path.join(KISS_HOME, 'sorcar.sock');

// ---- 2. vscode module stub ------------------------------------------
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

// ---- 3. Stub the heavy dependency modules ----------------------------
// SorcarTab.js is NOT stubbed: the tips logic under test lives there.
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
stubModule(path.join(OUT_DIR, 'UpdateChecker.js'), {
  checkForExtensionUpdate: () => Promise.resolve(),
});

const extension = require(path.join(OUT_DIR, 'extension.js'));
const {buildChatHtml} = require(path.join(OUT_DIR, 'SorcarTab.js'));

// ---- 4. Helpers -------------------------------------------------------
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
      // ignore
    }
  }
  fs.rmSync(ctx._tmpExtPath, {recursive: true, force: true});
}

/**
 * Render the chat webview HTML exactly like SorcarTab does when the
 * chat view resolves, and return the injected ``window.__TIPS__``.
 */
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

/**
 * Simulate the final steps of ``./scripts/build-extension.sh`` by
 * executing the ACTUAL marker-write command extracted from the script
 * (so the test tracks the script's real behaviour), plus the kiss-web
 * daemon restart side effect it performs (``rm -f ~/.kiss/sorcar.sock``
 * before the LaunchAgent respawns the daemon).
 */
function runBuildExtensionFinalSteps() {
  const script = fs.readFileSync(BUILD_SCRIPT, 'utf-8');
  const lines = script
    .split('\n')
    .filter(l => /^[^#]*\.extension-updated"/.test(l) && !/^\s*#/.test(l));
  assert.ok(
    lines.length >= 1,
    'build-extension.sh must write the ~/.kiss/.extension-updated marker',
  );
  const snippet =
    'rm -f "$HOME/.kiss/sorcar.sock"\nmkdir -p "$HOME/.kiss"\n' +
    lines.join('\n');
  execFileSync('bash', ['-c', snippet], {
    env: {...process.env, HOME: TMP_HOME},
  });
  assert.ok(
    fs.existsSync(markerPath) && fs.statSync(markerPath).size > 0,
    'the extracted script command must have written the update marker',
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

// ---- 5. Scenario ------------------------------------------------------
async function run() {
  // (a) Fresh install: first window activation, chat opens, tips show
  //     exactly once, then never again on ordinary reopens.
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

  // (b) A plain window reload WITHOUT a rebuild must not re-open tips.
  const ctx2 = makeContext();
  extension.activate(ctx2);
  check('plain reload without rebuild: tips stay closed', () => {
    assert.strictEqual(renderTipsConfig().show, false);
  });
  extension.deactivate();
  disposeContext(ctx2);

  // (c) User runs ./scripts/build-extension.sh: the VSIX is
  //     reinstalled, the kiss-web daemon is killed (LaunchAgent
  //     restarts it) and the .extension-updated marker is written,
  //     which makes every VS Code window reload.
  runBuildExtensionFinalSteps();
  fs.writeFileSync(sockPath, ''); // daemon respawned by LaunchAgent

  // (d) The reloaded window activates with the marker present.  The
  //     Tips window MUST auto-open again — this is the reported bug.
  const ctx3 = makeContext();
  extension.activate(ctx3);
  check('after build-extension.sh + kiss-web restart: tips re-open', () => {
    assert.deepStrictEqual(renderTipsConfig(), {
      tips: ['Hello **rebuild** tips.'],
      show: true,
    });
  });
  check('after rebuild: tips shown once, closed on later renders', () => {
    assert.strictEqual(renderTipsConfig().show, false);
  });
  extension.deactivate();
  disposeContext(ctx3);

  // (e) The next plain reload (marker already consumed by the
  //     dependency installer in real life) must not re-open tips.
  fs.rmSync(markerPath, {force: true});
  const ctx4 = makeContext();
  extension.activate(ctx4);
  check('next plain reload after rebuild: tips stay closed', () => {
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
      // ignore
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
      // ignore
    }
    fs.rmSync(TMP_HOME, {recursive: true, force: true});
    console.error('FAIL:', err && err.stack ? err.stack : err);
    process.exit(1);
  },
);
