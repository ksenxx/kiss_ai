// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// Integration test for the SCM commit-message countdown in
// ``extension.ts`` (drives the real compiled ``out/extension.js``
// activate() with only ``vscode`` and sibling modules stubbed, the same
// pattern as activationUpdateNotificationAction.test.js).
//
// Bugs locked in:
//
//   1. Interval leak — when the daemon never replies,
//      ``generateCommitMessage`` resolves via its internal timeout
//      WITHOUT firing ``onCommitMessage``, so nothing ever called
//      ``stopCommitCountdown``: the countdown interval kept overwriting
//      the SCM input box with "Generating in 0s ..." every second
//      FOREVER.  The generation promise must tear the countdown down on
//      settle and wipe the stale text.
//
//   2. Focus steal — ``setScmMessage`` executed ``workbench.view.scm``
//      on EVERY countdown tick, yanking the SCM sidebar open once per
//      second for the whole generation.  The reveal must happen only on
//      user-visible transitions (start / final message).
//
//   3. The success path must be untouched: a real ``onCommitMessage``
//      result lands in the input box and is NOT clobbered by the
//      teardown.
//
// Run directly with ``node`` (after ``npm run compile``):
//
//     node src/kiss/agents/vscode/test/commitCountdownTeardown.test.js

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
const registeredCommands = new Map();

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
    registerCommand: (name, fn) => {
      registeredCommands.set(name, fn);
      return makeDisposable();
    },
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

// --- Fake sidebar: mirrors the real generateCommitMessage contract ------
// (resolves when onCommitMessage fires OR after a timeout, whichever
// comes first) — with the 30 s production timeout compressed to 300 ms.
const commitListeners = [];
const FAKE_TIMEOUT_MS = 300;

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
  onCommitMessage(cb) {
    commitListeners.push(cb);
    return makeDisposable();
  }
  generateCommitMessage() {
    return new Promise(resolve => {
      let done = false;
      const finish = () => {
        if (done) return;
        done = true;
        const i = commitListeners.indexOf(finish);
        if (i >= 0) commitListeners.splice(i, 1);
        resolve();
      };
      commitListeners.push(finish);
      setTimeout(finish, FAKE_TIMEOUT_MS);
    });
  }
  handleMergeCommand() {}
  onFirstResolve() {}
  widenToOneThird() {
    return Promise.resolve();
  }
  runUpdate() {}
  dispose() {}
}

function fireCommitMessage(ev) {
  for (const cb of commitListeners.slice()) cb(ev);
}

// --- Fake git repo whose inputBox we observe ----------------------------
const inputBox = {value: ''};
const fakeGitApi = {
  repositories: [{inputBox, state: {indexChanges: [{}]}}],
};

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
  getGitApi: () => Promise.resolve(fakeGitApi),
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
  checkForExtensionUpdate: async () => ({
    checked: false,
    notified: false,
    latest: null,
    current: null,
    reason: 'test',
  }),
});

delete require.cache[require.resolve(extensionPath)];
const extension = require(extensionPath);

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

function scmRevealCount() {
  return executedCommands.filter(c => c.cmd === 'workbench.view.scm').length;
}

async function runTest() {
  const tmpExtPath = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-ext-cd-'));
  const ctx = {
    extensionUri: vscodeStub.Uri.file(tmpExtPath),
    extensionPath: tmpExtPath,
    subscriptions: [],
    workspaceState: makeMemento(),
    globalState: makeMemento(),
  };
  try {
    extension.activate(ctx);
    const trigger = registeredCommands.get('kissSorcar.generateCommitMessage');
    assert.ok(trigger, 'kissSorcar.generateCommitMessage must be registered');

    // ------------------------------------------------------------------
    // Bug 1: daemon never replies — the generation promise settles via
    // timeout without an onCommitMessage event.  The countdown must be
    // torn down and the stale "Generating in Ns ..." text wiped.
    // ------------------------------------------------------------------
    await trigger(undefined, undefined, undefined);
    await sleep(50);
    assert.strictEqual(
      inputBox.value,
      '',
      'BUG 1: stale "Generating in Ns ..." text must be wiped when the ' +
        'generation ends without a reply',
    );
    const valueAfterSettle = inputBox.value;
    const revealsAfterSettle = scmRevealCount();
    await sleep(2300);
    assert.strictEqual(
      inputBox.value,
      valueAfterSettle,
      'BUG 1: the countdown interval must be cleared when the generation ' +
        'promise settles — the SCM input box kept being overwritten with ' +
        '"Generating in 0s ..." every second forever',
    );
    assert.strictEqual(
      scmRevealCount(),
      revealsAfterSettle,
      'BUG 1: no further workbench.view.scm executions after the ' +
        'generation settled',
    );

    // ------------------------------------------------------------------
    // Bug 2: during the countdown, workbench.view.scm must be executed
    // only once (at start) — not on every tick.
    // ------------------------------------------------------------------
    executedCommands.length = 0;
    const p2 = trigger(undefined, undefined, undefined);
    await sleep(2300); // > 2 countdown ticks; fake timeout already passed
    await p2;
    const revealsDuring = scmRevealCount();
    assert.ok(
      revealsDuring <= 1,
      `BUG 2: workbench.view.scm executed ${revealsDuring} times during a ` +
        'single generation — the per-second countdown ticks must not ' +
        'steal the sidebar focus',
    );

    // ------------------------------------------------------------------
    // Success path: a real result lands and is NOT clobbered.
    // ------------------------------------------------------------------
    const p3 = trigger(undefined, undefined, undefined);
    await sleep(50);
    assert.ok(
      /^Generating in \d+s \.\.\.$/.test(inputBox.value),
      `countdown text expected during generation, got: "${inputBox.value}"`,
    );
    fireCommitMessage({message: 'feat: add countdown teardown'});
    await p3;
    await sleep(1500);
    assert.strictEqual(
      inputBox.value,
      'feat: add countdown teardown',
      'the generated commit message must land in the SCM input box and ' +
        'survive the teardown (no clobber, no countdown overwrite)',
    );

    console.log('\nAll commit countdown teardown tests passed');
  } finally {
    extension.deactivate();
    for (const d of ctx.subscriptions) {
      try {
        if (d && typeof d.dispose === 'function') d.dispose();
      } catch {}
    }
    fs.rmSync(tmpExtPath, {recursive: true, force: true});
    Module._load = origLoad;
  }
}

runTest().then(
  () => process.exit(0),
  err => {
    console.error('FAIL:', err);
    process.exit(1);
  },
);
