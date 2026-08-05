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

    executedCommands.length = 0;
    const p2 = trigger(undefined, undefined, undefined);
    await sleep(2300);
    await p2;
    const revealsDuring = scmRevealCount();
    assert.ok(
      revealsDuring <= 1,
      `BUG 2: workbench.view.scm executed ${revealsDuring} times during a ` +
        'single generation — the per-second countdown ticks must not ' +
        'steal the sidebar focus',
    );

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
