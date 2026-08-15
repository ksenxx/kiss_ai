// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// End-to-end test of commit-message generation in a workspace holding
// more than one git repository.
//
// VS Code hands the SCM sparkle's callback the repository the user
// clicked in. The extension used that repository to decide whether
// anything was staged and to choose which SCM input box gets the
// answer -- but then asked the daemon for a message WITHOUT it, so the
// daemon diffed the workspace's first folder instead. Click the sparkle
// in repository B and you got repository A's commit message written
// into B's box, or "nothing staged" because A had nothing staged.
//
// The two were also collapsed onto one generation: the extension kept a
// single in-flight promise and the daemon claims one generation per
// tabId -- and every SCM request carried the same empty tabId. So a
// generation running for A silently swallowed a request for B.
//
// The real compiled extension and the real SorcarSidebarView are driven
// here; the daemon is a real unix domain socket speaking the real line
// protocol, and it answers the way the daemon does: stamping back the
// tabId it was asked with. Only the VS Code API surface and the git
// extension -- which do not exist outside VS Code -- are stubbed.

const assert = require('assert');
const fs = require('fs');
const net = require('net');
const os = require('os');
const path = require('path');
const Module = require('module');

const EXT_ROOT = path.join(__dirname, '..');
const OUT_DIR = path.join(EXT_ROOT, 'out');
const extensionPath = path.join(OUT_DIR, 'extension.js');

assert.ok(
  fs.existsSync(extensionPath),
  `compiled extension missing: ${extensionPath} — run \`npm run compile\``,
);

if (process.platform === 'win32') {
  console.log('SKIP: UDS tests require a POSIX platform');
  process.exit(0);
}

const tmpHome = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-commit-multi-'));
fs.mkdirSync(path.join(tmpHome, '.kiss'), {recursive: true});
const sockPath = path.join(tmpHome, '.kiss', 'sorcar.sock');
process.env.HOME = tmpHome;
process.env.USERPROFILE = tmpHome;
process.env.KISS_HOME = path.join(tmpHome, '.kiss');
process.env.KISS_SORCAR_SOCK = sockPath;

// Two repositories, side by side, the way a multi-root workspace or a
// repo with a vendored sub-checkout looks to the git extension.
const REPO_A = path.join(tmpHome, 'alpha');
const REPO_B = path.join(tmpHome, 'beta');
fs.mkdirSync(REPO_A, {recursive: true});
fs.mkdirSync(REPO_B, {recursive: true});

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

function makeUri(p) {
  return {fsPath: p, scheme: 'file', toString: () => `file://${p}`};
}

class StubEventEmitter {
  constructor() {
    this._listeners = [];
    this.event = cb => {
      this._listeners.push(cb);
      return {
        dispose: () => {
          const i = this._listeners.indexOf(cb);
          if (i >= 0) this._listeners.splice(i, 1);
        },
      };
    };
  }
  fire(arg) {
    for (const cb of this._listeners.slice()) cb(arg);
  }
  dispose() {
    this._listeners = [];
  }
}

const registeredCommands = new Map();
// The workspace opens on repository A, so A is what _getWorkDir()
// answers: the bug is invisible unless the two differ.
const workspaceFolders = [{uri: makeUri(REPO_A)}];

const vscodeStub = {
  window: {
    registerWebviewViewProvider: () => makeDisposable(),
    createTreeView: () => ({
      onDidChangeVisibility: () => makeDisposable(),
      dispose: () => {},
    }),
    withProgress: (_opts, task) =>
      task(
        {report: () => {}},
        {onCancellationRequested: () => makeDisposable()},
      ),
    showInformationMessage: () => Promise.resolve(undefined),
    showErrorMessage: () => Promise.resolve(undefined),
    showWarningMessage: () => Promise.resolve(undefined),
    showTextDocument: () => Promise.resolve({}),
    activeTextEditor: undefined,
    tabGroups: {all: []},
  },
  commands: {
    registerCommand: (name, fn) => {
      registeredCommands.set(name, fn);
      return makeDisposable();
    },
    executeCommand: () => Promise.resolve(),
  },
  workspace: {
    get workspaceFolders() {
      return workspaceFolders;
    },
    asRelativePath: p => String(p),
    getConfiguration: () => ({get: () => 'stub-default-model'}),
    onDidChangeWorkspaceFolders: () => makeDisposable(),
    openTextDocument: () => Promise.resolve({getText: () => ''}),
    textDocuments: [],
  },
  Uri: {
    file: makeUri,
    joinPath: (base, ...parts) => makeUri(path.join(base.fsPath, ...parts)),
    parse: makeUri,
  },
  EventEmitter: StubEventEmitter,
  CancellationTokenSource: class {
    constructor() {
      this.token = {onCancellationRequested: () => makeDisposable()};
    }
    dispose() {}
  },
  Position: class {},
  Range: class {},
  Selection: class {},
  TextEditorRevealType: {InCenter: 2, AtTop: 3},
  ProgressLocation: {Notification: 15},
  ViewColumn: {One: 1},
  TreeItem: class {
    constructor(label) {
      this.label = label;
    }
  },
};

const origResolve = Module._resolveFilename;
Module._resolveFilename = function (request, parent, ...rest) {
  if (request === 'vscode') return require.resolve('./_vscode-stub.js');
  return origResolve.call(this, request, parent, ...rest);
};
global.__kissVscodeStub = vscodeStub;

function makeRepo(root) {
  return {
    rootUri: makeUri(root),
    inputBox: {value: ''},
    state: {indexChanges: [{}]},
  };
}

const repoA = makeRepo(REPO_A);
const repoB = makeRepo(REPO_B);
const fakeGitApi = {repositories: [repoA, repoB]};

function stubModule(filePath, exports) {
  const fakeMod = new Module(filePath);
  fakeMod.filename = filePath;
  fakeMod.loaded = true;
  fakeMod.exports = exports;
  require.cache[filePath] = fakeMod;
}

stubModule(path.join(OUT_DIR, 'gitApi.js'), {
  getGitApi: () => Promise.resolve(fakeGitApi),
});
stubModule(path.join(OUT_DIR, 'DependencyInstaller.js'), {
  ensureLocalBinInPath: () => {},
  ensureDependencies: () => Promise.resolve(),
});
stubModule(path.join(OUT_DIR, 'reloadGuard.js'), {
  isReloadReady: () => ({codeReady: false, socketUp: false, size: 0}),
});
stubModule(path.join(OUT_DIR, 'kissPaths.js'), {
  findKissProject: () => path.join(tmpHome, 'kiss_project'),
});
stubModule(path.join(OUT_DIR, 'WebviewNotifications.js'), {
  showInformationNotification: () => Promise.resolve(undefined),
  showWarningNotification: () => Promise.resolve(undefined),
  showErrorNotification: () => Promise.resolve(undefined),
  setWebviewNotificationPoster: () => {},
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

// A real daemon: a real unix socket speaking the real line protocol.
const daemonLines = [];
let daemonSock = null;
const server = net.createServer(sock => {
  daemonSock = sock;
  let buf = '';
  sock.on('data', d => {
    buf += d.toString();
    const parts = buf.split('\n');
    buf = parts.pop();
    for (const line of parts) {
      if (line.trim()) daemonLines.push(JSON.parse(line));
    }
  });
});

// The daemon stamps the requesting tabId on every commitMessage it
// broadcasts, so the printer routes it back to the tab that asked.
function answerGeneration(tabId, payload) {
  assert.ok(daemonSock, 'the extension must have connected to the daemon');
  daemonSock.write(JSON.stringify({type: 'commitMessage', tabId, ...payload}));
  daemonSock.write('\n');
}

function generations() {
  return daemonLines.filter(m => m.type === 'generateCommitMessage');
}

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

async function waitFor(predicate, message, timeoutMs = 4000) {
  const start = Date.now();
  for (;;) {
    const v = predicate();
    if (v) return v;
    if (Date.now() - start >= timeoutMs) throw new Error(message);
    await sleep(10);
  }
}

async function runTests() {
  await new Promise((res, rej) =>
    server.listen(sockPath, err => (err ? rej(err) : res())),
  );

  delete require.cache[require.resolve(extensionPath)];
  const extension = require(extensionPath);

  const ctx = {
    extensionUri: makeUri(EXT_ROOT),
    extensionPath: EXT_ROOT,
    subscriptions: [],
    workspaceState: makeMemento(),
    globalState: makeMemento(),
  };
  try {
    extension.activate(ctx);
    const trigger = registeredCommands.get('kissSorcar.generateCommitMessage');
    assert.ok(trigger, 'kissSorcar.generateCommitMessage must be registered');

    // --- the daemon must be asked about the repo the user clicked -----
    daemonLines.length = 0;
    repoA.inputBox.value = '';
    repoB.inputBox.value = '';
    const bOnly = trigger(makeUri(REPO_B), undefined, undefined);
    await waitFor(
      () => generations().length,
      'the extension must ask the daemon for a commit message',
    );
    assert.strictEqual(
      generations()[0].workDir,
      REPO_B,
      'the daemon must be asked to diff the repository the sparkle was ' +
        'clicked in, not whichever folder the workspace opened on',
    );

    answerGeneration(generations()[0].tabId, {message: 'feat: beta change'});
    await bOnly;
    await sleep(150);
    assert.strictEqual(
      repoB.inputBox.value,
      'feat: beta change',
      "the answer must land in the clicked repository's SCM input",
    );
    assert.strictEqual(
      repoA.inputBox.value,
      '',
      'and must not touch the other repository',
    );

    // --- two repositories generate at the same time -------------------
    daemonLines.length = 0;
    repoA.inputBox.value = '';
    repoB.inputBox.value = '';
    const pA = trigger(makeUri(REPO_A), undefined, undefined);
    const pB = trigger(makeUri(REPO_B), undefined, undefined);
    await waitFor(
      () => generations().length >= 2,
      'a request for a second repository must not be swallowed by the ' +
        'one already running for the first',
      2000,
    );
    const two = generations();
    assert.strictEqual(two.length, 2, 'exactly two generations, one each');
    const byDir = new Map(two.map(g => [g.workDir, g]));
    assert.ok(byDir.has(REPO_A) && byDir.has(REPO_B), 'one per repository');
    assert.notStrictEqual(
      byDir.get(REPO_A).tabId,
      byDir.get(REPO_B).tabId,
      'the two must reach the daemon under different tab ids: it claims ' +
        'one generation per tab and would otherwise drop the second',
    );

    // They come back out of order, as two independent LLM calls do.
    answerGeneration(byDir.get(REPO_B).tabId, {message: 'fix: beta'});
    await sleep(80);
    answerGeneration(byDir.get(REPO_A).tabId, {message: 'chore: alpha'});
    await Promise.all([pA, pB]);
    await sleep(150);
    assert.strictEqual(
      repoA.inputBox.value,
      'chore: alpha',
      "each repository must receive its own message, not the other's",
    );
    assert.strictEqual(repoB.inputBox.value, 'fix: beta');

    // --- a double click on ONE repository still joins ------------------
    daemonLines.length = 0;
    repoB.inputBox.value = '';
    const d1 = trigger(makeUri(REPO_B), undefined, undefined);
    const d2 = trigger(makeUri(REPO_B), undefined, undefined);
    await waitFor(() => generations().length, 'a generation must be sent');
    await sleep(120);
    assert.strictEqual(
      generations().length,
      1,
      'two clicks on the SAME repository must join one generation, not ' +
        'pay for two',
    );
    answerGeneration(generations()[0].tabId, {message: 'docs: joined'});
    await Promise.all([d1, d2]);
    await sleep(150);
    assert.strictEqual(
      repoB.inputBox.value,
      'docs: joined',
      "the joined generation's result must still land",
    );

    // --- staged-ness is judged per repository -------------------------
    daemonLines.length = 0;
    repoA.inputBox.value = '';
    repoB.inputBox.value = '';
    repoB.state.indexChanges = [];
    await trigger(makeUri(REPO_B), undefined, undefined);
    await sleep(100);
    assert.strictEqual(
      generations().length,
      0,
      'a repository with nothing staged must not cost an LLM call',
    );
    assert.strictEqual(
      repoB.inputBox.value,
      'Error: nothing staged',
      'and must say so in its own input box',
    );
    assert.strictEqual(
      repoA.inputBox.value,
      '',
      "...without disturbing the other repository's box",
    );

    // The other repository is still staged and still works.
    const stillWorks = trigger(makeUri(REPO_A), undefined, undefined);
    await waitFor(
      () => generations().length,
      'the staged repository must still generate',
    );
    assert.strictEqual(generations()[0].workDir, REPO_A);
    answerGeneration(generations()[0].tabId, {message: 'chore: alpha again'});
    await stillWorks;
    await sleep(150);
    assert.strictEqual(repoA.inputBox.value, 'chore: alpha again');
    repoB.state.indexChanges = [{}];

    console.log('  ok - the clicked repository is the one that is diffed');
    console.log('  ok - two repositories generate independently');
    console.log('  ok - two clicks on one repository still join');
    console.log('  ok - staged-ness and errors are per repository');
  } finally {
    try {
      extension.deactivate();
    } catch {}
    for (const d of ctx.subscriptions) {
      try {
        if (d && typeof d.dispose === 'function') d.dispose();
      } catch {}
    }
    Module._resolveFilename = origResolve;
  }
}

runTests()
  .then(() => {
    console.log('commitMultiRepoRouting.test.js: all tests passed');
  })
  .catch(err => {
    console.error('FAIL:', err && err.stack ? err.stack : err);
    process.exitCode = 1;
  })
  .finally(() => {
    if (daemonSock) daemonSock.destroy();
    server.close();
    fs.rmSync(tmpHome, {recursive: true, force: true});
    setTimeout(() => process.exit(process.exitCode || 0), 100).unref();
  });
