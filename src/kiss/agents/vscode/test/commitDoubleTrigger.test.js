// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// End-to-end test of a SECOND commit-message generation arriving while
// the first is still in flight -- a double click on the SCM sparkle, or
// another extension invoking one of the two ids this extension hijacks
// (github.copilot.git.generateCommitMessage / git.generateCommitMessage,
// both bound to the same handler with the same empty tabId).
//
// The sidebar de-duplicates per tab by returning an ALREADY-RESOLVED
// promise, meaning "someone else owns this generation". extension.ts
// treated every returned promise as "my generation finished", so the
// second call's .finally immediately cleared the in-flight flag, stopped
// the countdown and blanked the SCM input -- and when the daemon's real
// commit message arrived seconds later it was dropped on the floor. The
// user was left with an empty box, no countdown and no error.
//
// The real compiled extension and the real SorcarSidebarView are driven
// here; the daemon is a real unix domain socket speaking the real line
// protocol. Only the VS Code API surface and the git extension -- which
// do not exist outside VS Code -- are stubbed.

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

const tmpHome = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-commit2-'));
fs.mkdirSync(path.join(tmpHome, '.kiss'), {recursive: true});
const sockPath = path.join(tmpHome, '.kiss', 'sorcar.sock');
process.env.HOME = tmpHome;
process.env.USERPROFILE = tmpHome;
process.env.KISS_HOME = path.join(tmpHome, '.kiss');
process.env.KISS_SORCAR_SOCK = sockPath;

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
const workspaceFolders = [{uri: makeUri(tmpHome)}];

const vscodeStub = {
  window: {
    registerWebviewViewProvider: () => makeDisposable(),
    createTreeView: () => ({
      onDidChangeVisibility: () => makeDisposable(),
      dispose: () => {},
    }),
    withProgress: (_opts, task) =>
      task({report: () => {}}, {onCancellationRequested: () => makeDisposable()}),
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

const inputBox = {value: ''};
const fakeGitApi = {
  repositories: [
    {
      rootUri: makeUri(tmpHome),
      inputBox,
      state: {indexChanges: [{}]},
    },
  ],
};

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

function sendFromDaemon(obj) {
  assert.ok(daemonSock, 'the extension must have connected to the daemon');
  daemonSock.write(JSON.stringify(obj) + '\n');
}

// The daemon stamps the requesting tabId on every commitMessage it
// broadcasts, so the printer routes the answer back to the tab that
// asked for it (server.py::_generate_commit_message).  Answering with
// anything else would be testing a daemon that does not exist.
function answerLastGeneration(payload) {
  const asked = daemonLines.filter(m => m.type === 'generateCommitMessage');
  assert.ok(asked.length, 'nothing asked the daemon for a commit message');
  sendFromDaemon({
    type: 'commitMessage',
    tabId: asked[asked.length - 1].tabId,
    ...payload,
  });
}

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

async function waitFor(predicate, message, timeoutMs = 4000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    const v = predicate();
    if (v) return v;
    await sleep(10);
  }
  throw new Error(message || 'waitFor timed out');
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

    // --- the double trigger -------------------------------------------
    daemonLines.length = 0;
    inputBox.value = '';
    const p1 = trigger(makeUri(tmpHome), undefined, undefined);
    const p2 = trigger(makeUri(tmpHome), undefined, undefined);

    await waitFor(
      () => daemonLines.filter(m => m.type === 'generateCommitMessage').length,
      'the extension must ask the daemon for a commit message',
    );
    assert.strictEqual(
      daemonLines.filter(m => m.type === 'generateCommitMessage').length,
      1,
      'a second trigger must join the generation already in flight, not ' +
        'start a competing one',
    );

    // The daemon answers a few moments later, as a real LLM call would.
    await sleep(120);
    answerLastGeneration({message: 'feat: real message'});
    await Promise.all([p1, p2]);
    await sleep(150);

    assert.strictEqual(
      inputBox.value,
      'feat: real message',
      'the generated commit message must land in the SCM input box; the ' +
        "second trigger's teardown must not discard it",
    );

    // --- the single-click control case --------------------------------
    daemonLines.length = 0;
    inputBox.value = '';
    const p3 = trigger(makeUri(tmpHome), undefined, undefined);
    await waitFor(
      () => daemonLines.some(m => m.type === 'generateCommitMessage'),
      'a fresh generation must reach the daemon after the first finished',
    );
    answerLastGeneration({message: 'fix: single click'});
    await p3;
    await sleep(150);
    assert.strictEqual(
      inputBox.value,
      'fix: single click',
      'a single generation must still work',
    );

    // --- an error is still reported, and does not wedge the next one ---
    daemonLines.length = 0;
    inputBox.value = '';
    const p4 = trigger(makeUri(tmpHome), undefined, undefined);
    await waitFor(
      () => daemonLines.some(m => m.type === 'generateCommitMessage'),
      'a generation must reach the daemon',
    );
    answerLastGeneration({error: 'no staged changes'});
    await p4;
    await sleep(150);
    assert.strictEqual(
      inputBox.value,
      '',
      'an error must clear the countdown text rather than leave it',
    );

    daemonLines.length = 0;
    const p5 = trigger(makeUri(tmpHome), undefined, undefined);
    await waitFor(
      () => daemonLines.some(m => m.type === 'generateCommitMessage'),
      'a generation after an error must still reach the daemon',
    );
    answerLastGeneration({message: 'chore: after error'});
    await p5;
    await sleep(150);
    assert.strictEqual(
      inputBox.value,
      'chore: after error',
      'an error must not wedge later generations',
    );

    console.log('  ok - a second trigger joins the generation in flight');
    console.log('  ok - single generation, error, and recovery still work');
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
    console.log('commitDoubleTrigger.test.js: all tests passed');
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
