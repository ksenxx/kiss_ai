// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const fs = require('fs');
const net = require('net');
const os = require('os');
const path = require('path');
const Module = require('module');

class StubEventEmitter {
  constructor() {
    this._listeners = [];
    this.event = (cb) => {
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

let workspaceFolders = [];
const folderChangeListeners = [];

const vscodeStub = {
  workspace: {
    get workspaceFolders() {
      return workspaceFolders;
    },
    getConfiguration: () => ({get: () => 'stub-default-model'}),
    onDidChangeWorkspaceFolders: (cb) => {
      folderChangeListeners.push(cb);
      return {
        dispose: () => {
          const i = folderChangeListeners.indexOf(cb);
          if (i >= 0) folderChangeListeners.splice(i, 1);
        },
      };
    },
  },
  EventEmitter: StubEventEmitter,
  Uri: {
    file: (p) => ({fsPath: p, scheme: 'file'}),
    joinPath: (uri, ...parts) => ({
      fsPath: path.join(uri.fsPath, ...parts),
      scheme: uri.scheme || 'file',
    }),
  },
  ProgressLocation: {Notification: 15},
  window: {
    withProgress: (_opts, task) =>
      task({report: () => {}}, {
        onCancellationRequested: () => ({dispose: () => {}}),
      }),
  },
  commands: {executeCommand: () => Promise.resolve()},
};

const origResolve = Module._resolveFilename;
Module._resolveFilename = function (request, parent, ...rest) {
  if (request === 'vscode') return require.resolve('./_vscode-stub.js');
  return origResolve.call(this, request, parent, ...rest);
};

global.__kissVscodeStub = vscodeStub;

const tmpHome = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-mpr-'));
process.env.HOME = tmpHome;
process.env.USERPROFILE = tmpHome;
fs.mkdirSync(path.join(tmpHome, '.kiss'), {recursive: true});
const sockPath = path.join(tmpHome, '.kiss', 'sorcar.sock');

if (process.platform === 'win32') {
  console.log('  skipped on win32 (UDS test)');
  fs.rmSync(tmpHome, {recursive: true, force: true});
  process.exit(0);
}

const received = [];
const perConn = [];
let server = null;
let lastServerSock = null;
let lastConnIndex = -1;

function startServer() {
  return new Promise((resolve, reject) => {
    server = net.createServer((sock) => {
      lastServerSock = sock;
      const connIndex = perConn.length;
      perConn.push([]);
      lastConnIndex = connIndex;
      let buf = '';
      sock.on('data', (chunk) => {
        buf += chunk.toString();
        const lines = buf.split('\n');
        buf = lines.pop();
        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const msg = JSON.parse(line);
            received.push(msg);
            perConn[connIndex].push(msg);
          } catch (err) {
            console.error('bad json from client:', line, err);
          }
        }
      });
      sock.on('error', () => {});
    });
    server.on('error', reject);
    server.listen(sockPath, (err) => (err ? reject(err) : resolve()));
  });
}

function stopServer() {
  return new Promise((resolve) => {
    if (lastServerSock) {
      try {
        lastServerSock.destroy();
      } catch {}
      lastServerSock = null;
    }
    if (!server) return resolve();
    server.close(() => {
      try {
        fs.unlinkSync(sockPath);
      } catch {}
      server = null;
      resolve();
    });
  });
}

function waitFor(predicate, opts = {}) {
  const timeout = opts.timeout || 5000;
  const interval = opts.interval || 25;
  const start = Date.now();
  return new Promise((resolve, reject) => {
    const tick = () => {
      let ok;
      try {
        ok = predicate();
      } catch (err) {
        return reject(err);
      }
      if (ok) return resolve(ok);
      if (Date.now() - start > timeout) {
        return reject(new Error(opts.message || 'waitFor timed out'));
      }
      setTimeout(tick, interval);
    };
    tick();
  });
}

function makeStubWebviewView(extensionUri) {
  const posted = [];
  const messageListeners = [];
  const visibilityListeners = [];
  const webview = {
    options: {},
    html: '',
    cspSource: 'vscode-webview://stub',
    asWebviewUri: (uri) => ({toString: () => `vscode-webview://${uri.fsPath}`}),
    postMessage: (msg) => {
      posted.push(msg);
      return Promise.resolve(true);
    },
    onDidReceiveMessage: (cb) => {
      messageListeners.push(cb);
      return {dispose: () => {}};
    },
  };
  const view = {
    webview,
    visible: true,
    show: () => {},
    onDidDispose: () => ({dispose: () => {}}),
    onDidChangeVisibility: (cb) => {
      visibilityListeners.push(cb);
      return {dispose: () => {}};
    },
  };
  return {
    view,
    posted,
    extensionUri,
    fireMessage: (m) => {
      for (const cb of messageListeners.slice()) cb(m);
    },
  };
}

async function runTests() {
  const projectRoot = path.resolve(__dirname, '..');
  const extensionUri = {fsPath: projectRoot, scheme: 'file'};

  const sourcePath = path.join(projectRoot, 'out', 'SorcarSidebarView.js');
  assert.ok(
    fs.existsSync(sourcePath),
    `compiled extension missing: ${sourcePath} — run \`tsc -p .\` first`,
  );
  delete require.cache[require.resolve(sourcePath)];
  const {SorcarSidebarView} = require(sourcePath);

  let passed = 0;
  let failed = 0;
  const ok = (msg) => {
    passed += 1;
    console.log('  ok -', msg);
  };
  const fail = (msg, err) => {
    failed += 1;
    console.error('  FAIL -', msg);
    if (err) console.error('       ', err.message || err);
  };

  await startServer();

  const ws = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-mpr-ws-'));
  workspaceFolders = [{uri: {fsPath: ws, scheme: 'file'}}];

  const view = new SorcarSidebarView(extensionUri);
  view.syncWorkDir();

  const stub = makeStubWebviewView(extensionUri);
  view.resolveWebviewView(
    stub.view,
    {state: undefined},
    {
      isCancellationRequested: false,
      onCancellationRequested: () => ({dispose: () => {}}),
    },
  );

  try {
    await waitFor(
      () => received.some((m) => m.type === 'setWorkDir'),
      {message: 'no initial setWorkDir reached server'},
    );
    ok('initial connect sends setWorkDir');
  } catch (err) {
    fail('initial setWorkDir', err);
  }

  stub.fireMessage({type: 'ready', tabId: 'tab-1'});

  const INIT_TYPES = ['getModels', 'getInputHistory', 'getConfig'];
  try {
    await waitFor(
      () => INIT_TYPES.every((t) => received.some((m) => m.type === t)),
      {
        message:
          'ready handler did not dispatch all of getModels/getInputHistory/getConfig',
      },
    );
    ok('ready handler dispatches getModels + getInputHistory + getConfig');
  } catch (err) {
    fail('ready handler init commands', err);
  }

  const initialCounts = {};
  for (const t of [...INIT_TYPES, 'setWorkDir']) {
    initialCounts[t] = received.filter((m) => m.type === t).length;
  }

  await stopServer();

  await new Promise((res) => setTimeout(res, 100));

  await startServer();

  try {
    await waitFor(
      () => {
        const seen = received.filter((m) => m.type === 'setWorkDir').length;
        return seen > initialCounts.setWorkDir;
      },
      {timeout: 8000, message: 'reconnect did not re-send setWorkDir'},
    );
    ok('reconnect re-sends setWorkDir');
  } catch (err) {
    fail('reconnect re-send setWorkDir', err);
  }

  for (const t of INIT_TYPES) {
    try {
      await waitFor(
        () => received.filter((m) => m.type === t).length > initialCounts[t],
        {
          timeout: 8000,
          message: `reconnect did not re-send ${t} — model picker / settings panel would stay blank after a daemon restart`,
        },
      );
      ok(`reconnect re-sends ${t}`);
    } catch (err) {
      fail(`reconnect re-send ${t}`, err);
    }
  }

  const view2 = new SorcarSidebarView(extensionUri);
  view2.syncWorkDir();
  const view2InitialConnIndex = perConn.length;
  try {
    await waitFor(
      () =>
        perConn.length > view2InitialConnIndex &&
        perConn[view2InitialConnIndex].some((m) => m.type === 'setWorkDir'),
      {message: 'view2 initial setWorkDir never reached server'},
    );
    ok('view2 initial connect sends setWorkDir');
  } catch (err) {
    fail('view2 initial setWorkDir', err);
  }
  if (lastServerSock) {
    lastServerSock.destroy();
    lastServerSock = null;
  }
  const view2ReconnectIndex = perConn.length;
  try {
    await waitFor(
      () =>
        perConn.length > view2ReconnectIndex &&
        perConn[view2ReconnectIndex].some((m) => m.type === 'setWorkDir'),
      {
        timeout: 8000,
        message: 'view2 reconnect did not send setWorkDir',
      },
    );
    ok('view2 reconnect (no resolved webview) sends setWorkDir');
  } catch (err) {
    fail('view2 reconnect setWorkDir', err);
  }
  await new Promise((res) => setTimeout(res, 200));
  const view2ReconnectMsgs = perConn[view2ReconnectIndex] || [];
  const leaked = view2ReconnectMsgs
    .map((m) => m.type)
    .filter((t) => INIT_TYPES.includes(t));
  try {
    assert.deepStrictEqual(
      leaked,
      [],
      `view2 (no resolved webview) leaked init commands on reconnect: ${leaked.join(', ')}`,
    );
    ok('view2 reconnect does NOT spray getModels/getInputHistory/getConfig');
  } catch (err) {
    fail('view2 reconnect init-command gating', err);
  }

  if (typeof view.dispose === 'function') view.dispose();
  if (typeof view2.dispose === 'function') view2.dispose();
  fs.rmSync(ws, {recursive: true, force: true});
  await stopServer();

  console.log(`\n${passed} passed, ${failed} failed`);
  if (failed > 0) process.exit(1);
}

runTests().then(
  () => {
    fs.rmSync(tmpHome, {recursive: true, force: true});
    process.exit(0);
  },
  async (err) => {
    console.error('FAIL:', err);
    await stopServer().catch(() => {});
    fs.rmSync(tmpHome, {recursive: true, force: true});
    process.exit(1);
  },
);
