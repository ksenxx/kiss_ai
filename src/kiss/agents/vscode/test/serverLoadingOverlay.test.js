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

const tmpHome = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-srvload-'));
process.env.HOME = tmpHome;
process.env.USERPROFILE = tmpHome;
fs.mkdirSync(path.join(tmpHome, '.kiss'), {recursive: true});
const sockPath = path.join(tmpHome, '.kiss', 'sorcar.sock');

if (process.platform === 'win32') {
  console.log('  skipped on win32 (UDS test)');
  fs.rmSync(tmpHome, {recursive: true, force: true});
  process.exit(0);
}

let server = null;
let lastServerSock = null;

function startServer() {
  return new Promise((resolve, reject) => {
    server = net.createServer((sock) => {
      lastServerSock = sock;
      sock.on('data', () => {});
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
  const fail = (msg, err) => {
    failed += 1;
    console.error('  FAIL -', msg);
    if (err) console.error('       ', err.message || err);
  };
  const ok = (msg) => {
    passed += 1;
    console.log('  ok -', msg);
  };

  const wsA = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-ws-overlay-'));
  workspaceFolders = [{uri: {fsPath: wsA, scheme: 'file'}}];

  const view = new SorcarSidebarView(extensionUri);
  const stub = makeStubWebviewView(extensionUri);
  view.resolveWebviewView(
    stub.view,
    {state: undefined},
    {isCancellationRequested: false, onCancellationRequested: () => ({dispose: () => {}})},
  );

  const html = stub.view.webview.html;
  try {
    assert.ok(
      /id="kiss-server-loading"/.test(html),
      'chat.html must render the loading overlay element',
    );
    assert.ok(
      /KISS Sorcar Server is starting \.\.\./.test(html),
      'overlay must contain the "KISS Sorcar Server is starting ..." message',
    );
    assert.ok(
      /<div id="app" style="display:none;?"/.test(html),
      '#app must start hidden so the overlay is what the user sees',
    );
    ok('chat.html paints overlay over a hidden #app on first load');
  } catch (err) {
    fail('chat.html initial overlay assertions', err);
  }

  stub.posted.length = 0;
  stub.fireMessage({type: 'ready', tabId: 't1'});

  try {
    await waitFor(
      () =>
        stub.posted.some(
          (m) => m && m.type === 'daemonStatus' && m.connected === false,
        ),
      {message: 'no daemonStatus(connected:false) posted while daemon down'},
    );
    ok('ready posts daemonStatus(connected:false) when daemon is unreachable');
  } catch (err) {
    fail('ready -> daemonStatus(false) while daemon down', err);
  }

  await startServer();

  try {
    await waitFor(
      () =>
        stub.posted.some(
          (m) => m && m.type === 'daemonStatus' && m.connected === true,
        ),
      {
        timeout: 8000,
        message: 'no daemonStatus(connected:true) after daemon started',
      },
    );
    ok('connect posts daemonStatus(connected:true), revealing #app');
  } catch (err) {
    fail('connect -> daemonStatus(true)', err);
  }

  stub.posted.length = 0;
  if (lastServerSock) {
    lastServerSock.destroy();
    lastServerSock = null;
  }

  try {
    await waitFor(
      () =>
        stub.posted.some(
          (m) => m && m.type === 'daemonStatus' && m.connected === false,
        ),
      {message: 'no daemonStatus(connected:false) after socket dropped'},
    );
    ok('disconnect posts daemonStatus(connected:false) after socket drop');
  } catch (err) {
    fail('disconnect -> daemonStatus(false)', err);
  }

  stub.posted.length = 0;

  try {
    await waitFor(
      () =>
        stub.posted.some(
          (m) => m && m.type === 'daemonStatus' && m.connected === true,
        ),
      {
        timeout: 8000,
        message: 'no daemonStatus(connected:true) after auto-reconnect',
      },
    );
    ok('auto-reconnect posts daemonStatus(connected:true) again');
  } catch (err) {
    fail('reconnect -> daemonStatus(true)', err);
  }

  if (typeof view.dispose === 'function') view.dispose();
  await stopServer();
  fs.rmSync(wsA, {recursive: true, force: true});

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
