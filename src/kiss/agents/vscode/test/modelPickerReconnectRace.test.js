// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end regression test for the "model picker blank on VS Code
// launch" race.
//
// Bug being locked in:
//
//   When VS Code launches, the dependency installer often restarts the
//   kiss-web daemon (DependencyInstaller.runFinalization()).  Each VS
//   Code window owns ONE UDS connection to the daemon via AgentClient,
//   and the daemon keeps every per-connection state (work_dir,
//   conn_id, endpoint binding) tied to that socket.  When the socket
//   drops mid-flight — e.g. the user opened the sidebar, the webview
//   posted ``ready``, the extension forwarded ``getModels`` to the
//   daemon, and the daemon was killed before its ``models`` event
//   reached the client — AgentClient transparently reconnects, but
//   the extension's ``on('connect', ...)`` preamble only re-sent
//   ``setWorkDir``.  The ``getModels``, ``getInputHistory`` and
//   ``getConfig`` requests that the ``ready`` handler had already
//   dispatched are NOT re-issued, so:
//
//     * the webview's ``allModels`` stays empty,
//     * ``modelName.textContent`` is whatever the HTML template
//       substituted at first paint (often ``"No model"`` or blank),
//     * the welcome panel's settings form (filled by ``configData``)
//       stays empty.
//
//   The fix:
//     * ``SorcarSidebarView._getClient()``'s on-connect listener now
//       re-issues ``getModels``, ``getInputHistory`` and ``getConfig``
//       in addition to ``setWorkDir`` whenever the daemon socket
//       (re)connects AND a webview has been resolved (i.e. there is a
//       UI that depends on those replies).
//
// This test exercises the real compiled ``SorcarSidebarView.js`` and
// ``AgentClient.js`` (no mocks of project code) — only the ``vscode``
// module is stubbed and the daemon UDS endpoint is replaced by a real
// in-process socket server we can stop / restart on demand to
// simulate a daemon restart.
//
// Run directly with ``node``:
//
//     node src/kiss/agents/vscode/test/modelPickerReconnectRace.test.js

'use strict';

const assert = require('assert');
const fs = require('fs');
const net = require('net');
const os = require('os');
const path = require('path');
const Module = require('module');

// ---------------------------------------------------------------------------
// Stub the ``vscode`` module before any project source loads it.
// ---------------------------------------------------------------------------

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

// ``_vscode-stub.js`` is a git-tracked fixture shared by tests running
// in parallel; it already re-exports ``global.__kissVscodeStub`` — never
// rewrite or delete it here (writeFileSync truncates first, racing a
// concurrent ``require('vscode')`` in sibling test processes).
global.__kissVscodeStub = vscodeStub;

// ---------------------------------------------------------------------------
// Redirect HOME so AgentClient connects to our test UDS socket.
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// A throwaway UDS server that parses JSON lines into a shared array.
// We can stop it (simulating a daemon kill) and restart it on the same
// socket path (simulating daemon respawn).
// ---------------------------------------------------------------------------

const received = [];
// Per-connection received messages, keyed by a monotonically growing
// connection index so we can write per-socket assertions (e.g. the
// negative invariant that view2's connection never sends getModels).
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

// ---------------------------------------------------------------------------
// Build a fake VS Code WebviewView the sidebar provider can resolve.
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

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

  // Start the UDS server BEFORE constructing the sidebar view so the
  // first connect attempt succeeds immediately (mirroring the
  // realistic "daemon already up" startup path).
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

  // Wait for the first ``setWorkDir`` so we know the AgentClient is
  // attached to the server before we fire ``ready``.
  try {
    await waitFor(
      () => received.some((m) => m.type === 'setWorkDir'),
      {message: 'no initial setWorkDir reached server'},
    );
    ok('initial connect sends setWorkDir');
  } catch (err) {
    fail('initial setWorkDir', err);
  }

  // ------------------------------------------------------------------
  // Simulate the webview's init() posting ``ready``.  The extension's
  // _handleMessage('ready') handler fans out into getModels +
  // getInputHistory + getConfig (plus setWorkDir is already in flight
  // from the on-connect preamble).
  // ------------------------------------------------------------------
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

  // Snapshot count of each init type so we can detect the *second*
  // occurrence after reconnect.
  const initialCounts = {};
  for (const t of [...INIT_TYPES, 'setWorkDir']) {
    initialCounts[t] = received.filter((m) => m.type === t).length;
  }

  // ------------------------------------------------------------------
  // Simulate a daemon restart: kill the server, wait for AgentClient
  // to notice the drop, then bring the server back up on the same
  // path.  AgentClient's auto-reconnect loop will reconnect.
  // ------------------------------------------------------------------
  await stopServer();

  // Give AgentClient a moment to register the disconnect (its socket
  // 'close' fires on the next tick) before restarting the server.
  await new Promise((res) => setTimeout(res, 100));

  await startServer();

  // ------------------------------------------------------------------
  // After reconnect, the on('connect', ...) preamble MUST re-issue
  // every init command the webview depends on — not just setWorkDir.
  // Without the fix, only setWorkDir is re-sent and the model picker
  // stays blank because the original ``models`` event never reached
  // the webview (it was lost when the socket dropped mid-flight).
  // ------------------------------------------------------------------
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

  // ------------------------------------------------------------------
  // Negative invariant: the re-issue must be GATED on a resolved
  // webview.  A reconnect that happens BEFORE the user has opened the
  // sidebar (no webview resolved yet) must not spray
  // ``getModels`` / ``getInputHistory`` / ``getConfig`` into the
  // daemon — otherwise an idle window the user never interacted with
  // would still pull models and configData on every daemon restart.
  //
  // Spin up a SECOND sidebar view whose webview is NEVER resolved,
  // wait for its initial connect, then drop ITS socket and assert
  // the reconnect re-sends only ``setWorkDir`` — no init commands.
  // The per-connection bucket lets us inspect view2's socket(s) in
  // isolation from view's traffic on the shared UDS path.
  // ------------------------------------------------------------------
  const view2 = new SorcarSidebarView(extensionUri);
  view2.syncWorkDir();
  // Wait for view2's initial connection to be accepted by the server.
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
  // Drop view2's socket (the most recently-accepted one).
  if (lastServerSock) {
    lastServerSock.destroy();
    lastServerSock = null;
  }
  // Wait for the reconnect's setWorkDir to confirm view2 actually
  // reconnected, then snapshot its per-connection messages.
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
  // Give the reconnect handler a beat to flush any (buggy) extra
  // init commands before asserting their absence.
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
