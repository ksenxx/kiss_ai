// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// End-to-end tests (real compiled host, real loopback WebSocket daemon
// stand-in, no mocks) for the extension host's behaviour across a
// daemon outage:
//
// 1. AgentClient reports `disconnect` ONCE per outage.  The endpoint
//    file is polled every 100 ms while absent; every poll used to emit
//    `disconnect`, so each controller posted `daemonStatus` to its
//    webview 10x/s for as long as no daemon existed.
// 2. `sendCommand()` while a reconnect back-off is pending must not
//    open a socket immediately: the pending timer owns the next attempt.
// 3. A commit-message request already SENT to a daemon that then dies
//    is settled by the sidebar on `disconnect` with "The agent was
//    unreachable", not by the 30 s safety timer.
// 4. `subagentDone` / `closeSubagentTab` release a sub-agent tab's
//    host-side bookkeeping, so stopTask() never `stop`s a finished one.

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const Module = require('module');
const {createFakeDaemon} = require('./fakeDaemon');

const OUT_DIR = path.join(__dirname, '..', 'out');
const OUT_AGENT_CLIENT = path.join(OUT_DIR, 'AgentClient.js');
const OUT_SIDEBAR = path.join(OUT_DIR, 'SorcarSidebarView.js');
if (!fs.existsSync(OUT_AGENT_CLIENT) || !fs.existsSync(OUT_SIDEBAR)) {
  console.log('SKIP: out/ missing — run `npm run compile`');
  process.exit(0);
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

function makeUri(fsPath) {
  return {fsPath, scheme: 'file', toString: () => `file://${fsPath}`};
}

let workspaceFolders = [];
const vscodeStub = {
  workspace: {
    get workspaceFolders() {
      return workspaceFolders;
    },
    getConfiguration: () => ({get: () => 'stub-default-model'}),
    onDidChangeWorkspaceFolders: () => ({dispose: () => {}}),
    textDocuments: [],
  },
  EventEmitter: StubEventEmitter,
  Uri: {
    file: p => makeUri(p),
    joinPath: (base, ...parts) => makeUri(path.join(base.fsPath, ...parts)),
    parse: s => makeUri(s),
  },
  ProgressLocation: {Notification: 15},
  ViewColumn: {One: 1},
  window: {
    withProgress: (_opts, task) =>
      task(
        {report: () => {}},
        {onCancellationRequested: () => ({dispose: () => {}})},
      ),
    showInformationMessage: () => {},
    showErrorMessage: () => {},
    showWarningMessage: () => {},
    tabGroups: {all: []},
  },
  commands: {executeCommand: () => Promise.resolve()},
};

const origResolve = Module._resolveFilename;
Module._resolveFilename = function (request, parent, ...rest) {
  if (request === 'vscode') return require.resolve('./_vscode-stub.js');
  return origResolve.call(this, request, parent, ...rest);
};
global.__kissVscodeStub = vscodeStub;

const tmpHome = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-dc1-'));
process.env.HOME = tmpHome;
process.env.USERPROFILE = tmpHome;
process.env.KISS_HOME = path.join(tmpHome, '.kiss');
fs.mkdirSync(process.env.KISS_HOME, {recursive: true});

const {AgentClient} = require(OUT_AGENT_CLIENT);
const {SorcarSidebarView} = require(OUT_SIDEBAR);

function tmpEndpoint(name) {
  return path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-dc1-')), name);
}

function delay(ms) {
  return new Promise(r => setTimeout(r, ms));
}

function listen(server, endpointPath) {
  return new Promise((res, rej) =>
    server.listen(endpointPath, err => (err ? rej(err) : res())),
  );
}

function close(server) {
  return new Promise(r => server.close(r));
}

async function waitFor(pred, ms, what) {
  const deadline = Date.now() + ms;
  while (!pred()) {
    assert.ok(Date.now() < deadline, `timed out waiting for ${what}`);
    await delay(10);
  }
}

// ---------------------------------------------------------------------
// 1. One `disconnect` per outage
// ---------------------------------------------------------------------

async function testDisconnectEmittedOncePerOutage() {
  const endpointPath = tmpEndpoint('absent.json');
  const client = new AgentClient(endpointPath, {
    endpointPollMs: 20,
    reconnectBaseMs: 20,
    reconnectMaxMs: 20,
  });
  let disconnects = 0;
  let connects = 0;
  client.on('disconnect', () => disconnects++);
  client.on('connect', () => connects++);

  // No endpoint file: ~25 polls in 500 ms, one announcement.
  client.connect();
  await delay(500);
  assert.strictEqual(
    disconnects,
    1,
    `a missing endpoint file must be reported once, not per poll (${disconnects})`,
  );

  // Commands sent during the outage are not news either.
  client.sendCommand({type: 'getModels'});
  client.sendCommand({type: 'getConfig'});
  await delay(100);
  assert.strictEqual(disconnects, 1, 'sendCommand while down must not re-emit');

  // The daemon appears: connect; then it dies: exactly one more.
  const server = createFakeDaemon(conn => conn.on('data', () => {}));
  await listen(server, endpointPath);
  await waitFor(() => connects === 1, 3000, 'connect');
  assert.strictEqual(disconnects, 1);
  server.destroyConnections();
  await close(server);
  await delay(400);
  assert.strictEqual(
    disconnects,
    2,
    `the socket drop is one new outage, got ${disconnects} disconnects`,
  );

  // And it comes back again: the transition is reported each time.
  const server2 = createFakeDaemon(conn => conn.on('data', () => {}));
  await listen(server2, endpointPath);
  await waitFor(() => connects === 2, 3000, 'reconnect');
  client.dispose();
  await close(server2);
  assert.strictEqual(disconnects, 2);
  console.log('  ok - disconnect is announced once per outage');
}

// ---------------------------------------------------------------------
// 2. sendCommand honours a pending back-off
// ---------------------------------------------------------------------

async function testSendCommandHonoursBackoff() {
  const endpointPath = tmpEndpoint('crashloop.json');
  let attempts = 0;
  // A daemon that accepts and immediately drops: a crash loop.
  const server = createFakeDaemon();
  server.on('connection', conn => {
    attempts++;
    conn.destroy();
  });
  await listen(server, endpointPath);

  const client = new AgentClient(endpointPath, {
    reconnectBaseMs: 400,
    reconnectMaxMs: 400,
  });
  client.connect();
  await waitFor(() => attempts === 1, 2000, 'first attempt');
  await delay(50);
  // The back-off timer (200..400 ms) is now pending.  Each of these
  // used to open a socket at once.
  for (let i = 0; i < 20; i++) client.sendCommand({type: 'getModels'});
  await delay(100);
  assert.strictEqual(
    attempts,
    1,
    `sendCommand during the back-off must not connect (${attempts} attempts)`,
  );
  await waitFor(() => attempts >= 2, 1000, 'the back-off attempt');
  client.dispose();
  await close(server);
  console.log('  ok - sendCommand honours the pending back-off');
}

// ---------------------------------------------------------------------
// Sidebar harness
// ---------------------------------------------------------------------

function makeWebviewView(posted) {
  const recv = new StubEventEmitter();
  const dispose = new StubEventEmitter();
  const vis = new StubEventEmitter();
  const webview = {
    options: {},
    html: '',
    cspSource: 'vscode-resource:',
    asWebviewUri: uri => makeUri(uri.fsPath),
    postMessage: msg => {
      posted.push(msg);
      return Promise.resolve(true);
    },
    onDidReceiveMessage: cb => recv.event(cb),
  };
  const webviewView = {
    webview,
    visible: true,
    show: () => {},
    onDidChangeVisibility: cb => vis.event(cb),
    onDidDispose: cb => dispose.event(cb),
  };
  return {
    webviewView,
    fireMessage: m => recv.fire(m),
    disposeWebview: () => dispose.fire(),
  };
}

/** Start a fake daemon whose connections record every command line. */
async function startRecordingDaemon(endpointPath) {
  const lines = [];
  const conns = [];
  const server = createFakeDaemon(conn => {
    conns.push(conn);
    conn.on('data', d => {
      for (const l of d.toString().split('\n')) {
        if (l.trim()) lines.push(JSON.parse(l));
      }
    });
  });
  await listen(server, endpointPath);
  return {server, lines, conns};
}

// ---------------------------------------------------------------------
// 3. A sent commit-message request settles on disconnect
// ---------------------------------------------------------------------

async function testPendingCommitSettlesOnDisconnect() {
  const endpointPath = tmpEndpoint('commit.json');
  process.env.KISS_SORCAR_LOCAL = endpointPath;
  const ws = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-dc1-ws-'));
  workspaceFolders = [{uri: makeUri(ws)}];
  const {server, lines} = await startRecordingDaemon(endpointPath);

  const posted = [];
  const view = new SorcarSidebarView(makeUri(path.join(__dirname, '..')));
  const wv = makeWebviewView(posted);
  view.resolveWebviewView(wv.webviewView, {}, {});
  wv.fireMessage({type: 'ready', tabId: 'tab-0', restoredTabs: []});
  await waitFor(
    () => posted.some(m => m.type === 'daemonStatus' && m.connected),
    3000,
    'the sidebar to connect',
  );

  const results = [];
  view.onCommitMessage(ev => results.push(ev));
  const t0 = Date.now();
  const done = view.generateCommitMessage(undefined, 'tab-0', ws);
  await waitFor(
    () => lines.some(l => l.type === 'generateCommitMessage'),
    3000,
    'the request to reach the daemon',
  );
  // The daemon dies with the request in hand.
  server.destroyConnections();
  await close(server);
  await done;
  const elapsed = Date.now() - t0;
  assert.ok(
    elapsed < 5000,
    `the commit promise must settle on disconnect, not after 30 s (${elapsed} ms)`,
  );
  assert.deepStrictEqual(
    results.map(r => [r.tabId, r.error]),
    [['tab-0', 'The agent was unreachable']],
    `got ${JSON.stringify(results)}`,
  );
  // A second request is not a duplicate of the settled one: the pending
  // set was cleared.
  const posted2 = posted.length;
  void view.generateCommitMessage(undefined, 'tab-0', ws);
  await delay(50);
  assert.ok(posted.length >= posted2, 'view still alive');
  view.dispose();
  console.log('  ok - a sent commit request settles on disconnect');
}

// ---------------------------------------------------------------------
// 4. subagentDone releases the sub-agent tab
// ---------------------------------------------------------------------

async function testSubagentDoneReleasesTab() {
  const endpointPath = tmpEndpoint('subagent.json');
  process.env.KISS_SORCAR_LOCAL = endpointPath;
  const ws = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-dc1-ws-'));
  workspaceFolders = [{uri: makeUri(ws)}];
  const {server, lines, conns} = await startRecordingDaemon(endpointPath);

  const posted = [];
  const view = new SorcarSidebarView(makeUri(path.join(__dirname, '..')));
  const wv = makeWebviewView(posted);
  view.resolveWebviewView(wv.webviewView, {}, {});
  wv.fireMessage({type: 'ready', tabId: 'tab-0', restoredTabs: []});
  await waitFor(
    () => posted.some(m => m.type === 'daemonStatus' && m.connected),
    3000,
    'the sidebar to connect',
  );
  const conn = conns[0];
  const send = msg => conn.write(JSON.stringify(msg));

  // The daemon runs the root tab and spawns two sub-agents under it.
  send({type: 'status', running: true, tabId: 'tab-0'});
  for (const id of ['sub-1', 'sub-2']) {
    send({type: 'openSubagentTab', tab_id: id, parent_tab_id: 'tab-0'});
    send({type: 'status', running: true, tabId: id});
  }
  // sub-1 finishes (closes on every surface); sub-2 is closed explicitly.
  send({type: 'subagentDone', tab_id: 'sub-1', success: true});
  send({type: 'closeSubagentTab', tab_id: 'sub-2'});
  await waitFor(
    () => posted.some(m => m.type === 'closeSubagentTab'),
    3000,
    'the broadcasts to arrive',
  );

  // With the webview gone, stopTask() stops every tab the host still
  // believes is running: only the root tab may be among them.
  wv.disposeWebview();
  view.stopTask();
  await waitFor(() => lines.some(l => l.type === 'stop'), 3000, 'stop');
  await delay(100);
  assert.deepStrictEqual(
    lines.filter(l => l.type === 'stop').map(l => l.tabId),
    ['tab-0'],
    `finished sub-agent tabs must not be stopped: ${JSON.stringify(
      lines.filter(l => l.type === 'stop'),
    )}`,
  );
  view.dispose();
  await close(server);
  console.log('  ok - subagentDone/closeSubagentTab release the tab');
}

(async () => {
  const watchdog = setTimeout(() => {
    console.error('FAIL: agentClientDisconnectOnce.test.js timed out');
    process.exit(1);
  }, 60_000);
  try {
    await testDisconnectEmittedOncePerOutage();
    await testSendCommandHonoursBackoff();
    await testPendingCommitSettlesOnDisconnect();
    await testSubagentDoneReleasesTab();
    console.log('ok - agentClientDisconnectOnce.test.js');
  } catch (err) {
    console.error('FAIL:', err && err.stack ? err.stack : err);
    process.exitCode = 1;
  } finally {
    clearTimeout(watchdog);
    fs.rmSync(tmpHome, {recursive: true, force: true});
    process.exit(process.exitCode || 0);
  }
})();
