// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// End-to-end tests for a run the daemon never received.
//
// Sending a prompt is optimistic: the tab is shown as running -- spinner
// on, composer locked -- the moment the user presses enter, before any
// daemon has confirmed a thing.  The only thing that ever turns that
// back off is a `status running:false`, and the only thing that sends
// one is the daemon.
//
// AgentClient may legitimately decide never to deliver a queued
// command: one queued against a daemon that then died must not be
// replayed into the DIFFERENT daemon that answers ten seconds later (it
// would start an agent nobody asked for), and the queue is bounded so a
// long outage cannot grow it without limit.  Both were silent, so the
// tab that had already been marked running stayed running for ever,
// with no agent behind it and no way to type in it.
//
// Two levels are covered, both for real:
//   1. the REAL compiled AgentClient over a REAL unix domain socket:
//      dropping a command must be announced;
//   2. the REAL compiled SorcarSidebarView driving a webview: the
//      announcement must actually put the tab back.

const assert = require('assert');
const fs = require('fs');
const net = require('net');
const os = require('os');
const path = require('path');
const Module = require('module');

const OUT_DIR = path.join(__dirname, '..', 'out');
const OUT_AGENT_CLIENT = path.join(OUT_DIR, 'AgentClient.js');
const OUT_SIDEBAR = path.join(OUT_DIR, 'SorcarSidebarView.js');

if (!fs.existsSync(OUT_AGENT_CLIENT) || !fs.existsSync(OUT_SIDEBAR)) {
  console.log('SKIP: out/ missing — run `npm run compile`');
  process.exit(0);
}
if (process.platform === 'win32') {
  console.log('SKIP: UDS tests require a POSIX platform');
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
    openTextDocument: () =>
      Promise.resolve({uri: makeUri('/x'), getText: () => ''}),
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
    showTextDocument: () => Promise.resolve({}),
    activeTextEditor: undefined,
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

const tmpHome = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-drop-'));
process.env.HOME = tmpHome;
process.env.USERPROFILE = tmpHome;
fs.mkdirSync(path.join(tmpHome, '.kiss'), {recursive: true});

// The sidebar must find no daemon at all: that is the whole scenario.
const deadSock = path.join(tmpHome, '.kiss', 'nothing-here.sock');
process.env.KISS_SORCAR_SOCK = deadSock;

const {AgentClient} = require(OUT_AGENT_CLIENT);
const {SorcarSidebarView} = require(OUT_SIDEBAR);

const tmpDirs = [tmpHome];

function tmpSock(name) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-drop-sock-'));
  tmpDirs.push(dir);
  return path.join(dir, name);
}

function delay(ms) {
  return new Promise(r => setTimeout(r, ms));
}

function listen(server, sockPath) {
  return new Promise((res, rej) =>
    server.listen(sockPath, err => (err ? rej(err) : res())),
  );
}

function close(server) {
  return new Promise(r => server.close(r));
}

// ---------------------------------------------------------------------
// 1. AgentClient must say when it gives up on a command
// ---------------------------------------------------------------------

async function testExpiredCommandIsAnnounced() {
  const sockPath = tmpSock('expired.sock');
  const client = new AgentClient(sockPath, {
    reconnectBaseMs: 40,
    reconnectMaxMs: 120,
    pendingTtlMs: 250,
  });
  const dropped = [];
  client.on('commandDropped', (cmd, reason) => dropped.push({cmd, reason}));

  // Nothing is listening: the frame is queued and then goes stale.
  client.sendCommand({type: 'run', prompt: 'do the thing', tabId: 't1'});
  await delay(600);

  const received = [];
  const server = net.createServer(conn => {
    conn.on('data', d => received.push(d.toString()));
  });
  await listen(server, sockPath);
  await new Promise(resolve => {
    client.on('connect', resolve);
    client.connect();
  });
  await delay(150);
  client.dispose();
  await close(server);

  assert.ok(
    !received.join('').includes('"type":"run"'),
    'the stale run must still not be replayed into the new daemon',
  );
  assert.strictEqual(
    dropped.length,
    1,
    `giving up on a command must be announced, got ${JSON.stringify(dropped)}`,
  );
  assert.strictEqual(dropped[0].cmd.type, 'run');
  assert.strictEqual(
    dropped[0].cmd.tabId,
    't1',
    'the announcement must carry the command, so the tab that was ' +
      'optimistically marked running can be put back',
  );
  assert.strictEqual(dropped[0].reason, 'expired');
  console.log('  ok - a command dropped as stale is announced');
}

async function testOverflowingCommandsAreAnnounced() {
  const sockPath = tmpSock('overflow.sock');
  const client = new AgentClient(sockPath, {
    reconnectBaseMs: 40,
    reconnectMaxMs: 120,
    pendingTtlMs: 5000,
    maxPendingSends: 3,
  });
  const dropped = [];
  client.on('commandDropped', (cmd, reason) => dropped.push({cmd, reason}));

  for (let i = 0; i < 5; i += 1) {
    client.sendCommand({type: 'run', prompt: 'p', tabId: `t${i}`});
  }
  client.dispose();

  assert.deepStrictEqual(
    dropped.map(d => [d.cmd.tabId, d.reason]),
    [
      ['t0', 'overflow'],
      ['t1', 'overflow'],
    ],
    'the commands pushed out of the bounded queue must be announced, ' +
      `got ${JSON.stringify(dropped)}`,
  );
  console.log('  ok - commands pushed out of the queue are announced');
}

async function testDeliveredCommandsAreNotAnnounced() {
  const sockPath = tmpSock('delivered.sock');
  const received = [];
  const server = net.createServer(conn => {
    conn.on('data', d => received.push(d.toString()));
  });
  await listen(server, sockPath);

  const client = new AgentClient(sockPath, {
    reconnectBaseMs: 40,
    reconnectMaxMs: 120,
    pendingTtlMs: 5000,
  });
  const dropped = [];
  client.on('commandDropped', cmd => dropped.push(cmd));

  // One queued moments before the daemon answers, one sent afterwards.
  client.sendCommand({type: 'run', prompt: 'queued', tabId: 't1'});
  await new Promise(resolve => {
    client.on('connect', resolve);
    client.connect();
  });
  client.sendCommand({type: 'run', prompt: 'live', tabId: 't2'});
  await delay(150);
  client.dispose();
  await close(server);

  const lines = received
    .join('')
    .split('\n')
    .filter(l => l.trim());
  assert.strictEqual(lines.length, 2, 'both commands must be delivered');
  assert.deepStrictEqual(
    dropped,
    [],
    'a command that WAS delivered must not be announced as dropped',
  );
  console.log('  ok - delivered commands are not announced');
}

// ---------------------------------------------------------------------
// 2. The sidebar must put the tab back
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
  return {webviewView, fireMessage: m => recv.fire(m)};
}

function statusFor(posted, tabId) {
  return posted
    .filter(m => m && m.type === 'status' && m.tabId === tabId)
    .map(m => m.running);
}

// The daemon is unreachable for the whole test. The user sends prompts
// in a long-lived window until the bounded queue has to give some up.
async function testTabsWithADroppedRunAreNotLeftRunning() {
  const ws = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-drop-ws-'));
  tmpDirs.push(ws);
  workspaceFolders = [{uri: makeUri(ws)}];

  const posted = [];
  const view = new SorcarSidebarView(makeUri(path.join(__dirname, '..')));
  const wv = makeWebviewView(posted);
  view.resolveWebviewView(wv.webviewView, {}, {});
  wv.fireMessage({type: 'ready', tabId: 'tab-0', restoredTabs: []});

  // A newline keeps the prompt off the "open this file" shortcut path.
  const submit = tabId =>
    wv.fireMessage({
      type: 'submit',
      prompt: 'please do the thing\nfor me',
      model: 'm',
      tabId,
    });

  submit('tab-0');
  assert.deepStrictEqual(
    statusFor(posted, 'tab-0'),
    [true],
    'sending a prompt must show the tab as running straight away',
  );

  // The window stays open through a long outage. The queue is bounded
  // at 256 frames, so the oldest ones eventually have to go.
  for (let i = 1; i <= 300; i += 1) submit(`tab-${i}`);
  await delay(100);

  assert.deepStrictEqual(
    statusFor(posted, 'tab-0'),
    [true, false],
    'a tab whose run was dropped must be put back to not-running: it was ' +
      'marked running before the daemon had seen anything, and nothing ' +
      'else will ever unmark it',
  );
  assert.deepStrictEqual(
    statusFor(posted, 'tab-300'),
    [true],
    'a tab whose run is still queued must stay running',
  );

  const notices = posted.filter(m => m && m.type === 'notification');
  assert.ok(
    notices.length > 0,
    'the user must be told their request was not started, or the tab ' +
      'simply appears to have forgotten it',
  );

  if (typeof view.dispose === 'function') view.dispose();
  console.log('  ok - a tab whose run was dropped is not left running');
}

(async () => {
  try {
    await testExpiredCommandIsAnnounced();
    await testOverflowingCommandsAreAnnounced();
    await testDeliveredCommandsAreNotAnnounced();
    await testTabsWithADroppedRunAreNotLeftRunning();
    console.log('droppedRunUnsticksTab.test.js: all tests passed');
  } finally {
    Module._resolveFilename = origResolve;
    for (const dir of tmpDirs.reverse()) {
      fs.rmSync(dir, {recursive: true, force: true});
    }
  }
})().catch(err => {
  console.error('FAIL:', err && err.stack ? err.stack : err);
  process.exit(1);
});
