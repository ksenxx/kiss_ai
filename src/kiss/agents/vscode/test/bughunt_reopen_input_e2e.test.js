// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// END-TO-END integration test: the REAL compiled extension
// (``SorcarSidebarView.js`` + ``AgentClient.js``) wired to the REAL
// webview (``media/main.js`` running in jsdom) over a real
// Unix-domain-socket stub daemon.  Only the ``vscode`` module and the
// daemon's agent logic are stubbed — every layer the bug touches
// (webview send logic, extension forwarding + _runningTabs gate,
// daemon command routing) is exercised for real.
//
// REQUIREMENT (the bug): when a task is started in an extension tab and
// the tab (the secondary-sidebar view) is CLOSED and re-OPENED while the
// task is still running, user input must NOT be ignored — neither during
// the task nor after it finishes.  A tab where a task started must behave
// exactly like a tab that loads the task:
//   * a message typed WHILE the task runs must reach the daemon as an
//     ``appendUserMessage`` (injected into the live agent), and
//   * a message typed AFTER the task finishes must reach the daemon as a
//     ``run`` (a brand-new task) — i.e. it must NOT be swallowed by the
//     extension's ``submit`` → ``_runningTabs`` drop guard.
//
// Run directly with ``node``:
//
//     node src/kiss/agents/vscode/test/bughunt_reopen_input_e2e.test.js

'use strict';

const assert = require('assert');
const fs = require('fs');
const net = require('net');
const os = require('os');
const path = require('path');
const Module = require('module');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

// ---- vscode module stub (for the compiled extension) ----------------
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
const stubPath = path.join(__dirname, '_vscode-stub.js');
fs.writeFileSync(
  stubPath,
  `'use strict';\nmodule.exports = global.__kissVscodeStub;\n`,
);
global.__kissVscodeStub = vscodeStub;

const tmpHome = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-e2e-'));
process.env.HOME = tmpHome;
process.env.USERPROFILE = tmpHome;
fs.mkdirSync(path.join(tmpHome, '.kiss'), {recursive: true});
const sockPath = path.join(tmpHome, '.kiss', 'sorcar.sock');

if (process.platform === 'win32') {
  console.log('  skipped on win32 (UDS test)');
  fs.rmSync(tmpHome, {recursive: true, force: true});
  process.exit(0);
}

// ---- stub daemon (UDS) ----------------------------------------------
// Records every command the extension forwards and emulates the real
// daemon's reply pattern: ``run`` → clear + status:running; and
// ``resumeSession`` → status:running + task_events (server.py ordering).
const daemonCmds = [];
let lastServerSock = null;
function daemonReply(obj) {
  if (lastServerSock) lastServerSock.write(JSON.stringify(obj) + '\n');
}
const server = net.createServer(sock => {
  lastServerSock = sock;
  let buf = '';
  sock.on('data', chunk => {
    buf += chunk.toString();
    const lines = buf.split('\n');
    buf = lines.pop();
    for (const line of lines) {
      if (!line.trim()) continue;
      let cmd;
      try {
        cmd = JSON.parse(line);
      } catch {
        continue;
      }
      daemonCmds.push(cmd);
      if (cmd.type === 'run') {
        const tabId = cmd.tabId;
        daemonReply({type: 'clear', chat_id: 'chat-1', tabId});
        daemonReply({
          type: 'status',
          running: true,
          tabId,
          startTs: Date.now(),
        });
      } else if (cmd.type === 'resumeSession') {
        const tabId = cmd.tabId;
        daemonReply({
          type: 'status',
          running: true,
          tabId,
          startTs: Date.now(),
        });
        daemonReply({
          type: 'task_events',
          events: [],
          task: 'do a long task',
          tabId,
          chat_id: 'chat-1',
        });
      }
    }
  });
});

// ---- jsdom webview running the REAL main.js -------------------------
let persistedState; // shared across reopen (vscode.getState)

function buildWebviewWindow() {
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
  html = html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');
  const dom = new JSDOM(html, {
    runScripts: 'dangerously',
    pretendToBeVisual: true,
    url: 'https://localhost/',
  });
  const win = dom.window;
  win.Element.prototype.scrollIntoView = function () {};
  win.Element.prototype.scrollTo = function () {};
  win.HTMLElement.prototype.scrollTo = function () {};
  // ``toExtension`` is wired to the extension's onDidReceiveMessage
  // dispatcher once resolveWebviewView has run.
  const ctx = {toExtension: null};
  win.acquireVsCodeApi = function () {
    return {
      postMessage: msg => {
        if (ctx.toExtension) ctx.toExtension(msg);
      },
      getState: () => persistedState,
      setState: s => {
        persistedState = s;
      },
    };
  };
  ctx.win = win;
  return ctx;
}

function evalWebviewScripts(win) {
  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
}

/**
 * Build a ``WebviewView`` whose ``webview.postMessage`` delivers
 * extension→webview messages into the jsdom window, and whose
 * ``onDidReceiveMessage`` is wired (via ctx.toExtension) to the jsdom
 * webview's ``vscode.postMessage``.
 */
function makeWebviewView(ctx) {
  const recv = new StubEventEmitter();
  const dispose = new StubEventEmitter();
  const vis = new StubEventEmitter();
  const webview = {
    options: {},
    html: '',
    cspSource: 'vscode-resource:',
    asWebviewUri: uri => uri,
    postMessage: msg => {
      ctx.win.dispatchEvent(new ctx.win.MessageEvent('message', {data: msg}));
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
    wire: () => {
      ctx.toExtension = msg => recv.fire(msg);
    },
    fireDispose: () => dispose.fire(),
  };
}

function typeAndSend(win, text) {
  const inp = win.document.getElementById('task-input');
  const sendBtn = win.document.getElementById('send-btn');
  inp.value = text;
  inp.dispatchEvent(new win.Event('input', {bubbles: true}));
  sendBtn.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

const sleep = ms => new Promise(r => setTimeout(r, ms));

async function runTests() {
  await new Promise((res, rej) =>
    server.listen(sockPath, err => (err ? rej(err) : res())),
  );

  const sourcePath = path.join(__dirname, '..', 'out', 'SorcarSidebarView.js');
  assert.ok(
    fs.existsSync(sourcePath),
    `compiled extension missing: ${sourcePath} — run \`tsc -p .\` first`,
  );
  delete require.cache[require.resolve(sourcePath)];
  const {SorcarSidebarView} = require(sourcePath);

  const ws = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-e2e-ws-'));
  workspaceFolders = [{uri: makeUri(ws)}];
  const extUri = makeUri(path.join(__dirname, '..'));
  const view = new SorcarSidebarView(extUri);

  // ---- Open tab 1 and start a task --------------------------------
  const ctx1 = buildWebviewWindow();
  const wvv1 = makeWebviewView(ctx1);
  view.resolveWebviewView(wvv1.webviewView, {}, {});
  wvv1.wire();
  evalWebviewScripts(ctx1.win); // IIFE posts ``ready`` → extension
  await sleep(50);

  typeAndSend(ctx1.win, 'do a long task');
  await sleep(80);
  const TAB = view._activeTabId;
  assert.ok(
    daemonCmds.some(c => c.type === 'run' && c.tabId === TAB),
    'daemon must receive the initial run command',
  );
  assert.ok(TAB, 'extension must have learned the active tab id from submit');

  // ---- Close + reopen the view while the task is running ----------
  // VS Code re-resolves with the fresh webview FIRST, then fires the
  // old webview's dispose (the real ordering).
  const ctx2 = buildWebviewWindow();
  const wvv2 = makeWebviewView(ctx2);
  view.resolveWebviewView(wvv2.webviewView, {}, {});
  wvv2.wire();
  wvv1.fireDispose();
  evalWebviewScripts(ctx2.win); // IIFE posts ``ready`` w/ restoredTabs
  await sleep(120);

  // (1) DURING the task: typed input must reach the daemon as
  //     appendUserMessage (NOT a dropped submit/run).
  daemonCmds.length = 0;
  typeAndSend(ctx2.win, 'please also update the docs');
  await sleep(80);
  const duringTypes = daemonCmds
    .filter(c => c.type === 'appendUserMessage' || c.type === 'run')
    .map(c => c.type);
  assert.deepStrictEqual(
    duringTypes,
    ['appendUserMessage'],
    'BUG: after close+reopen, a message typed while the task runs must ' +
      'reach the daemon as appendUserMessage (was: ' +
      JSON.stringify(duringTypes) +
      '). The re-opened tab ignored the user input.',
  );

  // (2) AFTER the task finishes: typed input must reach the daemon as a
  //     new run (must NOT be dropped by the submit/_runningTabs guard).
  daemonReply({type: 'status', running: false, tabId: TAB});
  await sleep(80);
  daemonCmds.length = 0;
  typeAndSend(ctx2.win, 'now do a follow-up task');
  await sleep(80);
  const afterTypes = daemonCmds
    .filter(c => c.type === 'appendUserMessage' || c.type === 'run')
    .map(c => c.type);
  assert.deepStrictEqual(
    afterTypes,
    ['run'],
    'BUG: after the task finished in a re-opened tab, a typed message ' +
      'must start a new run (was: ' +
      JSON.stringify(afterTypes) +
      '). The extension dropped it as a submit for a still-"running" tab.',
  );

  if (typeof view.dispose === 'function') view.dispose();
  ctx1.win.close();
  ctx2.win.close();
  fs.rmSync(ws, {recursive: true, force: true});
}

function cleanup() {
  try {
    if (lastServerSock) lastServerSock.destroy();
  } catch {}
  try {
    server.close();
  } catch {}
  try {
    fs.unlinkSync(sockPath);
  } catch {}
  try {
    fs.unlinkSync(stubPath);
  } catch {}
  fs.rmSync(tmpHome, {recursive: true, force: true});
}

runTests().then(
  () => {
    cleanup();
    console.log('\nAll tests passed');
    process.exit(0);
  },
  err => {
    console.error('FAIL:', err && err.message ? err.message : err);
    cleanup();
    process.exit(1);
  },
);
