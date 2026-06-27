// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// Integration test for the "Server reset" confirmation dialog —
// guard-bypass via the fast path.
//
// Bug reproduced
// --------------
// The first fix to BUG 1 added an in-flight guard around
// ``showWarningMessage`` so a rapid double-click cannot stack two
// modals.  But the guard is checked only inside the
// ``if (message.agentRunning)`` branch — the fast path that fires
// ``sendCommand({type: 'serverReset'})`` directly when
// ``agentRunning`` is false is NOT guarded.  So:
//
//   1. Click 1: webview snapshots ``agentRunning=true`` → extension
//      opens the modal dialog.
//   2. While the dialog is open, the agent finishes
//      (``status: {running: false}``).
//   3. Click 2: webview snapshots ``agentRunning=false`` →
//      extension takes the fast path → immediately sends
//      ``{type: "serverReset"}`` to the daemon.
//   4. User picks OK on the dialog → extension sends a SECOND
//      ``{type: "serverReset"}`` to the daemon.
//
// The daemon receives two resets — the second one will hit the
// freshly respawned daemon and tear it down again.
//
// Contract pinned by this test
// ----------------------------
// While a server-reset confirmation dialog is open, every subsequent
// ``serverReset`` arriving in the extension must be dropped — regardless
// of its ``agentRunning`` flag.  At most ONE ``serverReset`` must reach
// the daemon per confirmed click cycle.

'use strict';

const assert = require('assert');
const fs = require('fs');
const net = require('net');
const os = require('os');
const path = require('path');
const vm = require('vm');
const Module = require('module');
const {JSDOM} = require('jsdom');

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

class StubCancellationTokenSource {
  constructor() {
    this.token = {onCancellationRequested: () => ({dispose: () => {}})};
  }
  dispose() {}
}

function makeUri(fsPath) {
  return {fsPath, scheme: 'file', toString: () => `file://${fsPath}`};
}

let workspaceFolders = [];
let warningCalls = [];
let warningResolvers = [];

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
  CancellationTokenSource: StubCancellationTokenSource,
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
    showInformationMessage: () => Promise.resolve(undefined),
    showWarningMessage: (message, options, ...actions) => {
      warningCalls.push({message, options, actions});
      return new Promise(resolve => {
        warningResolvers.push(resolve);
      });
    },
    showErrorMessage: () => Promise.resolve(undefined),
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

const tmpHome = fs.mkdtempSync(
  path.join(os.tmpdir(), 'kiss-server-reset-bypass-'),
);
const tmpDirs = [tmpHome];
process.env.HOME = tmpHome;
process.env.USERPROFILE = tmpHome;
fs.mkdirSync(path.join(tmpHome, '.kiss'), {recursive: true});
const sockPath = path.join(tmpHome, '.kiss', 'sorcar.sock');

if (process.platform === 'win32') {
  console.log('  skipped on win32 (UDS test)');
  for (const dir of tmpDirs.slice().reverse()) {
    fs.rmSync(dir, {recursive: true, force: true});
  }
  process.exit(0);
}

let lastServerSock = null;
const daemonLines = [];
let daemonBuffer = '';
const server = net.createServer(sock => {
  lastServerSock = sock;
  sock.on('data', chunk => {
    daemonBuffer += chunk.toString('utf8');
    let i;
    while ((i = daemonBuffer.indexOf('\n')) >= 0) {
      const line = daemonBuffer.slice(0, i);
      daemonBuffer = daemonBuffer.slice(i + 1);
      if (!line.trim()) continue;
      try {
        daemonLines.push(JSON.parse(line));
      } catch {
        daemonLines.push({raw: line});
      }
    }
  });
});

async function waitForClient() {
  for (let i = 0; i < 100 && !lastServerSock; i++) {
    await new Promise(r => setTimeout(r, 20));
  }
  assert.ok(lastServerSock, 'client never connected to daemon');
}

async function waitFor(predicate, message, attempts = 150) {
  for (let i = 0; i < attempts; i++) {
    const value = predicate();
    if (value) return value;
    await new Promise(r => setTimeout(r, 20));
  }
  throw new Error(message || 'waitFor timed out');
}

function makeWebviewView(domWindow) {
  const recv = new StubEventEmitter();
  const webview = {
    options: {},
    html: '',
    cspSource: 'vscode-resource:',
    asWebviewUri: uri => makeUri(uri.fsPath),
    postMessage: msg => {
      domWindow.dispatchEvent(
        new domWindow.MessageEvent('message', {data: msg}),
      );
      return Promise.resolve(true);
    },
    onDidReceiveMessage: cb => recv.event(cb),
  };
  const webviewView = {
    webview,
    visible: true,
    show: () => {},
    onDidChangeVisibility: () => ({dispose: () => {}}),
    onDidDispose: () => ({dispose: () => {}}),
  };
  return {webviewView, fireMessage: m => recv.fire(m)};
}

const PRELOADED_TAB_ID = 'tab-server-reset-bypass';

function makeDomWebview() {
  const mediaDir = path.join(__dirname, '..', 'media');
  let html = fs.readFileSync(path.join(mediaDir, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
  html = html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');

  const dom = new JSDOM(html, {
    runScripts: 'outside-only',
    pretendToBeVisual: true,
    url: 'https://localhost/',
  });
  const win = dom.window;
  win.Element.prototype.scrollIntoView = function () {};
  win.Element.prototype.scrollTo = function () {};
  win.HTMLElement.prototype.scrollTo = function () {};

  const posted = [];
  let state = {
    tabs: [
      {
        title: 'new chat',
        chatId: PRELOADED_TAB_ID,
        backendChatId: '',
        parentTabId: '',
      },
    ],
    activeTabIndex: 0,
    chatId: PRELOADED_TAB_ID,
  };
  win.acquireVsCodeApi = function () {
    return {
      postMessage: msg => posted.push(msg),
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };
  vm.runInContext(
    fs.readFileSync(path.join(mediaDir, 'panelCopy.js'), 'utf8'),
    dom.getInternalVMContext(),
  );
  vm.runInContext(
    fs.readFileSync(path.join(mediaDir, 'main.js'), 'utf8'),
    dom.getInternalVMContext(),
  );
  return {win, posted, close: () => win.close()};
}

function clickInWindow(win, id) {
  const el = win.document.getElementById(id);
  assert.ok(el, `element #${id} must exist`);
  el.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

async function setupView() {
  const ws = fs.mkdtempSync(
    path.join(os.tmpdir(), 'kiss-server-reset-bypass-ws-'),
  );
  tmpDirs.push(ws);
  workspaceFolders = [{uri: makeUri(ws)}];

  const sourcePath = path.join(
    __dirname,
    '..',
    'out',
    'SorcarSidebarView.js',
  );
  assert.ok(
    fs.existsSync(sourcePath),
    `compiled extension missing: ${sourcePath}`,
  );
  delete require.cache[require.resolve(sourcePath)];
  const {SorcarSidebarView} = require(sourcePath);
  const view = new SorcarSidebarView(makeUri(path.join(__dirname, '..')));

  const domWebview = makeDomWebview();
  const wv = makeWebviewView(domWebview.win);
  view.resolveWebviewView(wv.webviewView, {}, {});

  const ready = domWebview.posted.find(m => m.type === 'ready');
  if (ready) wv.fireMessage(ready);

  await waitForClient();
  return {view, domWebview, wv};
}

async function runTests() {
  await new Promise((res, rej) =>
    server.listen(sockPath, err => (err ? rej(err) : res())),
  );

  // ================================================================
  // Case: dialog open + agent finishes + second click (fast path).
  // ================================================================
  //
  //   1. Mark the tab as running.
  //   2. Click 1: webview posts {agentRunning:true}; extension opens
  //      a modal (we don't resolve it yet).
  //   3. The agent finishes (status running:false).
  //   4. Click 2: webview now posts {agentRunning:false}; without the
  //      fix the extension takes the fast path and immediately
  //      sends serverReset.
  //   5. User picks OK on the dialog from click 1.
  //
  // Required:
  //   * Daemon receives at most ONE serverReset for this click cycle.
  {
    warningCalls = [];
    warningResolvers = [];
    const before = daemonLines.length;
    const {view, domWebview, wv} = await setupView();
    try {
      send(domWebview.win, {
        type: 'status',
        tabId: PRELOADED_TAB_ID,
        running: true,
        startTs: Date.now(),
      });

      // Click 1 → modal opens.
      clickInWindow(domWebview.win, 'cfg-server-reset-btn');
      const post1 = await waitFor(
        () => domWebview.posted.find(m => m.type === 'serverReset'),
        'click 1 must post serverReset',
      );
      assert.strictEqual(post1.agentRunning, true);
      wv.fireMessage(post1);
      await waitFor(
        () => warningCalls.length === 1,
        'click 1 must raise the modal dialog',
      );

      // Agent finishes while modal is open.
      send(domWebview.win, {
        type: 'status',
        tabId: PRELOADED_TAB_ID,
        running: false,
      });

      // Click 2 — the webview now reports agentRunning:false.
      clickInWindow(domWebview.win, 'cfg-server-reset-btn');
      await waitFor(
        () =>
          domWebview.posted.filter(m => m.type === 'serverReset').length >= 2,
        'click 2 must post serverReset',
      );
      const allPosts = domWebview.posted.filter(
        m => m.type === 'serverReset',
      );
      const post2 = allPosts[1];
      assert.strictEqual(
        post2.agentRunning,
        false,
        'after the agent stopped, click 2 must report agentRunning=false',
      );
      wv.fireMessage(post2);

      // Give the extension a beat to process click 2 (the buggy fast
      // path would fire sendCommand right here).
      await new Promise(r => setTimeout(r, 200));

      // The modal from click 1 is still open.  Resolve it as OK now.
      assert.strictEqual(
        warningResolvers.length,
        1,
        'exactly one pending modal dialog must still be open',
      );
      warningResolvers.shift()('OK');

      // Let everything settle.
      await new Promise(r => setTimeout(r, 300));

      // At most ONE serverReset must reach the daemon.
      const resetsToDaemon = daemonLines
        .slice(before)
        .filter(m => m && m.type === 'serverReset');
      assert.strictEqual(
        resetsToDaemon.length,
        1,
        'at most one serverReset must reach the daemon per click cycle; ' +
          'got ' +
          resetsToDaemon.length +
          ' resets (click 2 fast-path must be blocked while a ' +
          'confirmation modal is open)',
      );
    } finally {
      view.dispose();
      domWebview.close();
    }
  }

  // ================================================================
  // Case: dialog open + Cancel + later legit no-running click.
  // ================================================================
  //
  // After the dialog from click 1 is cancelled, a later
  // ``agentRunning:false`` click must still fast-path through (the
  // guard must clear).
  {
    warningCalls = [];
    warningResolvers = [];
    const before = daemonLines.length;
    const {view, domWebview, wv} = await setupView();
    try {
      send(domWebview.win, {
        type: 'status',
        tabId: PRELOADED_TAB_ID,
        running: true,
        startTs: Date.now(),
      });

      clickInWindow(domWebview.win, 'cfg-server-reset-btn');
      const post1 = await waitFor(
        () => domWebview.posted.find(m => m.type === 'serverReset'),
        'click 1 must post serverReset',
      );
      wv.fireMessage(post1);
      await waitFor(
        () => warningCalls.length === 1,
        'click 1 must raise the modal',
      );

      // Cancel.
      warningResolvers.shift()(undefined);
      await new Promise(r => setTimeout(r, 150));

      // Agent finished after cancel.
      send(domWebview.win, {
        type: 'status',
        tabId: PRELOADED_TAB_ID,
        running: false,
      });

      // Fresh click after cancel — fast path must work.
      clickInWindow(domWebview.win, 'cfg-server-reset-btn');
      const posts = domWebview.posted.filter(m => m.type === 'serverReset');
      wv.fireMessage(posts[1]);
      await waitFor(
        () =>
          daemonLines
            .slice(before)
            .some(m => m && m.type === 'serverReset'),
        'after Cancel cleared the guard, a fresh no-agent click must ' +
          'fast-path the reset to the daemon',
      );
    } finally {
      view.dispose();
      domWebview.close();
    }
  }
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
  for (const dir of tmpDirs.slice().reverse()) {
    fs.rmSync(dir, {recursive: true, force: true});
  }
}

runTests().then(
  () => {
    cleanup();
    console.log('\nAll tests passed');
    process.exit(0);
  },
  err => {
    console.error('FAIL:', err && err.stack ? err.stack : err);
    cleanup();
    process.exit(1);
  },
);
