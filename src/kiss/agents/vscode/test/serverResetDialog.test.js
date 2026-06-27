// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// Integration test for the "Server reset" confirmation **dialog**.
//
// Bug reproduced
// --------------
// When the user clicks the settings-panel "Server reset" button while
// an agent is still running on any tab, the extension must raise a
// native VS Code modal dialog with OK and Cancel buttons.  Only when
// the user picks OK may the daemon receive ``{type: 'serverReset'}``;
// Cancel must abort the reset and leave the in-flight agent
// untouched.  The previous in-webview toast confirmation did not
// satisfy the user, so this test pins the contract to a real VS Code
// modal dialog (``vscode.window.showWarningMessage`` with
// ``{modal: true}`` and an ``OK`` button — VS Code auto-renders the
// Cancel button for modal warning messages).
//
// What this test does
// -------------------
// It drives the *compiled* ``SorcarSidebarView`` (the extension host
// side) against:
//   * a real loopback Unix-domain-socket daemon stub,
//   * a real JSDOM-rendered ``media/main.js`` (the webview).
// The only stubbed piece is the ``vscode`` module — in particular
// ``vscode.window.showWarningMessage`` records every call and returns
// a test-controlled answer ("OK" → confirm, ``undefined`` → cancel).
//
// Three cases:
//   1. Agent running + user picks OK → daemon receives serverReset.
//   2. Agent running + user picks Cancel → daemon does NOT receive
//      serverReset, no toast leaks into the webview.
//   3. No agent running → daemon receives serverReset immediately and
//      ``showWarningMessage`` is never called.

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

// ``warningCalls`` records every call to ``showWarningMessage`` so the
// test can assert that the modal dialog was raised with the expected
// options and label set.  ``warningAnswer`` is a function that returns
// the value the stubbed modal "resolves" to (OK / undefined).
let warningCalls = [];
let warningAnswer = () => undefined;
let nativeInfoErrorCount = 0;

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
    showInformationMessage: () => {
      nativeInfoErrorCount++;
      return Promise.resolve(undefined);
    },
    showWarningMessage: (message, options, ...actions) => {
      warningCalls.push({message, options, actions});
      return Promise.resolve(warningAnswer());
    },
    showErrorMessage: () => {
      nativeInfoErrorCount++;
      return Promise.resolve(undefined);
    },
    showTextDocument: () => Promise.resolve({}),
    activeTextEditor: undefined,
    tabGroups: {all: []},
  },
  commands: {executeCommand: () => Promise.resolve()},
};

// Route ``require('vscode')`` to the stub before loading the compiled
// extension.
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

const tmpHome = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-server-reset-'));
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

// Stub daemon: collects every line written by the AgentClient so the
// test can grep for ``{type:'serverReset'}``.
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

async function waitFor(predicate, message, attempts = 100) {
  for (let i = 0; i < attempts; i++) {
    const value = predicate();
    if (value) return value;
    await new Promise(r => setTimeout(r, 20));
  }
  throw new Error(message || 'waitFor timed out');
}

async function waitForFalsy(predicate, message, attempts = 50) {
  for (let i = 0; i < attempts; i++) {
    if (!predicate()) return;
    await new Promise(r => setTimeout(r, 20));
  }
  throw new Error(message || 'waitForFalsy timed out');
}

function makeWebviewView(domWindow) {
  // ``recv`` carries webview→extension messages.  Anything the
  // extension posts back to the webview is dispatched into JSDOM as a
  // real ``MessageEvent`` so ``media/main.js`` reacts to it the way it
  // would inside VS Code.
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

const PRELOADED_TAB_ID = 'tab-server-reset-integration';

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
  // Preload a single-tab persisted state with a deterministic chatId.
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
  // Fresh workspace each scenario so SorcarSidebarView state is clean.
  const ws = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-server-reset-ws-'));
  tmpDirs.push(ws);
  workspaceFolders = [{uri: makeUri(ws)}];

  const sourcePath = path.join(
    __dirname,
    '..',
    'out',
    'SorcarSidebarView.js',
  );
  assert.ok(fs.existsSync(sourcePath), `compiled extension missing: ${sourcePath}`);
  delete require.cache[require.resolve(sourcePath)];
  const {SorcarSidebarView} = require(sourcePath);
  const view = new SorcarSidebarView(makeUri(path.join(__dirname, '..')));

  const domWebview = makeDomWebview();
  const wv = makeWebviewView(domWebview.win);
  view.resolveWebviewView(wv.webviewView, {}, {});

  // The DOM webview's own ``ready`` post was queued before the extension
  // was wired in; replay it so the extension treats this tab as live.
  const ready = domWebview.posted.find(m => m.type === 'ready');
  if (ready) wv.fireMessage(ready);

  await waitForClient();
  return {view, domWebview, wv};
}

async function runTests() {
  await new Promise((res, rej) =>
    server.listen(sockPath, err => (err ? rej(err) : res())),
  );

  // ============================================================
  // Case 1 — agent running, user picks OK → daemon receives reset.
  // ============================================================
  {
    warningCalls = [];
    warningAnswer = () => 'OK';
    const before = daemonLines.length;
    const {view, domWebview, wv} = await setupView();
    try {
      // Mark the active tab as running.
      send(domWebview.win, {
        type: 'status',
        tabId: PRELOADED_TAB_ID,
        running: true,
        startTs: Date.now(),
      });

      clickInWindow(domWebview.win, 'cfg-server-reset-btn');

      // Webview must have posted serverReset with agentRunning=true.
      const posted = await waitFor(
        () => domWebview.posted.find(m => m.type === 'serverReset'),
        'webview must post {type:"serverReset"} on click',
      );
      assert.strictEqual(
        posted.agentRunning,
        true,
        'webview must report agentRunning=true so the extension can confirm',
      );

      // Forward the webview→extension message into the extension.
      wv.fireMessage(posted);

      // Extension must raise the modal dialog.
      await waitFor(
        () => warningCalls.length === 1,
        'extension must call vscode.window.showWarningMessage exactly once',
      );
      const call = warningCalls[0];
      assert.ok(
        call.options && call.options.modal === true,
        'dialog must be modal (options.modal === true)',
      );
      assert.ok(
        Array.isArray(call.actions) && call.actions.includes('OK'),
        'dialog must expose an OK action button (Cancel is added by VS Code for modal warnings)',
      );
      const text = String(call.message || '').toLowerCase();
      assert.ok(
        text.includes('agent') && text.includes('running'),
        'dialog message must mention that an agent is still running, got: ' +
          JSON.stringify(call.message),
      );

      // User picked OK → daemon must receive the reset.
      await waitFor(
        () =>
          daemonLines
            .slice(before)
            .some(m => m && m.type === 'serverReset'),
        'after user picks OK, daemon must receive {type:"serverReset"}',
      );
      assert.strictEqual(
        nativeInfoErrorCount,
        0,
        'no native info/error notification must fire for the reset flow',
      );
    } finally {
      view.dispose();
      domWebview.close();
    }
  }

  // ============================================================
  // Case 2 — agent running, user picks Cancel → no reset.
  // ============================================================
  {
    warningCalls = [];
    warningAnswer = () => undefined; // VS Code returns undefined on Cancel.
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
      const posted = await waitFor(
        () => domWebview.posted.find(m => m.type === 'serverReset'),
        'webview must post {type:"serverReset"} on click',
      );
      assert.strictEqual(posted.agentRunning, true);
      wv.fireMessage(posted);

      await waitFor(
        () => warningCalls.length === 1,
        'extension must call showWarningMessage once',
      );

      // Give the extension time to act on the dialog answer.
      await new Promise(r => setTimeout(r, 200));

      // No reset must reach the daemon.
      const resetLines = daemonLines
        .slice(before)
        .filter(m => m && m.type === 'serverReset');
      assert.strictEqual(
        resetLines.length,
        0,
        'after user picks Cancel, daemon must NOT receive serverReset; got: ' +
          JSON.stringify(resetLines),
      );

      // The webview must NOT have rendered a confirmation toast in
      // place of the modal — the dialog is the extension's job.
      assert.strictEqual(
        domWebview.win.document.querySelectorAll('.kiss-notification').length,
        0,
        'no in-webview confirmation toast must be raised — the dialog is a native VS Code modal',
      );
    } finally {
      view.dispose();
      domWebview.close();
    }
  }

  // ============================================================
  // Case 3 — no agent running → reset goes through immediately.
  // ============================================================
  {
    warningCalls = [];
    warningAnswer = () => undefined;
    const before = daemonLines.length;
    const {view, domWebview, wv} = await setupView();
    try {
      clickInWindow(domWebview.win, 'cfg-server-reset-btn');
      const posted = await waitFor(
        () => domWebview.posted.find(m => m.type === 'serverReset'),
        'webview must post {type:"serverReset"} on click',
      );
      assert.strictEqual(
        posted.agentRunning,
        false,
        'with no running agent, agentRunning must be false',
      );
      wv.fireMessage(posted);

      await waitFor(
        () =>
          daemonLines
            .slice(before)
            .some(m => m && m.type === 'serverReset'),
        'with no running agent, daemon must receive the reset immediately',
      );
      assert.strictEqual(
        warningCalls.length,
        0,
        'with no running agent the extension must NOT raise the modal dialog',
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
