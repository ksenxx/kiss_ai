// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// Integration test for the "Server reset" confirmation dialog —
// concurrent / rapid double-click case.
//
// Bug reproduced
// --------------
// The settings-panel "Server reset" button has no debounce in the
// webview, and the extension's ``case 'serverReset'`` has no
// in-flight guard around ``vscode.window.showWarningMessage``.  A
// rapid double-click therefore:
//
//   1. raises TWO stacked modal dialogs in VS Code (poor UX), and
//   2. if the user picks OK on each, sends TWO ``{type:"serverReset"}``
//      lines to the daemon — the second one tears down the freshly
//      respawned daemon.
//
// Contract pinned by this test
// ----------------------------
//   * On rapid back-to-back clicks while an agent is running, the
//     extension MUST raise at most ONE modal dialog at a time.
//   * After the user picks OK on that dialog, the daemon MUST
//     receive at most ONE ``{type:"serverReset"}`` per confirmed
//     click cycle (subsequent clicks that landed while the dialog
//     was open must NOT each trigger their own reset).
//
// This is an end-to-end test: real compiled ``SorcarSidebarView``,
// real UDS daemon stub, real JSDOM-rendered ``media/main.js``, with
// only the ``vscode`` module stubbed.

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

// Each ``showWarningMessage`` call returns a deferred Promise so the
// test can control overlap.  ``warningCalls`` records the call args,
// ``warningResolvers`` records the resolve() functions, and
// ``peakConcurrent`` is the maximum number of dialogs that were ever
// open simultaneously.
let warningCalls = [];
let warningResolvers = [];
let openCount = 0;
let peakConcurrent = 0;
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
      openCount++;
      if (openCount > peakConcurrent) peakConcurrent = openCount;
      return new Promise(resolve => {
        warningResolvers.push(value => {
          openCount--;
          resolve(value);
        });
      });
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

const tmpHome = fs.mkdtempSync(
  path.join(os.tmpdir(), 'kiss-server-reset-dbl-'),
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

// Stub daemon: collects every line written by the AgentClient.
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

const PRELOADED_TAB_ID = 'tab-server-reset-dbl';

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
    path.join(os.tmpdir(), 'kiss-server-reset-dbl-ws-'),
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

  // ========================================================
  // Case: rapid double-click while an agent is running.
  // ========================================================
  //
  // The user clicks Server reset twice in quick succession before
  // they get a chance to answer the first dialog.  Both clicks land
  // in the extension while the first ``showWarningMessage`` is still
  // awaiting an answer.  We then resolve every pending dialog as OK
  // and assert:
  //
  //   * peak concurrent open dialogs is 1 (not 2 — no stacking),
  //   * the daemon receives exactly ONE serverReset (not two —
  //     subsequent clicks that landed during the open dialog must
  //     not each fire their own reset).
  {
    warningCalls = [];
    warningResolvers = [];
    openCount = 0;
    peakConcurrent = 0;
    const before = daemonLines.length;
    const {view, domWebview, wv} = await setupView();
    try {
      send(domWebview.win, {
        type: 'status',
        tabId: PRELOADED_TAB_ID,
        running: true,
        startTs: Date.now(),
      });

      // Two rapid clicks before any modal answer.
      clickInWindow(domWebview.win, 'cfg-server-reset-btn');
      clickInWindow(domWebview.win, 'cfg-server-reset-btn');

      // Both webview posts arrive.
      await waitFor(
        () =>
          domWebview.posted.filter(m => m.type === 'serverReset').length >= 2,
        'webview must post serverReset twice for two clicks',
      );
      const resets = domWebview.posted.filter(m => m.type === 'serverReset');
      // Forward both to the extension as fast as possible.
      wv.fireMessage(resets[0]);
      wv.fireMessage(resets[1]);

      // Give the extension a beat to process both messages.
      await new Promise(r => setTimeout(r, 200));

      // At this point any well-behaved extension has at most ONE
      // dialog open.  A buggy extension has TWO stacked dialogs.
      assert.ok(
        warningCalls.length >= 1,
        'at least one modal dialog must be raised by the first click',
      );
      assert.strictEqual(
        peakConcurrent,
        1,
        'peak concurrent open modal dialogs must be 1 (no stacking); ' +
          'observed peak=' +
          peakConcurrent +
          ', total dialogs raised so far=' +
          warningCalls.length,
      );

      // Resolve every pending dialog as OK so any racing handlers
      // would happily send their daemon command.
      while (warningResolvers.length) {
        const resolve = warningResolvers.shift();
        resolve('OK');
        // Yield to let microtasks run between resolves.
        await new Promise(r => setTimeout(r, 30));
      }

      // Let everything settle, including a second possible dialog if
      // the in-flight guard is missing.
      await new Promise(r => setTimeout(r, 200));

      // The daemon must have received exactly ONE serverReset.
      const resetsToDaemon = daemonLines
        .slice(before)
        .filter(m => m && m.type === 'serverReset');
      assert.strictEqual(
        resetsToDaemon.length,
        1,
        'daemon must receive exactly ONE serverReset per confirmed click ' +
          'cycle (the duplicate click while dialog was open must not ' +
          'fire its own reset); got ' +
          resetsToDaemon.length +
          ' resets',
      );

      assert.strictEqual(
        nativeInfoErrorCount,
        0,
        'no native info/error notifications must fire for the reset flow',
      );
    } finally {
      view.dispose();
      domWebview.close();
    }
  }

  // ========================================================
  // Sanity: after the dialog closes, a fresh click works again.
  // ========================================================
  //
  // The in-flight guard must clear once the modal resolves so the
  // user can click again afterwards.
  {
    warningCalls = [];
    warningResolvers = [];
    openCount = 0;
    peakConcurrent = 0;
    const before = daemonLines.length;
    const {view, domWebview, wv} = await setupView();
    try {
      send(domWebview.win, {
        type: 'status',
        tabId: PRELOADED_TAB_ID,
        running: true,
        startTs: Date.now(),
      });

      // First click → modal opens.
      clickInWindow(domWebview.win, 'cfg-server-reset-btn');
      const firstPost = await waitFor(
        () => domWebview.posted.find(m => m.type === 'serverReset'),
        'first click must post serverReset',
      );
      wv.fireMessage(firstPost);
      await waitFor(
        () => warningCalls.length === 1,
        'first click must raise the modal',
      );

      // Resolve first dialog as Cancel (undefined).
      warningResolvers.shift()(undefined);
      await new Promise(r => setTimeout(r, 100));

      // No reset reached the daemon for the cancelled click.
      let resetsToDaemon = daemonLines
        .slice(before)
        .filter(m => m && m.type === 'serverReset');
      assert.strictEqual(
        resetsToDaemon.length,
        0,
        'cancel must not send serverReset',
      );

      // Second click — should raise a fresh modal.
      clickInWindow(domWebview.win, 'cfg-server-reset-btn');
      const posts2 = domWebview.posted.filter(m => m.type === 'serverReset');
      assert.strictEqual(
        posts2.length,
        2,
        'second click must produce a second serverReset post in the webview',
      );
      wv.fireMessage(posts2[1]);
      await waitFor(
        () => warningCalls.length === 2,
        'after the previous dialog closed, a new click must be able to ' +
          'raise a fresh modal',
      );

      // OK this time.
      warningResolvers.shift()('OK');

      // Daemon must receive exactly one reset.
      await waitFor(
        () =>
          daemonLines
            .slice(before)
            .some(m => m && m.type === 'serverReset'),
        'after OK the daemon must receive serverReset',
      );
      await new Promise(r => setTimeout(r, 100));
      resetsToDaemon = daemonLines
        .slice(before)
        .filter(m => m && m.type === 'serverReset');
      assert.strictEqual(
        resetsToDaemon.length,
        1,
        'exactly one serverReset must reach the daemon',
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
