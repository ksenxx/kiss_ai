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
      return Promise.resolve(undefined);
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

const origResolve = Module._resolveFilename;
Module._resolveFilename = function (request, parent, ...rest) {
  if (request === 'vscode') return require.resolve('./_vscode-stub.js');
  return origResolve.call(this, request, parent, ...rest);
};
const stubPath = path.join(__dirname, '_vscode-stub.js');
if (!fs.existsSync(stubPath)) {
  fs.writeFileSync(
    stubPath,
    `'use strict';\nmodule.exports = global.__kissVscodeStub || {};\n`,
  );
}
global.__kissVscodeStub = vscodeStub;

const tmpHome = fs.mkdtempSync(
  path.join(os.tmpdir(), 'kiss-server-reset-float-'),
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

const PRELOADED_TAB_ID = 'tab-server-reset-float';

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
    fs.readFileSync(path.join(mediaDir, 'api.js'), 'utf8'),
    dom.getInternalVMContext(),
  );
  vm.runInContext(
    fs.readFileSync(path.join(mediaDir, 'main.js'), 'utf8'),
    dom.getInternalVMContext(),
  );
  // The daemon's canonical tab snapshot (tabs are server-owned now);
  // combined with the persisted chatId above it restores the selection.
  win.dispatchEvent(
    new win.MessageEvent('message', {
      data: {
        type: 'tabs_state',
        tabs: [
          {tabId: PRELOADED_TAB_ID, chatId: '', title: 'new chat',
           workDir: ''},
        ],
      },
    }),
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
    path.join(os.tmpdir(), 'kiss-server-reset-float-ws-'),
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

function getFloatingModal(win) {
  return win.document.getElementById('server-reset-confirm-modal');
}

function isFloatingModalOpen(win) {
  const modal = getFloatingModal(win);
  return !!(modal && modal.classList.contains('open'));
}

async function runTests() {
  await new Promise((res, rej) =>
    server.listen(sockPath, err => (err ? rej(err) : res())),
  );

  {
    const domWebview = makeDomWebview();
    try {
      const settingsPanel =
        domWebview.win.document.getElementById('settings-panel');
      assert.ok(settingsPanel, '#settings-panel must exist');
      const modal = getFloatingModal(domWebview.win);
      assert.ok(
        modal,
        'settings panel must contain a #server-reset-confirm-modal floating box',
      );
      assert.ok(
        settingsPanel.contains(modal),
        'the floating modal must live INSIDE #settings-panel (not at the document root)',
      );
      assert.ok(
        domWebview.win.document.getElementById('server-reset-confirm-ok'),
        'modal must expose an OK button (#server-reset-confirm-ok)',
      );
      assert.ok(
        domWebview.win.document.getElementById('server-reset-confirm-cancel'),
        'modal must expose a Cancel button (#server-reset-confirm-cancel)',
      );
      assert.strictEqual(
        isFloatingModalOpen(domWebview.win),
        false,
        'the floating modal must be CLOSED by default',
      );
      assert.strictEqual(
        modal.getAttribute('role'),
        'dialog',
        'modal must declare role="dialog" for assistive tech',
      );
    } finally {
      domWebview.close();
    }
  }

  {
    warningCalls = [];
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

      await waitFor(
        () => isFloatingModalOpen(domWebview.win),
        'click must open the in-settings-panel floating confirmation box',
      );

      assert.strictEqual(
        domWebview.posted.filter(m => m.type === 'serverReset').length,
        0,
        'the webview must NOT post serverReset until the user clicks OK',
      );

      assert.strictEqual(
        warningCalls.length,
        0,
        'the new contract: vscode.window.showWarningMessage must NOT be called for the reset confirmation; got ' +
          warningCalls.length +
          ' calls',
      );

      const forwardPosted = (() => {
        const seen = new Set();
        return () => {
          for (const m of domWebview.posted) {
            if (m.type !== 'serverReset') continue;
            if (seen.has(m)) continue;
            seen.add(m);
            wv.fireMessage(m);
          }
        };
      })();

      clickInWindow(domWebview.win, 'server-reset-confirm-ok');

      await waitFor(
        () => !isFloatingModalOpen(domWebview.win),
        'OK must close the floating confirmation box',
      );

      const resets = domWebview.posted.filter(m => m.type === 'serverReset');
      assert.strictEqual(
        resets.length,
        1,
        'OK must trigger exactly one webview->extension serverReset post',
      );

      forwardPosted();
      await waitFor(
        () =>
          daemonLines
            .slice(before)
            .some(m => m && m.type === 'serverReset'),
        'after OK, daemon must receive exactly one serverReset',
      );
      await new Promise(r => setTimeout(r, 100));
      const daemonResets = daemonLines
        .slice(before)
        .filter(m => m && m.type === 'serverReset');
      assert.strictEqual(
        daemonResets.length,
        1,
        'daemon must receive exactly ONE serverReset per OK; got ' +
          daemonResets.length,
      );

      assert.strictEqual(
        warningCalls.length,
        0,
        'showWarningMessage must STILL not have been called (extension is just a forwarder now)',
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

  {
    warningCalls = [];
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
      await waitFor(
        () => isFloatingModalOpen(domWebview.win),
        'click must open the floating confirmation box',
      );

      clickInWindow(domWebview.win, 'server-reset-confirm-cancel');
      await waitFor(
        () => !isFloatingModalOpen(domWebview.win),
        'Cancel must close the floating confirmation box',
      );

      for (const m of domWebview.posted) {
        if (m.type === 'serverReset') wv.fireMessage(m);
      }
      await new Promise(r => setTimeout(r, 200));

      assert.strictEqual(
        domWebview.posted.filter(m => m.type === 'serverReset').length,
        0,
        'Cancel must NOT post serverReset to the extension',
      );
      const daemonResets = daemonLines
        .slice(before)
        .filter(m => m && m.type === 'serverReset');
      assert.strictEqual(
        daemonResets.length,
        0,
        'Cancel must NOT result in any serverReset reaching the daemon',
      );
      assert.strictEqual(
        warningCalls.length,
        0,
        'the new contract: vscode.window.showWarningMessage must NOT be called',
      );
    } finally {
      view.dispose();
      domWebview.close();
    }
  }

  {
    warningCalls = [];
    const before = daemonLines.length;
    const {view, domWebview, wv} = await setupView();
    try {
      clickInWindow(domWebview.win, 'cfg-server-reset-btn');

      await new Promise(r => setTimeout(r, 100));
      assert.strictEqual(
        isFloatingModalOpen(domWebview.win),
        false,
        'with no running agent, the floating confirmation box must NOT open',
      );

      const posted = await waitFor(
        () => domWebview.posted.find(m => m.type === 'serverReset'),
        'no-agent click must post serverReset immediately',
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
        'fast path must NOT raise a native VS Code modal',
      );
    } finally {
      view.dispose();
      domWebview.close();
    }
  }

  {
    warningCalls = [];
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
      await waitFor(
        () => isFloatingModalOpen(domWebview.win),
        'first click must open the floating confirmation box',
      );
      clickInWindow(domWebview.win, 'cfg-server-reset-btn');
      assert.strictEqual(
        isFloatingModalOpen(domWebview.win),
        true,
        'second click must NOT close or re-stack the floating dialog',
      );

      clickInWindow(domWebview.win, 'server-reset-confirm-ok');
      await waitFor(
        () => !isFloatingModalOpen(domWebview.win),
        'OK must close the dialog',
      );

      const resets = domWebview.posted.filter(m => m.type === 'serverReset');
      assert.strictEqual(
        resets.length,
        1,
        'double-click + OK must produce exactly ONE serverReset post (not two)',
      );
      wv.fireMessage(resets[0]);
      await waitFor(
        () =>
          daemonLines
            .slice(before)
            .some(m => m && m.type === 'serverReset'),
        'daemon must receive one serverReset',
      );
      await new Promise(r => setTimeout(r, 100));
      const daemonResets = daemonLines
        .slice(before)
        .filter(m => m && m.type === 'serverReset');
      assert.strictEqual(
        daemonResets.length,
        1,
        'daemon must receive exactly ONE serverReset per confirmed click cycle, got ' +
          daemonResets.length,
      );
      assert.strictEqual(
        warningCalls.length,
        0,
        'no native VS Code modal must ever be raised',
      );
    } finally {
      view.dispose();
      domWebview.close();
    }
  }

  {
    warningCalls = [];
    const {view, domWebview, wv} = await setupView();
    try {
      send(domWebview.win, {
        type: 'status',
        tabId: PRELOADED_TAB_ID,
        running: true,
        startTs: Date.now(),
      });

      clickInWindow(domWebview.win, 'cfg-server-reset-btn');
      await waitFor(
        () => isFloatingModalOpen(domWebview.win),
        'first click must open the floating box',
      );
      clickInWindow(domWebview.win, 'server-reset-confirm-cancel');
      await waitFor(
        () => !isFloatingModalOpen(domWebview.win),
        'Cancel must close the box',
      );

      clickInWindow(domWebview.win, 'cfg-server-reset-btn');
      await waitFor(
        () => isFloatingModalOpen(domWebview.win),
        'after the previous box closed, a fresh click must re-open it',
      );
      clickInWindow(domWebview.win, 'server-reset-confirm-cancel');
      await waitFor(
        () => !isFloatingModalOpen(domWebview.win),
        'second Cancel must close again',
      );

      assert.strictEqual(
        domWebview.posted.filter(m => m.type === 'serverReset').length,
        0,
        'two Cancel cycles must produce zero serverReset posts',
      );
      assert.strictEqual(
        warningCalls.length,
        0,
        'no native VS Code modal must ever be raised',
      );
      for (const m of domWebview.posted) {
        if (m.type === 'serverReset') wv.fireMessage(m);
      }
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
