// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// Bridged end-to-end test: the REAL chat webview (media/main.js in
// jsdom) talks to the REAL compiled extension host
// (out/SorcarSidebarView.js) over the real message channel, against a
// REAL temp workspace on disk.  No component in the checkPaths ->
// pathsExist -> click -> openFile chain is faked: webview postMessage
// feeds the host's onDidReceiveMessage, and host postMessage feeds the
// webview's message event.

const assert = require('assert');
const fs = require('fs');
const net = require('net');
const os = require('os');
const path = require('path');
const Module = require('module');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

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
const openedDocs = [];

const vscodeStub = {
  workspace: {
    get workspaceFolders() {
      return workspaceFolders;
    },
    getConfiguration: () => ({get: () => 'stub-default-model'}),
    onDidChangeWorkspaceFolders: () => ({dispose: () => {}}),
    openTextDocument: uri => {
      openedDocs.push(uri.fsPath);
      return Promise.resolve({getText: () => '', uri});
    },
    textDocuments: [],
  },
  EventEmitter: StubEventEmitter,
  CancellationTokenSource: class {
    constructor() {
      this.token = {onCancellationRequested: () => ({dispose: () => {}})};
    }
    dispose() {}
  },
  Uri: {
    file: p => makeUri(p),
    joinPath: (base, ...parts) => makeUri(path.join(base.fsPath, ...parts)),
    parse: s => makeUri(s),
  },
  Position: class {
    constructor(line, ch) {
      this.line = line;
      this.character = ch;
    }
  },
  Range: class {
    constructor(a, b) {
      this.start = a;
      this.end = b;
    }
  },
  Selection: class {
    constructor(a, b) {
      this.anchor = a;
      this.active = b;
    }
  },
  TextEditorRevealType: {InCenter: 2, AtTop: 3},
  ProgressLocation: {Notification: 15},
  ViewColumn: {One: 1},
  window: {
    withProgress: (_opts, task) =>
      task(
        {report: () => {}},
        {onCancellationRequested: () => ({dispose: () => {}})},
      ),
    showInformationMessage: () => Promise.resolve(undefined),
    showWarningMessage: () => Promise.resolve(undefined),
    showErrorMessage: () => Promise.resolve(undefined),
    showTextDocument: () =>
      Promise.resolve({
        selection: null,
        revealRange: () => {},
      }),
    activeTextEditor: undefined,
    tabGroups: {all: []},
  },
  commands: {
    executeCommand: () => Promise.resolve(),
  },
};

const origResolve = Module._resolveFilename;
Module._resolveFilename = function (request, parent, ...rest) {
  if (request === 'vscode') return require.resolve('./_vscode-stub.js');
  return origResolve.call(this, request, parent, ...rest);
};
global.__kissVscodeStub = vscodeStub;

const tmpHome = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-bridge-'));
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
const server = net.createServer(sock => {
  lastServerSock = sock;
  sock.on('data', () => {});
});

// makeBridgedWebview loads chat.html + panelCopy.js + api.js + main.js
// in jsdom and wires BOTH directions to the real extension host `view`:
// webview -> host via onDidReceiveMessage, host -> webview via a
// MessageEvent dispatch (exactly what VS Code's webview bridge does).
function makeBridgedWebview(view, fireToHost) {
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

  const posted = [];
  win.acquireVsCodeApi = function () {
    let state;
    return {
      postMessage: msg => {
        posted.push(msg);
        fireToHost(msg);
      },
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };
  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
  return {win, posted};
}

async function waitFor(predicate, message, timeoutMs = 3000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    const value = predicate();
    if (value) return value;
    await new Promise(r => setTimeout(r, 10));
  }
  throw new Error(message || 'waitFor timed out');
}

function findLinks(win, p) {
  return Array.from(
    win.document.querySelectorAll('#output [data-path]'),
  ).filter(el => el.dataset.path === p);
}

function clickEl(win, el) {
  el.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

async function runTests() {
  await new Promise((res, rej) =>
    server.listen(sockPath, err => (err ? rej(err) : res())),
  );

  const sourcePath = path.join(__dirname, '..', 'out', 'SorcarSidebarView.js');
  assert.ok(
    fs.existsSync(sourcePath),
    `compiled extension missing: ${sourcePath}`,
  );
  const {SorcarSidebarView} = require(sourcePath);

  const ws = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-bridge-ws-'));
  tmpDirs.push(ws);
  workspaceFolders = [{uri: makeUri(ws)}];
  const realFile = path.join(ws, 'src', 'app.py');
  fs.mkdirSync(path.dirname(realFile), {recursive: true});
  fs.writeFileSync(realFile, 'print("bridged")\n');
  const missingFile = path.join(ws, 'src', 'gone.py');

  // Real host side.
  const view = new SorcarSidebarView(makeUri(path.join(__dirname, '..')));
  const hostRecv = new StubEventEmitter();
  let webviewRef = null;
  const hostWebview = {
    options: {},
    html: '',
    cspSource: 'vscode-resource:',
    asWebviewUri: uri => makeUri(uri.fsPath),
    postMessage: msg => {
      // Host -> webview: the real VS Code bridge delivers this as a
      // 'message' event inside the webview page.
      if (webviewRef) {
        webviewRef.win.dispatchEvent(
          new webviewRef.win.MessageEvent('message', {data: msg}),
        );
      }
      return Promise.resolve(true);
    },
    onDidReceiveMessage: cb => hostRecv.event(cb),
  };
  view.resolveWebviewView(
    {
      webview: hostWebview,
      visible: true,
      show: () => {},
      onDidChangeVisibility: () => ({dispose: () => {}}),
      onDidDispose: () => ({dispose: () => {}}),
    },
    {},
    {},
  );

  // Real webview side, bridged to the host.
  webviewRef = makeBridgedWebview(view, msg => hostRecv.fire(msg));
  const {win, posted} = webviewRef;

  // An agent panel mentions one existing and one missing path.
  win.dispatchEvent(
    new win.MessageEvent('message', {
      data: {
        type: 'prompt',
        text: 'compare src/app.py with ' + realFile + ' and ' + missingFile,
      },
    }),
  );
  await waitFor(
    () => posted.some(m => m.type === 'checkPaths'),
    'webview must send checkPaths to the host',
  );
  // The real host answers asynchronously with pathsExist.
  await waitFor(
    () => findLinks(win, realFile).length === 1,
    'absolute existing path must become clickable via the REAL host reply',
  );
  await waitFor(
    () => findLinks(win, 'src/app.py').length === 1,
    'relative existing path must become clickable via the REAL host reply',
  );
  assert.strictEqual(
    findLinks(win, missingFile).length,
    0,
    'missing path must NOT be clickable after the REAL host reply',
  );
  const missing = Array.from(
    win.document.querySelectorAll('#output [data-path-missing]'),
  ).filter(el => el.textContent === missingFile);
  assert.strictEqual(missing.length, 1, 'missing path stays plain text');
  console.log('  ok - real host reply gates clickability end to end');

  // Clicking the promoted link must reach the REAL openFile handler
  // and open the file in the (stubbed) editor.
  clickEl(win, findLinks(win, realFile)[0]);
  await waitFor(
    () => openedDocs.indexOf(realFile) >= 0,
    'clicking the verified link must open the real file via the host',
  );
  console.log('  ok - clicking a verified link opens the file end to end');

  // Clicking the missing-path text posts nothing and opens nothing.
  const openedBefore = openedDocs.length;
  clickEl(win, missing[0]);
  await new Promise(r => setTimeout(r, 100));
  assert.strictEqual(
    openedDocs.length,
    openedBefore,
    'clicking a missing path must not open anything',
  );
  console.log('  ok - missing path click opens nothing end to end');

  view.dispose();
  win.close();
}

runTests()
  .then(() => {
    console.log('clickFilePathBridge.test.js: all tests passed');
  })
  .catch(err => {
    console.error('FAIL:', err && err.stack ? err.stack : err);
    process.exitCode = 1;
  })
  .finally(() => {
    server.close();
    if (lastServerSock) lastServerSock.destroy();
    for (const dir of tmpDirs.slice().reverse()) {
      fs.rmSync(dir, {recursive: true, force: true});
    }
  });
