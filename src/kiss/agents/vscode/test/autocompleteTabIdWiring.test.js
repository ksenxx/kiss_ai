// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// Wiring test for the autocomplete tab stamp.
//
// The webview stamps `tabId` on getFiles/complete so a reply can be
// matched against the conversation on screen.  The extension host
// rebuilds forwarded commands from a field whitelist, so a field the
// whitelist forgets is silently DROPPED before the daemon ever sees it
// -- and the webview's guard then rejects every (now untagged) reply,
// killing the @-mention picker and ghost text outright.
//
// A jsdom-only test cannot see that: it asserts on messages the webview
// posts, which are correct.  This test observes the raw JSON command
// lines arriving on a REAL unix socket from the REAL compiled host.

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

const vscodeStub = {
  workspace: {
    get workspaceFolders() {
      return workspaceFolders;
    },
    getConfiguration: () => ({get: () => 'stub-default-model'}),
    onDidChangeWorkspaceFolders: () => ({dispose: () => {}}),
    openTextDocument: uri => Promise.resolve({getText: () => '', uri}),
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
    showTextDocument: () => Promise.resolve({selection: null}),
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

const tmpHome = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-acwire-'));
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

// Every JSON command line the host actually writes to the daemon.
const daemonCommands = [];
let lastServerSock = null;
const server = net.createServer(sock => {
  lastServerSock = sock;
  let buf = '';
  sock.on('data', chunk => {
    buf += chunk.toString('utf8');
    let nl;
    while ((nl = buf.indexOf('\n')) >= 0) {
      const line = buf.slice(0, nl);
      buf = buf.slice(nl + 1);
      if (!line.trim()) continue;
      try {
        daemonCommands.push(JSON.parse(line));
      } catch (_e) {}
    }
  });
});

function makeBridgedWebview(fireToHost) {
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

function typeInto(win, text) {
  const inp = win.document.getElementById('task-input');
  inp.value = text;
  inp.selectionStart = text.length;
  inp.selectionEnd = text.length;
  inp.dispatchEvent(new win.Event('input', {bubbles: true}));
  return inp;
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

  const ws = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-acwire-ws-'));
  tmpDirs.push(ws);
  workspaceFolders = [{uri: makeUri(ws)}];

  const view = new SorcarSidebarView(makeUri(path.join(__dirname, '..')));
  const hostRecv = new StubEventEmitter();
  let webviewRef = null;
  const hostWebview = {
    options: {},
    html: '',
    cspSource: 'vscode-resource:',
    asWebviewUri: uri => makeUri(uri.fsPath),
    postMessage: msg => {
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

  webviewRef = makeBridgedWebview(msg => hostRecv.fire(msg));
  const {win} = webviewRef;

  // Typing an @-mention must reach the daemon WITH the tab stamp.
  typeInto(win, 'look at @sr');
  const getFiles = await waitFor(
    () => daemonCommands.find(c => c.type === 'getFiles'),
    'host must forward getFiles to the daemon',
  );
  assert.ok(
    getFiles.tabId,
    'getFiles must reach the daemon WITH tabId; the forwarded-command ' +
      'whitelist dropped it (@-mention picker would render nothing)',
  );
  console.log('  ok - getFiles reaches the daemon with tabId');

  // The ghost-text request carries the stamp too (300ms debounce).
  typeInto(win, 'write a haiku');
  const complete = await waitFor(
    () => daemonCommands.find(c => c.type === 'complete'),
    'host must forward complete to the daemon',
  );
  assert.ok(
    complete.tabId,
    'complete must reach the daemon WITH tabId; ghost text and inline ' +
      'completions would render nothing',
  );
  console.log('  ok - complete reaches the daemon with tabId');

  // The stamp is the tab that typed, not a stale host-side notion.
  assert.strictEqual(
    getFiles.tabId,
    complete.tabId,
    'both requests must name the same (active) tab',
  );

  view.dispose();
  win.close();
}

runTests()
  .then(() => {
    console.log('autocompleteTabIdWiring.test.js: all tests passed');
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
