// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Clicking a linkified .html/.htm file path must render the page in a
// webview TAB (like the remote web app's content tabs) instead of
// opening its source in the VS Code text editor.

'use strict';

const assert = require('assert');
const fs = require('fs');
const net = require('net');
const os = require('os');
const path = require('path');
const Module = require('module');

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
  return {
    fsPath,
    scheme: 'file',
    toString: () => `vscode-webview-resource://stub${fsPath}`,
  };
}

let workspaceFolders = [];
const openedTextDocs = [];
const shownTextDocs = [];
const executedCommands = [];
const createdPanels = [];

class StubWebviewPanel {
  constructor(viewType, title, viewColumn, options) {
    this.viewType = viewType;
    this.title = title;
    this.viewColumn = viewColumn;
    this.options = options;
    this.revealCalls = [];
    this.disposed = false;
    this._onDispose = new StubEventEmitter();
    this.webview = {
      html: '',
      asWebviewUri: uri => makeUri(uri.fsPath),
      postMessage: () => Promise.resolve(true),
    };
  }
  reveal(column) {
    this.revealCalls.push(column);
  }
  onDidDispose(cb) {
    return this._onDispose.event(cb);
  }
  dispose() {
    if (this.disposed) return;
    this.disposed = true;
    this._onDispose.fire();
  }
}

const vscodeStub = {
  workspace: {
    get workspaceFolders() {
      return workspaceFolders;
    },
    getConfiguration: () => ({get: () => 'stub-default-model'}),
    onDidChangeWorkspaceFolders: () => ({dispose: () => {}}),
    openTextDocument: uriOrPath => {
      const fsPath =
        uriOrPath && typeof uriOrPath === 'object' && uriOrPath.fsPath
          ? uriOrPath.fsPath
          : String(uriOrPath || '');
      openedTextDocs.push(fsPath);
      return Promise.resolve({uri: makeUri(fsPath), getText: () => ''});
    },
    textDocuments: [],
  },
  EventEmitter: StubEventEmitter,
  CancellationTokenSource: StubCancellationTokenSource,
  Uri: {
    file: p => makeUri(p),
    joinPath: (base, ...parts) => makeUri(path.join(base.fsPath, ...parts)),
    parse: s => makeUri(s),
  },
  Position: class {},
  Range: class {},
  Selection: class {},
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
    showTextDocument: doc => {
      shownTextDocs.push(doc && doc.uri ? doc.uri.fsPath : '');
      return Promise.resolve({
        document: doc,
        set selection(_v) {},
        revealRange: () => {},
      });
    },
    createWebviewPanel: (viewType, title, viewColumn, options) => {
      const panel = new StubWebviewPanel(viewType, title, viewColumn, options);
      createdPanels.push(panel);
      return panel;
    },
    activeTextEditor: undefined,
    tabGroups: {all: []},
  },
  commands: {
    executeCommand: (cmd, ...args) => {
      executedCommands.push({cmd, args});
      return Promise.resolve();
    },
  },
};

const origResolve = Module._resolveFilename;
Module._resolveFilename = function (request, parent, ...rest) {
  if (request === 'vscode') return require.resolve('./_vscode-stub.js');
  return origResolve.call(this, request, parent, ...rest);
};
global.__kissVscodeStub = vscodeStub;

const tmpHome = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-htmltab-'));
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

async function waitForClient() {
  for (let i = 0; i < 100 && !lastServerSock; i++) {
    await new Promise(r => setTimeout(r, 20));
  }
  assert.ok(lastServerSock, 'client never connected to daemon');
}

function makeWebviewView() {
  const recv = new StubEventEmitter();
  const posted = [];
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
    onDidChangeVisibility: () => ({dispose: () => {}}),
    onDidDispose: () => ({dispose: () => {}}),
  };
  return {webviewView, posted, fireMessage: m => recv.fire(m)};
}

async function waitFor(predicate, message, timeoutMs = 1500) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    const value = predicate();
    if (value) return value;
    await new Promise(r => setTimeout(r, 10));
  }
  throw new Error(message || 'waitFor timed out');
}

function clear() {
  openedTextDocs.length = 0;
  shownTextDocs.length = 0;
  executedCommands.length = 0;
  createdPanels.length = 0;
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
  delete require.cache[require.resolve(sourcePath)];
  const {SorcarSidebarView} = require(sourcePath);

  const ws = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-htmltab-ws-'));
  tmpDirs.push(ws);
  workspaceFolders = [{uri: makeUri(ws)}];

  const htmlFile = path.join(ws, 'reports', 'summary.html');
  fs.mkdirSync(path.dirname(htmlFile), {recursive: true});
  fs.writeFileSync(
    htmlFile,
    '<!DOCTYPE html><html><head><title>t</title></head>' +
      '<body><h1>Report</h1><img src="chart.png"></body></html>',
  );

  const htmFile = path.join(ws, 'reports', 'legacy.htm');
  fs.writeFileSync(htmFile, '<p>no head element here</p>');

  const pyFile = path.join(ws, 'src', 'main.py');
  fs.mkdirSync(path.dirname(pyFile), {recursive: true});
  fs.writeFileSync(pyFile, 'print("hello")\n');

  const view = new SorcarSidebarView(makeUri(path.join(__dirname, '..')));
  const wv = makeWebviewView();
  view.resolveWebviewView(wv.webviewView, {}, {});
  wv.fireMessage({type: 'ready', tabId: 'tab1', restoredTabs: []});
  await waitForClient();

  // 1. Clicking an .html link opens a rendered webview tab, NOT the
  //    text editor and NOT the native viewer.
  clear();
  wv.fireMessage({type: 'openFile', path: 'reports/summary.html'});
  await waitFor(
    () => createdPanels.length === 1,
    'html file: a webview panel must be created',
  );
  const panel = createdPanels[0];
  assert.strictEqual(panel.viewType, 'kissSorcarHtmlPreview');
  assert.strictEqual(panel.title, 'summary.html');
  assert.ok(
    panel.options && panel.options.enableScripts === true,
    'panel must allow scripts so the page behaves like in a browser tab',
  );
  assert.ok(
    panel.webview.html.includes('<h1>Report</h1>'),
    'panel must contain the rendered file content',
  );
  const baseAt = panel.webview.html.indexOf('<base href=');
  const headAt = panel.webview.html.indexOf('<head>');
  assert.ok(baseAt > headAt && headAt >= 0, '<base> injected after <head>');
  assert.ok(
    panel.webview.html.includes(
      `<base href="vscode-webview-resource://stub${path.dirname(htmlFile)}/">`,
    ),
    'base href must point at the file directory via asWebviewUri',
  );
  const roots = panel.options.localResourceRoots.map(u => u.fsPath);
  assert.ok(
    roots.includes(path.dirname(htmlFile)),
    'file directory must be a local resource root',
  );
  assert.ok(roots.includes(ws), 'workspace folder must be a resource root');
  assert.deepStrictEqual(openedTextDocs, [], 'no text document opened');
  assert.deepStrictEqual(shownTextDocs, [], 'no text editor shown');
  assert.deepStrictEqual(
    executedCommands.filter(c => c.cmd === 'vscode.open'),
    [],
    'vscode.open must NOT be invoked for html',
  );
  console.log('  ok - .html link opens a rendered webview tab');

  // 2. A second click on the same path reveals the SAME tab (no
  //    duplicate) and re-reads the file so fresh content shows.
  fs.writeFileSync(
    htmlFile,
    '<!DOCTYPE html><html><head></head><body><h1>Updated</h1></body></html>',
  );
  wv.fireMessage({type: 'openFile', path: 'reports/summary.html'});
  await waitFor(
    () => panel.revealCalls.length === 1,
    'second click must reveal the existing panel',
  );
  assert.strictEqual(createdPanels.length, 1, 'no duplicate panel');
  assert.ok(
    panel.webview.html.includes('<h1>Updated</h1>'),
    'reopened tab must show the file content re-read from disk',
  );
  console.log('  ok - second click reuses and refreshes the same tab');

  // 3. After the user closes the tab, the next click creates a new one.
  panel.dispose();
  wv.fireMessage({type: 'openFile', path: 'reports/summary.html'});
  await waitFor(
    () => createdPanels.length === 2,
    'after dispose a fresh panel must be created',
  );
  const recreatedPanel = createdPanels[1];
  console.log('  ok - closed tab is recreated on next click');

  // 4. .htm works too, and a document without <head> still gets a base.
  clear();
  wv.fireMessage({type: 'openFile', path: 'reports/legacy.htm'});
  await waitFor(
    () => createdPanels.length === 1,
    'htm file: a webview panel must be created',
  );
  const htmPanel = createdPanels[0];
  assert.ok(
    createdPanels[0].webview.html.startsWith('<base href='),
    'headless document gets the <base> prepended',
  );
  assert.ok(createdPanels[0].webview.html.includes('no head element here'));
  console.log('  ok - .htm renders in a tab, base prepended without <head>');

  // 4b. <base> placement is robust: a <head> inside a comment is not
  //     the insertion point, a quoted '>' inside the real head tag's
  //     attributes does not end the tag early, and the base lands right
  //     after the genuine opening tag.
  clear();
  const trickyFile = path.join(ws, 'reports', 'tricky.html');
  fs.writeFileSync(
    trickyFile,
    '<!-- fake <head> in a comment --><!DOCTYPE html><html>' +
      '<head data-x="a>b"><title>x</title></head><body>tricky</body></html>',
  );
  wv.fireMessage({type: 'openFile', path: 'reports/tricky.html'});
  await waitFor(
    () => createdPanels.length === 1,
    'tricky html: a webview panel must be created',
  );
  assert.ok(
    createdPanels[0].webview.html.includes('<head data-x="a>b"><base href='),
    'base must be injected after the real head tag, not in the comment',
  );
  console.log('  ok - base injection skips comments and quoted ">"');

  // 4c. A document with a doctype but no <head> gets the base injected
  //     AFTER the doctype, so the page is not demoted to quirks mode.
  clear();
  const noHeadFile = path.join(ws, 'reports', 'nohead.html');
  fs.writeFileSync(noHeadFile, '<!DOCTYPE html><p>only body</p>');
  wv.fireMessage({type: 'openFile', path: 'reports/nohead.html'});
  await waitFor(
    () => createdPanels.length === 1,
    'no-head html: a webview panel must be created',
  );
  assert.ok(
    createdPanels[0].webview.html.startsWith('<!DOCTYPE html><base href='),
    'base must come after the doctype, never before it',
  );
  console.log('  ok - doctype stays first when there is no <head>');

  // 5. Regression: a Python file still opens in the text editor and
  //    never creates a webview panel.
  clear();
  wv.fireMessage({type: 'openFile', path: 'src/main.py'});
  await waitFor(
    () => openedTextDocs.length === 1 && shownTextDocs.length === 1,
    'python file must still open in the text editor',
  );
  assert.strictEqual(openedTextDocs[0], pyFile);
  assert.deepStrictEqual(createdPanels, [], 'no webview panel for .py');
  console.log('  ok - non-html files still open in the text editor');

  // 6. An .html path outside the workspace is still refused.
  clear();
  const outside = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-out-'));
  tmpDirs.push(outside);
  const outsideHtml = path.join(outside, 'evil.html');
  fs.writeFileSync(outsideHtml, '<p>nope</p>');
  wv.fireMessage({type: 'openFile', path: outsideHtml});
  await new Promise(r => setTimeout(r, 100));
  assert.deepStrictEqual(
    createdPanels,
    [],
    'outside-workspace html: no panel may be created',
  );
  console.log('  ok - outside-workspace html path is refused');

  // 7. dispose() closes every remaining preview tab (the recreated
  //    summary.html panel from test 3 and the legacy.htm panel from
  //    test 4 are both still open).
  assert.ok(!htmPanel.disposed, 'legacy.htm panel still open');
  assert.ok(!recreatedPanel.disposed, 'summary.html panel still open');
  view.dispose();
  assert.ok(
    recreatedPanel.disposed && htmPanel.disposed,
    'view.dispose() must dispose all open preview panels',
  );
  console.log('  ok - view dispose closes preview tabs');

  server.close();
  for (const dir of tmpDirs.slice().reverse()) {
    fs.rmSync(dir, {recursive: true, force: true});
  }
}

(async () => {
  try {
    await runTests();
    console.log('\n9 passed, 0 failed');
    process.exit(0);
  } catch (err) {
    console.error('FAIL:', err && err.stack ? err.stack : err);
    process.exit(1);
  }
})();
