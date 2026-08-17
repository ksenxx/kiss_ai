// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Clicking a linkified .md/.markdown file path must open the file
// converted to HTML and rendered in a TAB — VS Code's built-in markdown
// preview (`markdown.showPreview`) — instead of showing the raw
// markdown source in the text editor, matching the remote web app,
// which converts .md content with marked and renders it in a content
// tab (renderContentView in media/main.js).

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
// When true, `markdown.showPreview` rejects, emulating a VS Code whose
// built-in markdown extension is disabled — the editor fallback path.
let markdownPreviewBroken = false;

class StubWebviewPanel {
  constructor(viewType, title, viewColumn, options) {
    this.viewType = viewType;
    this.title = title;
    this.viewColumn = viewColumn;
    this.options = options;
    this.disposed = false;
    this._onDispose = new StubEventEmitter();
    this.webview = {
      html: '',
      asWebviewUri: uri => makeUri(uri.fsPath),
      postMessage: () => Promise.resolve(true),
    };
  }
  reveal() {}
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
      if (cmd === 'markdown.showPreview' && markdownPreviewBroken) {
        return Promise.reject(new Error("command 'markdown.showPreview' not found"));
      }
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

const tmpHome = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-mdprev-'));
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

function previewCommands() {
  return executedCommands.filter(c => c.cmd === 'markdown.showPreview');
}

function clear() {
  openedTextDocs.length = 0;
  shownTextDocs.length = 0;
  executedCommands.length = 0;
  createdPanels.length = 0;
  markdownPreviewBroken = false;
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

  const ws = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-mdprev-ws-'));
  tmpDirs.push(ws);
  workspaceFolders = [{uri: makeUri(ws)}];

  const mdFile = path.join(ws, 'docs', 'notes.md');
  fs.mkdirSync(path.dirname(mdFile), {recursive: true});
  fs.writeFileSync(mdFile, '# Notes\n\nSome **bold** text.\n');

  const markdownFile = path.join(ws, 'docs', 'guide.markdown');
  fs.writeFileSync(markdownFile, '# Guide\n');

  const htmlFile = path.join(ws, 'reports', 'summary.html');
  fs.mkdirSync(path.dirname(htmlFile), {recursive: true});
  fs.writeFileSync(htmlFile, '<!DOCTYPE html><html><body>hi</body></html>');

  const pyFile = path.join(ws, 'src', 'main.py');
  fs.mkdirSync(path.dirname(pyFile), {recursive: true});
  fs.writeFileSync(pyFile, 'print("hello")\n');

  const view = new SorcarSidebarView(makeUri(path.join(__dirname, '..')));
  const wv = makeWebviewView();
  view.resolveWebviewView(wv.webviewView, {}, {});
  wv.fireMessage({type: 'ready', tabId: 'tab1', restoredTabs: []});
  await waitForClient();

  // 1. Clicking a .md link opens the built-in markdown preview tab
  //    (converted-to-HTML rendering), NOT the raw source in the editor
  //    and NOT an html webview panel.
  clear();
  wv.fireMessage({type: 'openFile', path: 'docs/notes.md'});
  await waitFor(
    () => previewCommands().length === 1,
    'md file: markdown.showPreview must be invoked',
  );
  assert.strictEqual(previewCommands()[0].args.length, 1);
  assert.strictEqual(
    previewCommands()[0].args[0].fsPath,
    mdFile,
    'preview must target the resolved absolute md path',
  );
  assert.deepStrictEqual(openedTextDocs, [], 'no text document opened');
  assert.deepStrictEqual(shownTextDocs, [], 'no text editor shown');
  assert.deepStrictEqual(createdPanels, [], 'no html webview panel for .md');
  assert.deepStrictEqual(
    executedCommands.filter(c => c.cmd === 'vscode.open'),
    [],
    'vscode.open must NOT be invoked for markdown',
  );
  console.log('  ok - .md link opens the rendered markdown preview tab');

  // 2. .markdown works the same way.
  clear();
  wv.fireMessage({type: 'openFile', path: 'docs/guide.markdown'});
  await waitFor(
    () => previewCommands().length === 1,
    'markdown file: markdown.showPreview must be invoked',
  );
  assert.strictEqual(previewCommands()[0].args[0].fsPath, markdownFile);
  assert.deepStrictEqual(shownTextDocs, [], 'no text editor shown');
  console.log('  ok - .markdown link opens the rendered preview tab too');

  // 3. A path-only submit (typing "docs/notes.md" and pressing Send)
  //    opens the very same preview tab as clicking the link.
  clear();
  wv.fireMessage({
    type: 'submit',
    prompt: 'docs/notes.md',
    workDir: ws,
    tabId: 'tab1',
  });
  await waitFor(
    () => previewCommands().length === 1,
    'submit shortcut: markdown.showPreview must be invoked',
  );
  assert.strictEqual(previewCommands()[0].args[0].fsPath, mdFile);
  assert.deepStrictEqual(shownTextDocs, [], 'no text editor shown');
  console.log('  ok - path-only submit renders markdown like a click');

  // 4. When the preview command fails (built-in markdown extension
  //    disabled), the file still opens — as source in the text editor.
  clear();
  markdownPreviewBroken = true;
  wv.fireMessage({type: 'openFile', path: 'docs/notes.md'});
  await waitFor(
    () => shownTextDocs.length === 1,
    'broken preview: the text editor fallback must open the file',
  );
  assert.strictEqual(previewCommands().length, 1, 'preview was attempted');
  assert.deepStrictEqual(openedTextDocs, [mdFile]);
  assert.deepStrictEqual(shownTextDocs, [mdFile]);
  console.log('  ok - editor fallback when markdown preview is unavailable');

  // 5. Regression: a Python file still opens in the text editor and
  //    never triggers the markdown preview.
  clear();
  wv.fireMessage({type: 'openFile', path: 'src/main.py'});
  await waitFor(
    () => openedTextDocs.length === 1 && shownTextDocs.length === 1,
    'python file must still open in the text editor',
  );
  assert.strictEqual(openedTextDocs[0], pyFile);
  assert.deepStrictEqual(previewCommands(), [], 'no preview for .py');
  console.log('  ok - non-markdown files still open in the text editor');

  // 6. Regression: .html still opens the html webview panel, not the
  //    markdown preview.
  clear();
  wv.fireMessage({type: 'openFile', path: 'reports/summary.html'});
  await waitFor(
    () => createdPanels.length === 1,
    'html file must still open its webview panel',
  );
  assert.deepStrictEqual(previewCommands(), [], 'no preview for .html');
  console.log('  ok - .html still renders in its own webview tab');

  // 7. A .md path outside the workspace is still refused.
  clear();
  const outside = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-mdprev-out-'));
  tmpDirs.push(outside);
  const outsideMd = path.join(outside, 'evil.md');
  fs.writeFileSync(outsideMd, '# nope\n');
  wv.fireMessage({type: 'openFile', path: outsideMd});
  await new Promise(r => setTimeout(r, 100));
  assert.deepStrictEqual(
    previewCommands(),
    [],
    'outside-workspace md: no preview may open',
  );
  assert.deepStrictEqual(shownTextDocs, [], 'and no editor either');
  console.log('  ok - outside-workspace md path is refused');

  view.dispose();
}

runTests()
  .then(() => {
    console.log('openFileMarkdownPreview tests passed');
    process.exit(0);
  })
  .catch(err => {
    console.error(err);
    process.exit(1);
  })
  .finally(() => {
    server.close();
    for (const dir of tmpDirs.slice().reverse()) {
      fs.rmSync(dir, {recursive: true, force: true});
    }
  });
