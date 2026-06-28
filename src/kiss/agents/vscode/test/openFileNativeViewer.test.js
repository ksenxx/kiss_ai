// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end regression test for the extension-side ``openFile``
// handler in ``SorcarSidebarView._handleMessage``.  Drives the
// compiled ``out/SorcarSidebarView.js`` against a stubbed ``vscode``
// module and a stub Unix-domain-socket daemon, exactly like
// ``autocommitProgressSticky.test.js``.
//
// Covered scenarios:
//   * Text source file → ``openTextDocument`` + ``showTextDocument``.
//   * Text source file with ``:line`` → cursor positioned on the line.
//   * Image file (.png) → native viewer via
//     ``vscode.commands.executeCommand('vscode.open', uri)`` (NOT
//     ``openTextDocument``, which would corrupt binary content).
//   * PDF file (.pdf) → native viewer.
//   * Path outside the configured workspace → silently refused.
//   * Non-existent path → silently refused.
//   * Directory → silently refused (only files open).
//
// The "native viewer" routing is the new behaviour the feature adds:
// binary/non-text files must NOT be loaded as text documents because
// VS Code's text editor cannot render them.  Sending them through
// ``vscode.open`` delegates to whichever viewer VS Code has
// registered for that extension (image preview, PDF preview, etc.).

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
  return {fsPath, scheme: 'file', toString: () => `file://${fsPath}`};
}

let workspaceFolders = [];
const openedTextDocs = [];
const shownTextDocs = [];
const executedCommands = [];
const positionCalls = [];
const selectionAssignments = [];
const revealCalls = [];

class StubPosition {
  constructor(line, character) {
    this.line = line;
    this.character = character;
    positionCalls.push({line, character});
  }
}
class StubRange {
  constructor(start, end) {
    this.start = start;
    this.end = end;
  }
}
class StubSelection extends StubRange {
  constructor(anchor, active) {
    super(anchor, active);
    this.anchor = anchor;
    this.active = active;
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
  Position: StubPosition,
  Range: StubRange,
  Selection: StubSelection,
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
    showTextDocument: (doc, _opts) => {
      shownTextDocs.push(doc && doc.uri ? doc.uri.fsPath : '');
      const editor = {
        document: doc,
        get selection() {
          return this._selection;
        },
        set selection(value) {
          this._selection = value;
          selectionAssignments.push({
            line: value && value.active ? value.active.line : null,
            character:
              value && value.active ? value.active.character : null,
          });
        },
        revealRange: (range, type) => {
          revealCalls.push({
            line: range && range.start ? range.start.line : null,
            type,
          });
        },
      };
      return Promise.resolve(editor);
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

const tmpHome = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-openfile-'));
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
  // Drain incoming bytes; the openFile handler never talks to the
  // daemon so we don't need to parse or respond.
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
  positionCalls.length = 0;
  selectionAssignments.length = 0;
  revealCalls.length = 0;
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

  const ws = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-openfile-ws-'));
  tmpDirs.push(ws);
  workspaceFolders = [{uri: makeUri(ws)}];

  // Real fixtures inside the workspace.
  const textFile = path.join(ws, 'src', 'main.py');
  fs.mkdirSync(path.dirname(textFile), {recursive: true});
  fs.writeFileSync(textFile, 'print("hello")\n');

  const imageFile = path.join(ws, 'assets', 'logo.png');
  fs.mkdirSync(path.dirname(imageFile), {recursive: true});
  // 8-byte PNG signature + IEND (enough to look like a binary file).
  fs.writeFileSync(
    imageFile,
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
  );

  const pdfFile = path.join(ws, 'docs', 'spec.pdf');
  fs.mkdirSync(path.dirname(pdfFile), {recursive: true});
  fs.writeFileSync(pdfFile, '%PDF-1.4\n%fakeeof\n');

  const dirPath = path.join(ws, 'src');

  const view = new SorcarSidebarView(makeUri(path.join(__dirname, '..')));
  const wv = makeWebviewView();
  view.resolveWebviewView(wv.webviewView, {}, {});
  wv.fireMessage({type: 'ready', tabId: 'tab1', restoredTabs: []});
  await waitForClient();

  // ---- 1. Text file ---------------------------------------------------
  clear();
  wv.fireMessage({type: 'openFile', path: 'src/main.py'});
  await waitFor(
    () => openedTextDocs.length === 1 && shownTextDocs.length === 1,
    'text file: openTextDocument + showTextDocument must be called',
  );
  assert.strictEqual(openedTextDocs[0], textFile);
  assert.strictEqual(shownTextDocs[0], textFile);
  assert.deepStrictEqual(
    executedCommands.filter(c => c.cmd === 'vscode.open'),
    [],
    'text file: vscode.open command must NOT be invoked',
  );
  console.log('  ok - text file opens via openTextDocument');

  // ---- 2. Text file with :line ----------------------------------------
  clear();
  wv.fireMessage({type: 'openFile', path: 'src/main.py', line: 7});
  await waitFor(
    () =>
      openedTextDocs.length === 1 &&
      shownTextDocs.length === 1 &&
      selectionAssignments.length === 1,
    'text file with line: openTextDocument + selection assignment expected',
  );
  // Production code passes ``line - 1`` to ``new vscode.Position(...)``
  // so the cursor lands on the 1-indexed line the user clicked.
  assert.strictEqual(
    selectionAssignments[0].line,
    6,
    'cursor must be on 0-indexed line 6 (== 1-indexed line 7)',
  );
  assert.strictEqual(revealCalls.length, 1, 'revealRange must fire once');
  console.log('  ok - text file with :line positions the cursor');

  // ---- 3. Image file → native viewer ---------------------------------
  clear();
  wv.fireMessage({type: 'openFile', path: 'assets/logo.png'});
  await waitFor(
    () =>
      executedCommands.some(
        c =>
          c.cmd === 'vscode.open' &&
          c.args.length >= 1 &&
          c.args[0] &&
          c.args[0].fsPath === imageFile,
      ),
    'image file: vscode.open command must be invoked',
  );
  assert.deepStrictEqual(
    openedTextDocs,
    [],
    'image file: openTextDocument must NOT be called (binary content)',
  );
  assert.deepStrictEqual(
    shownTextDocs,
    [],
    'image file: showTextDocument must NOT be called',
  );
  console.log('  ok - image file opens via native viewer (vscode.open)');

  // ---- 4. PDF file → native viewer -----------------------------------
  clear();
  wv.fireMessage({type: 'openFile', path: 'docs/spec.pdf'});
  await waitFor(
    () =>
      executedCommands.some(
        c =>
          c.cmd === 'vscode.open' &&
          c.args.length >= 1 &&
          c.args[0] &&
          c.args[0].fsPath === pdfFile,
      ),
    'pdf file: vscode.open command must be invoked',
  );
  assert.deepStrictEqual(
    openedTextDocs,
    [],
    'pdf file: openTextDocument must NOT be called',
  );
  console.log('  ok - pdf file opens via native viewer (vscode.open)');

  // ---- 5. Outside workspace → refused --------------------------------
  clear();
  const outside = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-outside-'));
  tmpDirs.push(outside);
  const outsideFile = path.join(outside, 'evil.py');
  fs.writeFileSync(outsideFile, 'x = 1\n');
  wv.fireMessage({type: 'openFile', path: outsideFile});
  // Give the handler a tick to run.
  await new Promise(r => setTimeout(r, 100));
  assert.deepStrictEqual(
    openedTextDocs,
    [],
    'outside-workspace path: openTextDocument must NOT be called',
  );
  assert.deepStrictEqual(
    executedCommands.filter(c => c.cmd === 'vscode.open'),
    [],
    'outside-workspace path: vscode.open must NOT be invoked',
  );
  console.log('  ok - outside-workspace path is refused');

  // ---- 6. Non-existent path → refused --------------------------------
  clear();
  wv.fireMessage({type: 'openFile', path: 'no/such/file.py'});
  await new Promise(r => setTimeout(r, 100));
  assert.deepStrictEqual(
    openedTextDocs,
    [],
    'non-existent path: openTextDocument must NOT be called',
  );
  assert.deepStrictEqual(
    executedCommands.filter(c => c.cmd === 'vscode.open'),
    [],
    'non-existent path: vscode.open must NOT be invoked',
  );
  console.log('  ok - non-existent path is refused');

  // ---- 7. Directory → refused ----------------------------------------
  clear();
  wv.fireMessage({type: 'openFile', path: 'src'});
  // ``src`` resolves to a real path inside the workspace but is a
  // directory; the handler must check ``isFile()`` and refuse.
  await new Promise(r => setTimeout(r, 100));
  assert.deepStrictEqual(
    openedTextDocs,
    [],
    'directory: openTextDocument must NOT be called',
  );
  assert.deepStrictEqual(
    executedCommands.filter(c => c.cmd === 'vscode.open'),
    [],
    'directory: vscode.open must NOT be invoked',
  );
  console.log('  ok - directory is refused (isFile guard)');

  // Cleanup
  view.dispose();
  server.close();
  for (const dir of tmpDirs.slice().reverse()) {
    fs.rmSync(dir, {recursive: true, force: true});
  }
}

(async () => {
  try {
    await runTests();
    console.log('\n7 passed, 0 failed');
    process.exit(0);
  } catch (err) {
    console.error('FAIL:', err && err.stack ? err.stack : err);
    process.exit(1);
  }
})();
