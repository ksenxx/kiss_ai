// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// END-TO-END test for the Cmd+E / Ctrl+E (``kissSorcar.runSelection``)
// keybinding: the REAL compiled extension entry point
// (``out/extension.js`` activate()) + the REAL
// ``out/SorcarSidebarView.js`` wired to the REAL webview
// (``media/main.js`` running in jsdom) over a real Unix-domain-socket
// stub daemon.  Only the ``vscode`` module and heavyweight activation
// side modules (DependencyInstaller, UpdateChecker, gitApi, kissPaths,
// WebviewNotifications) are stubbed.
//
// REQUIREMENT (the bug): when Cmd+E is pressed with text highlighted in
// the editor, the highlighted text must be
//   (1) copied out of the editor selection,
//   (2) PASTED INTO THE INPUT TEXTBOX of the chat webview, and
//   (3) SUBMITTED to the agent for execution through the webview's
//       normal send path (so tab id, selected model, running-task
//       steering etc. all behave exactly like pressing Send).
//
// The buggy behavior: ``runSelection`` bypassed the webview entirely —
// the text never appeared in the input textbox, the submit did not go
// through the webview's send path (no tabId, wrong model), and when the
// sidebar webview was not yet resolved the ``setTaskText`` message was
// silently dropped.
//
// Run directly with ``node`` (after ``npm run compile``):
//
//     node src/kiss/agents/vscode/test/runSelectionSubmit.test.js

'use strict';

const assert = require('assert');
const fs = require('fs');
const net = require('net');
const os = require('os');
const path = require('path');
const Module = require('module');
const {JSDOM} = require('jsdom');

const EXT_ROOT = path.join(__dirname, '..');
const OUT_DIR = path.join(EXT_ROOT, 'out');
const MEDIA = path.join(EXT_ROOT, 'media');

// ---- vscode module stub ----------------------------------------------
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

function makeDisposable() {
  return {dispose: () => {}};
}

function makeMemento(seed) {
  const store = new Map(Object.entries(seed || {}));
  return {
    get: (key, def) => (store.has(key) ? store.get(key) : def),
    update: (key, value) => {
      if (value === undefined) store.delete(key);
      else store.set(key, value);
      return Promise.resolve();
    },
  };
}

let workspaceFolders = [];
const registeredCommands = new Map();
const executedCommands = [];
// Optional hook: lets a test simulate VS Code resolving the sidebar
// webview when ``kissSorcar.chatViewSecondary.focus`` is executed.
let onFocusViewCommand = null;

const vscodeStub = {
  workspace: {
    get workspaceFolders() {
      return workspaceFolders;
    },
    getConfiguration: () => ({get: () => 'stub-default-model'}),
    onDidChangeWorkspaceFolders: () => makeDisposable(),
    openTextDocument: () =>
      Promise.resolve({uri: makeUri('/x'), getText: () => ''}),
    textDocuments: [],
    asRelativePath: p =>
      String(p && p.fsPath ? p.fsPath : p).replace(/^\//, ''),
  },
  EventEmitter: StubEventEmitter,
  Uri: {
    file: p => makeUri(p),
    joinPath: (base, ...parts) => makeUri(path.join(base.fsPath, ...parts)),
    parse: s => makeUri(s),
  },
  ProgressLocation: {Notification: 15},
  ViewColumn: {One: 1},
  TreeItem: class {
    constructor(label) {
      this.label = label;
    }
  },
  TabInputText: class {},
  window: {
    registerWebviewViewProvider: () => makeDisposable(),
    createTreeView: () => ({
      onDidChangeVisibility: () => makeDisposable(),
      dispose: () => {},
    }),
    withProgress: (_opts, task) =>
      task(
        {report: () => {}},
        {onCancellationRequested: () => makeDisposable()},
      ),
    showInformationMessage: () => Promise.resolve(undefined),
    showErrorMessage: () => Promise.resolve(undefined),
    showWarningMessage: () => Promise.resolve(undefined),
    showTextDocument: () => Promise.resolve({}),
    createTerminal: () => ({sendText: () => {}, show: () => {}}),
    onDidChangeVisibleTextEditors: () => makeDisposable(),
    visibleTextEditors: [],
    activeTextEditor: undefined,
    tabGroups: {all: [], close: () => Promise.resolve(true)},
  },
  commands: {
    registerCommand: (name, fn) => {
      registeredCommands.set(name, fn);
      return makeDisposable();
    },
    executeCommand: (cmd, ...args) => {
      executedCommands.push({cmd, args});
      if (cmd === 'kissSorcar.chatViewSecondary.focus' && onFocusViewCommand) {
        onFocusViewCommand();
      }
      return Promise.resolve();
    },
  },
  extensions: {getExtension: () => undefined},
};

const origResolve = Module._resolveFilename;
Module._resolveFilename = function (request, parent, ...rest) {
  if (request === 'vscode') return require.resolve('./_vscode-stub.js');
  return origResolve.call(this, request, parent, ...rest);
};
global.__kissVscodeStub = vscodeStub;

// ---- stub the heavyweight activation-side modules ---------------------
const notifications = [];

function stubModule(filePath, exports) {
  const fakeMod = new Module(filePath);
  fakeMod.filename = filePath;
  fakeMod.loaded = true;
  fakeMod.exports = exports;
  require.cache[filePath] = fakeMod;
}

stubModule(path.join(OUT_DIR, 'DependencyInstaller.js'), {
  ensureLocalBinInPath: () => {},
  ensureDependencies: () => Promise.resolve(),
});
stubModule(path.join(OUT_DIR, 'gitApi.js'), {
  getGitApi: () => Promise.resolve(undefined),
});
stubModule(path.join(OUT_DIR, 'kissPaths.js'), {
  findKissProject: () => undefined,
});
stubModule(path.join(OUT_DIR, 'UpdateChecker.js'), {
  checkForExtensionUpdate: async () => ({
    checked: false,
    notified: false,
    latest: null,
    current: null,
    reason: 'test',
  }),
});
stubModule(path.join(OUT_DIR, 'WebviewNotifications.js'), {
  setWebviewNotificationPoster: () => {},
  resolveWebviewNotificationAction: () => {},
  showInformationNotification: msg => {
    notifications.push(msg);
    return Promise.resolve(undefined);
  },
  showWarningNotification: () => Promise.resolve(undefined),
  showErrorNotification: () => Promise.resolve(undefined),
  withWebviewNotificationProgress: (_msg, task) =>
    task({report: () => {}}, {onCancellationRequested: () => {}}),
});

// ---- isolated HOME + stub daemon (UDS) --------------------------------
const tmpHome = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-cmde-'));
process.env.HOME = tmpHome;
process.env.USERPROFILE = tmpHome;
fs.mkdirSync(path.join(tmpHome, '.kiss'), {recursive: true});
const sockPath = path.join(tmpHome, '.kiss', 'sorcar.sock');

if (process.platform === 'win32') {
  console.log('  skipped on win32 (UDS test)');
  fs.rmSync(tmpHome, {recursive: true, force: true});
  process.exit(0);
}

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
      }
    }
  });
});

// ---- jsdom webview running the REAL main.js ---------------------------
let persistedState;

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
  };
}

/** Editor stub whose ``getText(selection)`` returns ``text``. */
function makeEditor(ws, text) {
  return {
    document: {
      uri: makeUri(path.join(ws, 'example.ts')),
      getText: sel => (sel === undefined ? text : text),
    },
    selection: {
      start: {line: 0, character: 0},
      end: {line: 0, character: text.length},
    },
  };
}

const sleep = ms => new Promise(r => setTimeout(r, ms));

async function runTests() {
  await new Promise((res, rej) =>
    server.listen(sockPath, err => (err ? rej(err) : res())),
  );

  const extensionPath = path.join(OUT_DIR, 'extension.js');
  assert.ok(
    fs.existsSync(extensionPath),
    `compiled extension missing: ${extensionPath} — run \`npm run compile\` first`,
  );
  delete require.cache[require.resolve(extensionPath)];
  const ext = require(extensionPath);

  const ws = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-cmde-ws-'));
  workspaceFolders = [{uri: makeUri(ws)}];

  // Capture the webview-view provider that activate() registers so the
  // test can resolve it exactly like VS Code does.
  let provider = null;
  vscodeStub.window.registerWebviewViewProvider = (_id, p) => {
    provider = p;
    return makeDisposable();
  };

  const context = {
    subscriptions: [],
    extensionUri: makeUri(EXT_ROOT),
    extensionPath: EXT_ROOT,
    // Pre-set the first-launch gates so activation does not schedule the
    // auto-open / auto-widen timers that would interfere with the test.
    workspaceState: makeMemento({firstLaunchDone: true, sidebarWidened: true}),
    globalState: makeMemento(),
  };
  ext.activate(context);
  assert.ok(provider, 'activate() must register the webview view provider');
  assert.ok(
    registeredCommands.has('kissSorcar.runSelection'),
    'activate() must register kissSorcar.runSelection (the Cmd+E command)',
  );

  const cmdE = () => registeredCommands.get('kissSorcar.runSelection')();

  // ================================================================
  // Scenario 1 — webview already open: Cmd+E pastes the highlighted
  // text into the chat input textbox and submits it through the
  // webview's send path (daemon gets a tab-stamped ``run``).
  // ================================================================
  const ctx1 = buildWebviewWindow();
  const wvv1 = makeWebviewView(ctx1);
  provider.resolveWebviewView(wvv1.webviewView, {}, {});
  wvv1.wire();
  evalWebviewScripts(ctx1.win);
  await sleep(50);

  // Record every value the chat input textbox goes through so the test
  // can prove the selection was PASTED into it (sendMessage clears the
  // textbox afterwards, exactly like a user pressing Send).
  const inp1 = ctx1.win.document.getElementById('task-input');
  const inputValues = [];
  inp1.addEventListener('input', () => inputValues.push(inp1.value));

  const SEL1 = 'refactor this function to be pure';
  vscodeStub.window.activeTextEditor = makeEditor(ws, `  ${SEL1}\n`);
  daemonCmds.length = 0;
  await cmdE();
  await sleep(300);

  assert.ok(
    inputValues.includes(SEL1),
    'BUG: Cmd+E must PASTE the highlighted text into the chat input ' +
      'textbox (input values seen: ' +
      JSON.stringify(inputValues) +
      ')',
  );
  assert.strictEqual(
    inp1.value,
    '',
    'the input textbox must be cleared after the submit (like Send)',
  );
  const runs1 = daemonCmds.filter(c => c.type === 'run');
  assert.strictEqual(
    runs1.length,
    1,
    'BUG: Cmd+E must SUBMIT the pasted text to the agent (daemon ' +
      'commands: ' +
      JSON.stringify(daemonCmds.map(c => c.type)) +
      ')',
  );
  assert.strictEqual(runs1[0].prompt, SEL1, 'run prompt must be the selection');
  const TAB1 = runs1[0].tabId;
  assert.ok(
    TAB1,
    'BUG: the run must be submitted through the webview send path, so ' +
      'it must carry the webview tab id (got: ' +
      JSON.stringify(runs1[0]) +
      ')',
  );

  // ================================================================
  // Scenario 2 — a task is RUNNING in the active tab: Cmd+E pastes and
  // submits as a steering follow-up (``appendUserMessage``) exactly
  // like typing into the box and pressing Send while running.
  // ================================================================
  const SEL2 = 'also add unit tests for it';
  vscodeStub.window.activeTextEditor = makeEditor(ws, SEL2);
  daemonCmds.length = 0;
  inputValues.length = 0;
  await cmdE();
  await sleep(300);

  assert.ok(
    inputValues.includes(SEL2),
    'BUG: Cmd+E while a task runs must still paste into the input ' +
      'textbox (input values seen: ' +
      JSON.stringify(inputValues) +
      ')',
  );
  const appends = daemonCmds.filter(c => c.type === 'appendUserMessage');
  assert.strictEqual(
    appends.length,
    1,
    'BUG: Cmd+E while a task runs must reach the agent as an ' +
      'appendUserMessage steering message (daemon commands: ' +
      JSON.stringify(daemonCmds.map(c => c.type)) +
      ')',
  );
  assert.strictEqual(appends[0].prompt, SEL2);
  assert.strictEqual(appends[0].tabId, TAB1);

  // ================================================================
  // Scenario 3 — no editor / empty selection: a notification is shown
  // and nothing is pasted or submitted.
  // ================================================================
  daemonReply({type: 'status', running: false, tabId: TAB1});
  await sleep(80);
  daemonCmds.length = 0;
  inputValues.length = 0;
  notifications.length = 0;

  vscodeStub.window.activeTextEditor = undefined;
  await cmdE();
  vscodeStub.window.activeTextEditor = makeEditor(ws, '   \n  ');
  await cmdE();
  await sleep(200);
  assert.strictEqual(
    notifications.filter(n => /no text selected/i.test(n)).length,
    1,
    'an empty selection must show the "No text selected" notification',
  );
  assert.strictEqual(inputValues.length, 0, 'nothing pasted for empty sel');
  assert.strictEqual(
    daemonCmds.filter(c => c.type === 'run' || c.type === 'appendUserMessage')
      .length,
    0,
    'nothing submitted for empty selection / missing editor',
  );

  // A stray insertAndSubmit with no text must be a no-op in the webview.
  wvv1.webviewView.webview.postMessage({type: 'insertAndSubmit', text: ''});
  await sleep(100);
  assert.strictEqual(
    daemonCmds.filter(c => c.type === 'run' || c.type === 'appendUserMessage')
      .length,
    0,
    'insertAndSubmit without text must not submit anything',
  );

  if (typeof ext.deactivate === 'function') ext.deactivate();
  ctx1.win.close();

  // ================================================================
  // Scenario 4 — the sidebar webview is NOT yet resolved when Cmd+E is
  // pressed: the command must open/resolve the webview (via the
  // ``kissSorcar.chatViewSecondary.focus`` command, like VS Code does),
  // paste the selection into the input textbox and submit it.  The old
  // code silently dropped every message here.
  // ================================================================
  registeredCommands.clear();
  let provider2 = null;
  vscodeStub.window.registerWebviewViewProvider = (_id, p) => {
    provider2 = p;
    return makeDisposable();
  };
  const context2 = {
    subscriptions: [],
    extensionUri: makeUri(EXT_ROOT),
    extensionPath: EXT_ROOT,
    workspaceState: makeMemento({firstLaunchDone: true, sidebarWidened: true}),
    globalState: makeMemento(),
  };
  delete require.cache[require.resolve(extensionPath)];
  const ext2 = require(extensionPath);
  ext2.activate(context2);
  assert.ok(provider2, 'second activate() must register the provider');

  persistedState = undefined; // fresh webview state
  const ctx2 = buildWebviewWindow();
  const wvv2 = makeWebviewView(ctx2);
  const lateInputValues = [];
  onFocusViewCommand = () => {
    if (ctx2.resolved) return;
    ctx2.resolved = true;
    provider2.resolveWebviewView(wvv2.webviewView, {}, {});
    wvv2.wire();
    evalWebviewScripts(ctx2.win);
    const inp2 = ctx2.win.document.getElementById('task-input');
    inp2.addEventListener('input', () => lateInputValues.push(inp2.value));
  };

  const SEL4 = 'document this module';
  vscodeStub.window.activeTextEditor = makeEditor(ws, SEL4);
  daemonCmds.length = 0;
  await registeredCommands.get('kissSorcar.runSelection')();
  await sleep(500);

  assert.ok(ctx2.resolved, 'Cmd+E must resolve/open the sidebar webview');
  assert.ok(
    lateInputValues.includes(SEL4),
    'BUG: with the sidebar closed, Cmd+E must open it and paste the ' +
      'selection into the input textbox (input values seen: ' +
      JSON.stringify(lateInputValues) +
      ')',
  );
  const runs4 = daemonCmds.filter(c => c.type === 'run');
  assert.strictEqual(
    runs4.length,
    1,
    'BUG: with the sidebar closed, Cmd+E must still submit the ' +
      'selection to the agent (daemon commands: ' +
      JSON.stringify(daemonCmds.map(c => c.type)) +
      ')',
  );
  assert.strictEqual(runs4[0].prompt, SEL4);
  assert.ok(runs4[0].tabId, 'the late-resolved run must carry a tab id');

  if (typeof ext2.deactivate === 'function') ext2.deactivate();
  ctx2.win.close();

  // ================================================================
  // Scenario 5 — cold-open race: the webview VIEW resolves quickly but
  // its SCRIPT (media/main.js) loads slowly (600ms — well past the
  // 150ms show delay in focusChatInput).  Cmd+E must wait for the
  // webview's ``ready`` handshake instead of posting into a webview
  // with no message listener yet — otherwise the paste + submit is
  // silently dropped.
  // ================================================================
  registeredCommands.clear();
  let provider3 = null;
  vscodeStub.window.registerWebviewViewProvider = (_id, p) => {
    provider3 = p;
    return makeDisposable();
  };
  const context3 = {
    subscriptions: [],
    extensionUri: makeUri(EXT_ROOT),
    extensionPath: EXT_ROOT,
    workspaceState: makeMemento({firstLaunchDone: true, sidebarWidened: true}),
    globalState: makeMemento(),
  };
  delete require.cache[require.resolve(extensionPath)];
  const ext3 = require(extensionPath);
  ext3.activate(context3);
  assert.ok(provider3, 'third activate() must register the provider');

  persistedState = undefined; // fresh webview state
  const ctx3 = buildWebviewWindow();
  const wvv3 = makeWebviewView(ctx3);
  const slowInputValues = [];
  onFocusViewCommand = () => {
    if (ctx3.resolved) return;
    ctx3.resolved = true;
    // The VIEW resolves immediately (like VS Code), but the webview
    // SCRIPT only finishes loading 600ms later.
    provider3.resolveWebviewView(wvv3.webviewView, {}, {});
    wvv3.wire();
    setTimeout(() => {
      evalWebviewScripts(ctx3.win);
      const inp3 = ctx3.win.document.getElementById('task-input');
      inp3.addEventListener('input', () => slowInputValues.push(inp3.value));
    }, 600);
  };

  const SEL5 = 'summarize the selected code';
  vscodeStub.window.activeTextEditor = makeEditor(ws, SEL5);
  daemonCmds.length = 0;
  await registeredCommands.get('kissSorcar.runSelection')();
  await sleep(500);

  assert.ok(
    slowInputValues.includes(SEL5),
    'BUG (race): Cmd+E right after a cold sidebar open must wait for ' +
      'the webview ready handshake and still paste the selection into ' +
      'the input textbox (input values seen: ' +
      JSON.stringify(slowInputValues) +
      ')',
  );
  const runs5 = daemonCmds.filter(c => c.type === 'run');
  assert.strictEqual(
    runs5.length,
    1,
    'BUG (race): the slow-loading webview must still submit the ' +
      'selection to the agent (daemon commands: ' +
      JSON.stringify(daemonCmds.map(c => c.type)) +
      ')',
  );
  assert.strictEqual(runs5[0].prompt, SEL5);
  assert.ok(runs5[0].tabId, 'the slow-webview run must carry a tab id');

  if (typeof ext3.deactivate === 'function') ext3.deactivate();
  ctx3.win.close();

  const wsDir = ws;
  fs.rmSync(wsDir, {recursive: true, force: true});
}

function cleanup() {
  try {
    if (lastServerSock) lastServerSock.destroy();
  } catch {
    // best-effort teardown
  }
  try {
    server.close();
  } catch {
    // best-effort teardown
  }
  try {
    fs.unlinkSync(sockPath);
  } catch {
    // best-effort teardown
  }
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
