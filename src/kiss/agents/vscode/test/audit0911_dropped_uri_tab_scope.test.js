// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end regression test for review-vscode.md #3: the dropped-URI
// round trip (webview `resolveDroppedPaths` -> extension host ->
// `droppedPaths` reply) carried no tab id, and the reply handler always
// edited the CURRENTLY VISIBLE composer.  Interleaving: drop a VS Code
// editor tab's URI on chat tab A, switch to chat tab B before the host's
// reply is delivered, and A's relative file path lands at B's caret.
//
// The fix stamps the originating tab id on the request, the host echoes
// it, and the webview rejects a reply whose owner is no longer on screen.
//
// Part 1 drives the real chat.html + api.js + main.js in JSDOM.
// Part 2 drives the real compiled SorcarSidebarView against a stubbed
// vscode module and asserts the host echoes the tab id.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const Module = require('module');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');
const OUT_DIR = path.join(__dirname, '..', 'out');

// ---------------------------------------------------------------- part 1

function makeWebview() {
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
  let state;
  win.acquireVsCodeApi = function () {
    return {
      postMessage: msg => posted.push(msg),
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };

  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(
    fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8') +
      '\n//# sourceURL=audit0911-droppeduri-main.js',
  );
  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function dropUris(win, uris) {
  const container = win.document.getElementById('input-container');
  assert.ok(container, '#input-container must exist');
  const ev = new win.Event('drop', {bubbles: true, cancelable: true});
  Object.defineProperty(ev, 'dataTransfer', {
    value: {
      getData: t => (t === 'text/uri-list' ? uris.join('\n') : ''),
      files: [],
    },
  });
  container.dispatchEvent(ev);
}

function switchTab(win, tabId) {
  const el = win.document.querySelector(
    `.chat-tab[data-tab-id=${JSON.stringify(tabId)}]`,
  );
  assert.ok(el, `tab ${tabId} must be in the tab bar`);
  el.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.strictEqual(win._testApi.getActiveTabId(), tabId);
}

function webviewSide() {
  const {win, posted} = makeWebview();
  win._testApi.endLaunch();
  const inp = win.document.getElementById('task-input');
  const tabA = win._testApi.getActiveTabId();

  // A drop on tab A must stamp A as the request's owner.
  inp.value = 'fix ';
  inp.setSelectionRange(4, 4);
  dropUris(win, ['file:///ws/src/a.py']);
  const req = posted.filter(m => m.type === 'resolveDroppedPaths').pop();
  assert.ok(req, 'the drop posted a resolveDroppedPaths request');
  assert.strictEqual(
    req.tabId,
    tabA,
    'the request must carry its originating tab id',
  );

  // The user switches to tab B before the host's reply arrives.
  win._testApi.createNewTab();
  const tabB = win._testApi.getActiveTabId();
  assert.notStrictEqual(tabB, tabA);
  inp.value = 'b draft';

  // The late reply is addressed to A: it must NOT touch B's composer.
  send(win, {type: 'droppedPaths', paths: ['src/a.py'], tabId: tabA});
  assert.strictEqual(
    inp.value,
    'b draft',
    "tab A's dropped file path was inserted into tab B's composer",
  );

  // Back on A, a reply addressed to A inserts at A's caret.
  switchTab(win, tabA);
  assert.strictEqual(inp.value, 'fix ', "A's draft survived the switch");
  inp.setSelectionRange(4, 4);
  send(win, {type: 'droppedPaths', paths: ['src/a.py'], tabId: tabA});
  assert.strictEqual(
    inp.value,
    'fix ./src/a.py',
    'a reply addressed to the visible tab inserts the path',
  );

  // A legacy reply without a tab id keeps the old behavior (insert into
  // the visible composer).
  send(win, {type: 'droppedPaths', paths: ['legacy.py'], tabId: undefined});
  assert.ok(
    inp.value.indexOf('./legacy.py') >= 0,
    'an unaddressed reply still inserts into the visible composer',
  );

  console.log('  webview side: OK');
}

// ---------------------------------------------------------------- part 2

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

function hostSide() {
  const vscodeStub = {
    workspace: {
      workspaceFolders: [{uri: makeUri('/ws')}],
      getConfiguration: () => ({get: () => ''}),
      onDidChangeWorkspaceFolders: () => ({dispose: () => {}}),
    },
    EventEmitter: StubEventEmitter,
    Uri: {
      file: p => makeUri(p),
      joinPath: (base, ...parts) => makeUri(path.join(base.fsPath, ...parts)),
      parse: s => makeUri(s.replace(/^file:\/\//, '')),
    },
    ViewColumn: {One: 1},
    window: {
      showInformationMessage: () => Promise.resolve(undefined),
      showWarningMessage: () => Promise.resolve(undefined),
      showErrorMessage: () => Promise.resolve(undefined),
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

  const sourcePath = path.join(OUT_DIR, 'SorcarSidebarView.js');
  assert.ok(
    fs.existsSync(sourcePath),
    `compiled extension missing: ${sourcePath} — run \`npm run compile\``,
  );
  const {SorcarSidebarView} = require(sourcePath);

  const recv = new StubEventEmitter();
  const postedByHost = [];
  const webviewView = {
    webview: {
      options: {},
      html: '',
      cspSource: 'vscode-resource:',
      asWebviewUri: uri => makeUri(uri.fsPath),
      postMessage: msg => {
        postedByHost.push(msg);
        return Promise.resolve(true);
      },
      onDidReceiveMessage: cb => recv.event(cb),
    },
    visible: true,
    show: () => {},
    onDidChangeVisibility: () => ({dispose: () => {}}),
    onDidDispose: () => ({dispose: () => {}}),
  };

  const view = new SorcarSidebarView(makeUri(path.join(__dirname, '..')));
  view.resolveWebviewView(webviewView, {}, {});

  recv.fire({
    type: 'resolveDroppedPaths',
    uris: ['file:///ws/src/a.py'],
    workDir: '/ws',
    tabId: 'tab-owner-A',
  });
  const reply = postedByHost.filter(m => m.type === 'droppedPaths').pop();
  assert.ok(reply, 'the host replied with droppedPaths');
  assert.strictEqual(
    JSON.stringify(reply.paths),
    JSON.stringify(['src/a.py']),
    'the host relativized the dropped URI',
  );
  assert.strictEqual(
    reply.tabId,
    'tab-owner-A',
    "the host's reply must preserve the request's originating tab id",
  );

  view.dispose();
  Module._resolveFilename = origResolve;
  console.log('  host side: OK');
}

function main() {
  webviewSide();
  hostSide();
  console.log('audit0911_dropped_uri_tab_scope: OK');
}

try {
  main();
  process.exit(0);
} catch (err) {
  console.error(err && err.stack ? err.stack : err);
  process.exit(1);
}
