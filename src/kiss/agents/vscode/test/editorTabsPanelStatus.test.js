// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end tests for editor-tabs mode's TAB STATUS (icon + title
// prefix) and finished-task reveal (out/SorcarPanelManager.js +
// out/SorcarSidebarView.js) against a real local-WSS daemon
// stub. The mode runs ONE chat editor tab whose webview mirrors the
// whole registry; `panelTitle {title, state}` describes the chat on
// screen and paints that single tab:
//  - `panelTitle {state:'running'}` swaps the tab ICON to the green ring
//    spinner (media/spinner-running.svg, SMIL-rotated like the
//    composer's wait spinner) and leaves the title undecorated;
//  - `state:'ok'` paints a steady green tick, `state:'fail'` a steady
//    red cross, and no state paints no prefix; all three show the KISS
//    logo icon again;
//  - `revealPanel` from the webview reveals the hosting editor tab
//    without stealing focus (preserveFocus);
//  - a serializer-revived panel drops the status prefix a previous
//    session persisted into its title, including the 🟢 / 🔴 circles
//    older versions used; a husk revived while a panel already exists
//    is dropped (one panel only).

'use strict';

const assert = require('assert');
const fs = require('fs');
const net = require('net');
const os = require('os');
const path = require('path');
const Module = require('module');
const {createFakeDaemon} = require('./fakeDaemon');

const EXT_ROOT = path.join(__dirname, '..');
const OUT_DIR = path.join(EXT_ROOT, 'out');
assert.ok(
  fs.existsSync(path.join(OUT_DIR, 'SorcarPanelManager.js')),
  'compiled extension missing — run `npm run compile` first',
);

const TICK = '\u2705 ';
const CROSS = '\u274C ';
const LEGACY_GREEN = '\u{1F7E2} ';
const LEGACY_RED = '\u{1F534} ';
// The braille frames older versions painted into a running title; a
// revived panel must still lose them.
const LEGACY_SPINNER_FRAME = '\u2839 ';
const KISS_ICON = path.join(EXT_ROOT, 'media', 'kiss-icon.svg');
const SPINNER_ICON = path.join(EXT_ROOT, 'media', 'spinner-running.svg');

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

const createdPanels = [];
let registeredSerializer = null;

function makeFakePanel(viewType, title) {
  const recvEmitter = new StubEventEmitter();
  const disposeEmitter = new StubEventEmitter();
  const viewStateEmitter = new StubEventEmitter();
  const panel = {
    viewType,
    title,
    iconPath: undefined,
    visible: true,
    active: true,
    revealCalls: [],
    disposed: false,
    webview: {
      options: {},
      html: '',
      cspSource: 'vscode-resource:',
      asWebviewUri: uri => makeUri(uri.fsPath),
      postMessage: () => Promise.resolve(true),
      onDidReceiveMessage: cb => recvEmitter.event(cb),
    },
    reveal: (column, preserveFocus) => {
      panel.revealCalls.push({column, preserveFocus});
    },
    onDidChangeViewState: cb => viewStateEmitter.event(cb),
    onDidDispose: cb => disposeEmitter.event(cb),
    dispose: () => {
      if (panel.disposed) return;
      panel.disposed = true;
      disposeEmitter.fire();
    },
    _recv: recvEmitter,
  };
  return panel;
}

const vscodeStub = {
  workspace: {
    workspaceFolders: [],
    getConfiguration: () => ({
      get: key => (key === 'editorTabsMode' ? true : ''),
      update: () => Promise.resolve(),
    }),
    onDidChangeWorkspaceFolders: () => ({dispose: () => {}}),
    openTextDocument: () =>
      Promise.resolve({uri: makeUri('/x'), getText: () => ''}),
    textDocuments: [],
  },
  EventEmitter: StubEventEmitter,
  Uri: {
    file: p => makeUri(p),
    joinPath: (base, ...parts) => makeUri(path.join(base.fsPath, ...parts)),
    parse: s => makeUri(s),
  },
  ProgressLocation: {Notification: 15},
  ViewColumn: {Active: -1, One: 1},
  window: {
    createWebviewPanel: (viewType, title) => {
      const panel = makeFakePanel(viewType, title);
      createdPanels.push(panel);
      return panel;
    },
    registerWebviewPanelSerializer: (viewType, serializer) => {
      registeredSerializer = {viewType, serializer};
      return {dispose: () => {}};
    },
    withProgress: (_opts, task) =>
      task(
        {report: () => {}},
        {onCancellationRequested: () => ({dispose: () => {}})},
      ),
    showInformationMessage: () => {},
    showWarningMessage: () => {},
    showErrorMessage: () => {},
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

const tmpHome = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-edstatus-'));
process.env.HOME = tmpHome;
process.env.USERPROFILE = tmpHome;
fs.mkdirSync(path.join(tmpHome, '.kiss'), {recursive: true});
const endpointPath = path.join(tmpHome, '.kiss', 'sorcar-local.json');

const server = createFakeDaemon(sock => {
  sock.on('data', () => {});
  sock.on('error', () => {});
});

async function waitFor(predicate, message) {
  for (let i = 0; i < 200; i++) {
    if (predicate()) return;
    await new Promise(r => setTimeout(r, 20));
  }
  throw new Error(message || 'waitFor timed out');
}

const {SorcarPanelManager, CHAT_PANEL_VIEW_TYPE} = require(
  path.join(OUT_DIR, 'SorcarPanelManager.js'),
);

/** Revive *panel* through the registered serializer, as the workbench
 * does after a window reload. */
function revive(panel) {
  panel.active = false;
  return registeredSerializer.serializer.deserializeWebviewPanel(panel, {});
}

async function runTest() {
  server.listen(endpointPath);
  const manager = new SorcarPanelManager(vscodeStub.Uri.file(EXT_ROOT));
  manager.registerSerializer();
  assert.strictEqual(registeredSerializer.viewType, CHAT_PANEL_VIEW_TYPE);

  const controller = manager.open();
  assert.strictEqual(createdPanels.length, 1, 'open() creates the panel');
  assert.strictEqual(manager.controller(), controller);
  const panel = createdPanels[0];
  assert.strictEqual(panel.viewType, CHAT_PANEL_VIEW_TYPE);
  assert.ok(
    /<body[^>]*class="editor-tab-mode"/.test(panel.webview.html),
    'the panel webview runs main.js in editor-tab mode',
  );
  assert.ok(
    !/data-kiss-tab-id=/.test(panel.webview.html),
    'the single panel is pinned to no root tab id',
  );

  // --- no state: plain title -------------------------------------------
  panel._recv.fire({type: 'panelTitle', title: 'fix the bug', state: ''});
  await waitFor(
    () => panel.title === 'fix the bug',
    'a stateless panelTitle must set the plain title',
  );

  assert.strictEqual(
    panel.iconPath && panel.iconPath.fsPath,
    KISS_ICON,
    'an idle panel shows the KISS logo as its tab icon',
  );

  // --- running: the green ring spinner becomes the tab icon -------------
  panel._recv.fire({
    type: 'panelTitle',
    title: 'fix the bug',
    state: 'running',
  });
  await waitFor(
    () => panel.iconPath && panel.iconPath.fsPath === SPINNER_ICON,
    'a running task must swap the tab icon to the ring spinner',
  );
  assert.strictEqual(
    panel.title,
    'fix the bug',
    'the running title carries no text spinner: the icon spins',
  );
  const spinnerUri = panel.iconPath;
  // A repaint that keeps the state must not push a fresh Uri (the
  // workbench would repaint the tab for nothing).
  panel._recv.fire({
    type: 'panelTitle',
    title: 'fix the bug (step 2)',
    state: 'running',
  });
  await waitFor(
    () => panel.title === 'fix the bug (step 2)',
    'a running retitle must repaint the title',
  );
  assert.strictEqual(
    panel.iconPath,
    spinnerUri,
    'a running retitle must keep the very same spinner icon Uri',
  );
  // The icon itself: the ring the composer's wait spinner draws (a
  // faint track and one bright arc, 2px thick, 12px across inside the
  // 16px icon box), in green, turned once every 0.8s by SMIL so the
  // workbench animates it as a plain CSS background image.
  const svg = fs.readFileSync(SPINNER_ICON, 'utf8');
  assert.ok(/<svg[^>]*viewBox="0 0 16 16"/.test(svg), 'a 16x16 icon box');
  assert.ok(
    /<circle[^>]*r="5"[^>]*stroke="#3fb950"[^>]*stroke-opacity="0\.25"[^>]*stroke-width="2"/.test(svg),
    'a faint green 2px track of 12px outer diameter: ' + svg,
  );
  assert.ok(
    /<path[^>]*stroke="#3fb950"[^>]*stroke-width="2"/.test(svg),
    'a bright green 2px leading arc: ' + svg,
  );
  assert.ok(
    /<animateTransform[^>]*type="rotate"[^>]*to="360 8 8"[^>]*dur="0\.8s"[^>]*repeatCount="indefinite"/.test(svg),
    'the arc turns a full circle every 0.8s, forever: ' + svg,
  );

  // --- ok: steady green tick ---------------------------------------------
  panel._recv.fire({type: 'panelTitle', title: 'fix the bug', state: 'ok'});
  await waitFor(
    () => panel.title === TICK + 'fix the bug',
    'a successful task must paint the green tick',
  );
  assert.strictEqual(
    panel.iconPath && panel.iconPath.fsPath,
    KISS_ICON,
    'a finished task shows the KISS logo again',
  );

  // --- fail: steady red cross --------------------------------------------
  panel._recv.fire({type: 'panelTitle', title: 'fix the bug', state: 'fail'});
  await waitFor(
    () => panel.title === CROSS + 'fix the bug',
    'a failed task must paint the red cross',
  );
  assert.strictEqual(
    panel.iconPath && panel.iconPath.fsPath,
    KISS_ICON,
    'a failed task shows the KISS logo, not the spinner',
  );

  // --- switching chats in place repaints from the shown chat ------------
  // The webview switched to another chat of the registry: its title
  // and status replace the previous chat's on the one tab.
  panel._recv.fire({type: 'panelTitle', title: 'other chat', state: ''});
  await waitFor(
    () => panel.title === 'other chat',
    'the tab describes whichever chat is on screen',
  );
  assert.strictEqual(panel.iconPath && panel.iconPath.fsPath, KISS_ICON);

  // --- an empty title falls back to the product name --------------------
  panel._recv.fire({type: 'panelTitle', title: '', state: ''});
  await waitFor(
    () => panel.title === require(path.join(OUT_DIR, 'brand.js')).PRODUCT_NAME,
    'a new chat with no title shows the product name',
  );

  // --- revealPanel brings the tab forward without stealing focus --------
  assert.strictEqual(panel.revealCalls.length, 0);
  panel._recv.fire({type: 'revealPanel'});
  await waitFor(
    () => panel.revealCalls.length === 1,
    'revealPanel must reveal the hosting editor tab',
  );
  assert.strictEqual(
    panel.revealCalls[0].preserveFocus,
    true,
    'the reveal must not steal keyboard focus',
  );

  // --- a legitimate tick-leading title is never corrupted ----------------
  panel._recv.fire({
    type: 'panelTitle',
    title: TICK + 'deploy status',
    state: 'fail',
  });
  await waitFor(
    () => panel.title === CROSS + TICK + 'deploy status',
    'a chat legitimately titled with a leading tick must keep it',
  );

  // --- a husk revived while the panel exists is dropped ------------------
  const duplicate = makeFakePanel(CHAT_PANEL_VIEW_TYPE, TICK + 'old chat');
  await revive(duplicate);
  assert.ok(duplicate.disposed, 'one panel only: the revived husk is dropped');
  assert.strictEqual(
    manager.controller(),
    controller,
    'the live panel keeps its controller',
  );

  // --- a revived panel drops the persisted status prefix ----------------
  // The panel closed before the reload; each revival adopts the husk
  // as THE panel, so the previous one is closed before the next.
  panel.dispose();
  assert.strictEqual(manager.controller(), undefined);
  for (const [prefix, what] of [
    [TICK, 'tick'],
    [CROSS, 'cross'],
    [LEGACY_SPINNER_FRAME, 'legacy spinner frame'],
    [LEGACY_GREEN, 'legacy green circle'],
    [LEGACY_RED, 'legacy red circle'],
  ]) {
    const revived = makeFakePanel(CHAT_PANEL_VIEW_TYPE, prefix + 'old chat');
    await revive(revived);
    assert.ok(!revived.disposed, 'the husk is adopted when no panel exists');
    assert.strictEqual(
      revived.title,
      'old chat',
      'revival must strip the previous session ' + what,
    );
    assert.strictEqual(
      revived.iconPath && revived.iconPath.fsPath,
      KISS_ICON,
      'a revived panel gets the KISS logo icon',
    );
    assert.ok(manager.controller(), 'a revived panel gets a controller');
    revived.dispose();
  }

  // --- ... and leaves braille that is not a spinner frame alone ---------
  const braille = makeFakePanel(CHAT_PANEL_VIEW_TYPE, '\u28FF load-test notes');
  await revive(braille);
  assert.strictEqual(
    braille.title,
    '\u28FF load-test notes',
    'revival must only strip the frames the spinner can have persisted',
  );
  braille.dispose();

  // --- ... but strips only the ONE decoration it added itself ------------
  const revived2 = makeFakePanel(
    CHAT_PANEL_VIEW_TYPE,
    CROSS + TICK + 'deploy status',
  );
  await revive(revived2);
  assert.strictEqual(
    revived2.title,
    TICK + 'deploy status',
    'revival must keep a legitimate leading tick in the chat title',
  );
  // A revived panel is painted like any other.
  revived2._recv.fire({
    type: 'panelTitle',
    title: 'deploy status',
    state: 'running',
  });
  await waitFor(
    () => revived2.iconPath && revived2.iconPath.fsPath === SPINNER_ICON,
    'a revived panel paints the running state',
  );
  assert.strictEqual(revived2.title, 'deploy status');

  manager.dispose();
  assert.ok(
    !revived2.disposed,
    'teardown leaves the panel standing for the serializer',
  );
}
/**
 * An installed (packaged) extension names its icons through
 * media/hashed/index.json (scripts/hash-icons.js); the tab icons must
 * follow that mapping so a same-version rebuild gets uncached URLs.
 */
async function runHashedIconsTest() {
  const packaged = fs.mkdtempSync(
    path.join(os.tmpdir(), 'kiss-edstatus-packaged-'),
  );
  fs.cpSync(path.join(EXT_ROOT, 'media'), path.join(packaged, 'media'), {
    recursive: true,
  });
  const hashedKiss = 'media/hashed/kiss-icon-deadbeef.svg';
  const hashedSpinner = 'media/hashed/spinner-running-0badf00d.svg';
  fs.mkdirSync(path.join(packaged, 'media', 'hashed'));
  fs.copyFileSync(KISS_ICON, path.join(packaged, hashedKiss));
  fs.copyFileSync(SPINNER_ICON, path.join(packaged, hashedSpinner));
  fs.writeFileSync(
    path.join(packaged, 'media', 'hashed', 'index.json'),
    JSON.stringify({
      'media/kiss-icon.svg': hashedKiss,
      'media/spinner-running.svg': hashedSpinner,
    }),
  );
  const manager = new SorcarPanelManager(vscodeStub.Uri.file(packaged));
  const before = createdPanels.length;
  manager.open();
  const panel = createdPanels[before];
  panel._recv.fire({type: 'panelTitle', title: 'hashed', state: ''});
  await waitFor(() => panel.title === 'hashed');
  assert.strictEqual(
    panel.iconPath.fsPath,
    path.join(packaged, hashedKiss),
    'idle icon follows index.json',
  );
  panel._recv.fire({type: 'panelTitle', title: 'hashed', state: 'running'});
  await waitFor(
    () => panel.iconPath.fsPath === path.join(packaged, hashedSpinner),
    'spinner follows index.json',
  );
  manager.dispose();
  fs.rmSync(packaged, {recursive: true, force: true});
}

runTest()
  .then(runHashedIconsTest)
  .then(() => {
    console.log('editorTabsPanelStatus: all tests passed');
    server.close();
    process.exit(0);
  })
  .catch(err => {
    console.error(err);
    process.exit(1);
  });
