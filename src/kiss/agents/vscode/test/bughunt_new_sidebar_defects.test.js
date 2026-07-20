// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// Bug-hunt regression tests for three SorcarSidebarView defects
// (harness mirrors autocommitProgressSticky.test.js: real compiled
// SorcarSidebarView + AgentClient against a real UDS daemon stub; only
// the ``vscode`` module is stubbed).
//
// Defect 1 — superseded action-progress toast never closes.
//   ``_showActionProgress`` blindly overwrote the pending resolver in
//   ``_worktreeActionResolves`` when a second action arrived for the
//   same tab.  The safety timeout of the FIRST dialog checks map
//   identity (``resolveMap.get(tabId) === resolve``) before resolving,
//   so once overwritten neither the completion event nor the timeout
//   could ever close the first toast — it stayed on screen forever.
//
// Defect 2 — commit-message generations cross-talk between tabs.
//   ``_onCommitMessage`` events carried no tabId, so
//   ``generateCommitMessage(token, tabId)`` resolved its promise on ANY
//   commitMessage reply — a reply for tab A settled tab B's pending
//   generation and deleted tab B's ``_commitPendingTabs`` entry while
//   its generation was still in flight (breaking the documented
//   per-tab independence, and letting a chat-tab commit message reach
//   the extension's SCM-input-box handler).
//
// Defect 3 — ntfy URL never reaches the webview when the topic file
//   appears after the first ``remote_url`` send.  The dedup key in
//   ``_tryReadAndSendUrl`` covered only ``tunnelActive|url``; when the
//   daemon wrote ``~/.kiss/ntfy_topic`` later (its usual startup
//   order), the watcher saw an unchanged URL, deduped, and the webview
//   never learned the ntfy link until a full webview reload.
//
// Run directly with ``node`` (after ``npm run compile``):
//
//     node src/kiss/agents/vscode/test/bughunt_new_sidebar_defects.test.js

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
    openTextDocument: () =>
      Promise.resolve({uri: makeUri('/x'), getText: () => ''}),
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
// ``_vscode-stub.js`` is a shared git-tracked fixture — never rewrite it.
global.__kissVscodeStub = vscodeStub;

const tmpHome = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-bh-new-'));
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
const daemonReceived = [];
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
        daemonReceived.push(JSON.parse(line));
      } catch {
        /* ignore */
      }
    }
  });
});

function daemonSend(msg) {
  assert.ok(lastServerSock, 'daemon has no connected client socket');
  lastServerSock.write(JSON.stringify(msg) + '\n');
  return new Promise(r => setTimeout(r, 60));
}

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

async function waitFor(predicate, message, timeoutMs = 2000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    const value = predicate();
    if (value) return value;
    await new Promise(r => setTimeout(r, 20));
  }
  throw new Error(message || 'waitFor timed out');
}

function notifications(posts) {
  return posts.filter(m => m && m.type === 'notification');
}

async function runTests() {
  await new Promise((res, rej) =>
    server.listen(sockPath, err => (err ? rej(err) : res())),
  );

  const sourcePath = path.join(__dirname, '..', 'out', 'SorcarSidebarView.js');
  assert.ok(
    fs.existsSync(sourcePath),
    `compiled extension missing: ${sourcePath} — run \`npm run compile\``,
  );
  delete require.cache[
    require.resolve(path.join(__dirname, '..', 'out', 'WebviewNotifications.js'))
  ];
  delete require.cache[require.resolve(sourcePath)];
  const {SorcarSidebarView} = require(sourcePath);

  const ws = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-bh-new-ws-'));
  tmpDirs.push(ws);
  workspaceFolders = [{uri: makeUri(ws)}];
  const view = new SorcarSidebarView(makeUri(path.join(__dirname, '..')));
  const wv = makeWebviewView();

  view.resolveWebviewView(wv.webviewView, {}, {});
  wv.fireMessage({type: 'ready', tabId: 'tab-main', restoredTabs: []});
  await waitForClient();

  // ----- Defect 1: superseded worktree progress toast must close -------
  const W = 'tab-worktree';
  const before1 = notifications(wv.posted).length;
  wv.fireMessage({type: 'worktreeAction', action: 'merge', tabId: W});
  const first = await waitFor(
    () =>
      notifications(wv.posted)
        .slice(before1)
        .find(m => m.message === 'Committing and merging worktree…' && !m.close),
    'first worktree progress toast was not posted',
  );
  const firstId = first.id;
  // Second action for the SAME tab supersedes the first dialog.
  const before2 = notifications(wv.posted).length;
  wv.fireMessage({type: 'worktreeAction', action: 'merge', tabId: W});
  const second = await waitFor(
    () =>
      notifications(wv.posted)
        .slice(before2)
        .find(
          m =>
            m.message === 'Committing and merging worktree…' &&
            !m.close &&
            m.id !== firstId,
        ),
    'second worktree progress toast was not posted',
  );
  const secondId = second.id;
  // The daemon completes (only) the CURRENT action.
  await daemonSend({
    type: 'worktree_result',
    success: true,
    message: 'Merged worktree branch.',
    tabId: W,
  });
  await waitFor(
    () =>
      notifications(wv.posted).some(m => m.id === secondId && m.close === true),
    'current worktree toast was not closed by worktree_result',
  );
  await waitFor(
    () =>
      notifications(wv.posted).some(m => m.id === firstId && m.close === true),
    'BUG: superseded worktree progress toast was never closed — its ' +
      'resolver was overwritten and its identity-checked timeout disarmed',
  );
  assert.strictEqual(
    view._worktreeActionResolves.size,
    0,
    'no worktree resolver may remain pending after worktree_result',
  );
  console.log('  ok - superseded worktree progress toast is closed');

  // ----- Defect 2: per-tab commit-message generations must not cross-talk
  view._ownTabs.add('tabA');
  view._ownTabs.add('tabB');
  let aResolved = false;
  let bResolved = false;
  const pA = view.generateCommitMessage(undefined, 'tabA').then(() => {
    aResolved = true;
  });
  const pB = view.generateCommitMessage(undefined, 'tabB').then(() => {
    bResolved = true;
  });
  assert.ok(view._commitPendingTabs.has('tabA'), 'tabA generation pending');
  assert.ok(view._commitPendingTabs.has('tabB'), 'tabB generation pending');
  // Reply for tabA only.
  await daemonSend({type: 'commitMessage', message: 'feat: a', tabId: 'tabA'});
  await pA;
  assert.ok(aResolved, 'tabA generation must resolve on its own reply');
  // Give any (buggy) cross-talk resolution a chance to land.
  await new Promise(r => setTimeout(r, 150));
  assert.strictEqual(
    bResolved,
    false,
    "BUG: tabB's pending generation was resolved by tabA's commitMessage",
  );
  assert.ok(
    view._commitPendingTabs.has('tabB'),
    "BUG: tabB's _commitPendingTabs entry was cleared by tabA's reply",
  );
  await daemonSend({type: 'commitMessage', message: 'feat: b', tabId: 'tabB'});
  await pB;
  assert.ok(bResolved, 'tabB generation must resolve on its own reply');
  assert.ok(!view._commitPendingTabs.has('tabB'), 'tabB no longer pending');
  console.log('  ok - commit-message generations are scoped per tab');

  // ----- Defect 3: ntfy topic appearing later must reach the webview ----
  const urlFile = path.join(tmpHome, '.kiss', 'remote-url.json');
  // The ``ready`` above already produced the initial remote_url send
  // (missing file ⇒ empty url).  Now the daemon writes the ntfy topic
  // AFTER that first send, with the URL file unchanged.
  await waitFor(
    () => wv.posted.some(m => m && m.type === 'remote_url'),
    'initial remote_url message was not posted',
  );
  fs.writeFileSync(path.join(tmpHome, '.kiss', 'ntfy_topic'), 'kiss-topic-1\n');
  const beforeNtfy = wv.posted.length;
  // Simulate the 10 s watcher tick (its interval is impractical to wait).
  view._tryReadAndSendUrl(urlFile);
  const ntfyMsg = wv.posted
    .slice(beforeNtfy)
    .find(m => m && m.type === 'remote_url');
  assert.ok(
    ntfyMsg,
    'BUG: remote_url resend was deduped even though the ntfy topic ' +
      'changed (dedup key ignored ntfyUrl)',
  );
  assert.strictEqual(
    ntfyMsg.ntfyUrl,
    'https://ntfy.sh/kiss-topic-1',
    'resent remote_url must carry the new ntfy URL',
  );
  // And an unchanged state must still dedup (no spam every 10 s).
  const beforeDedup = wv.posted.length;
  view._tryReadAndSendUrl(urlFile);
  assert.strictEqual(
    wv.posted.slice(beforeDedup).filter(m => m && m.type === 'remote_url')
      .length,
    0,
    'unchanged url+ntfy state must still be deduped',
  );
  console.log('  ok - ntfy topic written after first send reaches webview');

  view.dispose();
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
