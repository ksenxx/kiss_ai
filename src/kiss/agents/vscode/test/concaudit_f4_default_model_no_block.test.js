// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// End-to-end regression test: constructing the chat controller must not
// block the extension host on the default-model lookup.
//
// The bug: with `kissSorcar.defaultModel` unset, the SorcarSidebarView
// constructor ran `uv run python -c "...get_default_model()"`
// SYNCHRONOUSLY (15s deadline, then two more 2s synchronous `which`
// probes).  That is activation time -- and, in editor-tabs mode, every
// new panel -- frozen for as long as `uv` takes to import the model
// catalog.
//
// The fix starts from a spawn-free provisional value and adopts the
// asynchronously resolved default when it arrives.  This test drives the
// REAL compiled SorcarSidebarView against a fake HOME whose `uv` takes
// 1.5s to answer: the constructor must return in a small fraction of
// that, the first command sent to the (real UDS) daemon carries the
// provisional model, and one sent after `uv` answered carries the
// resolved one.

const assert = require('assert');
const fs = require('fs');
const net = require('net');
const os = require('os');
const path = require('path');
const Module = require('module');

const OUT_VIEW = path.join(__dirname, '..', 'out', 'SorcarSidebarView.js');
if (!fs.existsSync(OUT_VIEW)) {
  console.log('SKIP: out/SorcarSidebarView.js missing — run `npm run compile`');
  process.exit(0);
}
if (process.platform === 'win32') {
  console.log('SKIP: Unix domain sockets only');
  process.exit(0);
}

const UV_DELAY_S = '1.5';
const RESOLVED_MODEL = 'resolved-from-uv/model';

// --- fake HOME / project / uv ----------------------------------------
const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-defmodel-'));
const tmpHome = path.join(tmpRoot, 'home');
const fakeProj = path.join(tmpRoot, 'proj');
fs.mkdirSync(path.join(tmpHome, '.local', 'bin'), {recursive: true});
fs.mkdirSync(path.join(tmpHome, '.kiss'), {recursive: true});
fs.mkdirSync(fakeProj, {recursive: true});
fs.writeFileSync(
  path.join(fakeProj, 'pyproject.toml'),
  '[project]\nname = "kiss"\nversion = "0"\n',
);
const fakeUv = path.join(tmpHome, '.local', 'bin', 'uv');
fs.writeFileSync(
  fakeUv,
  `#!/bin/sh\nsleep ${UV_DELAY_S}\necho "${RESOLVED_MODEL}"\nexit 0\n`,
);
fs.chmodSync(fakeUv, 0o755);

process.env.HOME = tmpHome;
process.env.USERPROFILE = tmpHome;
process.env.KISS_HOME = path.join(tmpHome, '.kiss');
process.env.KISS_PROJECT_PATH = fakeProj;
// No API key may short-circuit the lookup: the provisional value must be
// the spawn-free "No model" placeholder.
for (const k of [
  'ANTHROPIC_API_KEY',
  'OPENAI_API_KEY',
  'GEMINI_API_KEY',
  'OPENROUTER_API_KEY',
  'TOGETHER_API_KEY',
]) {
  delete process.env[k];
}
const sockPath = path.join(tmpHome, '.kiss', 'sorcar.sock');

// --- vscode stub -----------------------------------------------------
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

const ws = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-defmodel-ws-'));
global.__kissVscodeStub = {
  workspace: {
    isTrusted: true,
    workspaceFolders: [{uri: makeUri(ws)}],
    // `kissSorcar.defaultModel` unset: the lookup path under test.
    getConfiguration: () => ({get: () => undefined}),
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
  ViewColumn: {One: 1},
  window: {
    withProgress: (_opts, task) =>
      task(
        {report: () => {}},
        {onCancellationRequested: () => ({dispose: () => {}})},
      ),
    showInformationMessage: () => {},
    showErrorMessage: () => {},
    showWarningMessage: () => {},
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

// --- real UDS daemon stand-in: records every frame ---------------------
const frames = [];
const server = net.createServer(sock => {
  let buf = '';
  sock.setEncoding('utf-8');
  sock.on('data', d => {
    buf += d;
    let nl;
    while ((nl = buf.indexOf('\n')) >= 0) {
      const line = buf.slice(0, nl);
      buf = buf.slice(nl + 1);
      try {
        frames.push(JSON.parse(line));
      } catch {}
    }
  });
  sock.on('error', () => {});
});

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

async function waitForFrame(pred, ms) {
  const deadline = Date.now() + ms;
  while (Date.now() < deadline) {
    const f = frames.find(pred);
    if (f) return f;
    await sleep(20);
  }
  return null;
}

function cleanup() {
  server.close();
  fs.rmSync(tmpRoot, {recursive: true, force: true});
  fs.rmSync(ws, {recursive: true, force: true});
}

async function main() {
  await new Promise((res, rej) =>
    server.listen(sockPath, err => (err ? rej(err) : res())),
  );
  const {SorcarSidebarView} = require(OUT_VIEW);

  // A 25ms heartbeat measures the longest event-loop stall around the
  // construction: a synchronous `uv run` shows up as one stall of at
  // least UV_DELAY_S.
  let last = Date.now();
  let maxStallMs = 0;
  const beat = setInterval(() => {
    const now = Date.now();
    if (now - last > maxStallMs) maxStallMs = now - last;
    last = now;
  }, 25);

  const t0 = Date.now();
  const view = new SorcarSidebarView(makeUri(path.join(__dirname, '..')));
  const ctorMs = Date.now() - t0;
  assert.ok(
    ctorMs < 500,
    `constructing the view took ${ctorMs}ms: the default-model lookup ` +
      'is blocking the extension host again',
  );
  console.log(`  ok - constructor returned in ${ctorMs}ms`);

  // 1. Immediately: the daemon-bound command carries the spawn-free
  //    provisional model (nothing has been resolved yet).
  void view.generateCommitMessage(undefined, 'tab-early', ws);
  const early = await waitForFrame(
    f => f.type === 'generateCommitMessage' && f.tabId === 'tab-early',
    5000,
  );
  assert.ok(early, 'the early command never reached the daemon');
  assert.strictEqual(
    early.model,
    'No model',
    'before the lookup settles the provisional placeholder must be used',
  );
  console.log('  ok - early command used the provisional model');

  // 2. After `uv` has answered: the resolved default is adopted.
  await sleep(2500);
  void view.generateCommitMessage(undefined, 'tab-late', ws);
  const late = await waitForFrame(
    f => f.type === 'generateCommitMessage' && f.tabId === 'tab-late',
    5000,
  );
  assert.ok(late, 'the late command never reached the daemon');
  assert.strictEqual(
    late.model,
    RESOLVED_MODEL,
    'the asynchronously resolved default model was not adopted',
  );
  console.log('  ok - late command used the resolved model');

  clearInterval(beat);
  assert.ok(
    maxStallMs < 500,
    `event loop stalled for ${maxStallMs}ms during construction`,
  );
  console.log(`  ok - longest event-loop stall: ${maxStallMs}ms`);

  // 3. A second controller (editor-tabs mode opens one per panel) starts
  //    from the already-resolved value: no placeholder, no new wait.
  const view2 = new SorcarSidebarView(makeUri(path.join(__dirname, '..')));
  void view2.generateCommitMessage(undefined, 'tab-second-view', ws);
  const second = await waitForFrame(
    f => f.type === 'generateCommitMessage' && f.tabId === 'tab-second-view',
    5000,
  );
  assert.ok(second, 'the second view command never reached the daemon');
  assert.strictEqual(second.model, RESOLVED_MODEL);
  console.log('  ok - a later controller starts from the resolved model');

  view.dispose();
  view2.dispose();
}

main()
  .then(() => {
    cleanup();
    console.log('concaudit_f4_default_model_no_block: all assertions passed');
    process.exit(0);
  })
  .catch(err => {
    cleanup();
    console.error(err && err.stack ? err.stack : err);
    process.exit(1);
  });
