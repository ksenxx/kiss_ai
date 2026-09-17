// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// End-to-end regression test: a worktree merge WITHOUT a tab id must not
// leave its progress notification open for ever.
//
// The bug: _showActionProgress() only registers its resolver in
// _worktreeActionResolves when the action carries a tabId, and its
// safety-net timeout was ALSO gated on `tabId !== undefined` — so for a
// `worktreeAction` message with no tabId (the protocol allows it:
// `tabId?: string` in src/types.ts) NOTHING could ever resolve the
// promise handed to withProgress.  The "Committing and merging
// worktree…" toast stayed open for the lifetime of the window;
// disconnects and dispose() resolve only the resolvers in the map, which
// this one never joined.
//
// The fix registers a tabId-less action under the '' sentinel key — the
// same value the daemon normalizes an omitted command tabId to and
// echoes on worktree_progress/worktree_result — so the terminal result,
// replacement, disconnect and dispose() all settle the toast, with the
// timeout as a safety net only.  The production ceiling is 120s;
// waiting that long per run is
// not acceptable in the suite, so the compiled module is copied to a
// temp dir with that ONE constant lowered (the copy asserts the
// substitution matched, so a rename fails loudly).  Everything else —
// the real compiled _handleMessage -> _showActionProgress ->
// withWebviewNotificationProgress -> vscode.window.withProgress chain —
// is the real code under real timers.

const assert = require('assert');
const childProcess = require('child_process');
const fs = require('fs');
const net = require('net');
const os = require('os');
const path = require('path');
const Module = require('module');

const OUT_DIR = path.join(__dirname, '..', 'out');
if (!fs.existsSync(path.join(OUT_DIR, 'SorcarSidebarView.js'))) {
  console.log('SKIP: out/SorcarSidebarView.js missing — run `npm run compile`');
  process.exit(0);
}

// ---- Process-liveness child (round-3 review regression) ----
// Re-run with --liveness-child: use the REAL compiled module with the
// PRODUCTION-DEFAULT 120s fallback timeout, start three same-key
// generations in one turn (each supersedes its predecessor), settle the
// last one, and then exit NATURALLY.  Before the fix each settled
// generation left its referenced 120s setTimeout alive, so this child —
// logically finished, maps empty, toast closed — was kept running until
// the parent's watchdog SIGKILLed it.  The fix stores the timer handle
// and clearTimeout()s it inside the idempotent settle closure, on every
// settlement path.  No process.exit() here: a natural exit IS the
// assertion.
if (process.argv.includes('--liveness-child')) {
  const childMakeUri = p => ({fsPath: p, scheme: 'file', toString: () => `file://${p}`});
  const childStub = {
    workspace: {
      workspaceFolders: [],
      getConfiguration: () => ({get: () => 'stub'}),
      onDidChangeWorkspaceFolders: () => ({dispose() {}}),
      textDocuments: [],
    },
    EventEmitter: class {
      constructor() {
        this.event = () => ({dispose() {}});
      }
      dispose() {}
    },
    Uri: {
      file: value => childMakeUri(value),
      joinPath: (base, ...parts) => childMakeUri(path.join(base.fsPath, ...parts)),
      parse: value => childMakeUri(value),
    },
    ProgressLocation: {Notification: 15},
    CancellationTokenSource: class {
      constructor() {
        this.token = {isCancellationRequested: false};
      }
      dispose() {}
    },
    window: {
      withProgress: (_options, task) =>
        Promise.resolve(task({report() {}}, {isCancellationRequested: false})),
      showInformationMessage: () => Promise.resolve(undefined),
      showWarningMessage: () => Promise.resolve(undefined),
      showErrorMessage: () => Promise.resolve(undefined),
    },
    commands: {executeCommand: () => Promise.resolve()},
  };
  const childResolve = Module._resolveFilename;
  Module._resolveFilename = function (request, parent, ...rest) {
    if (request === 'vscode') return require.resolve('./_vscode-stub.js');
    return childResolve.call(this, request, parent, ...rest);
  };
  global.__kissVscodeStub = childStub;
  const {SorcarSidebarView} = require(path.join(OUT_DIR, 'SorcarSidebarView.js'));
  const notifications = require(path.join(OUT_DIR, 'WebviewNotifications.js'));
  const posted = [];
  const poster = message => posted.push(message);
  notifications.setWebviewNotificationPoster(poster);
  const progressMap = new Map();
  const resolveMap = new Map();
  // Three same-key generations: each start settles (and must
  // clearTimeout) the previous one via the prev() replacement path, so
  // repeated merge clicks do not accumulate one retained timer per
  // generation.
  for (let i = 0; i < 3; i++) {
    SorcarSidebarView.prototype._showActionProgress.call(
      {},
      'settled action',
      undefined,
      progressMap,
      resolveMap,
    );
  }
  const settleLast = resolveMap.get('');
  assert.strictEqual(typeof settleLast, 'function');
  resolveMap.delete('');
  progressMap.delete('');
  settleLast();
  setImmediate(() => {
    notifications.clearWebviewNotificationPoster(poster);
    console.log(
      JSON.stringify({
        mapsEmpty: resolveMap.size === 0 && progressMap.size === 0,
        toastClosed: posted.some(message => message.close),
        generations: 3,
        timeoutMode: 'default-120000ms',
      }),
    );
  });
  // Top-level return: skip the parent-mode body below.  The child must
  // now exit on its own — any surviving fallback timer keeps it alive
  // and the parent's watchdog kill fails the test.
  return;
}

const SHORT_TIMEOUT_MS = 400;

const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-wtprog-'));
const outCopy = path.join(tmpRoot, 'out');
fs.mkdirSync(outCopy, {recursive: true});
for (const f of fs.readdirSync(OUT_DIR)) {
  if (!f.endsWith('.js')) continue;
  let src = fs.readFileSync(path.join(OUT_DIR, f), 'utf-8');
  if (f === 'SorcarSidebarView.js') {
    const needle = 'timeoutMs = 120_000';
    assert.ok(
      src.includes(needle),
      'SorcarSidebarView.js no longer defaults the action-progress ' +
        'timeout to 120_000 as expected',
    );
    src = src.replace(needle, `timeoutMs = ${SHORT_TIMEOUT_MS}`);
  }
  fs.writeFileSync(path.join(outCopy, f), src);
}

// A real (silent) UDS daemon stand-in.  Refusing the connection instead
// would fire the client's 'disconnect' on every reconnect attempt, and
// the view's disconnect handler resolves EVERY pending worktree action —
// masking exactly the hang this test exists to catch.
process.env.KISS_SORCAR_SOCK = path.join(tmpRoot, 'daemon.sock');
process.env.KISS_HOME = path.join(tmpRoot, 'kiss-home');
const daemon = net.createServer(sock => sock.on('error', () => {}));
daemon.listen(process.env.KISS_SORCAR_SOCK);

function makeUri(fsPath) {
  return {fsPath, scheme: 'file', toString: () => `file://${fsPath}`};
}

// Every withProgress invocation is recorded with its task promise and a
// settled flag, so the test can tell a toast that closed from one that
// hangs.
const progressCalls = [];

const vscodeStub = {
  workspace: {
    workspaceFolders: [],
    getConfiguration: () => ({get: () => 'stub-default-model'}),
    onDidChangeWorkspaceFolders: () => ({dispose: () => {}}),
    textDocuments: [],
  },
  EventEmitter: class {
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
  },
  Uri: {
    file: p => makeUri(p),
    joinPath: (base, ...parts) => makeUri(path.join(base.fsPath, ...parts)),
    parse: s => makeUri(s),
  },
  ProgressLocation: {Notification: 15},
  CancellationTokenSource: class {
    constructor() {
      this.token = {isCancellationRequested: false};
    }
    dispose() {}
  },
  window: {
    withProgress: (opts, task) => {
      const entry = {title: opts.title, settled: false};
      progressCalls.push(entry);
      const p = Promise.resolve(
        task(
          {report: () => {}},
          {onCancellationRequested: () => ({dispose: () => {}})},
        ),
      );
      p.then(
        () => {
          entry.settled = true;
        },
        () => {
          entry.settled = true;
        },
      );
      return p;
    },
    showInformationMessage: () => Promise.resolve(undefined),
    showWarningMessage: () => Promise.resolve(undefined),
    showErrorMessage: () => Promise.resolve(undefined),
  },
  commands: {executeCommand: () => Promise.resolve()},
};

const origResolve = Module._resolveFilename;
Module._resolveFilename = function (request, parent, ...rest) {
  if (request === 'vscode') return require.resolve('./_vscode-stub.js');
  return origResolve.call(this, request, parent, ...rest);
};
global.__kissVscodeStub = vscodeStub;

const {SorcarSidebarView} = require(path.join(outCopy, 'SorcarSidebarView.js'));
// The SAME module instance the copied view uses, so the production
// webview poster path (which defers the progress task into a
// microtask) is the one exercised below.
const notifications = require(path.join(outCopy, 'WebviewNotifications.js'));

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function main() {
  const view = new SorcarSidebarView(makeUri(path.join(__dirname, '..')));
  try {
    // 1) A merge with NO tabId: the protocol allows it, and before the
    //    fix its progress toast could never close.  The daemon
    //    normalizes an omitted command tabId to '' (commands.py:
    //    cmd.get("tabId", "")) and echoes it on worktree_result, so the
    //    toast must close IMMEDIATELY on the terminal result — not two
    //    minutes later via the safety-net timeout.
    await view._handleMessage({type: 'worktreeAction', action: 'merge'});
    assert.strictEqual(progressCalls.length, 1, 'one progress toast opens');
    assert.strictEqual(
      progressCalls[0].title,
      'Committing and merging worktree\u2026',
    );
    assert.ok(
      view._worktreeActionResolves.has(''),
      "a tabId-less merge registers its resolver under the '' sentinel " +
        'key the daemon echoes back',
    );
    view._client.emit('message', {
      type: 'worktree_result',
      tabId: '',
      success: true,
      message: 'merged',
    });
    await sleep(20);
    assert.ok(
      progressCalls[0].settled,
      'a worktreeAction without a tabId must close its progress toast ' +
        'on the terminal worktree_result (before the fix nothing but ' +
        'the 120s fallback could settle it)',
    );
    assert.strictEqual(
      view._worktreeActionResolves.size,
      0,
      'the terminal result removes the sentinel resolver entry',
    );
    assert.strictEqual(
      view._worktreeProgresses.size,
      0,
      'the terminal result removes the sentinel progress entry',
    );

    // 1b) With no terminal result at all, the safety-net timeout still
    //     closes a tabId-less toast.
    await view._handleMessage({type: 'worktreeAction', action: 'merge'});
    assert.strictEqual(progressCalls.length, 2, 'safety-net toast opens');
    await sleep(SHORT_TIMEOUT_MS + 600);
    assert.ok(
      progressCalls[1].settled,
      'a worktreeAction without a tabId must still close its progress ' +
        'toast when the timeout expires (before the fix the promise had ' +
        'no resolver at all and the toast stayed open for ever)',
    );
    assert.strictEqual(
      view._worktreeActionResolves.size,
      0,
      'the timeout removes the sentinel resolver entry',
    );
    // Re-number the remaining sections' expectations around the extra
    // toast opened above.
    const base = progressCalls.length;

    // 2) Regression guard: a merge WITH a tabId still registers its
    //    resolver and still times out cleanly, leaving no map entries.
    await view._handleMessage({
      type: 'worktreeAction',
      action: 'merge',
      tabId: 't1',
    });
    assert.strictEqual(progressCalls.length, base + 1, 'second toast opens');
    assert.ok(
      view._worktreeActionResolves.has('t1'),
      'a tabId-carrying merge registers its resolver',
    );
    await sleep(SHORT_TIMEOUT_MS + 600);
    assert.ok(progressCalls[base].settled, 'tabId merge times out cleanly');
    assert.strictEqual(
      view._worktreeActionResolves.size,
      0,
      'timeout removes the resolver entry',
    );
    assert.strictEqual(
      view._worktreeProgresses.size,
      0,
      'timeout removes the progress entry',
    );

    // 3) A second merge for the SAME tab resolves the first toast
    //    immediately (the prev() replacement path).
    await view._handleMessage({
      type: 'worktreeAction',
      action: 'merge',
      tabId: 't2',
    });
    await view._handleMessage({
      type: 'worktreeAction',
      action: 'merge',
      tabId: 't2',
    });
    assert.strictEqual(progressCalls.length, base + 3, 'both toasts opened');
    await sleep(20);
    assert.ok(
      progressCalls[base + 1].settled,
      'starting a new merge for the same tab closes the previous toast',
    );
    assert.ok(!progressCalls[base + 2].settled, 'the new toast is still open');
    await sleep(SHORT_TIMEOUT_MS + 600);
    assert.ok(progressCalls[base + 2].settled, 'and times out cleanly');

    // 4) Provider disposal settles a still-open tabId-less toast at
    //    once: dispose() must reach the sentinel resolver too.
    await view._handleMessage({type: 'worktreeAction', action: 'merge'});
    assert.strictEqual(
      progressCalls.length,
      base + 4,
      'disposal-test toast opens',
    );
    view.dispose();
    await sleep(20);
    assert.ok(
      progressCalls[base + 3].settled,
      'dispose() settles a pending tabId-less progress toast instead of ' +
        'leaving it to the 120s fallback',
    );

    // ---- Production webview poster orderings (round-2 review) ----
    // With the sidebar webview attached, toasts go through the
    // production poster, and withWebviewNotificationProgress() starts
    // its task only in Promise.resolve().then(...) — one microtask
    // AFTER _showActionProgress returned.  Before the fix the resolver
    // was registered only inside that deferred task, so two same-key
    // actions in ONE turn both missed the replacement check, the
    // second overwrote the first resolver, and the first toast was
    // stranded for ever (its safety timer's identity guard no longer
    // matched).  The fix publishes the lifecycle entry synchronously.
    const posted = [];
    const poster = m => posted.push(m);
    notifications.setWebviewNotificationPoster(poster);
    const openIds = () =>
      posted.filter(m => m.progress && !m.close).map(m => m.id);
    const closedIds = () => posted.filter(m => m.close).map(m => m.id);

    // 5) Two tabless merges in ONE turn (no intervening await), then a
    //    single terminal result: BOTH toasts must close.
    const viewB = new SorcarSidebarView(makeUri(path.join(__dirname, '..')));
    void viewB._handleMessage({type: 'worktreeAction', action: 'merge'});
    assert.ok(
      viewB._worktreeActionResolves.has(''),
      'the sentinel resolver is published SYNCHRONOUSLY, before the ' +
        'deferred progress task runs',
    );
    void viewB._handleMessage({type: 'worktreeAction', action: 'merge'});
    assert.strictEqual(
      viewB._worktreeActionResolves.size,
      1,
      'the second same-key action replaces (not stacks on) the first',
    );
    assert.strictEqual(openIds().length, 2, 'both toasts opened');
    await sleep(20);
    const afterReplace = closedIds();
    assert.strictEqual(
      afterReplace.length,
      1,
      'replacement settles the FIRST toast before any terminal result',
    );
    assert.strictEqual(afterReplace[0], openIds()[0], 'and only the first');
    viewB._client.emit('message', {
      type: 'worktree_result',
      tabId: '',
      success: true,
      message: 'merged',
    });
    await sleep(20);
    for (const id of openIds()) {
      assert.ok(
        closedIds().includes(id),
        `toast ${id} closed (no stranded first-generation toast)`,
      );
    }
    assert.strictEqual(viewB._worktreeActionResolves.size, 0);
    assert.strictEqual(viewB._worktreeProgresses.size, 0);

    // 6) Concurrent '' sentinel and a real tab id, started in one
    //    turn: each terminal result settles ONLY its own toast.
    const mixBase = posted.length;
    void viewB._handleMessage({type: 'worktreeAction', action: 'merge'});
    void viewB._handleMessage({
      type: 'worktreeAction',
      action: 'merge',
      tabId: 'tab-real',
    });
    assert.strictEqual(
      viewB._worktreeActionResolves.size,
      2,
      "'' and a real tab id occupy separate entries",
    );
    const mixOpen = posted
      .slice(mixBase)
      .filter(m => m.progress && !m.close)
      .map(m => m.id);
    assert.strictEqual(mixOpen.length, 2, 'both mixed-key toasts opened');
    viewB._client.emit('message', {
      type: 'worktree_result',
      tabId: '',
      success: true,
      message: 'merged',
    });
    await sleep(20);
    let mixClosed = posted
      .slice(mixBase)
      .filter(m => m.close)
      .map(m => m.id);
    assert.deepStrictEqual(
      mixClosed,
      [mixOpen[0]],
      "the sentinel result settles only the '' toast; the real-tab " +
        'toast stays open',
    );
    assert.ok(viewB._worktreeActionResolves.has('tab-real'));
    viewB._client.emit('message', {
      type: 'worktree_result',
      tabId: 'tab-real',
      success: true,
      message: 'merged',
    });
    await sleep(20);
    mixClosed = posted
      .slice(mixBase)
      .filter(m => m.close)
      .map(m => m.id);
    assert.deepStrictEqual(
      mixClosed.sort(),
      mixOpen.slice().sort(),
      'its own terminal result settles the real-tab toast',
    );
    assert.strictEqual(viewB._worktreeActionResolves.size, 0);

    // 7) Action followed IMMEDIATELY by a daemon disconnect, before
    //    any microtask turn: the toast must close at once, not at the
    //    fallback timeout.
    const discBase = posted.length;
    void viewB._handleMessage({type: 'worktreeAction', action: 'merge'});
    viewB._client.emit('disconnect');
    await sleep(20);
    const discOpen = posted
      .slice(discBase)
      .filter(m => m.progress && !m.close)
      .map(m => m.id);
    const discClosed = posted
      .slice(discBase)
      .filter(m => m.close)
      .map(m => m.id);
    assert.strictEqual(discOpen.length, 1, 'disconnect-test toast opened');
    assert.deepStrictEqual(
      discClosed,
      discOpen,
      'a disconnect in the same turn as the action still settles the ' +
        'toast immediately (the resolver is already published)',
    );
    viewB.dispose();

    // 8) Action followed IMMEDIATELY by dispose(), before any
    //    microtask turn: same requirement.
    const viewC = new SorcarSidebarView(makeUri(path.join(__dirname, '..')));
    const dispBase = posted.length;
    void viewC._handleMessage({type: 'worktreeAction', action: 'merge'});
    viewC.dispose();
    await sleep(20);
    const dispOpen = posted
      .slice(dispBase)
      .filter(m => m.progress && !m.close)
      .map(m => m.id);
    const dispClosed = posted
      .slice(dispBase)
      .filter(m => m.close)
      .map(m => m.id);
    assert.strictEqual(dispOpen.length, 1, 'dispose-test toast opened');
    assert.deepStrictEqual(
      dispClosed,
      dispOpen,
      'dispose() in the same turn as the action still settles the toast ' +
        'immediately instead of leaving it to the 120s fallback',
    );
    notifications.clearWebviewNotificationPoster(poster);

    // 9) Process liveness (round-3 review): with the PRODUCTION-DEFAULT
    //    120s fallback timeout, a child process whose toasts have all
    //    settled must exit promptly and normally.  Before the fix the
    //    child reached the same clean logical state (maps empty, toast
    //    closed) but its already-obsolete 120s timers were still
    //    referenced, so only the watchdog's SIGKILL ended it.
    const liveness = childProcess.spawnSync(
      process.execPath,
      [__filename, '--liveness-child'],
      {encoding: 'utf8', timeout: 10_000, killSignal: 'SIGKILL'},
    );
    assert.strictEqual(
      liveness.error,
      undefined,
      'a settled action with the production-default timeout must let ' +
        'the process exit instead of being watchdog-killed: ' +
        `${liveness.error && liveness.error.code} ${liveness.stderr}`,
    );
    assert.strictEqual(
      liveness.status,
      0,
      `liveness child failed: ${liveness.stderr}`,
    );
    assert.match(
      liveness.stdout,
      /"mapsEmpty":true/,
      'liveness child left resolver/progress entries behind',
    );
    assert.match(
      liveness.stdout,
      /"toastClosed":true/,
      'liveness child never closed its toast',
    );
  } finally {
    view.dispose();
    daemon.close();
    fs.rmSync(tmpRoot, {recursive: true, force: true});
  }
  console.log('PASS conc2026_action_progress_no_tab_hang');
}

main().catch(err => {
  console.error(err && err.stack ? err.stack : String(err));
  process.exit(1);
});
