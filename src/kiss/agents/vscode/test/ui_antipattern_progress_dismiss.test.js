// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// R3-3: closing a cancellable progress toast (the webview sends the
// toast's id with action undefined) must not disarm its Cancel.  The
// operation keeps running and its next progress report re-posts the
// toast with the same Cancel button, so that button has to cancel the
// operation's token.  The registration also outlives a replacement of
// the notification poster (sidebar view <-> editor tabs) and is removed
// only when the operation ends.  Runs against the REAL
// out/WebviewNotifications.js with a minimal vscode stub.

const assert = require('assert');
const path = require('path');
const Module = require('module');

const OUT_DIR = path.join(__dirname, '..', 'out');

class CancellationTokenSource {
  constructor() {
    this._listeners = [];
    this._cancelled = false;
    const self = this;
    this.token = {
      get isCancellationRequested() {
        return self._cancelled;
      },
      onCancellationRequested: cb => {
        self._listeners.push(cb);
        return {dispose: () => {}};
      },
    };
  }
  cancel() {
    if (this._cancelled) return;
    this._cancelled = true;
    for (const cb of this._listeners.slice()) cb();
  }
  dispose() {}
}

const nativeCalls = [];
const vscodeStub = {
  window: {
    withProgress: (opts, task) => {
      nativeCalls.push('withProgress');
      return task({report: () => {}}, new CancellationTokenSource().token);
    },
    showInformationMessage: () => Promise.resolve(undefined),
    showWarningMessage: () => Promise.resolve(undefined),
    showErrorMessage: () => Promise.resolve(undefined),
  },
  ProgressLocation: {Notification: 15},
  CancellationTokenSource,
};

const origLoad = Module._load;
Module._load = function (request, parent, isMain) {
  if (request === 'vscode') return vscodeStub;
  return origLoad.call(this, request, parent, isMain);
};

const notif = require(path.join(OUT_DIR, 'WebviewNotifications.js'));

/**
 * Start a cancellable "Setting up" progress operation whose task runs
 * until `finish()` is called or its token is cancelled.  Returns the
 * progress reporter, the token and the operation's promise.
 */
function startOperation(posted) {
  const started = {};
  started.done = notif.withWebviewNotificationProgress(
    {location: 15, title: 'KISS Sorcar: Setting up', cancellable: true},
    (progress, token) =>
      new Promise(resolve => {
        started.progress = progress;
        started.token = token;
        started.finish = () => resolve('finished');
        token.onCancellationRequested(() => resolve('cancelled'));
      }),
  );
  const toast = posted.find(
    m => m.type === 'notification' && m.progress === true,
  );
  assert.ok(toast, 'the progress toast is posted at once');
  assert.deepStrictEqual(toast.actions, ['Cancel']);
  started.id = toast.id;
  return started;
}

async function dismissThenCancelTest() {
  const posted = [];
  notif.setWebviewNotificationPoster(msg => posted.push(msg));
  const op = startOperation(posted);
  await new Promise(r => setImmediate(r));

  // The user closes the toast with its × (action undefined).
  notif.resolveWebviewNotificationAction(op.id, undefined);
  assert.strictEqual(
    op.token.isCancellationRequested,
    false,
    'closing the toast does not cancel the operation',
  );

  // The operation reports its next step: the toast reappears with Cancel.
  posted.length = 0;
  op.progress.report({message: 'Checking cloudflared...'});
  const reposted = posted.find(m => m.id === op.id && m.progress === true);
  assert.ok(reposted, 'the next report re-posts the toast');
  assert.deepStrictEqual(reposted.actions, ['Cancel']);
  assert.strictEqual(reposted.progressMessage, 'Checking cloudflared...');

  // Cancel on the reappeared toast must still cancel the token.
  notif.resolveWebviewNotificationAction(op.id, 'Cancel');
  assert.strictEqual(
    op.token.isCancellationRequested,
    true,
    'Cancel on the re-posted toast cancels the running operation',
  );
  assert.strictEqual(await op.done, 'cancelled');
  assert.ok(
    posted.some(m => m.id === op.id && m.close === true),
    'the toast is closed when the operation ends',
  );

  // Once the operation is over its id is forgotten: a late Cancel is a
  // no-op rather than an error.
  notif.resolveWebviewNotificationAction(op.id, 'Cancel');
  console.log('dismiss-then-cancel test passed');
}

async function posterReplacementTest() {
  const firstPosted = [];
  notif.setWebviewNotificationPoster(msg => firstPosted.push(msg));
  const op = startOperation(firstPosted);
  await new Promise(r => setImmediate(r));

  // The chat moves to another surface: the new poster takes over.
  const secondPosted = [];
  notif.setWebviewNotificationPoster(msg => secondPosted.push(msg));
  op.progress.report({message: 'Restarting kiss-web daemon...'});
  const onNewSurface = secondPosted.find(
    m => m.id === op.id && m.progress === true,
  );
  assert.ok(onNewSurface, 'progress reaches the new surface');
  assert.deepStrictEqual(onNewSurface.actions, ['Cancel']);
  notif.resolveWebviewNotificationAction(op.id, 'Cancel');
  assert.strictEqual(
    op.token.isCancellationRequested,
    true,
    'Cancel from the new surface cancels the operation',
  );
  assert.strictEqual(await op.done, 'cancelled');
  console.log('poster replacement test passed');
}

async function normalCompletionTest() {
  const posted = [];
  notif.setWebviewNotificationPoster(msg => posted.push(msg));
  const op = startOperation(posted);
  await new Promise(r => setImmediate(r));
  notif.resolveWebviewNotificationAction(op.id, undefined);
  op.finish();
  assert.strictEqual(await op.done, 'finished');
  assert.strictEqual(op.token.isCancellationRequested, false);
  assert.ok(posted.some(m => m.id === op.id && m.close === true));

  // Message toasts are unaffected: their action resolves the promise
  // once, and closing them resolves undefined.
  const answer = notif.showInformationNotification('Update?', 'Reload now');
  const toast = posted.find(m => m.message === 'Update?');
  notif.resolveWebviewNotificationAction(toast.id, 'Reload now');
  assert.strictEqual(await answer, 'Reload now');
  const closed = notif.showWarningNotification('Careful', 'Keep');
  const closedToast = posted.find(m => m.message === 'Careful');
  notif.resolveWebviewNotificationAction(closedToast.id, undefined);
  assert.strictEqual(await closed, undefined);
  console.log('normal completion / message toast tests passed');
}

async function runTest() {
  await dismissThenCancelTest();
  await posterReplacementTest();
  await normalCompletionTest();
  assert.deepStrictEqual(
    nativeCalls,
    [],
    'the native progress API was never used',
  );
  console.log('\nAll ui_antipattern_progress_dismiss tests passed');
}

runTest()
  .then(() => process.exit(0))
  .catch(err => {
    console.error(err);
    process.exit(1);
  });
