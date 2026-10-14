// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Audit 2026-10-13 (scope J, finding 1): an action toast whose webview
// surface went away (chat panel closed, mode flipped, window reloading)
// used to be resolved `undefined` by WebviewNotifications, exactly like
// a toast the user closed with its x.  The editor-actions consent
// question treats a dismissal as 'Keep my setting' and remembers it
// forever, so closing the chat panel while the question was open
// silently answered it: the actions were never moved and the user was
// never asked again.  ensureApiKeys had the same hole ('Skip' marker).
//
// Now a toast that loses its surface is shown again as a native VS
// Code notification (window-level, survives any panel) and its promise
// settles with THAT notification's answer.  Driven through the real
// compiled WebviewNotifications.js and editorActionsLocation.js with a
// vscode stub whose native show*Message records the call and lets the
// test answer it.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const Module = require('module');

const OUT = path.join(__dirname, '..', 'out');
assert.ok(
  fs.existsSync(path.join(OUT, 'WebviewNotifications.js')),
  'run `npm run compile` first',
);

// Native notifications the stub showed: {severity, message, actions, resolve}.
const native = [];
function nativeStub(severity) {
  return (message, ...items) =>
    new Promise(resolve => {
      const actions = items.filter(item => typeof item === 'string');
      native.push({severity, message, actions, resolve});
    });
}

const config = {'kissSorcar.editorTabsMode': true};
const settingUpdates = [];
const vscodeStub = {
  ConfigurationTarget: {Global: 1},
  ProgressLocation: {Notification: 15},
  workspace: {
    getConfiguration: section => ({
      get: key => config[section ? `${section}.${key}` : key],
      inspect: key => ({globalValue: config[key]}),
      update: async (key, value) => {
        settingUpdates.push({key, value});
        config[key] = value;
      },
    }),
  },
  window: {
    showInformationMessage: nativeStub('info'),
    showWarningMessage: nativeStub('warning'),
    showErrorMessage: nativeStub('error'),
  },
};
const origResolve = Module._resolveFilename;
Module._resolveFilename = function (request, parent, ...rest) {
  if (request === 'vscode') return path.join(__dirname, '_vscode-stub.js');
  return origResolve.call(this, request, parent, ...rest);
};
global.__kissVscodeStub = vscodeStub;

const notif = require(path.join(OUT, 'WebviewNotifications.js'));
const {LOCATION_CONSENT_KEY, syncEditorActionsLocation} = require(
  path.join(OUT, 'editorActionsLocation.js'),
);

function makeContext() {
  const store = new Map();
  return {
    globalState: {
      get: key => store.get(key),
      update: async (key, value) => {
        if (value === undefined) store.delete(key);
        else store.set(key, value);
      },
    },
  };
}

function tick() {
  return new Promise(resolve => setTimeout(resolve, 20));
}

async function consentSurvivesPanelClose() {
  const ctx = makeContext();
  const posted = [];
  const panelPoster = msg => posted.push(msg);
  notif.setWebviewNotificationPoster(panelPoster);

  await syncEditorActionsLocation(ctx, true);
  assert.strictEqual(posted.length, 1, 'the question went to the webview');
  assert.deepStrictEqual(posted[0].actions, [
    'Move actions to title bar',
    'Keep my setting',
  ]);
  assert.strictEqual(ctx.globalState.get(LOCATION_CONSENT_KEY), undefined);

  // The chat panel closes with the question still open.
  notif.clearWebviewNotificationPoster(panelPoster);
  await tick();
  assert.strictEqual(
    ctx.globalState.get(LOCATION_CONSENT_KEY),
    undefined,
    'closing the surface is not the user answering "keep"',
  );
  assert.strictEqual(native.length, 1, 'the question moved to a native toast');
  assert.strictEqual(native[0].severity, 'info');
  assert.strictEqual(native[0].message, posted[0].message);
  assert.deepStrictEqual(native[0].actions, posted[0].actions);

  // The user answers the native toast: the move happens right away.
  native[0].resolve('Move actions to title bar');
  await tick();
  assert.strictEqual(ctx.globalState.get(LOCATION_CONSENT_KEY), 'move');
  assert.deepStrictEqual(settingUpdates, [
    {key: 'workbench.editor.editorActionsLocation', value: 'titleBar'},
  ]);
  console.log('consent survives panel close: passed');
}

async function consentSurvivesPosterHandover() {
  delete config['workbench.editor.editorActionsLocation'];
  settingUpdates.length = 0;
  native.length = 0;
  const ctx = makeContext();
  const sidebarPosted = [];
  const panelPosted = [];
  const sidebarPoster = msg => sidebarPosted.push(msg);
  const panelPoster = msg => panelPosted.push(msg);
  notif.setWebviewNotificationPoster(sidebarPoster);

  await syncEditorActionsLocation(ctx, true);
  assert.strictEqual(sidebarPosted.length, 1);

  // Mode flip: the chat editor panel takes the poster over from the
  // sidebar while the question is open.
  notif.setWebviewNotificationPoster(panelPoster);
  await tick();
  assert.strictEqual(ctx.globalState.get(LOCATION_CONSENT_KEY), undefined);
  assert.strictEqual(
    panelPosted.length,
    0,
    'no toast re-posted to the webview',
  );
  assert.strictEqual(native.length, 1, 'the question moved to a native toast');

  // Dismissing the native toast is a real dismissal and still means keep.
  native[0].resolve(undefined);
  await tick();
  assert.strictEqual(ctx.globalState.get(LOCATION_CONSENT_KEY), 'keep');
  assert.deepStrictEqual(settingUpdates, []);
  notif.clearWebviewNotificationPoster(panelPoster);
  console.log('consent survives poster handover: passed');
}

async function genericToastRehomed() {
  // Any action toast, not only the consent question: a warning with
  // two buttons pending in the webview settles with the native answer
  // after its surface is torn down, and a later webview answer for the
  // old id is ignored.
  native.length = 0;
  const posted = [];
  const poster = msg => posted.push(msg);
  notif.setWebviewNotificationPoster(poster);
  const answer = notif.showWarningNotification('Keys?', 'Enter Key', 'Skip');
  assert.strictEqual(posted.length, 1);
  notif.clearWebviewNotificationPoster(poster);
  await tick();
  assert.strictEqual(native.length, 1);
  assert.strictEqual(native[0].severity, 'warning');
  assert.deepStrictEqual(native[0].actions, ['Enter Key', 'Skip']);
  notif.resolveWebviewNotificationAction(posted[0].id, 'Skip');
  native[0].resolve('Enter Key');
  assert.strictEqual(await answer, 'Enter Key');

  // A toast answered in the webview before the teardown is settled and
  // not shown natively again.
  native.length = 0;
  posted.length = 0;
  notif.setWebviewNotificationPoster(poster);
  const answered = notif.showErrorNotification('Broken', 'Retry');
  notif.resolveWebviewNotificationAction(posted[0].id, 'Retry');
  assert.strictEqual(await answered, 'Retry');
  notif.clearWebviewNotificationPoster(poster);
  await tick();
  assert.strictEqual(native.length, 0, 'an answered toast is not re-shown');

  // A native show that rejects settles the toast as dismissed instead
  // of leaving the caller hanging.
  posted.length = 0;
  vscodeStub.window.showInformationMessage = () =>
    Promise.reject(new Error('host gone'));
  notif.setWebviewNotificationPoster(poster);
  const rejected = notif.showInformationNotification('Info', 'OK');
  notif.clearWebviewNotificationPoster(poster);
  assert.strictEqual(await rejected, undefined);
  console.log('generic toast rehomed: passed');
}

(async () => {
  await consentSurvivesPanelClose();
  await consentSurvivesPosterHandover();
  await genericToastRehomed();
  console.log('audit1013_consent_teardown: all passed');
})().catch(err => {
  console.error(err);
  process.exit(1);
});
