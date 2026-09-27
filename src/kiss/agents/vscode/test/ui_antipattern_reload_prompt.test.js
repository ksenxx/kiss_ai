// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// A7 forced reload (audit-extension H3): after an in-place update the
// window is never reloaded behind the user's back.  Drives the real
// `.extension-updated` watcher in out/extension.js (fs.watchFile on a
// temp $KISS_HOME) with a reloadGuard that reports the bundle ready:
//  - the settle logic ends in ONE non-modal toast
//    'KISS Sorcar was updated.' with 'Reload now' / 'Later';
//  - 'Later' (and closing the toast) leaves the window alone;
//  - 'Reload now' runs workbench.action.reloadWindow, and only then;
//  - the prompt is shown once per activation even when the marker keeps
//    changing (reloadTriggered).

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');

const tmpHome = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-uap-reload-'));
process.env.KISS_HOME = tmpHome;
process.env.HOME = tmpHome;
process.env.USERPROFILE = tmpHome;

const {loadExtension, waitFor, sleep} = require('./_ui_antipattern_host');

const markerPath = path.join(tmpHome, '.extension-updated');
const RELOAD_CMD = 'workbench.action.reloadWindow';

const h = loadExtension({
  config: {'kissSorcar.editorTabsMode': false},
  modules: {
    'reloadGuard.js': {
      isReloadReady: () => ({codeReady: true, socketUp: true, size: 1}),
    },
    'UpdateChecker.js': {
      checkForExtensionUpdate: () => Promise.resolve({checked: false}),
      snoozeUpdateNotification: () => ({}),
      skipUpdateVersion: () => ({}),
    },
    'DependencyInstaller.js': {
      ensureLocalBinInPath: () => {},
      ensureDependencies: () => Promise.resolve(),
      promptApiKeysNow: () => Promise.resolve(true),
    },
  },
});

function reloads() {
  return h.executedCommands.filter(c => c.cmd === RELOAD_CMD).length;
}

function updatedToasts() {
  return h.notifications.filter(n => n.message === 'KISS Sorcar was updated.');
}

async function scenario(answer) {
  fs.rmSync(markerPath, {force: true});
  const ctx = h.makeContext();
  h.extension.activate(ctx);
  // Let fs.watchFile take its baseline before the marker appears.
  await sleep(300);
  const before = updatedToasts().length;
  fs.writeFileSync(markerPath, new Date().toISOString() + '\n');
  // watchFile polls every 2 s, the settle timer every 0.5 s.
  await waitFor(
    () => updatedToasts().length === before + 1,
    'the update marker must end in a "was updated" toast',
    400,
  );
  const toast = updatedToasts()[before];
  assert.strictEqual(toast.kind, 'info', 'a non-modal information toast');
  assert.deepStrictEqual(toast.actions, ['Reload now', 'Later']);
  const reloadsBefore = reloads();
  // Nothing reloads while the toast is open.
  await sleep(200);
  assert.strictEqual(reloads(), reloadsBefore, 'no reload before the answer');
  // The marker changing again must not stack a second prompt.
  fs.writeFileSync(markerPath, new Date().toISOString() + 'x\n');
  await sleep(2600);
  assert.strictEqual(
    updatedToasts().length,
    before + 1,
    'the prompt is shown once per activation',
  );
  toast.resolve(answer);
  await sleep(100);
  h.extension.deactivate();
  h.disposeContext(ctx);
  return reloads() - reloadsBefore;
}

async function runTest() {
  assert.strictEqual(await scenario('Later'), 0, "'Later' must not reload");
  assert.strictEqual(await scenario(undefined), 0, 'closing must not reload');
  assert.strictEqual(await scenario('Reload now'), 1, "'Reload now' reloads");
  fs.rmSync(tmpHome, {recursive: true, force: true});
  console.log('\nAll ui_antipattern_reload_prompt tests passed');
}

runTest()
  .then(() => process.exit(0))
  .catch(err => {
    console.error(err);
    process.exit(1);
  });
