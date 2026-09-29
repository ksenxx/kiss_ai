// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// After ./install.sh replaces the extension on disk, the window must
// reload on its own: no 'KISS Sorcar was updated. Reload now / Later'
// toast that leaves the update stuck until someone clicks it.  Drives
// the real `.extension-updated` watcher in out/extension.js
// (fs.watchFile on a temp $KISS_HOME) with a reloadGuard that reports
// the bundle ready:
//  - the settle logic ends in workbench.action.reloadWindow with no
//    notification of any kind;
//  - the reload runs once per activation even when the marker keeps
//    changing (reloadTriggered);
//  - an activation whose marker never changes does not reload.

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
  return h.notifications.filter(n => /was updated/.test(n.message));
}

async function markerWrittenReloadsOnce() {
  fs.rmSync(markerPath, {force: true});
  const ctx = h.makeContext();
  h.extension.activate(ctx);
  // Let fs.watchFile take its baseline before the marker appears.
  await sleep(300);
  const reloadsBefore = reloads();
  const toastsBefore = updatedToasts().length;
  fs.writeFileSync(markerPath, new Date().toISOString() + '\n');
  // watchFile polls every 2 s, the settle timer every 0.5 s.
  await waitFor(
    () => reloads() === reloadsBefore + 1,
    'the update marker must end in workbench.action.reloadWindow',
    400,
  );
  assert.strictEqual(
    updatedToasts().length,
    toastsBefore,
    'the reload must not be gated behind a "was updated" toast',
  );
  // The marker changing again must not stack a second reload.
  fs.writeFileSync(markerPath, new Date().toISOString() + 'x\n');
  await sleep(2600);
  assert.strictEqual(
    reloads(),
    reloadsBefore + 1,
    'the reload runs once per activation',
  );
  h.extension.deactivate();
  h.disposeContext(ctx);
}

async function noMarkerNoReload() {
  fs.rmSync(markerPath, {force: true});
  const ctx = h.makeContext();
  h.extension.activate(ctx);
  const reloadsBefore = reloads();
  await sleep(2600);
  assert.strictEqual(reloads(), reloadsBefore, 'no marker, no reload');
  h.extension.deactivate();
  h.disposeContext(ctx);
}

async function runTest() {
  await markerWrittenReloadsOnce();
  await noMarkerNoReload();
  fs.rmSync(tmpHome, {recursive: true, force: true});
  console.log('\nAll ui_antipattern_reload_prompt tests passed');
}

runTest()
  .then(() => process.exit(0))
  .catch(err => {
    console.error(err);
    process.exit(1);
  });
