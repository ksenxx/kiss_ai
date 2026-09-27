// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// A6 update nag (audit-extension H1): the "new release available" toast
// must not come back at every activation after the user closed it.
//
// Checker (src/UpdateChecker.js, real file I/O on a temp cache):
//  - skipUpdateVersion() persists the skipped version; that version and
//    older ones are never announced again, a newer one still is;
//  - the skip survives a cache refresh and a later snooze;
//  - `enabled: false` (the kissSorcar.checkForUpdates setting) makes
//    the check a no-op: no fetch, no cache, no toast.
// Host (out/extension.js activated against the stub):
//  - closing the toast without choosing an action snoozes it exactly
//    like 'Remind me later';
//  - 'Skip this version' calls skipUpdateVersion with the offered version;
//  - kissSorcar.checkForUpdates=false is passed to the checker as
//    enabled:false.

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');

const tmpHome = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-uap-upd-'));
process.env.KISS_HOME = tmpHome;
process.env.HOME = tmpHome;
process.env.USERPROFILE = tmpHome;

const {
  checkForExtensionUpdate,
  skipUpdateVersion,
  snoozeUpdateNotification,
} = require(path.join(__dirname, '..', 'src', 'UpdateChecker.js'));
const {loadExtension, waitFor} = require('./_ui_antipattern_host');

const cacheFilePath = path.join(tmpHome, 'update-cache.json');
const T0 = 1_000_000_000_000;

async function checkerTests() {
  const notified = [];
  const fetches = [];
  const base = {
    cacheFilePath,
    currentVersion: '1.0.0',
    cooldownMs: 6 * 3600_000,
    notify: n => notified.push(n),
  };
  const fetchLatest = latest => url => {
    fetches.push(url);
    return Promise.resolve(latest);
  };

  // A fresh check announces 1.1.0.
  let r = await checkForExtensionUpdate({
    ...base,
    now: () => T0,
    fetchLatest: fetchLatest('1.1.0'),
  });
  assert.strictEqual(r.reason, 'update-available');
  assert.strictEqual(notified.length, 1);

  // Skip 1.1.0: replayed from the cache inside the cooldown, and
  // re-fetched after it, the same version stays silent.
  const skip = skipUpdateVersion({latest: '1.1.0', cacheFilePath});
  assert.strictEqual(skip.skippedVersion, '1.1.0');
  r = await checkForExtensionUpdate({
    ...base,
    now: () => T0 + 60_000,
    fetchLatest: fetchLatest('1.1.0'),
  });
  assert.strictEqual(r.reason, 'skipped', 'cooldown replay of a skipped version');
  assert.strictEqual(r.notified, false);
  r = await checkForExtensionUpdate({
    ...base,
    now: () => T0 + 7 * 3600_000,
    fetchLatest: fetchLatest('1.1.0'),
  });
  assert.strictEqual(r.reason, 'skipped', 'fresh fetch of a skipped version');
  assert.strictEqual(notified.length, 1, 'no second toast for a skipped version');
  const refreshed = JSON.parse(fs.readFileSync(cacheFilePath, 'utf-8'));
  assert.strictEqual(
    refreshed.skippedVersion,
    '1.1.0',
    'the cache refresh must preserve the skipped version',
  );

  // A snooze on top of the skip keeps the skip.
  snoozeUpdateNotification({latest: '1.1.0', cacheFilePath, now: () => T0});
  assert.strictEqual(
    JSON.parse(fs.readFileSync(cacheFilePath, 'utf-8')).skippedVersion,
    '1.1.0',
    'snoozing must not erase the skipped version',
  );

  // A NEWER release is still announced.
  r = await checkForExtensionUpdate({
    ...base,
    now: () => T0 + 14 * 3600_000,
    fetchLatest: fetchLatest('1.2.0'),
  });
  assert.strictEqual(r.reason, 'update-available', 'a newer release notifies');
  assert.strictEqual(notified.length, 2);
  assert.strictEqual(notified[1].latest, '1.2.0');

  // Skipping with no explicit version skips the last release seen.
  skipUpdateVersion({cacheFilePath});
  assert.strictEqual(
    JSON.parse(fs.readFileSync(cacheFilePath, 'utf-8')).skippedVersion,
    '1.2.0',
  );

  // The setting: enabled:false never fetches, never notifies.
  const fetchesBefore = fetches.length;
  r = await checkForExtensionUpdate({
    ...base,
    enabled: false,
    now: () => T0 + 100 * 3600_000,
    fetchLatest: fetchLatest('9.9.9'),
  });
  assert.deepStrictEqual(r, {
    checked: false,
    notified: false,
    latest: null,
    current: null,
    reason: 'disabled',
  });
  assert.strictEqual(fetches.length, fetchesBefore, 'disabled: no fetch');
  assert.strictEqual(notified.length, 2, 'disabled: no toast');
  console.log('UpdateChecker skip / disabled tests passed');
}

async function hostTests() {
  const checks = [];
  const snoozes = [];
  const skips = [];
  const h = loadExtension({
    config: {'kissSorcar.editorTabsMode': false},
    modules: {
      'UpdateChecker.js': {
        checkForExtensionUpdate: async opts => {
          checks.push(opts);
          if (opts.enabled === false) {
            return {checked: false, notified: false, reason: 'disabled'};
          }
          opts.notify({latest: '2099.1.1', current: '2026.6.31'});
          return {checked: true, notified: true, reason: 'update-available'};
        },
        snoozeUpdateNotification: opts => {
          snoozes.push(opts);
          return {snoozeUntilMs: 0, snoozedLatest: opts && opts.latest};
        },
        skipUpdateVersion: opts => {
          skips.push(opts);
          return {skippedVersion: opts && opts.latest};
        },
      },
      'DependencyInstaller.js': {
        ensureLocalBinInPath: () => {},
        ensureDependencies: () => Promise.resolve(),
        promptApiKeysNow: () => Promise.resolve(true),
      },
    },
  });

  // --- 1. the toast is closed without an action: snoozed ---------------
  let ctx = h.makeContext();
  h.extension.activate(ctx);
  await waitFor(
    () => h.notifications.length === 1,
    'activation must show the update toast',
  );
  let toast = h.notifications[0];
  assert.strictEqual(toast.kind, 'info');
  assert.deepStrictEqual(toast.actions, [
    'Update now',
    'Update when idle',
    'Remind me later',
    'Skip this version',
  ]);
  assert.strictEqual(checks[0].enabled, true, 'default: checks enabled');
  toast.resolve(undefined);
  await waitFor(() => snoozes.length === 1, 'dismissing the toast must snooze');
  assert.deepStrictEqual(snoozes[0], {latest: '2099.1.1'});
  assert.strictEqual(skips.length, 0);
  h.extension.deactivate();
  h.disposeContext(ctx);

  // --- 2. 'Skip this version' persists the skipped version --------------
  ctx = h.makeContext();
  h.extension.activate(ctx);
  await waitFor(() => h.notifications.length === 2, 'second update toast');
  toast = h.notifications[1];
  toast.resolve('Skip this version');
  await waitFor(() => skips.length === 1, 'Skip this version must be recorded');
  assert.deepStrictEqual(skips[0], {latest: '2099.1.1'});
  assert.strictEqual(snoozes.length, 1, 'skipping must not also snooze');
  h.extension.deactivate();
  h.disposeContext(ctx);

  // --- 3. 'Remind me later' still snoozes; 'Update now' still updates ---
  ctx = h.makeContext();
  h.extension.activate(ctx);
  await waitFor(() => h.notifications.length === 3, 'third update toast');
  h.notifications[2].resolve('Remind me later');
  await waitFor(() => snoozes.length === 2, 'Remind me later snoozes');
  h.extension.deactivate();
  h.disposeContext(ctx);

  // --- 4. kissSorcar.checkForUpdates = false reaches the checker --------
  h.config['kissSorcar.checkForUpdates'] = false;
  ctx = h.makeContext();
  h.extension.activate(ctx);
  await waitFor(() => checks.length === 4, 'the checker is still invoked');
  assert.strictEqual(checks[3].enabled, false, 'the setting disables the check');
  await new Promise(r => setTimeout(r, 100));
  assert.strictEqual(h.notifications.length, 3, 'disabled: no update toast');
  h.extension.deactivate();
  h.disposeContext(ctx);
  console.log('Host update-toast dismissal / skip / setting tests passed');
}

checkerTests()
  .then(hostTests)
  .then(() => {
    fs.rmSync(tmpHome, {recursive: true, force: true});
    console.log('\nAll ui_antipattern_update_nag tests passed');
    process.exit(0);
  })
  .catch(err => {
    console.error(err);
    process.exit(1);
  });
