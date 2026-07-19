// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// Drain-aware macOS LaunchAgent restart for the kiss-web daemon.
//
// Why this exists
// ---------------
// ``restartKissWebDaemon`` (DependencyInstaller.ts) used to run the
// launchctl sequence ``bootout → bootstrap → kickstart -k`` back to
// back.  The macOS unified log from the 2026-07-19 14:30:58Z update
// shows why that made every post-update kiss-web launch take ~26s:
//
//   07:30:58.266  launchd: Setting service com.kiss.web-server to enabled
//   07:31:03.301  launchd: service inactive: com.kiss.web-server
//   07:31:03.301  launchd: removing service: com.kiss.web-server
//   07:31:13.539  launchd: service inactive: com.kiss.web-server
//   07:31:23.561  launchd: Successfully spawned kiss-web[90306]
//                 because inefficient
//
// The old daemon takes ~5s to drain after SIGTERM.  ``bootstrap``
// issued during that drain is accepted, but when the old instance
// finally exits the still-pending ``bootout`` REMOVES the whole
// service — including the fresh registration.  Nothing is loaded, so
// nothing respawns until ``verifyDaemonStartup`` re-issues the restart
// 15s later, and even that spawn is delayed several more seconds by
// launchd's default 10s ``ThrottleInterval`` ("because inefficient").
//
// :func:`restartLaunchAgent` fixes the first leg deterministically:
// after ``bootout`` it POLLS ``launchctl print`` until the service is
// POSITIVELY gone (the drain wait) and only then ``bootstrap``s the
// new registration.  The barrier fails CLOSED: an inconclusive probe
// (launchctl timing out under load, spawn failure) or a service that
// is still present at the drain deadline never authorizes a bootstrap
// — instead the existing registration is kicked and the caller's
// verifier (``verifyDaemonStartup``) retries the whole sequence later.
// (The second leg, the respawn throttle, is reduced by
// ``ThrottleInterval=5`` in the plist written by
// DependencyInstaller.ts, with an explicit ``kickstart`` for the
// managed-restart path.)
//
// All launchctl invocations are ASYNC (``child_process.execFile``) so
// the VS Code extension host's event loop is never blocked while
// launchd drains the old daemon.
//
// Implemented as a dependency-free CommonJS module (like
// ``daemonRestartVerify.js``) so it can be exercised end-to-end with
// plain ``node`` against the REAL launchd — no VS Code host required.

'use strict';

const {execFile} = require('child_process');

/** Default budget for the old job instance to drain after bootout
 * (ms).  launchd SIGKILLs a bootout'd instance that ignores SIGTERM
 * after its ExitTimeOut (20s default); 25s covers that plus slack. */
const DEFAULT_DRAIN_TIMEOUT_MS = 25_000;

/** Default delay between ``launchctl print`` drain polls (ms). */
const DEFAULT_POLL_INTERVAL_MS = 250;

/** Default number of ``bootstrap`` attempts before falling back to
 * ``launchctl load -w``. */
const DEFAULT_BOOTSTRAP_ATTEMPTS = 4;

/** Per-launchctl-command timeout (ms). */
const COMMAND_TIMEOUT_MS = 5_000;

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Run ``launchctl`` with the given argv, discarding output.
 *
 * @param {string} launchctlPath Binary to execute (the real
 *   ``launchctl`` in production; tests may point this at a controlled
 *   executable to reach failure branches).
 * @param {string[]} args argv passed verbatim (no shell).
 * @returns {Promise<{ok: boolean, conclusive: boolean}>} ``ok`` is
 *   true when the command exited 0.  ``conclusive`` is true when
 *   launchctl actually ran to completion and reported an exit status;
 *   false for a timeout / spawn failure (missing binary, EACCES, …),
 *   where the command outcome proves nothing about launchd state.
 */
function runLaunchctl(launchctlPath, args) {
  return new Promise(resolve => {
    execFile(
      launchctlPath,
      args,
      {timeout: COMMAND_TIMEOUT_MS},
      err => {
        if (!err) {
          resolve({ok: true, conclusive: true});
          return;
        }
        // ``err.code`` is the numeric exit status when the process
        // ran to completion, and a string errno (``'ENOENT'``, …) or
        // ``null``/``undefined`` (killed by the timeout) otherwise.
        const conclusive = typeof err.code === 'number';
        resolve({ok: false, conclusive});
      },
    );
  });
}

/**
 * Tri-state launchd service probe.
 *
 * @param {string} launchctlPath ``launchctl`` binary to execute.
 * @param {string} serviceTarget e.g. ``gui/501/com.kiss.web-server``.
 * @returns {Promise<'present'|'absent'|'unknown'>} ``'present'``
 *   while ``launchctl print`` succeeds (loaded, running, or
 *   draining), ``'absent'`` when launchctl positively reports the
 *   service is not there (non-zero exit), and ``'unknown'`` when the
 *   probe itself failed (timeout under load, missing binary) — which
 *   must NEVER be interpreted as "gone".
 */
async function probeService(launchctlPath, serviceTarget) {
  const res = await runLaunchctl(launchctlPath, ['print', serviceTarget]);
  if (res.ok) return 'present';
  return res.conclusive ? 'absent' : 'unknown';
}

/**
 * Boot out a LaunchAgent, wait until launchd POSITIVELY no longer
 * knows the service, bootstrap the (re)written plist, and kickstart
 * it.
 *
 * The drain wait is the essence: ``bootstrap`` is only issued once
 * ``launchctl print`` conclusively reports the service gone, so the
 * pending ``bootout`` can never remove the fresh registration (the
 * race that left kiss-web unloaded for 15s+ after every Update).
 * The barrier fails closed: if the service is still present — or the
 * probe is inconclusive — at the drain deadline, NO bootstrap is
 * attempted; the surviving registration is ``kickstart -k``ed
 * instead and the caller's ``verifyDaemonStartup`` retries the full
 * sequence later if the daemon is still down.
 *
 * @param {{
 *   serviceTarget: string,
 *   domainTarget: string,
 *   plistFile: string,
 *   launchctlPath?: string,
 *   log?: (msg: string) => void,
 *   drainTimeoutMs?: number,
 *   pollIntervalMs?: number,
 *   bootstrapAttempts?: number,
 * }} opts
 *   - ``serviceTarget``: ``gui/<uid>/<label>``.
 *   - ``domainTarget``: ``gui/<uid>``.
 *   - ``plistFile``: absolute path of the LaunchAgent plist.
 *   - ``launchctlPath``: launchctl binary (default ``'launchctl'``).
 *   - ``log``: diagnostic sink (defaults to a no-op).
 *   - ``drainTimeoutMs`` / ``pollIntervalMs`` / ``bootstrapAttempts``:
 *     timing overrides for tests.
 * @returns {Promise<{
 *   drainedMs: number,
 *   drained: boolean,
 *   bootstrapAttempts: number,
 *   bootstrapped: boolean,
 *   registered: boolean,
 *   kickstarted: boolean,
 * }>} Diagnostics: how long the drain wait took; whether the service
 *   was positively gone before bootstrap; how many bootstrap attempts
 *   ran (0 when the drain never completed and bootstrap was refused);
 *   whether a bootstrap attempt succeeded; whether SOME registration
 *   should exist afterwards (bootstrap, ``load -w`` fallback, or the
 *   undrained pre-existing registration); and whether the final
 *   kickstart succeeded.
 */
async function restartLaunchAgent(opts) {
  const serviceTarget = opts.serviceTarget;
  const domainTarget = opts.domainTarget;
  const plistFile = opts.plistFile;
  const launchctlPath = opts.launchctlPath || 'launchctl';
  const log = opts.log || (() => {});
  const drainTimeoutMs =
    typeof opts.drainTimeoutMs === 'number'
      ? opts.drainTimeoutMs
      : DEFAULT_DRAIN_TIMEOUT_MS;
  const pollIntervalMs =
    typeof opts.pollIntervalMs === 'number'
      ? opts.pollIntervalMs
      : DEFAULT_POLL_INTERVAL_MS;
  const bootstrapAttempts =
    typeof opts.bootstrapAttempts === 'number'
      ? opts.bootstrapAttempts
      : DEFAULT_BOOTSTRAP_ATTEMPTS;

  // 1. Ask launchd to unload the service.  Failure is fine (the
  //    service may simply not be loaded).
  await runLaunchctl(launchctlPath, ['bootout', serviceTarget]);

  // 2. Drain wait: poll until launchd POSITIVELY no longer knows the
  //    service (the old instance can take seconds to unwind; launchd
  //    only forgets the service once its process is gone).  Skipping
  //    this wait is the bug: a bootstrap issued mid-drain is wiped
  //    out together with the service when the old instance exits.
  //    An ``'unknown'`` probe (launchctl timed out under load, spawn
  //    failure) never counts as drained — bootstrapping on an
  //    inconclusive probe would reintroduce the race.
  const drainStart = Date.now();
  let state = await probeService(launchctlPath, serviceTarget);
  while (state !== 'absent' && Date.now() - drainStart < drainTimeoutMs) {
    await sleep(pollIntervalMs);
    state = await probeService(launchctlPath, serviceTarget);
  }
  const drained = state === 'absent';
  const drainedMs = Date.now() - drainStart;

  if (!drained) {
    // Fail closed: NEVER bootstrap while the service is (or may
    // still be) registered — the pending bootout could remove the
    // fresh registration, recreating the original race.  Kick the
    // surviving registration instead (``-k`` also kills a wedged
    // instance); if the daemon still does not come up, the caller's
    // verifier re-runs this whole sequence.
    const kick = await runLaunchctl(launchctlPath, [
      'kickstart',
      '-k',
      serviceTarget,
    ]);
    log(
      `launchd service ${serviceTarget} still ${state} after ` +
        `${drainedMs}ms drain wait — bootstrap refused (fail closed); ` +
        `kickstarted existing registration: ${kick.ok}`,
    );
    return {
      drainedMs,
      drained: false,
      bootstrapAttempts: 0,
      bootstrapped: false,
      registered: state === 'present',
      kickstarted: kick.ok,
    };
  }

  // 3. Bootstrap the new registration, retrying briefly: even after
  //    the drain wait launchd can transiently report EEXIST/EBUSY
  //    while its internal bookkeeping catches up.  Only conclusive
  //    non-zero exits are retried — a timeout or spawn failure will
  //    not resolve within this loop.
  let attempts = 0;
  let bootstrapped = false;
  let retryable = true;
  while (attempts < bootstrapAttempts && !bootstrapped && retryable) {
    attempts += 1;
    const res = await runLaunchctl(launchctlPath, [
      'bootstrap',
      domainTarget,
      plistFile,
    ]);
    bootstrapped = res.ok;
    retryable = res.conclusive;
    if (!bootstrapped && retryable && attempts < bootstrapAttempts) {
      await sleep(pollIntervalMs);
    }
  }
  let registered = bootstrapped;
  if (!bootstrapped) {
    // Fall back to the legacy load command; also best-effort.
    const loaded = await runLaunchctl(launchctlPath, ['load', '-w', plistFile]);
    registered = loaded.ok;
    log(
      `launchctl bootstrap ${serviceTarget} failed after ${attempts} ` +
        `attempt(s) — load -w fallback ${loaded.ok ? 'succeeded' : 'failed'}`,
    );
  }

  // 4. ``bootstrap``/``load`` with RunAtLoad already start the fresh
  //    instance; a plain ``kickstart`` (no ``-k`` — that would KILL
  //    the just-spawned daemon and force a second start) covers the
  //    "registered but not started" corner.  Best-effort: KeepAlive
  //    starts it eventually.
  const kick = await runLaunchctl(launchctlPath, ['kickstart', serviceTarget]);

  return {
    drainedMs,
    drained,
    bootstrapAttempts: attempts,
    bootstrapped,
    registered,
    kickstarted: kick.ok,
  };
}

module.exports = {
  restartLaunchAgent,
  probeService,
  DEFAULT_DRAIN_TIMEOUT_MS,
  DEFAULT_POLL_INTERVAL_MS,
  DEFAULT_BOOTSTRAP_ATTEMPTS,
  COMMAND_TIMEOUT_MS,
};
