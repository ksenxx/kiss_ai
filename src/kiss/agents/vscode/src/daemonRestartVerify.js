// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// Post-restart startup verification for the kiss-web daemon.
//
// Why this exists
// ---------------
// ``restartKissWebDaemon`` (DependencyInstaller.ts) used to kill the
// old daemon, run ``launchctl bootout/bootstrap/kickstart`` (or the
// systemd equivalent) with ``stdio: 'ignore'``, log "restarted" and
// write the fingerprint file — WITHOUT ever checking that a new daemon
// actually came up.  Two real-world races then stranded the user on
// the "KISS Sorcar Server is starting ..." overlay after running
// ``./install.sh``:
//
//   1. bootout → bootstrap race: ``launchctl bootstrap`` fails with
//      EEXIST while the old job instance is still draining; the
//      ``launchctl load -w`` fallback exits 0 without loading
//      anything; ``kickstart`` then has nothing to kick.  Nothing is
//      loaded, so the daemon NEVER respawns.  (Production logs show a
//      second VS Code window's dependency check healing this 42-136 s
//      later; a single-window user is stuck until they restart
//      VS Code by hand.)
//
//   2. venv-wipe race: a concurrent ``code --install-extension
//      --force`` (second ``install.sh`` run or second window) deletes
//      the extension directory — including ``.venv/bin/kiss-web``
//      that the LaunchAgent execs.  launchd's exec fails silently on
//      every KeepAlive tick; the daemon stays down until something
//      re-runs ``uv sync`` (observed outage: 2 m 16 s, log timestamp
//      2026-07-19T04:50:10Z).
//
// :func:`verifyDaemonStartup` closes both holes: after the caller's
// initial bootstrap it polls the REAL daemon state (TCP probe on the
// WSS port + existence of the ``~/.kiss/sorcar.sock`` UDS file) and,
// while the port is provably dead and the kiss-web binary exists,
// re-invokes the caller-supplied ``restart`` callback with a backoff.
// The caller writes the fingerprint file ONLY on a verified startup,
// so a failed restart is retried on the next activation instead of
// being masked forever.
//
// Implemented as a dependency-free CommonJS module (like
// ``daemonHealth.js``) so it can be exercised end-to-end with plain
// ``node`` against real TCP/UDS servers — no VS Code host required.

'use strict';

const fs = require('fs');
const {probeDaemonHealth, daemonHasActiveTasks} = require('./daemonHealth');

/** Default overall verification budget (ms).  Covers the worst
 * observed healthy-restart latency (the old daemon can take 30 s to
 * unwind plus launchd throttle) with ample headroom for a concurrent
 * ``uv sync`` to rebuild a wiped venv. */
const DEFAULT_TIMEOUT_MS = 180_000;

/** Default delay between state polls (ms). */
const DEFAULT_POLL_INTERVAL_MS = 1_000;

/** Default minimum spacing between two ``restart`` invocations (ms).
 * Long enough that a freshly-spawned daemon has time to bind its
 * listeners before we would consider bouncing it again. */
const DEFAULT_RESTART_EVERY_MS = 15_000;

/** Default per-poll TCP probe timeout (ms). */
const DEFAULT_PROBE_TIMEOUT_MS = 1_000;

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// ``fs.existsSync`` returns false (never throws) on any error, so no
// defensive wrapper is needed.
const fileExists = fs.existsSync;

/**
 * Poll until the restarted kiss-web daemon is provably up, re-issuing
 * the restart while it is provably down.
 *
 * The daemon counts as UP only when BOTH are true:
 *
 *   - the TCP probe on ``port`` reports ``'alive'`` (the WSS listener
 *     accepted a connection), AND
 *   - a REAL round-trip over the UDS at ``sockPath`` succeeds
 *     (``daemonHasActiveTasks`` — the endpoint the VS Code extension's
 *     ``AgentClient`` must reach; a mere socket FILE can be a stale
 *     leftover of a SIGKILLed daemon, and a daemon that is alive on
 *     TCP but unreachable over the UDS still leaves the overlay up).
 *
 * The ``restart`` callback is re-invoked only when the daemon is
 * provably absent on BOTH endpoints: the TCP probe reports ``'dead'``
 * (actively refused) AND the UDS round-trip reports ``'sock-missing'``
 * (no socket file, or a stale file with nothing listening behind it)
 * AND the kiss-web binary exists AND at least ``restartEveryMs`` has
 * elapsed since the previous restart attempt.  This gate is what makes
 * re-restarting safe:
 *
 *   - kiss-web binds its UDS listener during startup independently of
 *     the WSS port, so a mid-boot daemon can be answering on the UDS
 *     (and may already have picked up user tasks) while the TCP probe
 *     still says ``'dead'`` — bouncing it then would abort those
 *     tasks.  A successful UDS round-trip therefore always vetoes the
 *     restart.
 *   - A ``'unknown'`` TCP probe (timeout under load) never triggers a
 *     restart — bouncing a possibly-booting daemon on an inconclusive
 *     probe is how tasks get killed.
 *   - A missing binary never triggers a restart either: launchd/
 *     systemd cannot exec it, so the verifier just keeps waiting for
 *     the concurrent reinstall to put the binary back, then restarts.
 *
 * @param {{
 *   binPath: string,
 *   sockPath: string,
 *   port: number,
 *   restart?: (() => void | Promise<void>) | null,
 *   log?: (msg: string) => void,
 *   timeoutMs?: number,
 *   pollIntervalMs?: number,
 *   restartEveryMs?: number,
 *   probeTimeoutMs?: number,
 * }} opts
 *   - ``binPath``: absolute path to ``.venv/bin/kiss-web`` (may vanish
 *     mid-verification when a concurrent reinstall wipes the venv).
 *   - ``sockPath``: absolute path to the daemon's UDS socket file.
 *   - ``port``: the daemon's TCP (WSS) port, e.g. 8787.
 *   - ``restart``: callback that re-issues the full daemon (re)start
 *     sequence (bootout/bootstrap/kickstart, systemd restart, or a
 *     direct spawn).  May be async — a returned promise is awaited.
 *     Exceptions/rejections are caught and logged; ``null`` /
 *     omitted disables re-restarts (poll-only mode).
 *   - ``log``: diagnostic sink (defaults to a no-op).
 *   - ``timeoutMs`` / ``pollIntervalMs`` / ``restartEveryMs`` /
 *     ``probeTimeoutMs``: timing overrides for tests.
 * @returns {Promise<{
 *   ok: boolean,
 *   reason: 'alive'|'timeout'|'sock-missing'|'binary-missing',
 *   waitedMs: number,
 *   restarts: number,
 *   binaryVanished: boolean,
 * }>}
 *   - ``ok: true, reason: 'alive'`` — TCP alive and UDS file present.
 *   - ``ok: false, reason: 'binary-missing'`` — budget exhausted with
 *     the kiss-web binary absent (concurrent reinstall wiped the venv
 *     and nothing rebuilt it in time).
 *   - ``ok: false, reason: 'sock-missing'`` — budget exhausted with
 *     TCP alive but no UDS file (daemon unreachable from the
 *     extension).
 *   - ``ok: false, reason: 'timeout'`` — budget exhausted, binary
 *     present, daemon never came up.
 *   - ``restarts`` — how many times ``restart`` was re-invoked.
 *   - ``binaryVanished`` — whether the binary was observed missing at
 *     any point (diagnoses the venv-wipe race).
 */
async function verifyDaemonStartup(opts) {
  const binPath = opts.binPath;
  const sockPath = opts.sockPath;
  const port = opts.port;
  const restart = opts.restart || null;
  const log = opts.log || (() => {});
  const timeoutMs =
    typeof opts.timeoutMs === 'number' ? opts.timeoutMs : DEFAULT_TIMEOUT_MS;
  const pollIntervalMs =
    typeof opts.pollIntervalMs === 'number'
      ? opts.pollIntervalMs
      : DEFAULT_POLL_INTERVAL_MS;
  const restartEveryMs =
    typeof opts.restartEveryMs === 'number'
      ? opts.restartEveryMs
      : DEFAULT_RESTART_EVERY_MS;
  const probeTimeoutMs =
    typeof opts.probeTimeoutMs === 'number'
      ? opts.probeTimeoutMs
      : DEFAULT_PROBE_TIMEOUT_MS;

  const startedAt = Date.now();
  // Start the restart clock NOW: the caller invoked its own bootstrap
  // immediately before calling this function, so the first re-restart
  // must wait a full ``restartEveryMs`` — re-bouncing a daemon that is
  // merely slow to boot would kill it mid-startup.
  let lastRestartAt = startedAt;
  let restarts = 0;
  let binaryVanished = false;

  for (;;) {
    const health = await probeDaemonHealth(port, probeTimeoutMs);
    // Real UDS round-trip (not a mere file-existence check): a stale
    // socket file left behind by a SIGKILLed daemon must not count as
    // "up", and a UDS that ANSWERS proves a daemon is present (possibly
    // mid-boot, possibly already running tasks) even while the TCP
    // probe still reports dead.
    const uds = await daemonHasActiveTasks(sockPath, probeTimeoutMs);
    if (health === 'alive' && uds.ok) {
      return {
        ok: true,
        reason: 'alive',
        waitedMs: Date.now() - startedAt,
        restarts,
        binaryVanished,
      };
    }

    const binOk = fileExists(binPath);
    if (!binOk) binaryVanished = true;

    if (Date.now() - startedAt >= timeoutMs) {
      let reason;
      if (!binOk) reason = 'binary-missing';
      else if (health === 'alive') reason = 'sock-missing';
      else reason = 'timeout';
      return {
        ok: false,
        reason,
        waitedMs: Date.now() - startedAt,
        restarts,
        binaryVanished,
      };
    }

    if (
      restart &&
      binOk &&
      health === 'dead' &&
      !uds.ok &&
      uds.reason === 'sock-missing' &&
      Date.now() - lastRestartAt >= restartEveryMs
    ) {
      restarts += 1;
      log(
        `kiss-web still down ${Date.now() - startedAt}ms after restart ` +
          `(probe=${health}, uds=${uds.reason}) — re-issuing daemon ` +
          `restart (attempt ${restarts})`,
      );
      try {
        // ``restart`` may be synchronous (legacy) or async (the
        // drain-aware macLaunchd sequence).  Awaiting the returned
        // promise keeps async failures out of the unhandled-rejection
        // path and stops the poll loop from probing mid-restart.
        await restart();
      } catch (err) {
        log(
          'kiss-web re-restart attempt failed: ' +
            (err && err.message ? err.message : String(err)),
        );
      }
      lastRestartAt = Date.now();
    }

    await sleep(pollIntervalMs);
  }
}

module.exports = {
  verifyDaemonStartup,
  DEFAULT_TIMEOUT_MS,
  DEFAULT_POLL_INTERVAL_MS,
  DEFAULT_RESTART_EVERY_MS,
  DEFAULT_PROBE_TIMEOUT_MS,
};
