// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const fs = require('fs');
const {probeDaemonHealth, daemonHasActiveTasks} = require('./daemonHealth');

const DEFAULT_TIMEOUT_MS = 180_000;

const DEFAULT_POLL_INTERVAL_MS = 1_000;

const DEFAULT_RESTART_EVERY_MS = 15_000;

const DEFAULT_PROBE_TIMEOUT_MS = 1_000;

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

const fileExists = fs.existsSync;

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
  let lastRestartAt = startedAt;
  let restarts = 0;
  let binaryVanished = false;

  for (;;) {
    const health = await probeDaemonHealth(port, probeTimeoutMs);
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
