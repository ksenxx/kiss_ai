// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const {execFile} = require('child_process');

const DEFAULT_DRAIN_TIMEOUT_MS = 25_000;

const DEFAULT_POLL_INTERVAL_MS = 250;

const DEFAULT_BOOTSTRAP_ATTEMPTS = 4;

const COMMAND_TIMEOUT_MS = 5_000;

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

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
        const conclusive = typeof err.code === 'number';
        resolve({ok: false, conclusive});
      },
    );
  });
}

async function probeService(launchctlPath, serviceTarget) {
  const res = await runLaunchctl(launchctlPath, ['print', serviceTarget]);
  if (res.ok) return 'present';
  return res.conclusive ? 'absent' : 'unknown';
}

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

  await runLaunchctl(launchctlPath, ['bootout', serviceTarget]);

  const drainStart = Date.now();
  let state = await probeService(launchctlPath, serviceTarget);
  while (state !== 'absent' && Date.now() - drainStart < drainTimeoutMs) {
    await sleep(pollIntervalMs);
    state = await probeService(launchctlPath, serviceTarget);
  }
  const drained = state === 'absent';
  const drainedMs = Date.now() - drainStart;

  if (!drained) {
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
    const loaded = await runLaunchctl(launchctlPath, ['load', '-w', plistFile]);
    registered = loaded.ok;
    log(
      `launchctl bootstrap ${serviceTarget} failed after ${attempts} ` +
        `attempt(s) — load -w fallback ${loaded.ok ? 'succeeded' : 'failed'}`,
    );
  }

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
};
