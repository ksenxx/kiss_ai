// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const net = require('net');
const fs = require('fs');

function probeDaemonHealth(port, timeoutMs) {
  const timeout = typeof timeoutMs === 'number' ? timeoutMs : 1500;
  return new Promise(resolve => {
    let settled = false;
    const finish = result => {
      if (settled) return;
      settled = true;
      try {
        sock.destroy();
      } catch {
      }
      resolve(result);
    };
    const sock = net.connect({host: '127.0.0.1', port, timeout});
    sock.once('connect', () => finish('alive'));
    sock.once('timeout', () => finish('unknown'));
    sock.once('error', err => {
      const code = err && err.code;
      if (code === 'ECONNREFUSED') {
        finish('dead');
      } else {
        finish('unknown');
      }
    });
  });
}

/**
 * Ask the daemon named by the endpoint file how many tasks are running.
 *
 * Connects over the daemon's local WSS endpoint (token auth, CA
 * pinned), sends `activeTasksQuery` and resolves with
 * `{ok: true, count, tabs}` or `{ok: false, reason}`.  `reason` is
 * `endpoint-missing` when no daemon has published an endpoint or the
 * published one refuses connections (a stale file).
 *
 * @param {string} endpointPath The daemon's endpoint file.
 * @param {number} [timeoutMs] Overall probe budget (default 1500).
 * @returns {Promise<{ok: true, count: number, tabs: string[]} | {ok: false, reason: string}>}
 */
function daemonHasActiveTasks(endpointPath, timeoutMs) {
  // Required here, not at module load: these are compiled TypeScript
  // modules, and the pure helpers in this file (`sleep`, `decideRestart`,
  // `probeDaemonHealth`) are also loaded straight from src/ by tests.
  const {readLocalEndpoint} = require('./userAssets');
  const {WsClient} = require('./wsClient');
  const timeout = typeof timeoutMs === 'number' ? timeoutMs : 1500;
  return new Promise(resolve => {
    const endpoint = readLocalEndpoint(endpointPath);
    if (!endpoint) {
      resolve({ok: false, reason: 'endpoint-missing'});
      return;
    }
    let ca;
    if (endpoint.ca) {
      try {
        ca = fs.readFileSync(endpoint.ca, 'utf8');
      } catch (err) {
        resolve({ok: false, reason: 'ca-unreadable:' + (err && err.code)});
        return;
      }
    }

    let settled = false;
    let authenticated = false;
    const finish = result => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      try {
        ws.destroy();
      } catch {
      }
      resolve(result);
    };
    const timer = setTimeout(() => finish({ok: false, reason: 'timeout'}), timeout);
    const ws = new WsClient({url: endpoint.url, ca, connectTimeoutMs: timeout});
    ws.on('open', () => {
      ws.send(JSON.stringify({type: 'auth', token: endpoint.token}));
    });
    ws.on('message', text => {
      let parsed;
      try {
        parsed = JSON.parse(text);
      } catch {
        return;
      }
      if (!parsed || typeof parsed !== 'object') return;
      if (!authenticated) {
        if (parsed.type === 'auth_ok' && parsed.local === true) {
          authenticated = true;
          ws.send(JSON.stringify({type: 'activeTasksQuery'}));
        } else {
          finish({ok: false, reason: 'auth-rejected'});
        }
        return;
      }
      if (parsed.type === 'activeTasksResponse') {
        const count = typeof parsed.count === 'number' ? parsed.count : -1;
        const tabs = Array.isArray(parsed.tabs)
          ? parsed.tabs.filter(t => typeof t === 'string')
          : [];
        if (count < 0) {
          finish({ok: false, reason: 'missing-count'});
          return;
        }
        finish({ok: true, count, tabs});
        return;
      }
      if (
        parsed.type === 'error' &&
        typeof parsed.text === 'string' &&
        parsed.text.indexOf('Unknown command: activeTasksQuery') >= 0
      ) {
        // An old daemon that cannot answer the query conveys NO
        // information about whether it is running a task. Reporting
        // "zero active tasks" here would authorize a restart that can
        // abort in-flight work in exactly the process being upgraded.
        finish({ok: false, reason: 'unsupported-query'});
      }
    });
    ws.on('error', err => {
      const code = err && err.code;
      if (code === 'ECONNREFUSED') {
        finish({ok: false, reason: 'endpoint-missing'});
        return;
      }
      finish({ok: false, reason: 'error:' + (code || (err && err.message))});
    });
    ws.on('close', () => {
      finish({ok: false, reason: 'eof'});
    });
    ws.connect();
  });
}

function decideRestart(state) {
  const {fingerprintMatches, health, activeTasks, force} = state;
  if (force) {
    // The user answered "Restart now" to the deferred-update
    // notification: they accept aborting whatever the daemon reports
    // as running.  This is the only way out when the report is wrong
    // — a daemon wedged on a stale busy tab keeps deferring the very
    // restart that would load the code fixing it.
    return {skip: false, reason: 'forced-by-user'};
  }
  if (activeTasks && activeTasks.ok && activeTasks.count > 0) {
    return {skip: true, reason: 'active-tasks'};
  }
  if (
    health === 'alive' &&
    activeTasks && !activeTasks.ok &&
    activeTasks.reason === 'endpoint-missing'
  ) {
    return {
      skip: false,
      reason: 'unreachable-local (alive but endpoint file missing)',
    };
  }
  if (health === 'alive' && !(activeTasks && activeTasks.ok)) {
    const reason = activeTasks && activeTasks.reason
      ? activeTasks.reason : 'no-probe';
    return {
      skip: true,
      reason: `alive-uncertain (activeTasks=${reason})`,
    };
  }
  if (fingerprintMatches && health !== 'dead') {
    return {skip: true, reason: `healthy-unchanged (health=${health})`};
  }
  return {
    skip: false,
    reason:
      `restart-required (fingerprintMatches=${fingerprintMatches}, ` +
      `health=${health}, activeTasks=${activeTasks && activeTasks.ok ? activeTasks.count : 'unknown'})`,
  };
}

/**
 * Resolve after `ms` milliseconds.
 *
 * @param {number} ms How long to wait.
 * @returns {Promise<void>} Settles once the delay has elapsed.
 */
function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

module.exports = {
  probeDaemonHealth,
  daemonHasActiveTasks,
  decideRestart,
  sleep,
};
