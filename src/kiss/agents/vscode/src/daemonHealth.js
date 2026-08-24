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

function daemonHasActiveTasks(sockPath, timeoutMs) {
  const timeout = typeof timeoutMs === 'number' ? timeoutMs : 1500;
  return new Promise(resolve => {
    try {
      if (!fs.existsSync(sockPath)) {
        resolve({ok: false, reason: 'sock-missing'});
        return;
      }
    } catch {
      resolve({ok: false, reason: 'sock-stat-failed'});
      return;
    }

    let settled = false;
    let buf = '';
    const finish = result => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      try {
        sock.destroy();
      } catch {
      }
      resolve(result);
    };
    const timer = setTimeout(() => finish({ok: false, reason: 'timeout'}), timeout);
    const sock = net.createConnection(sockPath);
    sock.setEncoding('utf-8');
    sock.once('connect', () => {
      try {
        sock.write(JSON.stringify({type: 'activeTasksQuery'}) + '\n');
      } catch (err) {
        finish({ok: false, reason: 'write-failed:' + (err && err.code)});
      }
    });
    sock.on('data', chunk => {
      buf += chunk;
      let nl = buf.indexOf('\n');
      while (nl >= 0) {
        const line = buf.slice(0, nl);
        buf = buf.slice(nl + 1);
        nl = buf.indexOf('\n');
        if (line.length === 0) continue;
        let parsed;
        try {
          parsed = JSON.parse(line);
        } catch {
          continue;
        }
        if (!parsed || typeof parsed !== 'object') continue;
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
          return;
        }
      }
    });
    sock.once('error', err => {
      const code = err && err.code;
      if (code === 'ENOENT' || code === 'ECONNREFUSED' ||
          code === 'ENOTSOCK') {
        finish({ok: false, reason: 'sock-missing'});
        return;
      }
      finish({ok: false, reason: 'error:' + code});
    });
    sock.once('end', () => {
      finish({ok: false, reason: 'eof'});
    });
  });
}

function decideRestart(state) {
  const {fingerprintMatches, health, activeTasks} = state;
  if (activeTasks && activeTasks.ok && activeTasks.count > 0) {
    return {skip: true, reason: 'active-tasks'};
  }
  if (
    health === 'alive' &&
    activeTasks && !activeTasks.ok &&
    activeTasks.reason === 'sock-missing'
  ) {
    return {
      skip: false,
      reason: 'unreachable-uds (alive but socket file missing)',
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
