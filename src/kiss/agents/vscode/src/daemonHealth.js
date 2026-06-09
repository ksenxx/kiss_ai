// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// Health probes for the kiss-web daemon.
//
// Why this exists
// ---------------
// Before the dep-installer's ``restartKissWebDaemon`` was allowed to send
// SIGTERM to the running ``kiss-web`` daemon it tried to short-circuit when
// the daemon was already healthy.  The original probe used::
//
//     try {
//       execSync('lsof -i :8787 -t', {stdio:'ignore', timeout: 2000});
//       return fs.existsSync(sockPath);
//     } catch { return false; }
//
// which conflates THREE distinct outcomes into a single ``false``:
//
//   1. lsof exit status 1 — confirmed no listener on the port.
//   2. lsof was SIGTERMed by the 2 s ``timeout`` while the host was under
//      heavy load (the actual production failure mode).
//   3. lsof was not installed / spawn-error.
//
// Treating "I couldn't tell" as "definitely dead" caused the installer to
// SIGTERM a perfectly healthy daemon mid-task, killing a multi-hour agent
// run.  This module provides:
//
//   * :func:`probeDaemonHealth` — a tri-state TCP probe that returns
//     ``'alive' | 'dead' | 'unknown'``.  Uses Node's built-in ``net``
//     module instead of shelling out to ``lsof``, so it cannot stall on a
//     loaded host the way ``lsof`` can.
//   * :func:`daemonHasActiveTasks` — asks the daemon over its
//     ``~/.kiss/sorcar.sock`` UDS whether any agent tasks are currently
//     in flight.  When the answer is "yes" the installer must defer the
//     restart unconditionally — even if the fingerprint changed — because
//     a kiss-web restart aborts the in-flight task.
//
// Implemented as a dependency-free CommonJS module (rather than inline in
// ``DependencyInstaller.ts``) so it can be exercised directly with
// ``node`` in an integration test without spinning up a VS Code extension
// host.

'use strict';

const net = require('net');
const fs = require('fs');

/**
 * Probe whether a TCP listener exists on ``localhost:port``.
 *
 * Returns one of:
 *
 *   - ``'alive'``    — a TCP server accepted our connection within
 *                      ``timeoutMs``.
 *   - ``'dead'``     — the connect was actively refused (ECONNREFUSED)
 *                      or the host explicitly said "no such address"
 *                      (ECONNRESET on the very first packet); the
 *                      caller may treat the daemon as gone.
 *   - ``'unknown'``  — the probe could not be completed (timeout,
 *                      ENETUNREACH, EHOSTUNREACH, EACCES, spawn failure,
 *                      etc.).  Callers MUST NOT conflate this with
 *                      ``'dead'`` when deciding whether to SIGTERM the
 *                      daemon — under heavy load a healthy daemon can
 *                      legitimately fail to accept a probe inside a
 *                      short timeout window, and killing it then would
 *                      abort an in-flight agent task.
 *
 * @param {number} port           TCP port to probe (e.g. 8787).
 * @param {number} [timeoutMs=1500] Probe timeout in milliseconds.
 * @returns {Promise<'alive'|'dead'|'unknown'>}
 */
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
        /* ignore */
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
 * Ask the kiss-web daemon over its Unix-domain control socket whether
 * any agent tasks are currently in flight.
 *
 * Speaks the same newline-delimited JSON protocol as
 * :meth:`RemoteAccessServer._uds_handler`: the daemon responds to
 * ``{"type": "activeTasksQuery"}`` with a single
 * ``{"type": "activeTasksResponse", "count": N, "tabs": [...]}`` line.
 *
 * On any failure (socket missing, connect refused, timeout, malformed
 * response, unexpected message type) the function resolves with
 * ``{ok: false, reason: ...}`` so the caller can decide how
 * conservative to be.  The bug fix policy is:
 *
 *   * ``ok && count > 0``     — defer the restart unconditionally.
 *   * ``ok && count === 0``   — no in-flight tasks; restart is safe.
 *   * ``!ok``                 — UNKNOWN; the caller should fall back to
 *                               the fingerprint / health guard rather
 *                               than treating it as "no tasks".
 *
 * @param {string} sockPath        Path to ~/.kiss/sorcar.sock (or a test override).
 * @param {number} [timeoutMs=1500] Round-trip timeout in milliseconds.
 * @returns {Promise<
 *   {ok: true, count: number, tabs: string[]} |
 *   {ok: false, reason: string}
 * >}
 */
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
        /* ignore */
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
      // ``RemoteAccessServer._uds_handler`` registers every connected
      // client as a broadcast destination, so unrelated event lines
      // (or the OLD-daemon ``Unknown command: activeTasksQuery`` error
      // emitted by a pre-``activeTasksQuery`` build) can land on the
      // wire BEFORE the response we asked for.  Drain every complete
      // line and keep waiting until we see either the response, the
      // specific "old daemon" error (treated as ``count: 0`` so the
      // installer can replace it), an unrecoverable parse failure
      // is logged but ignored, or the outer ``setTimeout`` fires.
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
          // Non-JSON noise on the broadcast stream — skip rather than
          // fail; a malformed line should not be misread as "daemon
          // refusing to answer".
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
          // OLD daemon: cannot have in-flight-task accounting we need
          // to defer to.  Report ``count: 0`` so ``decideRestart`` lets
          // the caller replace it.
          finish({ok: true, count: 0, tabs: []});
          return;
        }
        // Any other broadcast line — keep draining.
      }
    });
    sock.once('error', err => {
      finish({ok: false, reason: 'error:' + (err && err.code)});
    });
    sock.once('end', () => {
      // Server closed before sending a response.
      finish({ok: false, reason: 'eof'});
    });
  });
}

/**
 * Combined decision helper used by ``restartKissWebDaemon``.
 *
 * Returns ``{skip: true, reason}`` when the daemon must NOT be
 * restarted, ``{skip: false, reason}`` when a restart is permitted.
 *
 * Decision table (evaluated top-to-bottom; first match wins)::
 *
 *     activeTasks.ok && activeTasks.count > 0   → skip ("active-tasks")
 *     health === 'alive' && !activeTasks.ok     → skip ("alive-uncertain")
 *     fingerprintMatches && health !== 'dead'   → skip ("healthy-unchanged")
 *     otherwise                                  → restart
 *
 * Two safety properties drive this table:
 *
 *   * ``health === 'unknown'`` is treated as "do not restart" when the
 *     fingerprint matches.  The original code conflated ``'unknown'``
 *     with ``'dead'`` and SIGTERMed the daemon during transient lsof
 *     timeouts.
 *
 *   * An ``'alive'`` daemon whose ``activeTasks`` status could NOT be
 *     confirmed (UDS timeout, socket missing during a transient
 *     startup window, parse failure, etc.) MUST NOT be SIGTERMed.
 *     This regression killed task_history row 3192 mid-flight: the
 *     fingerprint had changed (extension auto-update rewrote every
 *     bundled ``.py`` mtime), the UDS round-trip missed its 1500 ms
 *     deadline under load, and the previous decision table fell
 *     through to ``restart-required`` even though the daemon's own
 *     log line ``active_tasks=[a3b4ec24(task=3191)]`` proved it was
 *     mid-task.  The fingerprint change is applied lazily on the
 *     next activation that can prove the daemon is idle (or has died
 *     on its own).
 *
 * @param {{
 *   fingerprintMatches: boolean,
 *   health: 'alive'|'dead'|'unknown',
 *   activeTasks: {ok: true, count: number, tabs: string[]} |
 *                {ok: false, reason: string},
 * }} state
 * @returns {{skip: boolean, reason: string}}
 */
function decideRestart(state) {
  const {fingerprintMatches, health, activeTasks} = state;
  if (activeTasks && activeTasks.ok && activeTasks.count > 0) {
    return {skip: true, reason: 'active-tasks'};
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

module.exports = {
  probeDaemonHealth,
  daemonHasActiveTasks,
  decideRestart,
};
