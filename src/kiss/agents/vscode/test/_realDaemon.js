// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// Starts the REAL Sorcar daemon (test/_real_daemon.py, a real
// RemoteAccessServer) for an extension-host end-to-end test.
//
// The daemon publishes its loopback WSS endpoint (URL + token) in
// `env.KISS_HOME/sorcar-local.json`, which must be the same temp home
// the test points the compiled extension host at (kissHomeDir() ->
// sorcarEndpointPath()), so the host's AgentClient connects to this
// daemon and nothing in between is faked.

const {spawn} = require('child_process');
const path = require('path');

const EXT_DIR = path.join(__dirname, '..');
const KISS_PROJECT = path.resolve(EXT_DIR, '..', '..', '..', '..');
const DAEMON_PY = path.join(__dirname, '_real_daemon.py');

/**
 * Spawn the real daemon serving *workDir*; resolves once it listens.
 *
 * @param {string} uv Path of the uv binary (kissPaths findUvPath()).
 * @param {string} workDir The daemon's configured work dir.
 * @param {NodeJS.ProcessEnv} env Environment with HOME / KISS_HOME set to
 *   the test's temp home.
 * @param {number} [startTimeoutMs=120000] How long to wait for READY
 *   before killing the child and rejecting.
 * @returns {Promise<{broadcast(event: object): void, stop(): Promise<void>}>}
 *   `broadcast` has the daemon emit *event* to its clients as a running
 *   task would; `stop` closes stdin and waits for the daemon to exit.
 */
function startRealDaemon(uv, workDir, env, startTimeoutMs = 120000) {
  // `uv run` keeps python as a child, so killing uv alone would leave
  // python holding our stdout pipe.  On POSIX the child leads its own
  // process group (detached) so killTree can SIGKILL uv and python at
  // once.
  const posix = process.platform !== 'win32';
  const child = spawn(uv, ['run', 'python', DAEMON_PY, workDir], {
    cwd: KISS_PROJECT,
    env,
    stdio: ['pipe', 'pipe', 'inherit'],
    detached: posix,
  });
  const exited = new Promise(resolve => child.on('exit', resolve));
  return new Promise((resolve, reject) => {
    let out = '';
    // Kill before rejecting: no caller ever receives this daemon, so
    // nobody else can stop it, and a live child's stdio pipes would
    // keep the test process (and `npm test`) alive forever.
    const timer = setTimeout(() => {
      killTree(child, posix);
      reject(new Error('real daemon did not start in time'));
    }, startTimeoutMs);
    child.stdout.on('data', chunk => {
      out += chunk.toString();
      if (out.includes('READY')) {
        clearTimeout(timer);
        child.stdout.on('data', () => {});
        resolve({
          broadcast(event) {
            child.stdin.write(JSON.stringify(event) + '\n');
          },
          async stop() {
            child.stdin.end();
            const killer = setTimeout(() => killTree(child, posix), 15000);
            await exited;
            clearTimeout(killer);
          },
        });
      }
    });
    child.on('exit', code => {
      clearTimeout(timer);
      reject(new Error(`real daemon exited early with code ${code}`));
    });
  });
}

/**
 * SIGKILL *child* and, on POSIX, every process in its process group.
 *
 * @param {import('child_process').ChildProcess} child The spawned daemon.
 * @param {boolean} posix Whether the child leads its own process group.
 */
function killTree(child, posix) {
  if (posix) {
    try {
      process.kill(-child.pid, 'SIGKILL');
      return;
    } catch {
      // The group is already gone; fall through to the plain kill.
    }
  }
  child.kill('SIGKILL');
}

module.exports = {startRealDaemon};
