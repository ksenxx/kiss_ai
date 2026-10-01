// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// Starts the REAL Sorcar daemon (test/_real_daemon.py, a real
// RemoteAccessServer) for an extension-host end-to-end test.
//
// The daemon listens on the Unix socket under `env.KISS_HOME`, which
// must be the same temp home the test points the compiled extension
// host at (kissHomeDir() -> sorcarSockPath()), so the host's AgentClient
// connects to this daemon and nothing in between is faked.

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
 * @returns {Promise<{broadcast(event: object): void, stop(): Promise<void>}>}
 *   `broadcast` has the daemon emit *event* to its clients as a running
 *   task would; `stop` closes stdin and waits for the daemon to exit.
 */
function startRealDaemon(uv, workDir, env) {
  const child = spawn(uv, ['run', 'python', DAEMON_PY, workDir], {
    cwd: KISS_PROJECT,
    env,
    stdio: ['pipe', 'pipe', 'inherit'],
  });
  const exited = new Promise(resolve => child.on('exit', resolve));
  return new Promise((resolve, reject) => {
    let out = '';
    const timer = setTimeout(
      () => reject(new Error('real daemon did not start in time')),
      120000,
    );
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
            const killer = setTimeout(() => child.kill('SIGKILL'), 15000);
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

module.exports = {startRealDaemon};
