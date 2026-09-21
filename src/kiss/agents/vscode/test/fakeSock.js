// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

/**
 * Where a test's fake kiss-web daemon listens.
 *
 * Node cannot `listen()` on a filesystem path on Windows (`listen
 * EACCES`), so on win32 the fake daemon gets a named pipe instead.  The
 * production clients only `fs.existsSync(path)` and
 * `net.createConnection({path})`, and `sorcarSockPath()` honours
 * `KISS_SORCAR_SOCK` literally, so a pipe exercises the same code.
 * `fs.existsSync` is true for a pipe only while something listens on it;
 * tests that need a socket FILE independent of a listener (a stale file,
 * `rm` while the daemon is up) have no Windows equivalent and skip there
 * (see `SOCK_FILE_OPS`).
 */

const path = require('path');

const WIN32 = process.platform === 'win32';

/**
 * Return a path a fake daemon can listen on: `<dir>/<name>` on POSIX,
 * `\\.\pipe\kiss-test-<pid>-<dir basename>-<name>` on Windows.
 */
function fakeSockPath(dir, name) {
  if (!WIN32) return path.join(dir, name);
  return `\\\\.\\pipe\\kiss-test-${process.pid}-${path.basename(dir)}-${name}`;
}

/** False on Windows: a socket file cannot exist without a listener. */
const SOCK_FILE_OPS = !WIN32;

/** Console line for a case skipped for lack of socket-file semantics. */
const SOCK_FILE_SKIP = 'skipped on win32 (needs a UDS socket file, not a pipe)';

module.exports = {fakeSockPath, SOCK_FILE_OPS, SOCK_FILE_SKIP};
