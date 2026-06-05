// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// Reload-readiness guard for the KISS Sorcar extension.
//
// Why this exists
// ---------------
// When ``install.sh`` (or ``scripts/build-extension.sh``) reinstalls the
// extension while VS Code is open, ``code --install-extension --force`` first
// *deletes* the extension directory and then re-extracts it.  During that
// window ``out/extension.js`` is transiently missing (and, briefly, present
// but only partially written).  The extension watches ``out/extension.js`` to
// auto-reload after a reinstall; if it reloads *during* that window the
// webview comes up against a half-installed extension (its ``media`` chat
// resources are missing) and the kiss-web daemon has not been restarted yet —
// the chat view renders blank.
//
// These helpers let the reload logic wait until the reinstall has fully
// settled: the entry file is present and non-empty, its size is stable across
// consecutive polls, and the kiss-web daemon's Unix-domain socket is back so
// the reloaded webview can immediately reconnect.
//
// Implemented as a dependency-free CommonJS module (rather than inline in
// ``extension.ts``) so it can be exercised directly with ``node`` in an
// integration test without spinning up a full VS Code extension host.

'use strict';

const fs = require('fs');

/**
 * Size in bytes of the extension entry file, or ``-1`` when the path is
 * missing or is not a regular file.
 *
 * A regular ``statSync`` throws ``ENOENT`` while the reinstall has the old
 * directory deleted; returning ``-1`` lets callers treat "missing" and
 * "present but empty" uniformly as "not ready".
 *
 * @param {string} extJsPath Absolute path to ``out/extension.js``.
 * @returns {number} File size in bytes, or ``-1`` if missing / not a file.
 */
function extensionFileSize(extJsPath) {
  try {
    const st = fs.statSync(extJsPath);
    if (!st.isFile()) return -1;
    return st.size;
  } catch {
    return -1;
  }
}

/**
 * Return ``true`` when *p* exists on disk (any file type).
 *
 * Used to check for the kiss-web daemon's UDS socket so the reload is held
 * back until the daemon the reloaded webview connects to is listening again.
 *
 * @param {string} p Absolute path.
 * @returns {boolean} ``true`` if the path exists.
 */
function pathExists(p) {
  try {
    fs.accessSync(p);
    return true;
  } catch {
    return false;
  }
}

/**
 * Probe the post-reinstall state, given the previously observed entry-file
 * size, and report which reload preconditions currently hold.
 *
 * The two preconditions are reported separately so the caller can treat them
 * differently:
 *
 *   * ``codeReady`` — ``out/extension.js`` is present, non-empty, and its size
 *     is unchanged from the previous poll.  This is the *hard* precondition:
 *     reloading while the entry file is mid-delete / mid-write brings the
 *     webview up against a half-installed extension (its ``media`` chat
 *     resources are missing) and renders blank.  The caller must never reload
 *     until this is true.
 *
 *   * ``socketUp`` — the kiss-web daemon's Unix-domain socket exists, so a
 *     reloaded webview can reconnect immediately.  This is only a *soft*
 *     preference: ``install.sh`` deliberately kills the daemon and removes the
 *     socket before writing the update marker, and the socket only returns
 *     once a daemon is respawned.  On source installs the respawn happens in
 *     the post-reload ``ensureDependencies()`` (``restartKissWebDaemon``), so
 *     the socket *cannot* come back until after the reload.  Gating the reload
 *     on it would therefore dead-lock until the hard timeout.  The webview's
 *     ``AgentClient`` auto-reconnects when the socket reappears, so reloading
 *     while it is still down only briefly shows a reconnecting view rather
 *     than a permanently blank one.
 *
 * ``ready`` (= ``codeReady && socketUp``) is preserved for callers that want
 * the strict both-conditions gate.  The caller threads the returned ``size``
 * back in as ``prevSize`` on the next poll so stability is measured across
 * consecutive observations.
 *
 * @param {string} extJsPath Absolute path to ``out/extension.js``.
 * @param {string} sockPath Absolute path to ``~/.kiss/sorcar.sock``.
 * @param {number} prevSize Entry-file size observed on the previous poll.
 * @returns {{ready: boolean, codeReady: boolean, socketUp: boolean, size: number}}
 *   Readiness flags and current entry-file size.
 */
function isReloadReady(extJsPath, sockPath, prevSize) {
  const size = extensionFileSize(extJsPath);
  const codeReady = size > 0 && size === prevSize;
  const socketUp = pathExists(sockPath);
  return {ready: codeReady && socketUp, codeReady, socketUp, size};
}

module.exports = {extensionFileSize, pathExists, isReloadReady};
