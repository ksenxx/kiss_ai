// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// Locate the ``install.sh`` script used to update KISS Sorcar from the
// VS Code "Update" button in Settings.
//
// Why this exists
// ---------------
// The Update button used to look for ``install.sh`` inside the user's
// current workspace / PWD, which is the wrong place: the bootstrap
// script ``scripts/install.sh`` clones the repository to ``~/kiss_ai``
// and ``install.sh`` lives at the root of that clone (see
// ``scripts/install.sh`` — ``git clone … ~/kiss_ai``).  Users opening
// any other folder in VS Code therefore got::
//
//   Cannot update KISS Sorcar: install.sh not found in /some/user/specific/PWD.
//
// even though the script was sitting at ``~/kiss_ai/install.sh`` the
// whole time.  Centralising the lookup here keeps SorcarSidebarView
// free of installer-layout details and gives us a node-only,
// vscode-free helper we can drive from an integration test.
//
// Authored in plain JS (rather than TS) for the same reason as
// ``reloadGuard.js`` and ``daemonHealth.js``: the ``test/`` harness
// runs under bare ``node`` without a TypeScript compile step.

'use strict';

const fs = require('fs');
const os = require('os');
const path = require('path');

/**
 * Return the canonical KISS Sorcar source-checkout root: ``~/kiss_ai``.
 *
 * ``scripts/install.sh`` (the curl-piped bootstrapper documented on the
 * repo) clones the GitHub repo to this fixed location, and ``install.sh``
 * — which the Update button re-runs — lives at its root.  Resolved via
 * ``os.homedir()`` so it works on macOS, Linux, and Windows shells.
 */
function kissAiRoot() {
  return path.join(os.homedir(), 'kiss_ai');
}

/**
 * Return the absolute path of ``install.sh`` inside *root* (defaulting
 * to :func:`kissAiRoot`) if it is a regular file, otherwise ``null``.
 *
 * A regular-file check (not a bare existence check) keeps this in
 * lockstep with the Python twin ``web_server._find_install_script``,
 * which uses ``Path.is_file()`` — a directory named ``install.sh``
 * (e.g. from a botched checkout) must not be handed to ``bash``.
 *
 * The *root* override exists for integration tests; production callers
 * pass no argument so the real ``~/kiss_ai`` is probed.
 */
function findInstallScript(root) {
  const base = root || kissAiRoot();
  const candidate = path.join(base, 'install.sh');
  try {
    return fs.statSync(candidate).isFile() ? candidate : null;
  } catch {
    return null;
  }
}

module.exports = {kissAiRoot, findInstallScript};
