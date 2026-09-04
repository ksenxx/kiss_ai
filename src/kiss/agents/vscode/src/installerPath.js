// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const fs = require('fs');
const os = require('os');
const path = require('path');

// The curl installer (scripts/install.sh) clones the public repo into
// ~/.kiss/kiss_ai; the Update button must run the install.sh of that clone.
function kissAiRoot() {
  return path.join(os.homedir(), '.kiss', 'kiss_ai');
}

function findInstallScript(root) {
  const base = root || kissAiRoot();
  const candidate = path.join(base, 'install.sh');
  try {
    return fs.statSync(candidate).isFile() ? candidate : null;
  } catch {
    return null;
  }
}

// The public curl bootstrap (the README's install one-liner).  When
// ~/.kiss/kiss_ai/install.sh is missing — the extension was installed from a
// .vsix, or the clone was deleted — the Update button falls back to this
// script, which clones ~/.kiss/kiss_ai and hands over to its install.sh.
// $KISS_UPDATE_BOOTSTRAP_URL overrides it (forks, tests — curl accepts
// file:// URLs).  Twin of web_server._bootstrap_install_url().
const DEFAULT_BOOTSTRAP_INSTALL_URL =
  'https://raw.githubusercontent.com/ksenxx/kiss_ai/main/scripts/install.sh';

function bootstrapInstallUrl() {
  return process.env.KISS_UPDATE_BOOTSTRAP_URL || DEFAULT_BOOTSTRAP_INSTALL_URL;
}

module.exports = {kissAiRoot, findInstallScript, bootstrapInstallUrl};
