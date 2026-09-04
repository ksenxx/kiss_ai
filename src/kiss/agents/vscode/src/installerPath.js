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

module.exports = {kissAiRoot, findInstallScript};
