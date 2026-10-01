// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const fs = require('fs');

function extensionFileSize(extJsPath) {
  try {
    const st = fs.statSync(extJsPath);
    if (!st.isFile()) return -1;
    return st.size;
  } catch {
    return -1;
  }
}

function pathExists(p) {
  try {
    fs.accessSync(p);
    return true;
  } catch {
    return false;
  }
}

function isReloadReady(extJsPath, endpointPath, prevSize) {
  const size = extensionFileSize(extJsPath);
  const codeReady = size > 0 && size === prevSize;
  // The daemon publishes its endpoint file once it listens and removes
  // it on shutdown, so the file's presence is the daemon's presence.
  const daemonUp = pathExists(endpointPath);
  return {ready: codeReady && daemonUp, codeReady, daemonUp, size};
}

module.exports = {extensionFileSize, pathExists, isReloadReady};
