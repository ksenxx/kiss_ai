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

function isReloadReady(extJsPath, sockPath, prevSize) {
  const size = extensionFileSize(extJsPath);
  const codeReady = size > 0 && size === prevSize;
  const socketUp = pathExists(sockPath);
  return {ready: codeReady && socketUp, codeReady, socketUp, size};
}

module.exports = {extensionFileSize, pathExists, isReloadReady};
