#!/usr/bin/env node
// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const fs = require('fs');
const path = require('path');
const vscePackage = require('@vscode/vsce/out/package');
const {applyHashedIcons} = require('./hash-icons');

function parseArgs(argv) {
  const options = {
    cwd: process.cwd(),
    packagePath: undefined,
    dependencies: false,
    allowMissingRepository: false,
    allowPackageAllSecrets: false,
    skipLicense: false,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '-o' || arg === '--out') {
      i += 1;
      if (i >= argv.length) throw new Error(`${arg} requires a path`);
      options.packagePath = argv[i];
    } else if (arg.startsWith('--out=')) {
      options.packagePath = arg.slice('--out='.length);
    } else if (arg === '--no-dependencies') {
      options.dependencies = false;
    } else if (arg === '--dependencies') {
      options.dependencies = true;
    } else if (arg === '--allow-missing-repository') {
      options.allowMissingRepository = true;
    } else if (arg === '--allow-package-all-secrets') {
      options.allowPackageAllSecrets = true;
    } else if (arg === '--skip-license') {
      options.skipLicense = true;
    } else if (arg === '--help' || arg === '-h') {
      printHelp();
      process.exit(0);
    } else {
      throw new Error(`Unsupported package-vsix option: ${arg}`);
    }
  }
  return options;
}

function printHelp() {
  console.log(`Usage: node scripts/package-vsix.js [options]\n\nOptions:\n  -o, --out <path>              Output VSIX path\n  --no-dependencies             Do not include node dependencies\n  --dependencies                Include node dependencies\n  --allow-missing-repository    Match vsce package flag\n  --allow-package-all-secrets   Match vsce package flag\n  --skip-license                Match vsce package flag`);
}

/**
 * Run `vsce pack` with the manifest's icons renamed to content-hashed copies
 * (see scripts/hash-icons.js) and put the original package.json back
 * afterwards, also when packaging fails or the process is interrupted.
 */
async function packWithHashedIcons(options) {
  const restoreManifest = applyHashedIcons(options.cwd);
  function restoreAndExit() {
    restoreManifest();
    process.exit(1);
  }
  for (const signal of ['SIGINT', 'SIGTERM', 'SIGHUP'])
    process.on(signal, restoreAndExit);
  try {
    return await vscePackage.pack(options);
  } finally {
    restoreManifest();
    for (const signal of ['SIGINT', 'SIGTERM', 'SIGHUP'])
      process.off(signal, restoreAndExit);
  }
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const result = await packWithHashedIcons(options);
  const packagePath = path.resolve(options.cwd, result.packagePath);
  const stats = fs.statSync(packagePath);
  console.log(
    `Packaged: ${path.relative(options.cwd, packagePath)} ` +
      `(${result.files.length} files, ${(stats.size / (1024 * 1024)).toFixed(2)} MB)`,
  );
}

main().catch(err => {
  console.error(err && err.stack ? err.stack : String(err));
  process.exit(1);
});
