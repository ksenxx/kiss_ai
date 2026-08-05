// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const {spawnSync} = require('child_process');

const packageScript = path.join(__dirname, '..', 'scripts', 'package-vsix.js');

function writeJson(filePath, value) {
  fs.writeFileSync(filePath, JSON.stringify(value, null, 2) + '\n');
}

function runTest() {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-package-vsix-'));
  try {
    fs.mkdirSync(path.join(tmp, 'out'));
    fs.writeFileSync(path.join(tmp, 'README.md'), '# Fixture extension\n');
    fs.writeFileSync(path.join(tmp, 'out', 'extension.js'), 'module.exports = {};\n');
    const marker = path.join(tmp, 'implicit-prepublish-ran');
    writeJson(path.join(tmp, 'package.json'), {
      name: 'kiss-package-fixture',
      displayName: 'KISS Package Fixture',
      description: 'Fixture used by package-vsix regression tests',
      version: '1.0.0',
      publisher: 'ksenxx',
      engines: {vscode: '^1.98.0'},
      activationEvents: ['onStartupFinished'],
      main: './out/extension.js',
      scripts: {
        'vscode:prepublish':
          'node -e "require(\'fs\').writeFileSync(process.env.MARKER, \'ran\'); process.exit(42)"',
      },
    });

    const result = spawnSync(
      process.execPath,
      [packageScript, '--out', 'fixture.vsix'],
      {
        cwd: tmp,
        env: {...process.env, MARKER: marker},
        encoding: 'utf8',
        timeout: 120000,
      },
    );
    assert.strictEqual(
      result.status,
      0,
      `package-vsix failed:\nSTDOUT:\n${result.stdout}\nSTDERR:\n${result.stderr}`,
    );
    assert.ok(
      fs.existsSync(path.join(tmp, 'fixture.vsix')),
      'package-vsix must write the requested VSIX file',
    );
    assert.ok(
      !fs.existsSync(marker),
      'package-vsix must not execute vscode:prepublish implicitly',
    );
    assert.match(
      result.stdout,
      /Packaged: .*fixture\.vsix/,
      'package-vsix should print an explicit packaged message',
    );
  } finally {
    fs.rmSync(tmp, {recursive: true, force: true});
  }
}

runTest();
console.log('\npackage-vsix no-prepublish test passed');
