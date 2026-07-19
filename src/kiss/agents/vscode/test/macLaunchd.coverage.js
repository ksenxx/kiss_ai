// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// Coverage gate for ``macLaunchd.js``: re-runs
// ``macLaunchdRestart.test.js`` under V8's built-in coverage
// (``NODE_V8_COVERAGE``) and FAILS unless every executable line of the
// module was executed — i.e. the e2e suite exercises 100% of the fix
// for the "kiss-web launch takes a lot of time after an update" bug.
//
// Run with:
//
//     node src/kiss/agents/vscode/test/macLaunchd.coverage.js

'use strict';

const assert = require('assert');
const {spawnSync} = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');

const MODULE_FILE = path.resolve(__dirname, '..', 'src', 'macLaunchd.js');
const TEST_FILE = path.join(__dirname, 'macLaunchdRestart.test.js');

// Paint one script instance's char coverage from its V8 ranges.
// Ranges nest; applying them sorted by (startOffset asc, endOffset
// desc) paints outer ranges first so inner ranges override, matching
// V8 block-coverage semantics.
function paintInstance(functions, length, painted) {
  const ranges = [];
  for (const fn of functions) for (const r of fn.ranges) ranges.push(r);
  ranges.sort(
    (a, b) => a.startOffset - b.startOffset || b.endOffset - a.endOffset,
  );
  const instance = new Uint8Array(length);
  for (const r of ranges) {
    const v = r.count > 0 ? 1 : 0;
    const end = Math.min(r.endOffset, length);
    for (let i = Math.max(0, r.startOffset); i < end; i++) instance[i] = v;
  }
  // Merge: a char covered in ANY instance counts as covered.
  for (let i = 0; i < length; i++) if (instance[i]) painted[i] = 1;
}

function main() {
  if (process.platform !== 'darwin') {
    // The real-launchd half of the suite (which covers the happy
    // paths) only runs on macOS, so the 100% gate is only meaningful
    // there.
    console.log('  skip - macLaunchd coverage gate (not macOS)');
    return;
  }
  const src = fs.readFileSync(MODULE_FILE, 'utf-8');
  const covDir = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-maclaunchd-cov-'));
  try {
    run(src, covDir);
  } finally {
    fs.rmSync(covDir, {recursive: true, force: true});
  }
}

function run(src, covDir) {
  const res = spawnSync(process.execPath, [TEST_FILE], {
    env: {...process.env, NODE_V8_COVERAGE: covDir},
    encoding: 'utf-8',
  });
  assert.strictEqual(
    res.status,
    0,
    `test run failed:\n${res.stdout}\n${res.stderr}`,
  );

  const painted = new Uint8Array(src.length);
  let sawModule = false;
  for (const f of fs.readdirSync(covDir)) {
    if (!f.endsWith('.json')) continue;
    const data = JSON.parse(fs.readFileSync(path.join(covDir, f), 'utf-8'));
    for (const script of data.result || []) {
      const url = script.url || '';
      if (!url.endsWith('macLaunchd.js')) continue;
      sawModule = true;
      paintInstance(script.functions, src.length, painted);
    }
  }
  assert.ok(sawModule, 'coverage data for macLaunchd.js not found');

  const lines = src.split('\n');
  const uncovered = [];
  let offset = 0;
  for (let li = 0; li < lines.length; li++) {
    const line = lines[li];
    const trimmed = line.trim();
    const isCode =
      trimmed.length > 0 &&
      !trimmed.startsWith('//') &&
      !trimmed.startsWith('/*') &&
      !trimmed.startsWith('*');
    if (isCode) {
      let covered = false;
      for (let i = 0; i < line.length; i++) {
        if (line[i] !== ' ' && line[i] !== '\t' && painted[offset + i]) {
          covered = true;
          break;
        }
      }
      if (!covered) uncovered.push(`${li + 1}: ${line}`);
    }
    offset += line.length + 1;
  }

  assert.strictEqual(
    uncovered.length,
    0,
    'macLaunchd.js has uncovered executable lines:\n' + uncovered.join('\n'),
  );
  console.log(
    '  ok - macLaunchdRestart e2e suite covers 100% of macLaunchd.js',
  );
}

main();
