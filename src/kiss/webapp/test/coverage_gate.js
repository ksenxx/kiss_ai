// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Shared body of every `*.coverage.js` gate in this directory.
//
// A gate runs one functional jsdom suite under NODE_V8_COVERAGE and then
// requires 100% line coverage of the code fenced by
// `// <mark>-coverage:start` / `// <mark>-coverage:end` comments in one or
// more media/*.js files, keyed on the `//# sourceURL` pragma the suite
// appends when it evals each file.  The gates differ only in their suite,
// mark, targets and report label, so the spawn, V8-report painting and
// region accounting live here once.

'use strict';

const assert = require('assert');
const {spawn, spawnSync} = require('child_process');
const fs = require('fs');
const net = require('net');
const os = require('os');
const path = require('path');

const MEDIA = path.join(__dirname, '..', 'media');
// Below run-all.js's 120 s suite limit, so a hung suite is killed and
// reported here (and its coverage directory removed) first.
const CHILD_TIMEOUT_MS = 90 * 1000;
// How long the cleanup of a killed gate (`removeCoverageOnceGone`) waits
// for the suite's process group to be gone.
const CLEANUP_TIMEOUT_MS = 5 * 1000;

// Node 24+ enables the Maglev compiler and background Sparkplug compilation;
// either can hang `process.exit()` (nodejs/node#64274), which is why
// run-all.js passes these flags to every suite.  The child gets them even
// when the gate itself was started without them.
const V8_FLAGS = ['--no-maglev', '--no-concurrent-sparkplug'];

// Windows has no process groups to lead or kill, and a pipe handed to a
// child as fd 3 is not readable through `net.Socket({fd: 3})` there
// (`read ENOTCONN`): the suite runs without the guard below, and the
// timeout kills the suite itself.
const WINDOWS = process.platform === 'win32';

// The `identity` of every file under media/ when this gate's first suite
// started, by path (`runCovered` notes them, `measure` compares).
let mediaAtStart;

/**
 * What changes whenever `file`'s text does: its inode, size and
 * timestamps; undefined when there is no such file.
 */
function identity(file) {
  const s = fs.statSync(file, {throwIfNoEntry: false});
  return s && `${s.ino}:${s.size}:${s.mtimeMs}:${s.ctimeMs}`;
}

/**
 * The `identity` of every file under media/, by path.  A copy of this
 * file run from a scratch directory (the gate's own tests) has no media/
 * beside it and nothing to measure: none then.
 */
function mediaIdentities() {
  const ids = {};
  if (!fs.existsSync(MEDIA)) return ids;
  for (const name of fs.readdirSync(MEDIA, {recursive: true})) {
    const file = path.join(MEDIA, name);
    ids[file] = identity(file);
  }
  return ids;
}

/**
 * SIGKILL process group `pgid`; a group already gone is fine.  Anything
 * but a positive safe integer is refused: `process.kill(-0)` (0, '',
 * null, false, an empty pid file read as a number) would signal the
 * caller's own group, a negative number the process it negates to.
 */
function killGroup(pgid) {
  if (!Number.isSafeInteger(pgid) || pgid <= 0) return;
  try {
    process.kill(-pgid, 'SIGKILL');
  } catch {
    // ESRCH: every process of the group has exited.
  }
}

/**
 * Run `testFile` (a path relative to this directory) under NODE_V8_COVERAGE
 * and return the parsed V8 coverage reports it produced.
 *
 * The suite runs under a guard (`guardSuite`, this file run as a script)
 * that leads a new process group, so the suite and everything it starts
 * are killed together, whether or not they hold this process's output (a
 * worker started with stdio 'ignore' does not).  The group is SIGKILLed
 * as soon as the guard returns, when the suite exits or after
 * CHILD_TIMEOUT_MS (the guard is SIGKILLed first), so nothing the suite
 * started outlives the gate, run by run-all.js or on its own.  Should the
 * gate itself be killed first, the guard kills the group and has the
 * coverage directory removed (`retireGroup`).
 *
 * The child runs with V8_FLAGS only.  This process's own flags are not
 * passed on: a `node -e` / `-p` gate would make the child run that code
 * again (and recurse) instead of the suite, and `--watch` or `--inspect`
 * do not belong to it either; a preload in NODE_OPTIONS still reaches it
 * through the environment.  The suite writes straight to this process's
 * stdout and stderr, so what it printed survives a kill of the whole
 * process group.  When it fails or times out the coverage directory is
 * removed and the process exits with the suite's status.  The directory
 * is removed as well when a report cannot be read (one cut short does
 * not parse), which throws.
 *
 * Before the gate's first suite the identity of every media file is
 * noted, for `measure` to refuse one that changed since.
 */
function runCovered(testFile, mark) {
  mediaAtStart = mediaAtStart || mediaIdentities();
  const covDir = fs.mkdtempSync(path.join(os.tmpdir(), `${mark}-cov-`));
  const suite = path.join(__dirname, testFile);
  const res = spawnSync(
    process.execPath,
    WINDOWS ? [...V8_FLAGS, suite] : [...V8_FLAGS, __filename, suite],
    {
      env: Object.assign({}, process.env, {NODE_V8_COVERAGE: covDir}),
      // fd 3: the guard's lifeline to this process (see guardSuite).
      stdio: WINDOWS
        ? ['ignore', 'inherit', 'inherit']
        : ['ignore', 'inherit', 'inherit', 'pipe'],
      detached: !WINDOWS,
      timeout: CHILD_TIMEOUT_MS,
      killSignal: 'SIGKILL',
    },
  );
  // pid 0: the guard never started, and -0 would name our own group.
  if (res.pid && !WINDOWS) killGroup(res.pid);
  if (res.status !== 0) {
    fs.rmSync(covDir, {recursive: true, force: true});
    let why = 'the functional test itself FAILED';
    if (res.error && res.error.code === 'ETIMEDOUT') {
      why =
        'the functional test did not exit within ' +
        `${CHILD_TIMEOUT_MS / 1000}s, killed`;
    } else if (res.error) {
      why = `could not run the functional test: ${res.error}`;
    }
    console.error(`coverage gate: ${why}`);
    process.exit(res.status || 1);
  }
  try {
    return fs
      .readdirSync(covDir)
      .map(f => JSON.parse(fs.readFileSync(path.join(covDir, f), 'utf-8')));
  } finally {
    fs.rmSync(covDir, {recursive: true, force: true});
  }
}

/**
 * Paint one script's V8 block-coverage ranges onto a byte array of `length`
 * (1 = executed, 0 = not).  Outer ranges are painted first so that an inner
 * zero-count range overrides its executed parent.
 */
function paintInstance(functions, length) {
  const painted = new Uint8Array(length);
  const ranges = [];
  for (const fn of functions) for (const r of fn.ranges) ranges.push(r);
  ranges.sort(
    (a, b) => a.startOffset - b.startOffset || b.endOffset - a.endOffset,
  );
  for (const r of ranges) {
    const v = r.count > 0 ? 1 : 0;
    const end = Math.min(r.endOffset, length);
    for (let i = Math.max(0, r.startOffset); i < end; i++) painted[i] = v;
  }
  return painted;
}

/**
 * Union the coverage of every eval instance whose url ends with `sourceUrl`
 * across `reports`; returns the covered byte array and the instance count.
 */
function collectCoverage(reports, sourceUrl, length) {
  const covered = new Uint8Array(length);
  let instances = 0;
  for (const report of reports) {
    for (const script of report.result || []) {
      if (!script.url || !script.url.endsWith(sourceUrl)) continue;
      instances++;
      const painted = paintInstance(script.functions, length);
      for (let i = 0; i < length; i++) if (painted[i]) covered[i] = 1;
    }
  }
  return {covered, instances};
}

/**
 * The interior of every `mark` region of `lines` as zero-based half-open
 * [start, end) line intervals: `start` is the line after the opening fence,
 * `end` the index of the closing fence.  `target.regions` pins the exact
 * number of regions (so deleting a fence cannot silently shrink the gate),
 * `target.minRegions` (default 1) only a lower bound.
 */
function findRegions(lines, mark, name, target) {
  const startMark = `// ${mark}-coverage:start`;
  const endMark = `// ${mark}-coverage:end`;
  const regions = [];
  let start = -1;
  for (let i = 0; i < lines.length; i++) {
    const t = lines[i].trim();
    if (t === startMark) {
      assert.strictEqual(start, -1, `nested ${startMark} in ${name}`);
      start = i + 1;
    } else if (t === endMark) {
      assert.ok(start >= 0, `${endMark} without start in ${name}`);
      regions.push([start, i]);
      start = -1;
    }
  }
  assert.strictEqual(start, -1, `unclosed ${mark}-coverage region in ${name}`);
  if (target.regions !== undefined) {
    assert.strictEqual(
      regions.length,
      target.regions,
      `${name} must contain exactly ${target.regions} ${mark} regions`,
    );
  } else {
    assert.ok(
      regions.length >= (target.minRegions || 1),
      `expected at least ${target.minRegions || 1} ${mark} region(s) in ${name}`,
    );
  }
  return regions;
}

/** Blank lines and comment lines carry no code and are never counted. */
function isCodeLine(line) {
  const t = line.trim();
  return !(t === '' || t.startsWith('//') || t.startsWith('*') || t === '/**');
}

/**
 * Line coverage of every `mark` region of one target file, or of the
 * whole file when `target.wholeFile` is set.  A line counts as hit when
 * any of its non-whitespace bytes executed, or, with `everyByte`, only
 * when all of them did (V8 gives an untaken `return;` or `||` operand a
 * zero count even when it shares a line with run code).  Returns the
 * counts, the missed lines, and the file's text with its covered-byte
 * map for gates that also inspect branches.
 *
 * `reports` hold byte offsets into the text the suites eval'd, so they
 * fit the file only while it is the one the suites read: a file that
 * changed since `runCovered` started the first of them is refused by
 * name (exit 1) instead of measured, which would report lines the
 * suites ran as uncovered.
 */
function measure(target, mark, reports, everyByte) {
  const name = target.file;
  const file = path.join(MEDIA, name);
  const src = fs.readFileSync(file, 'utf-8');
  if (identity(file) !== mediaAtStart[file]) {
    console.error(
      `coverage gate: media/${name} changed while its suite ran, so the ` +
        'coverage cannot be mapped onto it; run the gate again',
    );
    process.exit(1);
  }
  const lines = src.split('\n');
  const regions = target.wholeFile
    ? [[0, lines.length]]
    : findRegions(lines, mark, name, target);
  const {covered, instances} = collectCoverage(
    reports,
    target.sourceUrl,
    src.length,
  );
  assert.ok(
    instances > 0,
    `no ${target.sourceUrl} coverage entries found -- did the test stop ` +
      'eval-ing it with the sourceURL pragma?',
  );
  const starts = new Array(lines.length);
  let offset = 0;
  for (let n = 0; n < lines.length; n++) {
    starts[n] = offset;
    offset += lines[n].length + 1;
  }
  let total = 0;
  let hit = 0;
  const missed = [];
  for (const [a, b] of regions) {
    for (let n = a; n < b; n++) {
      const line = lines[n];
      if (!isCodeLine(line)) continue;
      total++;
      let hitBytes = 0;
      let codeBytes = 0;
      for (let i = 0; i < line.length; i++) {
        if (/\s/.test(line[i])) continue;
        codeBytes++;
        if (covered[starts[n] + i]) hitBytes++;
      }
      if (everyByte ? hitBytes === codeBytes : hitBytes > 0) hit++;
      else missed.push({n: n + 1, line});
    }
  }
  assert.ok(total > 0, `${name} ${mark} regions must not be empty`);
  return {
    name,
    total,
    hit,
    missed,
    instances,
    regions: regions.length,
    src,
    covered,
  };
}

/**
 * Run the gate: spawn `opts.testFile` under coverage and require 100% line
 * coverage of every `opts.mark` region in each of `opts.targets`
 * (`{file, sourceUrl, regions | minRegions}`; `file` is relative to media/).
 * `opts.label` names the feature in the report; `opts.everyByte` demands
 * that every non-whitespace byte of each line ran, not just one.  Exits 1
 * on any miss.
 */
function gateRegions(opts) {
  const reports = runCovered(opts.testFile, opts.mark);
  const results = opts.targets.map(t =>
    measure(t, opts.mark, reports, opts.everyByte),
  );
  let failed = false;
  console.log('');
  for (const r of results) {
    const pct = ((100 * r.hit) / r.total).toFixed(1);
    console.log(
      `${opts.label} line coverage ${r.name}: ${r.hit}/${r.total} ` +
        `(${pct}%) over ${r.regions} regions, ${r.instances} eval instances`,
    );
    if (!r.missed.length) continue;
    failed = true;
    console.error(`\nUNCOVERED ${opts.label.toUpperCase()} LINES in ${r.name}:`);
    for (const m of r.missed) {
      console.error(`  ${String(m.n).padStart(5)}: ${m.line}`);
    }
  }
  if (failed) {
    console.error('\ncoverage gate FAILED: 100% line coverage required');
    process.exit(1);
  }
  console.log(
    `coverage gate passed: 100% line coverage of the ${opts.label} code.`,
  );
}

/**
 * The guard `runCovered` starts as `node coverage_gate.js <suite>`, the
 * leader of a new process group: run `suite` in that group and exit with
 * its status.  Fd 3 is a pipe from the gate, which never writes to it, so
 * its end means the gate is gone before it could kill the group (killed
 * with its own process group, as run-all.js stops a gate, or by a
 * terminal's Ctrl-C, which reaches only the gate's group); the guard then
 * retires the group (`retireGroup`).
 */
function guardSuite(suite) {
  const child = spawn(process.execPath, [...V8_FLAGS, suite], {
    stdio: ['ignore', 'inherit', 'inherit'],
  });
  child.on('exit', code => process.exit(code === null ? 1 : code));
  const gate = new net.Socket({fd: 3, readable: true, writable: false});
  gate.on('close', retireGroup);
  gate.resume();
}

/**
 * The guard, once its gate is gone: SIGKILL the group, itself, the suite
 * and what it started included.  The gate would have removed the coverage
 * directory and the guard cannot, dying with the group, so it first
 * starts a cleanup (`removeCoverageOnceGone`, this file run as a script)
 * outside the group: in a session of its own, holding none of the gate's
 * output, and without NODE_V8_COVERAGE, or it would write a report of its
 * own into the directory it removes (the variable is named with no value:
 * `spawn` copies it into an environment that does not name it).  `spawn`
 * returns once the cleanup runs there, and the group is killed whether or
 * not it could be started.
 */
function retireGroup() {
  const covDir = process.env.NODE_V8_COVERAGE;
  try {
    spawn(
      process.execPath,
      [...V8_FLAGS, __filename, '--cleanup', String(process.pid), covDir],
      {
        env: Object.assign({}, process.env, {NODE_V8_COVERAGE: undefined}),
        stdio: 'ignore',
        detached: true,
      },
    );
  } finally {
    killGroup(process.pid);
  }
}

/**
 * The cleanup `retireGroup` starts as `node coverage_gate.js --cleanup
 * <pgid> <covDir>`: remove `covDir` once no process of group `pgid` is
 * left to write a report into it (the guard writes its own as it kills
 * the group: node flushes coverage before a process signals itself), or
 * after CLEANUP_TIMEOUT_MS (a zombie nobody reaps stays in its group).
 * It kills nothing: the guard has, and a group whose processes are all
 * gone may be another's by now.
 */
function removeCoverageOnceGone(pgid, covDir) {
  const pause = new Int32Array(new SharedArrayBuffer(4));
  const deadline = Date.now() + CLEANUP_TIMEOUT_MS;
  while (Date.now() < deadline) {
    try {
      process.kill(-pgid, 0);
    } catch {
      break; // ESRCH: every process of the group has exited.
    }
    Atomics.wait(pause, 0, 0, 20);
  }
  fs.rmSync(covDir, {recursive: true, force: true});
}

module.exports = {runCovered, measure, gateRegions, killGroup};

if (require.main === module) {
  if (process.argv[2] === '--cleanup') {
    removeCoverageOnceGone(Number(process.argv[3]), process.argv[4]);
    // Exit rather than wait for the event loop to empty: a NODE_OPTIONS
    // preload may keep a timer on it, and nothing is left to kill this
    // process.
    process.exit(0);
  } else {
    guardSuite(process.argv[2]);
  }
}
