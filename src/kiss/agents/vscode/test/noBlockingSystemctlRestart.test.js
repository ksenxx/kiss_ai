// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

/* global require, __dirname, console */

'use strict';

// Static regression guard: no BLOCKING systemctl restart may ever be
// reintroduced in the extension source.
//
// The bug this guards against (fixed in daemonRestartNoBlock.test.js's
// commit): `execSync('systemctl --user restart kiss-web', {timeout})`
// blocks until the old daemon finishes shutting down.  A slow shutdown
// (>10s tunnel cleanup; the daemon's SIGTERM failsafe allows 30s) made
// execSync throw ETIMEDOUT, which was misread as "systemd failed" and
// triggered a direct-spawn fallback WHILE systemd's restart job was
// still in flight — two daemons then raced for port 8787 and systemd
// crash-looped, showing "KISS Sorcar Server is restarting" repeatedly.
//
// The safe pattern is `systemctl ... restart --no-block ...`, which
// queues the job and returns immediately.  This suite scans every
// production TypeScript/JavaScript file in the extension for
// synchronous child_process calls (execSync / execFileSync /
// spawnSync) whose arguments invoke a `systemctl ... restart` without
// `--no-block`, and fails with exact file:line locations if any are
// found.  A built-in self-check feeds known-bad snippets through the
// same scanner so a broken scanner can never silently pass.

const assert = require('assert');
const fs = require('fs');
const path = require('path');

const EXT_ROOT = path.join(__dirname, '..');
// Production code only.  test/ is excluded because tests legitimately
// exercise fake systemctl binaries; out/ is compiled from src/.
const SCAN_DIRS = ['src', 'media'];
const SCAN_EXTS = new Set(['.ts', '.tsx', '.js', '.mjs', '.cjs']);
const SYNC_CALL_RE = /\b(execSync|execFileSync|spawnSync)\s*\(/g;

function listSourceFiles(dir) {
  if (!fs.existsSync(dir)) return [];
  const out = [];
  for (const entry of fs.readdirSync(dir, {withFileTypes: true})) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      if (entry.name === 'node_modules') continue;
      out.push(...listSourceFiles(full));
    } else if (SCAN_EXTS.has(path.extname(entry.name))) {
      out.push(full);
    }
  }
  return out;
}

// Extracts the full argument text of a call starting at the opening
// paren, honoring strings and template literals so parens inside
// quoted commands cannot unbalance the scan.
function extractCallArgs(text, openParenIdx) {
  let depth = 0;
  let quote = null; // ', ", or ` when inside a string literal
  for (let i = openParenIdx; i < text.length; i++) {
    const ch = text[i];
    if (quote) {
      if (ch === '\\')
        i++; // skip escaped char
      else if (ch === quote) quote = null;
      continue;
    }
    if (ch === "'" || ch === '"' || ch === '`') quote = ch;
    else if (ch === '(') depth++;
    else if (ch === ')') {
      depth--;
      if (depth === 0) return text.slice(openParenIdx + 1, i);
    }
  }
  return text.slice(openParenIdx + 1); // unbalanced — scan what exists
}

// A call is a blocking systemctl restart when its argument text names
// systemctl AND the restart verb but never opts into --no-block.  This
// covers execSync('systemctl ... restart ...'), execFileSync(
// 'systemctl', ['restart', ...]) and spawnSync('/bin/sh', ['-c',
// 'systemctl restart ...']) alike.
function isBlockingSystemctlRestart(argText) {
  return (
    /systemctl/.test(argText) &&
    /\brestart\b/.test(argText) &&
    !/--no-block/.test(argText)
  );
}

// Returns [{line, fn, snippet}] for every offending call in `text`.
function findViolations(text) {
  const violations = [];
  SYNC_CALL_RE.lastIndex = 0;
  let m;
  while ((m = SYNC_CALL_RE.exec(text)) !== null) {
    const openParenIdx = m.index + m[0].length - 1;
    const argText = extractCallArgs(text, openParenIdx);
    if (isBlockingSystemctlRestart(argText)) {
      violations.push({
        line: text.slice(0, m.index).split('\n').length,
        fn: m[1],
        snippet: (m[1] + '(' + argText + ')')
          .replace(/\s+/g, ' ')
          .slice(0, 160),
      });
    }
  }
  return violations;
}

// --- Self-check: the scanner must flag known-bad and pass known-good ---

const badSamples = [
  // The exact shape of the original bug.
  "execSync('systemctl --user restart kiss-web', {stdio: 'ignore', timeout: 10000});",
  // Multi-line formatting.
  "execSync(\n  'systemctl --user restart kiss-web',\n  {timeout: 10_000},\n);",
  // execFileSync with an argv array.
  "execFileSync('systemctl', ['--user', 'restart', 'kiss-web'], {timeout: 5000});",
  // Shell wrapper around the same blocking command.
  "spawnSync('/bin/sh', ['-c', 'systemctl --user restart kiss-web'], {timeout: 10000});",
  // Property access on the child_process module.
  'cp.execSync(`systemctl --user restart ${unit}`, {timeout: 10000});',
];
for (const sample of badSamples) {
  assert.strictEqual(
    findViolations(sample).length,
    1,
    'scanner self-check failed to flag known-bad snippet: ' + sample,
  );
}

const goodSamples = [
  // The fixed, queued restart.
  "execSync('systemctl --user restart --no-block kiss-web', {stdio: 'ignore', timeout: 10000});",
  // Non-restart systemctl calls are fine.
  "execSync('systemctl --user daemon-reload', {timeout: 10000});",
  // restart of something that is not systemctl is fine.
  "execSync('docker restart kiss-web', {timeout: 10000});",
  // Async exec does not block the caller thread the same way.
  "exec('systemctl --user restart kiss-web', cb);",
  // Parens inside the quoted command must not unbalance extraction.
  "execSync('systemctl --user restart --no-block kiss-web # (queued)', {timeout: 1000}); execSync('echo (ok)');",
];
for (const sample of goodSamples) {
  assert.deepStrictEqual(
    findViolations(sample),
    [],
    'scanner self-check falsely flagged known-good snippet: ' + sample,
  );
}

// --- Scan the real extension source tree ---

const files = SCAN_DIRS.flatMap(d => listSourceFiles(path.join(EXT_ROOT, d)));
assert.ok(
  files.length > 0,
  'guard scanned zero source files — SCAN_DIRS is wrong, fix the test',
);
// DependencyInstaller.ts is where the original bug lived; if it ever
// stops being scanned the guard has silently gone blind.
assert.ok(
  files.some(f => f.endsWith(path.join('src', 'DependencyInstaller.ts'))),
  'guard no longer scans DependencyInstaller.ts — SCAN_DIRS is wrong',
);

const offenders = [];
for (const file of files) {
  const text = fs.readFileSync(file, 'utf-8');
  for (const v of findViolations(text)) {
    offenders.push(`${path.relative(EXT_ROOT, file)}:${v.line}  ${v.snippet}`);
  }
}

assert.deepStrictEqual(
  offenders,
  [],
  'BLOCKING systemctl restart detected — this reintroduces the ' +
    'restart/race bug that caused the "KISS Sorcar Server is ' +
    'restarting" churn.  Use `systemctl ... restart --no-block ...` ' +
    '(and keep verifyDaemonStartup() as the success check) instead of ' +
    'a blocking restart under an execSync timeout.\nOffending calls:\n' +
    offenders.join('\n'),
);

console.log(
  `noBlockingSystemctlRestart: scanned ${files.length} source files, ` +
    'no blocking systemctl restart calls found; scanner self-checks passed',
);
