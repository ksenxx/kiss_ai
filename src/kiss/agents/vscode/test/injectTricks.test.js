// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end test for ``getTricks`` — the helper in ``SorcarTab``
// that builds the "Inject instruction" panel trick list consumed by
// ``renderTricks`` in ``media/main.js`` via ``window.__TRICKS__``.
//
// Contract locked in here:
//
//   * The trick list is the concatenation of two Markdown files,
//     **in this order**:
//       1. ``~/.kiss/MY_INJECTION.md`` — user-curated tricks.
//          Auto-created on first read with the seed content
//
//              ## Trick
//
//              Write end-to-end 100% coverage tests for the feature first.  Then implement the feature.
//
//          so the file is always present and editable.
//       2. ``<kissRoot>/src/kiss/INJECTIONS.md`` — bundled tricks
//          shipped with the extension.  Read **directly from the
//          package**; never copied into ``~/.kiss/``.
//   * Each ``## Trick`` section becomes one entry, with the body
//     trimmed and mdformat backslash escapes reverted.
//   * Non-``Trick`` headings and empty bodies are skipped.
//   * ``KISS_INJECTIONS_PATH`` overrides the bundled file location for
//     testability (parallel to the Python ``tricks.py`` env override).
//   * A missing or unreadable file in either location contributes
//     zero entries (the panel still renders).
//
// Runs against the compiled extension under ``out/``:
//
//     tsc -p . && node test/injectTricks.test.js

'use strict';

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const Module = require('module');

// SorcarTab.js requires 'vscode' at load time.  Outside the extension
// host the module is unavailable; redirect that specifier to a tiny
// stub on disk.  ``getTricks`` itself never touches any vscode API,
// so an empty object suffices.
// ``_vscode-stub.js`` is a git-tracked fixture shared by tests running
// in parallel; it already re-exports ``global.__kissVscodeStub || {}`` —
// never rewrite or delete it here (writeFileSync truncates first, racing
// a concurrent ``require('vscode')`` in sibling test processes).
global.__kissVscodeStub = {};
const origResolve = Module._resolveFilename;
Module._resolveFilename = function (request, parent, ...rest) {
  if (request === 'vscode') return require.resolve('./_vscode-stub.js');
  return origResolve.call(this, request, parent, ...rest);
};

const sourcePath = path.join(__dirname, '..', 'out', 'SorcarTab.js');
assert.ok(
  fs.existsSync(sourcePath),
  `compiled extension missing: ${sourcePath} — run \`tsc -p .\` first`,
);
delete require.cache[require.resolve(sourcePath)];
const {getTricks, DEFAULT_MY_INJECTION, MY_INJECTION_DEFAULT_BODY} =
  require(sourcePath);

assert.strictEqual(
  typeof getTricks,
  'function',
  'getTricks must be exported from the compiled SorcarTab',
);
assert.strictEqual(
  typeof DEFAULT_MY_INJECTION,
  'string',
  'DEFAULT_MY_INJECTION must be exported',
);
assert.strictEqual(
  typeof MY_INJECTION_DEFAULT_BODY,
  'string',
  'MY_INJECTION_DEFAULT_BODY must be exported',
);

let passed = 0;
const failures = [];

function test(name, fn) {
  try {
    fn();
    passed += 1;
    console.log(`  ok - ${name}`);
  } catch (err) {
    failures.push({name, err});
    console.log(`  FAIL - ${name}: ${err && err.message}`);
  }
}

function mkTmp(prefix) {
  return fs.mkdtempSync(path.join(os.tmpdir(), prefix));
}

/**
 * Run ``fn`` inside a sandbox with KISS_HOME pointing at a fresh temp
 * dir and KISS_INJECTIONS_PATH pointing at a writable temp file.  The
 * caller can write any bundled tricks content via the returned
 * ``setBundled`` helper.
 */
function withSandbox(fn) {
  const kissHome = mkTmp('kiss-tricks-home-');
  const bundledFile = path.join(mkTmp('kiss-tricks-bundled-'), 'INJECTIONS.md');
  const prevHome = process.env.KISS_HOME;
  const prevInj = process.env.KISS_INJECTIONS_PATH;
  process.env.KISS_HOME = kissHome;
  process.env.KISS_INJECTIONS_PATH = bundledFile;
  const setBundled = (content) => fs.writeFileSync(bundledFile, content);
  try {
    fn({kissHome, bundledFile, setBundled});
  } finally {
    if (prevHome === undefined) delete process.env.KISS_HOME;
    else process.env.KISS_HOME = prevHome;
    if (prevInj === undefined) delete process.env.KISS_INJECTIONS_PATH;
    else process.env.KISS_INJECTIONS_PATH = prevInj;
    fs.rmSync(kissHome, {recursive: true, force: true});
    fs.rmSync(path.dirname(bundledFile), {recursive: true, force: true});
  }
}

test('exported default body matches the task spec verbatim', () => {
  assert.strictEqual(
    MY_INJECTION_DEFAULT_BODY,
    'Write end-to-end 100% coverage tests for the feature first.' +
      '  Then implement the feature.',
  );
  assert.strictEqual(
    DEFAULT_MY_INJECTION,
    '## Trick\n\n' + MY_INJECTION_DEFAULT_BODY + '\n',
  );
});

test('auto-creates MY_INJECTION.md with default trick when missing', () => {
  withSandbox(({kissHome, setBundled}) => {
    setBundled('## Trick\n\nbundled trick one\n');
    const myInj = path.join(kissHome, 'MY_INJECTION.md');
    assert.ok(!fs.existsSync(myInj), 'user copy must start absent');
    const tricks = getTricks();
    assert.ok(
      fs.existsSync(myInj),
      'getTricks must auto-create ~/.kiss/MY_INJECTION.md',
    );
    assert.strictEqual(
      fs.readFileSync(myInj, 'utf-8'),
      DEFAULT_MY_INJECTION,
      'seed content must be the default ## Trick section',
    );
    assert.deepStrictEqual(tricks, [
      MY_INJECTION_DEFAULT_BODY,
      'bundled trick one',
    ]);
  });
});

test('does NOT copy INJECTIONS.md into ~/.kiss/', () => {
  withSandbox(({kissHome, setBundled}) => {
    setBundled('## Trick\n\nbundled only\n');
    getTricks();
    assert.ok(
      !fs.existsSync(path.join(kissHome, 'INJECTIONS.md')),
      '~/.kiss/INJECTIONS.md must never be seeded',
    );
  });
});

test('MY_INJECTION.md tricks appear before bundled INJECTIONS.md tricks', () => {
  withSandbox(({kissHome, setBundled}) => {
    fs.writeFileSync(
      path.join(kissHome, 'MY_INJECTION.md'),
      '## Trick\n\nmy trick A\n\n## Trick\n\nmy trick B\n',
    );
    setBundled('## Trick\n\nbundled X\n\n## Trick\n\nbundled Y\n');
    assert.deepStrictEqual(getTricks(), [
      'my trick A',
      'my trick B',
      'bundled X',
      'bundled Y',
    ]);
  });
});

test('preserves user edits to MY_INJECTION.md across reads', () => {
  withSandbox(({kissHome, setBundled}) => {
    const myInj = path.join(kissHome, 'MY_INJECTION.md');
    fs.writeFileSync(myInj, '## Trick\n\nuser override\n');
    setBundled('## Trick\n\nfresh bundled\n');
    assert.deepStrictEqual(getTricks(), ['user override', 'fresh bundled']);
    // The user copy must NOT be overwritten by the default seed.
    assert.strictEqual(
      fs.readFileSync(myInj, 'utf-8'),
      '## Trick\n\nuser override\n',
    );
  });
});

test('returns only MY_INJECTION tricks when bundled INJECTIONS.md is missing', () => {
  withSandbox(({kissHome, bundledFile}) => {
    fs.writeFileSync(
      path.join(kissHome, 'MY_INJECTION.md'),
      '## Trick\n\nonly mine\n',
    );
    // Do NOT write the bundled file — keep it missing.
    assert.ok(!fs.existsSync(bundledFile), 'bundled file must start absent');
    assert.deepStrictEqual(getTricks(), ['only mine']);
  });
});

test('empty MY_INJECTION.md contributes nothing; only bundled tricks appear', () => {
  withSandbox(({kissHome, setBundled}) => {
    fs.writeFileSync(path.join(kissHome, 'MY_INJECTION.md'), '');
    setBundled('## Trick\n\nbundled only\n');
    assert.deepStrictEqual(getTricks(), ['bundled only']);
  });
});

test('parses multiple ## Trick sections in source order', () => {
  withSandbox(({kissHome, setBundled}) => {
    fs.writeFileSync(path.join(kissHome, 'MY_INJECTION.md'), '');
    setBundled(
      '## Trick\n\nfirst\n\n## Trick\n\nsecond\n\n## Trick\n\nthird\n',
    );
    assert.deepStrictEqual(getTricks(), ['first', 'second', 'third']);
  });
});

test('skips sections whose heading is not Trick', () => {
  withSandbox(({kissHome, setBundled}) => {
    fs.writeFileSync(
      path.join(kissHome, 'MY_INJECTION.md'),
      '## Task\n\nignored\n\n## Trick\n\nmy real trick\n',
    );
    setBundled(
      '## Notes\n\nignored\n\n## Trick\n\nbundled real\n\n## Other\n\nignored2\n',
    );
    assert.deepStrictEqual(getTricks(), ['my real trick', 'bundled real']);
  });
});

test('skips empty-bodied ## Trick sections', () => {
  withSandbox(({kissHome, setBundled}) => {
    fs.writeFileSync(path.join(kissHome, 'MY_INJECTION.md'), '');
    setBundled('## Trick\n\n   \n\n## Trick\n\nreal\n');
    assert.deepStrictEqual(getTricks(), ['real']);
  });
});

test('unescapes mdformat backslash escapes in trick bodies', () => {
  withSandbox(({kissHome, setBundled}) => {
    fs.writeFileSync(path.join(kissHome, 'MY_INJECTION.md'), '');
    // ``mdformat`` rewrites ``<<x>>`` to ``\<<x>>`` on save.  Tricks
    // are pasted verbatim into the chat; backslashes must be stripped.
    setBundled('## Trick\n\nRun on \\<<your repo>> with \\*literal\\*\n');
    assert.deepStrictEqual(getTricks(), [
      'Run on <<your repo>> with *literal*',
    ]);
  });
});

test('returns only bundled tricks when ~/.kiss/ is unwritable', () => {
  if (process.getuid && process.getuid() === 0) {
    console.log('  ok - SKIPPED (root) - read-only ~/.kiss/ fallback');
    passed += 1;
    return;
  }
  withSandbox(({kissHome, setBundled}) => {
    setBundled('## Trick\n\nbundled survives\n');
    // Recreate kissHome as an unwritable directory so the seed write
    // fails inside ensureUserAssetFromDefault.
    fs.chmodSync(kissHome, 0o500);
    try {
      assert.deepStrictEqual(getTricks(), ['bundled survives']);
      assert.ok(
        !fs.existsSync(path.join(kissHome, 'MY_INJECTION.md')),
        'seed file must not be created on read-only ~/.kiss/',
      );
    } finally {
      fs.chmodSync(kissHome, 0o700);
    }
  });
});

test('parses the shipped src/kiss/INJECTIONS.md without crashing', () => {
  const prevHome = process.env.KISS_HOME;
  const prevInj = process.env.KISS_INJECTIONS_PATH;
  const kissHome = mkTmp('kiss-tricks-real-');
  // Empty MY_INJECTION.md so the shipped tricks are observed alone.
  fs.writeFileSync(path.join(kissHome, 'MY_INJECTION.md'), '');
  process.env.KISS_HOME = kissHome;
  // Outside the VS Code host ``findKissProject`` cannot resolve a
  // workspace folder; pin the bundled file via the env override so
  // the test does not depend on the vscode API stub.
  const shipped = path.join(
    __dirname, '..', '..', '..', '..', '..', 'src', 'kiss', 'INJECTIONS.md',
  );
  process.env.KISS_INJECTIONS_PATH = shipped;
  try {
    assert.ok(
      fs.existsSync(shipped),
      `shipped INJECTIONS.md missing at ${shipped}`,
    );
    const tricks = getTricks();
    assert.ok(Array.isArray(tricks), 'must return an array');
    // The shipped INJECTIONS.md must have at least one section.
    assert.ok(tricks.length > 0, 'shipped INJECTIONS.md must yield ≥1 trick');
    for (const t of tricks) {
      assert.strictEqual(typeof t, 'string');
      assert.ok(t.length > 0, 'each trick body must be non-empty');
      assert.strictEqual(t, t.trim());
    }
  } finally {
    if (prevHome === undefined) delete process.env.KISS_HOME;
    else process.env.KISS_HOME = prevHome;
    if (prevInj === undefined) delete process.env.KISS_INJECTIONS_PATH;
    else process.env.KISS_INJECTIONS_PATH = prevInj;
    fs.rmSync(kissHome, {recursive: true, force: true});
  }
});

console.log(`\n${passed} passed, ${failures.length} failed`);
if (failures.length > 0) {
  for (const f of failures) {
    console.error(`\n${f.name}:\n`, f.err);
  }
  process.exit(1);
}
