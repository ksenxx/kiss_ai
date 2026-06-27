// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end test for ``readSampleTasks`` — the helper in
// ``SorcarTab`` that builds the welcome-screen chip list consumed by
// ``renderWelcomeSuggestions`` in ``media/main.js``.
//
// Contract locked in here:
//
//   * The chip list is the concatenation of two Markdown files,
//     **in this order**:
//       1. ``~/.kiss/MY_TASK_TEMPLATES.md`` — user-curated tasks,
//          source of truth for personal welcome chips.  Auto-created
//          on first read with the seed content ``## Task\n\nHi!\n``
//          so the file is always present and editable.
//       2. ``<extensionRoot>/kiss_project/src/kiss/SAMPLE_TASKS.md``
//          (or, in dev checkouts, ``<extensionRoot>/../../SAMPLE_TASKS.md``)
//          — bundled sample tasks shipped with the extension.
//     ``SAMPLE_TASKS.md`` is **never** copied into ``~/.kiss/``; the
//     package copy is read directly.  This makes the bundled chips
//     update automatically with every extension upgrade.
//   * Each ``## Task`` section becomes one ``{text}`` entry, with the
//     body trimmed and mdformat backslash escapes reverted.
//   * Non-``Task`` headings and empty bodies are skipped.
//   * A missing or unreadable file in either location contributes
//     zero chips (the welcome screen still renders).
//
// Runs against the compiled extension under ``out/``:
//
//     node test/sampleTasks.test.js

'use strict';

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const Module = require('module');

// SorcarTab.js requires 'vscode' at load time.  Outside the extension
// host the module is unavailable, so we redirect that specifier to a
// tiny generated stub on disk — same pattern as the other bughunt
// tests in this directory.  ``readSampleTasks`` itself never touches
// any vscode API, so an empty object suffices.
const stubPath = path.join(__dirname, '_vscode-stub.js');
fs.writeFileSync(
  stubPath,
  `'use strict';\nmodule.exports = global.__kissVscodeStub || {};\n`,
);
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
const {readSampleTasks} = require(sourcePath);
assert.strictEqual(
  typeof readSampleTasks,
  'function',
  'readSampleTasks must be exported from the compiled SorcarTab',
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

function mkExt() {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-sample-tasks-'));
}

function packageSampleTasksPath(ext) {
  return path.join(
    ext,
    'kiss_project',
    'src',
    'kiss',
    'SAMPLE_TASKS.md',
  );
}

function writePackageSampleTasks(ext, content) {
  const file = packageSampleTasksPath(ext);
  fs.mkdirSync(path.dirname(file), {recursive: true});
  fs.writeFileSync(file, content);
}

/**
 * Run ``fn`` with ``KISS_HOME`` redirected to a fresh temp directory
 * so the lazy ``~/.kiss/MY_TASK_TEMPLATES.md`` seed in
 * ``readSampleTasks`` operates on disposable state.  Each invocation
 * also gets its own temporary extension root so the bundled package
 * copy at ``ext/kiss_project/src/kiss/SAMPLE_TASKS.md`` is isolated.
 */
function withTempKissHome(fn) {
  const ext = mkExt();
  const kissHome = mkExt();
  const prev = process.env.KISS_HOME;
  process.env.KISS_HOME = kissHome;
  try {
    fn(ext, kissHome);
  } finally {
    if (prev === undefined) delete process.env.KISS_HOME;
    else process.env.KISS_HOME = prev;
    fs.rmSync(ext, {recursive: true, force: true});
    fs.rmSync(kissHome, {recursive: true, force: true});
  }
}

test('auto-creates MY_TASK_TEMPLATES.md with seed "Hi!" when missing', () => {
  withTempKissHome((ext, kissHome) => {
    const myTasks = path.join(kissHome, 'MY_TASK_TEMPLATES.md');
    assert.ok(!fs.existsSync(myTasks), 'user copy must start absent');
    const tasks = readSampleTasks(ext);
    assert.ok(
      fs.existsSync(myTasks),
      'reader must create ~/.kiss/MY_TASK_TEMPLATES.md when missing',
    );
    assert.strictEqual(
      fs.readFileSync(myTasks, 'utf-8'),
      '## Task\n\nHi!\n',
      'seed content must be a single Hi! Task section',
    );
    // No package SAMPLE_TASKS.md was written, so only the seeded
    // ``Hi!`` chip appears.
    assert.deepStrictEqual(tasks, [{text: 'Hi!'}]);
  });
});

test('does NOT copy SAMPLE_TASKS.md into ~/.kiss/', () => {
  withTempKissHome((ext, kissHome) => {
    writePackageSampleTasks(ext, '## Task\n\nPackage-only chip\n');
    const tasks = readSampleTasks(ext);
    // Bundled chip is rendered ...
    assert.deepStrictEqual(tasks, [
      {text: 'Hi!'},
      {text: 'Package-only chip'},
    ]);
    // ... but ~/.kiss/SAMPLE_TASKS.md is never created — the package
    // copy is read directly so updates land automatically.
    assert.ok(
      !fs.existsSync(path.join(kissHome, 'SAMPLE_TASKS.md')),
      '~/.kiss/SAMPLE_TASKS.md must never be seeded by readSampleTasks',
    );
  });
});

test('MY_TASK_TEMPLATES.md chips appear before SAMPLE_TASKS.md chips', () => {
  withTempKissHome((ext, kissHome) => {
    fs.mkdirSync(kissHome, {recursive: true});
    fs.writeFileSync(
      path.join(kissHome, 'MY_TASK_TEMPLATES.md'),
      '## Task\n\nMy first task\n\n## Task\n\nMy second task\n',
    );
    writePackageSampleTasks(
      ext,
      '## Task\n\nBundled A\n\n## Task\n\nBundled B\n',
    );
    assert.deepStrictEqual(readSampleTasks(ext), [
      {text: 'My first task'},
      {text: 'My second task'},
      {text: 'Bundled A'},
      {text: 'Bundled B'},
    ]);
  });
});

test('preserves user edits to MY_TASK_TEMPLATES.md across reads', () => {
  withTempKissHome((ext, kissHome) => {
    fs.mkdirSync(kissHome, {recursive: true});
    const myTasks = path.join(kissHome, 'MY_TASK_TEMPLATES.md');
    fs.writeFileSync(myTasks, '## Task\n\nCurated chip\n');
    writePackageSampleTasks(ext, '## Task\n\nFresh bundled\n');
    assert.deepStrictEqual(readSampleTasks(ext), [
      {text: 'Curated chip'},
      {text: 'Fresh bundled'},
    ]);
    // The user copy must NOT be overwritten by the seed default
    // ``Hi!`` even after re-reading.
    assert.strictEqual(
      fs.readFileSync(myTasks, 'utf-8'),
      '## Task\n\nCurated chip\n',
    );
  });
});

test('returns only MY_TASK_TEMPLATES.md chips when package SAMPLE_TASKS.md is missing', () => {
  withTempKissHome((ext, kissHome) => {
    fs.mkdirSync(kissHome, {recursive: true});
    fs.writeFileSync(
      path.join(kissHome, 'MY_TASK_TEMPLATES.md'),
      '## Task\n\nOnly mine\n',
    );
    assert.deepStrictEqual(readSampleTasks(ext), [{text: 'Only mine'}]);
  });
});

test('returns only seed Hi! + bundled chips when MY_TASK_TEMPLATES.md was empty', () => {
  withTempKissHome((ext, kissHome) => {
    fs.mkdirSync(kissHome, {recursive: true});
    // User explicitly emptied their template file — only ``## Task``
    // sections contribute, so the chip list is just the bundled
    // tasks (the seed default does not re-seed an existing file).
    fs.writeFileSync(path.join(kissHome, 'MY_TASK_TEMPLATES.md'), '');
    writePackageSampleTasks(ext, '## Task\n\nBundled only\n');
    assert.deepStrictEqual(readSampleTasks(ext), [{text: 'Bundled only'}]);
  });
});

test('parses multiple ## Task sections in source order', () => {
  withTempKissHome(ext => {
    // Suppress the seed ``Hi!`` chip by pre-creating an empty
    // MY_TASK_TEMPLATES.md so the assertion only sees the bundled
    // file's ordering.
    fs.mkdirSync(process.env.KISS_HOME, {recursive: true});
    fs.writeFileSync(
      path.join(process.env.KISS_HOME, 'MY_TASK_TEMPLATES.md'),
      '',
    );
    writePackageSampleTasks(
      ext,
      '## Task\n\nFirst task\n\n## Task\n\nSecond task\n\n## Task\n\nThird\n',
    );
    assert.deepStrictEqual(readSampleTasks(ext), [
      {text: 'First task'},
      {text: 'Second task'},
      {text: 'Third'},
    ]);
  });
});

test('preserves multi-line bodies and inline punctuation', () => {
  withTempKissHome(ext => {
    fs.mkdirSync(process.env.KISS_HOME, {recursive: true});
    fs.writeFileSync(
      path.join(process.env.KISS_HOME, 'MY_TASK_TEMPLATES.md'),
      '',
    );
    const body =
      'Line one with **bold**, "quotes", and a <<placeholder>>.\n' +
      'Line two continues the same task.';
    writePackageSampleTasks(ext, `## Task\n\n${body}\n`);
    assert.deepStrictEqual(readSampleTasks(ext), [{text: body}]);
  });
});

test('skips sections whose heading is not Task', () => {
  withTempKissHome(ext => {
    fs.mkdirSync(process.env.KISS_HOME, {recursive: true});
    fs.writeFileSync(
      path.join(process.env.KISS_HOME, 'MY_TASK_TEMPLATES.md'),
      '## Intro\n\nignored\n\n## Task\n\nmy kept\n',
    );
    writePackageSampleTasks(
      ext,
      '## Intro\n\nignored\n\n## Task\n\nkept\n\n## Notes\n\nalso ignored\n',
    );
    assert.deepStrictEqual(readSampleTasks(ext), [
      {text: 'my kept'},
      {text: 'kept'},
    ]);
  });
});

test('skips empty-bodied ## Task sections', () => {
  withTempKissHome(ext => {
    fs.mkdirSync(process.env.KISS_HOME, {recursive: true});
    fs.writeFileSync(
      path.join(process.env.KISS_HOME, 'MY_TASK_TEMPLATES.md'),
      '',
    );
    writePackageSampleTasks(ext, '## Task\n\n\n## Task\n\nreal body\n');
    assert.deepStrictEqual(readSampleTasks(ext), [{text: 'real body'}]);
  });
});

test('unescapes mdformat backslash escapes (\\<< -> <<)', () => {
  withTempKissHome(ext => {
    fs.mkdirSync(process.env.KISS_HOME, {recursive: true});
    fs.writeFileSync(
      path.join(process.env.KISS_HOME, 'MY_TASK_TEMPLATES.md'),
      '',
    );
    // ``mdformat`` rewrites ``<<x>>`` to ``\<<x>>`` on save.  The chip
    // renders ``s.text`` literally (only HTML-escaped) so the parser
    // MUST strip CommonMark backslash escapes — otherwise the user
    // sees a literal backslash on the welcome screen.
    writePackageSampleTasks(
      ext,
      '## Task\n\nRun on \\<<your dataset>> with **bold** \\*literal\\*\n',
    );
    assert.deepStrictEqual(readSampleTasks(ext), [
      {text: 'Run on <<your dataset>> with **bold** *literal*'},
    ]);
  });
});

test('skips user chips when ~/.kiss/ is unwritable (ensureUserAssetFromDefault returns null)', () => {
  // Locks in the read-only-FS branch in ``ensureUserAssetFromDefault``:
  // when the helper cannot write the seed file it returns ``null`` and
  // ``readSampleTasks`` silently drops the user-chips section instead
  // of crashing.  Only the bundled chips appear.
  if (process.getuid && process.getuid() === 0) {
    console.log('  ok - SKIPPED (root) - returns only bundled chips when ~/.kiss/ is unwritable');
    passed += 1;
    return;
  }
  withTempKissHome((ext, kissHome) => {
    // Recreate kissHome as an unwritable directory so the seed write
    // fails inside ensureUserAssetFromDefault.  The helper's
    // try/catch turns the error into ``null``.
    fs.mkdirSync(kissHome, {recursive: true});
    fs.chmodSync(kissHome, 0o500);
    try {
      writePackageSampleTasks(ext, '## Task\n\nOnly bundled\n');
      assert.deepStrictEqual(readSampleTasks(ext), [{text: 'Only bundled'}]);
      // The seed file was NOT created (write failed).
      assert.ok(
        !fs.existsSync(path.join(kissHome, 'MY_TASK_TEMPLATES.md')),
        'seed file must not be created on read-only ~/.kiss/',
      );
    } finally {
      fs.chmodSync(kissHome, 0o700);
    }
  });
});

test('falls back to dev-checkout SAMPLE_TASKS.md when packaged copy is absent', () => {
  // Locks in the dev-fallback branch: when
  // ``<ext>/kiss_project/src/kiss/SAMPLE_TASKS.md`` does not exist,
  // ``readSampleTasks`` reads ``<ext>/../../SAMPLE_TASKS.md`` instead
  // (the source-checkout layout used when running tsc out of the
  // monorepo without packaging).
  const prev = process.env.KISS_HOME;
  // Build a synthetic monorepo layout: <root>/SAMPLE_TASKS.md is the
  // dev-checkout file, and the extension lives at <root>/a/b/ so
  // ``<ext>/../../SAMPLE_TASKS.md`` resolves to <root>/SAMPLE_TASKS.md.
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-dev-fallback-'));
  const ext = path.join(root, 'a', 'b');
  fs.mkdirSync(ext, {recursive: true});
  const devFile = path.join(root, 'SAMPLE_TASKS.md');
  fs.writeFileSync(devFile, '## Task\n\nDev checkout chip\n');
  const kissHome = path.join(root, '.kiss');
  fs.mkdirSync(kissHome);
  // Suppress the seeded ``Hi!`` chip so the assertion only observes
  // the dev-checkout file.
  fs.writeFileSync(path.join(kissHome, 'MY_TASK_TEMPLATES.md'), '');
  process.env.KISS_HOME = kissHome;
  try {
    assert.deepStrictEqual(readSampleTasks(ext), [
      {text: 'Dev checkout chip'},
    ]);
  } finally {
    if (prev === undefined) delete process.env.KISS_HOME;
    else process.env.KISS_HOME = prev;
    fs.rmSync(root, {recursive: true, force: true});
  }
});

test('shipped SAMPLE_TASKS.md tasks never contain a leading backslash before <<', () => {
  withTempKissHome(() => {
    const ext = path.join(__dirname, '..');
    const tasks = readSampleTasks(ext);
    for (const t of tasks) {
      assert.ok(
        !/\\</.test(t.text),
        `task should not retain mdformat escapes: ${JSON.stringify(t.text)}`,
      );
    }
  });
});

test('parses the shipped SAMPLE_TASKS.md (sanity)', () => {
  withTempKissHome(() => {
    // The real source-checkout file lives at src/kiss/SAMPLE_TASKS.md,
    // two levels above the extension root.  This also guards against
    // accidentally checking in a malformed file.
    const ext = path.join(__dirname, '..');
    const shipped = path.join(ext, '..', '..', 'SAMPLE_TASKS.md');
    assert.ok(
      fs.existsSync(shipped),
      `shipped SAMPLE_TASKS.md missing at ${shipped}`,
    );
    const tasks = readSampleTasks(ext);
    assert.ok(Array.isArray(tasks), 'must return an array');
    assert.ok(tasks.length > 0, 'shipped file must contain at least one task');
    for (const t of tasks) {
      assert.strictEqual(typeof t.text, 'string');
      assert.ok(t.text.length > 0, 'each task body must be non-empty');
      // Each chip is displayed on the welcome screen — make sure the
      // text is trimmed so the chip label doesn't carry stray newlines.
      assert.strictEqual(t.text, t.text.trim());
    }
  });
});

console.log(`\n${passed} passed, ${failures.length} failed`);
if (failures.length > 0) {
  for (const f of failures) {
    console.error(`\n${f.name}:\n`, f.err);
  }
  process.exit(1);
}
